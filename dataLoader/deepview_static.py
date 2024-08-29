import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T
from scipy.spatial.transform import Rotation
import cv2
import json
from packaging import version as pver
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import mmengine

from .ray_utils import *


def interp_poses(poses, num_render_poses):
    out_poses = []
    per_pose_num = num_render_poses // len(poses)
    for k in range(len(poses)):
        rot1, trans1 = poses[k][:3,:3], poses[k][:3,3]
        rot2, trans2 = poses[(k+1)%len(poses)][:3,:3], poses[(k+1)%len(poses)][:3,3]
        rots = R.from_matrix(np.stack([rot1,rot2]))
        slerp = Slerp([0,1], rots)
        for i in range(per_pose_num):
            ratio = 1-i/per_pose_num
            trans = trans1*ratio+trans2*(1-ratio)
            pose = torch.zeros_like(poses[0])
            pose[:3,3] = trans
            pose[:3,:3] = torch.as_tensor(slerp([1-ratio])[0].as_matrix())
            out_poses.append(pose)
    return torch.stack(out_poses, 0)[:,:3,:4]

def custom_meshgrid(*args):
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def unproj(pt2d, pt=1, ps=1, K=None, dist=None, R=None, t=None):
    # pt2d: [2, N]
    # distCoef: [5]
    # R: w2c; t: c2w; R.T@p+t
    args = [K, dist, R, t]
    for i,v in enumerate(args):
        if isinstance(v, torch.Tensor):
            args[i] = v.cpu().numpy()
    K, dist, R, t = args
    N_points = pt2d.shape[1]
    D = np.zeros(4)
    D[:3] = dist
    unproj_pt = torch.as_tensor(cv2.fisheye.undistortPoints(pt2d.T[None,:], K, D)[0])
    if R is None:
        return unproj_pt
    unproj_pt = torch.cat([unproj_pt, torch.ones_like(unproj_pt[:,:1])], 1)
    rays_dir = torch.as_tensor(R).float().T @ unproj_pt.T
    rays_dir /= rays_dir.norm(dim=0, keepdim=True)
    rays_ori = torch.as_tensor(t).float().view([3,1])
    return rays_ori.expand_as(rays_dir).T, rays_dir.T

class Unproj:
    def __init__(self, params):
        self.params = params
    def __call__(self, *args, **kwds):
        return unproj(*args, **kwds, **self.params)

def proj(pt3d, pt, ps, dist, R, t, K=None):
    # pt, ps: point translation and point scale
    # t: w2c t; R.T@t is the cam location
    # R: w2c R
    if len(pt3d) == 1:
        pt3d = pt3d[0]
    else:
        assert 1
    pt3d = pt3d*ps+pt
    pt3d = pt3d.T # [3,n]
    pt3d = R@pt3d-R.T@t.view([-1,1])
    r = torch.sqrt(pt3d[0,:]*pt3d[0,:] + pt3d[1,:]*pt3d[1,:])
    theta = torch.atan2(r, pt3d[2,:])
    r2 = theta * theta
    distortion = 1.0 + r2 * (dist[0] + r2 * dist[1])
    return torch.stack([theta/r*pt3d[0,:]*distortion,
                        theta/r*pt3d[1,:]*distortion,
                        torch.ones_like(distortion),], 1)[None]

class Proj:
    def __init__(self, params):
        self.params = params
    def __call__(self, *args, **kwds):
        return proj(*args, **kwds, **self.params)

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center)], 1)
    return c2w

def recenter_poses(poses):
    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def get_c2w(R, t):
    tmp = torch.zeros([3,4])
    tmp[:3,:3] = torch.as_tensor(R).T
    tmp[:3,3] = torch.as_tensor(t).view([-1])
    return tmp

def get_K(view):
    return np.array(
        [[view['focal_length'], 0.0, view['principal_point'][0]],
        [0.0, view['focal_length'], view['principal_point'][1]],
        [0.0, 0.0, 1.0]])

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]

    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta)*3, -np.sin(theta)*3, -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return np.stack(render_poses, 0)


def load_deepview_static_data(basedir, subset, factor=8, frame=8, device='cuda', num_render_poses=27*3, only_render=False, only_names=False):
    all_cams = sorted([i.replace('.mp4', '') for i in 
                       os.listdir(f'{basedir}/') if 'mp4' in i])
    all_cams = [k for i,k in enumerate(all_cams) if i in subset]
    
    with open(f'{basedir}/models.json') as cfile:
        calib = json.load(cfile)
    cameras_raw = {cam['name']:cam for cam in calib}
    all_cams = sorted(list(set(all_cams).intersection(set(cameras_raw.keys()))))

    factor = int(factor)
    resize_root_path = f'{basedir}/frames{factor:d}x'
    if not os.path.isdir(resize_root_path):
        os.mkdir(resize_root_path)
    ori_w, ori_h = Image.open(f'{basedir}/{all_cams[0]}/{(frame+1):04d}.jpg').size
    w = ori_w//factor
    h = ori_h//factor

    all_imgs = []
    if not only_render:
        all_ori_imgs = [f'{basedir}/{all_cams[0]}/{(frame+1):04d}.jpg' for cam in all_cams]
        all_image_names = [f'{cam}_{frame:06d}.jpg' for cam in all_cams]
        for img_name, ori_img in zip(all_image_names, all_ori_imgs):
            resized_path = f'{resize_root_path}/{img_name}'
            if os.path.isfile(resized_path):
                if only_names:
                    all_imgs.append(resized_path)
                    continue
                all_imgs.append(np.array(Image.open(resized_path)))
            else:
                im = Image.open(ori_img).resize((w,h), Image.BILINEAR)
                im.save(resized_path)
                if only_names:
                    all_imgs.append(resized_path)
                    continue
                all_imgs.append(np.array(im))

    center_point = np.stack([c['position'] for c in cameras_raw.values()], axis=0)
    center_point = torch.as_tensor(center_point, device=device).mean(0).float()
    cam_params = {}
    for cam in all_cams:
        view = cameras_raw[cam]
        K = get_K(view)
        K /= factor; K[-1,-1] = 1
        cam_params[cam] = dict(
            K=torch.as_tensor(K, device=device),
            dist=torch.as_tensor(view['radial_distortion'], device=device),
            R=torch.as_tensor(Rotation.from_rotvec(view['orientation']).as_matrix(), device=device).float(),
            t=torch.as_tensor(view['position'], device=device)-center_point,)
    #! extrinsics(w2c) = np.concatenate((R, -np.dot(R, t)), axis=1)

    # w2c and K are for render poses and hwf
    all_c2w = [get_c2w(cam_params[cam]['R'], cam_params[cam]['t']) for cam in all_cams]
    # all_poses = torch.stack(all_c2w, 0).to(device)
    all_poses = torch.stack(all_c2w, 0).numpy()
    # all_poses = recenter_poses(all_poses.cpu().numpy())
    all_K = [cam_params[cam]['K'] for cam in all_cams]
    focal = all_K[0][0,0].item()

    rays_o, rays_d = [], []
    i, j = custom_meshgrid(torch.linspace(0, w-1, w, device='cpu'), 
                           torch.linspace(0, h-1, h, device='cpu'))
    i, j = i.T, j.T
    coords = torch.stack([i,j], dim=-1).view([-1,2]).T # [2,N]
    for cam_param in cam_params.values():
        if not only_names:
            o, d = unproj(coords.numpy(), **cam_param)
        else:
            o, d = torch.zeros([2])
        rays_o.append(o)
        rays_d.append(d)
    rays_o = torch.stack(rays_o, dim=0).to(device)
    rays_d = torch.stack(rays_d, dim=0).to(device)
    if not (only_render or only_names):
        all_imgs = torch.as_tensor(np.stack(all_imgs, 0), device=device)/255.

    # render poses
    if os.path.isfile(f'{basedir}/saved_poses.json'):
        with open(f'{basedir}/saved_poses.json', 'r') as f:
            render_poses = json.load(f)
        render_poses = torch.as_tensor(render_poses)[:,:3,:4]
        render_poses = interp_poses(render_poses, num_render_poses)
    else:
        up = np.array([0,1,0])
        rads = np.percentile(np.abs(all_poses[:,:3,3]), 90, 0)
        avg_pose = poses_avg(all_poses)
        render_poses = render_path_spiral(avg_pose, up, rads, focal, 
                                          zdelta=3, zrate=.5, rots=2, N=num_render_poses)


    c2w = np.array([Unproj(param) for param in cam_params.values()])
    w2c = np.array([Proj({**param, 'K':None}) for param in cam_params.values()])

    return (rays_o, rays_d, all_imgs, 
            torch.as_tensor(all_poses, device=device), [h, w, focal], 
            torch.as_tensor(render_poses, device=device).float(), 
            torch.stack([c['K'] for c in cam_params.values()], 0), 
            w2c, c2w)


def get_ray_directions(H, W, focal, center=None):
    """
    Get ray directions for all pixels in camera coordinate.
    Reference: https://www.scratchapixel.com/lessons/3d-basic-rendering/
               ray-tracing-generating-camera-rays/standard-coordinate-systems
    Inputs:
        H, W, focal: image height, width and focal length
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    grid = create_meshgrid(H, W, normalized_coordinates=False)[0] + 0.5

    i, j = grid.unbind(-1)
    # the direction here is without +0.5 pixel centering as calibration is not so accurate
    # see https://github.com/bmild/nerf/issues/24
    cent = center if center is not None else [W / 2, H / 2]
    directions = torch.stack([(i - cent[0]) / focal[0], (j - cent[1]) / focal[1], torch.ones_like(i)], -1)  # (H, W, 3)

    return directions


class DeepviewDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8, frame=0):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.root_dir = os.path.expanduser(datadir)
        self.split = split
        self.total_cams = 46
        self.hold_every = hold_every
        self.frame = frame
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()
        if self.split=='train':
            self.near_far = [0.05, 4]
        else:
            self.near_far = [0.2, 2]

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = False

        self.scene_bbox = torch.tensor([[-2., -2., -2.], [2., 2., 2.]])
        # self.scene_bbox = torch.tensor([[-0.5, -0.5, -0.5], [2., 2., 2.]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        # self.center = torch.zeros(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [self.focal]*2)

    def read_meta(self):
        subset = list(range(self.total_cams))
        if self.split=='train':
            subset = subset[1:]
            # subset = subset
        else:
            subset = [0]
            # subset = subset

        rays_o, rays_d, all_imgs, \
        all_poses, [h, w, focal], \
        render_poses, K, w2c, c2w = load_deepview_static_data(self.root_dir, subset, factor=self.downsample, device='cpu', frame=8)

        # near, far = self.near_far 
        # points = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
        # center = (points.amax((0,1,2))+points.amin((0,1,2)))/2
        # scale = (points.amax((0,1,2))-points.amin((0,1,2))).norm()/2
        # print(center, scale)
        # rays_o -= center
        # # center = torch.tensor([ 1.1362e-02, -8.1062e-04,  1.2077e+00]) 
        # # scale = 27.6588
        # rays_o /= scale
        # near /= scale
        # far /= scale

        data_dict = mmengine.load(f'{self.root_dir}/data_dict.json')
        center = torch.as_tensor(data_dict['center']).to(rays_o)
        scale = torch.as_tensor(data_dict['scale']).to(rays_o)
        print(center, scale)
        rays_o = (rays_o-center)/scale

        all_poses[...,:3,-1] = (all_poses[...,:3,-1]-center)/scale
        render_poses[...,:3,-1] = (render_poses[...,:3,-1]-center)/scale
        self.render_path = render_poses
        self.img_wh = np.array([w,h])
        self.focal = focal

        if not self.is_stack:
            self.all_rays = torch.cat([rays_o, rays_d], -1).reshape([-1,6])
            self.all_rgbs = all_imgs.reshape([-1,3])
        else:
            self.all_rays = torch.cat([rays_o, rays_d], -1).reshape([-1,h,w,6])
            self.all_rgbs = all_imgs


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample