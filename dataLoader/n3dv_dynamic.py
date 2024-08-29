import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os
from PIL import Image
from torchvision import transforms as T

from .ray_utils import *
from .deepview_dynamic import interp_poses


def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120):
    render_poses = []
    rads = np.array(list(rads) + [1.])

    for theta in np.linspace(0., 2. * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120, zrate=0.5):
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / (((1.0 - dt) / close_depth + dt / inf_depth))

    # Get radii for spiral path
    zdelta = near_fars.min() * .2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate=zrate, N=N_views)
    return np.stack(render_poses)


class N3DVDynamicDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8, num_frames=30, frame_list=[]):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.root_dir = datadir
        self.split = split
        self.hold_every = hold_every
        self.num_frames = num_frames
        self.frame_list = frame_list
        if len(self.frame_list) == 0:
            self.frame_list = range(self.num_frames)
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = False

        #         self.near_far = [np.min(self.near_fars[:,0]),np.max(self.near_fars[:,1])]
        self.near_far = [0.0, 1.0]
        self.scene_bbox = torch.tensor([[-1.5, -1.67, -1.0], [1.5, 1.67, 1.0]])
        # self.scene_bbox = torch.tensor([[-1.67, -1.5, -1.0], [1.67, 1.5, 1.0]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)

    def read_meta(self):
        all_cams = sorted([i.replace('.mp4', '') for i in 
                           os.listdir(f'{self.root_dir}/') if 'mp4' in i])
        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))
        
        if 'coffee_martini' in self.root_dir:
            print('====================deletting unsynchronized video==============')
            poses_bounds = np.concatenate([poses_bounds[:12], poses_bounds[13:]], axis=0)
            all_cams.pop(12)
            
        assert poses_bounds.shape[0] == len(all_cams), 'poses not aligned with cams'
        poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
        self.near_fars = poses_bounds[:, -2:]  # (N_images, 2)
        
        # Step 1: rescale focal length according to training resolution
        H, W, self.focal = poses[0, :, -1]  # original intrinsics, same for all images
        self.img_wh = np.array([int(W / self.downsample), int(H / self.downsample)])
        self.focal = [self.focal * self.img_wh[0] / W, self.focal * self.img_wh[1] / H]

        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right up back"
        # See https://github.com/bmild/nerf/issues/34
        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)
        # (N_images, 3, 4) exclude H, W, focal
        self.poses, self.pose_avg = center_poses(poses, self.blender2opencv)

        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        near_original = self.near_fars.min()*1.8
        scale_factor = near_original * 0.6
        # scale_factor = near_original * 0.75  # 0.75 is the default parameter
        # the nearest depth is at 1/0.75=1.33
        self.near_fars /= scale_factor
        self.poses[..., 3] /= scale_factor
        
        # build rendering path
        N_views, N_rots = 120, 2
        tt = self.poses[:, :3, 3]  # ptstocam(self.poses[:3,3,:].T, c2w).T
        up = normalize(self.poses[:, :3, 1].sum(0))
        rads = np.percentile(np.abs(tt), 90, 0)
        # self.render_path = get_spiral(self.poses, self.near_fars, rads_scale=0.4, zrate=-0.5, N_views=N_views)
        self.render_path = get_spiral(self.poses, self.near_fars, rads_scale=0.4, zrate=-0.5, N_views=self.num_frames)

        
        K = torch.zeros((len(all_cams), 3, 3), dtype=torch.float32)
        K[..., 0, 0] = self.focal[0].squeeze(-1)
        K[..., 1, 1] = self.focal[1].squeeze(-1)
        K[..., 0, 2] = self.img_wh[0] * 0.5
        K[..., 1, 2] = self.img_wh[1] * 0.5
        K[..., 2, 2] = 1.0
        self.intrinsics = K
        
        extrinsics = []
        for i in range(len(self.poses)):
            c2w = torch.from_numpy(self.poses[i]).float()
            c2w = torch.cat([c2w, torch.tensor([[0, 0, 0, 1]]).float()], dim=0)
            w2c = torch.inverse(c2w)
            extrinsics.append(w2c)
        extrinsics = torch.stack(extrinsics, dim=0) # (N_images, 4, 4)
        self.extrinsics = extrinsics

        # ray directions for all pixels, same for all images (same H, W, focal)
        W, H = self.img_wh
        self.directions = get_ray_directions_blender(H, W, self.focal)  # (H, W, 3)
        average_pose = average_poses(self.poses)
        dists = np.sum(np.square(average_pose[:3, 3] - self.poses[:, :3, 3]), -1)
        i_test = [0]
        cam_list = i_test if self.split != 'train' else list(set(np.arange(len(self.poses))))

        self.all_rays = []
        self.all_rgbs = []
        for i_frame, frame in enumerate(tqdm(self.frame_list, desc='frame')):
            for i in tqdm(cam_list, desc='camera', leave=False):
                image_name = f'{all_cams[i]}/{(frame+1):06d}.jpg'
                ori_image_path = f'{self.root_dir}/frames/{image_name}'
                image_path = f'{self.root_dir}/frames{int(self.downsample):d}x/{image_name}'
                os.makedirs(f'{self.root_dir}/frames{int(self.downsample):d}x/', exist_ok=True)

                c2w = torch.FloatTensor(self.poses[i])
                if os.path.isfile(image_path):
                    img = Image.open(image_path).convert('RGB')
                else:
                    os.makedirs(f'{self.root_dir}/frames{int(self.downsample):d}x/{all_cams[i]}', exist_ok=True)
                    img = Image.open(ori_image_path).convert('RGB')
                    img = img.resize(self.img_wh, Image.LANCZOS)
                    img.save(image_path)
                img = self.transform(img)  # (3, h, w)
                img = img.view(3, -1).permute(1, 0)  # (h*w, 3) RGB
                self.all_rgbs += [img]
                rays_o, rays_d = get_rays(self.directions, c2w)  # both (h*w, 3)
                rays_o, rays_d = ndc_rays_blender(H, W, self.focal[0], 1.0, rays_o, rays_d)
                # viewdir = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
                if self.num_frames > 1:
                    frame_vec = torch.zeros([*rays_o.shape[:-1],1])+i_frame/(self.num_frames-1)
                else:
                    frame_vec = torch.zeros([*rays_o.shape[:-1],1])
                self.all_rays += [torch.cat([rays_o, rays_d, frame_vec], 1)]

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0)   # (len(self.meta['frames]),h,w, 3)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)  # (len(self.meta['frames]),h,w,3)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample