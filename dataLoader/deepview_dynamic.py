import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import os
from torchvision import transforms as T
import mmcv

from .ray_utils import *
from .deepview_static import *


class DeepviewDynamicDataset(Dataset):
    def __init__(self, datadir, split='train', downsample=4, is_stack=False, hold_every=8, num_frames=30, frame_list=[]):
        """
        spheric_poses: whether the images are taken in a spheric inward-facing manner
                       default: False (forward-facing)
        val_num: number of val images (used for multigpu training, validate same image for all gpus)
        """

        self.root_dir = os.path.expanduser(datadir)
        self.split = split
        self.total_cams = 46
        self.hold_every = hold_every
        self.num_frames = num_frames
        self.frame_list = frame_list
        
        if len(self.frame_list) == 0:
            self.frame_list = range(self.num_frames)
        self.is_stack = is_stack
        self.downsample = downsample
        self.define_transforms()
        if self.split=='train':
            self.near_far = [0.02, 4]
        else:
            self.near_far = [0.12, 2]

        self.blender2opencv = np.eye(4)#np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        self.read_meta()
        self.white_bg = False

        self.scene_bbox = torch.tensor([[-2., -2., -2.], [2., 2., 2.]])
        self.center = torch.mean(self.scene_bbox, dim=0).float().view(1, 1, 3)
        self.invradius = 1.0 / (self.scene_bbox[1] - self.center).float().view(1, 1, 3)
        if self.split=='train':
            self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [self.focal]*2)
        else:
            self.directions = get_ray_directions(self.img_wh[1], self.img_wh[0], [self.focal*1.2]*2)

    def read_meta(self):
        subset = list(range(self.total_cams))
        if self.split=='train':
            subset = subset[1:]
            # subset = subset
        else:
            subset = [0]
            # subset = subset
            
        data_dict = mmcv.load(f'{self.root_dir}/data_dict.json')
        center = torch.as_tensor(data_dict['center'])
        scale = torch.as_tensor(data_dict['scale'])
        
        self.all_rays = []
        self.all_rgbs = []
        for i_frame, frame in enumerate(tqdm(self.frame_list, desc='frame')):
            rays_o, rays_d, all_imgs, \
            all_poses, [h, w, focal], \
            render_poses, K, w2c, c2w = load_deepview_static_data(self.root_dir, subset, 
                                                                  factor=self.downsample, device='cpu', 
                                                                  frame=frame, num_render_poses=max(60,self.num_frames))
            rays_o, rays_d, all_imgs = rays_o.view([-1,3]), rays_d.view([-1,3]), all_imgs.view([-1,3])
            if self.num_frames > 1:
                frame_vec = torch.zeros([*rays_o.shape[:-1],1])+i_frame/(self.num_frames-1)
            else:
                frame_vec = torch.zeros([*rays_o.shape[:-1],1])
                
            # near, far = self.near_far 
            # points = torch.stack([rays_o+rays_d*near, rays_o+rays_d*far])
            # center = (points.amax((0,1,2))+points.amin((0,1,2)))/2
            # scale = (points.amax((0,1,2))-points.amin((0,1,2))).norm()/2
            
            # data_dict = {'center': center.tolist(), 'scale': scale.item()}
            # import json
            # with open(f'{self.root_dir}/data_dict.json', 'w') as f:
            #     json.dump(data_dict, f)
                
            rays_o = (rays_o-center)/scale
            self.all_rays += [torch.cat([rays_o, rays_d, frame_vec], 1)]
            self.all_rgbs += [all_imgs]

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

        all_poses[...,:3,-1] = (all_poses[...,:3,-1]-center)/scale
        render_poses[...,:3,-1] = (render_poses[...,:3,-1]-center)/scale
        self.render_path = render_poses
        self.img_wh = np.array([w,h])
        self.focal = focal

        if not self.is_stack:
            self.all_rays = torch.cat(self.all_rays, 0) # (len(self.meta['frames])*h*w, 3)
            self.all_rgbs = torch.cat(self.all_rgbs, 0) # (len(self.meta['frames])*h*w,3)
        else:
            self.all_rays = torch.stack(self.all_rays, 0).reshape(-1,*self.img_wh[::-1], 7)
            self.all_rgbs = torch.stack(self.all_rgbs, 0).reshape(-1,*self.img_wh[::-1], 3)


    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        return len(self.all_rgbs)

    def __getitem__(self, idx):

        sample = {'rays': self.all_rays[idx],
                  'rgbs': self.all_rgbs[idx]}

        return sample