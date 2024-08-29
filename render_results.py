import os
from tqdm.auto import tqdm

import json, random
from stream_renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys
import torchvision

import mmengine
import configargparse
import argparse
from einops import rearrange
import math

import threading

from dataLoader.ray_utils import *

from ip2p import InstructPix2Pix
from itertools import cycle

from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import warnings; warnings.filterwarnings("ignore")

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
    parser.add_argument("--cache", type=str, default='./cache',
                        help='where to store some cache')
    parser.add_argument("--add_timestamp", type=int, default=0,
                        help='add timestamp to dir')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    parser.add_argument("--progress_refresh_rate", type=int, default=10,
                        help='how many iterations to show psnrs or iters')
    parser.add_argument('--with_depth', action='store_true')
    parser.add_argument('--downsample_train', type=float, default=1.0)
    parser.add_argument('--downsample_test', type=float, default=1.0)
    parser.add_argument('--model_name', type=str, default='StreamTensorVMSplit',
                        choices=['StreamTensorVMSplit', 'StreamTensorCP'])
    parser.add_argument('--tag', type=str, default='orig')
    
    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--patch_size", type=int, default=32)
    parser.add_argument("--n_iters", type=int, default=30000)
    parser.add_argument("--n_keyframe_iters", type=int, default=800)
    parser.add_argument('--dataset_name', type=str, default='blender', choices=['n3dv_dynamic','deepview_dynamic',])
    
    # sequence_ip2p options
    parser.add_argument('--prompt', type=str, default="don't change the image",
                        help='prompt for InstructPix2Pix')
    parser.add_argument('--guidance_scale', type=float, default=7.5,
                        help='(text) guidance scale for InstructPix2Pix')
    parser.add_argument('--image_guidance_scale', type=float, default=1.5,
                        help='image guidance scale for InstructPix2Pix')
    parser.add_argument('--diffusion_steps', type=int, default=20,
                        help='number of diffusion steps to take for InstructPix2Pix')
    parser.add_argument('--refine_diffusion_steps', type=int, default=6,
                        help='number of diffusion steps to take for refinement')
    parser.add_argument('--refine_num_steps', type=int, default=700,
                        help='number of denoise steps to take for refinement')
    parser.add_argument('--ip2p_device', type=str, default='cuda:1',
                        help='second device to place InstructPix2Pix on')
    parser.add_argument('--ip2p_use_full_precision', type=bool, default=False,
                        help='Whether to use full precision for InstructPix2Pix')
    parser.add_argument('--sequence_length', type=int, default=5,
                        help='length of the sequence')
    parser.add_argument('--overlap_length', type=int, default=1,
                        help='length of the overlap')
    
    # training options
    # learning rate
    parser.add_argument("--lr_init", type=float, default=0.02,
                        help='learning rate')    
    parser.add_argument("--lr_basis", type=float, default=1e-3,
                        help='learning rate')
    parser.add_argument("--lr_decay_iters", type=int, default=-1,
                        help = 'number of iterations the lr will decay to the target ratio; -1 will set it to n_iters')
    parser.add_argument("--lr_decay_target_ratio", type=float, default=0.1,
                        help='the target decay ratio; after decay_iters inital lr decays to lr*ratio')
    parser.add_argument("--lr_upsample_reset", type=int, default=1,
                        help='reset lr to inital after upsampling')
    # loss
    parser.add_argument("--L1_weight_inital", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--L1_weight_rest", type=float, default=0,
                        help='loss weight')
    parser.add_argument("--Ortho_weight", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_density", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--TV_weight_app", type=float, default=0.0,
                        help='loss weight')
    parser.add_argument("--feat_diff_weight", type=float, default=0.0,)
    parser.add_argument("--lpips_weight", type=float, default=0.0,)
    # model
    parser.add_argument("--num_frames", type=int,)
    parser.add_argument("--frame_list", type=str, default='[]')
    # parser.add_argument("--deform_field", type=int,)
    # parser.add_argument("--portion_decoder", type=int,)
    parser.add_argument("--virtual_cannonical", type=int, default=0)
    parser.add_argument("--target_portion", type=int, action="append", default=[])
    parser.add_argument("--share_portion_embeddings", type=int, default=1)
    parser.add_argument("--portion_weight", type=float, default=0)
    # volume options
    parser.add_argument("--n_lamb_sigma", type=int, action="append")
    parser.add_argument("--n_lamb_sh", type=int, action="append")
    parser.add_argument("--data_dim_color", type=int, default=27)
    parser.add_argument("--ld_per_frame", type=float, default=1)
    parser.add_argument("--rm_weight_mask_thre", type=float, default=0.0001,
                        help='mask points in ray marching')
    parser.add_argument("--alpha_mask_thre", type=float, default=0.0001,
                        help='threshold for creating alpha mask volume')
    parser.add_argument("--distance_scale", type=float, default=25,
                        help='scaling sampling distance for computation')
    parser.add_argument("--density_shift", type=float, default=-10,
                        help='shift density in softplus; making density = 0  when feature == 0')
    # network decoder
    parser.add_argument("--shadingMode", type=str, default="MLP_PE",
                        help='which shading mode to use')
    parser.add_argument("--pos_pe", type=int, default=6,
                        help='number of pe for pos')
    parser.add_argument("--view_pe", type=int, default=6,
                        help='number of pe for view')
    parser.add_argument("--fea_pe", type=int, default=6,
                        help='number of pe for features')
    parser.add_argument("--featureC", type=int, default=128,
                        help='hidden feature channel in MLP')
    parser.add_argument("--ckpt", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--render_only", type=int, default=0)
    parser.add_argument("--render_test", type=int, default=0)
    parser.add_argument("--render_train", type=int, default=0)
    parser.add_argument("--render_path", type=int, default=0)
    parser.add_argument("--export_mesh", type=int, default=0)
    # rendering options
    parser.add_argument('--lindisp', default=False, action="store_true",
                        help='use disparity depth sampling')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--accumulate_decay", type=float, default=0.998)
    parser.add_argument("--fea2denseAct", type=str, default='softplus')
    parser.add_argument('--ndc_ray', type=int, default=0)
    parser.add_argument('--nSamples', type=int, default=1e6,
                        help='sample point each ray, pass 1e6 if automatic adjust')
    parser.add_argument('--step_ratio',type=float,default=0.5)
    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument('--N_voxel_init',
                        type=int,
                        default=100**3)
    parser.add_argument('--N_voxel_final',
                        type=int,
                        default=300**3)
    parser.add_argument("--upsamp_list", type=int, action="append")
    parser.add_argument("--update_AlphaMask_list", type=int, action="append")
    parser.add_argument('--idx_view', type=int, default=0)
    # logging/saving options
    parser.add_argument("--N_vis", type=int, default=5,
                        help='N images to vis')
    parser.add_argument("--vis_every", type=int, default=10000,
                        help='frequency of visualize the image')
    parser.add_argument('--cfg_options', nargs='+', action=mmengine.DictAction,)
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast


class SimpleSampler:
    def __init__(self, total, total_frame, batch):
        self.total = total
        self.total_frame = total_frame
        self.batch = batch
        self.curr = total
        self.ids = None
        self.permute_base = self.gen_permute()

    def nextids(self):
        # self.curr+=self.batch
        # if self.curr + self.batch > self.total:
        #     self.ids = self.gen_permute()
        #     self.curr = 0
        # return self.ids[self.curr:self.curr+self.batch]
        frame = int(random.random()*self.total_frame)
        start = int(random.random()*(len(self.permute_base)-self.batch))
        return self.permute_base[start:start+self.batch]+frame*self.per_frame_length

    def gen_permute(self):
        # return torch.LongTensor(np.random.permutation(self.total))
        self.per_frame_length = self.total / self.total_frame
        assert self.per_frame_length.is_integer()
        self.per_frame_length = int(self.per_frame_length)
        return torch.LongTensor(np.random.permutation(self.per_frame_length))


class MotionSampler:
    def __init__(self, allrgbs, total_frame, batch):
        self.total = allrgbs.shape[0]
        self.total_frame = total_frame
        self.batch = batch
        self.curr = self.total
        self.ids = None
        self.per_frame_length = self.total / self.total_frame # 表示一个frame里rays的数量
        assert self.per_frame_length.is_integer()
        self.per_frame_length = int(self.per_frame_length) 
        self.permute_base = torch.LongTensor(np.random.permutation(self.per_frame_length)) # 一个frame里的rays的随机排列
        motion_mask = (allrgbs-torch.roll(allrgbs,self.per_frame_length,0)).abs().mean(-1)>(10/255) # 两帧之间的运动大于10/255的mask，shape为(total_frame*per_frame_length)
        get_mask = lambda x: motion_mask[x*self.per_frame_length:(x+1)*self.per_frame_length] # 获取第x帧的mask
        self.mi = {} # motion index
        for k in range(self.total_frame):
            # nearby 5 frames
            current_mask = get_mask(k) 
            for i in range(1,6): 
                if k-i >= 0: 
                    current_mask = current_mask|get_mask(k-i) 
                if k+i < self.total_frame: 
                    current_mask = current_mask|get_mask(k+i) 
            mask_idx = current_mask.nonzero() 
            if len(mask_idx)>0: 
                self.mi[k] = mask_idx[:,0] 
            else:
                self.mi[k] = []
        self.motion_num = self.batch//10

    def nextids(self):
        # self.curr+=self.batch
        # if self.curr + self.batch > self.total:
        #     self.ids = self.gen_permute()
        #     self.curr = 0
        # return self.ids[self.curr:self.curr+self.batch]
        frame = int(random.random()*self.total_frame)
        start = int(random.random()*(len(self.permute_base)))
        m_num = len(self.mi[frame])
        if m_num > 0:
            if m_num < self.motion_num:
                m_idx = self.mi[frame][torch.randperm(self.motion_num)%m_num]
            else:
                m_idx = self.mi[frame][torch.randperm(m_num)[:self.motion_num]]
        else:
            m_idx = self.permute_base[:1]
        # start = int(random.random()*(len(self.permute_base)-self.batch+len(m_idx)))
        end = min(start+self.batch-len(m_idx), self.per_frame_length)
        return  torch.cat([m_idx, self.permute_base[start:end]],0)+frame*self.per_frame_length


def render_results(args):
    
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, 
                            is_stack=False, num_frames=args.num_frames, frame_list=args.frame_list)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, 
                            is_stack=True, num_frames=args.num_frames, frame_list=args.frame_list)
    white_bg = train_dataset.white_bg
    ndc_ray = args.ndc_ray
    
    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return
    
    ckpt = torch.load(args.ckpt, map_location=device)
    print(f'ckpt loaded from {args.ckpt}')
    kwargs = ckpt['kwargs']
    kwargs.update({'device':device})
    aabb = test_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_final, aabb)
    tensorf = eval(args.model_name)(
        aabb, reso_cur, device,
        density_n_comp=args.n_lamb_sigma, appearance_n_comp=args.n_lamb_sh, app_dim=args.data_dim_color, near_far=test_dataset.near_far, 
        shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, 
        distance_scale=args.distance_scale, pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, 
        featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct, 
        num_frames=args.num_frames, ld_per_frame=args.ld_per_frame, 
        deform_field=args.deform_field, portion_decoder=args.portion_decoder, 
        virtual_cannonical=args.virtual_cannonical, target_portion=args.target_portion if args.target_portion else [0,0,1], share_portion_embeddings=args.share_portion_embeddings, portion_weight=args.portion_weight)
        
    tensorf.load(ckpt)
    
    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    W, H = train_dataset.img_wh
    num_frame = len(train_dataset.frame_list) 
    num_cam = len(train_dataset.poses)
    assert num_frame*num_cam*W*H == allrgbs.shape[0] == allrays.shape[0]
    
    save_results = './ablation_coffee_original_render'
    os.makedirs(save_results, exist_ok=True)
    
    for frame_idx in range(num_frame, 0, -1):
        for cam_idx in range(num_cam):
            if os.path.exists(os.path.join(save_results, f'{frame_idx}_{cam_idx}.png')):
                continue
            
            sample_rays = allrays.view(num_frame, num_cam, H*W, -1)[frame_idx][cam_idx]
            with torch.no_grad():
                rgb_map, _, _, _, _, _ = renderer(sample_rays, tensorf, chunk=4096, N_samples=-1, ndc_ray=ndc_ray, white_bg=white_bg, is_train=False, device=device)
            rgb_map = rgb_map.view(H, W, 3).permute(2, 0, 1)
         
            torchvision.utils.save_image(rgb_map, os.path.join(save_results, f'{frame_idx}_{cam_idx}.png'))
   

@torch.no_grad()
def render_test(args):
    # init dataset
    dataset = dataset_dict[args.dataset_name]
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, 
                            is_stack=True, num_frames=args.num_frames, frame_list=args.frame_list)
    white_bg = test_dataset.white_bg
    ndc_ray = args.ndc_ray

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!')
        return

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    # tensorf = eval(args.model_name)(**kwargs)
    aabb = test_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_final, aabb)
    tensorf = eval(args.model_name)(
        aabb, reso_cur, device,
        density_n_comp=args.n_lamb_sigma, appearance_n_comp=args.n_lamb_sh, app_dim=args.data_dim_color, near_far=test_dataset.near_far, 
        shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, 
        distance_scale=args.distance_scale, pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, 
        featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct, 
        num_frames=args.num_frames, ld_per_frame=args.ld_per_frame, 
        # deform
        deform_field=args.deform_field, portion_decoder=args.portion_decoder, 
        virtual_cannonical=args.virtual_cannonical, target_portion=args.target_portion if args.target_portion else [0,0,1], 
        share_portion_embeddings=args.share_portion_embeddings, portion_weight=args.portion_weight)

    tensorf.load(ckpt)

    logfolder = os.path.dirname(args.ckpt)
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/{args.expname}/imgs_test_all', exist_ok=True)
        evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/{args.expname}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

    if args.render_path:
        if ndc_ray:
            tensorf.near_far = [0.2,1]
        c2ws = test_dataset.render_path
        os.makedirs(f'{logfolder}/{args.expname}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/{args.expname}/imgs_path_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        if ndc_ray:
            tensorf.near_far = [0,1]
   
if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    args.datadir = os.path.expanduser(args.datadir)
    args.frame_list = eval(args.frame_list)
    args = mmengine.Config(vars(args))
    if args.cfg_options is not None:
        args.merge_from_dict(args.cfg_options)
    args.deform_field = None
    args.portion_decoder = None
    print(args)
        
    targs = mmengine.Config.fromfile(f'{os.path.dirname(args.ckpt)}/config.py')
    targs.merge_from_dict(args)
    
    render_results(args)
    

