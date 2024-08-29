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

import mmcv
import configargparse
import argparse
from einops import rearrange
import math

import threading

from dataLoader.ray_utils import *
from ip2p_sequence import SequenceInstructPix2Pix
from ip2p_models.RAFT.raft import RAFT
from ip2p_models.RAFT.utils.utils import InputPadder

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
    parser.add_argument('--keyframe_refine_diffusion_steps', type=int, default=10,
                        help='number of diffusion steps to take for keyframe refinement')
    parser.add_argument('--keyframe_refine_num_steps', type=int, default=700,
                        help='number of denoise steps to take for keyframe refinement')
    parser.add_argument('--restview_refine_diffusion_steps', type=int, default=10,
                        help='number of diffusion steps to take for keyframe refinement')
    parser.add_argument('--restview_refine_num_steps', type=int, default=800,
                        help='number of denoise steps to take for keyframe refinement')
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
    parser.add_argument('--raft_ckpt', type=str, default='./ip2p_models/weights/raft-things.pth',)
    
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
    parser.add_argument('--cfg_options', nargs='+', action=mmcv.DictAction,)
    if cmd is not None:
        return parser.parse_args(cmd)
    else:
        return parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

renderer = OctreeRender_trilinear_fast

enhomo0 = lambda x: torch.cat([x, torch.ones([1, *x.shape[1:]], device=x.device, dtype=x.dtype)], dim=0)
dehomo0 = lambda x: x[:-1] / x[-1:]

def warp_pts_AfromB(ptsA, intriB, w2cB):
    """
    Args:
        pts: (H, W, 3)
        intrinsics: (3, 3)
        w2c: (4, 4)
    """
    H, W, _ = ptsA.shape
    x = ptsA.reshape(-1, 3).T # (3, H*W)
    x = dehomo0(w2cB @ enhomo0(x)) # (3, H*W)
    x = intriB @ x # (3, H*W)
    pixB = dehomo0(x).T # (H*W, 3)
    pixB = pixB.reshape(H, W, 2)
    
    return pixB

def apply_warp(
    AfromB: torch.Tensor, 
    imgA: torch.Tensor, 
    imgB: torch.Tensor,
    ptsA: torch.Tensor,
    ptsB: torch.Tensor, 
    u=None, d=None, l=None, r=None, 
    diff_thres = 0.2, 
    default = torch.tensor([0, 0, 0]) 
):
    """Warp imgB to imgA
    """
    u, d, l, r = u or 0, d or imgA.shape[0], l or 0, r or imgA.shape[1]
    default = default.to(dtype=imgA.dtype, device=imgA.device)
    
    Y, X = (AfromB - 0.5).unbind(-1) # (H, W)
    mask = (u <= X) & (X <= d - 1) & (l <= Y) & (Y <= r - 1) 
    X = ((X - u) / (d - 1 - u) * 2 - 1) * mask 
    Y = ((Y - l) / (r - 1 - l) * 2 - 1) * mask 
    pix = torch.stack([-Y, X], dim=-1).unsqueeze(0) 
    imgA = F.grid_sample(imgB[None, u:d, l:r].permute(0, 3, 1, 2), pix, mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0)
    pts_AfromB = F.grid_sample(ptsB[None, u:d, l:r].permute(0, 3, 1, 2), pix, mode='bilinear', align_corners=True).squeeze(0).permute(1, 2, 0) 
    diff = (ptsA - pts_AfromB).norm(dim=-1) # (H, W)
    
    mask &= (diff < diff_thres)
    imgA[~mask] = default 
    
    return imgA, mask, diff

def warp_flow(img, flow): 
    # warp image according to flow
    h, w = flow.shape[:2]
    flow_new = flow.copy() 
    flow_new[:, :, 0] += np.arange(w) 
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis] 

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res

def compute_bwd_mask(fwd_flow, bwd_flow):
    # compute the backward mask
    alpha_1 = 0.5 
    alpha_2 = 0.5

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return bwd_mask


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
        self.per_frame_length = self.total / self.total_frame 
        assert self.per_frame_length.is_integer()
        self.per_frame_length = int(self.per_frame_length) 
        self.permute_base = torch.LongTensor(np.random.permutation(self.per_frame_length)) 
        motion_mask = (allrgbs-torch.roll(allrgbs,self.per_frame_length,0)).abs().mean(-1)>(10/255) 
        get_mask = lambda x: motion_mask[x*self.per_frame_length:(x+1)*self.per_frame_length] 
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


def nerf_editing(args):
    # init log file
    if args.add_timestamp:
        logfolder = f'{args.basedir}/{args.expname}{datetime.datetime.now().strftime("-%Y%m%d-%H%M%S")}'
    else:
        logfolder = f'{args.basedir}/{args.expname}'
    print(f'logfolder {logfolder}')
    os.makedirs(logfolder, exist_ok=True)
    args.dump(os.path.join(logfolder, 'config.py'))
    os.makedirs(f'{logfolder}/imgs_vis', exist_ok=True)
    summary_writer = SummaryWriter(logfolder)
    
    tag_prompt = args.prompt.split(' ')[-1].replace('?','')
    tag = f'{datetime.datetime.now().strftime("%d-%H-%M")}_{tag_prompt}'

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, 
                            is_stack=False, num_frames=args.num_frames, frame_list=args.frame_list)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, 
                            is_stack=True, num_frames=args.num_frames, frame_list=args.frame_list)
    white_bg = train_dataset.white_bg
    ndc_ray = args.ndc_ray
    
    os.makedirs(os.path.join(args.cache, args.expname), exist_ok=True) # cache folder
    print(f'cache folder {os.path.join(args.cache, args.expname)}')

    if not os.path.exists(args.ckpt):
        print('the ckpt path does not exists!!!')
        return
    
    ckpt = torch.load(args.ckpt, map_location=device)
    print(f'ckpt loaded from {args.ckpt}')
    kwargs = ckpt['kwargs']
    kwargs.update({'device':device})
    aabb = test_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_final, aabb)
    
    nSamples = min(args.nSamples, cal_n_samples(reso_cur, args.step_ratio))
    
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
    
    ip2p = SequenceInstructPix2Pix(device=args.ip2p_device, ip2p_use_full_precision=args.ip2p_use_full_precision)
    
    raft = torch.nn.DataParallel(RAFT(args=argparse.Namespace(small=False, mixed_precision=False)))
    raft.load_state_dict(torch.load(args.raft_ckpt))
    raft = raft.module
    
    raft = raft.to(args.ip2p_device)
    raft.requires_grad_(False)
    raft.eval()
    print('RAFT loaded!')
    
    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    W, H = train_dataset.img_wh
    num_frame = len(train_dataset.frame_list) 
    num_cam = len(train_dataset.poses)
    assert num_frame*num_cam*W*H == allrgbs.shape[0] == allrays.shape[0]
    
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = MotionSampler(allrgbs, args.num_frames, args.batch_size)
    simpleSampler = SimpleSampler(allrgbs.shape[0], args.num_frames, args.batch_size)
    
    Ortho_reg_weight = args.Ortho_weight
    print("initial Ortho_reg_weight", Ortho_reg_weight)

    L1_reg_weight = args.L1_weight_inital
    print("initial L1_reg_weight", L1_reg_weight)
    TV_weight_density, TV_weight_app = args.TV_weight_density, args.TV_weight_app
    tvreg = TVLoss()
    print(f"initial TV_weight density: {TV_weight_density} appearance: {TV_weight_app}")
    def save_ckpt():
        for f in os.listdir(f'{logfolder}'):
            if '.th' in f:
                os.remove(f'{logfolder}/{f}')
                print(f'removed {logfolder}/{f}')
        tensorf.save(f'{logfolder}/ckpt-{iteration}.th')
        print(f'ckpt saved: {logfolder}/ckpt-{iteration}.th')
        
    original_rgbs = allrgbs.clone().view(num_frame, num_cam, H, W, 3) # (num_frame, num_cam, H, W, 3)
    cam_idxs = list(range(0, num_cam)) 
    # thread lock for data and model
    data_lock = threading.Lock()
    model_lock = threading.Lock()

    cache_pts_path = os.path.join(args.cache, args.expname, 'pts_all.pt')
    if os.path.exists(cache_pts_path):
        pts_all = torch.load(cache_pts_path).cpu()
        print('all pts loaded from cache dir')
        torch.cuda.empty_cache()
    else:
        key_frame_index = 0
        if os.path.exists(os.path.join(args.cache, args.expname, 'depth_all.pt')):
            key_frame_depth_maps = torch.load(os.path.join(args.cache, args.expname, 'depth_all.pt')).cpu()
            print('all depth maps loaded from cache dir')
            torch.cuda.empty_cache()
        else:
            key_frame_depth_maps = []
            for j in range(num_cam):
                rays = allrays.view(num_frame, num_cam, H*W, -1)[key_frame_index][j] # (H*W, 7)
                with torch.no_grad():
                    rays = rays.to(device)
                    _, _, depth_map, _, _, _ = renderer(rays, tensorf, chunk=2048, N_samples=-1, ndc_ray=ndc_ray, white_bg=white_bg, device=device)
                depth_map = depth_map.view(-1, 1).cpu() # (H*W, 1)
                key_frame_depth_maps.append(depth_map)
            key_frame_depth_maps = torch.stack(key_frame_depth_maps, dim=0).cpu() # (num_cam, H*W, 1)
            torch.save(key_frame_depth_maps, os.path.join(args.cache, args.expname, f'depth_all.pt'))
            print('all depth maps saved to cache dir')
            del depth_map, rays
            torch.cuda.empty_cache()
            
        pts_all = []
        for j in range(num_cam):
            rays = allrays.view(num_frame, num_cam, H*W, -1)[key_frame_index][j] # (H*W, 7)
            depth = key_frame_depth_maps[j] # (H*W, 1)
            pts = rays[..., :3] + rays[..., 3:6] * depth # (H*W, 3)
            # ---------------project from ndc space to world space----------------------------
            pts[..., 2] = 2 / (pts[..., 2] - 1)
            pts[..., 0] = -pts[..., 0] * pts[..., 2] * W / 2 / train_dataset.focal[0]
            pts[..., 1] = -pts[..., 1] * pts[..., 2] * H / 2 / train_dataset.focal[1]
            # --------------------------------------------------------------------------------
            pts = pts.view(H, W, 3)
            pts_all.append(pts)
        pts_all = torch.stack(pts_all, dim=0) # (num_cam, H, W, 3)
        torch.save(pts_all, os.path.join(args.cache, args.expname, f'pts_all.pt'))
        del key_frame_depth_maps, pts
        torch.cuda.empty_cache()
    
    def key_frame_update(key_frame:int = 0, warp_ratio:float = 0.5, warm_up_steps:int = 10): 
        print(f'key frame {key_frame} editing')  
        for warm_up_idx in range(warm_up_steps):
            sample_idxs = sorted(list(np.random.choice(cam_idxs, args.sequence_length, replace=False)))
            remain_idxs = sorted(list(set(cam_idxs) - set(sample_idxs)))
            
            sample_images = allrgbs.view(num_frame, num_cam, H, W, -1)[key_frame][sample_idxs].to(device) 
            sample_images = rearrange(sample_images, 'f H W C -> f C H W') # (sample_length, C, H, W)
            sample_images_cond = original_rgbs[key_frame][sample_idxs].to(device) # (sample_length, H, W, 3)
            sample_images_cond = rearrange(sample_images_cond, 'f H W C -> f C H W')
            
            remain_images = allrgbs.view(num_frame, num_cam, H, W, -1)[key_frame][remain_idxs].to(device) 
            remain_images = rearrange(remain_images, 'f H W C -> f C H W') # (remain_length, C, H, W)
            remain_images_cond = original_rgbs[key_frame][remain_idxs].to(device) # (remain_length, H, W, 3)
            remain_images_cond = rearrange(remain_images_cond, 'f H W C -> f C H W') # (remain_length, 3, H, W)
            
            torchvision.utils.save_image(sample_images, f'{tag}_sample_images.png', nrow=sample_images.shape[0], normalize=True)
            torchvision.utils.save_image(sample_images_cond, f'{tag}_sample_images_cond.png', nrow=sample_images_cond.shape[0], normalize=True)
            
            # cosine annealing
            scale_min, scale_max = 0.8, 1.0
            scale = scale_min + 0.5 * (1 + math.cos(math.pi * warm_up_idx / warm_up_steps)) * (scale_max - scale_min)
            sample_images_edit = ip2p.edit_sequence(
                images=sample_images.unsqueeze(0), # (1, seq_len, C, H, W)
                images_cond=sample_images_cond.unsqueeze(0), # (1, seq_len, C, H, W)
                guidance_scale=args.guidance_scale,
                image_guidance_scale=args.image_guidance_scale,
                diffusion_steps=int(args.diffusion_steps * scale),
                prompt=args.prompt,
                noisy_latent_type="noisy_latent",
                T=int(1000 * scale),
            ) # (1, C, f, H, W)
            
            sample_images_edit = rearrange(sample_images_edit, '1 C f H W -> f C H W').to(device, dtype=torch.float32) # (f, C, H, W)
            if sample_images_edit.shape[-2:] != (H, W):
                sample_images_edit = F.interpolate(sample_images_edit, size=(H, W), mode='bilinear', align_corners=False)
                
            # warp average among sample images
            for idx_cur, i in enumerate(sample_idxs):
                warp_average = torch.zeros((H, W, 3), dtype=torch.float32, device=device)
                weights_mask = torch.zeros((H, W), dtype=torch.float32, device=device)
                for idx_ref, j in enumerate(sample_idxs):
                    intrinsic_ref = train_dataset.intrinsics[j] 
                    extrinsic_ref = train_dataset.extrinsics[j]
                    warp_cur_from_ref = warp_pts_AfromB(pts_all[i], intrinsic_ref, extrinsic_ref).to(device) # (H, W, 2) 
                    pts_ref = pts_all[j].to(device) # (H, W, 3)
                    pts_cur = pts_all[i].to(device) # (H, W, 3)
                    image_ref = sample_images_edit[idx_ref].permute(1, 2, 0).float() # (H, W, 3)
                    image_cur = sample_images_edit[idx_cur].permute(1, 2, 0).float() # (H, W, 3)
                    
                    warp, mask, diff = apply_warp(warp_cur_from_ref, image_cur, image_ref, pts_cur, pts_ref, diff_thres=0.05)
                    weight = (mask!=0).sum() / (mask).numel()
                    warp_average[mask] += warp[mask] * weight
                    weights_mask[mask] += weight
                    
                average_mask = (weights_mask!=0)
                warp_average[average_mask] /= weights_mask[average_mask].unsqueeze(-1)
                sample_images_edit[idx_cur].permute(1, 2, 0)[average_mask] = warp_average[average_mask]
                
            torchvision.utils.save_image(sample_images_edit, f'{tag}_sample_images_edit.png', nrow=sample_images_edit.shape[0], normalize=True)
            torchvision.utils.save_image(remain_images, f'{tag}_remain_images.png', nrow=remain_images.shape[0], normalize=True)
                
            remain_images_warped = remain_images.clone()
            for idx_cur, i in enumerate(remain_idxs):
                warp_average = torch.zeros((H, W, 3), dtype=torch.float32, device=device) # (H, W, 3)
                weights_mask = torch.zeros((H, W), dtype=torch.float32, device=device) # (H, W)
                for idx_ref, j in enumerate(sample_idxs):
                    intrinsic_ref = train_dataset.intrinsics[j] 
                    extrinsic_ref = train_dataset.extrinsics[j] 
                    warp_cur_from_ref = warp_pts_AfromB(pts_all[i], intrinsic_ref, extrinsic_ref).to(device) # (H, W, 2)
                    pts_ref = pts_all[j].to(device) # (H, W, 3)
                    pts_cur = pts_all[i].to(device) # (H, W, 3)
                    image_ref = sample_images_edit[idx_ref].permute(1, 2, 0).float() # (H, W, 3)
                    image_cur = remain_images[idx_cur].permute(1, 2, 0).float() # (H, W, 3)
                    warp, mask, diff = apply_warp(warp_cur_from_ref, image_cur, image_ref, pts_cur, pts_ref, diff_thres=0.02)
                    weight = (mask!=0).sum() / (mask).numel() 
                    warp_average[mask] += warp[mask] * weight
                    weights_mask[mask] += weight
                    
                average_mask = (weights_mask!=0) # (H, W)
                warp_average[average_mask] /= weights_mask[average_mask].unsqueeze(-1) # (H, W, 3)
                remain_images_warped[idx_cur].permute(1, 2, 0)[average_mask] = warp_average[average_mask] * warp_ratio + remain_images[idx_cur].permute(1, 2, 0)[average_mask] * (1-warp_ratio)
                
            torchvision.utils.save_image(remain_images_warped, f'{tag}_remain_images_warped.png', nrow=remain_images_warped.shape[0], normalize=True)
            
            if warm_up_idx == warm_up_steps-1:
                for i in range(0, remain_images_warped.shape[0], args.sequence_length):
                    
                    anchor_idx = min(i, remain_images_warped.shape[0]-1)
                    anchor_image = remain_images_warped[anchor_idx].unsqueeze(0) # (1, C, H, W)
                    anchor_image_cond = remain_images_cond[anchor_idx].unsqueeze(0) # (1, C, H, W)
                    
                    start_idx = i
                    end_idx = min(i+args.sequence_length, remain_images_warped.shape[0])
                    selected_remain_images_warped = remain_images_warped[start_idx:end_idx] # (seq_len, H, W, 3)
                    selected_remain_images_cond = remain_images_cond[start_idx:end_idx] # (seq_len, 3, H, W)
                    
                    images_input = torch.cat([anchor_image, selected_remain_images_warped], dim=0) 
                    images_cond = torch.cat([anchor_image_cond, selected_remain_images_cond], dim=0)
                    
                    images_edit = ip2p.edit_sequence(
                        images=images_input.unsqueeze(0), # (1, seq_len+1, C, H, W)
                        images_cond=images_cond.unsqueeze(0), # (1, seq_len+1, C, H, W)
                        guidance_scale=args.guidance_scale,
                        image_guidance_scale=args.image_guidance_scale,
                        diffusion_steps=args.restview_refine_diffusion_steps,
                        prompt=args.prompt,
                        noisy_latent_type="noisy_latent",
                        T=args.restview_refine_num_steps,
                    ) # (1, C, f, H, W)
                    
                    images_edit = rearrange(images_edit, '1 C f H W -> f C H W').to(device, dtype=torch.float32) # (f, C, H, W)
                    if images_edit.shape[-2:] != (H, W):
                        images_edit = F.interpolate(images_edit, size=(H, W), mode='bilinear', align_corners=False)
                    
                    images_edit = images_edit[1:] # (seq_len, C, H, W)
                    remain_images_warped[start_idx:end_idx] = images_edit
                    
                torchvision.utils.save_image(remain_images_warped, f'{tag}_remain_images_warped_refined.png', nrow=remain_images_warped.shape[0], normalize=True)
            
            sample_images_update = rearrange(sample_images_edit, 'f C H W -> f H W C').to(allrgbs)
            allrgbs.view(num_frame, num_cam, H, W, -1)[key_frame][sample_idxs] = sample_images_update
            remain_images_update = rearrange(remain_images_warped, 'f C H W -> f H W C').to(allrgbs)  
            allrgbs.view(num_frame, num_cam, H, W, -1)[key_frame][remain_idxs] = remain_images_update
                
    key_frame_update(key_frame=0, warp_ratio=0.5, warm_up_steps=12)
    print('Key frame editing done!')
    
    keyframe_images = allrgbs.view(num_frame, num_cam, H, W, -1)[0] # (num_cam, H, W, 3)
    keyframe_images = rearrange(keyframe_images, 'f H W C -> f C H W') # (num_cam, 3, H, W)
    torchvision.utils.save_image(keyframe_images, f'{tag}_keyframe_images.png', nrow=args.sequence_length, normalize=True)
        
    def all_frame_update(key_frame:int = 0):  
        for cam_idx in range(num_cam):
            
            # edited key frame
            keyframe_image = allrgbs.view(num_frame, num_cam, H, W, -1)[key_frame, cam_idx] # (H, W, 3)
            keyframe_image = rearrange(keyframe_image, 'H W C -> 1 C H W') # (1, C, H, W)
            keyframe_image_cond = original_rgbs[key_frame, cam_idx] # (H, W, 3)
            keyframe_image_cond = rearrange(keyframe_image_cond, 'H W C -> 1 C H W') # (1, C, H, W)
            
            for frame_idx in range(0, num_frame, args.sequence_length):
                start_idx = frame_idx
                end_idx = min(frame_idx+args.sequence_length, num_frame)
                selected_frame_idxs = list(range(start_idx, end_idx))
                sequence_length = len(selected_frame_idxs)
                
                images = allrgbs.view(num_frame, num_cam, H, W, -1)[selected_frame_idxs, cam_idx] # (f, H, W, 3)
                images_condition = original_rgbs[selected_frame_idxs, cam_idx] # (f, H, W, 3)
                
                for i in range(0, sequence_length):
                    if start_idx == 0 and i == 0:
                        continue
                    ref_idx = max(start_idx-1, 0)
                    ref_image = allrgbs.view(num_frame, num_cam, H, W, -1)[ref_idx, cam_idx] # (H, W, 3)
                    ref_image = rearrange(ref_image, 'H W C -> 1 C H W') # (1, C, H, W)
                    ref_image_cond = original_rgbs[ref_idx, cam_idx] # (H, W, 3)
                    ref_image_cond = rearrange(ref_image_cond, 'H W C -> 1 C H W') # (1, C, H, W)
                    cur_image = rearrange(images[i], 'H W C -> 1 C H W', H=H, W=W) # (1, C, H, W)
                    cur_image_cond = rearrange(images_condition[i], 'H W C -> 1 C H W', H=H, W=W) # (1, C, H, W)
                    
                    ref_image = (ref_image * 255.0).float().to(args.ip2p_device)
                    ref_image_cond = (ref_image_cond * 255.0).float().to(args.ip2p_device)
                    cur_image = (cur_image * 255.0).float().to(args.ip2p_device)
                    cur_image_cond = (cur_image_cond * 255.0).float().to(args.ip2p_device)
                    
                    padder = InputPadder(cur_image.shape)
                    ref_image, ref_image_cond, cur_image, cur_image_cond = padder.pad(ref_image, ref_image_cond, cur_image, cur_image_cond)
                    
                    _, flow_fwd_ref = raft(ref_image_cond, cur_image_cond, iters=20, test_mode=True) 
                    _, flow_bwd_ref = raft(cur_image_cond, ref_image_cond, iters=20, test_mode=True)
                    
                    flow_fwd_ref = padder.unpad(flow_fwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
                    flow_bwd_ref = padder.unpad(flow_bwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
                    
                    ref_image = padder.unpad(ref_image[0]).cpu().numpy().transpose(1, 2, 0).astype(np.uint8) 
                    cur_image = padder.unpad(cur_image[0]).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                    
                    mask_bwd_ref = compute_bwd_mask(flow_fwd_ref, flow_bwd_ref) # (h, w)
                    warp_cur_from_ref_proj = warp_flow(ref_image, flow_bwd_ref) # (h, w, c)
                    
                    warp_image = warp_cur_from_ref_proj * mask_bwd_ref[..., None] + cur_image * (1 - mask_bwd_ref[..., None]) # (h, w, c)
                    
                    warp_image = torch.from_numpy(warp_image / 255.0).to(images)
                    if warp_image.shape[:2] != (H, W):
                        warp_image = rearrange(warp_image, 'H W C -> 1 C H W')
                        warp_image = F.interpolate(warp_image, size=(H, W), mode='bilinear',align_corners=False)
                        warp_image = rearrange(warp_image, '1 C H W -> H W C')
                        
                    images[i] = warp_image # (H, W, 3)
                
                images = rearrange(images, 'f H W C -> f C H W') # (f, C, H, W)
                images_condition = rearrange(images_condition, 'f H W C -> f C H W') # (f, C, H, W)
                torchvision.utils.save_image(images, f'{tag}_images_flow_warped.png', nrow=images.shape[0], normalize=True)
                torchvision.utils.save_image(images_condition, f'{tag}_images_flow_cond.png', nrow=images_condition.shape[0], normalize=True)
                
                images = torch.cat([keyframe_image, images], dim=0) # (f+1, C, H, W)
                images_condition = torch.cat([keyframe_image_cond, images_condition], dim=0) # (f+1, C, H, W)
                
                images_flow = ip2p.edit_sequence(
                    images=images.unsqueeze(0).to(args.ip2p_device), # (1, f+1, C, H, W)
                    images_cond=images_condition.unsqueeze(0).to(args.ip2p_device), # (1, f+1, C, H, W)
                    guidance_scale=args.guidance_scale,
                    image_guidance_scale=args.image_guidance_scale,
                    diffusion_steps=args.refine_diffusion_steps,
                    prompt=args.prompt,
                    noisy_latent_type="noisy_latent",
                    T=args.refine_num_steps,
                ) # (1, C, f+1, H, W)
                
                images_flow = rearrange(images_flow, '1 C f H W -> f C H W').cpu().to(allrgbs.dtype)
                images_flow = images_flow[1:] # (f, C, H, W)
                
                if images_flow.shape[-2:] != (H, W):
                    images_flow = F.interpolate(images_flow, size=(H, W), mode='bilinear', align_corners=False)
                    
                torchvision.utils.save_image(images_flow, f'{tag}_images_flow_refine.png', nrow=images_flow.shape[0], normalize=True)
                
                images_update = rearrange(images_flow, 'N C H W -> N H W C').to(allrgbs)
                with data_lock:
                    allrgbs.view(num_frame, num_cam, H, W, -1)[selected_frame_idxs, cam_idx] = images_update
                    
    thread_all_frames_update = threading.Thread(target=all_frame_update, name='dataset_update')
    thread_all_frames_update.start()
    print('all frame update thread started')
    
    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        if iteration < 0.5 * args.n_iters:
            ray_idx = trainingSampler.nextids()
        else:
            ray_idx = simpleSampler.nextids()
            
        with data_lock:
            rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)
            
        #rgb_map, alphas_map, depth_map, weights, uncertainty
        with model_lock:
            rgb_map, alphas_map, depth_map, weights, uncertainty, extra_loss = renderer(
                rays_train, tensorf, chunk=args.batch_size,
                N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, device=device, is_train=True)

        loss = torch.mean((rgb_map - rgb_train) ** 2) + extra_loss
        
        # loss
        total_loss = loss
        if Ortho_reg_weight > 0:
            loss_reg = tensorf.vector_comp_diffs()
            total_loss += Ortho_reg_weight*loss_reg
            summary_writer.add_scalar('train/reg', loss_reg.detach().item(), global_step=iteration)
        if L1_reg_weight > 0:
            loss_reg_L1 = tensorf.density_L1()
            total_loss += L1_reg_weight*loss_reg_L1
            summary_writer.add_scalar('train/reg_l1', loss_reg_L1.detach().item(), global_step=iteration)
        if TV_weight_density>0:
            TV_weight_density *= lr_factor
            loss_tv = tensorf.TV_loss_density(tvreg) * TV_weight_density
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_density', loss_tv.detach().item(), global_step=iteration)
        if TV_weight_app>0:
            TV_weight_app *= lr_factor
            loss_tv = loss_tv + tensorf.TV_loss_app(tvreg)*TV_weight_app
            total_loss = total_loss + loss_tv
            summary_writer.add_scalar('train/reg_tv_app', loss_tv.detach().item(), global_step=iteration)
        if args.feat_diff_weight>0:
            total_loss +=  tensorf.feat_diff_loss(
                torch.randint(0,tensorf.num_frames,[1]).item()/tensorf.num_frames
            )*args.feat_diff_weight

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        loss = loss.detach().item()
        
        PSNRs.append(-10.0 * np.log(loss) / np.log(10.0))
        summary_writer.add_scalar('train/PSNR', PSNRs[-1], global_step=iteration)
        summary_writer.add_scalar('train/mse', loss, global_step=iteration)

        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * lr_factor

        # Print the current values of the losses.
        if iteration % args.progress_refresh_rate == 0:
            pbar.set_description(
                f'Iteration {iteration:05d}:'
                + f' train_psnr = {float(np.mean(PSNRs)):.2f}'
                + f' test_psnr = {float(np.mean(PSNRs_test)):.2f}'
                + f' mse = {loss:.6f}'
            )
            PSNRs = []
        if iteration % args.vis_every == args.vis_every - 1 and args.N_vis!=0:
            PSNRs_test = evaluation(test_dataset, tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis, prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)
            c2ws = test_dataset.render_path
            evaluation_path(test_dataset, tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray, device=device)
        if iteration % (args.n_iters//5) == 0: # save at the first iter to make sure disk space available
            save_ckpt()

    thread_all_frames_update.join()
    save_ckpt()

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset, tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_test:
        os.makedirs(f'{logfolder}/imgs_test_all', exist_ok=True)
        PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_test_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        summary_writer.add_scalar('test/psnr_all', np.mean(PSNRs_test), global_step=iteration)
        print(f'======> {args.expname} test all psnr: {np.mean(PSNRs_test)} <========================')

    if args.render_path:
        c2ws = test_dataset.render_path
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)

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
            
    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
                                N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)
        print(f'======> {args.expname} train all psnr: {np.mean(PSNRs_test)} <========================')
            

if __name__ == '__main__':

    torch.set_default_dtype(torch.float32)
    torch.manual_seed(20211202)
    np.random.seed(20211202)

    args = config_parser()
    args.datadir = os.path.expanduser(args.datadir)
    args.frame_list = eval(args.frame_list)
    args = mmcv.Config(vars(args))
    if args.cfg_options is not None:
        args.merge_from_dict(args.cfg_options)
    args.deform_field = None
    args.portion_decoder = None
    print(args)
        
    targs = mmcv.Config.fromfile(f'{os.path.dirname(args.ckpt)}/config.py')
    targs.merge_from_dict(args)
    
    if args.render_only and (args.render_test or args.render_path):
        render_test(targs)
    else:
        nerf_editing(args)
    

