import os
from tqdm.auto import tqdm

import json, random
from stream_renderer import *
from utils import *
from torch.utils.tensorboard import SummaryWriter
import datetime

from dataLoader import dataset_dict
import sys

import mmengine
import configargparse

def config_parser(cmd=None):
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./log',
                        help='where to store ckpts and logs')
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
    # loader options
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--n_iters", type=int, default=30000)
    parser.add_argument('--dataset_name', type=str, default='blender', choices=['n3dv_dynamic','deepview_dynamic',])
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


@torch.no_grad()
def export_mesh(args):

    ckpt = torch.load(args.ckpt, map_location=device)
    kwargs = ckpt['kwargs']
    kwargs.update({'device': device})
    tensorf = eval(args.model_name)(**kwargs)
    tensorf.load(ckpt)

    alpha,_ = tensorf.getDenseAlpha()
    convert_sdf_samples_to_ply(alpha.cpu(), f'{args.ckpt[:-3]}.ply',bbox=tensorf.aabb.cpu(), level=0.005)

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

def reconstruction(args):
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

    # init dataset
    dataset = dataset_dict[args.dataset_name]
    train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, 
                            is_stack=False, num_frames=args.num_frames, frame_list=args.frame_list)
    test_dataset = dataset(args.datadir, split='test', downsample=args.downsample_train, 
                            is_stack=True, num_frames=args.num_frames, frame_list=args.frame_list)
    white_bg = train_dataset.white_bg
    near_far = train_dataset.near_far
    ndc_ray = args.ndc_ray

    # init resolution
    upsamp_list = args.upsamp_list
    update_AlphaMask_list = args.update_AlphaMask_list
    n_lamb_sigma = args.n_lamb_sigma
    n_lamb_sh = args.n_lamb_sh
    
    # init parameters
    # tensorVM, renderer = init_parameters(args, train_dataset.scene_bbox.to(device), reso_list[0])
    aabb = train_dataset.scene_bbox.to(device)
    reso_cur = N_to_reso(args.N_voxel_init, aabb)
    nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
    
    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt, map_location=device)
        kwargs = ckpt['kwargs']
        kwargs.update({'device':device})
        tensorf = eval(args.model_name)(**kwargs)
        tensorf.load(ckpt)
    else:
        tensorf = eval(args.model_name)(
            aabb, reso_cur, device,
            density_n_comp=n_lamb_sigma, appearance_n_comp=n_lamb_sh, app_dim=args.data_dim_color, near_far=near_far, 
            shadingMode=args.shadingMode, alphaMask_thres=args.alpha_mask_thre, density_shift=args.density_shift, 
            distance_scale=args.distance_scale, pos_pe=args.pos_pe, view_pe=args.view_pe, fea_pe=args.fea_pe, 
            featureC=args.featureC, step_ratio=args.step_ratio, fea2denseAct=args.fea2denseAct, 
            num_frames=args.num_frames, ld_per_frame=args.ld_per_frame, 
            # deform
            deform_field=args.deform_field, portion_decoder=args.portion_decoder, 
            virtual_cannonical=args.virtual_cannonical, target_portion=args.target_portion if args.target_portion else [0,0,1], 
            share_portion_embeddings=args.share_portion_embeddings, portion_weight=args.portion_weight)

    grad_vars = tensorf.get_optparam_groups(args.lr_init, args.lr_basis)
    if args.lr_decay_iters > 0:
        lr_factor = args.lr_decay_target_ratio**(1/args.lr_decay_iters)
    else:
        args.lr_decay_iters = args.n_iters
        lr_factor = args.lr_decay_target_ratio**(1/args.n_iters)

    print("lr decay", args.lr_decay_target_ratio, args.lr_decay_iters)
    
    optimizer = torch.optim.Adam(grad_vars, betas=(0.9,0.99))

    #linear in logrithmic space
    N_voxel_list = (torch.round(torch.exp(torch.linspace(np.log(args.N_voxel_init), np.log(args.N_voxel_final), len(upsamp_list)+1))).long()).tolist()[1:]

    torch.cuda.empty_cache()
    PSNRs,PSNRs_test = [],[0]

    allrays, allrgbs = train_dataset.all_rays, train_dataset.all_rgbs
    if not args.ndc_ray:
        allrays, allrgbs = tensorf.filtering_rays(allrays, allrgbs, bbox_only=True)
    trainingSampler = MotionSampler(allrgbs, args.num_frames, args.batch_size)

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

    pbar = tqdm(range(args.n_iters), miniters=args.progress_refresh_rate, file=sys.stdout)
    for iteration in pbar:
        ray_idx = trainingSampler.nextids()
        rays_train, rgb_train = allrays[ray_idx], allrgbs[ray_idx].to(device)

        #rgb_map, alphas_map, depth_map, weights, uncertainty
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
            PSNRs_test = evaluation(test_dataset,tensorf, args, renderer, f'{logfolder}/imgs_vis/', N_vis=args.N_vis,
                                    prtx=f'{iteration:06d}_', N_samples=nSamples, white_bg = white_bg, ndc_ray=ndc_ray, compute_extra_metrics=False)
            summary_writer.add_scalar('test/psnr', np.mean(PSNRs_test), global_step=iteration)

        if iteration in update_AlphaMask_list:
            if reso_cur[0] * reso_cur[1] * reso_cur[2]<256**3:# update volume resolution
                reso_mask = reso_cur
            new_aabb = tensorf.updateAlphaMask(tuple(reso_mask))
            if iteration == update_AlphaMask_list[0]:
                tensorf.shrink(new_aabb)
                # tensorVM.alphaMask = None
                L1_reg_weight = args.L1_weight_rest
                print("continuing L1_reg_weight", L1_reg_weight)
            if not args.ndc_ray and iteration == update_AlphaMask_list[1]:
                # filter rays outside the bbox
                allrays,allrgbs = tensorf.filtering_rays(allrays,allrgbs)
                trainingSampler = SimpleSampler(allrgbs.shape[0], args.batch_size)

        if iteration in upsamp_list:
            n_voxels = N_voxel_list.pop(0)
            reso_cur = N_to_reso(n_voxels, tensorf.aabb)
            nSamples = min(args.nSamples, cal_n_samples(reso_cur,args.step_ratio))
            tensorf.upsample_volume_grid(reso_cur)

            if args.lr_upsample_reset:
                print("reset lr to initial")
                lr_scale = 1 #0.1 ** (iteration / args.n_iters)
            else:
                lr_scale = args.lr_decay_target_ratio ** (iteration / args.n_iters)
            grad_vars = tensorf.get_optparam_groups(args.lr_init*lr_scale, args.lr_basis*lr_scale)
            optimizer = torch.optim.Adam(grad_vars, betas=(0.9, 0.99))

        if iteration % (args.n_iters//10) == 0: # save at the first iter to make sure disk space available
            save_ckpt()
            
    save_ckpt()

    if args.render_train:
        os.makedirs(f'{logfolder}/imgs_train_all', exist_ok=True)
        train_dataset = dataset(args.datadir, split='train', downsample=args.downsample_train, is_stack=True)
        PSNRs_test = evaluation(train_dataset,tensorf, args, renderer, f'{logfolder}/imgs_train_all/',
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
        # c2ws = test_dataset.poses
        print('========>',c2ws.shape)
        os.makedirs(f'{logfolder}/imgs_path_all', exist_ok=True)
        evaluation_path(test_dataset,tensorf, c2ws, renderer, f'{logfolder}/imgs_path_all/',
                        N_vis=-1, N_samples=-1, white_bg = white_bg, ndc_ray=ndc_ray,device=device)


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
    # args.deform_field = dict(xyz_freq=10, t_freq=8, num_layers=4, hidden_dim=128)
    # args.portion_decoder = dict(num_layers=3, hidden_dim=64)
    print(args)

    if args.export_mesh:
        export_mesh(args)

    if args.render_only and (args.render_test or args.render_path):
        targs = mmengine.Config.fromfile(f'{os.path.dirname(args.ckpt)}/config.py')
        targs.merge_from_dict(args)
        render_test(targs)
    else:
        reconstruction(args)

