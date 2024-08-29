import torch
from torch import nn
import torch.nn.functional as F
from .sh import eval_sh_bases
import numpy as np
import time


def positional_encoding(positions, freqs):
    
        freq_bands = (2**torch.arange(freqs).float()).to(positions.device)  # (F,)
        pts = (positions[..., None] * freq_bands).reshape(
            positions.shape[:-1] + (freqs * positions.shape[-1], ))  # (..., DF)
        pts = torch.cat([torch.sin(pts), torch.cos(pts)], dim=-1)
        return pts

def raw2alpha(sigma, dist):
    # sigma, dist  [N_rays, N_samples]
    alpha = 1. - torch.exp(-sigma*dist)

    T = torch.cumprod(torch.cat([torch.ones(alpha.shape[0], 1).to(alpha.device), 1. - alpha + 1e-10], -1), -1)

    weights = alpha * T[:, :-1]  # [N_rays, N_samples]
    return alpha, weights, T[:,-1:]


def SHRender(xyz_sampled, viewdirs, features):
    sh_mult = eval_sh_bases(2, viewdirs)[:, None]
    rgb_sh = features.view(-1, 3, sh_mult.shape[-1])
    rgb = torch.relu(torch.sum(sh_mult * rgb_sh, dim=-1) + 0.5)
    return rgb


def RGBRender(xyz_sampled, viewdirs, features):

    rgb = features
    return rgb

class AlphaGridMask(nn.Module):
    def __init__(self, device, aabb, alpha_volume):
        super(AlphaGridMask, self).__init__()
        self.device = device

        self.aabb=aabb.to(self.device)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invgridSize = 1.0/self.aabbSize * 2
        self.alpha_volume = alpha_volume.view(1,1,*alpha_volume.shape[-3:])
        self.gridSize = torch.LongTensor([alpha_volume.shape[-1],alpha_volume.shape[-2],alpha_volume.shape[-3]]).to(self.device)

    def sample_alpha(self, xyz_sampled):
        xyz_sampled = self.normalize_coord(xyz_sampled)
        alpha_vals = F.grid_sample(self.alpha_volume, xyz_sampled.view(1,-1,1,1,3), align_corners=True).view(-1)

        return alpha_vals

    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invgridSize - 1


class MLPRender_Fea(nn.Module):
    def __init__(self,inChanel, viewpe=6, feape=6, featureC=128):
        super(MLPRender_Fea, self).__init__()

        self.in_mlpC = 2*viewpe*3 + 2*feape*inChanel + 3 + inChanel
        self.viewpe = viewpe
        self.feape = feape
        layer1 = nn.Linear(self.in_mlpC, featureC)
        layer2 = nn.Linear(featureC, featureC)
        layer3 = nn.Linear(featureC,3)

        self.mlp = nn.Sequential(layer1, nn.ReLU(inplace=True), layer2, nn.ReLU(inplace=True), layer3)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.feape > 0:
            indata += [positional_encoding(features, self.feape)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender_PE(nn.Module):
    def __init__(self,inChanel, viewpe=6, pospe=6, featureC=128):
        super(MLPRender_PE, self).__init__()

        self.in_mlpC = (3+2*viewpe*3)+ (3+2*pospe*3)  + inChanel #
        self.viewpe = viewpe
        self.pospe = pospe
        layer1 = nn.Linear(self.in_mlpC, featureC)
        layer2 = nn.Linear(featureC, featureC)
        layer3 = nn.Linear(featureC,3)

        self.mlp = nn.Sequential(layer1, nn.ReLU(inplace=True), layer2, nn.ReLU(inplace=True), layer3)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.pospe > 0:
            indata += [positional_encoding(pts, self.pospe)]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb

class MLPRender(nn.Module):
    def __init__(self,inChanel, viewpe=6, featureC=128):
        super(MLPRender, self).__init__()

        self.in_mlpC = (3+2*viewpe*3) + inChanel
        self.viewpe = viewpe
        
        layer1 = nn.Linear(self.in_mlpC, featureC)
        layer2 = nn.Linear(featureC, featureC)
        layer3 = nn.Linear(featureC,3)

        self.mlp = nn.Sequential(layer1, nn.ReLU(inplace=True), layer2, nn.ReLU(inplace=True), layer3)
        nn.init.constant_(self.mlp[-1].bias, 0)

    def forward(self, pts, viewdirs, features):
        indata = [features, viewdirs]
        if self.viewpe > 0:
            indata += [positional_encoding(viewdirs, self.viewpe)]
        mlp_in = torch.cat(indata, dim=-1)
        rgb = self.mlp(mlp_in)
        rgb = torch.sigmoid(rgb)

        return rgb


def get_fc_net(num_layers, in_dim, hidden_dim, out_dim, use_bias=True):
    net =  []
    for l in range(num_layers):
        if l == 0:
            in_dim = in_dim
        else:
            in_dim = hidden_dim
        if l == num_layers - 1:
            l_out_dim = out_dim
            net.append(nn.Linear(in_dim, l_out_dim, bias=use_bias))
        else:
            l_out_dim = hidden_dim
            net.append(nn.Linear(in_dim, l_out_dim, bias=use_bias))
            net.append(nn.ReLU())
    net = nn.Sequential(*net)
    return net


class DeformField(nn.Module):
    def __init__(self, xyz_dim=3, xyz_freq=5, t_freq=5, num_layers=2, hidden_dim=64, out_feat=32):
        super().__init__()
        self.register_buffer('xyz_freq', 2**torch.arange(xyz_freq).float())
        self.register_buffer('t_freq', 2**torch.arange(t_freq).float())
        nets = [nn.Linear(xyz_freq*2*xyz_dim+t_freq*2, hidden_dim), nn.ReLU(),]
        for _ in range(num_layers-1):
            nets += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        nets += [nn.Linear(hidden_dim, out_feat),]
        self.net = nn.Sequential(*nets)

    def forward(self, xyz, frame):
        n = xyz.shape[0]
        frame_encoded = torch.cat([torch.sin(frame*self.t_freq), 
                                   torch.cos(frame*self.t_freq)])[None]
        frame_encoded = frame_encoded.expand([n, -1])
        xyz_encoded = torch.cat([torch.sin(xyz[:,:,None]*self.xyz_freq[None,None]).view([n,-1]), 
                                 torch.cos(xyz[:,:,None]*self.xyz_freq[None,None]).view([n,-1])],1)
        h = torch.cat([xyz_encoded, frame_encoded], 1)
        return self.net(h)



class StreamTensorBase(nn.Module):
    def __init__(self, aabb, gridSize, device, density_n_comp = 8, appearance_n_comp = 24, app_dim = 27,
                    shadingMode = 'MLP_PE', alphaMask = None, near_far=[2.0,6.0],
                    density_shift = -10, alphaMask_thres=0.001, distance_scale=25, rayMarch_weight_thres=0.0001,
                    pos_pe = 6, view_pe = 6, fea_pe = 6, featureC=128, step_ratio=2.0,
                    fea2denseAct = 'softplus', num_frames=1, ld_per_frame=1, 
                    # deform
                    deform_field=None, portion_decoder=None, virtual_cannonical=False, 
                    target_portion=[0,0,1], share_portion_embeddings=True, portion_weight=0):
        super(StreamTensorBase, self).__init__()

        self.density_n_comp = density_n_comp
        self.app_n_comp = appearance_n_comp
        self.app_dim = app_dim
        self.aabb = aabb
        self.alphaMask = alphaMask
        self.device=device
        self.num_frames = num_frames

        self.density_shift = density_shift
        self.alphaMask_thres = alphaMask_thres
        self.distance_scale = distance_scale
        self.rayMarch_weight_thres = rayMarch_weight_thres
        self.fea2denseAct = fea2denseAct

        self.near_far = near_far
        self.step_ratio = step_ratio

        self.update_stepSize(gridSize)
        self.matMode = [[0,1], [0,2], [1,2]]
        self.vecMode =  [2, 1, 0]
        self.comp_w = [1,1,1]

        self.ld_per_frame = ld_per_frame
        self.level_dim_multi = 1
        if self.ld_per_frame > 1:
            assert self.ld_per_frame%1==0
            self.level_dim_multi = int(self.ld_per_frame*2)
            for k in [*self.density_n_comp, *self.app_n_comp]:
                assert k%self.level_dim_multi==0, 'should be a factor of given feat dim'
            self.ld_per_frame = 0.5
        self.new_dim_interval = int(1/self.ld_per_frame)
        self.density_list = [self.register_frame_permute_index(k) for k in self.density_n_comp]
        self.app_list = [self.register_frame_permute_index(k) for k in self.app_n_comp]

        self.init_svd_volume(gridSize[0], device)
        self.shadingMode, self.pos_pe, self.view_pe, self.fea_pe, self.featureC = shadingMode, pos_pe, view_pe, fea_pe, featureC
        self.init_render_func(shadingMode, pos_pe, view_pe, fea_pe, featureC, device)

        # deform
        if deform_field is not None:
            self.deform_field = DeformField(**deform_field, out_feat=3,).to(device)
        else:
            self.deform_field = None
        if portion_decoder is not None:
            self.portion_decoder = get_fc_net(in_dim=self.app_dim, out_dim=3, **portion_decoder).to(device)
        self.register_buffer('target_portion', torch.as_tensor(target_portion).to(device))
        self.virtual_cannonical = virtual_cannonical
        self.share_portion_embeddings = share_portion_embeddings
        self.portion_weight = portion_weight
        if not self.share_portion_embeddings:
            raise NotImplementedError

    def init_render_func(self, shadingMode, pos_pe, view_pe, fea_pe, featureC, device):
        if shadingMode == 'MLP_PE':
            self.renderModule = MLPRender_PE(self.app_dim, view_pe, pos_pe, featureC).to(device)
        elif shadingMode == 'MLP_Fea':
            self.renderModule = MLPRender_Fea(self.app_dim, view_pe, fea_pe, featureC).to(device)
        elif shadingMode == 'MLP':
            self.renderModule = MLPRender(self.app_dim, view_pe, featureC).to(device)
        elif shadingMode == 'SH':
            self.renderModule = SHRender
        elif shadingMode == 'RGB':
            assert self.app_dim == 3
            self.renderModule = RGBRender
        else:
            print("Unrecognized shading module")
            exit()
        print("pos_pe", pos_pe, "view_pe", view_pe, "fea_pe", fea_pe)
        print(self.renderModule)

    def update_stepSize(self, gridSize):
        print("aabb", self.aabb.view(-1))
        print("grid size", gridSize)
        self.aabbSize = self.aabb[1] - self.aabb[0]
        self.invaabbSize = 2.0/self.aabbSize
        self.gridSize = torch.LongTensor(gridSize).to(self.device)
        self.units = self.aabbSize / (self.gridSize-1)
        self.stepSize = torch.mean(self.units)*self.step_ratio
        self.aabbDiag = torch.sqrt(torch.sum(torch.square(self.aabbSize)))
        self.nSamples = int((self.aabbDiag / self.stepSize).item()) + 1
        print("sampling step size: ", self.stepSize)
        print("sampling number: ", self.nSamples)

    def init_svd_volume(self, res, device):
        pass

    def compute_features(self, xyz_sampled, frame):
        pass
    
    def compute_densityfeature(self, xyz_sampled, frame):
        pass
    
    def compute_appfeature(self, xyz_sampled, frame):
        pass
    
    def normalize_coord(self, xyz_sampled):
        return (xyz_sampled-self.aabb[0]) * self.invaabbSize - 1

    def get_optparam_groups(self, lr_init_spatial = 0.02, lr_init_network = 0.001):
        pass

    def get_kwargs(self):
        return {
            'aabb': self.aabb,
            'gridSize': self.gridSize.tolist(),
            'density_n_comp': self.density_n_comp,
            'appearance_n_comp': self.app_n_comp,
            'app_dim': self.app_dim,
            'density_shift': self.density_shift,
            'alphaMask_thres': self.alphaMask_thres,
            'distance_scale': self.distance_scale,
            'rayMarch_weight_thres': self.rayMarch_weight_thres,
            'fea2denseAct': self.fea2denseAct,
            'near_far': self.near_far,
            'step_ratio': self.step_ratio,
            'shadingMode': self.shadingMode,
            'pos_pe': self.pos_pe,
            'view_pe': self.view_pe,
            'fea_pe': self.fea_pe,
            'featureC': self.featureC,
            'num_frames': 1, 
            'ld_per_frame': 1, 
            'deform_field': None, 
            'portion_decoder': None, 
            'virtual_cannonical': False, 
            'target_portion': [0,0,1], 
            'share_portion_embeddings': True, 
            'portion_weight': 0
        }

    def save(self, path):
        kwargs = self.get_kwargs()
        ckpt = {'kwargs': kwargs, 'state_dict': self.state_dict()}
        if self.alphaMask is not None:
            alpha_volume = self.alphaMask.alpha_volume.bool().cpu().numpy()
            ckpt.update({'alphaMask.shape':alpha_volume.shape})
            ckpt.update({'alphaMask.mask':np.packbits(alpha_volume.reshape(-1))})
            ckpt.update({'alphaMask.aabb': self.alphaMask.aabb.cpu()})
        torch.save(ckpt, path)

    def load(self, ckpt):
        if 'alphaMask.aabb' in ckpt.keys():
            length = np.prod(ckpt['alphaMask.shape'])
            alpha_volume = torch.from_numpy(np.unpackbits(ckpt['alphaMask.mask'])[:length].reshape(ckpt['alphaMask.shape']))
            self.alphaMask = AlphaGridMask(self.device, ckpt['alphaMask.aabb'].to(self.device), alpha_volume.float().to(self.device))
        self.load_state_dict(ckpt['state_dict'])

    def sample_ray_ndc(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples > 0 else self.nSamples
        near, far = self.near_far
        interpx = torch.linspace(near, far, N_samples).unsqueeze(0).to(rays_o)
        if is_train:
            interpx += torch.rand_like(interpx).to(rays_o) * ((far - near) / N_samples)

        rays_pts = rays_o[..., None, :] + rays_d[..., None, :] * interpx[..., None]
        mask_outbbox = ((self.aabb[0] > rays_pts) | (rays_pts > self.aabb[1])).any(dim=-1)
        return rays_pts, interpx, ~mask_outbbox

    def sample_ray(self, rays_o, rays_d, is_train=True, N_samples=-1):
        N_samples = N_samples if N_samples>0 else self.nSamples
        stepsize = self.stepSize
        near, far = self.near_far
        vec = torch.where(rays_d==0, torch.full_like(rays_d, 1e-6), rays_d)
        rate_a = (self.aabb[1] - rays_o) / vec
        rate_b = (self.aabb[0] - rays_o) / vec
        t_min = torch.minimum(rate_a, rate_b).amax(-1).clamp(min=near, max=far)

        rng = torch.arange(N_samples)[None].float()
        if is_train:
            rng = rng.repeat(rays_d.shape[-2],1)
            rng += torch.rand_like(rng[:,[0]])
        step = stepsize * rng.to(rays_o.device)
        interpx = (t_min[...,None] + step)

        rays_pts = rays_o[...,None,:] + rays_d[...,None,:] * interpx[...,None]
        mask_outbbox = ((self.aabb[0]>rays_pts) | (rays_pts>self.aabb[1])).any(dim=-1)

        return rays_pts, interpx, ~mask_outbbox

    def shrink(self, new_aabb, voxel_size):
        pass

    @torch.no_grad()
    def getDenseAlpha(self,gridSize=None):
        gridSize = self.gridSize if gridSize is None else gridSize

        samples = torch.stack(torch.meshgrid(
            torch.linspace(0, 1, gridSize[0]),
            torch.linspace(0, 1, gridSize[1]),
            torch.linspace(0, 1, gridSize[2]),
        ), -1).to(self.device)
        dense_xyz = self.aabb[0] * (1-samples) + self.aabb[1] * samples

        # dense_xyz = dense_xyz
        # print(self.stepSize, self.distance_scale*self.aabbDiag)
        alpha = torch.zeros_like(dense_xyz[...,0])
        for i in range(gridSize[0]):
            alpha[i] = self.compute_alpha(dense_xyz[i].view(-1,3), self.stepSize).view((gridSize[1], gridSize[2]))
        return alpha, dense_xyz

    @torch.no_grad()
    def updateAlphaMask(self, gridSize=(200,200,200)):
        raise NotImplementedError('alpha mask should not get updated')

        alpha, dense_xyz = self.getDenseAlpha(gridSize)
        dense_xyz = dense_xyz.transpose(0,2).contiguous()
        alpha = alpha.clamp(0,1).transpose(0,2).contiguous()[None,None]
        total_voxels = gridSize[0] * gridSize[1] * gridSize[2]

        ks = 3
        alpha = F.max_pool3d(alpha, kernel_size=ks, padding=ks // 2, stride=1).view(gridSize[::-1])
        alpha[alpha>=self.alphaMask_thres] = 1
        alpha[alpha<self.alphaMask_thres] = 0

        self.alphaMask = AlphaGridMask(self.device, self.aabb, alpha)

        valid_xyz = dense_xyz[alpha>0.5]

        xyz_min = valid_xyz.amin(0)
        xyz_max = valid_xyz.amax(0)

        new_aabb = torch.stack((xyz_min, xyz_max))

        total = torch.sum(alpha)
        print(f"bbox: {xyz_min, xyz_max} alpha rest %%%f"%(total/total_voxels*100))
        return new_aabb

    @torch.no_grad()
    def filtering_rays(self, all_rays, all_rgbs, N_samples=256, chunk=10240*5, bbox_only=False):
        print('========> filtering rays ...')
        tt = time.time()

        N = torch.tensor(all_rays.shape[:-1]).prod()

        mask_filtered = []
        idx_chunks = torch.split(torch.arange(N), chunk)
        for idx_chunk in idx_chunks:
            rays_chunk = all_rays[idx_chunk].to(self.device)

            rays_o, rays_d = rays_chunk[..., :3], rays_chunk[..., 3:6]
            if bbox_only:
                vec = torch.where(rays_d == 0, torch.full_like(rays_d, 1e-6), rays_d)
                rate_a = (self.aabb[1] - rays_o) / vec
                rate_b = (self.aabb[0] - rays_o) / vec
                t_min = torch.minimum(rate_a, rate_b).amax(-1)#.clamp(min=near, max=far)
                t_max = torch.maximum(rate_a, rate_b).amin(-1)#.clamp(min=near, max=far)
                mask_inbbox = t_max > t_min

            else:
                xyz_sampled, _,_ = self.sample_ray(rays_o, rays_d, N_samples=N_samples, is_train=False)
                mask_inbbox= (self.alphaMask.sample_alpha(xyz_sampled).view(xyz_sampled.shape[:-1]) > 0).any(-1)

            mask_filtered.append(mask_inbbox.cpu())

        mask_filtered = torch.cat(mask_filtered).view(all_rgbs.shape[:-1])

        print(f'Ray filtering done! takes {time.time()-tt} s. ray mask ratio: {torch.sum(mask_filtered) / N}')
        return all_rays[mask_filtered], all_rgbs[mask_filtered]

    def feature2density(self, density_features):
        if self.fea2denseAct == "softplus":
            return F.softplus(density_features+self.density_shift)
        elif self.fea2denseAct == "relu":
            return F.relu(density_features)

    def compute_alpha(self, xyz_locs, length=1):
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_locs)
            alpha_mask = alphas > 0
        else:
            alpha_mask = torch.ones_like(xyz_locs[:,0], dtype=bool)
        sigma = torch.zeros(xyz_locs.shape[:-1], device=xyz_locs.device)
        if alpha_mask.any():
            xyz_sampled = self.normalize_coord(xyz_locs[alpha_mask])
            sigma_feature = self.compute_densityfeature(xyz_sampled)
            validsigma = self.feature2density(sigma_feature)
            sigma[alpha_mask] = validsigma
        alpha = 1 - torch.exp(-sigma*length).view(xyz_locs.shape[:-1])
        return alpha

    def forward(self, rays_chunk, white_bg=True, is_train=False, ndc_ray=False, N_samples=-1):
        # sample points
        viewdirs = rays_chunk[:, 3:6]
        frame = rays_chunk[:, -1]
        frame = frame.unique()
        assert len(frame) == 1, f'got {frame}'

        if ndc_ray:
            xyz_sampled, z_vals, ray_valid = self.sample_ray_ndc(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
            rays_norm = torch.norm(viewdirs, dim=-1, keepdim=True)
            dists = dists * rays_norm
            viewdirs = viewdirs / rays_norm
        else:
            xyz_sampled, z_vals, ray_valid = self.sample_ray(rays_chunk[:, :3], viewdirs, is_train=is_train,N_samples=N_samples)
            dists = torch.cat((z_vals[:, 1:] - z_vals[:, :-1], torch.zeros_like(z_vals[:, :1])), dim=-1)
        viewdirs = viewdirs.view(-1, 1, 3).expand(xyz_sampled.shape)
        
        if self.alphaMask is not None:
            alphas = self.alphaMask.sample_alpha(xyz_sampled[ray_valid])
            alpha_mask = alphas > 0
            ray_invalid = ~ray_valid
            ray_invalid[ray_valid] |= (~alpha_mask)
            ray_valid = ~ray_invalid

        sigma = torch.zeros(xyz_sampled.shape[:-1], device=xyz_sampled.device)
        rgb = torch.zeros((*xyz_sampled.shape[:2], 3), device=xyz_sampled.device)

        if ray_valid.any():
            xyz_sampled = self.normalize_coord(xyz_sampled)
            sigma_feature = self.compute_densityfeature(xyz_sampled[ray_valid], frame)
            validsigma = self.feature2density(sigma_feature)
            sigma[ray_valid] = validsigma

        alpha, weight, bg_weight = raw2alpha(sigma, dists * self.distance_scale)

        app_mask = weight > self.rayMarch_weight_thres

        zero = torch.zeros(1).to(rgb)
        extra_loss = zero
        if app_mask.any():
            novel_feat = self.compute_appfeature(xyz_sampled[app_mask], frame)
            
            if self.deform_field is not None:
                xyz = (xyz_sampled[app_mask]+1)/2 # to [0,1]
                xyz.requires_grad = True
                if not self.virtual_cannonical:
                    if frame == 0:
                        new_xyz = xyz
                    else:
                        # delta_xyz = self.deform_field(xyz, frame)*frame
                        delta_xyz = self.deform_field(xyz, frame)
                        new_xyz = (xyz+delta_xyz) - (xyz+delta_xyz).floor().detach()
                    stationary_feat = self.compute_appfeature(xyz_sampled[app_mask], zero)
                else:
                    # delta_xyz = self.deform_field(xyz, frame)*frame
                    delta_xyz = self.deform_field(xyz, frame)
                    new_xyz = (xyz+delta_xyz) - (xyz+delta_xyz).floor().detach()
                    delta_can = self.deform_field(xyz, 0).detach()
                    cannonical_xyz = (xyz+delta_can) - (xyz+delta_can).floor().detach()
                    stationary_feat = self.compute_appfeature((cannonical_xyz-1)*2, zero)
                deform_feat = self.compute_appfeature((new_xyz-1)*2, zero)
                if self.portion_weight > 0:
                    if self.share_portion_embeddings:
                        portion = self.portion_decoder(novel_feat).softmax(dim=-1)
                    else:
                        raise NotImplementedError
                        # portion = self.portion_decoder(self.encode(xyz, frame, self.portion_embeddings)).softmax(dim=-1)
                    if -1 in self.target_portion:
                        extra_loss += (portion[:,self.target_portion==-1]).mean()*self.portion_weight
                    else:
                        extra_loss += (portion.mean(0)-self.target_portion).pow(2).mean()*self.portion_weight
                else:
                    portion = self.target_portion[None]
                app_features = torch.stack([stationary_feat, deform_feat, novel_feat],dim=-1)
                app_features = (app_features*portion[:,None]).sum(-1)
            else:
                app_features = novel_feat
            
            valid_rgbs = self.renderModule(xyz_sampled[app_mask], viewdirs[app_mask], app_features)
            rgb[app_mask] = valid_rgbs

        acc_map = torch.sum(weight, -1)
        rgb_map = torch.sum(weight[..., None] * rgb, -2)

        if white_bg or (is_train and torch.rand((1,))<0.5):
            rgb_map = rgb_map + (1. - acc_map[..., None])
        rgb_map = rgb_map.clamp(0,1)

        with torch.no_grad():
            depth_map = torch.sum(weight * z_vals, -1)
            # depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -1]
            depth_map = depth_map + (1. - acc_map) * rays_chunk[..., -2]

        return rgb_map, depth_map, extra_loss # rgb, sigma, alpha, weight, bg_weight

    def register_frame_permute_index(self, level_dim):
        level_dim = int(level_dim/self.level_dim_multi)
        if self.ld_per_frame > 1:
            raise NotImplementedError
        index_init = [0,level_dim]+list(range(1,level_dim))
        permute_base = list(range(1,level_dim))
        last_entry = 0 # insert into ith place
        permute_init = permute_base[:last_entry] + [0] + permute_base[last_entry:]
        index_list = [torch.as_tensor(index_init, dtype=torch.long),]
        permute_list = [torch.as_tensor(permute_init, dtype=torch.long),]
        for frame_i in range(1, self.num_frames-1):
            if self.new_dim_interval==0 or frame_i%self.new_dim_interval!=0:
                continue
            last_entry += 1
            if last_entry >= level_dim:
                last_entry = 0
            last_index_max = index_list[-1].max().item()
            last_index_min = index_list[-1].min().item()
            prev = index_list[-1][1:][permute_list[-1]].tolist()
            prev.pop(last_entry)
            new_index = [last_index_min+1, last_index_max+1] + prev
            index_list.append(torch.as_tensor(new_index, dtype=torch.long))
            new_permute = permute_base[:last_entry] + [0] + permute_base[last_entry:]
            permute_list.append(torch.as_tensor(new_permute, dtype=torch.long))
        return torch.stack(index_list,0), torch.stack(permute_list,0)

    def get_embeds(self, frame, embedding, frame_index, frame_index_permute):
        if self.new_dim_interval == 0:
            return embedding
        else:
            _, ncom, ndim0, ndim1 = embedding.shape
            embedding = embedding.reshape([1,-1,self.level_dim_multi,ndim0,ndim1])
            if frame == 1:
                row_idx = -1
                left, right = 1, 0
            else:
                # use which row
                row_value = frame*len(frame_index)
                row_idx = int(row_value)
                # blending weights
                left, right = row_value-row_idx, row_idx+1-row_value
            feat_idx = frame_index[row_idx]
            # blending with blending weights
            if left == 0:
                feat_idx = torch.cat([feat_idx[[0]], feat_idx[2:]],0)
                feat = embedding[:,feat_idx]
            elif right == 0: # only the right most case; when frame == 1
                feat_idx = feat_idx[1:]
                feat = embedding[:,feat_idx]
            else:
                # selecting weights
                feat = embedding[:,feat_idx]
                feat = torch.cat([feat[:,[0]]*right+feat[:,[1]]*left, feat[:,2:]],1)
            # return with permute
            return feat[:, frame_index_permute[row_idx]].flatten(1,2)

    def feat_diff(self, frame, embedding, frame_index):
        _, ncom, ndim0, ndim1 = embedding.shape
        embedding = embedding.reshape([1,-1,self.level_dim_multi,ndim0,ndim1])
        row_value = frame*len(frame_index)
        row_idx = int(row_value)
        feat_idx = frame_index[row_idx]
        feat = embedding[:,feat_idx]
        return (feat[:,[0]]-feat[:,[1]]).abs().mean()

    def feat_diff_loss(self, frame):
        raise NotImplementedError

