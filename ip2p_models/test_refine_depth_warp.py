import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn.functional as F

from einops import rearrange
from warp_utils import *
import os

from pytorch_lightning import seed_everything
seed_everything(7070)

from ip2p_sequence import SequenceInstructPix2Pix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

ip2p = SequenceInstructPix2Pix(device=device, ip2p_use_full_precision=False)

data_path = 'data/coffee_cam_2x'
prompt = "What if it was painted by Van Gogh?"
guidance_scale = 8.5
image_guidance_scale = 1.5
diffusion_steps = 20
num_train_timesteps = 1000
noisy_latent_type = 'noise' # 'ddim_inv', 'ddim_inv_classifier', 'noise', 'noisy_latents'
num_key_views = 5

pts_all = torch.load('/scratch/bbsh/linzhan/cache/neural_3d/edit_coffee_50_2/pts_0.pt').to(device)
warp_all = torch.load('/scratch/bbsh/linzhan/cache/neural_3d/edit_coffee_50_2/warp_0.pt').to(device)

files = sorted(os.listdir(data_path), key=lambda x: int(x.split('.')[0]))
files = [os.path.join(data_path, file) for file in files]
num_cams = len(files)
print('num_cams', num_cams)
indexs = list(range(len(files)))

images = []
for file in files:
    image = Image.open(file).convert('RGB')
    image = torch.from_numpy(np.array(image) / 255).permute(2, 0, 1).unsqueeze(0).to(torch.float32).to(device)
    images.append(image)
images = torch.cat(images, dim=0) # (f, c, h, w)

cam_indexs = list(range(num_cams))
sample_indexs = sorted(np.random.choice(cam_indexs, num_key_views, replace=False))
remain_indexs = sorted(list(set(cam_indexs) - set(sample_indexs)))

images_sample = images[sample_indexs] # (sequence_length, c, h, w)
images_sample_cond = images_sample.clone() # (sequence_length, c, h, w)
images_remain = images[remain_indexs] # (total_length-sequence_length, c, h, w)
images_remain_cond = images_remain.clone() # (total_length-sequence_length, c, h, w)

torchvision.utils.save_image(images_sample, 'images_sample.png', nrow=images_sample.shape[0], padding=0, pad_value=1)

images_sample_edit = ip2p.edit_sequence(
    images=images_sample.to(device), # (sq_len, C, H, W)
    images_cond=images_sample_cond.to(device), # (sq_len, C, H, W)
    guidance_scale=guidance_scale,
    image_guidance_scale=image_guidance_scale,
    diffusion_steps=diffusion_steps,
    prompt=prompt,
    noisy_latent_type=noisy_latent_type,
    T=num_train_timesteps,
).to(device, dtype=torch.float32) # (1, C, f, H, W)

if images_sample_edit[-2:] != images.shape[-2:]:
    images_sample_edit = F.interpolate(images_sample_edit, size=images.shape[-2:], mode='bilinear', align_corners=False)
    
torchvision.utils.save_image(images_sample_edit, 'images_sample_edit.png', nrow=images_sample_edit.shape[0], padding=0, pad_value=1)

for idx_cur, i in enumerate(sample_indexs):
    warp_average = torch.zeros_like(images_sample_edit[idx_cur].permute(1, 2, 0)) # (H, W, 3)
    weights_mask = torch.zeros(warp_average.shape[:-1]).to(device) # (H, W)
    for idx_ref, j in enumerate(sample_indexs):
        warp_ref = warp_all[j][i] # (H, W, 2) 
        warp, mask, diff = apply_warp(warp_ref, images_sample_edit[idx_cur].permute(1, 2, 0).float(), images_sample_edit[idx_ref].permute(1, 2, 0).float(), pts_all[i], pts_all[j], diff_thres=0.02) # (H, W, 3), (H, W)
        import ipdb; ipdb.set_trace()
        weight = (mask!=0).sum() / (mask).numel()
        warp_average[mask] += warp[mask] * weight # (H, W, 3)
        weights_mask[mask] += weight
        weight = torch.nn.Softmax(dim=0)(diff[mask]).max()
    average_mask = (weights_mask!=0) # (H, W)
    images_sample_edit[idx_cur].permute(1, 2, 0)[average_mask] = (warp_average[average_mask]) / weights_mask[average_mask].unsqueeze(-1) 
    
torchvision.utils.save_image(images_sample_edit, 'images_sample_edit_warp.png', nrow=images_sample_edit.shape[0], padding=0, pad_value=1)

torchvision.utils.save_image(images_remain, 'images_remain.png', nrow=images_remain.shape[0]//3, padding=0, pad_value=1)

for idx_cur, i in enumerate(remain_indexs):
    warp_average = torch.zeros(images_remain[idx_cur].permute(1, 2, 0).shape).to(device) # (H, W, 3)
    weights_mask = torch.zeros(warp_average.shape[:-1]).to(device) # (H, W)
    for idx_ref, j in enumerate(sample_indexs):
        warp_ref = warp_all[j][i] # (H, W, 2)
        warp, mask, diff = apply_warp(warp_ref, images_remain[idx_cur].permute(1, 2, 0).float(), images_sample_edit[idx_ref].permute(1, 2, 0).float(), pts_all[i], pts_all[j], diff_thres=0.2)
        weight = (mask!=0).sum() / (mask).numel()
        warp_average[mask] += warp[mask] * weight
        weights_mask[mask] += weight
    
    average_mask = (weights_mask!=0) # (H, W)
    warp_average[average_mask] /= weights_mask[average_mask].unsqueeze(-1) # (H, W, 3)
    images_remain[idx_cur].permute(1, 2, 0)[average_mask] = warp_average[average_mask]

torchvision.utils.save_image(images_remain, 'images_remain_warp.png', nrow=images_remain.shape[0]//3, padding=0, pad_value=1)

images_remain_edit = torch.zeros_like(images_remain)
for idx in range(0, images_remain.shape[0] // num_key_views + 1):
    start_idx = idx * num_key_views
    end_idx = min((idx+1) * num_key_views, images_remain.shape[0])
    
    images_remain_selected = images_remain[start_idx:end_idx] # (num_key_views, c, h, w)
    images_remain_selected_cond = images_remain_cond[start_idx:end_idx] # (num_key_views, c, h, w)

    images_remain_selected_edit = ip2p.edit_sequence(
        images=images_remain_selected.to(device), # (seq_len, C, H, W)
        images_cond=images_remain_selected_cond.to(device), # (seq_len, C, H, W)
        guidance_scale=guidance_scale,
        image_guidance_scale=image_guidance_scale,
        diffusion_steps=6,
        prompt=prompt,
        noisy_latent_type="noisy_latent",
        T=750,
    ).to(device, dtype=torch.float32) # (f, c, H, W)

    if images_remain_selected_edit[-2:] != images.shape[-2:]:
        images_remain_selected_edit = F.interpolate(images_remain_selected_edit, size=images.shape[-2:], mode='bilinear', align_corners=False)
    
    images_remain_edit[start_idx:end_idx] = images_remain_selected_edit
     
torchvision.utils.save_image(images_remain_edit, 'images_remain_edit.png', nrow=images_remain_edit.shape[0]//3, padding=0, pad_value=1)