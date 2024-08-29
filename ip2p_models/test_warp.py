import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn.functional as F
from einops import rearrange
import argparse

from warp_utils import *
import os

from pytorch_lightning import seed_everything
seed_everything(7070)

from ip2p_sequence import SequenceInstructPix2Pix

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

ip2p = SequenceInstructPix2Pix(device=device, ip2p_use_full_precision=False)

parser = argparse.ArgumentParser()
parser.add_argument("--image_dir", type=str, default="./examples/coffee_cam_2x")
parser.add_argument("--prompt", type=str, default="What if it was painted by Van Gogh?")
parser.add_argument("--guidance_scale", type=float, default=10.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5)
parser.add_argument("--pts_path", type=str, default="./examples/pts_0.pt")
parser.add_argument("--warp_path", type=str, default="./examples/warp_0.pt")

args = parser.parse_args()
data_path = args.image_dir
prompt = args.prompt
guidance_scale = args.guidance_scale
image_guidance_scale = args.image_guidance_scale
diffusion_steps = 20
num_train_timesteps = 1000
pts_all = torch.load(args.pts_path).to(device)
warp_all = torch.load(args.warp_path).to(device)

files = sorted(os.listdir(data_path), key=lambda x: int(x.split('.')[0]))
files = [os.path.join(data_path, file) for file in files]
num_cams = len(files)
print('num_cams:', num_cams)
indexs = list(range(len(files)))

sequence_length = num_cams // 3
selected_indexs = list(np.random.choice(indexs, sequence_length, replace=False))
remaining_indexs = [index for index in indexs if index not in selected_indexs]

images = []
for file in files:
    image = Image.open(file).convert('RGB')
    image = torch.from_numpy(np.array(image) / 255).permute(2, 0, 1).unsqueeze(0).to(torch_dtype).to(device)
    images.append(image)
images = torch.cat(images, dim=0) # (f, c, h, w)

images_input = images[selected_indexs] # (sequence_length, c, h, w)
images_cond = images_input.clone() # (sequence_length, c, h, w)
images_remain = images[remaining_indexs] # (total_length-sequence_length, c, h, w)
images_remain_cond = images_remain.clone() # (total_length-sequence_length, c, h, w)

images_edit = ip2p.edit_sequence(
    images=images_input.to(device), # (seq_len, C, H, W)
    images_cond=images_cond.to(device), # (seq_len, C, H, W)
    guidance_scale=guidance_scale,
    image_guidance_scale=image_guidance_scale,
    diffusion_steps=diffusion_steps,
    prompt=prompt,
    noisy_latent_type="noisy_latent",
    T=num_train_timesteps,
) # (f, C, H, W)

images_remain = images_remain.to(device, dtype=torch.float32) #
video = images_edit.to(device, dtype=torch.float32) # (f, c, h, w)
if video.shape[-2:] != images_cond.shape[-2:]:
    video = F.interpolate(video, size=images.shape[-2:], mode='bilinear', align_corners=False)
    
torchvision.utils.save_image(images_input, 'images_input.png', nrow=sequence_length, padding=0, pad_value=1)
torchvision.utils.save_image(video, 'images_edit.png', nrow=sequence_length, padding=0, pad_value=1)
    
for idx_cur, i in enumerate(selected_indexs):
    warp_average = torch.zeros_like(video[idx_cur].permute(1, 2, 0)) # (H, W, 3)
    weights_mask = torch.zeros(warp_average.shape[:-1]).to(device) # (H, W)
    for idx_ref, j in enumerate(selected_indexs):
        # if i == j:
        #     continue
        warp_ref = warp_all[j][i] # (H, W, 2) 
        warp, mask = apply_warp(warp_ref, video[idx_cur].permute(1, 2, 0).float(), video[idx_ref].permute(1, 2, 0).float(), pts_all[i], pts_all[j], diff_thres=0.02) # (H, W, 3), (H, W)
        weight = (mask!=0).sum() / (mask).numel()
        warp_average[mask] += warp[mask] * weight # (H, W, 3)
        weights_mask[mask] += weight
        
    average_mask = (weights_mask!=0) # (H, W)
    # video[idx_cur].permute(1, 2, 0)[average_mask] = (video[idx_cur].permute(1, 2, 0)[average_mask]+ warp_average[average_mask]) / (1+weights_mask[average_mask].unsqueeze(-1)) 
    video[idx_cur].permute(1, 2, 0)[average_mask] = (warp_average[average_mask]) / weights_mask[average_mask].unsqueeze(-1)
    
    
torchvision.utils.save_image(video, 'images_edit_warp.png', nrow=sequence_length, padding=0, pad_value=1)

# images_warp_refine = ip2p.edit_sequence(
#     images=video.unsqueeze(0).to(device), # (1, seq_len, C, H, W)
#     images_cond=images_cond.unsqueeze(0).to(device), # (1, seq_len, C, H, W)
#     guidance_scale=guidance_scale,
#     image_guidance_scale=image_guidance_scale,
#     diffusion_steps=3,
#     prompt=prompt,
#     noisy_latent_type="noisy_latent",
#     T=600,
# ) # (1, C, f, H, W)

# images_warp_refine = images_warp_refine.to(device, dtype=torch.float32)
# video = rearrange(images_warp_refine, '1 c f h w -> f c h w').to(device, dtype=torch.float32) # (f, c, h, w)
# if video.shape[-2:] != images_cond.shape[-2:]:
#     video = F.interpolate(video, size=images.shape[-2:], mode='bilinear', align_corners=False)

# torchvision.utils.save_image(video, 'images_edit_warp_edit.png', nrow=sequence_length, padding=0, pad_value=1)

torchvision.utils.save_image(images_remain, 'images_remain.png', nrow=len(remaining_indexs)//2, padding=0, pad_value=1)

for idx_cur, i in enumerate(remaining_indexs):
    warp_average = torch.zeros(images_remain[idx_cur].permute(1, 2, 0).shape).to(device) # (H, W, 3)
    weights_mask = torch.zeros(warp_average.shape[:-1]).to(device) # (H, W)
    for idx_ref, j in enumerate(selected_indexs):
        warp_ref = warp_all[j][i] # (H, W, 2)
        warp, mask = apply_warp(warp_ref, images_remain[idx_cur].permute(1, 2, 0).float(), video[idx_ref].permute(1, 2, 0).float(), pts_all[i], pts_all[j], diff_thres=0.2)
        weight = (mask!=0).sum() / (mask).numel()
        warp_average[mask] += warp[mask] * weight
        weights_mask[mask] += weight
    
    average_mask = (weights_mask!=0) # (H, W)
    warp_average[average_mask] /= weights_mask[average_mask].unsqueeze(-1) # (H, W, 3)
    images_remain[idx_cur].permute(1, 2, 0)[average_mask] = warp_average[average_mask] * 0.8 + images_remain[idx_cur].permute(1, 2, 0)[average_mask] * 0.2

torchvision.utils.save_image(images_remain, 'images_remain_warp.png', nrow=len(remaining_indexs)//2, padding=0, pad_value=1)

images_remain_edit = ip2p.edit_sequence(
    images=images_remain.to(device), # (1, seq_len, C, H, W)
    images_cond=images_remain_cond.to(device), # (1, seq_len, C, H, W)
    guidance_scale=guidance_scale,
    image_guidance_scale=image_guidance_scale,
    diffusion_steps=5,
    prompt=prompt,
    noisy_latent_type="noisy_latent",
    T=600,
) # (1, C, f, H, W)

images_remain_edit = rearrange(images_remain_edit, '1 c f h w -> f c h w').to(device, dtype=torch.float32) # (f, c, h, w)
if images_remain_edit.shape[-2:] != images_remain.shape[-2:]:
    images_remain_edit = F.interpolate(images_remain_edit, size=images_remain.shape[-2:], mode='bilinear', align_corners=False)
    
torchvision.utils.save_image(images_remain_edit, 'images_remain_edit.png', nrow=len(remaining_indexs)//2, padding=0, pad_value=1)


    




    
    
    