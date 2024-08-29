import torch
import torchvision
import numpy as np
from PIL import Image
import torch.nn.functional as F
from einops import rearrange
import argparse

from warp_utils import *
import os

from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
)

from pytorch_lightning import seed_everything
seed_everything(7070)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

ip2p = StableDiffusionInstructPix2PixPipeline.from_pretrained("timbrooks/instruct-pix2pix", torch_dtype=torch_dtype, safety_checker=None).to(device)

ip2p.enable_xformers_memory_efficient_attention()
ip2p.enable_model_cpu_offload()

parser = argparse.ArgumentParser()
parser.add_argument("--source_img", type=str, default="./examples/coffee_cam_2x/0.png")
parser.add_argument("--target_img", type=str, default="./examples/coffee_cam_2x/1.png")
parser.add_argument("--prompt", type=str, default="What if it was painted by Van Gogh?")
parser.add_argument("--guidance_scale", type=float, default=10.5)
parser.add_argument("--image_guidance_scale", type=float, default=1.5)
parser.add_argument("--pts_path", type=str, default="./examples/pts_0.pt")
parser.add_argument("--warp_path", type=str, default="./examples/warp_0.pt")

args = parser.parse_args()
prompt = args.prompt
guidance_scale = args.guidance_scale
image_guidance_scale = args.image_guidance_scale
diffusion_steps = 20
num_train_timesteps = 1000
pts_all = torch.load(args.pts_path).to(device)
warp_all = torch.load(args.warp_path).to(device)

source_img = Image.open(args.source_img).convert('RGB')
target_img = Image.open(args.target_img).convert('RGB')
width, height = source_img.size
source_index = int(args.source_img.split('/')[-1].split('.')[0].split('_')[-1])
target_index = int(args.target_img.split('/')[-1].split('.')[0].split('_')[-1])

with torch.no_grad():   
    edited_image = ip2p(prompt, image=source_img, num_inference_steps=diffusion_steps, guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale, output_type="pt").images # [1, 3, H, W] [0, 1]

if edited_image.shape[-2:] != (height, width):
    edited_image = F.interpolate(edited_image, (height, width), mode='bilinear', align_corners=False)
    
edited_image = edited_image.squeeze(0).permute(1, 2, 0).float() # (H, W, 3)
source_image = torch.from_numpy(np.array(source_img) / 255).to(device).float() # (H, W, 3)
target_image = torch.from_numpy(np.array(target_img) / 255).to(device).float() # (H, W, 3)

warp, mask, diff = apply_warp(warp_all[source_index][target_index], target_image, edited_image, pts_all[target_index], pts_all[source_index], diff_thres=0.2) # (H, W, 3), (H, W), (H, W, 3)

orig_image = torch.cat([source_image, target_image], dim=1) # (H, 2*W, 3)
updated_image = torch.cat([edited_image, warp], dim=1) # (H, 2*W, 3)
save_image = torch.cat([orig_image, updated_image], dim=0) # (2*H, 2*W, 3)
torchvision.utils.save_image(save_image.permute(2, 0, 1), 'output.png')
    