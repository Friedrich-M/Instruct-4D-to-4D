from diffusers import (
    DDIMScheduler,
    AutoencoderKL,
)
from transformers import (
    CLIPTextModel, 
    CLIPTokenizer
)
from models.ip2p_pipeline import InstructPix2PixPipeline
from models.ip2p_unet import UNet3DConditionModel
from models.ip2p_utils import ddim_inversion, ddim_inversion_classifier
import torch
import torchvision
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F

from RAFT.raft import RAFT
from RAFT.utils.utils import InputPadder
from generate_flow import compute_fwdbwd_mask, warp_flow

import argparse
from einops import rearrange
from tqdm import tqdm
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

from pytorch_lightning import seed_everything
seed_everything(7070)

import warnings; warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="restore RAFT checkpoint")
parser.add_argument("--small", action="store_true", help="use small model")
parser.add_argument(
    "--mixed_precision", action="store_true", help="use mixed precision"
)
args = parser.parse_args()

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"
RAFT_SOURCE = "weights/raft-things.pth"
tokenizer = CLIPTokenizer.from_pretrained(IP2P_SOURCE, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(IP2P_SOURCE, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(IP2P_SOURCE, subfolder="vae")
unet = UNet3DConditionModel.from_pretrained_2d(IP2P_SOURCE, subfolder="unet")

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

vae = vae.to(device, dtype=torch_dtype)
text_encoder = text_encoder.to(device, dtype=torch_dtype)
unet = unet.to(device, dtype=torch_dtype)
        
pipe = InstructPix2PixPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder="scheduler"),
    )

raft = torch.nn.DataParallel(RAFT(args))
raft.load_state_dict(torch.load(RAFT_SOURCE))
raft = raft.module
raft.to(device)
raft.requires_grad_(False)
raft.eval()

data_path = '/scratch/bbsh/linzhan/data/neural_3d/cook_spinach/frames/cam10'
prompt = "Make it look like a Fauvism painting"
guidance_scale = 9.5
image_guidance_scale = 1.5
diffusion_step = 5
num_train_timesteps = 600
sequence_length = 3
overlap_length = 1

selected_latents_type = 'noisy_latents' 

files = sorted(os.listdir(data_path))
files_input = [os.path.join(data_path, file) for file in files][::50]

images = []
for file in files_input:
    image = Image.open(file).convert('RGB')
    width, height = image.size
    factor = 1024 / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    image = torch.from_numpy(np.array(image) / 255).permute(2, 0, 1).unsqueeze(0).to(torch_dtype).to(device)
    images.append(image)
images = torch.cat(images, dim=0) # (f, c, h, w)

dataset_length, _, H, W = images.shape
RH, RW = H // 8 * 8, W // 8 * 8

images = images.to(device, dtype=torch_dtype)
images_input = F.interpolate(images, size=(RH, RW), mode='bilinear', align_corners=False) # (f, c, h, w)
images_condition = images_input.clone()

prompt_embeds = pipe._encode_prompt(
    prompt,
    device=device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
) # [3b, 77, 768]

ref_image = images_input[[0]] # (1, c, h, w)
ref_image_cond = images_condition[[0]] # (1, c, h, w)

with torch.no_grad():
    ref_latents = pipe.vae.encode(2*ref_image-1).latent_dist.sample() * 0.18215  # (b*f, 4, h//4, w//4)
    ref_image_latents = pipe.vae.encode(2*ref_image_cond-1).latent_dist.mode() # (b*f, 4, h//4, w//4)
    
ref_latents = rearrange(ref_latents, "1 c h w -> 1 c 1 h w") # (1, 4, 1, h//4, w//4)
ref_image_latents = rearrange(ref_image_latents, "1 c h w -> 1 c 1 h w") # (1, 4, 1, h//4, w//4)
uncond_ref_image_latents = torch.zeros_like(ref_image_latents)
ref_image_latents = torch.cat([ref_image_latents, ref_image_latents, uncond_ref_image_latents], dim=0) # (3b, 4, f, h//4, w//4)

pipe.scheduler.config.num_train_timesteps = 1000
pipe.scheduler.set_timesteps(20)

noise = torch.randn_like(ref_latents) # (b, 4, f, h//4, w//4)
ref_latents = noise

for i, t in tqdm(enumerate(pipe.scheduler.timesteps), total=len(pipe.scheduler.timesteps), desc="Inference"):
    latent_model_input = torch.cat([ref_latents] * 3) # [3b, 4, sequence_length, h//4, w//4]
    latent_model_input = torch.cat([latent_model_input, ref_image_latents], dim=1) # [3b, 8, sequence_length, h//4, w//4]
    
    # predict the noise residual
    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, t, prompt_embeds, None, None, False)[0] # [3b, 4, sequence_length, h//4, w//4]
    
    # perform classifier-free guidance
    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
    
    noise_pred = (
        noise_pred_uncond
        + guidance_scale * (noise_pred_text - noise_pred_image)
        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
    )
    
    # compute the previous noisy sample x_t -> x_t-1
    ref_latents = pipe.scheduler.step(noise_pred, t, ref_latents, return_dict=False)[0] # [b, c, f, h//4, w//4]
    
latents = rearrange(ref_latents, "b c f h w -> (b f) c h w")
latents = 1 / 0.18215 * latents # (b*f, 4, h//4, w//4)
with torch.no_grad():
    images_refine = vae.decode(latents, False)[0] # (b*f, 3, h, w)
    
ref_image = (images_refine / 2 + 0.5).clamp(0, 1) # (b*f, 3, h, w) [-1, 1] -> [0, 1]

images_sequence = images_input[:3] # (sequence_length, c, h, w)
images_sequence_cond = images_condition[:3] # (sequence_length, c, h, w)

images_sequence = torch.cat([ref_image, images_sequence], dim=0) # (sequence_length+1, c, h, w) 
images_sequence_cond = torch.cat([ref_image_cond, images_sequence_cond], dim=0) # (sequence_length+1, c, h, w)

for i in range(1, images_sequence.shape[0]):
    image0 = (images_sequence[[0]] * 255.0).float()
    image2 = (images_sequence[[i]] * 255.0).float()
    image0_cond = (images_sequence_cond[[0]] * 255.0).float()
    image2_cond = (images_sequence_cond[[i]] * 255.0).float()
    
    padder = InputPadder(image0.shape) 
    image0, image2, image0_cond, image2_cond = padder.pad(image0, image2, image0_cond, image2_cond)
    
    _, flow_fwd_ref = raft(image0_cond, image2_cond, iters=20, test_mode=True) 
    _, flow_bwd_ref = raft(image2_cond, image0_cond, iters=20, test_mode=True)

    flow_fwd_ref = padder.unpad(flow_fwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
    flow_bwd_ref = padder.unpad(flow_bwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
    
    _, mask_bwd_ref = compute_fwdbwd_mask(flow_fwd_ref, flow_bwd_ref) # (h, w)
    image0 = (image0[0].permute(1,2,0).cpu().numpy()).astype(np.uint8) # (h, w, c)
    image2 = (image2[0].permute(1,2,0).cpu().numpy()).astype(np.uint8) # (h, w, c)
    
    warp_to_image2 = warp_flow(image0, flow_bwd_ref) # (h, w, c)
    warp_to_image2 = warp_to_image2 * mask_bwd_ref[..., None] + image2 * (1 - mask_bwd_ref[..., None])
    warp_to_image2 = torch.from_numpy(warp_to_image2 / 255.0).permute(2, 0, 1).to(torch_dtype).to(device) # (c, h, w)
    
    warp_to_image2 = F.interpolate(warp_to_image2.unsqueeze(0), size=(RH, RW), mode='bilinear', align_corners=False).squeeze(0) # (c, h, w)
    
    images_sequence[i] = warp_to_image2
    
with torch.no_grad():
    latents = pipe.vae.encode(2*images_sequence-1).latent_dist.sample() * 0.18215  # (b*f, 4, h//4, w//4)
    image_latents = pipe.vae.encode(2*images_sequence_cond-1).latent_dist.mode() # (b*f, 4, h//4, w//4)

latents = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length+1) # (b, 4, f, h//4, w//4)
image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length+1) # (b, 4, f, h//4, w//4)
uncond_image_latents = torch.zeros_like(image_latents)

pipe.scheduler.config.num_train_timesteps = num_train_timesteps
pipe.scheduler.set_timesteps(diffusion_step)

noise = torch.randn_like(latents) # (b, 4, f, h//4, w//4)
latents = pipe.scheduler.add_noise(latents, noise, pipe.scheduler.timesteps[0])  
    
image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0) # (3b, 4, f, h//4, w//4)

for i, t in tqdm(enumerate(pipe.scheduler.timesteps), total=len(pipe.scheduler.timesteps), desc="Inference"):
    latent_model_input = torch.cat([latents] * 3) # [3b, 4, sequence_length, h//4, w//4]
    latent_model_input = torch.cat([latent_model_input, image_latents], dim=1) # [3b, 8, sequence_length, h//4, w//4]
    
    # predict the noise residual
    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, t, prompt_embeds, None, None, False)[0] # [3b, 4, sequence_length, h//4, w//4]
    
    # perform classifier-free guidance
    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
    
    noise_pred = (
        noise_pred_uncond
        + guidance_scale * (noise_pred_text - noise_pred_image)
        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
    )
    
    # compute the previous noisy sample x_t -> x_t-1
    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0] # [b, c, f, h//4, w//4]
    
latents = rearrange(latents, "b c f h w -> (b f) c h w")
latents = 1 / 0.18215 * latents # (b*f, 4, h//4, w//4)
with torch.no_grad():
    images_refine = vae.decode(latents, False)[0] # (b*f, 3, h, w)
    
images_refine = (images_refine / 2 + 0.5).clamp(0, 1) # (b*f, 3, h, w) [-1, 1] -> [0, 1]

for idx, j in enumerate(range(1, images_refine.shape[0])):
    torchvision.utils.save_image(images_refine[j], f"ip2p_anthor_{prompt.split(' ')[-1].replace('?', '')}_{files[idx].split('/')[-1]}")

images_sequence = images_input[-3:] # (sequence_length, c, h, w)
images_sequence_cond = images_condition[-3:] # (sequence_length, c, h, w)

images_sequence = torch.cat([ref_image, images_sequence], dim=0) # (sequence_length+1, c, h, w) 
images_sequence_cond = torch.cat([ref_image_cond, images_sequence_cond], dim=0) # (sequence_length+1, c, h, w)

for i in range(1, images_sequence.shape[0]):
    image0 = (images_sequence[[0]] * 255.0).float()
    image2 = (images_sequence[[i]] * 255.0).float()
    image0_cond = (images_sequence_cond[[0]] * 255.0).float()
    image2_cond = (images_sequence_cond[[i]] * 255.0).float()
    
    padder = InputPadder(image0.shape) 
    image0, image2, image0_cond, image2_cond = padder.pad(image0, image2, image0_cond, image2_cond)
    
    _, flow_fwd_ref = raft(image0_cond, image2_cond, iters=20, test_mode=True) 
    _, flow_bwd_ref = raft(image2_cond, image0_cond, iters=20, test_mode=True)

    flow_fwd_ref = padder.unpad(flow_fwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
    flow_bwd_ref = padder.unpad(flow_bwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
    
    _, mask_bwd_ref = compute_fwdbwd_mask(flow_fwd_ref, flow_bwd_ref) # (h, w)
    image0 = (image0[0].permute(1,2,0).cpu().numpy()).astype(np.uint8) # (h, w, c)
    image2 = (image2[0].permute(1,2,0).cpu().numpy()).astype(np.uint8) # (h, w, c)
    
    warp_to_image2 = warp_flow(image0, flow_bwd_ref) # (h, w, c)
    warp_to_image2 = warp_to_image2 * mask_bwd_ref[..., None] + image2 * (1 - mask_bwd_ref[..., None])
    warp_to_image2 = torch.from_numpy(warp_to_image2 / 255.0).permute(2, 0, 1).to(torch_dtype).to(device) # (c, h, w)
    
    warp_to_image2 = F.interpolate(warp_to_image2.unsqueeze(0), size=(RH, RW), mode='bilinear', align_corners=False).squeeze(0) # (c, h, w)
    
    images_sequence[i] = warp_to_image2
    
with torch.no_grad():
    latents = pipe.vae.encode(2*images_sequence-1).latent_dist.sample() * 0.18215  # (b*f, 4, h//4, w//4)
    image_latents = pipe.vae.encode(2*images_sequence_cond-1).latent_dist.mode() # (b*f, 4, h//4, w//4)

latents = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length+1) # (b, 4, f, h//4, w//4)
image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length+1) # (b, 4, f, h//4, w//4)
uncond_image_latents = torch.zeros_like(image_latents)

pipe.scheduler.config.num_train_timesteps = num_train_timesteps
pipe.scheduler.set_timesteps(diffusion_step)

noise = torch.randn_like(latents) # (b, 4, f, h//4, w//4)
latents = pipe.scheduler.add_noise(latents, noise, pipe.scheduler.timesteps[0])  
    
image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0) # (3b, 4, f, h//4, w//4)

for i, t in tqdm(enumerate(pipe.scheduler.timesteps), total=len(pipe.scheduler.timesteps), desc="Inference"):
    latent_model_input = torch.cat([latents] * 3) # [3b, 4, sequence_length, h//4, w//4]
    latent_model_input = torch.cat([latent_model_input, image_latents], dim=1) # [3b, 8, sequence_length, h//4, w//4]
    
    # predict the noise residual
    with torch.no_grad():
        noise_pred = pipe.unet(latent_model_input, t, prompt_embeds, None, None, False)[0] # [3b, 4, sequence_length, h//4, w//4]
    
    # perform classifier-free guidance
    noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
    
    noise_pred = (
        noise_pred_uncond
        + guidance_scale * (noise_pred_text - noise_pred_image)
        + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
    )
    
    # compute the previous noisy sample x_t -> x_t-1
    latents = pipe.scheduler.step(noise_pred, t, latents, return_dict=False)[0] # [b, c, f, h//4, w//4]
    
latents = rearrange(latents, "b c f h w -> (b f) c h w")
latents = 1 / 0.18215 * latents # (b*f, 4, h//4, w//4)
with torch.no_grad():
    images_refine = vae.decode(latents, False)[0] # (b*f, 3, h, w)
    
images_refine = (images_refine / 2 + 0.5).clamp(0, 1) # (b*f, 3, h, w) [-1, 1] -> [0, 1]

for idx, j in enumerate(range(1, images_refine.shape[0])):
    torchvision.utils.save_image(images_refine[j], f"ip2p_anthor_{prompt.split(' ')[-1].replace('?', '')}_{files[idx+3].split('/')[-1]}")
    
    
    
