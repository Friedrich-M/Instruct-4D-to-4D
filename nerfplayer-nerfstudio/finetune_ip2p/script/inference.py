from models.ip2p_pipeline import InstructPix2PixPipeline
from models.ip2p_unet import UNet3DConditionModel
from models.ip2p_utils import save_videos_grid, ddim_inversion, write_video_to_file
import torch

import decord
decord.bridge.set_bridge('torch')
from einops import rearrange
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

pretrained_model_path = "timbrooks/instruct-pix2pix"
my_model_path = "timbrooks/instruct-pix2pix"
unet = UNet3DConditionModel.from_pretrained_2d(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')
pipe = InstructPix2PixPipeline.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to("cuda")

pipe.unet.eval()
pipe.vae.eval()

# Freeze vae and text_encoder
pipe.vae.requires_grad_(False)
pipe.text_encoder.requires_grad_(False)
pipe.unet.requires_grad_(False)

vr = decord.VideoReader('data/drum_cam.mp4', width=512, height=256)
sample_index = list(range(0, len(vr), 1))[:10]
video = vr.get_batch(sample_index) # (f, h, w, c)
video = rearrange(video, "f h w c -> f c h w").unsqueeze(0).to(torch.float16).to("cuda") # (1, f, c, h, w)
sequence_length = video.shape[1]

image = rearrange(video, "b f c h w -> (b f) c h w").to(torch.float16) # (b*f, c, h, w)
latents = pipe.vae.encode(image).latent_dist.sample().to(torch.float16) # (b*f, 4, h//4, w//4)
image_latents = pipe.vae.encode(image).latent_dist.mode().to(torch.float16) # (b*f, 4, h//4, w//4)

latents = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
latents = latents * 0.18215 
image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)

prompt = "turn it into a chocolate cake"

from diffusers import DDIMScheduler
ddim_inv_scheduler = DDIMScheduler.from_pretrained("timbrooks/instruct-pix2pix", subfolder='scheduler', torch_dtype=torch.float16)
ddim_inv_scheduler.set_timesteps(50)
ddim_inv_latent = ddim_inversion(pipe, ddim_inv_scheduler, latent=latents, image_latents=image_latents, num_inv_steps=50, prompt="")[-1].to(torch.float16)

generator = torch.Generator(device=ddim_inv_latent.device)
generator.manual_seed(33)
video = pipe(prompt, image=video, generator=generator, latents=ddim_inv_latent, video_length=30, height=256, width=512, num_inference_steps=50, guidance_scale=7.5, image_guidance_scale=1.5).videos

save_videos_grid(video, f"./{prompt}.gif")