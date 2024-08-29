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
import configargparse as argparse
import torch
import torchvision
import numpy as np
from PIL import Image, ImageOps
import torch.nn.functional as F

from einops import rearrange; 
from PIL import Image

from einops import rearrange
from tqdm import tqdm
import math
import os

from pytorch_lightning import seed_everything
seed_everything(17070)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="./examples/coffee_frame_2x")
    parser.add_argument("--prompt", type=str, default="What if it was painted by Van Gogh?")
    parser.add_argument("--resize", type=int, default=None)
    parser.add_argument("--sequence_length", type=int, default=6)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--image_guidance_scale", type=float, default=1.5)
    return parser.parse_args()

args = parse_args()
data_path = args.image_dir
sequence_length = args.sequence_length
prompt = args.prompt
guidance_scale = args.guidance_scale
image_guidance_scale = args.image_guidance_scale
diffusion_step = args.steps
num_train_timesteps = 1000
latents_type = 'noisy_latents' # 'noise', 'noisy_latents'

tag = prompt.split(' ')[-1].replace('?', '')

files = sorted(os.listdir(data_path), key=lambda x: int(x.split('.')[0]))
files = [os.path.join(data_path, file) for file in files]
files = files[:sequence_length]
print(f'Loaded {len(files)} images from {data_path}')

images = []
for file in files:
    image = Image.open(file).convert('RGB')
    width, height = image.size
    if args.resize is None:
        args.resize = max(width, height)
    factor = args.resize / max(width, height)
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
images = F.interpolate(images, size=(RH, RW), mode='bilinear', align_corners=False) # (f, c, h, w)
images_cond = images.clone().to(device, dtype=torch_dtype) # (f, c, h, w)

with torch.no_grad():
    latents = pipe.vae.encode(2*images-1).latent_dist.sample() * 0.18215  # (b*f, 4, h//4, w//4)
    image_latents = pipe.vae.encode(2*images_cond-1).latent_dist.mode() # (b*f, 4, h//4, w//4)

latents = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
uncond_image_latents = torch.zeros_like(image_latents)

prompt_embeds = pipe._encode_prompt(
    prompt,
    device=device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
) # [3b, 77, 768]

pipe.scheduler.config.num_train_timesteps = num_train_timesteps
pipe.scheduler.set_timesteps(diffusion_step)

if latents_type == 'noise':
    latents = torch.randn_like(latents)
elif latents_type == 'noisy_latents':
    noise = torch.randn_like(latents) # (b, 4, f, h//4, w//4)
    latents = pipe.scheduler.add_noise(latents, noise, pipe.scheduler.timesteps[0])  
else:
    raise NotImplementedError
    
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
    video = vae.decode(latents, False)[0] # (b*f, 3, h, w)
    
video = (video / 2 + 0.5).clamp(0, 1) # (b*f, 3, h, w) [-1, 1] -> [0, 1]

for i in range(sequence_length):
    torchvision.utils.save_image(video[i], f"ip2p_sequence_{prompt.split(' ')[-1].replace('?', '')}_{files[i].split('/')[-1]}")

