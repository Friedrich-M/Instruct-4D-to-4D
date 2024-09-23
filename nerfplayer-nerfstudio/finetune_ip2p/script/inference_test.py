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
from PIL import Image
import torch.nn.functional as F

from einops import rearrange
from tqdm import tqdm
import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"
tokenizer = CLIPTokenizer.from_pretrained(IP2P_SOURCE, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(IP2P_SOURCE, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(IP2P_SOURCE, subfolder="vae")
unet = UNet3DConditionModel.from_pretrained_2d(IP2P_SOURCE, subfolder="unet")
ddim_inv_scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder='scheduler', torch_dtype=torch_dtype)

vae.requires_grad_(False)
text_encoder.requires_grad_(False)
unet.requires_grad_(False)

slide_window_length = 5

vae = vae.to(device, dtype=torch_dtype)
text_encoder = text_encoder.to(device, dtype=torch_dtype)
unet = unet.to(device, dtype=torch_dtype)
        
pipe = InstructPix2PixPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=ddim_inv_scheduler,
    )

data_path = 'data/cat'
files = sorted(os.listdir(data_path))
files = [os.path.join(data_path, file) for file in files]

images = []
for file in files:
    img = Image.open(file).convert('RGB')
    img = torch.from_numpy(np.array(img) / 255).permute(2, 0, 1).unsqueeze(0).to(torch.float16).to("cuda")
    images.append(img)
    
images = torch.cat(images, dim=0) # (f, c, h, w)
images = images[:slide_window_length] # (f, c, h, w)

sequence_length, _, H, W = images.shape
RH, RW = 960, 720

images = images.to(device, dtype=torch_dtype)
images = F.interpolate(images, size=(RH, RW), mode='bilinear', align_corners=False) # (f, c, h, w)
images_cond = images.clone().to(device, dtype=torch_dtype) # (f, c, h, w)

edited_image_0 = 'noisy_latents_image_0.png'
edited_image_0 = Image.open(edited_image_0).convert('RGB')
edited_image_0 = torch.from_numpy(np.array(edited_image_0) / 255).permute(2, 0, 1).unsqueeze(0).to(torch.float16).to("cuda") # (1, c, h, w)

edited_image_1 = 'noisy_latents_image_1.png'
edited_image_1 = Image.open(edited_image_1).convert('RGB')
edited_image_1 = torch.from_numpy(np.array(edited_image_1) / 255).permute(2, 0, 1).unsqueeze(0).to(torch.float16).to("cuda") # (1, c, h, w)

edited_image_0 = edited_image_0.to(device, dtype=torch_dtype)
edited_image_1 = edited_image_1.to(device, dtype=torch_dtype)
edited_image_0 = F.interpolate(edited_image_0, size=(RH, RW), mode='bilinear', align_corners=False) # (f, c, h, w)
edited_image_1 = F.interpolate(edited_image_1, size=(RH, RW), mode='bilinear', align_corners=False) # (f, c, h, w)
images[0] = edited_image_0[0]
images[1] = edited_image_1[0]

torchvision.utils.save_image(images, 'original_images_test.png', nrow=sequence_length, padding=0, pad_value=1)
torchvision.utils.save_image(images_cond, 'conditional_images_test.png', nrow=sequence_length, padding=0, pad_value=1)

latents = pipe.vae.encode(2*images-1).latent_dist.sample() * 0.18215  # (b*f, 4, h//4, w//4)
image_latents = pipe.vae.encode(2*images_cond-1).latent_dist.mode() # (b*f, 4, h//4, w//4)

latents = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
uncond_image_latents = torch.zeros_like(image_latents)

ddim_inv_prompt = ""
prompt = "Make it look like a Van Gogh painting"
diffusion_step = 20
num_train_timesteps = 1000
guidance_scale = 8.5
image_guidance_scale = 1.5
latents_type = 'noisy_latents' # 'ddim_inv', 'ddim_inv_classifier', 'noise', 'noisy_latents'

prompt_embeds = pipe._encode_prompt(
    prompt,
    device=device,
    num_images_per_prompt=1,
    do_classifier_free_guidance=True,
) # [3b, 77, 768]

pipe.scheduler.config.num_train_timesteps = num_train_timesteps
ddim_inv_scheduler.config.num_train_timesteps = num_train_timesteps

pipe.scheduler.set_timesteps(diffusion_step)
ddim_inv_scheduler.set_timesteps(diffusion_step)

if latents_type == 'ddim_inv':
    ddim_inv_latent = ddim_inversion(pipe, ddim_inv_scheduler, latent=latents, image_latents=image_latents, num_inv_steps=diffusion_step, prompt=ddim_inv_prompt)[-1].to(torch_dtype)
    latents = ddim_inv_latent
elif latents_type == 'ddim_inv_classifier':
    ddim_inv_latent = ddim_inversion_classifier(pipe, ddim_inv_scheduler, latent=latents, image_latents=image_latents, num_inv_steps=diffusion_step, prompt=ddim_inv_prompt, guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale).to(torch_dtype)
    latents = ddim_inv_latent
elif latents_type == 'noise':
    latents = torch.randn_like(latents)
elif latents_type == 'noisy_latents':
    noise = torch.randn_like(latents)
    latents = pipe.scheduler.add_noise(latents, noise, pipe.scheduler.timesteps[0])  # type: ignore
    
image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0) # (3b, 4, f, h//4, w//4)

for i, t in tqdm(enumerate(pipe.scheduler.timesteps), total=len(pipe.scheduler.timesteps), desc="Inference"):
    latent_model_input = torch.cat([latents] * 3) # [3b, 4, sequence_length, h//4, w//4]
    latent_model_input = torch.cat([latent_model_input, image_latents], dim=1) # [3b, 8, sequence_length, h//4, w//4]
    
    # predict the noise residual
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
video = vae.decode(latents, False)[0] # (b*f, 3, h, w)
    
video = (video / 2 + 0.5).clamp(0, 1) # (b*f, 3, h, w) [-1, 1] -> [0, 1]

# for i in range(sequence_length):
#     torchvision.utils.save_image(video[i:i+1], f'{latents_type}_image_{i}.png', nrow=1, padding=0, pad_value=1)

torchvision.utils.save_image(video, f'{latents_type}_images_test.png', nrow=sequence_length, padding=0, pad_value=1)
