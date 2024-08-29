import sys
from typing import Union

import torch
from rich.console import Console
from torch import Tensor, nn
from jaxtyping import Float
import torch.nn.functional as F

CONSOLE = Console(width=120)

import sys; import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

try:
    from diffusers import (
        DDIMScheduler,
        AutoencoderKL,
    )
    from transformers import (
        CLIPTextModel, 
        CLIPTokenizer
    )
except ImportError:
    CONSOLE.print("[bold red]Missing Stable Diffusion packages.")
    CONSOLE.print(r"Install using [yellow]pip install nerfstudio\[gen][/yellow]")
    CONSOLE.print(r"or [yellow]pip install -e .\[gen][/yellow] if installing from source.")
    sys.exit(1)
    
from ip2p_models.models.ip2p_pipeline import InstructPix2PixPipeline
from ip2p_models.models.ip2p_unet import UNet3DConditionModel
from ip2p_models.models.ip2p_utils import ddim_inversion, ddim_inversion_classifier

from einops import rearrange

CONST_SCALE = 0.18215

DDIM_SOURCE = "CompVis/stable-diffusion-v1-4"
IP2P_SOURCE = "timbrooks/instruct-pix2pix"

class SequenceInstructPix2Pix(nn.Module):

    def __init__(self, device: Union[torch.device, str], ip2p_use_full_precision=False,) -> None:
        super().__init__()
        
        CONSOLE.print("Loading Sequence InstructPix2Pix...")
        self.device = device
        self.weights_dtype = torch.float32 if ip2p_use_full_precision else torch.float16
                
        tokenizer = CLIPTokenizer.from_pretrained(IP2P_SOURCE, subfolder="tokenizer")
        text_encoder = CLIPTextModel.from_pretrained(IP2P_SOURCE, subfolder="text_encoder")
        vae = AutoencoderKL.from_pretrained(IP2P_SOURCE, subfolder="vae")
        unet = UNet3DConditionModel.from_pretrained_2d(IP2P_SOURCE, subfolder="unet")
        
        self.ddim_inv_scheduler = DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder='scheduler')
        
        vae.requires_grad_(False)
        text_encoder.requires_grad_(False)
        unet.requires_grad_(False)
        
        vae = vae.to(self.device, dtype=self.weights_dtype)
        text_encoder = text_encoder.to(self.device, dtype=self.weights_dtype)
        unet = unet.to(self.device, dtype=self.weights_dtype)
        
        pipe = InstructPix2PixPipeline(
            vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
            scheduler=DDIMScheduler.from_pretrained(DDIM_SOURCE, subfolder='scheduler'),
        )
        
        self.pipe = pipe
        self.vae = vae
        self.unet = unet
        
        self.scheduler = self.pipe.scheduler
        
        CONSOLE.print("Sequence InstructPix2Pix loaded!")
    
    def edit_sequence(self,
        images: Float[Tensor, "BS F 3 H W"],
        images_cond: Float[Tensor, "BS F 3 H W"],
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        diffusion_steps: int = 20,
        prompt: str = "",
        noisy_latent_type: str = "ddim_inv",
        T: int = 1000,
    ) -> torch.Tensor:
        
        batch_size, sequence_length, _, H, W = images.shape
        RH, RW = H // 8 * 8, W // 8 * 8
            
        images = rearrange(images, "b f c h w -> (b f) c h w") # (b*f, 3, h, w)
        images = F.interpolate(images, size=(RH, RW), mode="bilinear", align_corners=False)
        images = images.to(self.device, dtype=self.weights_dtype)
        
        images_cond = rearrange(images_cond, "b f c h w -> (b f) c h w") # (b*f, 3, h, w)
        images_cond = F.interpolate(images_cond, size=(RH, RW), mode="bilinear", align_corners=False)
        images_cond = images_cond.to(self.device, dtype=self.weights_dtype)
        
        with torch.no_grad():
            latents = self.imgs_to_latent(images) # (b*f, 4, h//4, w//4)
            image_latents = self.prepare_image_latents(images_cond) # (b*f, 4, h//4, w//4)
            
        # reshape back to batch and frame dimensions
        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
        image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length) 
        images_cond = rearrange(images_cond, "(b f) c h w -> b f c h w", f=sequence_length) # (b, f, 3, h, w)
            
        self.set_num_steps(T)
        self.scheduler.set_timesteps(diffusion_steps)
        self.ddim_inv_scheduler.set_timesteps(diffusion_steps)
        
        if noisy_latent_type == 'noise':
            latents = None
        elif noisy_latent_type == 'noisy_latent':
            latents = self.scheduler.add_noise(latents, torch.randn_like(latents), self.scheduler.timesteps[0])
        elif noisy_latent_type == 'ddim_inv':
            latents = ddim_inversion(self.pipe, self.ddim_inv_scheduler, latent=latents, image_latents=image_latents, num_inv_steps=diffusion_steps, prompt=prompt)[-1].to(self.device, dtype=self.weights_dtype)
            
        with torch.no_grad():
            video = self.pipe(prompt=prompt, video_length=sequence_length, image=images_cond, latents=latents, height=RH, width=RW, num_inference_steps=diffusion_steps, guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale).videos 
        
        return video # (b, 3, f, h, w)
    
    def latents_to_img(self, latents: Float[Tensor, "BS 4 H W"]) -> Float[Tensor, "BS 3 H W"]:
        latents = 1 / CONST_SCALE * latents

        with torch.no_grad():
            imgs = self.vae.decode(latents).sample

        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs
    
    def imgs_to_latent(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * CONST_SCALE

        return latents

    def prepare_image_latents(self, imgs: Float[Tensor, "BS 3 H W"]) -> Float[Tensor, "BS 4 H W"]:
        imgs = 2 * imgs - 1

        image_latents = self.vae.encode(imgs).latent_dist.mode()

        return image_latents
    
    def set_num_steps(self, num_steps: int = None) -> None:
        num_steps = num_steps or 1000
        self.scheduler.config.num_train_timesteps = num_steps
        self.ddim_inv_scheduler.config.num_train_timesteps = num_steps 
    
   