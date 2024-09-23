import os
import imageio
import numpy as np
from typing import Union

import torch
import torchvision

from tqdm import tqdm
from einops import rearrange

import cv2

def write_video_to_file(frames: torch.Tensor, file_name: str):
    frames = rearrange(frames, "b c t h w -> t b c h w") # (num_frames, b, c, h, w)
    outputs = []
    for x in frames:
        x = torchvision.utils.make_grid(x)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        x = (x * 255).cpu().numpy().astype(np.uint8)
        outputs.append(x)
        
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    # Photo tourism image sizes differ
    frames = outputs
    sizes = np.array([frame.shape[:2] for frame in frames])
    same_size_frames = np.unique(sizes, axis=0).shape[0] == 1
    if same_size_frames:
        height, width = frames[0].shape[:2]
        video = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        for img in frames:
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            video.write(img[:, :, ::-1])  # opencv uses BGR instead of RGB
        cv2.destroyAllWindows()
        video.release()
    else:
        height = sizes[:, 0].max()
        width = sizes[:, 1].max()
        video = cv2.VideoWriter(
            file_name, cv2.VideoWriter_fourcc(*'mp4v'), 5, (width, height))
        for img in frames:
            image = np.zeros((height, width, 3), dtype=np.uint8)
            h, w = img.shape[:2]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            image[(height-h)//2:(height-h)//2+h, (width-w)//2:(width-w)//2+w, :] = img
            video.write(image[:, :, ::-1])  # opencv uses BGR instead of RGB
        cv2.destroyAllWindows()
        video.release()


def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).cpu().numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, duration=1000 * 1 / fps)
     

# DDIM Inversion
@torch.no_grad()
def init_prompt(prompt, pipeline):
    uncond_input = pipeline.tokenizer(
        [""], padding="max_length", max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt"
    )
    uncond_embeddings = pipeline.text_encoder(uncond_input.input_ids.to(pipeline.device))[0]
    text_input = pipeline.tokenizer(
        [prompt],
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = pipeline.text_encoder(text_input.input_ids.to(pipeline.device))[0]
    context = torch.cat([uncond_embeddings, text_embeddings])

    return context


def next_step(model_output: Union[torch.FloatTensor, np.ndarray], timestep: int,
              sample: Union[torch.FloatTensor, np.ndarray], ddim_scheduler):
    timestep, next_timestep = min(
        timestep - ddim_scheduler.config.num_train_timesteps // ddim_scheduler.num_inference_steps, 999), timestep
    alpha_prod_t = ddim_scheduler.alphas_cumprod[timestep] if timestep >= 0 else ddim_scheduler.final_alpha_cumprod
    alpha_prod_t_next = ddim_scheduler.alphas_cumprod[next_timestep]
    beta_prod_t = 1 - alpha_prod_t
    next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
    next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
    next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
    return next_sample


def get_noise_pred_single(latents, t, context, unet):
    noise_pred = unet(latents, t, encoder_hidden_states=context)["sample"]
    return noise_pred


@torch.no_grad()
def ddim_loop(pipeline, ddim_scheduler, latent, image_latents, num_inv_steps, prompt):
    context = pipeline._encode_prompt(
        prompt, device=latent.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
    ) # (3b, 77, 768)
    cond_embeddings, uncond_embeddings, uncond_embeddings = context.chunk(3)
    all_latent = [latent]
    latent = latent.clone().detach()
    for i in range(num_inv_steps):
        latent_input = torch.cat([latent, image_latents], dim=1) # (b, 8, num_frames, h//4, w//4)
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent_input, t, cond_embeddings, pipeline.unet) # (b, 4, num_frames, h//4, w//4)
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
        all_latent.append(latent)
    return all_latent

@torch.no_grad()
def ddim_loop_classifier(pipeline, ddim_scheduler, latent, image_latents, num_inv_steps, prompt, guidance_scale, image_guidance_scale):
    text_embeddings = pipeline._encode_prompt(
        prompt, device=latent.device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
    ) # (3b, 77, 768)
    uncond_image_latents = torch.zeros_like(image_latents)
    image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)
    latent = latent.clone().detach()
    for i in range(num_inv_steps):
        latent_model_input = torch.cat([latent] * 3)
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=1) 
        t = ddim_scheduler.timesteps[len(ddim_scheduler.timesteps) - i - 1]
        noise_pred = get_noise_pred_single(latent_model_input, t, text_embeddings, pipeline.unet) # (b, 4, num_frames, h//4, w//4)
        # perform classifier-free guidance
        noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
        noise_pred = (
            noise_pred_uncond
            + guidance_scale * (noise_pred_text - noise_pred_image)
            + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
        )
        latent = next_step(noise_pred, t, latent, ddim_scheduler)
    return latent


@torch.no_grad()
def ddim_inversion(pipeline, ddim_scheduler, latent, image_latents, num_inv_steps, prompt=""):
    ddim_latents = ddim_loop(pipeline, ddim_scheduler, latent, image_latents, num_inv_steps, prompt)
    return ddim_latents

@torch.no_grad()
def ddim_inversion_classifier(pipeline, ddim_scheduler, latent, image_latents, num_inv_steps, prompt="", guidance_scale=7.5, image_guidance_scale=1.5):
    ddim_latents = ddim_loop_classifier(pipeline, ddim_scheduler, latent, image_latents, num_inv_steps, prompt, guidance_scale, image_guidance_scale)
    return ddim_latents
