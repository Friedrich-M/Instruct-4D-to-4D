import torch
from tqdm import tqdm
from typing import Dict, Optional, Tuple, Union
from omegaconf import OmegaConf
import logging

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

import diffusers
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from einops import rearrange

from models.ip2p_pipeline import InstructPix2PixPipeline
from models.ip2p_unet import UNet3DConditionModel
from models.ip2p_dataset import VideoDataset, MultiFrameDataset, MultiViewDataset
from models.ip2p_utils import save_videos_grid, ddim_inversion, write_video_to_file

import sys; import os
dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(dir, os.path.pardir)))

logger = get_logger(__name__, log_level="INFO")

def main(
    train_data: Dict,
    validation_data: Dict,
    pretrained_model_path: str = "timbrooks/instruct-pix2pix",
    ddim_source = "CompVis/stable-diffusion-v1-4",
    learning_rate=3e-5,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    train_batch_size: int = 1,
    lr_warmup_steps: int = 0,
    max_train_steps: int = 1000,
    lr_scheduler: str = "constant",
    seed: int = 1314,
    output_dir: str = "./output/lego4",
    trainable_modules: Tuple[str] = (
        "attn1.to_q",
        "attn2.to_q",
        "attn_temp",
    ),
    max_grad_norm: float = 1.0,
    validation_steps: int = 100,
    gradient_accumulation_steps: int = 1,
    mixed_precision: str = "fp16",
):
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)
    
    # Handle the output folder creation
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        
    noise_scheduler = DDPMScheduler.from_pretrained(ddim_source, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    unet = UNet3DConditionModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    for name, module in unet.named_modules():
        if name.endswith(tuple(trainable_modules)):
            for params in module.parameters():
                params.requires_grad = True
    
    unet.enable_gradient_checkpointing() 
    
    optimizer_cls = torch.optim.AdamW
    
    optimizer = optimizer_cls(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    
    # Get the training dataset
    train_dataset = MultiFrameDataset(**train_data)
    
    train_dataset[0]
        
    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]
        
    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=4
    )
    
    # Get the validation pipeline
    validation_pipeline = InstructPix2PixPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
  
    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)
    
    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=max_train_steps,
    )
    
    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )
    
    weight_dtype = torch.float16
    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("ip2p-fine-tune")
        
    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes
    
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    
    global_step = 0
    first_epoch = 0
    
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    
    for epoch in range(first_epoch, max_train_steps):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                
                image = pixel_values.clone()
                sequence_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w") # (b*f, 3, h, w)
                
                latents = vae.encode(pixel_values).latent_dist.sample() # (b*f, 4, h//4, w//4)
                image_latents = vae.encode(pixel_values).latent_dist.mode() # (b*f, 4, h//4, w//4) 
                
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
                latents = latents * 0.18215 
                
                image_latents = rearrange(image_latents, "(b f) c h w -> b c f h w", f=sequence_length) # (b, 4, f, h//4, w//4)
                # uncond_image_latents = torch.zeros_like(image_latents)
                
                # image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0) # (3*b, 4, f, h//4, w//4)
                
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents) # (b, 4, f, h//4, w//4)
                bsz = latents.shape[0]
                                
                # Sample a random timestep for each video
                timesteps = torch.randint(
                    0,
                    noise_scheduler.num_train_timesteps, 
                    (bsz,), 
                    device=latents.device,
                    dtype=torch.long
                )
                                
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps) # (b, 4, f, h//4, w//4)
                
                # Get the text embedding for conditioning
                text_prompt = text_encoder(batch["prompt_ids"])[0] # (b, 77, 768)
                non_text_prompt = torch.zeros_like(text_prompt) # (b, 77, 768)
                
                encoder_hidden_states = text_prompt # (b, 77, 768)
                # encoder_hidden_states = torch.cat([text_prompt, non_text_prompt, non_text_prompt], dim=0) # (3*b, 77, 768)
                                
                latent_model_input = torch.cat([noisy_latents, image_latents], dim=1) # (b, 8, f, h//4, w//4)
                # latent_model_input = torch.cat([noisy_latents] * 3) # (3*b, 4, f, h//4, w//4)
                # latent_model_input = torch.cat([latent_model_input, image_latents], dim=1) # (3*b, 8, f, h//4, w//4)
                
                # Predict the noise residual and compute loss
                noise_pred = unet(latent_model_input, timesteps, encoder_hidden_states).sample # (3*b, 4, f, h//4, w//4)
                
                # perform classifier-free guidance
                # noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                # noise_pred = (
                #     noise_pred_uncond
                #     + text_guidance_scale * (noise_pred_text - noise_pred_image)
                #     + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                # ) # (b, 4, f, h//4, w//4)
            
                loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean") 
                
                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps # We manually accumulate loss for Gradient Accumulation

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0
                
                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(seed)
                        
                        ddim_inv_latent = None
                        if validation_data.use_inv_latent:
                            inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                            ddim_inv_latent = ddim_inversion(
                                validation_pipeline, ddim_inv_scheduler, latent=latents, image_latents=image_latents, num_inv_steps=validation_data.num_inv_steps, prompt="")[-1].to(weight_dtype)
                            torch.save(ddim_inv_latent, inv_latents_path)
                        
                        for idx, prompt in enumerate(validation_data.prompts):
                            sample = validation_pipeline(prompt=prompt, image=image, generator=generator, latents=ddim_inv_latent, **validation_data).videos
                            # save_videos_grid(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                            write_video_to_file(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.mp4")
                            samples.append(sample)
                            
                        samples = torch.concat(samples)
                        save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                        save_videos_grid(samples, save_path)
                        logger.info(f"Saved samples to {save_path}")
                
            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break
    
    # Create the pipeline using the trained modules and save it.
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = InstructPix2PixPipeline.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)
    
    accelerator.end_training()
    
    accelerator.end_training()
        
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/lego.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))





















