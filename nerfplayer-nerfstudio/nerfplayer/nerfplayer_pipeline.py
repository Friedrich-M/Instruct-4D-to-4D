# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""InstructPix2Pix Pipeline and trainer"""

from dataclasses import dataclass, field
from itertools import cycle
import math
from typing import Optional, Type, Any, Mapping
from einops import rearrange
import torchvision
import torch
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText

import threading
import numpy as np
import cv2
import argparse
import random

from nerfplayer.nerfplayer_datamanager import NerfplayerDataManagerConfig
from nerfplayer.ip2p_sequence import SequenceInstructPix2Pix 

from finetune_ip2p.RAFT.raft import RAFT
from finetune_ip2p.RAFT.utils.utils import InputPadder

import warnings; warnings.filterwarnings("ignore")

@dataclass
class NerfplayerPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: NerfplayerPipeline)
    """target class to instantiate"""
    datamanager: NerfplayerDataManagerConfig = NerfplayerDataManagerConfig()
    """specifies the datamanager config"""
    prompt: str = "Original"
    """prompt for InstructPix2Pix"""
    guidance_scale: float = 7.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    refine_diffusion_steps: int = 3
    """Number of diffusion steps to take for refinement"""
    refine_num_steps: int = 600
    """Number of denoise steps to take for refinement"""
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = False
    """Whether to use full precision for InstructPix2Pix"""
    resize_512: bool = False
    """whether to resize the images to 512x512"""
    small: bool = False
    """whether to use the small model for RAFT"""
    mixed_precision: bool = False
    """whether to use mixed precision for RAFT"""
    model_path: str = "finetune_ip2p/weights/raft-things.pth"
    
class NerfplayerPipeline(VanillaPipeline):
    """InstructNeRF2NeRF pipeline"""

    config: NerfplayerPipelineConfig

    def __init__(
        self,
        config: NerfplayerPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        # select device for InstructPix2Pix
        self.ip2p_device = (
            torch.device(device)
            if self.config.ip2p_device is None
            else torch.device(self.config.ip2p_device)
        )

        self.ip2p = SequenceInstructPix2Pix(device=self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)
        
        raft = torch.nn.DataParallel(RAFT(args=argparse.Namespace(small=self.config.small, mixed_precision=self.config.mixed_precision)))
        raft.load_state_dict(torch.load(self.config.model_path))
        raft = raft.module
        self.raft = raft
        
        self.raft.to(self.ip2p_device)
        self.raft.requires_grad_(False)
        self.raft.eval()
        
        # keep track of spot in dataset
        self.sequence_length = 5 # jointly edit X images
        self.overlap_length = 1 # overlap X images for inference
        
        self.train_dataset_length = len(self.datamanager.train_dataparser_outputs.image_filenames)
        self.train_indices_order = cycle(range(self.train_dataset_length))
        
        self.buffer_value = 0.9 # buffer value for dataset update
        
        # lock for data and model, since we are using multiple threads
        self.data_lock = threading.Lock() 
        self.model_lock = threading.Lock()
        
        # keep track of current step
        self.current_step = 0
        self.start_step = 0 
        self.end_step = 0
        
        # viewer elements
        self.prompt_box = ViewerText(name="Prompt", default_value=self.config.prompt, cb_hook=self.prompt_callback)
        self.guidance_scale_box = ViewerNumber(name="Text Guidance Scale", default_value=self.config.guidance_scale, cb_hook=self.guidance_scale_callback)
        self.image_guidance_scale_box = ViewerNumber(name="Image Guidance Scale", default_value=self.config.image_guidance_scale, cb_hook=self.image_guidance_scale_callback)


    def guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for guidance scale slider"""
        self.config.guidance_scale = handle.value

    def image_guidance_scale_callback(self, handle: ViewerText) -> None:
        """Callback for text guidance scale slider"""
        self.config.image_guidance_scale = handle.value

    def prompt_callback(self, handle: ViewerText) -> None:
        """Callback for prompt box, change prompt in config and update text embedding"""
        self.config.prompt = handle.value
            
    def test_edit(self, key_frame:int=0):
        update_batch_index = 0
        
        while (self.current_step - self.start_step) < self.buffer_value * (self.end_step - self.start_step):    
            self.frame_batch_index = 0
            tag = self.config.prompt.split(" ")[-1].replace(" ", "").replace(",", "").replace(".", "").replace("?", "").replace("!", "")
            
            key_frame_spot = int(torch.where(self.datamanager.image_batch["image_idx"] == torch.tensor(key_frame))[0][0])
            key_frame_camera_transforms = self.datamanager.train_camera_optimizer(torch.tensor(key_frame).unsqueeze(dim=0))
            key_frame_camera = self.datamanager.train_dataparser_outputs.cameras[key_frame].to(self.device)
            
            key_frame_condition_image = self.datamanager.original_image_batch["image"][key_frame_spot]
            
            key_frame_ray_bundle = key_frame_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=key_frame_camera_transforms)
            camera_outputs = self.model.get_outputs_for_camera_ray_bundle(key_frame_ray_bundle)
            
            key_frame_render_image = camera_outputs["rgb"] # (H, W, 3)
            key_frame_render_image = rearrange(key_frame_render_image, 'H W C -> 1 C H W') # (N, C, H, W)
            key_frame_condition_image = rearrange(key_frame_condition_image, 'H W C -> 1 C H W') # (N, C, H, W)
            
            if self.config.resize_512:
                height, width = key_frame_render_image.size()[-2:]
                factor = 512 / max(height, width)
                factor = math.ceil(min(height, width) * factor / 64) * 64 / min(height, width)
                height = int((height * factor) // 64) * 64
                width = int((width * factor) // 64) * 64
                key_frame_render_image = F.interpolate(key_frame_render_image, size=(height, width), mode="bilinear", align_corners=False) # (N, C, RH, RW)
                key_frame_condition_image = F.interpolate(key_frame_condition_image, size=(height, width), mode="bilinear", align_corners=False) # (N, C, RH, RW)
                
            torchvision.utils.save_image(key_frame_render_image, f'key_frame_render_image_{tag}.png', nrow=key_frame_render_image.shape[0], padding=0)
            torchvision.utils.save_image(key_frame_condition_image, f'key_frame_condition_image_{tag}.png', nrow=key_frame_condition_image.shape[0], padding=0)
            
            key_frame_edit_image = self.ip2p.edit_sequence(
                images=key_frame_render_image.to(self.ip2p_device), 
                images_cond=key_frame_condition_image.to(self.ip2p_device),
                guidance_scale=self.config.guidance_scale,
                image_guidance_scale=self.config.image_guidance_scale,
                diffusion_steps=20,
                prompt=self.config.prompt,
                noisy_latent_type="noisy_latent",
                T=1000 if update_batch_index == 0 else 800,
            ).to(key_frame_render_image) # (1, C, H, W)
            
            torchvision.utils.save_image(key_frame_edit_image, f'key_frame_edit_image_{tag}.png', nrow=key_frame_edit_image.shape[0], padding=0)
            
            if key_frame_edit_image.size()[-2:] != self.datamanager.image_batch["image"].size()[1:-1]:
                key_frame_edit_image = F.interpolate(key_frame_edit_image, size=self.datamanager.image_batch["image"].size()[1:-1], mode="bilinear", align_corners=False)
                key_frame_condition_image = F.interpolate(key_frame_condition_image, size=self.datamanager.image_batch["image"].size()[1:-1], mode="bilinear", align_corners=False)
            
            with self.data_lock:
                self.datamanager.image_batch["image"][key_frame_spot] = key_frame_edit_image[0].squeeze().permute(1, 2, 0)
            
            while (self.current_step - self.start_step) < self.buffer_value * (self.end_step - self.start_step):
                self.frame_batch_index = 0
                
                while self.frame_batch_index * (self.sequence_length - self.overlap_length) < self.train_dataset_length:
                    start_idx = self.frame_batch_index * (self.sequence_length - self.overlap_length)
                    end_idx = min(((self.frame_batch_index + 1) * (self.sequence_length - self.overlap_length) + self.overlap_length), self.train_dataset_length)
                    
                    current_indexs = sorted(list(range(start_idx, end_idx)))
                    render_list = []
                    original_list = []
                    current_spots = []
                    
                    with self.model_lock:
                        for i, current_index in enumerate(current_indexs):
                            camera_transforms = self.datamanager.train_camera_optimizer(torch.tensor(current_index).unsqueeze(dim=0))
                            current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
                            
                            current_spot = int(torch.where(self.datamanager.image_batch["image_idx"] == torch.tensor(current_index))[0][0])
                            current_spots.append(current_spot)
                            
                            original_img = self.datamanager.original_image_batch["image"][current_spot].to(self.device) # (H, W, 3)
                            original_list.append(original_img)
                            
                            current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)
                            camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
                            
                            render_img = camera_outputs["rgb"] # (H, W, 3)
                            render_list.append(render_img)
                            
                    # clear memory
                    del camera_transforms, current_camera, current_ray_bundle, camera_outputs
                    torch.cuda.empty_cache()
                    
                    render_images = torch.stack(render_list) # (N, H, W, 3)
                    original_images = torch.stack(original_list) # (N, H, W, 3)
                    
                    render_images = rearrange(render_images, 'N H W C -> N C H W') # (N, C, H, W)
                    original_images = rearrange(original_images, 'N H W C -> N C H W') # (N, C, H, W)
                    
                    torchvision.utils.save_image(render_images, f'images_render_{tag}.png', nrow=render_images.shape[0], padding=0)
                    torchvision.utils.save_image(original_images, f'images_condition_{tag}.png', nrow=original_images.shape[0], padding=0)
                    
                    update_images = self.datamanager.image_batch["image"][current_spots].to(self.device) 
                    update_images = rearrange(update_images, 'N H W C -> N C H W') # (N, C, H, W)
                    
                    # apply the optical flow warp
                    for i in range(1, len(update_images)):
                        ref_image = (update_images[[0]] * 255.0).float().to(self.ip2p_device) # (1, C, H, W)
                        cur_image = (update_images[[i]] * 255.0).float().to(self.ip2p_device) # (1, C, H, W)
                        
                        ref_image_cond = (original_images[[0]] * 255.0).float().to(self.ip2p_device) # (1, C, H, W)
                        cur_image_cond = (original_images[[i]] * 255.0).float().to(self.ip2p_device) # (1, C, H, W)
                        
                        padder = InputPadder(ref_image.shape) 
                        ref_image, cur_image, ref_image_cond, cur_image_cond = padder.pad(ref_image, cur_image, ref_image_cond, cur_image_cond)
                        
                        _, flow_fwd_ref = self.raft(ref_image_cond, cur_image_cond, iters=20, test_mode=True) 
                        _, flow_bwd_ref = self.raft(cur_image_cond, ref_image_cond, iters=20, test_mode=True)
                    
                        flow_fwd_ref = padder.unpad(flow_fwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
                        flow_bwd_ref = padder.unpad(flow_bwd_ref[0]).cpu().numpy().transpose(1, 2, 0) 
                        
                        ref_image = padder.unpad(ref_image[0]).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                        cur_image = padder.unpad(cur_image[0]).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)
                        
                        mask_bwd_ref = compute_bwd_mask(flow_fwd_ref, flow_bwd_ref) # (h, w)
                        warp_to_cur_image_ref = warp_flow(ref_image, flow_bwd_ref) # (h, w, c)
                        warp_to_cur_image = warp_to_cur_image_ref * mask_bwd_ref[..., None] + cur_image * (1 - mask_bwd_ref[..., None]) # (h, w, c)
                        
                        warp_to_cur_image = torch.from_numpy(warp_to_cur_image / 255.0)
                        warp_to_cur_image = rearrange(warp_to_cur_image, 'H W C -> 1 C H W').to(update_images)
                        
                        if warp_to_cur_image.size()[-2:] != update_images.size()[-2:]:
                            warp_to_cur_image = F.interpolate(warp_to_cur_image, size=update_images.size()[-2:], mode="bilinear", align_corners=False)
                        
                        update_images[[i]] = warp_to_cur_image.to(update_images)
                            
                    torchvision.utils.save_image(update_images, f'images_warped_{tag}.png', nrow=update_images.shape[0], padding=0)
                    
                    # use key frame image as the reference image
                    update_images[0] = key_frame_edit_image[0].to(update_images) # (C, H, W)
                    original_images[0] = key_frame_condition_image[0].to(original_images) # (C, H, W)
                    
                    if self.config.resize_512:
                        height, width = update_images.size()[-2:]
                        factor = 512 / max(height, width)
                        factor = math.ceil(min(height, width) * factor / 64) * 64 / min(height, width)
                        height = int((height * factor) // 64) * 64
                        width = int((width * factor) // 64) * 64
                        update_images = F.interpolate(update_images, size=(height, width), mode="bilinear", align_corners=False)
                        original_images = F.interpolate(original_images, size=(height, width), mode="bilinear", align_corners=False)
                    
                    edited_images = self.ip2p.edit_sequence(
                        images=update_images.to(self.ip2p_device), 
                        images_cond=original_images.to(self.ip2p_device),
                        guidance_scale=self.config.guidance_scale,
                        image_guidance_scale=self.config.image_guidance_scale,
                        diffusion_steps=5 if update_batch_index == 0 else 3,
                        prompt=self.config.prompt,
                        noisy_latent_type="noisy_latent",
                        T=700 if update_batch_index == 0 else 550,
                    ).to(update_images) # (N, C, H, W)
                    
                    torchvision.utils.save_image(edited_images, f'images_edited_{tag}.png', nrow=edited_images.shape[0], padding=0)
                    
                    if edited_images.size()[-2:] != self.datamanager.image_batch["image"].size()[1:-1]:
                        edited_images = F.interpolate(edited_images, size=self.datamanager.image_batch["image"].size()[1:-1], mode="bilinear", align_corners=False) # (N, 3, H, W)
                        
                    with self.data_lock:
                        for i, current_spot in enumerate(current_spots):
                            if i == 0 and end_idx < self.train_dataset_length:
                                continue
                            elif i == 0 and end_idx >= self.train_dataset_length:
                                current_spot = key_frame_spot
                            else:
                                current_spot = current_spot
                            
                            if update_batch_index == 0:
                                self.datamanager.image_batch["image"][current_spot] = edited_images[i].squeeze().permute(1, 2, 0) # (H, W, 3)
                            
                    # batch index update
                    self.frame_batch_index += 1
                    
                update_batch_index += 1
                
    
    def init_start_step(self, step):
        self.start_step = step
        self.current_step = step
                    
    def init_end_step(self, step):
        self.end_step = step
        
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        
        self.current_step = step # update current step

        with self.data_lock:
            # if step % 10 == 0:
            #     largest_index = max(self.frame_batch_index * (self.sequence_length - self.overlap_length), self.train_dataset_length-1)
            #     current_index = random.randint(0, largest_index)
            #     current_spot = int(torch.where(self.datamanager.image_batch["image_idx"] == torch.tensor(current_index))[0][0])
            #     images = self.datamanager.image_batch["image"][current_spot].to(self.device)
            #     cameras = self.datamanager.train_dataparser_outputs.cameras[current_index].to(
            #         self.device)
            #     scaling_factor = 512 / max(images.shape[:2])
            #     cameras.rescale_output_resolution(scaling_factor=scaling_factor)
            #     ray_bundle = cameras.generate_rays(
            #         torch.tensor(list(range(1))).unsqueeze(-1)
            #     ).flatten()
                
            #     height, width = images.size()[:2] # (H, W)
            #     images = F.interpolate(images.unsqueeze(0).permute(0, 3, 1, 2), size=(int(height*scaling_factor), int(width*scaling_factor)), mode="bilinear", align_corners=False).squeeze(0).permute(1, 2, 0) # (H, W, 3)
                
            #     batch = {
            #         "image": images.reshape(-1, 3)
            #     }
            # else:
            ray_bundle, batch = self.datamanager.next_train(step)
            
        with self.model_lock:
            model_outputs = self.model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        
        return model_outputs, loss_dict, metrics_dict
    

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        is_ddp_model_state = True
        model_state = {}
        for key, value in state_dict.items():
            if key.startswith("_model."):
                # remove the "_model." prefix from key
                model_state[key[len("_model.") :]] = value
                # make sure that the "module." prefix comes from DDP,
                # rather than an attribute of the model named "module"
                if not key.startswith("_model.module."):
                    is_ddp_model_state = False
        # remove "module." prefix added by DDP
        if is_ddp_model_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {key: value for key, value in state_dict.items() if not key.startswith("_model.")}
        self.model.load_state_dict(model_state, strict=False)
        super().load_state_dict(pipeline_state, strict=False)
        
        
def warp_flow(img, flow): 
    # warp image according to flow
    h, w = flow.shape[:2]
    flow_new = flow.copy() 
    flow_new[:, :, 0] += np.arange(w) 
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis] 

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res

def compute_bwd_mask(fwd_flow, bwd_flow):
    # compute the backward mask
    alpha_1 = 0.5 
    alpha_2 = 0.5

    fwd2bwd_flow = warp_flow(fwd_flow, bwd_flow)
    bwd_lr_error = np.linalg.norm(bwd_flow + fwd2bwd_flow, axis=-1)

    bwd_mask = (
        bwd_lr_error
        < alpha_1
        * (np.linalg.norm(bwd_flow, axis=-1) + np.linalg.norm(fwd2bwd_flow, axis=-1))
        + alpha_2
    )

    return bwd_mask
    
