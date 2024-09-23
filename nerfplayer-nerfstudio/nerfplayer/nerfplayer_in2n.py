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

from nerfplayer.nerfplayer_datamanager import NerfplayerDataManagerConfig
from nerfplayer.ip2p import InstructPix2Pix

import warnings; warnings.filterwarnings("ignore")

@dataclass
class IN2NPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: IN2NPipeline)
    """specifies the datamanager config"""
    prompt: str = "don't change the image"
    """prompt for InstructPix2Pix"""
    guidance_scale: float = 7.5
    """(text) guidance scale for InstructPix2Pix"""
    image_guidance_scale: float = 1.5
    """image guidance scale for InstructPix2Pix"""
    diffusion_steps: int = 20
    """Number of diffusion steps to take for InstructPix2Pix"""
    lower_bound: float = 0.02
    """Lower bound for diffusion timesteps to use for image editing"""
    upper_bound: float = 0.98
    """Upper bound for diffusion timesteps to use for image editing"""
    ip2p_device: Optional[str] = None
    """Second device to place InstructPix2Pix on. If None, will use the same device as the pipeline"""
    ip2p_use_full_precision: bool = True
    """Whether to use full precision for InstructPix2Pix"""
    
class IN2NPipeline(VanillaPipeline):
    """InstructNeRF2NeRF pipeline"""

    config: IN2NPipelineConfig

    def __init__(
        self,
        config: IN2NPipelineConfig,
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

        self.ip2p = InstructPix2Pix(self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision)
        
        # load base text embedding using classifier free guidance
        self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt, device=self.ip2p_device, num_images_per_prompt=1, do_classifier_free_guidance=True, negative_prompt=""
        )
        
        self.train_indices_order = cycle(range(len(self.datamanager.train_dataparser_outputs.image_filenames)))
        
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
            
    def test_edit(self):
        while (self.current_step - self.start_step) < 0.99 * (self.end_step - self.start_step):  
            
            current_spot = next(self.train_indices_order)
            # get original image from dataset
            original_image = self.datamanager.original_image_batch["image"][current_spot].to(self.device)
            # generate current index in datamanger
            current_index = self.datamanager.image_batch["image_idx"][current_spot]

            # get current camera, include camera transforms from original optimizer
            camera_transforms = self.datamanager.train_camera_optimizer(current_index.unsqueeze(dim=0))
            current_camera = self.datamanager.train_dataparser_outputs.cameras[current_index].to(self.device)
            current_ray_bundle = current_camera.generate_rays(torch.tensor(list(range(1))).unsqueeze(-1), camera_opt_to_camera=camera_transforms)

            # get current render of nerf
            original_image = original_image.unsqueeze(dim=0).permute(0, 3, 1, 2) # (N, C, H, W)
            with self.model_lock:
                camera_outputs = self.model.get_outputs_for_camera_ray_bundle(current_ray_bundle)
            rendered_image = camera_outputs["rgb"].unsqueeze(dim=0).permute(0, 3, 1, 2) # (N, C, H, W)
            
            # delete to free up memory
            del camera_outputs
            del current_camera
            del current_ray_bundle
            del camera_transforms
            torch.cuda.empty_cache()
            
            height, width = rendered_image.size()[-2:]
            factor = 512 / max(height, width)
            factor = math.ceil(min(height, width) * factor / 64) * 64 / min(height, width)
            height = int((height * factor) // 64) * 64
            width = int((width * factor) // 64) * 64
            
            image_input = torch.nn.functional.interpolate(rendered_image, size=(height, width), mode="bilinear", align_corners=False) # (N, C, RH, RW)
            image_condition = torch.nn.functional.interpolate(original_image, size=(height, width), mode="bilinear", align_corners=False) # (N, C, RH, RW)
             
            edited_image = self.ip2p.edit_image(
                    self.text_embedding.to(self.ip2p_device, dtype=torch.float32),
                    image_input.to(self.ip2p_device),
                    image_condition.to(self.ip2p_device),
                    guidance_scale=self.config.guidance_scale,
                    image_guidance_scale=self.config.image_guidance_scale,
                    diffusion_steps=self.config.diffusion_steps,
                    lower_bound=self.config.lower_bound,
                    upper_bound=self.config.upper_bound,
                )

            # resize to original image size (often not necessary)
            if (edited_image.size() != rendered_image.size()):
                edited_image = torch.nn.functional.interpolate(edited_image, size=rendered_image.size()[2:], mode='bilinear')
                
            torchvision.utils.save_image(edited_image, f"edited_images.png")

            # write edited image to dataloader
            with self.data_lock:
                self.datamanager.image_batch["image"][current_spot] = edited_image.squeeze().permute(1,2,0)
                
    
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
        
        