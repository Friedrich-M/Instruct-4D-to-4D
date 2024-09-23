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

"""
Instruct-NeRF2NeRF Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import torch
from typing import Dict, Tuple, Type

from rich.progress import Console

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.model_components.ray_generators import RayGenerator
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.data.datasets.depth_dataset import DepthDataset


CONSOLE = Console(width=120)

@dataclass
class NerfplayerDataManagerConfig(VanillaDataManagerConfig):
    """Configuration for the KplanesDataManager."""

    _target: Type = field(default_factory=lambda: NerfplayerDataManager[DepthDataset])

class NerfplayerDataManager(VanillaDataManager[DepthDataset]):
    """Data manager for Nerfplayer."""

    config: NerfplayerDataManagerConfig
    
    def setup_train(self):
        """Sets up the data loaders for training"""
        assert self.train_dataset is not None
        CONSOLE.print("Setting up training dataset...")
        self.train_image_dataloader = CacheDataloader(
            self.train_dataset,
            num_images_to_sample_from=self.config.train_num_images_to_sample_from,
            num_times_to_repeat_images=self.config.train_num_times_to_repeat_images,
            device=self.device,
            num_workers=self.world_size * 4,
            pin_memory=True,
            collate_fn=self.config.collate_fn,
        )
        self.iter_train_image_dataloader = iter(self.train_image_dataloader)
        self.train_pixel_sampler = self._get_pixel_sampler(self.train_dataset, self.config.train_num_rays_per_batch)
        self.train_camera_optimizer = self.config.camera_optimizer.setup(
            num_cameras=self.train_dataset.cameras.size, device=self.device
        )
        self.train_ray_generator = RayGenerator(
            self.train_dataset.cameras.to(self.device),
            self.train_camera_optimizer,
        )

        # pre-fetch the image batch (how images are replaced in dataset)
        self.image_batch = next(self.iter_train_image_dataloader)

        # keep a copy of the original image batch
        self.original_image_batch = {}
        self.original_image_batch['image_idx'] = self.image_batch['image_idx'].clone()
        self.original_image_batch['image'] = self.image_batch['image'].clone()
        self.original_image_batch['depth_image'] = self.image_batch['depth_image'].clone()

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Returns the next batch of data from the train dataloader."""
        self.train_count += 1
        assert self.train_pixel_sampler is not None
        batch = self.train_pixel_sampler.sample(self.image_batch)
        ray_indices = batch["indices"]
        ray_bundle = self.train_ray_generator(ray_indices)
        
        return ray_bundle, batch
