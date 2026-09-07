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

"""Instruct-NeRF2NeRF baseline, for comparison against the 4D method.

The baseline edits one frame at a time with stock InstructPix2Pix and writes it
straight back, cycling through the dataset until training is nearly done.  There
is nothing tying consecutive edits together, which is exactly the failure this
paper addresses: the appearance drifts from frame to frame and the field
averages the disagreement into blur.
"""

from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from instruct4d.ip2p import InstructPix2Pix

from nerfplayer.editing_pipeline import EditingPipeline, EditingPipelineConfig

#: Editing stops this far into the run, leaving the tail purely for the field to
#: converge on the images it has been given.
EDIT_UNTIL_PROGRESS = 0.99


@dataclass
class IN2NPipelineConfig(EditingPipelineConfig):
    """Configuration for :class:`IN2NPipeline`."""

    _target: Type = field(default_factory=lambda: IN2NPipeline)
    """Class this config instantiates."""
    lower_bound: float = 0.02
    """Lowest fraction of the noise schedule to sample an edit timestep from."""
    upper_bound: float = 0.98
    """Highest fraction of the noise schedule to sample an edit timestep from."""
    ip2p_use_full_precision: bool = True
    """Run the diffusion model in fp32; the baseline is more sensitive to fp16."""


class IN2NPipeline(EditingPipeline):
    """Edits one training view at a time, with no cross-frame consistency."""

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
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

        self.ip2p = InstructPix2Pix(
            self.ip2p_device, ip2p_use_full_precision=self.config.ip2p_use_full_precision
        )
        # The prompt is fixed for the run, so its embedding is computed once.
        self.text_embedding = self.ip2p.pipe._encode_prompt(
            self.config.prompt,
            device=self.ip2p_device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
            negative_prompt="",
        )
        self.train_indices_order = cycle(
            range(len(self.datamanager.train_dataparser_outputs.image_filenames))
        )

    def test_edit(self) -> None:
        """Cycle through training views, editing each one in place."""
        while self.training_progress < EDIT_UNTIL_PROGRESS:
            slot = next(self.train_indices_order)
            index = self.datamanager.image_batch["image_idx"][slot]

            original = self.datamanager.original_image_batch["image"][slot].to(self.device)
            original = original.unsqueeze(0).permute(0, 3, 1, 2)
            rendered = self.render_view(int(index)).unsqueeze(0).permute(0, 3, 1, 2)

            target = self.diffusion_resolution(*rendered.shape[-2:])
            edited = self.ip2p.edit_image(
                self.text_embedding.to(self.ip2p_device, dtype=torch.float32),
                self.resize(rendered, target).to(self.ip2p_device),
                self.resize(original, target).to(self.ip2p_device),
                guidance_scale=self.config.guidance_scale,
                image_guidance_scale=self.config.image_guidance_scale,
                diffusion_steps=self.config.diffusion_steps,
                lower_bound=self.config.lower_bound,
                upper_bound=self.config.upper_bound,
            )
            edited = self.resize(edited, rendered.shape[-2:])

            with self.data_lock:
                self.datamanager.image_batch["image"][slot] = edited.squeeze(0).permute(1, 2, 0)
