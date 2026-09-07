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

"""Shared machinery for the two single-view editing pipelines.

Both :class:`~nerfplayer.nerfplayer_pipeline.NerfplayerPipeline` (ours) and
:class:`~nerfplayer.nerfplayer_in2n.IN2NPipeline` (the Instruct-NeRF2NeRF
baseline) follow the same outer structure: a background thread rewrites the
training images while the main thread keeps optimising the field against them.
Everything that structure needs -- the two locks, the progress bookkeeping the
editing thread reads, the viewer controls and the nerfstudio plumbing -- lives
here, so each pipeline only has to implement its own ``test_edit``.
"""

import math
import threading
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText

from nerfplayer.nerfplayer_datamanager import NerfplayerDataManagerConfig

#: Both the VAE and the UNet's downsampling stack need a side length divisible
#: by this.
SIZE_MULTIPLE = 64


@dataclass
class EditingPipelineConfig(VanillaPipelineConfig):
    """Options common to both editing pipelines.

    Not instantiated directly: the two concrete configs override ``_target``.
    """

    _target: Type = field(default_factory=lambda: EditingPipeline)
    """Class this config instantiates."""
    datamanager: NerfplayerDataManagerConfig = NerfplayerDataManagerConfig()
    """Data manager, which keeps the edited and original images side by side."""
    prompt: str = "don't change the image"
    """The editing instruction."""
    guidance_scale: float = 7.5
    """Text classifier-free guidance weight."""
    image_guidance_scale: float = 1.5
    """Image classifier-free guidance weight; higher stays closer to the input."""
    diffusion_steps: int = 20
    """Denoising steps for a full edit."""
    ip2p_device: Optional[str] = None
    """Device for the diffusion model. Defaults to the pipeline's own device,
    but editing and optimisation run concurrently, so a second GPU is faster."""
    ip2p_use_full_precision: bool = False
    """Run the diffusion model in fp32 instead of fp16."""


class EditingPipeline(VanillaPipeline):
    """A pipeline whose training images are rewritten as training proceeds.

    Subclasses implement :meth:`test_edit`, which the trainer runs on a
    background thread.  It may read :attr:`current_step`, :attr:`start_step` and
    :attr:`end_step` to pace itself against training, and must take
    :attr:`data_lock` before writing into ``datamanager.image_batch``.
    """

    config: EditingPipelineConfig

    def __init__(
        self,
        config: EditingPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.ip2p_device = (
            torch.device(device)
            if self.config.ip2p_device is None
            else torch.device(self.config.ip2p_device)
        )

        # The editing thread writes training images while the main thread reads
        # them, and renders from the field while the main thread optimises it.
        self.data_lock = threading.Lock()
        self.model_lock = threading.Lock()

        # Training progress, so the editing thread can tell how far along it is.
        self.current_step = 0
        self.start_step = 0
        self.end_step = 0

        self.prompt_box = ViewerText(
            name="Prompt", default_value=self.config.prompt, cb_hook=self.prompt_callback
        )
        self.guidance_scale_box = ViewerNumber(
            name="Text Guidance Scale",
            default_value=self.config.guidance_scale,
            cb_hook=self.guidance_scale_callback,
        )
        self.image_guidance_scale_box = ViewerNumber(
            name="Image Guidance Scale",
            default_value=self.config.image_guidance_scale,
            cb_hook=self.image_guidance_scale_callback,
        )

    # ------------------------------------------------------------------
    # Viewer controls
    # ------------------------------------------------------------------
    def guidance_scale_callback(self, handle: ViewerNumber) -> None:
        """Apply a text-guidance change made in the viewer."""
        self.config.guidance_scale = handle.value

    def image_guidance_scale_callback(self, handle: ViewerNumber) -> None:
        """Apply an image-guidance change made in the viewer."""
        self.config.image_guidance_scale = handle.value

    def prompt_callback(self, handle: ViewerText) -> None:
        """Apply a prompt change made in the viewer."""
        self.config.prompt = handle.value

    # ------------------------------------------------------------------
    # Progress, read by the editing thread
    # ------------------------------------------------------------------
    def init_start_step(self, step: int) -> None:
        self.start_step = step
        self.current_step = step

    def init_end_step(self, step: int) -> None:
        self.end_step = step

    @property
    def training_progress(self) -> float:
        """Fraction of the run completed, in ``[0, 1]``."""
        span = self.end_step - self.start_step
        return (self.current_step - self.start_step) / span if span > 0 else 1.0

    # ------------------------------------------------------------------
    # Editing hook
    # ------------------------------------------------------------------
    def test_edit(self) -> None:
        """Rewrite the training images. Runs on a background thread."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def diffusion_resolution(height: int, width: int, long_side: int = 512) -> Tuple[int, int]:
        """Nearest resolution the diffusion model accepts, near ``long_side``.

        Rounds the short side to a multiple of :data:`SIZE_MULTIPLE` first, so
        the aspect ratio moves as little as possible.
        """
        scale = long_side / max(height, width)
        scale = (
            math.ceil(min(height, width) * scale / SIZE_MULTIPLE)
            * SIZE_MULTIPLE
            / min(height, width)
        )
        return int(height * scale) // SIZE_MULTIPLE * SIZE_MULTIPLE, int(
            width * scale
        ) // SIZE_MULTIPLE * SIZE_MULTIPLE

    @staticmethod
    def resize(images: torch.Tensor, size: Tuple[int, int]) -> torch.Tensor:
        """Bilinearly resize ``(N, C, H, W)`` images, skipping a no-op resize."""
        if tuple(images.shape[-2:]) == tuple(size):
            return images
        return F.interpolate(images, size=size, mode="bilinear", align_corners=False)

    def render_view(self, index: int) -> torch.Tensor:
        """Render training view ``index`` from the current field, as ``(H, W, 3)``.

        Takes :attr:`model_lock`, so this is safe to call from the editing
        thread while the main thread is optimising.
        """
        transforms = self.datamanager.train_camera_optimizer(torch.tensor(index).unsqueeze(dim=0))
        camera = self.datamanager.train_dataparser_outputs.cameras[index].to(self.device)
        ray_bundle = camera.generate_rays(
            torch.tensor([0]).unsqueeze(-1), camera_opt_to_camera=transforms
        )
        with self.model_lock:
            outputs = self.model.get_outputs_for_camera_ray_bundle(ray_bundle)
        rgb = outputs["rgb"]
        del outputs, camera, ray_bundle, transforms
        torch.cuda.empty_cache()
        return rgb

    def batch_slot(self, index: int) -> int:
        """Position of training view ``index`` inside the data manager's batch."""
        matches = torch.where(
            self.datamanager.image_batch["image_idx"] == torch.tensor(index)
        )[0]
        return int(matches[0])

    # ------------------------------------------------------------------
    # nerfstudio plumbing
    # ------------------------------------------------------------------
    def get_train_loss_dict(self, step: int):
        """One training iteration, guarded against the editing thread's writes."""
        self.current_step = step

        with self.data_lock:
            ray_bundle, batch = self.datamanager.next_train(step)

        with self.model_lock:
            model_outputs = self.model(ray_bundle)
            metrics_dict = self.model.get_metrics_dict(model_outputs, batch)

        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        raise NotImplementedError("The pipeline is driven by the trainer, not called directly.")

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        """Load a checkpoint, tolerating the ``module.`` prefix that DDP adds."""
        model_state, is_ddp_state = {}, True
        for key, value in state_dict.items():
            if key.startswith("_model."):
                model_state[key[len("_model.") :]] = value
                if not key.startswith("_model.module."):
                    is_ddp_state = False
        if is_ddp_state:
            model_state = {key[len("module.") :]: value for key, value in model_state.items()}

        pipeline_state = {k: v for k, v in state_dict.items() if not k.startswith("_model.")}
        self.model.load_state_dict(model_state, strict=False)
        super().load_state_dict(pipeline_state, strict=False)
