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

"""Instruct 4D-to-4D editing for a single-view (monocular) 4D scene.

The single-view setting has no other cameras to be consistent with, so the
spatial half of the method does not apply: what remains is the temporal half.
The key frame is edited once, and that edit is carried along the sequence by a
flow-guided sliding window, exactly as in the multi-view pipeline.

The whole edit runs on a background thread started by
:class:`~nerfplayer.nerfplayer_trainer.NerfplayerTrainer`, so the field keeps
absorbing edited frames as they are produced.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Type

import numpy as np
import torch
import torchvision
from einops import rearrange
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from instruct4d.flow import (
    blend_with_mask,
    consistency_mask,
    estimate_flow_pair,
    load_raft,
    warp_by_flow,
)
from instruct4d.ip2p import SequenceInstructPix2Pix

from nerfplayer.editing_pipeline import EditingPipeline, EditingPipelineConfig


@dataclass
class NerfplayerPipelineConfig(EditingPipelineConfig):
    """Configuration for :class:`NerfplayerPipeline`."""

    _target: Type = field(default_factory=lambda: NerfplayerPipeline)
    """Class this config instantiates."""
    refine_diffusion_steps: int = 5
    """Denoising steps when repainting a flow-warped window."""
    refine_num_steps: int = 700
    """Noise level when repainting; lower preserves more of the warp."""
    sequence_length: int = 5
    """Frames edited together in one anchor-aware batch. Reduce this first if
    you run out of GPU memory."""
    overlap_length: int = 1
    """Frames shared between consecutive windows. The overlapping frame is the
    anchor slot, so this is normally 1."""
    resize_512: bool = False
    """Edit at roughly 512px on the long side instead of the native resolution."""
    raft_ckpt: str = "weights/raft-things.pth"
    """RAFT optical-flow checkpoint."""
    save_debug_images: bool = False
    """Write the intermediate renders, warps and edits to the working directory."""


class NerfplayerPipeline(EditingPipeline):
    """Edits a monocular 4D scene with a flow-guided sliding window."""

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
        super().__init__(config, device, test_mode, world_size, local_rank, grad_scaler)

        self.ip2p = SequenceInstructPix2Pix(
            device=self.ip2p_device,
            use_full_precision=self.config.ip2p_use_full_precision,
            # Consecutive frames of a monocular capture differ more than
            # neighbouring views of one timestamp, so each frame keeps its own
            # content alongside the anchor's.
            self_attention="anchor_self",
        )
        self.raft = load_raft(self.config.raft_ckpt, self.ip2p_device)
        self.num_frames = len(self.datamanager.train_dataparser_outputs.image_filenames)

    # ------------------------------------------------------------------
    def test_edit(self, key_frame: int = 0) -> None:
        """Edit the key frame, then propagate it along the sequence.

        Runs a single pass: the key frame is edited once and every window is
        written back once.  Editing is far slower than an optimisation step, so
        the field continues training against the frames already produced.
        """
        anchor, anchor_cond = self._edit_key_frame(key_frame)

        stride = self.config.sequence_length - self.config.overlap_length
        for start in range(0, self.num_frames, stride):
            end = min(start + self.config.sequence_length, self.num_frames)
            if end <= start:
                break
            self._edit_window(
                list(range(start, end)), anchor, anchor_cond, key_frame,
                is_last=end >= self.num_frames,
            )

        print("editing complete; training continues on the edited frames")

    # ------------------------------------------------------------------
    def _edit_key_frame(self, key_frame: int):
        """Edit the key frame and write it back; return it and its condition.

        The result becomes the anchor for every window, which is what stops the
        appearance drifting as the edit travels along the sequence.
        """
        slot = self.batch_slot(key_frame)
        render = rearrange(self.render_view(key_frame), "h w c -> 1 c h w")
        condition = rearrange(
            self.datamanager.original_image_batch["image"][slot], "h w c -> 1 c h w"
        )

        target = self._edit_resolution(render)
        render = self.resize(render, target)
        condition = self.resize(condition, target)
        self._save_debug("key_frame_render", render)

        edited = self.ip2p.edit_sequence(
            images=render.to(self.ip2p_device),
            images_cond=condition.to(self.ip2p_device),
            prompt=self.config.prompt,
            guidance_scale=self.config.guidance_scale,
            image_guidance_scale=self.config.image_guidance_scale,
            diffusion_steps=self.config.diffusion_steps,
            noisy_latent_type="noisy_latent",
        ).to(render)
        self._save_debug("key_frame_edited", edited)

        native = self.datamanager.image_batch["image"].shape[1:-1]
        with self.data_lock:
            self.datamanager.image_batch["image"][slot] = (
                self.resize(edited, native)[0].permute(1, 2, 0)
            )
        return edited, condition

    def _edit_window(
        self,
        indices: List[int],
        anchor: torch.Tensor,
        anchor_cond: torch.Tensor,
        key_frame: int,
        is_last: bool,
    ) -> None:
        """Warp, repaint and write back one window of frames.

        Windows overlap by ``overlap_length``, and slot 0 of each batch is given
        over to the anchor rather than to the window's own first frame. That
        frame is therefore covered by the *previous* window, and slot 0's output
        belongs to the key frame instead.
        """
        slots = [self.batch_slot(i) for i in indices]

        originals = [
            self.datamanager.original_image_batch["image"][slot].to(self.device) for slot in slots
        ]
        original_images = rearrange(torch.stack(originals), "n h w c -> n c h w")
        current = rearrange(
            self.datamanager.image_batch["image"][slots].to(self.device), "n h w c -> n c h w"
        )

        if self.debug_enabled:
            # The window is built from the stored training images, not from
            # fresh renders, so this costs a full render per frame and is only
            # worth paying to look at.
            renders = torch.stack([self.render_view(index) for index in indices])
            self._save_debug("window_render", rearrange(renders, "n h w c -> n c h w"))

        current = self._warp_window(current, original_images)
        self._save_debug("window_warped", current)

        # Slot 0 is the anchor slot: replacing it with the edited key frame is
        # what ties this window to the same appearance as every other one.
        target = self._edit_resolution(current)
        current = self.resize(current, target)
        original_images = self.resize(original_images, target)
        current[0] = self.resize(anchor, target)[0].to(current)
        original_images[0] = self.resize(anchor_cond, target)[0].to(original_images)

        edited = self.ip2p.edit_sequence(
            images=current.to(self.ip2p_device),
            images_cond=original_images.to(self.ip2p_device),
            prompt=self.config.prompt,
            guidance_scale=self.config.guidance_scale,
            image_guidance_scale=self.config.image_guidance_scale,
            diffusion_steps=self.config.refine_diffusion_steps,
            noisy_latent_type="noisy_latent",
            T=self.config.refine_num_steps,
        ).to(current)
        self._save_debug("window_edited", edited)

        native = self.datamanager.image_batch["image"].shape[1:-1]
        edited = self.resize(edited, native)
        with self.data_lock:
            for position, slot in enumerate(slots):
                if position == 0:
                    if not is_last:
                        # Slot 0 carried the anchor, and this window's own first
                        # frame was already written by the previous window.
                        continue
                    # On the final window nothing comes after, so slot 0's
                    # output is folded back into the key frame it came from.
                    slot = self.batch_slot(key_frame)
                self.datamanager.image_batch["image"][slot] = edited[position].permute(1, 2, 0)

    def _warp_window(self, current: torch.Tensor, conditions: torch.Tensor) -> torch.Tensor:
        """Warp the window's first frame onto the rest with optical flow.

        Flow is estimated on the *unedited* frames: the edit changes colours
        everywhere, which would confound a photometric flow estimator, while the
        underlying motion is the same either way.
        """
        warped = current.clone()
        reference = (current[:1] * 255.0).float().to(self.ip2p_device)
        reference_cond = (conditions[:1] * 255.0).float().to(self.ip2p_device)

        for i in range(1, len(current)):
            frame = (current[i : i + 1] * 255.0).float().to(self.ip2p_device)
            frame_cond = (conditions[i : i + 1] * 255.0).float().to(self.ip2p_device)

            forward, backward = estimate_flow_pair(self.raft, reference_cond, frame_cond)
            reliable = consistency_mask(backward, forward)

            ref_np = reference[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            cur_np = frame[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            blended = blend_with_mask(warp_by_flow(ref_np, backward), cur_np, reliable)

            result = rearrange(torch.from_numpy(blended / 255.0), "h w c -> 1 c h w").to(current)
            warped[i] = self.resize(result, current.shape[-2:])[0]
        return warped

    def _edit_resolution(self, images: torch.Tensor):
        """Resolution to run the diffusion model at for this batch."""
        if not self.config.resize_512:
            return images.shape[-2:]
        return self.diffusion_resolution(*images.shape[-2:])

    @property
    def debug_enabled(self) -> bool:
        return self.config.save_debug_images

    def _save_debug(self, name: str, images: torch.Tensor) -> None:
        if not self.debug_enabled:
            return
        tag = self.config.prompt.strip().split(" ")[-1].strip("?!.,")
        torchvision.utils.save_image(
            images.float(), f"{name}_{tag}.png", nrow=images.shape[0], padding=0
        )
