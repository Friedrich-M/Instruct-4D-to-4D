"""Temporal propagation with a flow-guided sliding window (Section 3.3).

Once the key frame is edited, the edit has to travel along time without the
appearance drifting.  Editing each frame independently would drift immediately;
editing the whole video at once does not fit in memory.

The sliding window is the compromise.  Frames are processed in windows, and each
window is initialised by warping the last already-edited frame forward with RAFT
optical flow.  The warp carries the appearance across; wherever the flow is
unreliable -- occlusions, disocclusions, anything entering the shot -- the
forward-backward consistency check rejects it and the original frame shows
through.  Anchor-aware InstructPix2Pix then repaints the window, with the edited
key frame prepended as the anchor so every window in the sequence is pulled back
toward the same appearance rather than toward its own predecessor.

Because only a short, low-noise diffusion pass is needed to repair the warp, the
per-frame cost is a fraction of a full edit.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange

from ..flow import blend_with_mask, consistency_mask, estimate_flow_pair, warp_by_flow
from ..ip2p import SequenceInstructPix2Pix
from .buffer import FrameBuffer
from .debug import DebugImageWriter


class TemporalPropagator:
    """Carries the key frame's edit across time, one camera at a time."""

    def __init__(
        self,
        ip2p: SequenceInstructPix2Pix,
        raft,
        frames: FrameBuffer,
        originals: torch.Tensor,
        args,
        debug: Optional[DebugImageWriter] = None,
    ):
        """
        Args:
            ip2p: The anchor-aware editor.
            raft: A RAFT model from :func:`instruct4d.flow.load_raft`, on the
                same device as ``ip2p``.
            frames: The training colours, edited in place.
            originals: ``(frame, camera, H, W, 3)`` unedited images.
            args: Parsed configuration.
            debug: Optional writer for intermediate visualisations.
        """
        self.ip2p = ip2p
        self.raft = raft
        self.frames = frames
        self.originals = originals
        self.args = args
        self.device = args.ip2p_device
        self.debug = debug or DebugImageWriter(None)

    def run(self, key_frame: int = 0) -> None:
        """Propagate the key frame's edit to every other frame, in place."""
        for camera in range(self.frames.num_cameras):
            self._propagate_camera(camera, key_frame)

    def _propagate_camera(self, camera: int, key_frame: int) -> None:
        """Slide a window along time for one camera."""
        # The conditioning image is the untouched original, so it never changes.
        anchor_cond = self.originals[key_frame, camera].to(self.device).permute(2, 0, 1)[None]

        window = self.args.sequence_length
        for start in range(0, self.frames.num_frames, window):
            end = min(start + window, self.frames.num_frames)
            frame_ids = list(range(start, end))

            # Re-read the anchor every window rather than hoisting it. The first
            # window covers the key frame and writes its own result back over it,
            # and later windows must anchor to that updated value.
            anchor = self.frames.get(key_frame, camera, self.device)

            # Advanced indexing copies, so these are scratch buffers rather than
            # views into the shared training colours.
            images = self.frames.images[frame_ids, camera].clone()
            images_cond = self.originals[frame_ids, camera]

            images = self._warp_window_forward(images, images_cond, camera, start, len(frame_ids))

            images = rearrange(images, "f h w c -> f c h w").to(self.device)
            images_cond = rearrange(images_cond, "f h w c -> f c h w").to(self.device)
            self.debug.save(start, f"cam{camera:02d}_1_flow_warped", images)

            refined = self._refine_window(images, images_cond, anchor, anchor_cond)
            self.debug.save(start, f"cam{camera:02d}_2_refined", refined)

            self.frames.set_many(frame_ids, camera, refined)

    def _warp_window_forward(
        self,
        images: torch.Tensor,
        images_cond: torch.Tensor,
        camera: int,
        start: int,
        length: int,
    ) -> torch.Tensor:
        """Warp the previous window's last edited frame into every slot.

        Args:
            images: ``(f, H, W, 3)`` current colours for this window.
            images_cond: ``(f, H, W, 3)`` matching unedited images.
            camera: Camera being processed.
            start: Index of the window's first frame.
            length: Number of frames in the window.

        Returns:
            ``images`` with every slot replaced by the warped reference where the
            flow is trustworthy.
        """
        # The frame immediately before the window; for the first window that is
        # the key frame itself, which is already edited.
        reference_id = max(start - 1, 0)
        # The reference is fixed for the whole window, so it is converted once
        # rather than once per frame.
        ref_rgb = self._to_raft_input(self.frames.images[reference_id, camera])
        ref_cond = self._to_raft_input(self.originals[reference_id, camera])
        ref_np = ref_rgb[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        for i in range(length):
            if start == 0 and i == 0:
                # The key frame is the source of the edit; leave it alone.
                continue

            cur_rgb = self._to_raft_input(images[i])
            cur_cond = self._to_raft_input(images_cond[i])

            # Flow is estimated on the *unedited* pair. The edit changes colours
            # everywhere, which would confound a photometric flow estimator; the
            # underlying motion is the same either way.
            forward, backward = estimate_flow_pair(self.raft, ref_cond, cur_cond)
            reliable = consistency_mask(backward, forward)

            cur_np = cur_rgb[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            warped = blend_with_mask(warp_by_flow(ref_np, backward), cur_np, reliable)

            images[i] = self._from_raft_output(warped, images)
        return images

    def _to_raft_input(self, image: torch.Tensor) -> torch.Tensor:
        """``(H, W, 3)`` in ``[0, 1]`` to ``(1, 3, H, W)`` in ``[0, 255]``."""
        return (image.permute(2, 0, 1)[None] * 255.0).float().to(self.device)

    def _from_raft_output(self, warped: np.ndarray, like: torch.Tensor) -> torch.Tensor:
        """``(H, W, 3)`` in ``[0, 255]`` back to the buffer's layout and size."""
        image = torch.from_numpy(warped / 255.0).to(like)
        if image.shape[:2] != (self.frames.height, self.frames.width):
            image = rearrange(image, "h w c -> 1 c h w")
            image = F.interpolate(
                image, size=(self.frames.height, self.frames.width), mode="bilinear", align_corners=False
            )
            image = rearrange(image, "1 c h w -> h w c")
        return image

    def _refine_window(
        self,
        images: torch.Tensor,
        images_cond: torch.Tensor,
        anchor: torch.Tensor,
        anchor_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Repaint a warped window, anchored on the edited key frame."""
        batch = torch.cat([anchor, images], dim=0)
        batch_cond = torch.cat([anchor_cond, images_cond], dim=0)

        edited = self.ip2p.edit_sequence(
            images=batch,
            images_cond=batch_cond,
            prompt=self.args.prompt,
            guidance_scale=self.args.guidance_scale,
            image_guidance_scale=self.args.image_guidance_scale,
            diffusion_steps=self.args.refine_diffusion_steps,
            noisy_latent_type="noisy_latent",
            T=self.args.refine_num_steps,
        )
        # Drop the anchor's own output; it is only there to condition the rest.
        return edited[1:].cpu()
