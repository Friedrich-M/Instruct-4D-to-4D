"""Ray samplers for training the 4D field.

Rays are stored frame-major, so index ``frame * rays_per_frame + pixel`` selects
one ray.  Both samplers here draw a whole batch from a *single* frame, which the
field requires: its feature planes are gathered once per timestamp, so a batch
spanning several frames could not be rendered in one pass.
"""

import random
from typing import Dict

import numpy as np
import torch

#: A pixel counts as moving when its colour changes by more than this between
#: consecutive frames (8-bit levels out of 255).
MOTION_THRESHOLD = 10 / 255

#: Motion is unioned over this many frames either side, so a pixel stays flagged
#: for a while after something passes through it.
MOTION_WINDOW = 5

#: Fraction of each batch reserved for moving pixels.
MOTION_FRACTION = 0.1


class UniformSampler:
    """Draws a contiguous slice of a fixed random pixel permutation.

    The permutation is generated once and then re-used, with only the frame and
    the offset resampled.  That is much cheaper than reshuffling every step and,
    because the permutation is random to begin with, still gives each batch a
    spatially scattered set of rays.
    """

    def __init__(self, total_rays: int, total_frames: int, batch: int):
        """
        Args:
            total_rays: Total number of rays across all frames.
            total_frames: Number of frames the rays are divided into.
            batch: Rays per batch.
        """
        self.total_frames = total_frames
        self.batch = batch
        self.rays_per_frame = _rays_per_frame(total_rays, total_frames)
        self.permutation = torch.LongTensor(np.random.permutation(self.rays_per_frame))

    def nextids(self) -> torch.Tensor:
        """Return indices for the next batch."""
        frame = int(random.random() * self.total_frames)
        start = int(random.random() * (len(self.permutation) - self.batch))
        return self.permutation[start : start + self.batch] + frame * self.rays_per_frame


class MotionSampler:
    """Oversamples pixels where the scene is moving.

    Most of a DyNeRF scene is static, so uniform sampling spends nearly all its
    budget on background that converges quickly and starves the moving subject.
    This sampler reserves :data:`MOTION_FRACTION` of every batch for pixels
    flagged as moving, and fills the rest uniformly.
    """

    def __init__(self, all_rgbs: torch.Tensor, total_frames: int, batch: int):
        """
        Args:
            all_rgbs: ``(N, 3)`` colours for every ray, laid out frame-major.
            total_frames: Number of frames.
            batch: Rays per batch.
        """
        self.total_frames = total_frames
        self.batch = batch
        self.rays_per_frame = _rays_per_frame(all_rgbs.shape[0], total_frames)
        self.permutation = torch.LongTensor(np.random.permutation(self.rays_per_frame))
        self.motion_count = int(batch * MOTION_FRACTION)
        self.motion_indices = self._build_motion_index(all_rgbs)

    def _build_motion_index(self, all_rgbs: torch.Tensor) -> Dict[int, torch.Tensor]:
        """Flag, for each frame, the within-frame pixels that are moving.

        Comparing a frame against the one before it only catches motion at the
        instant it happens, so each frame's mask is unioned with its
        :data:`MOTION_WINDOW` neighbours on both sides.  That keeps a pixel in
        the budget for the whole time an object is passing through it.
        """
        # Rolling by one frame's worth of rays aligns each pixel with itself in
        # the previous frame.
        moving = (
            all_rgbs - torch.roll(all_rgbs, self.rays_per_frame, 0)
        ).abs().mean(-1) > MOTION_THRESHOLD

        def frame_mask(frame: int) -> torch.Tensor:
            return moving[frame * self.rays_per_frame : (frame + 1) * self.rays_per_frame]

        indices: Dict[int, torch.Tensor] = {}
        for frame in range(self.total_frames):
            mask = frame_mask(frame)
            for offset in range(1, MOTION_WINDOW + 1):
                if frame - offset >= 0:
                    mask = mask | frame_mask(frame - offset)
                if frame + offset < self.total_frames:
                    mask = mask | frame_mask(frame + offset)
            hits = mask.nonzero()
            indices[frame] = hits[:, 0] if len(hits) > 0 else torch.empty(0, dtype=torch.long)
        return indices

    def nextids(self) -> torch.Tensor:
        """Return indices for the next batch.

        The batch can come back slightly shorter than ``batch`` when the
        uniform slice runs into the end of the frame.
        """
        frame = int(random.random() * self.total_frames)
        start = int(random.random() * len(self.permutation))

        available = len(self.motion_indices[frame])
        if available == 0:
            # A completely static frame: fall back to a single uniform ray so
            # the batch shape stays predictable.
            motion = self.permutation[:1]
        elif available < self.motion_count:
            # Fewer moving pixels than the quota, so sample them with repeats.
            motion = self.motion_indices[frame][torch.randperm(self.motion_count) % available]
        else:
            motion = self.motion_indices[frame][torch.randperm(available)[: self.motion_count]]

        end = min(start + self.batch - len(motion), self.rays_per_frame)
        uniform = self.permutation[start:end]
        return torch.cat([motion, uniform], dim=0) + frame * self.rays_per_frame


def _rays_per_frame(total_rays: int, total_frames: int) -> int:
    """Rays per frame, requiring an exact split."""
    if total_rays % total_frames != 0:
        raise ValueError(
            f"{total_rays} rays do not divide evenly into {total_frames} frames; "
            "ray filtering must preserve the frame-major layout"
        )
    return total_rays // total_frames
