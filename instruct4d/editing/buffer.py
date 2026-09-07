"""A frame/camera view over the flat training colours.

The dataset stores every pixel of every (frame, camera) image in one flat
``(N, 3)`` tensor, which is what the ray sampler wants.  The editing pipeline
instead wants whole images, addressed by frame and camera.  Both need to refer
to the *same* memory: the editor overwrites training targets in place while the
optimiser is reading them from another thread.

:class:`FrameBuffer` provides that second addressing scheme over the same
storage, and keeps the layout conversions in one place instead of scattering
``rearrange`` calls through the editing code.  Writes go straight through to the
flat buffer; reads hand back a copy, so a caller cannot accidentally hold a view
that keeps changing underneath it.
"""

import threading
from typing import Sequence

import torch
from einops import rearrange


class FrameBuffer:
    """Addresses the flat colour buffer as ``(frame, camera, H, W, 3)`` images.

    Attributes:
        num_frames: Length of the time axis.
        num_cameras: Number of cameras per frame.
        height: Image height in pixels.
        width: Image width in pixels.
        lock: Guards writes, because the editor runs on a background thread
            while the optimiser reads from the main one.
    """

    def __init__(
        self,
        all_rgbs: torch.Tensor,
        num_frames: int,
        num_cameras: int,
        height: int,
        width: int,
    ):
        """
        Args:
            all_rgbs: ``(num_frames * num_cameras * H * W, 3)`` colours, laid out
                frame-major.  Held by reference, not copied.
            num_frames: Length of the time axis.
            num_cameras: Number of cameras per frame.
            height: Image height in pixels.
            width: Image width in pixels.
        """
        expected = num_frames * num_cameras * height * width
        if all_rgbs.shape[0] != expected:
            raise ValueError(
                f"expected {expected} colours for {num_frames} frames x {num_cameras} cameras "
                f"x {height}x{width}, got {all_rgbs.shape[0]}"
            )
        self._flat = all_rgbs
        self.num_frames = num_frames
        self.num_cameras = num_cameras
        self.height = height
        self.width = width
        self.lock = threading.Lock()

    @property
    def images(self) -> torch.Tensor:
        """The buffer as ``(frame, camera, H, W, 3)``; a view, not a copy."""
        return self._flat.view(self.num_frames, self.num_cameras, self.height, self.width, 3)

    def snapshot(self) -> torch.Tensor:
        """A detached copy of the whole buffer, shaped as images.

        Used to keep the *original* unedited images around as the conditioning
        signal for InstructPix2Pix, which must not drift as editing proceeds.
        """
        return self._flat.clone().view(
            self.num_frames, self.num_cameras, self.height, self.width, 3
        )

    def get(self, frame: int, cameras, device=None) -> torch.Tensor:
        """Read images as ``(N, 3, H, W)``, the layout the diffusion model wants.

        Always returns a copy.  Indexing a single camera would otherwise hand
        back a view that keeps tracking later writes, so the same call would
        behave differently depending on whether ``device`` forces a transfer.

        Args:
            frame: Frame index.
            cameras: Camera index or sequence of camera indices.
            device: Optional device to move the result to.
        """
        images = self.images[frame, cameras]
        if images.dim() == 3:  # a single camera
            images = images.unsqueeze(0)
        images = images.clone()
        if device is not None:
            images = images.to(device)
        return rearrange(images, "n h w c -> n c h w")

    def set(self, frame: int, cameras, images: torch.Tensor) -> None:
        """Write ``(N, 3, H, W)`` images back, converting layout and dtype.

        Takes :attr:`lock`, so this is safe to call while the optimiser is
        sampling rays from the same buffer.
        """
        update = rearrange(images, "n c h w -> n h w c").to(self._flat)
        with self.lock:
            self.images[frame, cameras] = update

    def set_many(self, frames: Sequence[int], camera: int, images: torch.Tensor) -> None:
        """Write one camera across several frames, as ``(N, 3, H, W)``."""
        update = rearrange(images, "n c h w -> n h w c").to(self._flat)
        with self.lock:
            self.images[frames, camera] = update
