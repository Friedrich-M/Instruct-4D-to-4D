"""Optional dumps of intermediate editing results.

Editing is slow and hard to inspect from metrics alone: it is usually obvious
from a contact sheet whether the diffusion model took the prompt, whether the
warps line up, and whether the refinement pass closed the holes.  Writing those
sheets is off by default and, when enabled, goes to a directory under the run's
log folder rather than the working directory.
"""

import os
from typing import Optional

import torch
import torchvision


class DebugImageWriter:
    """Writes batches of frames as contact sheets, or does nothing.

    A writer constructed with ``directory=None`` is inert, so callers can invoke
    :meth:`save` unconditionally instead of guarding every call.
    """

    def __init__(self, directory: Optional[str]):
        """
        Args:
            directory: Where to write the sheets, or ``None`` to disable.
        """
        self.directory = directory
        if directory:
            os.makedirs(directory, exist_ok=True)

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def save(self, step: int, name: str, images: torch.Tensor) -> None:
        """Write ``(N, 3, H, W)`` images as a single row.

        Args:
            step: Iteration number, used to order the files.
            name: Short label for what the images are.
            images: The batch to write.
        """
        if not self.directory:
            return
        path = os.path.join(self.directory, f"{step:04d}_{name}.png")
        torchvision.utils.save_image(images, path, nrow=images.shape[0], normalize=True)
