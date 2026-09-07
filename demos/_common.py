"""Shared helpers for the component demos.

Importing this module also puts the repository root on ``sys.path``, so the
demos run straight from a checkout without ``pip install -e .`` first.  Import
it before anything from ``instruct4d``.
"""

import math
import os
import sys
from typing import List, Optional, Sequence

import numpy as np
import torch
from PIL import Image, ImageOps

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

#: Stable Diffusion's VAE and the UNet's downsampling stack together require a
#: side length divisible by 64.
SIZE_MULTIPLE = 64

#: File types :func:`sorted_image_paths` will pick up.
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def prepare_output_dir(path: str) -> str:
    """Create the output directory and return it."""
    os.makedirs(path, exist_ok=True)
    return path


def fit_to_multiple(image: Image.Image, long_side: Optional[int] = None) -> Image.Image:
    """Rescale an image so both sides are multiples of :data:`SIZE_MULTIPLE`.

    Args:
        image: The image to rescale.
        long_side: Target length of the longer side.  Defaults to leaving the
            resolution as it is, rounded to a valid multiple.

    Returns:
        The rescaled image.
    """
    width, height = image.size
    if long_side is None:
        long_side = max(width, height)

    scale = long_side / max(width, height)
    # Round the *short* side to a multiple first, then derive the long side, so
    # the aspect ratio moves as little as possible.
    scale = math.ceil(min(width, height) * scale / SIZE_MULTIPLE) * SIZE_MULTIPLE / min(width, height)
    target = (int(width * scale) // SIZE_MULTIPLE * SIZE_MULTIPLE,
              int(height * scale) // SIZE_MULTIPLE * SIZE_MULTIPLE)
    return ImageOps.fit(image, target, method=Image.Resampling.LANCZOS)


def sorted_image_paths(directory: str) -> List[str]:
    """List the images in a directory, in frame order.

    Frames are usually named ``0.png``, ``1.png``, ``10.png``, which sort wrong
    lexicographically, so numeric names are ordered by value.  Anything that is
    not an image is skipped, and mixed naming falls back to plain sorting.

    Raises:
        FileNotFoundError: If the directory holds no images.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"not a directory: {directory}")

    names = [n for n in os.listdir(directory) if n.lower().endswith(IMAGE_SUFFIXES)]
    if not names:
        raise FileNotFoundError(
            f"no images in {directory} (looked for {', '.join(IMAGE_SUFFIXES)})"
        )

    stems = [os.path.splitext(n)[0] for n in names]
    if all(stem.isdigit() for stem in stems):
        names.sort(key=lambda n: int(os.path.splitext(n)[0]))
    else:
        names.sort()
    return [os.path.join(directory, n) for n in names]


def load_frames(
    paths: Sequence[str], resize: Optional[int], device, dtype=torch.float32
) -> torch.Tensor:
    """Load images as a ``(F, 3, H, W)`` tensor in ``[0, 1]``.

    Raises:
        ValueError: If the images are not all the same size, which would make
            them impossible to stack into one batch.
    """
    frames, sizes = [], set()
    for path in paths:
        image = fit_to_multiple(Image.open(path).convert("RGB"), resize)
        sizes.add(image.size)
        if len(sizes) > 1:
            raise ValueError(
                f"frames must share one resolution, found {sorted(sizes)}; "
                "pass --resize to bring them to a common size"
            )
        frames.append(torch.from_numpy(np.asarray(image) / 255.0).permute(2, 0, 1))
    return torch.stack(frames).to(device=device, dtype=dtype)


def save_sheet(images: torch.Tensor, path: str) -> str:
    """Write ``(F, 3, H, W)`` frames as a single-row contact sheet."""
    import torchvision

    torchvision.utils.save_image(images.float().cpu(), path, nrow=images.shape[0], padding=0)
    return path


def output_name(prompt: str, stem: str) -> str:
    """Build a filename tagged with the last word of the prompt."""
    tag = prompt.strip().split(" ")[-1].strip("?!.,") or "edit"
    return f"{stem}_{tag}.png"
