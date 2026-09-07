"""Chunked volumetric rendering."""

from typing import Tuple

import torch


def render_rays(
    rays: torch.Tensor,
    field,
    chunk: int = 4096,
    N_samples: int = -1,
    ndc_ray: bool = False,
    white_bg: bool = True,
    is_train: bool = False,
    device: str = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render a batch of rays in fixed-size chunks.

    A full image is far too many rays to march at once, so they are split into
    chunks that fit in memory and the results concatenated.

    Args:
        rays: ``(N, 7)`` rays as ``[origin, direction, time]``.  All rays in a
            single chunk must share a timestamp, so a batch spanning several
            frames must already be grouped by frame.
        field: A :class:`~instruct4d.fields.tensor_base.StreamTensorBase`.
        chunk: Rays per forward pass.
        N_samples: Samples per ray; ``-1`` uses the field's own estimate.
        ndc_ray: Sample in normalised device coordinates.
        white_bg: Composite over a white background.
        is_train: Enables the field's training-time stochasticity.
        device: Device to move each chunk onto.

    Returns:
        ``(rgb, depth)`` of shapes ``(N, 3)`` and ``(N,)``.
    """
    rgbs, depths = [], []
    n_rays = rays.shape[0]
    n_chunks = n_rays // chunk + int(n_rays % chunk > 0)

    for i in range(n_chunks):
        rays_chunk = rays[i * chunk : (i + 1) * chunk].to(device)
        rgb_map, depth_map = field(
            rays_chunk,
            is_train=is_train,
            white_bg=white_bg,
            ndc_ray=ndc_ray,
            N_samples=N_samples,
        )
        rgbs.append(rgb_map)
        depths.append(depth_map)

    return torch.cat(rgbs), torch.cat(depths)
