"""Rendering a dataset or a camera path, and scoring the result.

Two entry points share the same back half -- march rays, colourise the depth,
write per-frame PNGs, and mux everything into a video:

:func:`evaluate`
    Renders the held-out views a dataset already carries and, since ground truth
    is available, reports PSNR/SSIM/LPIPS.
:func:`render_path`
    Renders novel views along a synthesised camera path.  No ground truth, so no
    metrics.
"""

import os
import sys
from copy import deepcopy
from typing import List, Sequence

import imageio
import numpy as np
import torch
from tqdm.auto import tqdm

from ..data.ray_utils import get_rays, ndc_rays_blender
from ..utils.image import colorize_depth
from ..utils.metrics import mse_to_psnr, rgb_lpips, rgb_ssim
from .renderer import render_rays

#: Frame rate and quality for the written videos.
VIDEO_FPS = 15
VIDEO_QUALITY = 9


class _FrameWriter:
    """Collects rendered frames, writes them out, then muxes videos.

    Individual frames go into a ``buffer/`` subdirectory so the videos sit at
    the top of the output directory rather than being buried among thousands of
    PNGs.
    """

    def __init__(self, save_path: str, prefix: str = ""):
        self.save_path = save_path
        self.prefix = prefix
        self.rgb_frames: List[np.ndarray] = []
        self.depth_frames: List[np.ndarray] = []
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "buffer"), exist_ok=True)

    def add(self, index: int, rgb: np.ndarray, depth: np.ndarray) -> None:
        """Record one frame and write it to disk."""
        self.rgb_frames.append(rgb)
        self.depth_frames.append(depth)
        stem = f"{self.save_path}/buffer/{self.prefix}{index:03d}"
        imageio.imwrite(f"{stem}.png", rgb)
        imageio.imwrite(f"{stem}-rgbd.png", np.concatenate((rgb, depth), axis=1))

    def finish(self, quality: int = VIDEO_QUALITY) -> None:
        """Write the colour and depth videos."""
        for name, frames in (("video", self.rgb_frames), ("depthvideo", self.depth_frames)):
            stack = np.stack(frames)
            imageio.mimwrite(
                f"{self.save_path}/{self.prefix}{name}.mp4",
                stack,
                fps=VIDEO_FPS,
                quality=quality,
            )
            imageio.mimwrite(f"{self.save_path}/{self.prefix}{name}.gif", stack)


def _write_metrics(save_path: str, prefix: str, psnrs, ssims, lpips_alex, lpips_vgg) -> None:
    """Write mean metrics next to the rendered frames."""
    values = [np.mean(psnrs)]
    if ssims:
        values += [np.mean(ssims), np.mean(lpips_alex), np.mean(lpips_vgg)]
    np.savetxt(f"{save_path}/{prefix}mean.txt", np.asarray(values))


@torch.no_grad()
def evaluate(
    test_dataset,
    field,
    save_path: str,
    N_vis: int = 5,
    prefix: str = "",
    N_samples: int = -1,
    white_bg: bool = False,
    ndc_ray: bool = False,
    compute_extra_metrics: bool = True,
    device: str = "cuda",
) -> List[float]:
    """Render a dataset's views and score them against ground truth.

    Args:
        test_dataset: A stacked dataset, i.e. built with ``is_stack=True``.
        field: The radiance field to render.
        save_path: Directory for frames, videos and the metric summary.
        N_vis: Number of views to render, evenly spaced; ``-1`` renders all.
        prefix: Filename prefix, used to tag intermediate visualisations.
        N_samples: Samples per ray; ``-1`` uses the field's own estimate.
        white_bg: Composite over white.
        ndc_ray: Sample in normalised device coordinates.
        compute_extra_metrics: Also compute SSIM and LPIPS, which are far slower
            than PSNR.
        device: Device to render on.

    Returns:
        Per-view PSNR values.
    """
    psnrs, ssims, lpips_alex, lpips_vgg = [], [], [], []
    writer = _FrameWriter(save_path, prefix)

    # The field's own near/far comes from training; evaluation may use a
    # different range, so swap it in and restore it afterwards.
    original_near_far = deepcopy(field.near_far)
    field.near_far = test_dataset.near_far

    interval = 1 if N_vis < 0 else max(test_dataset.all_rays.shape[0] // N_vis, 1)
    indices = list(range(0, test_dataset.all_rays.shape[0], interval))
    width, height = test_dataset.img_wh

    try:
        for output_index, rays in enumerate(
            tqdm(test_dataset.all_rays[::interval], file=sys.stdout, desc="evaluating")
        ):
            rgb_map, depth_map = render_rays(
                rays.view(-1, rays.shape[-1]),
                field,
                chunk=4096,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )
            rgb_map = rgb_map.clamp(0.0, 1.0).reshape(height, width, 3).cpu()
            depth_map = depth_map.reshape(height, width).cpu()
            depth_vis, _ = colorize_depth(depth_map.numpy(), test_dataset.near_far)

            if len(test_dataset.all_rgbs):
                gt_rgb = test_dataset.all_rgbs[indices[output_index]].view(height, width, 3)
                psnrs.append(mse_to_psnr(torch.mean((rgb_map - gt_rgb) ** 2).item()))
                if compute_extra_metrics:
                    ssims.append(rgb_ssim(rgb_map, gt_rgb, 1))
                    lpips_alex.append(rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "alex", field.device))
                    lpips_vgg.append(rgb_lpips(gt_rgb.numpy(), rgb_map.numpy(), "vgg", field.device))

            writer.add(output_index, (rgb_map.numpy() * 255).astype("uint8"), depth_vis)

        writer.finish()
        if psnrs:
            _write_metrics(save_path, prefix, psnrs, ssims, lpips_alex, lpips_vgg)
    finally:
        field.near_far = original_near_far

    return psnrs


@torch.no_grad()
def render_path(
    test_dataset,
    field,
    c2ws: Sequence[np.ndarray],
    save_path: str,
    prefix: str = "",
    N_samples: int = -1,
    white_bg: bool = False,
    ndc_ray: bool = False,
    device: str = "cuda",
) -> None:
    """Render novel views along a camera path and write a fly-through video.

    Args:
        test_dataset: Supplies intrinsics, resolution and near/far.
        field: The radiance field to render.
        c2ws: Camera-to-world matrices, one per output frame.
        save_path: Directory for frames and videos.
        prefix: Filename prefix.
        N_samples: Samples per ray; ``-1`` uses the field's own estimate.
        white_bg: Composite over white.
        ndc_ray: Sample in normalised device coordinates.
        device: Device to render on.
    """
    writer = _FrameWriter(save_path, prefix)
    original_near_far = deepcopy(field.near_far)
    field.near_far = test_dataset.near_far
    width, height = test_dataset.img_wh

    try:
        for index, c2w in enumerate(tqdm(c2ws, file=sys.stdout, desc="rendering path")):
            rays_o, rays_d = get_rays(test_dataset.directions, torch.FloatTensor(c2w))
            if ndc_ray:
                rays_o, rays_d = ndc_rays_blender(
                    height, width, test_dataset.focal[0], 1.0, rays_o, rays_d
                )
            # Sweep time linearly along the path, so the camera move and the
            # scene's own motion play out together.
            time_channel = torch.full([*rays_o.shape[:-1], 1], index / len(c2ws))
            rays = torch.cat([rays_o, rays_d, time_channel], dim=1)

            rgb_map, depth_map = render_rays(
                rays,
                field,
                chunk=1024,
                N_samples=N_samples,
                ndc_ray=ndc_ray,
                white_bg=white_bg,
                device=device,
            )
            rgb_map = rgb_map.clamp(0.0, 1.0).reshape(height, width, 3).cpu()
            depth_map = depth_map.reshape(height, width).cpu()
            depth_vis, _ = colorize_depth(depth_map.numpy(), test_dataset.near_far)

            writer.add(index, (rgb_map.numpy() * 255).astype("uint8"), depth_vis)

        writer.finish(quality=8)
    finally:
        field.near_far = original_near_far
