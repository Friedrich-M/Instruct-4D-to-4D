"""Depth-based warping between camera views (Section 3.4).

Once a pseudo-view has been edited, the same edit has to appear in every other
view of the same timestamp.  Because the radiance field can render depth, each
pixel of the target view can be unprojected to a 3D point, projected into the
source view, and its colour resampled -- a perspective transform that is exact
wherever the surface is visible in both views.

Points that are occluded in one of the two views land on a *different* surface,
so the resampled colour would be wrong.  Comparing the two views' 3D points at
the same pixel detects exactly that case: when they disagree by more than a
threshold, the correspondence is rejected.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

#: Points that disagree by more than this (in world units) are treated as an
#: occlusion and left unwarped.
DEFAULT_DIFF_THRESHOLD = 0.2


def _to_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Append a row of ones to a ``(D, N)`` stack of column vectors."""
    return torch.cat([x, torch.ones([1, *x.shape[1:]], device=x.device, dtype=x.dtype)], dim=0)


def _from_homogeneous(x: torch.Tensor) -> torch.Tensor:
    """Divide a ``(D, N)`` stack of column vectors by its last row."""
    return x[:-1] / x[-1:]


def project_points_to_view(
    points: torch.Tensor, intrinsics: torch.Tensor, world_to_camera: torch.Tensor
) -> torch.Tensor:
    """Project world-space points into another camera's image plane.

    Args:
        points: ``(H, W, 3)`` world-space points, one per pixel of the *target*
            view, typically obtained by unprojecting its rendered depth.
        intrinsics: ``(3, 3)`` intrinsics of the *source* view.
        world_to_camera: ``(4, 4)`` extrinsics of the *source* view.

    Returns:
        ``(H, W, 2)`` pixel coordinates in the source view.
    """
    height, width, _ = points.shape
    x = points.reshape(-1, 3).T
    x = _from_homogeneous(world_to_camera @ _to_homogeneous(x))
    x = intrinsics @ x
    return _from_homogeneous(x).T.reshape(height, width, 2)


def apply_warp(
    target_from_source: torch.Tensor,
    target_image: torch.Tensor,
    source_image: torch.Tensor,
    target_points: torch.Tensor,
    source_points: torch.Tensor,
    diff_thres: float = DEFAULT_DIFF_THRESHOLD,
    default: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Resample ``source_image`` into the target view, rejecting occlusions.

    Args:
        target_from_source: ``(H, W, 2)`` from :func:`project_points_to_view`:
            where each target pixel lands in the source image.
        target_image: ``(H, W, 3)`` target view; only its shape and dtype are
            used, the content is fully replaced.
        source_image: ``(H, W, 3)`` already-edited source view to pull from.
        target_points: ``(H, W, 3)`` world-space points of the target view.
        source_points: ``(H, W, 3)`` world-space points of the source view.
        diff_thres: Rejection threshold on the distance between the target's own
            3D point and the one sampled from the source.
        default: Colour written where the warp is rejected; black by default.

    Returns:
        ``(warped, mask, diff)``.  ``mask`` is ``True`` where the warp is
        trustworthy, and ``diff`` is the per-pixel 3D disagreement.
    """
    height, width = target_image.shape[0], target_image.shape[1]
    if default is None:
        default = torch.zeros(3)
    default = default.to(dtype=target_image.dtype, device=target_image.device)

    # Pixel centres sit at half-integer coordinates.
    y, x = (target_from_source - 0.5).unbind(-1)
    mask = (0 <= x) & (x <= height - 1) & (0 <= y) & (y <= width - 1)

    # grid_sample wants normalised [-1, 1] coordinates. Multiplying by the mask
    # clamps out-of-frame samples to the centre rather than letting them wrap.
    x_norm = (x / (height - 1) * 2 - 1) * mask
    y_norm = (y / (width - 1) * 2 - 1) * mask
    grid = torch.stack([-y_norm, x_norm], dim=-1).unsqueeze(0)

    warped = _grid_sample_hwc(source_image, grid)
    warped_points = _grid_sample_hwc(source_points, grid)

    # Where the two views disagree about what surface a pixel sees, one of them
    # is looking at an occluder.
    diff = (target_points - warped_points).norm(dim=-1)
    mask = mask & (diff < diff_thres)
    warped[~mask] = default
    return warped, mask, diff


def _grid_sample_hwc(image: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """``F.grid_sample`` for an ``(H, W, C)`` image, returning ``(H, W, C)``."""
    sampled = F.grid_sample(
        image[None].permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=True
    )
    return sampled.squeeze(0).permute(1, 2, 0)


def unproject_depth_ndc(
    rays: torch.Tensor,
    depth: torch.Tensor,
    height: int,
    width: int,
    focal: Tuple[float, float],
) -> torch.Tensor:
    """Recover world-space points from NDC rays and a rendered depth map.

    The field renders depth in normalised device coordinates, but the warp needs
    metric world coordinates, so the NDC projection has to be inverted.

    Args:
        rays: ``(H * W, 7)`` NDC rays as ``[origin, direction, time]``.
        depth: ``(H * W, 1)`` rendered NDC depth.
        height: Image height in pixels.
        width: Image width in pixels.
        focal: ``(fx, fy)`` focal lengths in pixels.

    Returns:
        ``(H, W, 3)`` world-space points.
    """
    points = rays[..., :3] + rays[..., 3:6] * depth

    # Invert the NDC projection (NeRF, Appendix C).
    z = 2 / (points[..., 2] - 1)
    x = -points[..., 0] * z * width / 2 / focal[0]
    y = -points[..., 1] * z * height / 2 / focal[1]
    return torch.stack([x, y, z], dim=-1).view(height, width, 3)
