"""Camera-ray construction for the forward-facing multi-view datasets.

The DyNeRF scenes are forward-facing, so the radiance field is parameterised in
normalised device coordinates (NDC): rays are projected into a unit cube where
the far plane maps to a finite value, which is what lets a bounded voxel grid
cover an unbounded scene.

Two conventions appear here.  Directions are built in the OpenGL/Blender camera
frame (``+x`` right, ``+y`` up, ``-z`` forward) to match the pose convention of
the LLFF-style ``poses_bounds.npy`` these datasets ship.
"""

from typing import Optional, Sequence, Tuple

import torch


def get_ray_directions(
    height: int,
    width: int,
    focal: Sequence[float],
    center: Optional[Sequence[float]] = None,
) -> torch.Tensor:
    """Ray directions for every pixel, in the OpenGL/Blender camera frame.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        focal: ``(fx, fy)`` focal lengths in pixels.
        center: ``(cx, cy)`` principal point; defaults to the image centre.

    Returns:
        ``(height, width, 3)`` unnormalised directions.  They are deliberately
        left unnormalised so that the ``z`` component stays ``-1``, which the NDC
        projection below relies on.
    """
    # `+ 0.5` puts the sample at the centre of each pixel.
    j, i = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    i, j = i + 0.5, j + 0.5
    cx, cy = center if center is not None else (width / 2, height / 2)
    return torch.stack(
        [(i - cx) / focal[0], -(j - cy) / focal[1], -torch.ones_like(i)], dim=-1
    )


def get_rays(directions: torch.Tensor, c2w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Place camera-frame directions into world space.

    Args:
        directions: ``(H, W, 3)`` from :func:`get_ray_directions`.
        c2w: ``(3, 4)`` or ``(4, 4)`` camera-to-world matrix.

    Returns:
        ``(rays_o, rays_d)``, both ``(H * W, 3)``.  Every ray shares the camera
        centre as its origin.
    """
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o.reshape(-1, 3), rays_d.reshape(-1, 3)


def ndc_rays_blender(
    height: int,
    width: int,
    focal: float,
    near: float,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Project world-space rays into normalised device coordinates.

    Follows Appendix C of the original NeRF paper.  Origins are first pushed
    onto the near plane, then both origin and direction are mapped so that the
    scene occupies ``[-1, 1]^2 x [-1, 1]`` with the far plane at ``z = 1``.

    Args:
        height: Image height in pixels.
        width: Image width in pixels.
        focal: Focal length in pixels.
        near: Distance to the near plane.
        rays_o: ``(N, 3)`` world-space ray origins.
        rays_d: ``(N, 3)`` world-space ray directions.

    Returns:
        ``(rays_o, rays_d)`` in NDC, both ``(N, 3)``.
    """
    # Shift ray origins onto the near plane.
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    o0 = -1.0 / (width / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (height / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (width / (2.0 * focal)) * (
        rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2]
    )
    d1 = -1.0 / (height / (2.0 * focal)) * (
        rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2]
    )
    d2 = -2.0 * near / rays_o[..., 2]

    return torch.stack([o0, o1, o2], dim=-1), torch.stack([d0, d1, d2], dim=-1)
