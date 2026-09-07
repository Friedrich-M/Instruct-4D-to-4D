"""Camera geometry: ray construction, NDC, and depth-based warping.

These are the parts where a sign error is silent -- the pipeline still runs and
produces a plausible-looking but wrong warp -- so they are worth pinning down.
"""

import numpy as np
import torch

from instruct4d.data.ray_utils import get_ray_directions, get_rays, ndc_rays_blender
from instruct4d.data.pose_utils import average_poses, center_poses, get_spiral
from instruct4d.editing.warping import apply_warp, project_points_to_view, unproject_depth_ndc


def test_ray_directions_use_the_opengl_convention():
    dirs = get_ray_directions(8, 12, [10.0, 10.0])
    assert dirs.shape == (8, 12, 3)
    # -z is forward, and the directions are left unnormalised so z stays -1.
    assert torch.all(dirs[..., 2] == -1.0)
    # +x is right: the column index increases with x.
    assert dirs[0, -1, 0] > dirs[0, 0, 0]
    # +y is up: the row index decreases with y.
    assert dirs[-1, 0, 1] < dirs[0, 0, 1]


def test_get_rays_shares_one_origin():
    dirs = get_ray_directions(4, 6, [10.0, 10.0])
    c2w = torch.eye(4)[:3]
    c2w[:, 3] = torch.tensor([1.0, 2.0, 3.0])
    rays_o, rays_d = get_rays(dirs, c2w)
    assert rays_o.shape == (24, 3) and rays_d.shape == (24, 3)
    assert torch.allclose(rays_o, torch.tensor([1.0, 2.0, 3.0]).expand(24, 3))


def test_ndc_maps_the_far_plane_to_one():
    dirs = get_ray_directions(6, 8, [12.0, 12.0])
    rays_o, rays_d = get_rays(dirs, torch.eye(4)[:3])
    ndc_o, ndc_d = ndc_rays_blender(6, 8, 12.0, 1.0, rays_o, rays_d)
    # Marching the full parameter range lands exactly on z = 1.
    assert torch.allclose(ndc_o[:, 2] + ndc_d[:, 2], torch.ones(48), atol=1e-5)


def test_unproject_recovers_a_world_plane():
    height, width, focal = 10, 14, 18.0
    dirs = get_ray_directions(height, width, [focal, focal])
    rays_o, rays_d = get_rays(dirs, torch.eye(4)[:3])
    ndc_o, ndc_d = ndc_rays_blender(height, width, focal, 1.0, rays_o, rays_d)
    rays = torch.cat([ndc_o, ndc_d, torch.zeros(height * width, 1)], dim=1)

    points = unproject_depth_ndc(rays, torch.full((height * width, 1), 0.5), height, width, (focal, focal))
    depth = points[..., 2]
    assert torch.allclose(depth, depth.reshape(-1)[0].expand_as(depth), atol=1e-4)
    assert (depth < 0).all(), "the camera looks down -z"

    # At a fixed depth the rays fan out linearly across the image.
    row = points[height // 2, :, 0]
    steps = row[1:] - row[:-1]
    assert torch.allclose(steps, steps[0].expand_as(steps), atol=1e-4)


def _plane_scene(height=24, width=32, focal=30.0, depth=-4.0):
    """A fronto-parallel plane, expressed the way the pipeline's rays are."""
    ys, xs = torch.meshgrid(
        torch.arange(height).float(), torch.arange(width).float(), indexing="ij"
    )
    z = torch.full((height, width), depth)
    points = torch.stack(
        [-(xs + 0.5 - width / 2) / focal * z, (ys + 0.5 - height / 2) / focal * z, z], dim=-1
    )
    intrinsics = torch.tensor([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    return points, intrinsics, torch.eye(4)


def test_warp_between_identical_views_is_the_identity():
    points, intrinsics, extrinsics = _plane_scene()
    pixel_map = project_points_to_view(points, intrinsics, extrinsics)
    image = torch.rand(*points.shape[:2], 3)

    warped, mask, diff = apply_warp(
        pixel_map, torch.zeros_like(image), image, points, points, diff_thres=1e-4
    )
    assert mask.float().mean() > 0.8
    assert diff[mask].max() < 1e-4
    assert torch.allclose(warped[mask], image[mask], atol=1e-4)


def test_warp_rejects_disagreeing_geometry():
    points, intrinsics, extrinsics = _plane_scene()
    pixel_map = project_points_to_view(points, intrinsics, extrinsics)
    image = torch.rand(*points.shape[:2], 3)

    # The source view claims a completely different surface at every pixel.
    elsewhere = points + torch.tensor([0.0, 0.0, 10.0])
    _, mask, _ = apply_warp(
        pixel_map, torch.zeros_like(image), image, points, elsewhere, diff_thres=0.2
    )
    assert not mask.any()


def test_center_poses_moves_the_average_to_the_origin():
    rng = np.random.RandomState(0)
    poses = rng.randn(9, 3, 4)
    centered, _ = center_poses(poses)
    assert np.allclose(average_poses(centered)[:, 3], 0.0, atol=1e-10)


def test_spiral_path_has_the_requested_length():
    rng = np.random.RandomState(1)
    poses, _ = center_poses(rng.randn(7, 3, 4))
    near_fars = np.abs(rng.randn(7, 2)) + 1.0
    path = get_spiral(poses, near_fars, rads_scale=0.4, n_views=13, zrate=-0.5)
    assert path.shape == (13, 4, 4)
    assert np.isfinite(path).all()
