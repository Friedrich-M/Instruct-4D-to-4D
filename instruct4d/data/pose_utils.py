"""Camera-pose bookkeeping shared by the forward-facing dataset loaders.

Adapted from the LLFF/NeRF reference implementations.  Two jobs live here:
recentring a set of camera poses around their average (a prerequisite for the
NDC parameterisation), and synthesising the smooth spiral camera path used to
render fly-through videos.
"""

from typing import Tuple

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    """Return ``v`` scaled to unit length."""
    return v / np.linalg.norm(v)


def average_poses(poses: np.ndarray) -> np.ndarray:
    """Compute the average of a set of camera poses.

    The translation is a plain mean, but the rotation has to be rebuilt so it
    stays orthonormal: the mean ``y`` axis is generally not perpendicular to the
    mean ``z`` axis, so ``x`` is recovered as ``y x z`` and ``y`` is then
    recomputed as ``z x x``.

    Args:
        poses: ``(N, 3, 4)`` camera-to-world matrices.

    Returns:
        ``(3, 4)`` average camera-to-world matrix.
    """
    center = poses[..., 3].mean(0)
    z = normalize(poses[..., 2].mean(0))
    y_mean = poses[..., 1].mean(0)
    x = normalize(np.cross(z, y_mean))
    y = np.cross(x, z)
    return np.stack([x, y, z, center], axis=1)


def center_poses(poses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Move all poses into the frame of their average pose.

    NDC assumes the cameras look down ``-z`` from near the origin, so the poses
    must be recentred before :func:`~instruct4d.data.ray_utils.ndc_rays_blender`
    can be applied.  See https://github.com/bmild/nerf/issues/34.

    Args:
        poses: ``(N, 3, 4)`` camera-to-world matrices.

    Returns:
        ``(centered_poses, average_pose)`` of shapes ``(N, 3, 4)`` and ``(4, 4)``.
    """
    pose_avg = np.eye(4)
    pose_avg[:3] = average_poses(poses)

    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))
    poses_homo = np.concatenate([poses, last_row], axis=1)
    poses_centered = (np.linalg.inv(pose_avg) @ poses_homo)[:, :3]
    return poses_centered, pose_avg


def view_matrix(z: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Build a camera-to-world matrix looking along ``z`` from ``position``."""
    vec2 = normalize(z)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, position], axis=1)
    return m


def get_spiral(
    c2ws: np.ndarray,
    near_fars: np.ndarray,
    rads_scale: float = 1.0,
    n_views: int = 120,
    zrate: float = 0.5,
    n_rots: int = 2,
) -> np.ndarray:
    """Synthesise a spiral camera path around the average training pose.

    The camera orbits in ``xy`` while oscillating in ``z``, always looking at a
    focus depth chosen from the scene's near/far bounds.  This is the path used
    for the rendered fly-through videos.

    Args:
        c2ws: ``(N, 3, 4)`` training camera-to-world matrices.
        near_fars: ``(N, 2)`` per-camera near and far bounds.
        rads_scale: Shrinks or grows the orbit radius.
        n_views: Number of poses to generate.
        zrate: Vertical oscillations per full rotation.
        n_rots: Number of full rotations along the path.

    Returns:
        ``(n_views, 4, 4)`` camera-to-world matrices.
    """
    c2w = average_poses(c2ws)
    up = normalize(c2ws[:, :3, 1].sum(0))

    # Blend the closest and furthest depths into a single focus distance; the
    # 0.75 weighting comes from the LLFF reference implementation.
    blend = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - blend) / close_depth + blend / inf_depth)

    # Orbit radius: the 90th percentile of camera offsets, so a few outlying
    # cameras cannot blow the path up.
    radii = np.percentile(np.abs(c2ws[:, :3, 3]), 90, axis=0) * rads_scale
    radii = np.array(list(radii) + [1.0])

    render_poses = []
    for theta in np.linspace(0.0, 2.0 * np.pi * n_rots, n_views + 1)[:-1]:
        offset = np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0]) * radii
        position = np.dot(c2w[:3, :4], offset)
        look_at = np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.0]))
        render_poses.append(view_matrix(normalize(position - look_at), up, position))
    return np.stack(render_poses)
