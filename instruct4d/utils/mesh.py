"""Mesh extraction from the density field."""

from typing import Optional, Sequence

import numpy as np
import plyfile
import skimage.measure
import torch


def convert_sdf_samples_to_ply(
    sdf_volume: torch.Tensor,
    ply_filename_out: str,
    bbox: torch.Tensor,
    level: float = 0.5,
    offset: Optional[Sequence[float]] = None,
    scale: Optional[float] = None,
) -> None:
    """Run marching cubes over a density volume and write the result as PLY.

    Adapted from https://github.com/RobotLocomotion/spartan.

    Args:
        sdf_volume: ``(nx, ny, nz)`` scalar field on the CPU.
        ply_filename_out: Destination path.
        bbox: ``(2, 3)`` tensor holding the volume's min and max corner in world
            space, used to map voxel indices back to world coordinates.
        level: Iso-level at which to extract the surface.
        offset: Optional translation subtracted from the vertices.
        scale: Optional divisor applied to the vertices.
    """
    volume = sdf_volume.numpy()
    voxel_size = list((bbox[1] - bbox[0]) / np.array(volume.shape))

    verts, faces, _, _ = skimage.measure.marching_cubes(volume, level=level, spacing=voxel_size)
    # marching_cubes emits inward-facing triangles for this convention.
    faces = faces[..., ::-1]

    mesh_points = verts + np.asarray(bbox[0]).reshape(1, 3)
    if scale is not None:
        mesh_points = mesh_points / scale
    if offset is not None:
        mesh_points = mesh_points - offset

    verts_tuple = np.zeros((len(mesh_points),), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    verts_tuple[:] = list(map(tuple, mesh_points))
    faces_tuple = np.array(
        [(f.tolist(),) for f in faces], dtype=[("vertex_indices", "i4", (3,))]
    )

    plyfile.PlyData(
        [
            plyfile.PlyElement.describe(verts_tuple, "vertex"),
            plyfile.PlyElement.describe(faces_tuple, "face"),
        ]
    ).write(ply_filename_out)
    print(f"mesh saved to {ply_filename_out}")
