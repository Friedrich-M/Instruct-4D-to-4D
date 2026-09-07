"""Voxel-grid resolution bookkeeping for the TensoRF field."""

from typing import List, Sequence

import numpy as np
import torch


def n_to_reso(n_voxels: int, bbox: torch.Tensor) -> List[int]:
    """Split a voxel budget across the axes of a bounding box.

    Voxels are kept cubic, so the per-axis resolution is proportional to the
    box's extent along that axis.

    Args:
        n_voxels: Total number of voxels to spend.
        bbox: ``(2, 3)`` tensor holding the min and max corner.

    Returns:
        ``[nx, ny, nz]`` resolution.
    """
    xyz_min, xyz_max = bbox
    voxel_size = ((xyz_max - xyz_min).prod() / n_voxels).pow(1 / 3)
    return ((xyz_max - xyz_min) / voxel_size).long().tolist()


def cal_n_samples(reso: Sequence[int], step_ratio: float = 0.5) -> int:
    """Number of samples needed to march a ray across the grid.

    A ray crossing the volume diagonally covers ``||reso||`` voxels, so taking
    one sample every ``step_ratio`` voxels gives this count.
    """
    return int(np.linalg.norm(reso) / step_ratio)
