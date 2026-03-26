"""Spatial block partitioning utilities.

Block partitioning provides a simple way to allocate budget locally, ensuring
small geometric structures are not entirely removed by global sampling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BlockPartitionConfig:
    voxel_size: float = 0.5


def build_blocks(points_xyz: np.ndarray, config: BlockPartitionConfig | None = None) -> Dict[str, np.ndarray]:
    """Partition point cloud into 3D voxels.

    Parameters
    ----------
    points_xyz:
        [N, 3] float array.
    config:
        Optional config containing voxel size.

    Returns
    -------
    dict
        {
            "voxel_coords": [N, 3] int voxel coordinate per point,
            "block_ids": [N] int compact block index,
            "num_blocks": int,
            "block_counts": [B] number of points in each block,
        }
    """
    if config is None:
        config = BlockPartitionConfig()
    if config.voxel_size <= 0:
        raise ValueError("voxel_size must be > 0")

    xyz_min = points_xyz.min(axis=0, keepdims=True)
    voxel_coords = np.floor((points_xyz - xyz_min) / config.voxel_size).astype(np.int32)

    uniq, inverse, counts = np.unique(voxel_coords, axis=0, return_inverse=True, return_counts=True)

    return {
        "voxel_coords": voxel_coords,
        "block_ids": inverse.astype(np.int32),
        "num_blocks": int(len(uniq)),
        "block_counts": counts.astype(np.int32),
    }
