"""Convert sampled point cloud into an initial Gaussian parameter set for 3DGS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class GaussianInitConfig:
    default_opacity: float = 0.1
    default_scale: float = 0.02


def initialize_gaussians(
    sampled_points_xyzrgb: np.ndarray,
    config: GaussianInitConfig | None = None,
) -> Dict[str, np.ndarray]:
    """Build initial Gaussian parameters from sampled points.

    Parameters
    ----------
    sampled_points_xyzrgb:
        [M, 6] containing xyz and rgb.
    config:
        Optional GaussianInitConfig.

    Returns
    -------
    dict
        {
            "xyz": [M, 3],
            "rgb": [M, 3],
            "opacity": [M, 1],
            "scale": [M, 3],
            "rotation_quat": [M, 4],
        }
    """
    if config is None:
        config = GaussianInitConfig()

    if sampled_points_xyzrgb.ndim != 2 or sampled_points_xyzrgb.shape[1] != 6:
        raise ValueError(f"sampled_points_xyzrgb must be [M, 6], got {sampled_points_xyzrgb.shape}")

    m = sampled_points_xyzrgb.shape[0]
    xyz = sampled_points_xyzrgb[:, :3].astype(np.float32)
    rgb = sampled_points_xyzrgb[:, 3:6].astype(np.float32)

    if rgb.max() > 1.5:
        rgb = rgb / 255.0

    opacity = np.full((m, 1), fill_value=config.default_opacity, dtype=np.float32)
    scale = np.full((m, 3), fill_value=config.default_scale, dtype=np.float32)
    rotation_quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (m, 1))

    return {
        "xyz": xyz,
        "rgb": rgb,
        "opacity": opacity,
        "scale": scale,
        "rotation_quat": rotation_quat,
    }
