"""Geometry feature extraction for adaptive point-cloud downsampling.

This module focuses on local geometric complexity signals, such as curvature,
normal variation, and boundary likelihood. These signals are later consumed by
`adaptive_sampler.py` to allocate point budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class GeometryFeatureConfig:
    """Configuration for geometry feature extraction.

    Attributes
    ----------
    k_neighbors:
        Number of neighbors used for local covariance / PCA.
    boundary_radius:
        Radius for boundary score estimation.
    eps:
        Numerical stability epsilon.
    """

    k_neighbors: int = 24
    boundary_radius: float = 0.08
    eps: float = 1e-9


def estimate_geometry_features(
    points_xyz: np.ndarray,
    config: GeometryFeatureConfig | None = None,
) -> Dict[str, np.ndarray]:
    """Estimate geometry features for each point.

    Parameters
    ----------
    points_xyz:
        Float array of shape [N, 3].
    config:
        Optional GeometryFeatureConfig.

    Returns
    -------
    dict
        {
            "curvature": np.ndarray [N],
            "normal_consistency": np.ndarray [N],
            "boundary_score": np.ndarray [N],
            "geometry_score": np.ndarray [N],
        }

    Notes
    -----
    - Current implementation is intentionally simple and readable.
    - It favors correctness over speed for first-stage prototyping.
    """
    if config is None:
        config = GeometryFeatureConfig()
    if points_xyz.ndim != 2 or points_xyz.shape[1] != 3:
        raise ValueError(f"points_xyz must be [N, 3], got {points_xyz.shape}")

    n = points_xyz.shape[0]
    tree = cKDTree(points_xyz)
    _, knn_idx = tree.query(points_xyz, k=min(config.k_neighbors, n))
    if knn_idx.ndim == 1:
        knn_idx = knn_idx[:, None]

    curvature = np.zeros(n, dtype=np.float32)
    normal_consistency = np.zeros(n, dtype=np.float32)

    normals = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        nbr = points_xyz[knn_idx[i]]
        centered = nbr - nbr.mean(axis=0, keepdims=True)
        cov = centered.T @ centered / max(len(nbr) - 1, 1)
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)
        evals = evals[order]
        evecs = evecs[:, order]

        lam0, lam1, lam2 = evals
        curvature[i] = lam0 / (lam0 + lam1 + lam2 + config.eps)
        normals[i] = evecs[:, 0]

    for i in range(n):
        nbr_normals = normals[knn_idx[i]]
        cos_sim = np.abs(nbr_normals @ normals[i])
        normal_consistency[i] = 1.0 - np.mean(cos_sim)

    counts = np.array([len(tree.query_ball_point(p, r=config.boundary_radius)) for p in points_xyz])
    boundary_score = 1.0 / (counts.astype(np.float32) + config.eps)

    geometry_score = normalize_01(0.45 * curvature + 0.35 * normal_consistency + 0.20 * normalize_01(boundary_score))

    return {
        "curvature": curvature,
        "normal_consistency": normal_consistency,
        "boundary_score": boundary_score,
        "geometry_score": geometry_score,
    }


def normalize_01(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Min-max normalize to [0, 1].

    Parameters
    ----------
    values:
        Array of shape [...].
    eps:
        Stability term.

    Returns
    -------
    np.ndarray
        Same shape as `values`, normalized to [0, 1].
    """
    vmin = float(values.min())
    vmax = float(values.max())
    return (values - vmin) / (vmax - vmin + eps)
