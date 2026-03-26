"""Geometry features from kNN + PCA.

This module exposes a reusable function:
    compute_geometry_features(xyz, k=20)

Input
-----
xyz: np.ndarray, shape [N, 3]

Output
------
curvature: np.ndarray, shape [N]
planarity: np.ndarray, shape [N]
linearity: np.ndarray, shape [N]
normals: np.ndarray, shape [N, 3]

PCA eigenvalue convention (ascending): λ1 <= λ2 <= λ3
- curvature = λ1 / (λ1 + λ2 + λ3)
- planarity = (λ2 - λ1) / λ3
- linearity = (λ3 - λ2) / λ3
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def compute_geometry_features(xyz: np.ndarray, k: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute local geometric features using kNN + PCA.

    Parameters
    ----------
    xyz:
        Point coordinates, shape [N, 3], float-like.
    k:
        Number of neighbors for local PCA. Effective k is clamped to [3, N].

    Returns
    -------
    curvature:
        [N], local surface variation.
    planarity:
        [N], high on planar structures.
    linearity:
        [N], high on line-like structures (e.g., poles/wires).
    normals:
        [N, 3], estimated normal from eigenvector of smallest eigenvalue.

    Notes
    -----
    Robust handling:
    - If N == 0: return empty arrays with correct shapes.
    - If N < 3: returns zeros for scalar features and default z-up normals.
    - k is automatically clamped to avoid invalid neighbor queries.
    - Division denominators are protected by epsilon.
    """
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}")

    n = xyz.shape[0]
    eps = 1e-12

    if n == 0:
        return (
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )

    if n < 3:
        curvature = np.zeros((n,), dtype=np.float64)
        planarity = np.zeros((n,), dtype=np.float64)
        linearity = np.zeros((n,), dtype=np.float64)
        normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype=np.float64), (n, 1))
        return curvature, planarity, linearity, normals

    k_eff = int(np.clip(k, 3, n))
    tree = cKDTree(xyz)

    # dists: [N, k_eff], nn_idx: [N, k_eff]
    _, nn_idx = tree.query(xyz, k=k_eff)
    if nn_idx.ndim == 1:
        nn_idx = nn_idx[:, None]

    curvature = np.zeros((n,), dtype=np.float64)
    planarity = np.zeros((n,), dtype=np.float64)
    linearity = np.zeros((n,), dtype=np.float64)
    normals = np.zeros((n, 3), dtype=np.float64)

    for i in range(n):
        local_pts = xyz[nn_idx[i]]  # [k_eff, 3]
        mu = local_pts.mean(axis=0, keepdims=True)
        centered = local_pts - mu

        # Covariance: C = (X^T X) / (k_eff - 1), shape [3, 3]
        cov = (centered.T @ centered) / max(k_eff - 1, 1)

        # eigh for symmetric covariance; eigenvalues are ascending.
        evals, evecs = np.linalg.eigh(cov)
        lam1, lam2, lam3 = np.maximum(evals, 0.0)

        denom_sum = lam1 + lam2 + lam3 + eps
        denom_l3 = lam3 + eps

        # Formulas:
        # curvature = λ1 / (λ1 + λ2 + λ3)
        # planarity = (λ2 - λ1) / λ3
        # linearity = (λ3 - λ2) / λ3
        curvature[i] = lam1 / denom_sum
        planarity[i] = (lam2 - lam1) / denom_l3
        linearity[i] = (lam3 - lam2) / denom_l3

        # Normal is eigenvector corresponding to smallest eigenvalue λ1
        nrm = evecs[:, 0]
        norm = np.linalg.norm(nrm)
        normals[i] = nrm / (norm + eps)

    # Keep ranges stable against tiny negative numerical spill.
    curvature = np.clip(curvature, 0.0, 1.0)
    planarity = np.clip(planarity, 0.0, 1.0)
    linearity = np.clip(linearity, 0.0, 1.0)

    return curvature, planarity, linearity, normals


def _test_compute_geometry_features_random() -> None:
    """Small random-cloud smoke test.

    - builds a tiny random cloud [64, 3]
    - runs compute_geometry_features
    - checks shape and finite values
    """
    rng = np.random.default_rng(123)
    xyz = rng.normal(size=(64, 3))

    curvature, planarity, linearity, normals = compute_geometry_features(xyz, k=16)

    assert curvature.shape == (64,)
    assert planarity.shape == (64,)
    assert linearity.shape == (64,)
    assert normals.shape == (64, 3)
    assert np.isfinite(curvature).all()
    assert np.isfinite(planarity).all()
    assert np.isfinite(linearity).all()
    assert np.isfinite(normals).all()


if __name__ == "__main__":
    _test_compute_geometry_features_random()
    print("geometry_features random smoke test passed.")
