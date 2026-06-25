"""Adaptive sampling core logic.

Combines geometry and texture scores with block-aware budget allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class AdaptiveSamplerConfig:
    w_geometry: float = 0.6
    w_texture: float = 0.4
    min_points_per_block: int = 8
    eps: float = 1e-9


def sample_adaptively(
    points_xyzrgb: np.ndarray,
    geometry_features: Dict[str, np.ndarray],
    texture_features: Dict[str, np.ndarray],
    block_info: Dict[str, np.ndarray],
    target_num_points: int,
    config: AdaptiveSamplerConfig | None = None,
) -> Dict[str, np.ndarray]:
    """Adaptive downsampling with block-level quota.

    Parameters
    ----------
    points_xyzrgb:
        [N, 6] float array.
    geometry_features:
        Must include "geometry_score" [N].
    texture_features:
        Must include "texture_score" [N].
    block_info:
        Output of block_partition.build_blocks.
    target_num_points:
        Desired number of sampled points.

    Returns
    -------
    dict
        {
            "selected_indices": [M] int index in original cloud,
            "sampled_points_xyzrgb": [M, 6] float,
            "importance_score": [N] float,
        }
    """
    if config is None:
        config = AdaptiveSamplerConfig()

    n = points_xyzrgb.shape[0]
    if target_num_points >= n:
        idx = np.arange(n, dtype=np.int32)
        return {
            "selected_indices": idx,
            "sampled_points_xyzrgb": points_xyzrgb,
            "importance_score": np.ones(n, dtype=np.float32),
        }

    g = geometry_features["geometry_score"]
    t = texture_features["texture_score"]
    importance = normalize_01(config.w_geometry * g + config.w_texture * t)

    block_ids = block_info["block_ids"]
    num_blocks = int(block_info["num_blocks"])

    block_mass = np.zeros(num_blocks, dtype=np.float64)
    for b in range(num_blocks):
        mask = block_ids == b
        block_mass[b] = float(np.sum(importance[mask])) + config.eps

    raw_quota = block_mass / block_mass.sum() * target_num_points
    quotas = np.floor(raw_quota).astype(np.int32)
    quotas = np.maximum(quotas, config.min_points_per_block)

    # Prevent over-allocation when block count is large.
    if quotas.sum() > target_num_points:
        scale = target_num_points / float(quotas.sum())
        quotas = np.maximum(1, np.floor(quotas * scale).astype(np.int32))

    # Distribute remaining slots by largest remainder.
    remain = target_num_points - int(quotas.sum())
    if remain > 0:
        remainders = raw_quota - np.floor(raw_quota)
        order = np.argsort(remainders)[::-1]
        for b in order[:remain]:
            quotas[b] += 1

    selected = []
    rng = np.random.default_rng(42)
    for b in range(num_blocks):
        idx = np.where(block_ids == b)[0]
        if len(idx) == 0:
            continue

        k = int(min(quotas[b], len(idx)))
        p = importance[idx] + config.eps
        p = p / p.sum()
        pick = rng.choice(idx, size=k, replace=False, p=p)
        selected.append(pick)

    selected_indices = np.concatenate(selected, axis=0) if selected else np.array([], dtype=np.int32)

    # Exact budget adjustment.
    if len(selected_indices) > target_num_points:
        order = np.argsort(importance[selected_indices])[::-1]
        selected_indices = selected_indices[order[:target_num_points]]
    elif len(selected_indices) < target_num_points:
        missing = target_num_points - len(selected_indices)
        unselected_mask = np.ones(n, dtype=bool)
        unselected_mask[selected_indices] = False
        candidates = np.where(unselected_mask)[0]
        if len(candidates) > 0:
            order = np.argsort(importance[candidates])[::-1]
            selected_indices = np.concatenate([selected_indices, candidates[order[:missing]]])

    return {
        "selected_indices": np.asarray(selected_indices, dtype=np.int32),
        "sampled_points_xyzrgb": points_xyzrgb[selected_indices],
        "importance_score": importance.astype(np.float32),
    }


def normalize_01(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    return (values - vmin) / (vmax - vmin + eps)
