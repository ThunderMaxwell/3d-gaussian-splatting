"""Texture feature extraction from multi-view images for 3D points.

Given points, images, and camera parameters, this module computes per-point
texture richness indicators (e.g., image gradient magnitude, color variance).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from scipy.ndimage import sobel


@dataclass
class Camera:
    """Simple camera container.

    Attributes
    ----------
    K:
        Intrinsic matrix, shape [3, 3].
    T_wc:
        World-to-camera transform, shape [4, 4].
    image_size:
        (H, W).
    """

    K: np.ndarray
    T_wc: np.ndarray
    image_size: tuple[int, int]


@dataclass
class TextureFeatureConfig:
    gradient_clip_percentile: float = 95.0
    eps: float = 1e-9


def estimate_texture_features(
    points_xyz: np.ndarray,
    points_rgb: np.ndarray,
    images: List[np.ndarray],
    cameras: List[Camera],
    config: TextureFeatureConfig | None = None,
) -> Dict[str, np.ndarray]:
    """Estimate per-point texture features using multi-view projection.

    Parameters
    ----------
    points_xyz:
        [N, 3] float world coordinates.
    points_rgb:
        [N, 3] float or uint8 in RGB.
    images:
        List of RGB images, each [H, W, 3].
    cameras:
        Same length as images.
    config:
        Optional TextureFeatureConfig.

    Returns
    -------
    dict
        {
            "view_count": np.ndarray [N],
            "mean_grad": np.ndarray [N],
            "rgb_var": np.ndarray [N],
            "texture_score": np.ndarray [N],
        }
    """
    if config is None:
        config = TextureFeatureConfig()
    n = points_xyz.shape[0]

    grad_maps = [compute_gradient_map(img) for img in images]
    observed_grads = [[] for _ in range(n)]
    observed_colors = [[] for _ in range(n)]

    for img, gmap, cam in zip(images, grad_maps, cameras):
        uv, depth, valid = project_points(points_xyz, cam)
        uv_int = np.round(uv).astype(np.int32)
        h, w = img.shape[:2]

        inside = (
            valid
            & (uv_int[:, 0] >= 0)
            & (uv_int[:, 0] < w)
            & (uv_int[:, 1] >= 0)
            & (uv_int[:, 1] < h)
        )

        idx = np.where(inside)[0]
        for i in idx:
            x, y = uv_int[i]
            observed_grads[i].append(float(gmap[y, x]))
            observed_colors[i].append(img[y, x].astype(np.float32))

    view_count = np.array([len(v) for v in observed_grads], dtype=np.float32)
    mean_grad = np.array([np.mean(v) if len(v) > 0 else 0.0 for v in observed_grads], dtype=np.float32)
    rgb_var = np.array(
        [np.mean(np.var(np.stack(c, axis=0), axis=0)) if len(c) > 1 else 0.0 for c in observed_colors],
        dtype=np.float32,
    )

    mean_grad = clip_percentile(mean_grad, config.gradient_clip_percentile)

    texture_score = normalize_01(0.7 * normalize_01(mean_grad) + 0.3 * normalize_01(rgb_var))
    return {
        "view_count": view_count,
        "mean_grad": mean_grad,
        "rgb_var": rgb_var,
        "texture_score": texture_score,
    }


def project_points(points_xyz: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project world points to image plane.

    Parameters
    ----------
    points_xyz:
        [N, 3]
    camera:
        Camera object

    Returns
    -------
    uv:
        [N, 2] projected image coordinates.
    depth:
        [N] depth in camera frame.
    valid:
        [N] bool mask where depth > 0.
    """
    n = points_xyz.shape[0]
    xyz1 = np.concatenate([points_xyz, np.ones((n, 1), dtype=points_xyz.dtype)], axis=1)
    pc = (camera.T_wc @ xyz1.T).T[:, :3]
    depth = pc[:, 2]
    valid = depth > 1e-6

    uv_h = (camera.K @ pc.T).T
    uv = uv_h[:, :2] / np.maximum(uv_h[:, 2:3], 1e-6)
    return uv, depth, valid


def compute_gradient_map(image: np.ndarray) -> np.ndarray:
    """Compute Sobel gradient magnitude map.

    Input shape: [H, W, 3] or [H, W]
    Output shape: [H, W]
    """
    if image.ndim == 3:
        gray = 0.299 * image[..., 0] + 0.587 * image[..., 1] + 0.114 * image[..., 2]
    else:
        gray = image
    gx = sobel(gray, axis=1)
    gy = sobel(gray, axis=0)
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def clip_percentile(values: np.ndarray, percentile: float) -> np.ndarray:
    hi = np.percentile(values, percentile)
    return np.minimum(values, hi)


def normalize_01(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    vmin = float(values.min())
    vmax = float(values.max())
    return (values - vmin) / (vmax - vmin + eps)
