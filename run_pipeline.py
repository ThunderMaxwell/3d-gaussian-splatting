"""End-to-end demo pipeline for adaptive point-cloud downsampling for 3DGS init."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import open3d as o3d

from adaptive_sampler import AdaptiveSamplerConfig, sample_adaptively
from block_partition import BlockPartitionConfig, build_blocks
from gaussian_initializer import GaussianInitConfig, initialize_gaussians
from geometry_features import GeometryFeatureConfig, estimate_geometry_features
from texture_features import Camera, TextureFeatureConfig, estimate_texture_features


@dataclass
class PipelineInputs:
    points_xyzrgb: np.ndarray  # [N, 6]
    images: List[np.ndarray]  # each [H, W, 3]
    cameras: List[Camera]  # len == len(images)
    target_num_points: int


@dataclass
class PipelineOutputs:
    sampled_points_xyzrgb: np.ndarray  # [M, 6]
    selected_indices: np.ndarray  # [M]
    gaussian_params: dict


def run_adaptive_downsample_pipeline(inputs: PipelineInputs) -> PipelineOutputs:
    """Run geometry+texture aware adaptive sampling pipeline.

    Input Shapes
    ------------
    points_xyzrgb: [N, 6], images: list([H, W, 3]), target_num_points: int

    Output Shapes
    -------------
    sampled_points_xyzrgb: [M, 6], selected_indices: [M]
    """
    points_xyz = inputs.points_xyzrgb[:, :3]
    points_rgb = inputs.points_xyzrgb[:, 3:6]

    geom_feats = estimate_geometry_features(points_xyz, GeometryFeatureConfig())
    tex_feats = estimate_texture_features(
        points_xyz=points_xyz,
        points_rgb=points_rgb,
        images=inputs.images,
        cameras=inputs.cameras,
        config=TextureFeatureConfig(),
    )
    block_info = build_blocks(points_xyz, BlockPartitionConfig(voxel_size=0.5))

    sample_result = sample_adaptively(
        points_xyzrgb=inputs.points_xyzrgb,
        geometry_features=geom_feats,
        texture_features=tex_feats,
        block_info=block_info,
        target_num_points=inputs.target_num_points,
        config=AdaptiveSamplerConfig(),
    )

    gaussian_params = initialize_gaussians(
        sampled_points_xyzrgb=sample_result["sampled_points_xyzrgb"],
        config=GaussianInitConfig(),
    )

    return PipelineOutputs(
        sampled_points_xyzrgb=sample_result["sampled_points_xyzrgb"],
        selected_indices=sample_result["selected_indices"],
        gaussian_params=gaussian_params,
    )


def load_xyzrgb_pcd(pcd_path: str | Path) -> np.ndarray:
    """Load a colored point cloud from .ply/.pcd as [N, 6]."""
    pcd = o3d.io.read_point_cloud(str(pcd_path))
    xyz = np.asarray(pcd.points, dtype=np.float32)
    rgb = np.asarray(pcd.colors, dtype=np.float32)
    if rgb.size == 0:
        rgb = np.zeros_like(xyz, dtype=np.float32)
    return np.concatenate([xyz, rgb], axis=1)


def save_xyzrgb_ply(points_xyzrgb: np.ndarray, out_path: str | Path) -> None:
    """Save [N, 6] point cloud to .ply."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyzrgb[:, :3])
    colors = points_xyzrgb[:, 3:6]
    if colors.max() > 1.5:
        colors = colors / 255.0
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(str(out_path), pcd)


if __name__ == "__main__":
    # Pseudo demo. Replace cameras/images loading with your real dataset parser.
    n = 10000
    xyz = np.random.randn(n, 3).astype(np.float32)
    rgb = np.random.rand(n, 3).astype(np.float32)
    points_xyzrgb = np.concatenate([xyz, rgb], axis=1)

    dummy_img = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    dummy_cam = Camera(
        K=np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32),
        T_wc=np.eye(4, dtype=np.float32),
        image_size=(480, 640),
    )

    inputs = PipelineInputs(
        points_xyzrgb=points_xyzrgb,
        images=[dummy_img],
        cameras=[dummy_cam],
        target_num_points=2000,
    )
    outputs = run_adaptive_downsample_pipeline(inputs)
    print("Input points:", points_xyzrgb.shape[0])
    print("Sampled points:", outputs.sampled_points_xyzrgb.shape[0])
