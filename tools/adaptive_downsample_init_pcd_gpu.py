#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
分区域下采样脚本（适合“着色后的激光点云 -> 3DGS 初始化”流程）
增强版：
1. 法向量估计优先使用显卡（GPU）
2. 若显卡不可用或 Open3D 不支持，则自动回退到中央处理器（CPU）
3. 去掉当前流程里通常不必要的全局法向一致化，以减少耗时
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
import open3d as o3d
from tqdm import tqdm


@dataclass
class RegionConfig:
    plane_voxel: float = 0.15
    edge_voxel: float = 0.01
    thin_voxel: float = 0.01
    other_voxel: float = 0.03
    nb_neighbors_outlier: int = 20
    std_ratio_outlier: float = 2.0
    normal_radius: float = 0.30
    normal_max_nn: int = 30
    feature_knn: int = 24
    normal_knn: int = 16
    plane_curvature_max: float = 0.018
    plane_normal_disagreement_max: float = 0.08
    plane_planarity_min: float = 0.45
    edge_curvature_min: float = 0.035
    edge_normal_disagreement_min: float = 0.18
    thin_linearity_min: float = 0.78
    thin_planarity_max: float = 0.22
    thin_curvature_max: float = 0.08
    min_points_to_downsample: int = 200
    final_dedup_voxel: float = 0.005
    use_gpu_normals: bool = True
    gpu_device: str = 'CUDA:0'


@dataclass
class FeaturePack:
    curvature: np.ndarray
    linearity: np.ndarray
    planarity: np.ndarray
    normal_disagreement: np.ndarray


LABEL_PLANE = 0
LABEL_EDGE = 1
LABEL_THIN = 2
LABEL_OTHER = 3
LABEL_NAME = {0: 'plane', 1: 'edge', 2: 'thin', 3: 'other'}


def read_point_cloud(path: str) -> o3d.geometry.PointCloud:
    pcd = o3d.io.read_point_cloud(path)
    if len(pcd.points) == 0:
        raise RuntimeError(f'点云为空：{path}')
    return pcd


def ensure_color_range_01(colors: np.ndarray) -> np.ndarray:
    if colors.size == 0:
        return colors
    return colors / 255.0 if colors.max() > 1.1 else colors


def remove_outliers(pcd, cfg):
    print('[1/6] 统计离群点去除...')
    t0 = time.time()
    clean_pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=cfg.nb_neighbors_outlier,
        std_ratio=cfg.std_ratio_outlier
    )
    print(f'    原始点数: {len(pcd.points)}')
    print(f'    去噪后点数: {len(clean_pcd.points)}')
    print(f'    耗时: {time.time() - t0:.2f}s')
    return clean_pcd


def _try_gpu_estimate_normals(pcd, cfg):
    """
    使用 Open3D 张量接口在 GPU 上估计法向量。
    返回：
        success: bool
        msg: str
    """
    if not hasattr(o3d, 't'):
        return False, '当前 Open3D 不包含张量接口（o3d.t）'
    if not hasattr(o3d, 'core'):
        return False, '当前 Open3D 不包含核心张量模块（o3d.core）'

    try:
        cuda_ok = False
        if hasattr(o3d.core, 'cuda') and hasattr(o3d.core.cuda, 'is_available'):
            cuda_ok = bool(o3d.core.cuda.is_available())
        if not cuda_ok:
            return False, '当前 Open3D 未检测到可用 GPU'

        device = o3d.core.Device(cfg.gpu_device)
        t_pcd = o3d.t.geometry.PointCloud.from_legacy(
            pcd,
            dtype=o3d.core.float32,
            device=device,
        )

        # Open3D 官方更推荐在 GPU 上使用“半径 + 最大近邻数”的混合搜索。
        t_pcd.estimate_normals(
            max_nn=cfg.normal_max_nn,
            radius=cfg.normal_radius,
        )

        normals = t_pcd.point.normals.cpu().numpy()
        pcd.normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
        return True, f'已使用 GPU({cfg.gpu_device}) 完成法向量估计'
    except Exception as e:
        return False, f'GPU 法向量估计失败：{e}'



def estimate_normals(pcd, cfg):
    print('[2/6] 估计法向量...')
    t0 = time.time()

    if cfg.use_gpu_normals:
        ok, msg = _try_gpu_estimate_normals(pcd, cfg)
        if ok:
            print(f'    {msg}')
            print('    已跳过全局法向一致化（当前流程通常不需要）')
            print(f'    耗时: {time.time() - t0:.2f}s')
            return
        else:
            print(f'    [提示] {msg}')
            print('    自动回退到 CPU 法向量估计...')

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.normal_radius,
            max_nn=cfg.normal_max_nn,
        )
    )
    # 这里不做 orient_normals_consistent_tangent_plane。
    # 因为后续 normal_disagreement 使用的是 abs(dot)，
    # 全局正负朝向一致性对当前分类流程通常不是必须条件。
    print('    已使用 CPU 完成法向量估计')
    print('    已跳过全局法向一致化（当前流程通常不需要）')
    print(f'    耗时: {time.time() - t0:.2f}s')



def _safe_eigvals(cov: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(np.maximum(eigvals, 1e-12))[::-1]
    return eigvals



def compute_features(pcd, cfg):
    print('[3/6] 计算局部几何特征（可能稍慢，这是正常的）...')
    t0 = time.time()
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    n = len(points)
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    curvature = np.zeros(n, dtype=np.float32)
    linearity = np.zeros(n, dtype=np.float32)
    planarity = np.zeros(n, dtype=np.float32)
    normal_disagreement = np.zeros(n, dtype=np.float32)

    for i in tqdm(range(n), desc='    特征计算', unit='点'):
        _, idx, _ = kdtree.search_knn_vector_3d(points[i], cfg.feature_knn)
        idx = np.asarray(idx, dtype=np.int64)
        nbr_pts = points[idx]
        center = nbr_pts.mean(axis=0, keepdims=True)
        diffs = nbr_pts - center
        cov = diffs.T @ diffs / max(len(idx) - 1, 1)
        lam1, lam2, lam3 = _safe_eigvals(cov)
        denom = lam1 + lam2 + lam3 + 1e-12
        curvature[i] = lam3 / denom
        linearity[i] = (lam1 - lam2) / (lam1 + 1e-12)
        planarity[i] = (lam2 - lam3) / (lam1 + 1e-12)

        _, n_idx, _ = kdtree.search_knn_vector_3d(points[i], cfg.normal_knn)
        n_idx = np.asarray(n_idx, dtype=np.int64)
        nbr_normals = normals[n_idx]
        dots = np.abs(nbr_normals @ normals[i])
        normal_disagreement[i] = 1.0 - np.mean(np.clip(dots, 0.0, 1.0))

    print(f'    耗时: {time.time() - t0:.2f}s')
    return FeaturePack(curvature, linearity, planarity, normal_disagreement)



def classify_regions(pcd, feat, cfg):
    print('[4/6] 按几何特征分类区域...')
    labels = np.full(len(pcd.points), LABEL_OTHER, dtype=np.int32)

    thin_mask = (
        (feat.linearity >= cfg.thin_linearity_min)
        & (feat.planarity <= cfg.thin_planarity_max)
        & (feat.curvature <= cfg.thin_curvature_max)
    )
    labels[thin_mask] = LABEL_THIN

    edge_mask = (
        ((feat.curvature >= cfg.edge_curvature_min)
         | (feat.normal_disagreement >= cfg.edge_normal_disagreement_min))
        & (~thin_mask)
    )
    labels[edge_mask] = LABEL_EDGE

    plane_mask = (
        (feat.curvature <= cfg.plane_curvature_max)
        & (feat.normal_disagreement <= cfg.plane_normal_disagreement_max)
        & (feat.planarity >= cfg.plane_planarity_min)
        & (~thin_mask)
        & (~edge_mask)
    )
    labels[plane_mask] = LABEL_PLANE
    return labels



def subset_pcd(pcd, mask):
    idx = np.flatnonzero(mask)
    return pcd.select_by_index(idx.tolist())



def downsample_region(name, pcd, voxel, cfg):
    n = len(pcd.points)
    if n == 0:
        return pcd
    if n < cfg.min_points_to_downsample or voxel <= 0:
        print(f'    {name:>6s}: 点数 {n}，直接保留')
        return pcd
    down = pcd.voxel_down_sample(voxel)
    print(f'    {name:>6s}: {n:8d} -> {len(down.points):8d}  (voxel={voxel:.3f})')
    return down



def merge_point_clouds(pcd_list):
    merged = o3d.geometry.PointCloud()
    for p in pcd_list:
        merged += p
    return merged



def colorize_labels(points, labels):
    color_map = {
        LABEL_PLANE: np.array([0.0, 0.8, 0.0]),
        LABEL_EDGE: np.array([1.0, 0.2, 0.2]),
        LABEL_THIN: np.array([0.2, 0.4, 1.0]),
        LABEL_OTHER: np.array([0.8, 0.8, 0.8]),
    }
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.stack([color_map[int(x)] for x in labels], axis=0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd



def save_stats(out_dir, cfg, raw_count, clean_count, labels, final_count):
    stats = {
        'config': asdict(cfg),
        'raw_count': int(raw_count),
        'clean_count': int(clean_count),
        'label_counts': {LABEL_NAME[k]: int((labels == k).sum()) for k in LABEL_NAME},
        'final_count': int(final_count),
        'reduction_from_raw': float(1.0 - final_count / max(raw_count, 1)),
        'reduction_from_clean': float(1.0 - final_count / max(clean_count, 1)),
    }
    with open(os.path.join(out_dir, 'downsample_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)



def run(args):
    cfg = RegionConfig(
        plane_voxel=args.plane_voxel,
        edge_voxel=args.edge_voxel,
        thin_voxel=args.thin_voxel,
        other_voxel=args.other_voxel,
        nb_neighbors_outlier=args.nb_neighbors_outlier,
        std_ratio_outlier=args.std_ratio_outlier,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        feature_knn=args.feature_knn,
        normal_knn=args.normal_knn,
        plane_curvature_max=args.plane_curvature_max,
        plane_normal_disagreement_max=args.plane_normal_disagreement_max,
        plane_planarity_min=args.plane_planarity_min,
        edge_curvature_min=args.edge_curvature_min,
        edge_normal_disagreement_min=args.edge_normal_disagreement_min,
        thin_linearity_min=args.thin_linearity_min,
        thin_planarity_max=args.thin_planarity_max,
        thin_curvature_max=args.thin_curvature_max,
        min_points_to_downsample=args.min_points_to_downsample,
        final_dedup_voxel=args.final_dedup_voxel,
        use_gpu_normals=(not args.cpu_normals),
        gpu_device=args.gpu_device,
    )

    os.makedirs(args.output_dir, exist_ok=True)
    print(f'读取点云: {args.input}')
    pcd = read_point_cloud(args.input)
    raw_count = len(pcd.points)

    if not pcd.has_colors():
        print('[警告] 输入点云没有颜色，将自动设为白色。')
        points = np.asarray(pcd.points)
        pcd.colors = o3d.utility.Vector3dVector(np.ones_like(points))
    else:
        colors = ensure_color_range_01(np.asarray(pcd.colors))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    clean_pcd = remove_outliers(pcd, cfg)
    clean_count = len(clean_pcd.points)
    estimate_normals(clean_pcd, cfg)
    feat = compute_features(clean_pcd, cfg)
    labels = classify_regions(clean_pcd, feat, cfg)
    points = np.asarray(clean_pcd.points)

    print('[5/6] 分区下采样...')
    out_parts = []
    for label, voxel in [
        (LABEL_PLANE, cfg.plane_voxel),
        (LABEL_EDGE, cfg.edge_voxel),
        (LABEL_THIN, cfg.thin_voxel),
        (LABEL_OTHER, cfg.other_voxel),
    ]:
        mask = labels == label
        sub = subset_pcd(clean_pcd, mask)
        down = downsample_region(LABEL_NAME[label], sub, voxel, cfg)
        out_parts.append(down)

    merged = merge_point_clouds(out_parts)
    if cfg.final_dedup_voxel > 0:
        merged = merged.voxel_down_sample(cfg.final_dedup_voxel)

    print('[6/6] 保存结果...')
    out_ply = os.path.join(args.output_dir, args.output_name)
    o3d.io.write_point_cloud(out_ply, merged, write_ascii=False)
    print(f'    下采样点云已保存: {out_ply}')

    label_pcd = colorize_labels(points, labels)
    label_path = os.path.join(args.output_dir, 'region_labels_debug.ply')
    o3d.io.write_point_cloud(label_path, label_pcd, write_ascii=False)
    print(f'    区域分类调试点云已保存: {label_path}')

    save_stats(args.output_dir, cfg, raw_count, clean_count, labels, len(merged.points))
    print(f"    统计信息已保存: {os.path.join(args.output_dir, 'downsample_stats.json')}")

    print('\n===== 汇总 =====')
    print(f'原始点数      : {raw_count}')
    print(f'去噪后点数    : {clean_count}')
    print(f'最终点数      : {len(merged.points)}')
    print(f'相对原始压缩比: {(1 - len(merged.points) / max(raw_count, 1)) * 100:.2f}%')
    print(f'相对去噪压缩比: {(1 - len(merged.points) / max(clean_count, 1)) * 100:.2f}%')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='适合 3DGS 初始化的分区域下采样（GPU 法向量估计增强版）')
    parser.add_argument('--input', type=str, required=True, help='输入点云（PLY/PCD）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出目录')
    parser.add_argument('--output_name', type=str, default='init_downsampled.ply', help='输出点云文件名')

    parser.add_argument('--plane_voxel', type=float, default=0.15)
    parser.add_argument('--edge_voxel', type=float, default=0.06)
    parser.add_argument('--thin_voxel', type=float, default=0.03)
    parser.add_argument('--other_voxel', type=float, default=0.08)

    parser.add_argument('--nb_neighbors_outlier', type=int, default=20)
    parser.add_argument('--std_ratio_outlier', type=float, default=2.0)

    parser.add_argument('--normal_radius', type=float, default=0.30)
    parser.add_argument('--normal_max_nn', type=int, default=30)
    parser.add_argument('--feature_knn', type=int, default=24)
    parser.add_argument('--normal_knn', type=int, default=16)

    parser.add_argument('--plane_curvature_max', type=float, default=0.018)
    parser.add_argument('--plane_normal_disagreement_max', type=float, default=0.08)
    parser.add_argument('--plane_planarity_min', type=float, default=0.45)
    parser.add_argument('--edge_curvature_min', type=float, default=0.035)
    parser.add_argument('--edge_normal_disagreement_min', type=float, default=0.18)
    parser.add_argument('--thin_linearity_min', type=float, default=0.78)
    parser.add_argument('--thin_planarity_max', type=float, default=0.22)
    parser.add_argument('--thin_curvature_max', type=float, default=0.08)

    parser.add_argument('--min_points_to_downsample', type=int, default=200)
    parser.add_argument('--final_dedup_voxel', type=float, default=0.005)

    parser.add_argument('--cpu_normals', action='store_true', help='强制使用 CPU 估计法向量')
    parser.add_argument('--gpu_device', type=str, default='CUDA:0', help='GPU 设备名，例如 CUDA:0')

    args = parser.parse_args()
    run(args)
