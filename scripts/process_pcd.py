import open3d as o3d
import numpy as np
import os

# =================配置区域=================
input_pcd_path = "/home/ysc/3dgs-npy/data/taijie/other/color_SS_0.002.pcd"
output_ply_path = "/home/ysc/3dgs-npy/data/taijie/sparse/0/points3D.ply"

# 体素大小 (单位: 米，假设你的点云单位是米)
# 建议先试 0.05 (5cm) 或 0.1 (10cm)。
# 目标是把点数降到 200万左右。
VOXEL_SIZE = 0.05 #400万
# ========================================

def pcd_to_ply_downsampled(input_path, output_path, voxel_size):
    print(f"1. 正在读取点云: {input_path}")
    pcd = o3d.io.read_point_cloud(input_path)
    original_count = len(pcd.points)
    print(f"   原始点数: {original_count}")

    # --- 核心步骤：下采样 ---
    print(f"2. 正在进行体素下采样 (Voxel Size = {voxel_size})...")
    downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    new_count = len(downpcd.points)
    print(f"   下采样后点数: {new_count} (约为原图的 {new_count/original_count:.2%})")

    if new_count > 5000000:
        print("⚠️ 警告：点数依然超过 500万，显存可能还是会爆！建议调大 VOXEL_SIZE。")
    
    # --- 补全属性 (颜色 & 法向量) ---
    # 颜色
    if not downpcd.has_colors():
        print("   补全随机颜色...")
        points = np.asarray(downpcd.points)
        colors = np.random.rand(len(points), 3)
        downpcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 法向量 (3DGS 强制要求)
    if not downpcd.has_normals():
        print("   补全虚拟法向量 (全0)...")
        points_count = len(downpcd.points)
        normals = np.zeros((points_count, 3))
        downpcd.normals = o3d.utility.Vector3dVector(normals)

    # --- 保存 ---
    print(f"3. 正在保存为 PLY: {output_path}")
    o3d.io.write_point_cloud(output_path, downpcd, write_ascii=False)
    print("✅ 处理完成！")

if __name__ == "__main__":
    pcd_to_ply_downsampled(input_pcd_path, output_ply_path, VOXEL_SIZE)