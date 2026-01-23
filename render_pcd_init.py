# render_pcd_init_flythrough.py
# [修改版] 增加了下采样功能，减小 PLY 体积

import torch
import os
import sys
import numpy as np
import open3d as o3d
from argparse import ArgumentParser
import math
import time
from tqdm import tqdm

from scene import GaussianModel
from gaussian_renderer import render
from arguments import PipelineParams
from scipy.interpolate import CubicSpline

# ... (前面的 DummyInfo, BasicPointCloud, MiniCam, getProjectionMatrix, look_at, get_custom_path 类和函数保持不变，无需修改) ...
# 为了节省篇幅，这里省略了上面未改动的辅助类和函数，请保留你原文件中的这部分代码
# ... 

# ================= 辅助类 (保留原样) =================
class DummyInfo:
    def __init__(self, name):
        self.image_name = name

class BasicPointCloud:
    def __init__(self, points, colors, normals=None):
        self.points = points
        self.colors = colors
        self.normals = normals if normals is not None else np.zeros_like(points)

class MiniCam:
    def __init__(self, c2w, width, height, fov_deg=60.0, znear=0.01, zfar=100.0, name="init_view"):
        w2c = np.linalg.inv(c2w)
        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).cuda().float()
        self.FoVy = fov_deg * np.pi / 180
        self.FoVx = self.FoVy 
        self.projection_matrix = getProjectionMatrix(znear=znear, zfar=zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda().float()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        self.image_width = width
        self.image_height = height
        self.zfar = zfar
        self.znear = znear
        self.image_name = name

# ================= 数学辅助函数 (保留原样) =================
def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4)
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def look_at(eye, center, up):
    z_axis = (center - eye)
    norm_z = np.linalg.norm(z_axis)
    if norm_z == 0: return np.eye(4)
    z_axis = z_axis / norm_z
    
    x_axis = np.cross(up, z_axis)
    norm_x = np.linalg.norm(x_axis)
    if norm_x == 0: x_axis = np.array([1,0,0])
    else: x_axis = x_axis / norm_x
    
    y_axis = np.cross(z_axis, x_axis)
    
    mat = np.eye(4)
    mat[:3, 0] = x_axis
    mat[:3, 1] = y_axis
    mat[:3, 2] = z_axis
    mat[:3, 3] = eye
    return mat

def get_custom_path(points, num_frames=120):
    # 此处省略 get_custom_path 具体实现，保持你原文件内容不变
    # 请确保这里有你的 get_custom_path 函数
    # ...
    # 为了演示，我简单复制之前那个基本的 spline 逻辑占位，实际请用你上一版代码
    MY_POINTS = [
        [0.548535,  -0.765724,  0], 
        [9.840271,  -0.085796,  0], 
        [10.113586,  4.328422,  0], 
        [13.171944, 4.59255583,  0], 
        [13.038110, 8.036506,  0], 
        [25.687349,  8.733245,  0], 
    ]
    waypoints = np.array(MY_POINTS)
    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2
    fixed_z = center[2] 
    
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cum_dist = np.r_[0, np.cumsum(dists)]
    total_dist = cum_dist[-1]
    t_points = cum_dist / total_dist
    cs_x = CubicSpline(t_points, x, bc_type='natural')
    cs_y = CubicSpline(t_points, y, bc_type='natural')

    cams = []
    t_samples = np.linspace(0, 1, num_frames)
    for i, t in enumerate(t_samples):
        cur_x = float(cs_x(t))
        cur_y = float(cs_y(t))
        eye = np.array([cur_x, cur_y, fixed_z])
        look_ahead = 0.05
        t_target = min(t + look_ahead, 1.0)
        target_x = float(cs_x(t_target))
        target_y = float(cs_y(t_target))
        if t >= 1.0 - 1e-3:
             deriv_x = cs_x(t, 1)
             deriv_y = cs_y(t, 1)
             target_x = cur_x + deriv_x
             target_y = cur_y + deriv_y
        look_target = np.array([target_x, target_y, fixed_z])
        c2w = look_at(eye, look_target, up=np.array([0, 0, 1]))
        cams.append(c2w)
    return cams

# ================= 主逻辑 (关键修改区) =================
def main(args):
    # 1. 读取 PCD
    print(f"[{time.strftime('%H:%M:%S')}] Loading PCD from {args.input}...")
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return

    pcd = o3d.io.read_point_cloud(args.input)
    original_count = len(pcd.points)
    print(f"[Info] Original Point Count: {original_count}")

    # ================== 【新增】下采样逻辑 ==================
    if args.voxel_size > 0:
        print(f"[{time.strftime('%H:%M:%S')}] Downsampling with voxel_size={args.voxel_size}...")
        # voxel_down_sample 会同时对坐标和颜色进行平均
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        new_count = len(pcd.points)
        print(f"[Info] Downsampled Point Count: {new_count} (Reduction: {100*(1-new_count/original_count):.2f}%)")
    else:
        print("[Info] No downsampling applied.")
    # =======================================================

    pts = np.asarray(pcd.points)
    
    if not pcd.has_colors():
        print("[Warning] PCD has no color. Using default white.")
        clrs = np.ones_like(pts)
    else:
        clrs = np.asarray(pcd.colors) 
        if clrs.max() > 1.1:
            clrs = clrs / 255.0
            
    print(f"[{time.strftime('%H:%M:%S')}] Final points to initialize: {len(pts)}")

    # 2. 初始化 Gaussian Model
    print(f"[{time.strftime('%H:%M:%S')}] Initializing Gaussian Model...")
    gaussians = GaussianModel(sh_degree=0) 
    
    pcd_wrapper = BasicPointCloud(points=pts, colors=clrs, normals=None)
    dummy_infos = [DummyInfo("init_view")]
    
    gaussians.create_from_pcd(pcd_wrapper, dummy_infos, spatial_lr_scale=1.0)
    
    # 保存初始化后的 PLY (体积会变小)
    init_ply_path = os.path.join(args.output_dir, "init_gaussians.ply")
    os.makedirs(args.output_dir, exist_ok=True)
    gaussians.save_ply(init_ply_path)
    print(f"[Success] Initialized Gaussian PLY saved to {init_ply_path}")

    # 3. 准备渲染轨迹 (如果你只想要 PLY，其实后面这些可以注释掉，但为了保持完整功能还是留着)
    print(f"[{time.strftime('%H:%M:%S')}] Preparing FLY-THROUGH path...")
    
    # 这里的轨迹生成使用的是下采样后的点云来计算边界，但这不影响，
    # 因为下采样后的 BBox 和原始点云几乎是一样的。
    orbit_c2ws = get_custom_path(pts, num_frames=args.frames)

    bg_color = [1, 1, 1] if args.white_bg else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    pipeline = PipelineParams(ArgumentParser())

    save_dir = os.path.join(args.output_dir, "render_frames")
    os.makedirs(save_dir, exist_ok=True)

    # 4. 循环渲染
    print(f"[{time.strftime('%H:%M:%S')}] Start Rendering {args.frames} frames...")
    import torchvision

    for idx, c2w in enumerate(tqdm(orbit_c2ws, desc="Rendering Path", unit="frame")):
        cam = MiniCam(c2w=c2w, width=args.width, height=args.height, name="init_view")
        
        with torch.no_grad():
            render_pkg = render(cam, gaussians, pipeline, background)
            image = render_pkg["render"]
            
            save_path = os.path.join(save_dir, f"{idx:04d}.png")
            torchvision.utils.save_image(image, save_path)

    print(f"\n[Success] All frames saved to {save_dir}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Render PCD using 3DGS initialization")
    parser.add_argument("--input", type=str, required=True, help="Path to .pcd file")
    parser.add_argument("--output_dir", type=str, default="output/flythrough_check", help="Directory to save results")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--white_bg", action="store_true", help="Use white background")
    
    # ================== 【新增】参数 ==================
    parser.add_argument("--voxel_size", type=float, default=0.0, help="Voxel size for downsampling (e.g., 0.05 or 0.1). 0 means no downsampling.")
    # =================================================
    
    args = parser.parse_args()
    main(args)