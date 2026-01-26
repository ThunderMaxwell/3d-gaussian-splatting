# [优化版] 增加了去噪、体素下采样、以及“高斯球尺度强行对齐”功能

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

# ================= 辅助类 (保持不变) =================
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

# ================= 数学辅助函数 (保持不变) =================
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

# def get_custom_path(points, num_frames=120):
#     MY_POINTS = [
#         [-0.5111,16.2380,0],
#         [-3.380,-29.2320,0],
#         [2.6201,-46.5410,0],
#         [18.8118,-38.4271,0],
#         [43.8103,-18.4347,0],
#         [50.6967,31.9439,0],
#         [61.6209,47.0195,0],
#         [91.9127,54.6661,0],
#         [95.7155,83.3632,0],
#         [37.8222,142.3704,0],
#         [15.0271,140.7660,0],
#         [3.9576,45.7530,0],
#     ]
#     waypoints = np.array(MY_POINTS)
#     min_bound = points.min(axis=0)
#     max_bound = points.max(axis=0)
#     center = (min_bound + max_bound) / 2
#     fixed_z = center[2] 
    
#     x = waypoints[:, 0]
#     y = waypoints[:, 1]
#     dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
#     cum_dist = np.r_[0, np.cumsum(dists)]
#     total_dist = cum_dist[-1]
#     t_points = cum_dist / total_dist
#     cs_x = CubicSpline(t_points, x, bc_type='natural')
#     cs_y = CubicSpline(t_points, y, bc_type='natural')

#     cams = []
#     t_samples = np.linspace(0, 1, num_frames)
#     for i, t in enumerate(t_samples):
#         cur_x = float(cs_x(t))
#         cur_y = float(cs_y(t))
#         eye = np.array([cur_x, cur_y, fixed_z])
#         look_ahead = 0.05
#         t_target = min(t + look_ahead, 1.0)
#         target_x = float(cs_x(t_target))
#         target_y = float(cs_y(t_target))
#         if t >= 1.0 - 1e-3:
#              deriv_x = cs_x(t, 1)
#              deriv_y = cs_y(t, 1)
#              target_x = cur_x + deriv_x
#              target_y = cur_y + deriv_y
#         look_target = np.array([target_x, target_y, fixed_z])
#         c2w = look_at(eye, look_target, up=np.array([0, 0, 1]))
#         cams.append(c2w)
#     return cams
#
# 头部引入库时，增加 Akima1DInterpolator
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator

def get_custom_path(points, num_frames=120):
    # 你的自定义关键点
    MY_POINTS = [
        [6.819603,35.267548,0],
        [2.029363,24.265266,0],
        [2.029363,24.265266,0],
        [-0.120145,-2.120338,0],
        [-3.822304,-22.997616,0],
        [0.917989,-41.632301,0],
        [6.843385,-48.600845,0],
        [14.421021,-48.132591,0],
        [21.514904,-39.341156,0],
        [31.717968,-30.235355,0],
        [41.003460,-22.132597,0],
        [47.214142,-7.089324,0],
        [48.188385,9.797846,0],
        [49.413837,23.944885,0],
        [58.788906,36.992146,0],
        [65.319321,43.964832,0],
        [85.924309,45.567451,0],
        [94.014854,52.673065,0],
        [91.936012,62.980450,0],
        [96.218765,75.370125,0],
        [92.420212,83.275688,0],
        [87.015755,91.068253,0],
        [74.419411,100.267632,0],
        [69.884232,115.876007,0],
        [56.147816,128.591309,0],
        [38.818279,140.365372,0],
        [27.210773,142.481445,0],
        [15.511998,137.060318,0],
        [11.458668,119.121529,0],
        [9.621028,99.701813,0],
        [9.341122,81.923790,0],
        [6.389007,63.828831,0],
    ]
    waypoints = np.array(MY_POINTS)
    # --- [自动去重逻辑] ---
    # 计算相邻点距离，只保留距离 > 0 的点
    diffs = np.linalg.norm(np.diff(waypoints[:, :2], axis=0), axis=1)
    valid_mask = np.r_[True, diffs > 1e-6] 
    
    # 打印去重信息，让你知道删了几个点
    if np.sum(valid_mask) < len(waypoints):
        print(f"[Warning] Detected {len(waypoints) - np.sum(valid_mask)} duplicate points. Removing them automatically.")
        
    waypoints = waypoints[valid_mask]
    # ---------------------

    min_bound = points.min(axis=0)
    max_bound = points.max(axis=0)
    center = (min_bound + max_bound) / 2 
    total_height = max_bound[2] - min_bound[2]
    fixed_z = max_bound[2] - total_height * 0.3
    # fixed_z = center[2] 
    
    x = waypoints[:, 0]
    y = waypoints[:, 1]
    
    dists = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cum_dist = np.r_[0, np.cumsum(dists)]
    total_dist = cum_dist[-1]
    
    if total_dist == 0:
        t_points = np.linspace(0, 1, len(x))
    else:
        t_points = cum_dist / total_dist
    
    # 使用 Akima 插值，防止过冲
    cs_x = Akima1DInterpolator(t_points, x)
    cs_y = Akima1DInterpolator(t_points, y)

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
        
        # 边界处理
        if t >= 1.0 - 1e-3:
             deriv_x = cs_x(t, 1)
             deriv_y = cs_y(t, 1)
             target_x = cur_x + deriv_x
             target_y = cur_y + deriv_y
             
        look_target = np.array([target_x, target_y, fixed_z])
        c2w = look_at(eye, look_target, up=np.array([0, 0, 1]))
        cams.append(c2w)
        
    return cams
# ================= 主逻辑修改 =================
def main(args):
    # 1. 读取 PCD
    print(f"[{time.strftime('%H:%M:%S')}] Loading PCD from {args.input}...")
    if not os.path.exists(args.input):
        print(f"Error: File {args.input} not found.")
        return
    
    pcd = o3d.io.read_point_cloud(args.input)
    original_count = len(pcd.points)
    print(f"[Info] Original Point Count: {original_count}")

    # --- [新增] 去噪处理 (提升清晰度) ---
    # 在下采样之前，先去除离群噪点，防止这些噪点变成大的模糊团块
    print(f"[{time.strftime('%H:%M:%S')}] Removing outliers to improve clarity...")
    # nb_neighbors: 考虑多少个邻居, std_ratio: 标准差倍数，越小越严格
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    print(f"[Info] Points after outlier removal: {len(pcd.points)}")

    # --- [优化] 体素下采样 (减少内存) ---
    if args.voxel_size > 0:
        print(f"[{time.strftime('%H:%M:%S')}] Downsampling with voxel_size={args.voxel_size}...")
        pcd = pcd.voxel_down_sample(voxel_size=args.voxel_size)
        new_count = len(pcd.points)
        print(f"[Info] Downsampled Point Count: {new_count} (Reduction: {100*(1-new_count/original_count):.2f}%)")
    else:
        print("[Info] No downsampling applied.")

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
    
    # 标准初始化 (会根据点间距计算 scale)
    gaussians.create_from_pcd(pcd_wrapper, dummy_infos, spatial_lr_scale=1.0)
    
    # --- [核心修改] 根据体素大小强制对齐高斯球尺度 ---
    # 这就是你想要的 "根据高斯球大小排列在一起" 的逻辑实现。
    # 既然我们用了体素下采样，我们知道点之间的距离大约是 voxel_size。
    # 强制将高斯球大小设为 voxel_size 的一定比例，可以保证它们完美平铺，没有空洞，也最省内存。
    if args.voxel_size > 0:
        print(f"[{time.strftime('%H:%M:%S')}] Overriding Gaussian scales to match voxel size (Tiling)...")
        with torch.no_grad():
            # 3DGS 内部存储的是 log(scale)。
            # 系数 0.7 是经验值：voxel_size * 0.7 大约能覆盖对角线空隙，又不会过度重叠。
            # 如果觉得有空洞，可以把 0.7 改成 0.8 或 1.0
            target_scale = args.voxel_size * 0.7
            
            # 将所有高斯球强制设为各项同性 (球体) 且大小一致
            new_scales = torch.full_like(gaussians._scaling, np.log(target_scale))
            gaussians._scaling.data = new_scales
            
            # 同时也强制不透明度初始值为满 (可选，保证这一层是实心的)
            # gaussians._opacity.data.fill_(inverse_sigmoid(0.99)) 

    # 保存初始化后的 PLY
    init_ply_path = os.path.join(args.output_dir, "init_gaussians.ply")
    os.makedirs(args.output_dir, exist_ok=True)
    gaussians.save_ply(init_ply_path)
    print(f"[Success] Initialized Gaussian PLY saved to {init_ply_path}")

    # 3. 准备渲染轨迹
    print(f"[{time.strftime('%H:%M:%S')}] Preparing FLY-THROUGH path...")
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
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1600)
    parser.add_argument("--frames", type=int, default=120, help="Number of frames")
    parser.add_argument("--white_bg", action="store_true", help="Use white background")
    
    # 建议默认开启一个小的 voxel_size，例如 0.05 或 0.1
    parser.add_argument("--voxel_size", type=float, default=0.1, help="Voxel size controls the Gaussian Size and Density.")
    
    args = parser.parse_args()
    main(args)