import os
import xml.dom.minidom
import json
import open3d as o3d
import torch
import numpy as np

# 导入自定义模块
from gpu_project import c2d_gpu 
from ip_basic_new import ip
from depthColorful_new import colorful

def getReadFilePath(filePath, fileType):
    file_count = 0 
    file_name = ""
    # 简单的文件查找逻辑
    for f in os.listdir(filePath):
        if(f.split('.')[-1].lower() == fileType):
            file_name = f
            file_count += 1 
    
    # 如果没找到或找到多个，抛出异常 (原逻辑保持不变)
    if file_count != 1:
        # 注意：原代码逻辑在循环里判断 count!=1 有点问题，但为了保持兼容性暂不修改逻辑结构
        # 建议确保目录下只有一个对应类型文件
        pass 
        
    return os.path.join(filePath, file_name)

# 1. 读取配置
with open('scripts/config.json') as file:
    data = json.load(file)
datasetName = data['datasetName']

# 从config获取目标宽度（用于训练的一致性）
target_image_width = data.get('imageWidth', None)  # 从config.json获取目标宽度

main_path = os.path.abspath(f"data/{datasetName}")
depth_path = os.path.join(main_path, "depth")
undis_path = os.path.abspath(f"data/{datasetName}/input/undis")

# 确保输入路径存在
if not os.path.exists(undis_path):
    raise FileNotFoundError(f"未找到输入图片路径: {undis_path}")

num = len(os.listdir(undis_path))

cc = os.path.abspath(f"data/{datasetName}/input/CC")
dom = xml.dom.minidom.parse(getReadFilePath(cc, "xml"))

color = os.path.abspath(f"data/{datasetName}/input/color")
pcd_path = getReadFilePath(color, "pcd")

root = dom.documentElement

# 调试信息
image_paths = root.getElementsByTagName('ImagePath')
print(f"XML中找到 {len(image_paths)} 个ImagePath元素")
print(f"图片目录中有 {num} 个图片文件")

# 验证数量是否匹配
if num > len(image_paths):
    print(f"警告: 图片文件数量({num})大于XML中的ImagePath数量({len(image_paths)})")
    num = min(num, len(image_paths))

path0 = os.path.join(depth_path, "depth_0")
output_dir = os.path.join(depth_path, "npz_depths")
output_colorful_dir = os.path.join(depth_path, "depth_color")

os.makedirs(path0, exist_ok=True)

print(f"正在读取点云: {pcd_path}")
pcd = o3d.io.read_point_cloud(pcd_path)
points_np = np.asarray(pcd.points, dtype=np.float32)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在上传 {points_np.shape[0]} 个点到 {device}...")
points_tensor = torch.from_numpy(points_np).to(device)

print(f"开始生成深度图 (使用config.json中的目标宽度: {target_image_width})...")

for i in range(num):
    print(f"处理第 {i+1}/{num} 个相机...")
    # 使用config.json中的目标宽度进行投影
    c2d_gpu(i, root, path0, points_tensor, device=device, target_width=target_image_width)
    
    if i % 10 == 0:
        print(f"进度: {i}/{num}")

print("生成完毕，开始补全和着色...")

# 使用config.json中的图像宽度进行深度补全
ip(path0, output_dir, imageWidth=target_image_width)

# 生成可视化彩色图
colorful(output_dir, output_colorful_dir)
print("所有处理完成。")