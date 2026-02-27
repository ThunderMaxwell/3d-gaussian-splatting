#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image downsampling for 3DGS

功能描述：
    - 将3DGS数据中的照片按照指定宽度进行缩放，保持原始宽高比
    - 调整相机内参以适应新的图像尺寸
    - 复制images.txt文件到输出目录
    - 使用OpenCV进行图像处理，提高速度
    - 使用多线程并行处理，进一步提高速度

使用方法：
    python3 image_downsampling.py <workpath> <outpath> <width>
    
    参数说明：
    - workpath: 工作目录路径
    - outpath: 输出目录路径，用于保存处理后的文件
    - width: 目标图像宽度

输入结构：
    <workpath>/
    ├── images/          # 原始照片目录
    │   ├── 0/           # 相机0的照片
    │   └── 1/           # 相机1的照片
    ├── cameras.txt      # 原始相机内参文件
    └── images.txt       # 原始图像外参文件

输出结构：
    <outpath>/
    ├── images/          # 缩放后的照片（保持0和1文件夹结构）
    ├── cameras.txt      # 调整后的相机内参
    └── images.txt       # 复制的原始images.txt文件

依赖：
    - opencv-python
    - numpy
"""
import os
import sys
import argparse
import cv2



def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Image downsampling for 3DGS')
    parser.add_argument('workpath', type=str, help='Working directory path')
    parser.add_argument('outpath', type=str, help='Output directory path')
    parser.add_argument('width', type=int, help='Target image width')
    
    args = parser.parse_args()
    
    # 验证参数
    if args.width <= 0:
        raise ValueError("Width must be a positive integer")
    
    return args

def get_image_files(images_dir):
    """获取所有照片文件路径"""
    image_files = []
    
    # 遍历0和1文件夹
    for cam_folder in ['0', '1']:
        cam_dir = os.path.join(images_dir, cam_folder)
        if os.path.exists(cam_dir):
            for file_name in os.listdir(cam_dir):
                if file_name.endswith('.jpg'):
                    file_path = os.path.join(cam_dir, file_name)
                    image_files.append((cam_folder, file_name, file_path))
    
    return image_files

def process_image(cam_folder, file_name, file_path, output_dir, target_width):
    """处理单个图像"""
    # 创建输出目录
    cam_output_dir = os.path.join(output_dir, cam_folder)
    os.makedirs(cam_output_dir, exist_ok=True)
    
    # 处理图像
    try:
        # 使用OpenCV读取图像
        img = cv2.imread(file_path)
        if img is None:
            print(f"Error reading {file_path}")
            return
        
        # 计算新的高度，保持宽高比
        height, width = img.shape[:2]
        aspect_ratio = height / width
        new_height = int(target_width * aspect_ratio)
        
        # 调整大小
        resized_img = cv2.resize(img, (target_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 保存调整后的图像
        output_path = os.path.join(cam_output_dir, file_name)
        cv2.imwrite(output_path, resized_img)
        
        print(f"Resized {file_path} to {target_width}x{new_height}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def resize_images(image_files, output_dir, target_width):
    """调整图像大小并保存（并行处理）"""
    import concurrent.futures
    
    # 使用线程池并行处理图像
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for cam_folder, file_name, file_path in image_files:
            futures.append(executor.submit(process_image, cam_folder, file_name, file_path, output_dir, target_width))
        
        # 等待所有任务完成
        for future in concurrent.futures.as_completed(futures):
            pass

def read_cameras_file(cameras_path):
    """读取相机内参文件"""
    cameras = []
    
    with open(cameras_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 8:
                    camera = {
                        'id': int(parts[0]),
                        'model': parts[1],
                        'width': int(parts[2]),
                        'height': int(parts[3]),
                        'fx': float(parts[4]),
                        'fy': float(parts[5]),
                        'px': float(parts[6]),
                        'py': float(parts[7])
                    }
                    cameras.append(camera)
    
    return cameras

def adjust_camera_intrinsics(cameras, target_width):
    """调整相机内参"""
    adjusted_cameras = []
    
    for camera in cameras:
        # 计算缩放比例
        scale = target_width / camera['width']
        
        # 调整内参
        adjusted_camera = {
            'id': camera['id'],
            'model': camera['model'],
            'width': target_width,
            'height': int(camera['height'] * scale),
            'fx': camera['fx'] * scale,
            'fy': camera['fy'] * scale,
            'px': camera['px'] * scale,
            'py': camera['py'] * scale
        }
        adjusted_cameras.append(adjusted_camera)
    
    return adjusted_cameras

def save_cameras_file(cameras, output_path):
    """保存相机内参文件"""
    with open(output_path, 'w') as f:
        for camera in cameras:
            line = "{} {} {} {} {} {} {} {}".format(
                camera['id'],
                camera['model'],
                camera['width'],
                camera['height'],
                camera['fx'],
                camera['fy'],
                camera['px'],
                camera['py']
            )
            f.write(line + '\n')

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    workpath = args.workpath
    outpath = args.outpath
    target_width = args.width
    
    # 构建路径
    images_dir = os.path.join(workpath, 'images')
    output_base_dir = outpath
    output_images_dir = os.path.join(output_base_dir, 'images')
    cameras_path = os.path.join(workpath, 'cameras.txt')
    output_cameras_path = os.path.join(output_base_dir, 'cameras.txt')
    images_txt_path = os.path.join(workpath, 'images.txt')
    output_images_txt_path = os.path.join(output_base_dir, 'images.txt')
    
    # 检查输入路径
    if not os.path.exists(images_dir):
        print(f"Error: {images_dir} does not exist")
        sys.exit(1)
    
    if not os.path.exists(cameras_path):
        print(f"Error: {cameras_path} does not exist")
        sys.exit(1)
    
    # 1. 遍历照片文件
    print("Finding image files...")
    image_files = get_image_files(images_dir)
    print(f"Found {len(image_files)} image files")
    
    # 2. 照片下采样处理
    print("Resizing images...")
    resize_images(image_files, output_images_dir, target_width)
    
    # 3. 读取相机内参文件
    print("Reading camera intrinsics...")
    cameras = read_cameras_file(cameras_path)
    print(f"Read {len(cameras)} camera(s)")
    
    # 4. 调整相机内参
    print("Adjusting camera intrinsics...")
    adjusted_cameras = adjust_camera_intrinsics(cameras, target_width)
    
    # 5. 保存调整后的内参
    print("Saving adjusted camera intrinsics...")
    os.makedirs(output_base_dir, exist_ok=True)
    save_cameras_file(adjusted_cameras, output_cameras_path)
    
    # 6. 复制images.txt文件
    if os.path.exists(images_txt_path):
        print("Copying images.txt file...")
        import shutil
        shutil.copy2(images_txt_path, output_images_txt_path)
    
    print("Processing completed!")

if __name__ == "__main__":
    main()