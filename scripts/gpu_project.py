import torch
import numpy as np
import os
import json
from xml.dom import minidom

def get_target_width_from_config():
    """从config.json中获取目标宽度"""
    try:
        with open('scripts/config.json') as file:
            data = json.load(file)
        return data.get('imageWidth', None)
    except Exception as e:
        print(f"无法读取config.json中的imageWidth: {e}")
        return None

def c2d_gpu(i, root, path0, points_tensor, device="cuda", target_width=None):
    """
    GPU 加速版点云投影 (支持强制分辨率缩放)
    修正了M矩阵索引超出范围的问题，采用正确的XML解析方式
    """
    # 1. 获取第i个Photo节点
    photo_elements = root.getElementsByTagName('Photo')
    if i >= len(photo_elements):
        print(f"错误: 索引 {i} 超出 Photo 数组范围 ({len(photo_elements)})")
        return
    
    current_photo = photo_elements[i]
    
    # 解析 ImagePath
    image_path_element = current_photo.getElementsByTagName('ImagePath')
    if not image_path_element:
        print(f"错误: 第 {i} 个 Photo 节点缺少 ImagePath")
        return
        
    path_str = image_path_element[0].firstChild.data.split('/')[-1]
    cam = path_str[6:10]
    id = int(cam[-1])
    
    # 2. 解析 XML 里的原始宽高 (例如 Cam1 可能是 3535)
    # 从Photogroup中获取相机参数
    photogroups = root.getElementsByTagName('Photogroup')
    if id >= len(photogroups):
        print(f"错误: 相机ID {id} 超出 Photogroup 数组范围")
        return
        
    photogroup = photogroups[id]
    
    # 获取宽高
    width_elements = photogroup.getElementsByTagName('Width')
    height_elements = photogroup.getElementsByTagName('Height')
    if not width_elements or not height_elements:
        print(f"错误: Photogroup {id} 缺少 Width 或 Height")
        return
        
    W_xml = int(width_elements[0].firstChild.data)
    H_xml = int(height_elements[0].firstChild.data)
    
    # [修改] 优先使用传入的target_width，如果没有则从config.json获取
    if target_width is not None:
        TARGET_WIDTH = int(target_width)
    else:
        # 尝试从config.json获取目标宽度
        config_target_width = get_target_width_from_config()
        if config_target_width is not None:
            TARGET_WIDTH = int(config_target_width)
        else:
            # 如果config.json也没有，则使用XML中的原始尺寸
            TARGET_WIDTH = W_xml
    
    # 保持长宽比计算目标高度
    scale_val = TARGET_WIDTH / W_xml
    TARGET_HEIGHT = int(H_xml * scale_val)

    # 打印一下调试信息，确保缩放生效
    print(f"Processing {path_str}: XML_W={W_xml} -> Target_W={TARGET_WIDTH}, Scale={scale_val:.4f}")

    # 3. 解析并缩放内参
    sensor_sizes = photogroup.getElementsByTagName('SensorSize')
    principal_points = photogroup.getElementsByTagName('PrincipalPoint')
    
    if not sensor_sizes:
        print(f"错误: Photogroup {id} 缺少 SensorSize")
        return
        
    Focal_mm = float(photogroup.getElementsByTagName('FocalLength')[0].firstChild.data)
    Sensor_mm = float(sensor_sizes[0].firstChild.data)
    
    # 检查主点元素
    if not principal_points:
        print(f"错误: Photogroup {id} 缺少 PrincipalPoint")
        return
        
    pp_element = principal_points[0]
    cx_xml = float(pp_element.getElementsByTagName('x')[0].firstChild.data)
    cy_xml = float(pp_element.getElementsByTagName('y')[0].firstChild.data)
    
    # 原始 fx, fy
    fx_xml = W_xml * Focal_mm / Sensor_mm
    fy_xml = fx_xml 
    
    # [修改] 应用缩放比例 (关键步骤：把原始的内参拉伸适配到目标尺寸)
    fx = fx_xml * scale_val
    fy = fy_xml * scale_val
    cx = cx_xml * scale_val
    cy = cy_xml * scale_val
    
    Width = TARGET_WIDTH
    Height = TARGET_HEIGHT
    
    # 4. 解析外参 - 从当前Photo节点内部解析
    pose_elements = current_photo.getElementsByTagName('Pose')
    if not pose_elements:
        print(f"错误: 第 {i} 个 Photo 节点缺少 Pose")
        return
        
    pose = pose_elements[0]
    
    # --- 解析旋转矩阵 (R) ---
    rotation_element = pose.getElementsByTagName('Rotation')
    if not rotation_element:
        print(f"错误: 第 {i} 个 Pose 节点缺少 Rotation")
        return
        
    rotation = rotation_element[0]
    M_tags = ['M_00', 'M_01', 'M_02', 'M_10', 'M_11', 'M_12', 'M_20', 'M_21', 'M_22']
    vals = []
    for tag in M_tags:
        m_elements = rotation.getElementsByTagName(tag)
        if not m_elements:
            print(f"错误: Rotation 缺少 {tag} 元素")
            return
        vals.append(float(m_elements[0].firstChild.data))
    
    R = torch.tensor(vals, device=device).reshape(3, 3)
    
    # --- 解析相机中心 (Center) ---
    center_elements = pose.getElementsByTagName('Center')
    if not center_elements:
        print(f"错误: 第 {i} 个 Pose 节点缺少 Center")
        return
        
    center_element = center_elements[0]
    x = float(center_element.getElementsByTagName('x')[0].firstChild.data)
    y = float(center_element.getElementsByTagName('y')[0].firstChild.data)
    z = float(center_element.getElementsByTagName('z')[0].firstChild.data)
    center = torch.tensor([x, y, z], device=device)
    
    t = -torch.matmul(R, center)
    
    # ==================== GPU 投影计算 ====================
    
    # World -> Camera
    xyz_cam = torch.matmul(points_tensor, R.t()) + t

    # 提取深度 Z
    z_cam = xyz_cam[:, 2]
    valid_mask = z_cam > 0.1 
    
    xyz_valid = xyz_cam[valid_mask]
    z_valid = z_cam[valid_mask]
    
    if z_valid.shape[0] == 0:
        print(f"警告: 相机 {path_str} 没有有效的深度点")
        # 创建空的深度图
        depth_map = torch.zeros((Height, Width), dtype=torch.float32, device=device)
    else:
        # 透视投影: Camera -> Pixel
        u = (fx * xyz_valid[:, 0] / z_valid) + cx
        v = (fy * xyz_valid[:, 1] / z_valid) + cy
        
        # 像素边界过滤 (使用新的 Width/Height)
        u_int = torch.round(u).long()
        v_int = torch.round(v).long()
        
        valid_pixel_mask = (u_int >= 0) & (u_int < Width) & \
                           (v_int >= 0) & (v_int < Height)
                           
        u_final = u_int[valid_pixel_mask]
        v_final = v_int[valid_pixel_mask]
        z_final = z_valid[valid_pixel_mask]
        
        if z_final.shape[0] == 0:
            print(f"警告: 相机 {path_str} 没有有效的像素点")
            # 创建空的深度图
            depth_map = torch.zeros((Height, Width), dtype=torch.float32, device=device)
        else:
            # Z-Buffer 处理
            sort_idx = torch.argsort(z_final, descending=True)
            u_sorted = u_final[sort_idx]
            v_sorted = v_final[sort_idx]
            z_sorted = z_final[sort_idx]
            
            # 赋值
            depth_map = torch.zeros((Height, Width), dtype=torch.float32, device=device)
            depth_map[v_sorted, u_sorted] = z_sorted
    
    # 保存
    depth_numpy = depth_map.cpu().numpy()
    
    if not os.path.exists(path0):
        os.makedirs(path0, exist_ok=True)

    save_name = os.path.join(path0, path_str[:-3] + "npz")
    np.savez(save_name, depth=depth_numpy)