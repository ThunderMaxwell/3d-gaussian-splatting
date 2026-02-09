# ip_basic_new.py

import glob
import os
import numpy as np
import depth_map_utils
import cv2

# [修改 1] 将 imageWidth 默认值设为 None
def ip(input_depth_dir, output_dir, imageWidth=None):
    os.makedirs(output_dir, exist_ok=True)
    fill_type = 'fast'
    extrapolate = True
    blur_type = 'bilateral'

    images_to_use = sorted(glob.glob(os.path.join(input_depth_dir, '*.npz')))
    print(f"DEBUG: Found {len(images_to_use)} files in {input_depth_dir}")
    
    num_images = len(images_to_use)
    for i in range(num_images):
        depth_image = images_to_use[i]
        depth = np.load(depth_image)
        depth = depth[depth.files[0]].astype(np.float32)
        depth_copy = depth.copy()
        
        if fill_type == 'fast':
            final_depth = depth_map_utils.fill_in_fast(
                depth_copy, extrapolate=extrapolate, blur_type=blur_type)
        elif fill_type == 'multiscale':
            final_depth, process_dict = depth_map_utils.fill_in_multiscale(
                depth_copy, extrapolate=extrapolate, blur_type=blur_type)
        else:
            raise ValueError('Invalid fill_type {}'.format(fill_type))

        depth[depth<1] = final_depth[depth<1]
        
        # [修改 2] 只有当 imageWidth 被指定(不是 None)时才进行强制缩放
        if imageWidth is not None:
            # 确保转为 int
            target_w = int(imageWidth)
            # 如果目标宽度和当前宽度不一样，才缩放
            if target_w != depth.shape[1]:
                depth = cv2.resize(depth, (target_w, int(target_w / depth.shape[1] * depth.shape[0])))
        
        # 如果 imageWidth 是 None，代码会直接跳过 resize，保留原始尺寸 (3535 或 3574)

        filename = os.path.basename(depth_image) 
        save_path = os.path.join(output_dir, filename)
        np.savez(save_path, depth)
        
        # 可以减少打印频率，或者保留
        if i % 10 == 0:
            print(f"Saved {i}/{num_images}: {save_path} (Shape: {depth.shape})")