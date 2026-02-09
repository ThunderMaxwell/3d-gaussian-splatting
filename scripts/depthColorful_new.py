import os
import cv2
import numpy as np

"""created by jxf @ 2023/01/24
"""

def colorful(output_dir,output_colorful_dir):
    # modified by zc @ 2024/02/26
    """可视化深度图，生成深度图片

    Args:
        output_dir (str): 深度信息文件路径
        output_colorful_dir (str): 深度图片输出路径
    """
    # end modified
    os.makedirs(output_colorful_dir, exist_ok=True)
    input_depth_dir = os.path.expanduser(output_dir)
    image_list = []
    depth_pathes = os.listdir(input_depth_dir)
    for file in depth_pathes:
        if file[-3:] == "npz" or file[-3:] == "npy":
            image_list.append(input_depth_dir + '/' + file)
    for depth_image in image_list:
        depth = np.load(depth_image)
        if depth_image[-3:] == "npz":
            depth = depth[depth.files[0]].astype(np.float32)
        img = cv2.applyColorMap(np.uint8(depth / np.amax(depth) * 255), cv2.COLORMAP_JET)
        img_name = depth_image.split('/')[-1].split('.')[0]
        outpath = output_colorful_dir + '/' + img_name + '.png'
        cv2.imwrite(outpath, img)
        print(outpath)

# end created @ jxf