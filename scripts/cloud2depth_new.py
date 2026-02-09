import numpy as np
import os
import open3d as o3d

"""create by jxf @ 2023/01/24
"""

def c2d(i, root, path0, pcd_path, r):
    # modified by zc @ 2024/02/26
    """根据CC xml获取位姿，配合点云生成深度信息，并保存为.npz格式文件

    Args:
        i (int): 生成数量
        root (dom): xml文件
        path0 (str): 深度信息文件保存路径
        pcd_path (str): 点云文件路径
        r (int): 用于计算点云中隐藏点去除的半径。
    """
    # end modified @ zc
    os.makedirs(path0, exist_ok=True)
    ImagePath = root.getElementsByTagName('ImagePath')
    path = ImagePath[i].firstChild.data.split('/')[-1]
    cam = path[6:10]
    id = int(cam[-1])
    Width = int(root.getElementsByTagName('Width')[id].firstChild.data)                             # 宽
    Height = int(root.getElementsByTagName('Height')[id].firstChild.data)                           # 高
    FocalLength = float(root.getElementsByTagName('Photogroup')[id].childNodes[13].firstChild.data) # 焦距
    SensorSize = float(root.getElementsByTagName('SensorSize')[id].firstChild.data)                 # 传感器尺寸
    fx = Width * FocalLength / SensorSize
    fy = fx
    cx = float(root.getElementsByTagName('PrincipalPoint')[id].childNodes[1].firstChild.data)
    cy = float(root.getElementsByTagName('PrincipalPoint')[id].childNodes[3].firstChild.data)
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    M_00 = float(root.getElementsByTagName('M_00')[i * 2].firstChild.data)
    M_01 = float(root.getElementsByTagName('M_01')[i * 2].firstChild.data)
    M_02 = float(root.getElementsByTagName('M_02')[i * 2].firstChild.data)
    M_10 = float(root.getElementsByTagName('M_10')[i * 2].firstChild.data)
    M_11 = float(root.getElementsByTagName('M_11')[i * 2].firstChild.data)
    M_12 = float(root.getElementsByTagName('M_12')[i * 2].firstChild.data)
    M_20 = float(root.getElementsByTagName('M_20')[i * 2].firstChild.data)
    M_21 = float(root.getElementsByTagName('M_21')[i * 2].firstChild.data)
    M_22 = float(root.getElementsByTagName('M_22')[i * 2].firstChild.data)
    R = np.array([[M_00,M_01,M_02],[M_10,M_11,M_12],[M_20,M_21,M_22]])

    x = float(root.getElementsByTagName('Center')[i * 2].childNodes[1].firstChild.data)
    y = float(root.getElementsByTagName('Center')[i * 2].childNodes[3].firstChild.data)
    z = float(root.getElementsByTagName('Center')[i * 2].childNodes[5].firstChild.data)
    t = np.array([x,y,z])
    t = -np.dot(R,t)
    T = np.hstack((R, t.reshape(3, 1)))
    # 读取点云数据
    camera = [x, y, z]
    pcd = o3d.io.read_point_cloud(pcd_path)
    diameter = np.linalg.norm(
        np.asarray(pcd.get_max_bound()) - np.asarray(pcd.get_min_bound()))
    radius = diameter * r
    _, pt_map = pcd.hidden_point_removal(camera, radius)
    pcd1 = pcd.select_by_index(pt_map)
    points = np.asarray(pcd1.points)
    proj_cloud = np.dot(K, np.dot(T, np.vstack((points.T, np.ones(points.shape[0]))))).T
    # 计算深度图
    depth = np.zeros((Height, Width), dtype=np.float32)
    for p in proj_cloud:
        u, v = int(p[0]/p[2]), int(p[1]/p[2])
        if 0 <= u < Width and 0 <= v < Height and p[2] >= 0:
            depth[v, u] = p[2]

    np.savez(os.path.join(path0, path[:-3] + "npz"), depth)


# end create @ jxf
