import open3d as o3d
import numpy as np
import matplotlib.cm as cm


# =========================================================================
#               点云着色可视化脚本
# =========================================================================
#
# 功能:
#   - 加载点云文件
#   - 支持多种点云着色方式:
#     1. 按高度着色（使用 viridis 颜色映射）
#     2. 按视点深度着色（使用 plasma 颜色映射）
#     3. 按法向量着色
#   - 可视化着色后的点云
#
# 输入:
#   - 点云文件路径（默认：bun_zipper.ply）
#
# 输出:
#   - 着色后的点云可视化窗口
#
# 使用方法 (在终端中运行):
#   python 05_pointcloud_colorization.py
#
# =========================================================================

# 加载点云文件
def load_point_cloud(file_path):
    """
    加载点云文件，支持多种格式
    
    参数:
        file_path: 点云文件路径
    
    返回:
        o3d.geometry.PointCloud: 点云对象
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == '.npy':
        # 加载 npy 格式
        points = np.load(file_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    elif ext == '.ply' or ext == '.pcd':
        # 加载 ply 或 pcd 格式
        pcd = o3d.io.read_point_cloud(file_path)
    elif ext == '.txt':
        # 加载 txt 格式
        points = np.loadtxt(file_path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    else:
        raise ValueError(f"不支持的文件格式: {ext}")
    
    return pcd

import os
pcd = load_point_cloud(r'D:\Document\low_score3\merged_points.txt')

# ------------------ 方式1：按高度 viridis ------------------
points = np.asarray(pcd.points)
z = points[:, 2]
z_norm = (z - z.min()) / (z.max() - z.min() + 1e-6)

colors = cm.viridis(z_norm)[:, :3]          # 或 cm.plasma / cm.turbo
pcd.colors = o3d.utility.Vector3dVector(colors)

# ------------------ 方式2：按视点深度 -----------------------
viewpoint = np.array([0, -3, 2])           # 假设相机位置
dist = np.linalg.norm(points - viewpoint, axis=1)
dist_norm = (dist - dist.min()) / (dist.max() - dist.min() + 1e-6)
colors = cm.plasma(dist_norm)[:, :3]
pcd.colors = o3d.utility.Vector3dVector(colors)

# ------------------ 方式3：法向量着色 -----------------------
pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
normals = np.asarray(pcd.normals)
colors = (normals + 1) / 2
pcd.colors = o3d.utility.Vector3dVector(colors)

# 可视化着色后的点云
o3d.visualization.draw_geometries([pcd])
