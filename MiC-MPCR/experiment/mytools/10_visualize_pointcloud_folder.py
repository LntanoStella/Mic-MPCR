# -*- coding: utf-8 -*-
"""
点云文件夹可视化工具

功能：
1. 读取指定文件夹中的所有点云文件
2. 支持多种点云格式（.ply, .pcd, .npy, .txt）
3. 为每个点云分配不同的颜色
4. 分开可视化每个点云
5. 支持命令行参数设置

使用方法：
python 10_visualize_pointcloud_folder.py --input /path/to/pointclouds --color-scheme sci_professional
"""

import os
import argparse
import numpy as np
import open3d as o3d

# 定义配色方案
color_schemes = {
    # 标准配色方案
    'standard': {
        'name': '标准配色',
        'colors': [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
    },
    # SCI专业学术配色方案
    'sci_professional': {
        'name': 'SCI专业配色',
        'colors': [[0.0, 0.447, 0.741], [0.85, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    },
    # 灰度配色方案
    'grayscale': {
        'name': '灰度配色',
        'colors': [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9], [1, 1, 1]]
    },
    # 暖色配色方案
    'warm': {
        'name': '暖色配色',
        'colors': [[1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0.5, 0.25, 0], [0.75, 0.37, 0], [0.25, 0.12, 0]]
    },
    # 冷色配色方案
    'cool': {
        'name': '冷色配色',
        'colors': [[0, 0, 1], [0, 0.5, 1], [0, 1, 1], [0, 0.25, 0.5], [0, 0.37, 0.75], [0, 0.12, 0.25]]
    }
}

def load_point_cloud(file_path):
    """
    加载点云文件
    
    参数:
        file_path: 点云文件路径
    
    返回:
        o3d.geometry.PointCloud: 点云对象
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.ply':
            pcd = o3d.io.read_point_cloud(file_path)
        elif ext == '.pcd':
            pcd = o3d.io.read_point_cloud(file_path)
        elif ext == '.npy':
            points = np.load(file_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
        elif ext == '.txt':
            points = np.loadtxt(file_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
        else:
            print(f"错误: 不支持的文件格式: {ext}")
            return None
        
        if not pcd.has_points():
            print(f"错误: 点云文件 {os.path.basename(file_path)} 为空")
            return None
        
        print(f"成功加载点云: {os.path.basename(file_path)}, 点数: {len(pcd.points)}")
        return pcd
    except Exception as e:
        print(f"错误: 加载点云文件 {file_path} 时出错: {e}")
        return None

def get_point_cloud_files(folder_path):
    """
    获取文件夹中的所有点云文件
    
    参数:
        folder_path: 文件夹路径
    
    返回:
        list: 点云文件路径列表
    """
    supported_extensions = ['.ply', '.pcd', '.npy', '.txt']
    point_cloud_files = []
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                file_path = os.path.join(root, file)
                point_cloud_files.append(file_path)
    
    # 按文件名排序
    point_cloud_files.sort()
    return point_cloud_files

def visualize_point_clouds(point_clouds, file_names, color_scheme='standard'):
    """
    可视化多个点云
    
    参数:
        point_clouds: 点云对象列表
        file_names: 文件名列表
        color_scheme: 配色方案
    """
    if not point_clouds:
        print("错误: 没有点云可可视化")
        return
    
    print(f"\n开始可视化点云（使用 {color_schemes[color_scheme]['name']}）...")
    print("提示: 按ESC键退出当前窗口，将显示下一个点云")
    
    colors = color_schemes[color_scheme]['colors']
    
    for i, (pcd, file_name) in enumerate(zip(point_clouds, file_names)):
        if pcd and pcd.has_points():
            # 为点云设置颜色
            color_idx = i % len(colors)
            pcd.paint_uniform_color(colors[color_idx])
            
            # 计算点云范围
            points = np.asarray(pcd.points)
            min_coords = np.min(points, axis=0)
            max_coords = np.max(points, axis=0)
            
            print(f"\n可视化 {file_name}:")
            print(f"  点数: {len(pcd.points)}")
            print(f"  坐标范围: X[{min_coords[0]:.4f}, {max_coords[0]:.4f}], Y[{min_coords[1]:.4f}, {max_coords[1]:.4f}], Z[{min_coords[2]:.4f}, {max_coords[2]:.4f}]")
            
            # 可视化
            o3d.visualization.draw_geometries(
                [pcd],
                window_name=f"点云可视化 - {file_name}",
                width=1024,
                height=768
            )
        else:
            print(f"错误: 点云 {file_name} 无效，跳过可视化")
    
    print("\n所有点云可视化完成")

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='点云文件夹可视化工具')
    parser.add_argument('--input', type=str, default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\PCR_test\data2', help='点云文件夹路径')
    parser.add_argument('--color-scheme', type=str, default='standard',
                        choices=list(color_schemes.keys()),
                        help='配色方案')
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入文件夹 {args.input} 不存在")
        return
    
    # 获取点云文件
    point_cloud_files = get_point_cloud_files(args.input)
    
    if not point_cloud_files:
        print(f"错误: 文件夹 {args.input} 中没有找到点云文件")
        return
    
    print(f"找到 {len(point_cloud_files)} 个点云文件:")
    for file_path in point_cloud_files:
        print(f"  - {os.path.basename(file_path)}")
    
    # 加载点云
    point_clouds = []
    file_names = []
    
    for file_path in point_cloud_files:
        pcd = load_point_cloud(file_path)
        if pcd:
            point_clouds.append(pcd)
            file_names.append(os.path.basename(file_path))
    
    # 可视化点云
    if point_clouds:
        visualize_point_clouds(point_clouds, file_names, args.color_scheme)
    else:
        print("错误: 未能加载任何点云文件")


if __name__ == '__main__':
    main()
