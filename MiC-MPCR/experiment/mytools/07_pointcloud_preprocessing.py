# -*- coding: utf-8 -*-
"""
点云数据预处理工具

功能：
- 加载点云数据
- 地面分割（RANSAC平面拟合）
- 点云去噪（统计滤波）
- 杂物去除（基于大小的过滤）
- 保存处理后的点云
- 可视化前后对比

输入：
- 点云文件路径（.npy格式）

输出：
- 处理后的点云文件
"""

import os
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def load_point_cloud(file_path):
    """
    加载点云数据
    
    参数:
        file_path: 点云文件路径
    
    返回:
        o3d.geometry.PointCloud: 点云对象
    """
    print(f"加载点云: {file_path}")
    points = np.load(file_path)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd


def remove_ground(pcd, distance_threshold=0.1, ransac_n=3, num_iterations=1000):
    """
    去除地面点云
    
    参数:
        pcd: 原始点云
        distance_threshold: 距离阈值
        ransac_n: RANSAC采样点数
        num_iterations: 迭代次数
    
    返回:
        o3d.geometry.PointCloud: 去除地面后的点云
        o3d.geometry.PointCloud: 地面点云
    """
    print("执行地面分割...")
    
    # 执行RANSAC平面拟合
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )
    
    # 提取地面点和非地面点
    inlier_cloud = pcd.select_by_index(inliers)
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    
    print(f"地面点数量: {len(inlier_cloud.points)}")
    print(f"非地面点数量: {len(outlier_cloud.points)}")
    
    return outlier_cloud, inlier_cloud


def denoise_point_cloud(pcd, nb_neighbors=20, std_ratio=1.0):
    """
    点云去噪
    
    参数:
        pcd: 原始点云
        nb_neighbors: 邻域点数量
        std_ratio: 标准差倍数
    
    返回:
        o3d.geometry.PointCloud: 去噪后的点云
    """
    print("执行点云去噪...")
    
    # 统计滤波 - 调整参数以更严格地过滤离群点
    cl, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio
    )
    
    print(f"去噪前点数量: {len(pcd.points)}")
    print(f"去噪后点数量: {len(cl.points)}")
    
    # 可选：添加半径滤波进一步去除离群点
    cl, ind = cl.remove_radius_outlier(nb_points=10, radius=0.1)
    print(f"半径滤波后点数量: {len(cl.points)}")
    
    return cl


def remove_small_objects(pcd, min_points=100, eps=0.1):
    """
    去除小物体
    
    参数:
        pcd: 原始点云
        min_points: 最小物体点数
        eps: DBSCAN聚类半径
    
    返回:
        o3d.geometry.PointCloud: 去除小物体后的点云
    """
    print("执行小物体去除...")
    
    # 使用DBSCAN聚类
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    # 统计每个聚类的大小
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # 只保留大于等于min_points的聚类
    large_clusters = []
    for label, count in zip(unique_labels, counts):
        if label != -1 and count >= min_points:
            large_clusters.append(label)
    
    # 提取大物体
    if large_clusters:
        mask = np.isin(labels, large_clusters)
        indices = np.where(mask)[0]
        filtered_pcd = pcd.select_by_index(indices.tolist())
        print(f"去除小物体后点数量: {len(filtered_pcd.points)}")
        return filtered_pcd
    else:
        print("没有找到符合条件的物体")
        return pcd


def save_point_cloud(pcd, output_path, format='npy'):
    """
    保存点云数据
    
    参数:
        pcd: 点云对象
        output_path: 输出文件路径
        format: 保存格式 ('npy' 或 'ply')
    """
    print(f"保存点云: {output_path}")
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if format == 'npy':
        points = np.asarray(pcd.points)
        np.save(output_path, points)
    elif format == 'ply':
        o3d.io.write_point_cloud(output_path, pcd)
    else:
        print(f"不支持的格式: {format}")


def visualize_step(step_name, point_clouds, titles):
    """
    可视化单个步骤的效果
    
    参数:
        step_name: 步骤名称
        point_clouds: 点云列表
        titles: 点云标题列表
    """
    print(f"可视化{step_name}...")
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=step_name, width=1200, height=800)
    
    # 根据步骤名称设置不同的颜色方案，确保对比鲜明
    if step_name == "地面分割效果":
        # 地面点云（蓝色）和去除地面后的点云（绿色）
        colors = [[0, 0.5, 1], [0, 1, 0]]  # 蓝色、绿色
    elif step_name == "去噪效果":
        # 去除地面后的点云（绿色）和去噪后的点云（红色）
        colors = [[0, 1, 0], [1, 0, 0]]  # 绿色、红色
    elif step_name == "小物体去除效果":
        # 去噪后的点云（红色）和去除小物体后的点云（深蓝色）
        colors = [[1, 0, 0], [0, 0, 1]]  # 红色、深蓝色
    elif step_name == "整体处理前后对比":
        # 原始点云（淡橙色）和最终处理后的点云（深绿色）
        colors = [[0.8, 0.6, 0.4], [0, 0.7, 0]]  # 淡橙色、深绿色
    else:
        # 默认颜色方案
        colors = [[1, 0.5, 0], [0, 1, 0], [0, 0.5, 1], [1, 0, 0], [0, 0, 1]]  # 橙色、绿色、蓝色、红色、深蓝色
    
    # 设置点云颜色和添加到窗口
    for i, (pcd, title) in enumerate(zip(point_clouds, titles)):
        if i < len(colors):
            pcd.paint_uniform_color(colors[i])
        vis.add_geometry(pcd)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def visualize_comparison(original, no_ground, denoised, final, ground=None):
    """
    可视化处理前后的对比，分步骤展示
    
    参数:
        original: 原始点云
        no_ground: 去除地面后的点云
        denoised: 去噪后的点云
        final: 最终处理后的点云（去除小物体）
        ground: 地面点云
    """
    print("开始分步骤可视化...")
    
    # 1. 可视化地面分割效果（只显示地面点云和去除地面后的点云）
    if ground:
        visualize_step(
            "地面分割效果",
            [ground, no_ground],
            ["地面点云", "去除地面后的点云"]
        )
    
    # 2. 可视化去噪效果
    visualize_step(
        "去噪效果",
        [no_ground, denoised],
        ["去除地面后的点云", "去噪后的点云"]
    )
    
    # 3. 可视化小物体去除效果
    visualize_step(
        "小物体去除效果",
        [denoised, final],
        ["去噪后的点云", "去除小物体后的点云"]
    )
    
    # 4. 可视化整体处理前后对比（使用不同颜色，避免覆盖）
    visualize_step(
        "整体处理前后对比",
        [original, final],
        ["原始点云", "最终处理后的点云"]
    )


def process_point_cloud(input_path, output_path, 
                       remove_ground_flag=True, 
                       denoise_flag=True, 
                       remove_small_objects_flag=True, 
                       visualize_flag=False, 
                       save_format='npy'):
    """
    处理单个点云文件
    
    参数:
        input_path: 输入点云文件路径
        output_path: 输出点云文件路径
        remove_ground_flag: 是否去除地面
        denoise_flag: 是否去噪
        remove_small_objects_flag: 是否去除小物体
        visualize_flag: 是否可视化
        save_format: 保存格式
    """
    print(f"\n处理文件: {input_path}")
    
    # 加载点云
    pcd = load_point_cloud(input_path)
    # 创建原始点云的深拷贝
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    if pcd.has_colors():
        original_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    
    # 存储中间步骤的点云
    no_ground_pcd = None
    denoised_pcd = None
    final_pcd = None
    ground_pcd = None
    
    # 去除地面
    if remove_ground_flag:
        pcd, ground_pcd = remove_ground(pcd)
        # 创建去除地面后的点云副本
        no_ground_pcd = o3d.geometry.PointCloud()
        no_ground_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            no_ground_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    else:
        # 如果不去除地面，使用原始点云
        no_ground_pcd = o3d.geometry.PointCloud()
        no_ground_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            no_ground_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    
    # 去噪
    if denoise_flag:
        pcd = denoise_point_cloud(pcd)
        # 创建去噪后的点云副本
        denoised_pcd = o3d.geometry.PointCloud()
        denoised_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            denoised_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    else:
        # 如果不去噪，使用去除地面后的点云
        denoised_pcd = o3d.geometry.PointCloud()
        denoised_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
        if pcd.has_colors():
            denoised_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    
    # 去除小物体
    if remove_small_objects_flag:
        pcd = remove_small_objects(pcd)
    
    # 最终处理后的点云
    final_pcd = o3d.geometry.PointCloud()
    final_pcd.points = o3d.utility.Vector3dVector(np.asarray(pcd.points))
    if pcd.has_colors():
        final_pcd.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors))
    
    # 保存处理后的点云
    save_point_cloud(pcd, output_path, save_format)
    
    # 可视化对比
    if visualize_flag:
        visualize_comparison(original_pcd, no_ground_pcd, denoised_pcd, final_pcd, ground_pcd)
    
    print(f"处理完成: {output_path}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='点云数据预处理工具')
    parser.add_argument('--input-dir', type=str, 
                        default='d:\\Application\\PycharmProfessional\\pycharm\\PointCloud_registration\\GeoTransformer-main\\data\\custom\\RawData_local\\MiC-3-ground-enhanced-2\\point_clouds',
                        help='输入点云目录')
    parser.add_argument('--output-dir', type=str, 
                        default='d:\\Application\\PycharmProfessional\\pycharm\\PointCloud_registration\\GeoTransformer-main\\data\\custom\\RawData_local\\MiC-3-ground-enhanced\\processed_point_clouds',
                        help='输出点云目录')
    parser.add_argument('--remove-ground', action='store_true', default=True,
                        help='是否去除地面')
    parser.add_argument('--denoise', action='store_true', default=True,
                        help='是否去噪')
    parser.add_argument('--remove-small-objects', action='store_true', default=True,
                        help='是否去除小物体')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='是否可视化前后对比')
    parser.add_argument('--format', type=str, default='npy', choices=['npy', 'ply'],
                        help='保存格式')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理所有点云文件
    for filename in os.listdir(args.input_dir):
        if filename.endswith('.npy'):
            input_path = os.path.join(args.input_dir, filename)
            output_filename = filename.replace('.npy', f'_processed.{args.format}')
            output_path = os.path.join(args.output_dir, output_filename)
            
            process_point_cloud(
                input_path=input_path,
                output_path=output_path,
                remove_ground_flag=args.remove_ground,
                denoise_flag=args.denoise,
                remove_small_objects_flag=args.remove_small_objects,
                visualize_flag=args.visualize,
                save_format=args.format
            )
    
    print("\n所有点云处理完成！")


if __name__ == '__main__':
    main()
