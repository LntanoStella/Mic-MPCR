# -*- coding: utf-8 -*-
"""
点云配准可视化工具

功能：
1. 加载点云文件和变换矩阵
2. 选择两个点云进行可视化
3. 应用ground truth变换矩阵进行配准
4. 添加用户指定的扰动参数
5. 可视化不同状态的点云

输入：
- 数据目录路径（包含points_clouds和transforms子目录）
- 两个点云索引（1-4）
- 扰动参数（旋转扰动角度，平移扰动距离）

输出：
- 可视化窗口：原始点云、真值配准点云、带扰动的配准点云
"""

import os
import argparse
import numpy as np
import open3d as o3d


def load_point_clouds(data_dir):
    """
    加载点云文件
    
    参数:
        data_dir: 数据目录路径
    
    返回:
        dict: 点云字典 {frame_index: o3d.geometry.PointCloud}
    """
    point_clouds = {}
    points_dir = os.path.join(data_dir, 'point_clouds')
    
    # 获取所有npy格式的点云文件
    point_files = [f for f in os.listdir(points_dir) if f.endswith('.npy')]
    point_files.sort()
    
    for filename in point_files:
        file_path = os.path.join(points_dir, filename)
        try:
            # 从npy文件加载点云
            points = np.load(file_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            # 从文件名提取帧索引
            # 假设文件名格式为 lidar_1.npy
            frame_idx = int(filename.split('_')[1].split('.')[0])
            point_clouds[frame_idx] = pcd
            print(f"加载点云: {filename} -> 帧 {frame_idx}, 点数: {len(points)}")
        except Exception as e:
            print(f"警告: 无法加载点云文件 {filename}: {e}")
    
    return point_clouds


def load_transforms(data_dir):
    """
    加载变换矩阵文件
    
    参数:
        data_dir: 数据目录路径
    
    返回:
        dict: 变换矩阵字典 {(source_frame, target_frame): 4x4矩阵}
    """
    transforms = {}
    transforms_dir = os.path.join(data_dir, 'transforms')
    
    # 获取所有npy格式的变换矩阵文件
    transform_files = [f for f in os.listdir(transforms_dir) if f.endswith('.npy')]
    transform_files.sort()
    
    for filename in transform_files:
        file_path = os.path.join(transforms_dir, filename)
        try:
            # 从npy文件加载变换矩阵
            transform = np.load(file_path)
            
            # 从文件名提取帧索引
            # 假设文件名格式为 transform_1_2.npy
            base_name = os.path.splitext(filename)[0]
            parts = base_name.split('_')
            if len(parts) >= 3:
                source_frame = int(parts[1])
                target_frame = int(parts[2])
                transforms[(source_frame, target_frame)] = transform
                print(f"加载变换矩阵: {filename} -> {source_frame} -> {target_frame}")
        except Exception as e:
            print(f"警告: 无法加载变换矩阵文件 {filename}: {e}")
    
    return transforms


def create_rotation_matrix(angle_deg, axis=[0, 0, 1]):
    """
    创建旋转矩阵
    
    参数:
        angle_deg: 旋转角度（度）
        axis: 旋转轴
    
    返回:
        np.ndarray: 4x4旋转矩阵
    """
    angle_rad = np.deg2rad(angle_deg)
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
        np.array(axis) * angle_rad
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation_matrix
    return transform


def create_translation_matrix(translation):
    """
    创建平移矩阵
    
    参数:
        translation: 平移向量 [x, y, z]
    
    返回:
        np.ndarray: 4x4平移矩阵
    """
    transform = np.eye(4)
    transform[:3, 3] = translation
    return transform


def apply_perturbation(base_transform, rotation_perturb_deg, translation_perturb):
    """
    应用扰动到变换矩阵
    
    参数:
        base_transform: 基础变换矩阵
        rotation_perturb_deg: 旋转扰动角度（度）
        translation_perturb: 平移扰动距离（米）
    
    返回:
        np.ndarray: 带扰动的变换矩阵
    """
    # 创建旋转扰动矩阵
    rotation_perturb = create_rotation_matrix(rotation_perturb_deg)
    
    # 创建平移扰动矩阵
    translation_perturb = create_translation_matrix(translation_perturb)
    
    # 组合扰动
    perturbation = np.dot(translation_perturb, rotation_perturb)
    
    # 应用到基础变换
    perturbed_transform = np.dot(base_transform, perturbation)
    
    return perturbed_transform


def visualize_point_clouds(pcd1, pcd2, title):
    """
    可视化两个点云
    
    参数:
        pcd1: 第一个点云
        pcd2: 第二个点云
        title: 窗口标题
    """
    # 为点云设置不同颜色
    pcd1.paint_uniform_color([1, 0, 0])  # 红色
    pcd2.paint_uniform_color([0, 1, 0])  # 绿色
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(title, width=1200, height=800)
    
    # 添加点云
    vis.add_geometry(pcd1)
    vis.add_geometry(pcd2)
    
    # 设置视角
    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)
    
    # 运行可视化
    vis.run()
    vis.destroy_window()


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='点云配准可视化工具')
    parser.add_argument('--data-dir', type=str, 
                        default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\data\custom\random_Dataset\MiC_9\Variation_0001',
                        help='数据目录路径（包含point_clouds和transforms子目录）')
    parser.add_argument('--point-clouds', type=int, nargs=2, default=[2, 4],
                        help='选择的两个点云索引（1-3）')
    parser.add_argument('--perturbations', type=float, nargs='+', 
                        default=[6.71,0.12, 1.17,0.38, 0.67,0.24],
                        help='扰动参数，格式为 [旋转角度(度) 平移距离(米)] 或 [旋转1 平移1 旋转2 平移2]')
    
    args = parser.parse_args()
    
    # 加载数据
    print("="*60)
    print("加载数据...")
    print("="*60)
    
    point_clouds = load_point_clouds(args.data_dir)
    transforms = load_transforms(args.data_dir)
    
    # 检查点云索引是否有效
    cloud1_idx, cloud2_idx = args.point_clouds
    if cloud1_idx not in point_clouds or cloud2_idx not in point_clouds:
        print(f"错误: 点云索引 {cloud1_idx} 或 {cloud2_idx} 不存在")
        return
    
    # 获取点云
    pcd1 = o3d.geometry.PointCloud(point_clouds[cloud1_idx])
    pcd2 = o3d.geometry.PointCloud(point_clouds[cloud2_idx])
    
    # 1. 可视化原始点云
    print("\n" + "="*60)
    print("可视化1: 原始点云")
    print("="*60)
    visualize_point_clouds(pcd1, pcd2, f"原始点云 - 帧 {cloud1_idx} (红) 和 帧 {cloud2_idx} (绿)")
    
    # 2. 可视化真值配准点云
    print("\n" + "="*60)
    print("可视化2: 真值配准点云")
    print("="*60)
    
    # 查找真值变换矩阵
    transform_key = (cloud1_idx, cloud2_idx)
    if transform_key not in transforms:
        # 尝试反向变换
        transform_key = (cloud2_idx, cloud1_idx)
        if transform_key in transforms:
            # 使用逆变换
            base_transform = np.linalg.inv(transforms[transform_key])
            pcd2_transformed = o3d.geometry.PointCloud(pcd2)
            pcd2_transformed.transform(base_transform)
            print(f"使用变换矩阵: {cloud2_idx} -> {cloud1_idx} (逆变换)")
        else:
            print(f"错误: 未找到点云 {cloud1_idx} 和 {cloud2_idx} 之间的变换矩阵")
            return
    else:
        # 使用正向变换
        base_transform = transforms[transform_key]
        pcd2_transformed = o3d.geometry.PointCloud(pcd2)
        pcd2_transformed.transform(base_transform)
        print(f"使用变换矩阵: {cloud1_idx} -> {cloud2_idx}")
    
    visualize_point_clouds(pcd1, pcd2_transformed, f"真值配准点云 - 帧 {cloud1_idx} (红) 和 帧 {cloud2_idx} (绿)")
    
    # 3. 可视化带扰动的配准点云
    print("\n" + "="*60)
    print("可视化3: 带扰动的配准点云")
    print("="*60)
    
    # 解析扰动参数
    perturbations = []
    if len(args.perturbations) == 2:
        # 单组扰动参数
        perturbations = [args.perturbations]
    elif len(args.perturbations) % 2 == 0:
        # 多组扰动参数
        for i in range(0, len(args.perturbations), 2):
            perturbations.append(args.perturbations[i:i+2])
    else:
        print("错误: 扰动参数格式不正确，请提供偶数个值")
        return
    
    # 对每组扰动参数进行可视化
    for i, (rot_perturb, trans_perturb) in enumerate(perturbations):
        # 创建平移扰动向量（随机方向）
        # 这里简单使用x轴方向的平移，实际应用中可能需要随机方向
        trans_perturb_vec = [trans_perturb, 0, 0]
        
        # 应用扰动
        perturbed_transform = apply_perturbation(base_transform, rot_perturb, trans_perturb_vec)
        pcd2_perturbed = o3d.geometry.PointCloud(pcd2)
        pcd2_perturbed.transform(perturbed_transform)
        
        # 可视化
        title = f"带扰动的配准点云 {i+1} - 帧 {cloud1_idx} (红) 和 帧 {cloud2_idx} (绿)"
        title += f"\n旋转扰动: {rot_perturb}度, 平移扰动: {trans_perturb}米"
        visualize_point_clouds(pcd1, pcd2_perturbed, title)
    
    print("\n" + "="*60)
    print("处理完成！")
    print("="*60)


if __name__ == '__main__':
    main()
