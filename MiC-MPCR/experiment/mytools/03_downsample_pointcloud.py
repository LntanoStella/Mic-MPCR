import numpy as np
import open3d as o3d
import os
import argparse
import copy


# =========================================================================
#               点云降采样脚本
# =========================================================================
#
# 功能:
#   - 读取点云文件
#   - 支持多种降采样方法
#   - 将降采样后的点云保存到同目录下，自动重命名
#   - 支持可视化降采样结果
#
# 使用方法 (在终端中运行):
#   python 03_downsample_pointcloud.py --input "path/to/pointcloud.ply" --method voxel --size 0.01
#
# =========================================================================

def load_point_cloud(file_path):
    """
    加载点云文件
    """
    try:
        # 为txt文件明确指定格式为xyz
        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == '.txt':
            pcd = o3d.io.read_point_cloud(file_path, format='xyz')
        else:
            # 其他格式让Open3D自动检测
            pcd = o3d.io.read_point_cloud(file_path)
        
        if pcd.has_points():
            print(f"成功加载点云文件: {os.path.basename(file_path)}")
            print(f"原始点云点数: {len(pcd.points)}")
            return pcd
        else:
            print("错误: 点云文件为空。")
            return None
    except Exception as e:
        print(f"错误: 加载点云文件时出错: {e}")
        return None

def compute_point_curvature(pcd):
    """
    计算点云的曲率
    """
    # 计算法线
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    
    # 计算曲率
    # 这里使用简单的曲率估计方法：基于邻域点的法线变化
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    curvature = np.zeros(len(points))
    
    # 构建KD树
    kdtree = o3d.geometry.KDTreeFlann(pcd)
    
    for i in range(len(points)):
        # 查找最近邻
        [k, idx, _] = kdtree.search_knn_vector_3d(points[i], 10)  # 10个最近邻
        if k > 1:
            # 计算邻域法线的协方差矩阵
            neighbor_normals = normals[idx[1:], :]  # 排除自身
            cov_matrix = np.cov(neighbor_normals.T)
            # 计算特征值并排序
            eigenvalues = np.linalg.eigvals(cov_matrix)
            eigenvalues.sort()
            # 计算曲率
            curvature[i] = eigenvalues[0] / (eigenvalues.sum() + 1e-10)
    
    return curvature

def downsample_point_cloud(pcd, method, size):
    """
    对点云进行降采样
    """
    if method == 'voxel':
        # 体素降采样
        downsampled = pcd.voxel_down_sample(voxel_size=size)
        print(f"体素降采样完成，新点数: {len(downsampled.points)}")
    elif method == 'uniform':
        # 均匀降采样
        # 计算需要保留的点数
        target_points = int(len(pcd.points) * size)
        if target_points < 1:
            target_points = 1
        # 随机选择点
        indices = np.random.choice(len(pcd.points), size=target_points, replace=False)
        downsampled = pcd.select_by_index(indices)
        print(f"均匀降采样完成，新点数: {len(downsampled.points)}")
    elif method == 'farthest':
        # 最远点采样
        # 计算需要保留的点数
        target_points = int(len(pcd.points) * size)
        if target_points < 1:
            target_points = 1
        # 执行最远点采样
        downsampled = pcd.farthest_point_down_sample(number_of_points=target_points)
        print(f"最远点采样完成，新点数: {len(downsampled.points)}")
    elif method == 'curvature':
        # 基于曲率的降采样
        # 计算需要保留的点数
        target_points = int(len(pcd.points) * size)
        if target_points < 1:
            target_points = 1
        
        # 计算曲率
        curvature = compute_point_curvature(pcd)
        
        # 选择曲率较大的点
        # 曲率值越大，表示点所在区域越复杂，需要保留更多点
        indices = np.argsort(-curvature)[:target_points]  # 取曲率最大的点
        downsampled = pcd.select_by_index(indices)
        print(f"基于曲率的降采样完成，新点数: {len(downsampled.points)}")
    elif method == 'radius':
        # 半径降采样
        # 计算需要保留的点数
        target_points = int(len(pcd.points) * size)
        if target_points < 1:
            target_points = 1
        
        # 计算点云边界框，估计合适的半径
        bbox = pcd.get_axis_aligned_bounding_box()
        diagonal = np.linalg.norm(bbox.get_max_bound() - bbox.get_min_bound())
        radius = diagonal * 0.01  # 边界框对角线的1%
        
        # 执行半径降采样
        downsampled = pcd.uniform_down_sample(every_k_points=max(1, len(pcd.points) // target_points))
        print(f"半径降采样完成，新点数: {len(downsampled.points)}")
    elif method == 'random':
        # 随机降采样
        # 计算需要保留的点数
        target_points = int(len(pcd.points) * size)
        if target_points < 1:
            target_points = 1
        
        # 随机选择点
        indices = np.random.choice(len(pcd.points), size=target_points, replace=False)
        downsampled = pcd.select_by_index(indices)
        print(f"随机降采样完成，新点数: {len(downsampled.points)}")
    else:
        print(f"错误: 不支持的降采样方法: {method}")
        return None
    
    return downsampled

def save_downsampled_point_cloud(pcd, input_path, method, size):
    """
    保存降采样后的点云到同目录下，自动重命名
    """
    try:
        # 获取输入文件的目录和基本名称
        input_dir = os.path.dirname(input_path)
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        ext = os.path.splitext(input_path)[1]
        
        # 生成输出文件名
        if method == 'voxel':
            output_name = f"{base_name}_downsampled_voxel_{size:.4f}{ext}"
        elif method == 'uniform':
            output_name = f"{base_name}_downsampled_uniform_{size:.2f}{ext}"
        elif method == 'farthest':
            output_name = f"{base_name}_downsampled_farthest_{size:.2f}{ext}"
        elif method == 'curvature':
            output_name = f"{base_name}_downsampled_curvature_{size:.2f}{ext}"
        elif method == 'radius':
            output_name = f"{base_name}_downsampled_radius_{size:.2f}{ext}"
        elif method == 'random':
            output_name = f"{base_name}_downsampled_random_{size:.2f}{ext}"
        else:
            output_name = f"{base_name}_downsampled_{method}{ext}"
        
        # 生成输出路径
        output_path = os.path.join(input_dir, output_name)
        
        # 保存点云
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"成功保存降采样后的点云到: {output_path}")
        return True
    except Exception as e:
        print(f"错误: 保存点云时出错: {e}")
        return False

def visualize_downsampling_result(original, downsampled, method):
    """
    可视化降采样结果
    """
    try:
        # 创建点云副本并设置颜色
        original_copy = copy.deepcopy(original)
        downsampled_copy = copy.deepcopy(downsampled)
        
        # 设置颜色：原始点云为蓝色，降采样后的点云为红色
        original_copy.paint_uniform_color([0, 0, 1])    # 蓝色
        downsampled_copy.paint_uniform_color([1, 0, 0])  # 红色
        
        print("\n开始可视化降采样结果...")
        print("颜色说明: 蓝色=原始点云, 红色=降采样后的点云")
        
        # 可视化原始点云
        print("\n1. 原始点云:")
        o3d.visualization.draw_geometries(
            [original_copy], 
            window_name="原始点云 (蓝色)",
            width=720,
            height=405
        )
        
        # 可视化降采样后的点云
        print("\n2. 降采样后的点云:")
        o3d.visualization.draw_geometries(
            [downsampled_copy], 
            window_name=f"{method} 降采样结果 (红色)",
            width=720,
            height=405
        )
        
        # 可视化对比
        print("\n3. 对比视图:")
        o3d.visualization.draw_geometries(
            [original_copy, downsampled_copy], 
            window_name="降采样对比 (蓝色=原始, 红色=降采样)",
            width=720,
            height=405
        )
        
        print("\n可视化完成。")
    except Exception as e:
        print(f"错误: 可视化时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="点云降采样脚本")
    parser.add_argument('--input', type=str, default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\experiments\geotransformer.custom.stage4.gse.k3.max.oacl.stage2.sinkhorn\mytools\mypointcloud\MiC-3_uniform_30000.ply', help='点云文件路径。')
    parser.add_argument('--method', type=str, default='uniform', 
                        choices=['voxel', 'farthest', 'curvature', 'random'],
                        help='降采样方法: voxel (体素降采样), farthest (最远点采样), random (随机降采样), curvature (基于曲率的降采样), uniform (均匀降采样), radius (半径降采样)。')
    parser.add_argument('--size', type=float, default=0.1, 
                        help='降采样参数: 对于voxel是体素大小，对于其他方法是保留点数的比例。')
    parser.add_argument('--save', type=bool, default=False, help='是否保存降采样后的点云。')
    parser.add_argument('--visualize', type=bool, default=True, help='是否可视化降采样结果。')
    args = parser.parse_args()

    # 加载点云
    pcd = load_point_cloud(args.input)
    if not pcd:
        return

    # 执行降采样
    downsampled_pcd = downsample_point_cloud(pcd, args.method, args.size)
    if not downsampled_pcd:
        return

    # 保存降采样后的点云
    if args.save:
        save_downsampled_point_cloud(downsampled_pcd, args.input, args.method, args.size)

    # 可视化降采样结果
    if args.visualize:
        visualize_downsampling_result(pcd, downsampled_pcd, args.method)

    print("\n点云降采样完成。")


if __name__ == "__main__":
    main()
