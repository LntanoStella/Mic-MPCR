import numpy as np
import open3d as o3d
import os
import argparse
import copy


# =========================================================================
#               点云配准脚本
# =========================================================================
#
# 功能:
#   - 读取两个点云文件
#   - 读取转换矩阵文件
#   - 应用转换矩阵到点云
#   - 保存配准后的点云
#   - 可视化配准结果
#
# 使用方法 (在终端中运行):
#   python 04_register_pointclouds.py --source "path/to/source.ply" --target "path/to/target.ply" --transform "path/to/transform.txt"
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
            print(f"点云点数: {len(pcd.points)}")
            return pcd
        else:
            print("错误: 点云文件为空。")
            return None
    except Exception as e:
        print(f"错误: 加载点云文件时出错: {e}")
        return None

def load_transform_matrix(file_path):
    """
    加载转换矩阵文件
    """
    try:
        # 读取转换矩阵
        transform = np.loadtxt(file_path)
        
        # 确保矩阵是4x4的
        if transform.shape == (4, 4):
            print("成功加载转换矩阵:")
            print(transform)
            return transform
        else:
            print(f"错误: 转换矩阵形状不正确，期望4x4，实际为{transform.shape}")
            return None
    except Exception as e:
        print(f"错误: 加载转换矩阵时出错: {e}")
        return None

def apply_transform(pcd, transform):
    """
    应用转换矩阵到点云
    """
    try:
        # 应用变换
        transformed_pcd = copy.deepcopy(pcd)
        transformed_pcd.transform(transform)
        print("成功应用转换矩阵到点云")
        return transformed_pcd
    except Exception as e:
        print(f"错误: 应用转换矩阵时出错: {e}")
        return None

def save_registered_pointcloud(pcd, source_path, target_path):
    """
    保存配准后的点云
    """
    try:
        # 获取源文件的目录和基本名称
        source_dir = os.path.dirname(source_path)
        source_base = os.path.splitext(os.path.basename(source_path))[0]
        target_base = os.path.splitext(os.path.basename(target_path))[0]
        ext = os.path.splitext(source_path)[1]
        
        # 生成输出文件名
        output_name = f"{source_base}_registered_to_{target_base}{ext}"
        
        # 生成输出路径
        output_path = os.path.join(source_dir, output_name)
        
        # 保存点云
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"成功保存配准后的点云到: {output_path}")
        return output_path
    except Exception as e:
        print(f"错误: 保存配准后的点云时出错: {e}")
        return None

def visualize_registration(source, target, transformed_source):
    """
    可视化配准结果
    """
    try:
        # 创建点云副本并设置颜色
        source_copy = copy.deepcopy(source)
        target_copy = copy.deepcopy(target)
        transformed_copy = copy.deepcopy(transformed_source)
        
        # 设置颜色：源点云为红色，目标点云为蓝色，配准后的源点云为绿色
        source_copy.paint_uniform_color([1, 0, 0])  # 红色
        target_copy.paint_uniform_color([0, 0, 1])    # 蓝色
        transformed_copy.paint_uniform_color([0, 1, 0])  # 绿色
        
        print("\n开始可视化配准结果...")
        print("颜色说明: 红色=原始源点云, 蓝色=目标点云, 绿色=配准后的源点云")
        
        # 可视化原始点云
        print("\n1. 原始点云展示:")
        o3d.visualization.draw_geometries(
            [source_copy, target_copy], 
            window_name="原始点云 (红色=源, 蓝色=目标)",
            width=720,
            height=405
        )
        
        # 可视化配准结果
        print("\n2. 配准结果展示:")
        o3d.visualization.draw_geometries(
            [transformed_copy, target_copy], 
            window_name="配准结果 (绿色=配准后的源, 蓝色=目标)",
            width=720,
            height=405
        )
        
        print("\n可视化完成。")
    except Exception as e:
        print(f"错误: 可视化时出错: {e}")

def main():
    parser = argparse.ArgumentParser(description="点云配准脚本")
    parser.add_argument('--source', type=str, default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\experiments\geotransformer.custom.stage4.gse.k3.max.oacl.stage2.sinkhorn\mytools\bunny\test\cloud_bin_3.ply', help='源点云文件路径。')
    parser.add_argument('--target', type=str, default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\experiments\geotransformer.custom.stage4.gse.k3.max.oacl.stage2.sinkhorn\mytools\bunny\test\cloud_bin_0.ply', help='目标点云文件路径。')
    parser.add_argument('--transform', type=str, default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\experiments\geotransformer.custom.stage4.gse.k3.max.oacl.stage2.sinkhorn\mytools\bunny\test\gt.txt', help='转换矩阵文件路径。')
    parser.add_argument('--save', type=bool, default=True, help='是否保存配准后的点云。')
    parser.add_argument('--visualize', type=bool, default=True, help='是否可视化配准结果。')
    args = parser.parse_args()

    # 加载源点云
    source_pcd = load_point_cloud(args.source)
    if not source_pcd:
        return

    # 加载目标点云
    target_pcd = load_point_cloud(args.target)
    if not target_pcd:
        return

    # 加载转换矩阵
    transform = load_transform_matrix(args.transform)
    if transform is None:
        return

    # 应用转换矩阵到源点云
    transformed_source = apply_transform(source_pcd, transform)
    if not transformed_source:
        return

    # 保存配准后的点云
    if args.save:
        output_path = save_registered_pointcloud(transformed_source, args.source, args.target)
        if not output_path:
            return

    # 可视化配准结果
    if args.visualize:
        visualize_registration(source_pcd, target_pcd, transformed_source)

    print("\n点云配准完成。")


if __name__ == "__main__":
    main()
