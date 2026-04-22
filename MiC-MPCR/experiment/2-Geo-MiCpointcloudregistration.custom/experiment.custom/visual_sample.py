import numpy as np
import open3d as o3d
import os
import time
# 从sample.py导入需要的类和函数
from sample import HierarchicalStructurePreservingSampling

def load_point_cloud(pc_path):
    """加载点云数据"""
    if pc_path.endswith('.npy'):
        points = np.load(pc_path)
    elif pc_path.endswith('.txt') or pc_path.endswith('.pcd'):
        pcd = o3d.io.read_point_cloud(pc_path)
        points = np.asarray(pcd.points)
    else:
        raise ValueError(f"不支持的文件格式: {pc_path}")
    return points

def visualize_point_cloud(points, point_types=None, title="点云可视化"):
    """可视化点云，不同类型的点用不同颜色表示"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 设置颜色
    colors = np.zeros((len(points), 3))
    if point_types is None:
        # 默认为蓝色
        colors[:, 0] = 1.0  # 蓝色
    else:
        # 平面点: 蓝色 (0, 0, 1)
        # 边缘点: 绿色 (0, 1, 0)
        # 角点: 红色 (1, 0, 0)
        colors[:, 2] = (point_types == 0).astype(float)  # 蓝色 - 平面点
        colors[:, 1] = (point_types == 1).astype(float)  # 绿色 - 边缘点
        colors[:, 0] = (point_types == 2).astype(float)  # 红色 - 角点
    
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    vis.add_geometry(pcd)
    
    # 设置可视化参数
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 2.0
    
    print(f"可视化窗口 '{title}' 已创建。按 'Q' 键关闭窗口。")
    print("颜色说明：")
    if point_types is not None:
        print("- 红色: 角点")
        print("- 绿色: 边缘点")
        print("- 蓝色: 平面点")
    
    vis.run()
    vis.destroy_window()

def visualize_sampling_result(original_points, sampled_points, point_types, title="采样结果可视化"):
    """可视化采样结果，对比原始点云和采样点云"""
    # 为大型点云下采样以提高可视化性能
    if len(original_points) > 50000:
        indices = np.random.choice(len(original_points), 50000, replace=False)
        vis_original = original_points[indices]
    else:
        vis_original = original_points
    
    # 创建点云对象
    # 原始点云 - 灰色
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(vis_original)
    original_pcd.paint_uniform_color([0.5, 0.5, 0.5])  # 灰色
    
    # 采样点云 - 根据类型着色
    sampled_pcd = o3d.geometry.PointCloud()
    sampled_pcd.points = o3d.utility.Vector3dVector(sampled_points)
    
    # 为不同类型点设置颜色: 角点(红色), 边缘点(绿色), 平面点(蓝色)
    colors = np.zeros((len(sampled_points), 3))
    colors[point_types == 2] = [1, 0, 0]  # 角点 - 红色
    colors[point_types == 1] = [0, 1, 0]  # 边缘点 - 绿色
    colors[point_types == 0] = [0, 0, 1]  # 平面点 - 蓝色
    sampled_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title)
    
    # 添加点云
    vis.add_geometry(original_pcd)
    vis.add_geometry(sampled_pcd)
    
    # 设置可视化参数
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0])
    opt.point_size = 2.0
    
    print(f"可视化窗口 '{title}' 已创建。按 'Q' 键关闭窗口。")
    print("颜色说明：")
    print("- 灰色: 原始点云")
    print("- 红色: 采样后的角点")
    print("- 绿色: 采样后的边缘点")
    print("- 蓝色: 采样后的平面点")
    
    vis.run()
    vis.destroy_window()

def test_sampling_methods(points, num_samples=20000):
    """测试不同采样方法的用时"""
    results = {}
    
    # 测试随机采样
    start_time = time.time()
    try:
        # 尝试获取三个返回值（适配修改后的接口）
        try:
            sampled_points_random, indices_random, point_types_random = random_sampling(points, num_samples, method='random')
        except ValueError:
            # 兼容旧接口，只返回两个值
            sampled_points_random, indices_random = random_sampling(points, num_samples, method='random')
            point_types_random = np.zeros(len(sampled_points_random), dtype=int)  # 默认全为平面点
        random_time = time.time() - start_time
        results['random'] = {
            'time': random_time,
            'points': sampled_points_random,
            'indices': indices_random,
            'point_types': point_types_random
        }
        print(f"随机采样用时: {random_time:.4f} 秒")
    except Exception as e:
        print(f"随机采样失败: {e}")
    
    # 测试FPS采样
    start_time = time.time()
    try:
        # 尝试获取三个返回值
        try:
            sampled_points_fps, indices_fps, point_types_fps = random_sampling(points, num_samples, method='fps')
        except ValueError:
            # 兼容旧接口
            sampled_points_fps, indices_fps = random_sampling(points, num_samples, method='fps')
            point_types_fps = np.zeros(len(sampled_points_fps), dtype=int)  # 默认全为平面点
        fps_time = time.time() - start_time
        results['fps'] = {
            'time': fps_time,
            'points': sampled_points_fps,
            'indices': indices_fps,
            'point_types': point_types_fps
        }
        print(f"FPS采样用时: {fps_time:.4f} 秒")
    except Exception as e:
        print(f"FPS采样失败: {e}")
    
    # 测试结构保留采样
    start_time = time.time()
    try:
        sampled_points_struct, indices_struct, point_types_struct = random_sampling(points, num_samples, method='structure_preserving')
        struct_time = time.time() - start_time
        results['structure_preserving'] = {
            'time': struct_time,
            'points': sampled_points_struct,
            'indices': indices_struct,
            'point_types': point_types_struct
        }
        print(f"结构保留采样用时: {struct_time:.4f} 秒")
    except Exception as e:
        print(f"结构保留采样失败: {e}")
    
    return results

def main():
    # 设置点云文件路径和参数
    point_cloud_file = "D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\data\demo\cad_cloud2.npy"  # 替换为实际的点云文件路径
    target_num_points = 100000  # 目标采样点数
    
    # 加载点云数据
    points = np.load(point_cloud_file)
    print(f"原始点云点数: {len(points)}")
    
    # 创建采样器实例
    sampler = HierarchicalStructurePreservingSampling(
        target_num_points=target_num_points,
        corner_ratio=0.15,
        edge_ratio=0.35,
        plane_ratio=0.50
    )
    
    # 执行采样
    start_time = time.time()
    sampled_points, sampled_indices, point_types = sampler.sample(points)
    end_time = time.time()
    
    print(f"采样后点数: {len(sampled_points)}")
    print(f"采样耗时: {end_time - start_time:.4f} 秒")
    
    # 统计各类点的数量
    corner_count = np.sum(point_types == 2)
    edge_count = np.sum(point_types == 1)
    plane_count = np.sum(point_types == 0)
    
    print(f"角点数量: {corner_count} ({corner_count/len(sampled_points)*100:.1f}%)")
    print(f"边缘点数量: {edge_count} ({edge_count/len(sampled_points)*100:.1f}%)")
    print(f"平面点数量: {plane_count} ({plane_count/len(sampled_points)*100:.1f}%)")
    
    # 可视化结果
    visualize_sampling_result(points, sampled_points, point_types)
    
    # 单独可视化每种类型的点
    print("\n单独可视化各类型点:")
    
    # 可视化角点（红色）
    corner_mask = point_types == 2
    corner_points = sampled_points[corner_mask]
    corner_types = point_types[corner_mask]
    if len(corner_points) > 0:
        visualize_point_cloud(corner_points, corner_types, title="角点可视化 (红色)")
    else:
        print("没有检测到角点")
    
    # 可视化边缘点（绿色）
    edge_mask = point_types == 1
    edge_points = sampled_points[edge_mask]
    edge_types = point_types[edge_mask]
    if len(edge_points) > 0:
        visualize_point_cloud(edge_points, edge_types, title="边缘点可视化 (绿色)")
    else:
        print("没有检测到边缘点")
    
    # 可视化平面点（蓝色）
    plane_mask = point_types == 0
    plane_points = sampled_points[plane_mask]
    plane_types = point_types[plane_mask]
    if len(plane_points) > 0:
        visualize_point_cloud(plane_points, plane_types, title="平面点可视化 (蓝色)")
    else:
        print("没有检测到平面点")

if __name__ == "__main__":
    main()