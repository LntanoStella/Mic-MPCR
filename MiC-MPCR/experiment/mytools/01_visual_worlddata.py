import numpy as np
import open3d as o3d
import os
import argparse
import copy
import glob


# =========================================================================
#               点云可视化脚本
# =========================================================================
#
# 功能:
#   - 实现了两种模式的可视化，展示点云数据。
#   - 模式1: 单独展示。分别加载并显示每个点云文件。
#   - 模式2: 统一展示。将所有点云在同一窗口中显示。
#   - 支持多种点云文件格式: .txt, .pcd, .ply, .xyz
#   - 支持多种配色方案，适用于不同视角和不同采样密度的点云
#
# 使用方法 (在终端中运行):
#   python your_script_name.py --folder "path/to/your/point_cloud_folder"
#
# =========================================================================

# 定义配色方案
color_schemes = {
    # 标准配色方案（默认）
    'standard': {
        'name': '标准配色',
        'colors': [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 0]]
    },
    # 灰度配色方案（适合打印）
    'grayscale': {
        'name': '灰度配色',
        'colors': [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9], [1, 1, 1]]
    },
    # 暖色配色方案（适合不同视角）
    'warm': {
        'name': '暖色配色',
        'colors': [[1, 0, 0], [1, 0.5, 0], [1, 1, 0], [0.5, 0.25, 0], [0.75, 0.37, 0], [0.25, 0.12, 0]]
    },
    # 冷色配色方案（适合不同视角）
    'cool': {
        'name': '冷色配色',
        'colors': [[0, 0, 1], [0, 0.5, 1], [0, 1, 1], [0, 0.25, 0.5], [0, 0.37, 0.75], [0, 0.12, 0.25]]
    },
    # 采样密度配色方案（适合不同密度展示）
    'density': {
        'name': '密度配色',
        'colors': [[0.1, 0.1, 0.1], [0.3, 0.3, 0.3], [0.5, 0.5, 0.5], [0.7, 0.7, 0.7], [0.9, 0.9, 0.9], [1, 1, 1]]
    },
    # SCI专业学术配色方案1 - 对比鲜明且专业
    'sci_professional': {
        'name': 'SCI专业配色',
        'colors': [[0.0, 0.447, 0.741], [0.85, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
    },
    # SCI专业学术配色方案2 - 色盲友好
    'sci_colorblind': {
        'name': 'SCI色盲友好配色',
        'colors': [[0, 0.45, 0.7], [0.8, 0.4, 0], [0.9, 0.6, 0], [0.35, 0.7, 0.9], [0, 0.6, 0.5], [0.5, 0.5, 0.5]]
    },
    # SCI渐变色方案 - 用于表示连续变量
    'sci_gradient': {
        'name': 'SCI渐变色',
        'colors': [[0.267, 0.004, 0.329], [0.190, 0.400, 0.557], [0.204, 0.620, 0.545], [0.678, 0.847, 0.329]]
    },
    # SCI单色调配色 - 强调形状和结构
    'sci_monochrome': {
        'name': 'SCI单色调',
        'colors': [[0.1, 0.3, 0.5], [0.2, 0.4, 0.6], [0.3, 0.5, 0.7], [0.4, 0.6, 0.8], [0.5, 0.7, 0.9], [0.6, 0.8, 1.0]]
    },
    # 点云配准专用配色 - 参考和源点云区分
    'registration': {
        'name': '配准专用配色',
        'colors': [[0.0, 0.447, 0.741], [0.85, 0.325, 0.098], [0.929, 0.694, 0.125], [0.494, 0.184, 0.556]]
    }
}

def get_density_based_color(point_count):
    """
    根据点云密度返回对应的颜色
    点云点数越多，颜色越亮
    """
    # 密度等级划分
    density_levels = [1000, 5000, 10000, 50000, 100000]
    colors = color_schemes['density']['colors']
    
    # 根据点云点数确定密度等级
    for i, level in enumerate(density_levels):
        if point_count <= level:
            return colors[i]
    return colors[-1]  # 点数超过最大等级，返回最亮的颜色

def main():
    parser = argparse.ArgumentParser(description="点云可视化脚本")
    # D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\experiments\geotransformer.custom.stage4.gse.k3.max.oacl.stage2.sinkhorn\mytools\bunny\test
    # D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\data\custom\RawData_world\MiC3
    # 'd:\\Application\\PycharmProfessional\\pycharm\\PointCloud_registration\\GeoTransformer-main\\data\\custom\\RawData_local\\MiC-3-ground-enhanced-2\\point_clouds'
    parser.add_argument('--folder', type=str, default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\data\custom\RawData_world\MiC_9_01',
                        help='包含点云文件的文件夹路径。')
    parser.add_argument('--scheme', type=str, default='sci_professional', choices=list(color_schemes.keys()),
                        help='配色方案: standard, grayscale, warm, cool, density, sci_professional, sci_colorblind, sci_gradient, sci_monochrome, registration')
    args = parser.parse_args()

    data_folder = args.folder
    color_scheme = args.scheme
    print(f"--- 开始可视化数据文件夹: {data_folder} ---")
    print(f"--- 使用配色方案: {color_schemes[color_scheme]['name']} ---")

    if not os.path.isdir(data_folder):
        print(f"错误: 文件夹 '{data_folder}' 不存在。请检查路径。")
        return

    # 可视化窗口大小设置
    window_width = 720  # 窗口宽度
    window_height = 405   # 窗口高度

    # 支持的点云文件格式
    supported_formats = ['*.txt', '*.pcd', '*.ply', '*.xyz']
    point_cloud_files = []
    
    # 搜索所有支持的点云文件
    for fmt in supported_formats:
        point_cloud_files.extend(glob.glob(os.path.join(data_folder, fmt)))
    
    # 去重并排序
    point_cloud_files = sorted(list(set(point_cloud_files)))
    
    if not point_cloud_files:
        print("错误: 未找到任何支持的点云文件。")
        return
    
    print(f"找到 {len(point_cloud_files)} 个点云文件:")
    for i, file_path in enumerate(point_cloud_files):
        print(f"  {i+1}. {os.path.basename(file_path)}")
    
    # 加载点云
    point_clouds = []
    for file_path in point_cloud_files:
        try:
            # 为txt文件明确指定格式为xyz
            file_ext = os.path.splitext(file_path)[1].lower()
            if file_ext == '.txt':
                pcd = o3d.io.read_point_cloud(file_path, format='xyz')
            else:
                # 其他格式让Open3D自动检测
                pcd = o3d.io.read_point_cloud(file_path)
            
            if pcd.has_points():
                print(f"成功加载 {os.path.basename(file_path)} ({len(pcd.points)} 个点)。")
                point_clouds.append(pcd)
            else:
                print(f"警告: {os.path.basename(file_path)} 没有点数据。")
                point_clouds.append(o3d.geometry.PointCloud())
        except Exception as e:
            print(f"错误: 加载 {os.path.basename(file_path)} 时出错: {e}")
            point_clouds.append(o3d.geometry.PointCloud())

    if not any(pcd.has_points() for pcd in point_clouds):
        print("错误: 未能加载任何有效的点云数据。")
        return

    # ==============================================================
    #               可视化 1: 单独展示
    # ==============================================================
    print("\n--- 可视化 1: 单独展示每个点云 ---")
    print("    - 每个点云将单独显示在不同窗口中。")
    print("    - 关闭当前窗口后将显示下一个点云。")

    for i, (pcd, file_path) in enumerate(zip(point_clouds, point_cloud_files)):
        if pcd.has_points():
            pcd_copy = copy.deepcopy(pcd)  # 创建副本以避免修改原始数据
            
            # 根据选择的配色方案设置颜色
            if color_scheme == 'density':
                # 基于密度的配色
                color = get_density_based_color(len(pcd.points))
            else:
                # 基于索引的配色
                colors = color_schemes[color_scheme]['colors']
                color_idx = i % len(colors)
                color = colors[color_idx]
            
            pcd_copy.paint_uniform_color(color)
            file_name = os.path.basename(file_path)
            print(f"    - 显示 {file_name} ({len(pcd.points)} 个点)。")
            o3d.visualization.draw_geometries(
                [pcd_copy], 
                window_name=f"{file_name} (单独展示)",
                width=window_width,
                height=window_height
            )

    # ==============================================================
    #               可视化 2: 统一展示
    # ==============================================================
    print("\n--- 可视化 2: 统一展示所有点云 ---")
    print("    - 所有点云将在同一窗口中显示，使用不同颜色区分。")

    viz_list = []
    for i, (pcd, file_path) in enumerate(zip(point_clouds, point_cloud_files)):
        if pcd.has_points():
            pcd_copy = copy.deepcopy(pcd)  # 创建副本以避免修改原始数据
            
            # 根据选择的配色方案设置颜色
            if color_scheme == 'density':
                # 基于密度的配色
                color = get_density_based_color(len(pcd.points))
            else:
                # 基于索引的配色
                colors = color_schemes[color_scheme]['colors']
                color_idx = i % len(colors)
                color = colors[color_idx]
            
            pcd_copy.paint_uniform_color(color)
            viz_list.append(pcd_copy)

    o3d.visualization.draw_geometries(
        viz_list, 
        window_name=f"统一可视化 (所有点云) - {color_schemes[color_scheme]['name']}",
        width=window_width,
        height=window_height
    )

    # ==============================================================
    #               可视化 3: 密度对比展示
    # ==============================================================
    print("\n--- 可视化 3: 密度对比展示 ---")
    print("    - 所有点云将在同一窗口中显示，根据密度自动着色。")

    density_viz_list = []
    for i, (pcd, file_path) in enumerate(zip(point_clouds, point_cloud_files)):
        if pcd.has_points():
            pcd_copy = copy.deepcopy(pcd)  # 创建副本以避免修改原始数据
            # 基于密度的配色
            color = get_density_based_color(len(pcd.points))
            pcd_copy.paint_uniform_color(color)
            density_viz_list.append(pcd_copy)

    o3d.visualization.draw_geometries(
        density_viz_list, 
        window_name="密度对比可视化 (所有点云)",
        width=window_width,
        height=window_height
    )

    print("\n可视化完毕。")


if __name__ == "__main__":
    main()