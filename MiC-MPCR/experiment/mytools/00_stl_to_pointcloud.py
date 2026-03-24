import numpy as np
import open3d as o3d
import os
import argparse
import copy


# =========================================================================
#               STL模型转点云采样脚本
# =========================================================================
#
# 功能:
#   - 从STL模型中采样点云
#   - 支持多种采样模式
#   - 支持设置采样密度
#   - 支持保存采样后的点云
#   - 支持可视化，包括不同的配色方案
#   - 支持同时选择多种采样模式并进行可视化
#
# 使用方法 (在终端中运行):
#   python 00_stl_to_pointcloud.py --input "path/to/model.stl" --modes uniform poisson --density 10000
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

def load_stl_model(file_path):
    """
    加载STL模型
    """
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_vertices() or not mesh.has_triangles():
            print("错误: 加载的STL模型无效。")
            return None
        print(f"成功加载STL模型: {os.path.basename(file_path)}")
        print(f"模型信息: 顶点数={len(mesh.vertices)}, 三角面数={len(mesh.triangles)}")
        return mesh
    except Exception as e:
        print(f"错误: 加载STL模型时出错: {e}")
        return None

def sample_point_cloud(mesh, mode, density):
    """
    从网格模型中采样点云
    """
    if mode == 'uniform':
        # 均匀采样
        pcd = mesh.sample_points_uniformly(number_of_points=density)
    elif mode == 'poisson':
        # 泊松采样
        pcd = mesh.sample_points_poisson_disk(number_of_points=density, pcl=None)
    elif mode == 'random':
        # 随机采样
        pcd = mesh.sample_points_uniformly(number_of_points=density)
    elif mode == 'area':
        # 基于面积的采样
        # 计算每个三角面的面积
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        
        # 计算每个三角面的面积
        areas = []
        for tri in triangles:
            v1 = vertices[tri[0]]
            v2 = vertices[tri[1]]
            v3 = vertices[tri[2]]
            # 计算向量
            vec1 = v2 - v1
            vec2 = v3 - v1
            # 计算叉积的一半作为面积
            area = 0.5 * np.linalg.norm(np.cross(vec1, vec2))
            areas.append(area)
        
        areas = np.array(areas)
        total_area = np.sum(areas)
        probabilities = areas / total_area
        
        # 根据面积概率选择三角面
        selected_triangles = np.random.choice(len(triangles), size=density, p=probabilities)
        
        # 在选定的三角面上随机采样点
        points = []
        for tri_idx in selected_triangles:
            tri = triangles[tri_idx]
            v1, v2, v3 = vertices[tri[0]], vertices[tri[1]], vertices[tri[2]]
            # 生成重心坐标
            r1, r2 = np.random.random(2)
            if r1 + r2 > 1:
                r1, r2 = 1 - r1, 1 - r2
            r3 = 1 - r1 - r2
            # 计算采样点
            point = r1 * v1 + r2 * v2 + r3 * v3
            points.append(point)
        
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
    elif mode == 'curvature':
        # 基于曲率的采样
        # 计算顶点法线
        mesh.compute_vertex_normals()
        
        # 计算顶点曲率（基于法线变化）
        vertices = np.asarray(mesh.vertices)
        normals = np.asarray(mesh.vertex_normals)
        
        # 简单的曲率估计：计算每个顶点与其邻居法线的差异
        curvature = np.zeros(len(vertices))
        
        # 构建顶点邻居关系
        vertex_neighbors = {i: set() for i in range(len(vertices))}
        for tri in mesh.triangles:
            vertex_neighbors[tri[0]].add(tri[1])
            vertex_neighbors[tri[0]].add(tri[2])
            vertex_neighbors[tri[1]].add(tri[0])
            vertex_neighbors[tri[1]].add(tri[2])
            vertex_neighbors[tri[2]].add(tri[0])
            vertex_neighbors[tri[2]].add(tri[1])
        
        # 计算每个顶点的曲率
        for i in range(len(vertices)):
            neighbors = list(vertex_neighbors[i])
            if len(neighbors) > 0:
                # 计算与邻居法线的平均差异
                normal_diff = 0
                for j in neighbors:
                    # 计算法线夹角的余弦值
                    cos_angle = np.dot(normals[i], normals[j])
                    # 转换为角度差异
                    angle_diff = np.arccos(np.clip(cos_angle, -1, 1))
                    normal_diff += angle_diff
                curvature[i] = normal_diff / len(neighbors)
        
        # 归一化曲率
        curvature = curvature / np.max(curvature)
        # 曲率越高，采样概率越大
        probabilities = curvature + 0.1  # 添加一个小值确保所有点都有采样概率
        probabilities = probabilities / np.sum(probabilities)
        
        # 根据曲率概率选择顶点
        selected_vertices = np.random.choice(len(vertices), size=density, p=probabilities)
        selected_points = vertices[selected_vertices]
        
        # 创建点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(selected_points)
    elif mode == 'adaptive':
        # 自适应采样
        # 基于表面复杂度调整采样密度
        pcd = mesh.sample_points_poisson_disk(number_of_points=density, radius=0.01)
    elif mode == 'vertex':
        # 网格顶点采样
        # 直接使用网格的顶点作为点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = mesh.vertices
        # 如果顶点数超过目标密度，进行下采样
        if len(pcd.points) > density:
            pcd = pcd.voxel_down_sample(voxel_size=0.01)
            # 如果仍超过目标密度，随机采样
            if len(pcd.points) > density:
                indices = np.random.choice(len(pcd.points), size=density, replace=False)
                pcd = pcd.select_by_index(indices)
    elif mode == 'normal':
        # 法线指导采样
        # 考虑表面法线方向的采样
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_uniformly(number_of_points=density)
        # 复制法线信息
        pcd.normals = mesh.vertex_normals
    else:
        print(f"错误: 不支持的采样模式: {mode}")
        return None
    
    print(f"{mode}采样完成，获得 {len(pcd.points)} 个点。")
    
    # 将点云坐标从毫米转换为米
    if pcd and pcd.has_points():
        points = np.asarray(pcd.points)
        # 转换为米单位（除以1000）
        points_meters = points / 1000.0
        # 更新点云
        pcd.points = o3d.utility.Vector3dVector(points_meters)
        
        # 计算点云的数值范围（米单位）
        min_coords = np.min(points_meters, axis=0)
        max_coords = np.max(points_meters, axis=0)
        print(f"点云数值范围 (米):")
        print(f"  X: {min_coords[0]:.4f} ~ {max_coords[0]:.4f} m")
        print(f"  Y: {min_coords[1]:.4f} ~ {max_coords[1]:.4f} m")
        print(f"  Z: {min_coords[2]:.4f} ~ {max_coords[2]:.4f} m")
    
    return pcd

def save_point_cloud(pcd, file_path):
    """
    保存点云到文件（只保留三维坐标）
    """
    try:
        # 确保输出目录存在
        output_dir = os.path.dirname(file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 创建只包含三维坐标的新点云
        pcd_only_coords = o3d.geometry.PointCloud()
        pcd_only_coords.points = pcd.points  # 只复制点坐标，不复制其他属性
        
        # 保存点云
        o3d.io.write_point_cloud(file_path, pcd_only_coords)
        print(f"成功保存点云到: {file_path} (仅保留三维坐标)")
        return True
    except Exception as e:
        print(f"错误: 保存点云时出错: {e}")
        return False

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

def visualize_point_clouds(point_clouds, modes, color_scheme='standard'):
    """
    可视化多个点云（每种采样模式单独可视化）
    """
    if not point_clouds:
        print("错误: 没有点云可可视化。")
        return
    
    print("\n开始可视化点云...")
    print("提示: 按ESC键退出当前窗口，将显示下一个点云。")
    
    for i, (pcd, mode) in enumerate(zip(point_clouds, modes)):
        if pcd and pcd.has_points():
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
            print(f"可视化 {mode} 采样点云 ({len(pcd.points)} 个点)。")
            o3d.visualization.draw_geometries(
                [pcd_copy], 
                window_name=f"{mode} 采样结果 - {color_schemes[color_scheme]['name']}",
                width=720,
                height=405
            )
        else:
            print(f"错误: {mode} 采样点云无效，跳过可视化。")
    
    print("\n所有点云可视化完成。")

def main():
    parser = argparse.ArgumentParser(description="STL模型转点云采样脚本")
    parser.add_argument('--input', type=str, default=r'D:\Application\Matlab\matlab files\202504-pointcloud\MiC\MiC-1.stl', help='STL模型文件路径。')
    parser.add_argument('--modes', type=str, nargs='+', default=['uniform'], 
                        choices=['uniform', 'poisson', 'random', 'area', 'curvature', 'adaptive', 'vertex', 'normal'],
                        help='采样模式: uniform (均匀采样), poisson (泊松采样), random (随机采样), area (面积采样), curvature (曲率采样), adaptive (自适应采样), vertex (顶点采样), normal (法线指导采样)。')
    parser.add_argument('--density', type=int, default=50000, help='采样密度（点云点数）。')
    parser.add_argument('--save', type=bool, default=True, help='保存采样后的点云。')
    parser.add_argument('--output', type=str, default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\experiments\geotransformer.custom.stage4.gse.k3.max.oacl.stage2.sinkhorn\mytools\mypointcloud', help='保存采样后点云的文件路径。')
    parser.add_argument('--visualize', type=bool, default=True, help='可视化采样后的点云。')
    parser.add_argument('--color-scheme', type=str, default='sci_professional', 
                        choices=list(color_schemes.keys()),
                        help='可视化配色方案: standard, grayscale, sci_professional, sci_colorblind, sci_gradient, sci_monochrome, registration。')
    args = parser.parse_args()

    # 加载STL模型
    mesh = load_stl_model(args.input)
    if not mesh:
        return

    # 对每种采样模式进行采样
    point_clouds = []
    for mode in args.modes:
        pcd = sample_point_cloud(mesh, mode, args.density)
        if pcd:
            point_clouds.append(pcd)

    if not point_clouds:
        print("错误: 未能生成任何点云。")
        return

    # 保存点云
    if args.save:
        if args.output:
            # 保存到指定路径
            base_output = args.output
            # 确保输出是目录或文件路径
            if os.path.isdir(base_output):
                # 如果是目录，为每种采样模式创建单独的文件
                base_name = os.path.splitext(os.path.basename(args.input))[0]
                for pcd, mode in zip(point_clouds, args.modes):
                    save_path = os.path.join(base_output, f"{base_name}_{mode}_{args.density}.ply")
                    save_point_cloud(pcd, save_path)
            else:
                # 如果是文件路径，为每种采样模式创建单独的文件
                base_name = os.path.splitext(base_output)[0]
                ext = os.path.splitext(base_output)[1]
                if not ext:
                    ext = '.ply'
                for pcd, mode in zip(point_clouds, args.modes):
                    save_path = f"{base_name}_{mode}{ext}"
                    save_point_cloud(pcd, save_path)
        else:
            # 保存到默认路径
            base_name = os.path.splitext(os.path.basename(args.input))[0]
            output_dir = os.path.join(os.path.dirname(args.input), 'sampled_pointclouds')
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            for pcd, mode in zip(point_clouds, args.modes):
                save_path = os.path.join(output_dir, f"{base_name}_{mode}_{args.density}.ply")
                save_point_cloud(pcd, save_path)

    # 可视化点云
    if args.visualize:
        visualize_point_clouds(point_clouds, args.modes, args.color_scheme)

    print("\nSTL模型点云采样完成。")


if __name__ == "__main__":
    main()