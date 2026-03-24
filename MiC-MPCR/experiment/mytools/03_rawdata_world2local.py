import numpy as np
import os
import argparse
import glob
from scipy.io import loadmat
import itertools
import time


def normalize_path(path):
    """
    规范化路径，处理跨平台路径问题，特别是WSL和Windows路径转换
    确保路径中包含GeoTransformer-main目录
    """
    # 处理WSL路径
    if path.startswith('/mnt/'):
        # 确保路径包含GeoTransformer-main
        if 'PointCloud_registration' in path and 'GeoTransformer-main' not in path:
            parts = path.split('/')
            if 'PointCloud_registration' in parts:
                idx = parts.index('PointCloud_registration')
                # 插入GeoTransformer-main
                new_parts = parts[:idx+1] + ['GeoTransformer-main'] + parts[idx+1:]
                path = '/'.join(new_parts)
        return path
    # 处理Windows路径
    elif 'PointCloud_registration' in path and 'GeoTransformer-main' not in path:
        # 确保Windows路径也包含GeoTransformer-main
        parts = path.replace('\\', '/').split('/')
        if 'PointCloud_registration' in parts:
            idx = parts.index('PointCloud_registration')
            # 插入GeoTransformer-main
            new_parts = parts[:idx+1] + ['GeoTransformer-main'] + parts[idx+1:]
            path = '/'.join(new_parts)
    return path

# =========================================================================
#       从 MATLAB 数据创建结构化训练数据集的 Python 脚本
# =========================================================================
#
# 功能:
#   1. 读取由 simulation7.m 保存的点云文件夹 (Lidar_X_Points.txt) 和
#      .mat 配置文件。
#   2. 还原“真实传感器数据”(局部坐标系点云)，并将其保存到指定的
#      点云文件夹下 (例如 'output/point_clouds/lidar_1_points.npy')。
#   3. 计算所有传感器对之间的相对变换矩阵，并将其保存到指定的
#      变换文件夹下 (例如 'output/transforms/transform_1_2.npy')。
#   4. 在输出根目录生成一个 manifest.txt 文件，清晰地索引每一个
#      训练样本的参考点云、源点云和变换矩阵的相对路径。
#
# 使用方法 (在终端中运行):
#   python create_structured_dataset.py --pcd_folder "path/to/golden_standard" --mat_file "path/to/sensor_config.mat" --output_folder "path/to/training_dataset"
#
# =========================================================================

def calculate_poses_from_mat(mat_filepath, epsilon=1e-6):
    """
    从.mat配置文件中加载传感器设置，并计算每个传感器的绝对世界位姿。

    返回:
    - world_poses (list): 包含4个[4, 4] NumPy数组的列表。
    """
    try:
        data = loadmat(mat_filepath)
        if 'sensorConfig' not in data:
            print(f"错误: 在 {mat_filepath} 中未找到 'sensorConfig' 变量。")
            return None

        sensor_config = data['sensorConfig'][0]
        world_poses = []

        for s_config in sensor_config:
            pos = s_config['pos'][0]
            target = s_config['targetPos'][0]

            d_vec = target - pos
            norm_d = np.linalg.norm(d_vec)
            if norm_d > epsilon:
                d = d_vec / norm_d
            else:
                d = np.array([0, 0, -1])

            world_up = np.array([0, 0, 1])
            if abs(np.dot(d, world_up)) > (1 - epsilon):
                u = np.cross(np.array([0, 1, 0]), d)
            else:
                u = np.cross(world_up, d)
            u = u / np.linalg.norm(u)
            v = np.cross(d, u)

            R = np.vstack((u, v, d)).T
            T_matrix = np.eye(4)
            T_matrix[:3, :3] = R
            T_matrix[:3, 3] = pos
            world_poses.append(T_matrix)

        print(f"成功从 {mat_filepath} 计算了 {len(world_poses)} 个绝对世界位姿。")
        return world_poses

    except Exception as e:
        print(f"读取或处理 .mat 文件时出错: {e}")
        return None


def find_mat_file(pcd_folder):
    """
    在点云文件夹中查找.mat文件
    
    参数:
    - pcd_folder: 点云文件夹路径
    
    返回:
    - mat_file_path: 找到的.mat文件路径，如果未找到或找到多个则返回None
    """
    # 查找文件夹中的所有.mat文件
    mat_files = glob.glob(os.path.join(pcd_folder, '*.mat'))
    
    if len(mat_files) == 0:
        print(f"错误: 在 '{pcd_folder}' 中未找到任何.mat文件。")
        return None
    elif len(mat_files) > 1:
        print(f"警告: 在 '{pcd_folder}' 中找到多个.mat文件:")
        for i, mat_file in enumerate(mat_files):
            print(f"  {i+1}. {os.path.basename(mat_file)}")
        # 默认选择第一个文件
        selected_file = mat_files[0]
        print(f"  已自动选择: {os.path.basename(selected_file)}")
        return selected_file
    else:
        # 找到单个.mat文件
        print(f"成功找到.mat文件: {os.path.basename(mat_files[0])}")
        return mat_files[0]


def main():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 直接计算项目根目录 - 脚本在experiments/xxx/xxx/下，向上三级是GeoTransformer-main
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
    
    # 强制确保项目根目录包含GeoTransformer-main
    if not 'GeoTransformer-main' in os.path.basename(project_root):
        # 显式构建包含GeoTransformer-main的路径
        current_path = script_dir
        # 向上查找直到找到PointCloud_registration目录
        while current_path and os.path.dirname(current_path) != current_path:
            if 'PointCloud_registration' in os.path.basename(current_path):
                # 在PointCloud_registration后添加GeoTransformer-main
                project_root = os.path.join(current_path, 'GeoTransformer-main')
                break
            current_path = os.path.dirname(current_path)
    
    # 验证项目根目录是否存在
    if not os.path.isdir(project_root):
        print(f"警告: 计算的项目根目录 '{project_root}' 不存在，尝试使用默认路径结构")
        # 使用绝对路径构建正确的项目根目录
        default_project_root = 'd:\\Application\\PycharmProfessional\\pycharm\\PointCloud_registration\\GeoTransformer-main'
        if os.path.isdir(default_project_root):
            project_root = default_project_root
            print(f"使用默认项目根目录: {project_root}")
    
    parser = argparse.ArgumentParser(description="从 MATLAB 数据创建结构化的深度学习训练数据集。")

    # 构建默认路径，确保包含GeoTransformer-main
    default_pcd_folder = os.path.join(project_root, 'data', 'custom', 'RawData_world', 'MiC-3-ground-enhanced-2')
    default_output_folder = os.path.join(project_root, 'data', 'custom', 'RawData_local')
    
    parser.add_argument('--pcd_folder', type=str, 
                        default=default_pcd_folder,
                        help='包含 Lidar_X_Points.txt 文件和.mat配置文件的文件夹路径。')
    parser.add_argument('--output_folder', type=str, 
                        default=default_output_folder,
                        help='保存结构化数据集的根目录。')
    parser.add_argument('--scene_name', type=str, default='MiC-3-ground-enhanced-2',
                        help='场景名称，用于创建场景特定的子文件夹。')
    parser.add_argument('--pcd_subdir', type=str, default='point_clouds',
                        help='在输出目录中用于存放点云的子文件夹名称。')
    parser.add_argument('--transform_subdir', type=str, default='transforms',
                        help='在输出目录中用于存放变换矩阵的子文件夹名称。')

    args = parser.parse_args()

    # --- 0. 检查输入路径和查找.mat文件 ---
    # 规范化路径
    args.pcd_folder = normalize_path(args.pcd_folder)
    args.output_folder = normalize_path(args.output_folder)
    
    # 强制确保输出路径包含GeoTransformer-main
    if 'PointCloud_registration' in args.output_folder and 'GeoTransformer-main' not in args.output_folder:
        # 统一处理Windows和WSL路径
        separator = '/' if '/' in args.output_folder else '\\'
        parts = args.output_folder.split(separator)
        if 'PointCloud_registration' in parts:
            idx = parts.index('PointCloud_registration')
            # 插入GeoTransformer-main
            new_parts = parts[:idx+1] + ['GeoTransformer-main'] + parts[idx+1:]
            fixed_output_path = separator.join(new_parts)
            print(f"修正输出路径: {fixed_output_path}")
            args.output_folder = fixed_output_path
    
    # 检查并修复点云路径
    if not os.path.isdir(args.pcd_folder):
        # 尝试修复常见路径错误
        if 'PointCloud_registration' in args.pcd_folder and 'GeoTransformer-main' not in args.pcd_folder:
            # 统一处理Windows和WSL路径
            separator = '/' if '/' in args.pcd_folder else '\\'
            parts = args.pcd_folder.split(separator)
            if 'PointCloud_registration' in parts:
                idx = parts.index('PointCloud_registration')
                # 插入GeoTransformer-main
                new_parts = parts[:idx+1] + ['GeoTransformer-main'] + parts[idx+1:]
                fixed_path = separator.join(new_parts)
                if os.path.isdir(fixed_path):
                    print(f"修复路径: {fixed_path}")
                    args.pcd_folder = fixed_path
        
        if not os.path.isdir(args.pcd_folder):
            print(f"错误: 点云文件夹 '{args.pcd_folder}' 不存在。")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"项目根目录: {project_root}")
            return
    
    # 自动查找.mat文件
    mat_file = find_mat_file(args.pcd_folder)
    if mat_file is None:
        return

    # --- 1. 创建输出目录结构 ---
    # 创建场景特定的输出文件夹
    scene_output_folder = os.path.join(args.output_folder, args.scene_name)
    os.makedirs(scene_output_folder, exist_ok=True)
    
    pcd_output_dir = os.path.join(scene_output_folder, args.pcd_subdir)
    transform_output_dir = os.path.join(scene_output_folder, args.transform_subdir)
    os.makedirs(pcd_output_dir, exist_ok=True)
    os.makedirs(transform_output_dir, exist_ok=True)

    print(f"数据集将保存到: {scene_output_folder}")
    print(f"  - 点云子目录: {pcd_output_dir}")
    print(f"  - 变换子目录: {transform_output_dir}\n")
    
    # 更新输出文件夹为场景特定的文件夹，用于后续的相对路径计算
    args.output_folder = scene_output_folder

    start_time = time.time()

    # --- 2. 计算所有绝对世界位姿 ---
    world_poses = calculate_poses_from_mat(mat_file)
    if world_poses is None: return

    num_sensors = len(world_poses)

    # --- 3. 还原局部点云并保存 ---
    print("--- 正在还原并保存局部坐标系点云 ---")
    local_pcd_paths = []
    for i in range(num_sensors):
        sensor_idx = i + 1
        pcd_world_path = os.path.join(args.pcd_folder, f'Lidar_{sensor_idx}.txt')
        pcd_local_path = os.path.join(pcd_output_dir, f'lidar_{sensor_idx}_points.npy')
        local_pcd_paths.append(os.path.relpath(pcd_local_path, args.output_folder))

        if os.path.exists(pcd_world_path):
            pcd_world = np.loadtxt(pcd_world_path, usecols=(0, 1, 2), ndmin=2)
            if pcd_world.shape[0] > 0:
                T_sensor_world = np.linalg.inv(world_poses[i])
                pcd_world_hom = np.hstack((pcd_world, np.ones((pcd_world.shape[0], 1))))
                pcd_local = (T_sensor_world @ pcd_world_hom.T).T[:, :3]

                np.save(pcd_local_path, pcd_local.astype(np.float32))
                print(f"  已保存: {pcd_local_path} ({pcd_local.shape[0]} 个点)")
            else:
                # 如果原始文件为空，也创建一个空的 .npy 文件
                np.save(pcd_local_path, np.array([]).astype(np.float32))
                print(f"  警告: Lidar_{sensor_idx}.txt 为空, 已创建空的 .npy 文件。")
        else:
            print(f"  警告: 未找到 Lidar_{sensor_idx}.txt, 将创建空的 .npy 文件。")
            np.save(pcd_local_path, np.array([]).astype(np.float32))

    # --- 4. 计算相对变换矩阵并保存，同时生成 manifest ---
    print("\n--- 正在计算相对变换并生成 manifest 文件 ---")
    manifest_lines = []
    pair_indices = list(itertools.combinations(range(num_sensors), 2))

    for i, j in pair_indices:
        ref_idx = i + 1
        src_idx = j + 1

        # 计算 T_ij (从 j 变换到 i)
        T_ij = np.linalg.inv(world_poses[i]) @ world_poses[j]

        transform_path = os.path.join(transform_output_dir, f'transform_{ref_idx}_{src_idx}.npy')
        np.save(transform_path, T_ij.astype(np.float32))

        # 生成 manifest 行
        ref_pcd_path = local_pcd_paths[i]
        src_pcd_path = local_pcd_paths[j]
        transform_rel_path = os.path.relpath(transform_path, args.output_folder)

        manifest_lines.append(f"{ref_pcd_path} {src_pcd_path} {transform_rel_path}")

    # --- 5. 写入 manifest.txt 索引文件 ---
    manifest_path = os.path.join(args.output_folder, 'manifest.txt')
    try:
        with open(manifest_path, 'w') as f:
            f.write("# 参考点云路径 (ref)  源点云路径 (src)  变换矩阵路径 (transform)\n")
            for line in manifest_lines:
                f.write(line.replace('\\', '/') + '\n')  # 确保路径分隔符为 '/', 提高跨平台兼容性
        print(f"\n成功创建索引文件: {manifest_path}")
    except Exception as e:
        print(f"\n错误: 创建索引文件失败: {e}")

    end_time = time.time()
    print(f"\n所有数据处理完毕！总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == "__main__":
    main()