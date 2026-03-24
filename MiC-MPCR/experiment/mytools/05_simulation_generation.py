import numpy as np
import os
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import random
import time
import itertools


def normalize_path(path):
    """
    规范化路径，处理跨平台路径问题，包括Windows和WSL路径
    确保路径中正确包含GeoTransformer-main目录
    """
    # 处理Windows路径
    if '\\' in path:
        # 确保路径包含GeoTransformer-main
        if 'PointCloud_registration' in path and 'GeoTransformer-main' not in path:
            parts = path.split('\\')
            if 'PointCloud_registration' in parts:
                idx = parts.index('PointCloud_registration')
                # 插入GeoTransformer-main
                new_parts = parts[:idx+1] + ['GeoTransformer-main'] + parts[idx+1:]
                path = '\\'.join(new_parts)
        return path
    # 处理WSL路径
    elif path.startswith('/mnt/'):
        # 确保路径包含GeoTransformer-main
        if 'PointCloud_registration' in path and 'GeoTransformer-main' not in path:
            parts = path.split('/')
            if 'PointCloud_registration' in parts:
                idx = parts.index('PointCloud_registration')
                # 插入GeoTransformer-main
                new_parts = parts[:idx+1] + ['GeoTransformer-main'] + parts[idx+1:]
                path = '/'.join(new_parts)
        return path
    # 处理其他路径格式
    return path


# =========================================================================
#               批量数据增强Python脚本 (最终修正版)
# =========================================================================
#
# 功能:
#   - 修正了之前版本中点云与变换矩阵不匹配的严重逻辑错误。
#   - 读取“结构化黄金标准”数据集。
#   - 通过对局部点云和相对位姿施加多层次、且数学一致的随机扰动，
#     快速生成大量增强数据。
#   - 将增强后的数据按照与“黄金标准”相同的结构保存，并生成新的索引文件。
#
# =========================================================================

def get_random_transform(rotation_range_deg, translation_range_m):
    """生成一个小的随机4x4齐次变换矩阵。"""
    angle_x = np.deg2rad(np.random.uniform(rotation_range_deg[0], rotation_range_deg[1]))
    angle_y = np.deg2rad(np.random.uniform(rotation_range_deg[0], rotation_range_deg[1]))
    angle_z = np.deg2rad(np.random.uniform(rotation_range_deg[0], rotation_range_deg[1]))
    Rx = np.array([[1, 0, 0], [0, np.cos(angle_x), -np.sin(angle_x)], [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)], [0, 1, 0], [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0], [np.sin(angle_z), np.cos(angle_z), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    t = np.random.uniform(translation_range_m[0], translation_range_m[1], 3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def process_and_augment(variation_idx, golden_data, args):
    """
    处理单个数据增强迭代的核心工作函数。
    """

    # --- 1. 为每个传感器生成独立的位姿扰动 (模拟安装误差) ---
    sensor_perturbations = [get_random_transform(
        [-args.sensor_rot_deg, args.sensor_rot_deg],
        [-args.sensor_trans_m, args.sensor_trans_m]
    ) for _ in range(golden_data['num_sensors'])]

    # --- 2. 对“黄金标准”局部点云应用扰动和增强 ---
    final_local_pcds = []
    for i in range(golden_data['num_sensors']):
        pcd_orig_local = golden_data['local_pcds'][i]

        if pcd_orig_local.shape[0] == 0:
            final_local_pcds.append(pcd_orig_local)
            continue

        # 核心修正：首先，根据传感器位姿扰动，变换点云本身
        # P_new_local = inv(T_perturb) * P_old_local
        T_perturb_inv = np.linalg.inv(sensor_perturbations[i])
        pcd_orig_hom = np.hstack((pcd_orig_local, np.ones((pcd_orig_local.shape[0], 1))))
        pcd_perturbed_local = (T_perturb_inv @ pcd_orig_hom.T).T[:, :3]

        pcd_aug = pcd_perturbed_local.copy()

        # 接着，对这个新的、被扰动过的点云，再应用点云级增强
        # 添加高斯噪声
        noise = np.random.normal(0, args.noise_std, pcd_aug.shape)
        pcd_aug += noise

        # 随机下采样
        if args.downsample_n > 0 and pcd_aug.shape[0] > args.downsample_n:
            indices = np.random.choice(pcd_aug.shape[0], args.downsample_n, replace=False)
            pcd_aug = pcd_aug[indices, :]

        # 随机点丢失
        if random.random() < args.dropout_ratio and pcd_aug.shape[0] > 10:
            center_point = pcd_aug[np.random.choice(pcd_aug.shape[0]), :]
            distances = np.linalg.norm(pcd_aug - center_point, axis=1)
            pcd_aug = pcd_aug[distances > args.dropout_radius, :]

        final_local_pcds.append(pcd_aug)

    # --- 3. 计算与新位姿匹配的、新的“真值”相对变换 ---
    final_relative_transforms = {}
    for pair_key, T_orig_ij in golden_data['relative_transforms'].items():
        _, i_str, j_str = pair_key.split('_')
        i, j = int(i_str), int(j_str)

        T_perturb_i = sensor_perturbations[i - 1]
        T_perturb_j = sensor_perturbations[j - 1]

        # T_new = inv(T_perturb_i) * T_orig * T_perturb_j
        T_new_ij = np.linalg.inv(T_perturb_i) @ T_orig_ij @ T_perturb_j
        final_relative_transforms[pair_key] = T_new_ij

    # --- 4. 保存增强后的数据和新的 manifest ---
    variation_folder_name = f'Variation_{variation_idx:04d}'
    variation_folder_path = os.path.join(args.output_folder, variation_folder_name)
    pcd_output_dir = os.path.join(variation_folder_path, args.pcd_subdir)
    transform_output_dir = os.path.join(variation_folder_path, args.transform_subdir)
    os.makedirs(pcd_output_dir, exist_ok=True)
    os.makedirs(transform_output_dir, exist_ok=True)

    for i, pcd in enumerate(final_local_pcds):
        if pcd.shape[0] > 0:
            np.save(os.path.join(pcd_output_dir, f'lidar_{i + 1}.npy'), pcd.astype(np.float32))

    manifest_lines_for_this_variation = []
    for pair_key, T_new_ij in final_relative_transforms.items():
        _, i_str, j_str = pair_key.split('_')
        ref_idx, src_idx = int(i_str), int(j_str)

        ref_pcd_path = os.path.join(pcd_output_dir, f'lidar_{ref_idx}.npy')
        src_pcd_path = os.path.join(pcd_output_dir, f'lidar_{src_idx}.npy')
        if not os.path.exists(ref_pcd_path) or not os.path.exists(src_pcd_path):
            continue

        transform_path = os.path.join(transform_output_dir, f'transform_{ref_idx}_{src_idx}.npy')
        np.save(transform_path, T_new_ij.astype(np.float32))

        line = (f"{os.path.join(args.scene_name, variation_folder_name, args.pcd_subdir, f'lidar_{ref_idx}.npy')} "
                f"{os.path.join(args.scene_name, variation_folder_name, args.pcd_subdir, f'lidar_{src_idx}.npy')} "
                f"{os.path.join(args.scene_name, variation_folder_name, args.transform_subdir, f'transform_{ref_idx}_{src_idx}.npy')}")
        manifest_lines_for_this_variation.append(line)

    return manifest_lines_for_this_variation


def main():
    # 获取脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 获取项目根目录
    # 显式构建包含GeoTransformer-main的路径
    project_root_candidates = [
        # 直接向上三级目录（标准路径情况）
        os.path.abspath(os.path.join(script_dir, '..', '..', '..')),
        # 备选方案：如果路径中缺少GeoTransformer-main，手动构建正确路径
        os.path.abspath(os.path.join(script_dir, '..', '..', '..', 'GeoTransformer-main'))
    ]
    
    # 选择存在的、且包含GeoTransformer-main的项目根目录
    project_root = None
    for candidate in project_root_candidates:
        if os.path.exists(candidate) and 'GeoTransformer-main' in candidate.replace('\\', '/'):
            project_root = candidate
            break
    
    # 如果找不到合适的项目根目录，使用默认值但发出警告
    if project_root is None:
        project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
        print(f"警告: 未找到明确包含GeoTransformer-main的项目根目录，使用默认路径: {project_root}")
    
    parser = argparse.ArgumentParser(description="从“结构化黄金标准”数据批量生成增强后的训练数据集。")

    # 构建默认路径，确保包含GeoTransformer-main
    default_input_folder = os.path.join(project_root, 'data', 'custom', 'RawData_local')
    default_output_folder = os.path.join(project_root, 'data', 'custom', 'random_Dataset')
    
    parser.add_argument('--input_folder', type=str,
                        default=default_input_folder,
                        help='包含 manifest.txt 和 .npy 文件的“结构化黄金标准”文件夹路径。')
    parser.add_argument('--output_folder', type=str,
                        default=default_output_folder,
                        help='保存所有增强后数据的根目录。')
    parser.add_argument('--scene_name', type=str,
                        default='MiC_1',
                        help='场景名称，用于在manifest.txt中添加路径前缀。')
    parser.add_argument('--num_variations', type=int, default=2, help='要生成的数据增强组的总数。')
    parser.add_argument('--num_workers', type=int, default=max(1, cpu_count() - 2), help='用于并行处理的CPU核心数。')
    parser.add_argument('--pcd_subdir', type=str, default='point_clouds', help='存放点云的子文件夹名称。')
    parser.add_argument('--transform_subdir', type=str, default='transforms', help='存放变换矩阵的子文件夹名称。')

    parser.add_argument('--sensor_rot_deg', type=float, default=5.0, help='模拟安装误差的最大随机旋转角度(度)。')
    parser.add_argument('--sensor_trans_m', type=float, default=0.5, help='模拟安装误差的最大随机平移距离(米)。')
    parser.add_argument('--noise_std', type=float, default=0.01, help='高斯噪声的标准差(米)。')
    parser.add_argument('--downsample_n', type=int, default=0, help='随机下采样到的点数 (0表示不下采样)。')
    parser.add_argument('--dropout_ratio', type=float, default=0.1, help='触发随机点丢弃的概率。')
    parser.add_argument('--dropout_radius', type=float, default=0.3, help='随机点丢弃的半径(米)。')

    args = parser.parse_args()

    # 规范化路径
    args.input_folder = normalize_path(args.input_folder)
    args.output_folder = normalize_path(args.output_folder)
    
    # 检查并修复路径
    if not os.path.isdir(args.input_folder):
        # 尝试修复常见路径错误
        path_separator = '\\' if '\\' in args.input_folder else '/'
        parts = args.input_folder.split(path_separator)
        
        if 'PointCloud_registration' in parts and 'GeoTransformer-main' not in parts:
            idx = parts.index('PointCloud_registration')
            # 插入GeoTransformer-main
            new_parts = parts[:idx+1] + ['GeoTransformer-main'] + parts[idx+1:]
            fixed_path = path_separator.join(new_parts)
            if os.path.isdir(fixed_path):
                print(f"修复路径: {fixed_path}")
                args.input_folder = fixed_path
                
        # 再尝试另一种路径修复方案 - 手动构建包含GeoTransformer-main的完整路径
        if not os.path.isdir(args.input_folder):
            # 从项目根目录重新构建输入文件夹路径
            if 'RawData_local' in args.input_folder:
                scene_name = args.input_folder.split('RawData_local')[1].split(path_separator)[1] if path_separator in args.input_folder.split('RawData_local')[1] else ''
                fixed_path = os.path.join(project_root, 'data', 'custom', 'RawData_local', scene_name)
                if os.path.isdir(fixed_path):
                    print(f"使用项目根目录重新构建路径: {fixed_path}")
                    args.input_folder = fixed_path
        
        if not os.path.isdir(args.input_folder):
            print(f"错误: 输入文件夹 '{args.input_folder}' 不存在。")
            print(f"当前工作目录: {os.getcwd()}")
            print(f"项目根目录: {project_root}")
            return

    # 创建场景特定的输出文件夹
    scene_output_folder = os.path.join(args.output_folder, args.scene_name)
    os.makedirs(scene_output_folder, exist_ok=True)
    args.output_folder = scene_output_folder  # 更新输出文件夹为场景特定的文件夹
    print(f"增强数据将保存到: {args.output_folder}")

    print("--- 正在加载“黄金标准”数据 ---")
    golden_data = {}
    golden_data['local_pcds'] = []
    golden_data['relative_transforms'] = {}

    try:
        # 修正输入文件夹路径，确保包含场景名称
        scene_input_folder = os.path.join(args.input_folder, args.scene_name)
        pcd_input_dir = os.path.join(scene_input_folder, args.pcd_subdir)
        
        print(f"正在从以下路径加载数据: {scene_input_folder}")
        
        num_sensors = 4
        for i in range(1, num_sensors + 1):
            pcd_path = os.path.join(pcd_input_dir, f'lidar_{i}_points.npy')
            if not os.path.exists(pcd_path):
                print(f"警告: 未找到点云文件: {pcd_path}")
                # 尝试备选路径
                pcd_path = os.path.join(args.input_folder, args.scene_name, 'point_clouds', f'lidar_{i}_points.npy')
                if not os.path.exists(pcd_path):
                    raise FileNotFoundError(f"无法找到点云文件: {pcd_path}")
            golden_data['local_pcds'].append(np.load(pcd_path))
        golden_data['num_sensors'] = num_sensors

        transform_input_dir = os.path.join(scene_input_folder, args.transform_subdir)
        for i in range(1, num_sensors + 1):
            for j in range(i + 1, num_sensors + 1):
                key = f'transform_{i}_{j}'
                transform_path = os.path.join(transform_input_dir, f'{key}.npy')
                if not os.path.exists(transform_path):
                    # 尝试备选路径
                    transform_path = os.path.join(args.input_folder, args.scene_name, 'transforms', f'{key}.npy')
                    if not os.path.exists(transform_path):
                        raise FileNotFoundError(f"无法找到变换矩阵文件: {transform_path}")
                golden_data['relative_transforms'][key] = np.load(transform_path)

        print("“黄金标准”数据加载完毕。\n")
    except Exception as e:
        print(f"加载“黄金标准”数据失败: {e}")
        return

    start_time = time.time()
    worker_func = partial(process_and_augment, golden_data=golden_data, args=args)

    print(f"将使用 {args.num_workers} 个CPU核心并行生成 {args.num_variations} 组数据...\n")
    with Pool(processes=args.num_workers) as pool:
        list_of_manifest_lists = pool.map(worker_func, range(args.num_variations))

    all_manifest_lines = [line for sublist in list_of_manifest_lists for line in sublist]
    manifest_path = os.path.join(args.output_folder, 'manifest.txt')
    try:
        with open(manifest_path, 'w') as f:
            f.write("# ref_pcd_path  src_pcd_path  transform_path\n")
            for line in all_manifest_lines:
                f.write(line.replace('\\', '/') + '\n')
        print(f"\n成功创建总索引文件: {manifest_path}")
    except Exception as e:
        print(f"\n错误: 创建总索引文件失败: {e}")

    end_time = time.time()
    print(f"\n所有数据增强完毕！总耗时: {end_time - start_time:.2f} 秒。")


if __name__ == "__main__":
    main()