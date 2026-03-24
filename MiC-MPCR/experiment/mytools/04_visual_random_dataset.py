import numpy as np
import open3d as o3d
import os
import argparse
import copy


# =========================================================================
#               随机增强数据集可视化脚本
# =========================================================================
#
# 功能:
#   - 读取random_Dataset中的Variation_XXXX文件夹数据
#   - 逐对验证：循环显示每一对点云在配准前（局部坐标系）和配准后
#     （应用变换矩阵）的效果，以验证每个 transform_i_j.npy 的正确性。
#   - 全局验证：将所有点云变换到Lidar 1的坐标系下，显示最终的
#     完整拼接效果，以验证数据集的全局一致性。
#
# 使用方法 (在终端中运行):
#   python 04_visual_random_dataset.py --folder "path/to/your/Variation_XXXX"
#
# =========================================================================

def main():
    parser = argparse.ArgumentParser(description="可视化随机增强数据集的单个变异文件夹。")
    parser.add_argument('--folder', type=str, 
                        default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\data\custom\random_Dataset\MiC_8_01\Variation_0000',
                        help='要可视化的单个Variation_XXXX文件夹路径。')
    args = parser.parse_args()

    data_folder = args.folder
    print(f"--- 开始可视化随机增强数据集: {data_folder} ---")

    # --- 0. 路径和有效性检查 ---
    if not os.path.isdir(data_folder):
        print(f"错误: 文件夹 '{data_folder}' 不存在。")
        return

    pcd_dir = os.path.join(data_folder, 'point_clouds')
    transform_dir = os.path.join(data_folder, 'transforms')
    
    # 查找上一级目录中的manifest.txt
    parent_folder = os.path.dirname(data_folder)
    manifest_path = os.path.join(parent_folder, 'manifest.txt')

    if not os.path.isdir(pcd_dir) or not os.path.isdir(transform_dir):
        print("错误: 输入文件夹结构不完整。请确保包含 'point_clouds' 和 'transforms' 子文件夹。")
        return
    
    # manifest.txt可能不存在于当前变异文件夹中，只打印警告
    if not os.path.isfile(manifest_path):
        print(f"警告: 未找到索引文件 '{manifest_path}'，将跳过逐对验证环节。")

    # --- 1. 加载所有局部点云 ---
    num_sensors = 4
    local_point_clouds = []
    print("--- 正在加载所有局部坐标系点云 ---")
    for i in range(1, num_sensors + 1):
        # 注意：random_Dataset中的点云文件命名可能没有_points后缀
        pcd_path = os.path.join(pcd_dir, f'lidar_{i}.npy')
        if os.path.exists(pcd_path):
            points = np.load(pcd_path)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            local_point_clouds.append(pcd)
            print(f"  成功加载 Lidar {i} 的局部点云 ({len(pcd.points)} 个点)。")
        else:
            # 尝试带_points后缀的命名格式
            pcd_path = os.path.join(pcd_dir, f'lidar_{i}_points.npy')
            if os.path.exists(pcd_path):
                points = np.load(pcd_path)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                local_point_clouds.append(pcd)
                print(f"  成功加载 Lidar {i} 的局部点云 ({len(pcd.points)} 个点)。")
            else:
                local_point_clouds.append(o3d.geometry.PointCloud())
                print(f"  警告: 未找到 Lidar {i} 的点云文件。")

    # --- 2. 逐对可视化验证（如果manifest.txt存在）---
    if os.path.isfile(manifest_path):
        print("\n--- 可视化阶段 1: 逐对验证 ---")
        colors = [[1, 0.7, 0], [0, 0.65, 0.93]]  # 橙色 vs 蓝色

        with open(manifest_path, 'r') as f:
            # 跳过注释行
            lines = [line for line in f if not line.strip().startswith('#')]
        
        # 过滤出与当前变异文件夹相关的行
        variation_name = os.path.basename(data_folder)
        variation_lines = [line for line in lines if variation_name in line]
        
        if not variation_lines:
            print(f"警告: 在 {manifest_path} 中未找到与 {variation_name} 相关的条目。")
        else:
            for line in variation_lines:
                try:
                    ref_rel_path, src_rel_path, trans_rel_path = line.strip().split()
                    
                    # 从相对路径解析出传感器索引
                    try:
                        ref_idx = int(os.path.basename(ref_rel_path).split('_')[1])
                        src_idx = int(os.path.basename(src_rel_path).split('_')[1])
                    except:
                        ref_idx = int(os.path.basename(ref_rel_path).split('_')[1].split('.')[0])
                        src_idx = int(os.path.basename(src_rel_path).split('_')[1].split('.')[0])

                    print(f"\n>> 正在验证配准对: Lidar {ref_idx} (参考) vs Lidar {src_idx} (源)")

                    # 获取对应的局部点云
                    if ref_idx > 0 and ref_idx <= num_sensors and src_idx > 0 and src_idx <= num_sensors:
                        ref_pcd = local_point_clouds[ref_idx - 1]
                        src_pcd = local_point_clouds[src_idx - 1]
                    else:
                        print(f"  无效的传感器索引: ref={ref_idx}, src={src_idx}，跳过此对。")
                        continue

                    if not ref_pcd.has_points() or not src_pcd.has_points():
                        print("  其中一个点云为空，跳过此对。")
                        continue

                    # 直接从transforms文件夹加载变换矩阵
                    transform_filename = f'transform_{ref_idx}_{src_idx}.npy'
                    transform_path = os.path.join(transform_dir, transform_filename)
                    
                    if not os.path.exists(transform_path):
                        print(f"  未找到变换矩阵文件: {transform_path}，跳过此对。")
                        continue
                    
                    transform_matrix = np.load(transform_path)

                    # --- 可视化配准前 ---
                    print("  - 显示 [配准前] 状态 (两个点云都在各自的原点)...")
                    ref_pcd_before = copy.deepcopy(ref_pcd).paint_uniform_color(colors[0])
                    src_pcd_before = copy.deepcopy(src_pcd).paint_uniform_color(colors[1])
                    o3d.visualization.draw_geometries([ref_pcd_before, src_pcd_before],
                                                      window_name=f"配准前: Lidar {ref_idx} (橙) vs Lidar {src_idx} (蓝)")

                    # --- 可视化配准后 ---
                    print("  - 应用变换矩阵并显示 [配准后] 状态...")
                    ref_pcd_after = copy.deepcopy(ref_pcd).paint_uniform_color(colors[0])
                    src_pcd_after = copy.deepcopy(src_pcd).transform(transform_matrix).paint_uniform_color(colors[1])
                    o3d.visualization.draw_geometries([ref_pcd_after, src_pcd_after],
                                                      window_name=f"配准后: Lidar {ref_idx} vs Lidar {src_idx}")

                except Exception as e:
                    print(f"处理行 '{line.strip()}' 时出错: {e}")

    # --- 3. 全局可视化验证 ---
    print("\n--- 可视化阶段 2: 全局验证 ---")
    print("    - 将所有点云变换到 Lidar 1 的坐标系下进行最终拼接。")

    world_point_clouds = []

    # Lidar 1 不需要变换
    pcd1_world = copy.deepcopy(local_point_clouds[0])
    if pcd1_world.has_points():
        pcd1_world.paint_uniform_color([1, 0, 0])  # 红色
        world_point_clouds.append(pcd1_world)

    # 变换 Lidar 2, 3, 4 到 Lidar 1 的坐标系
    for i in range(2, num_sensors + 1):
        src_pcd_local = local_point_clouds[i - 1]
        if not src_pcd_local.has_points():
            continue

        # 寻找 T_1_i 的变换矩阵
        transform_path = os.path.join(transform_dir, f'transform_1_{i}.npy')
        if os.path.exists(transform_path):
            T_1_i = np.load(transform_path)
            src_pcd_world = copy.deepcopy(src_pcd_local).transform(T_1_i)

            # 根据索引上色
            color_map = {2: [0, 1, 0], 3: [0, 0, 1], 4: [1, 0, 1]}  # G, B, M
            src_pcd_world.paint_uniform_color(color_map[i])
            world_point_clouds.append(src_pcd_world)
        else:
            print(f"警告: 未找到全局配准所需的 transform_1_{i}.npy 文件。")

    # 创建最终的合并点云
    combined_pcd = o3d.geometry.PointCloud()
    for pcd in world_point_clouds:
        combined_pcd += pcd

    print("\n    - 显示所有点云拼接后的最终效果。")
    o3d.visualization.draw_geometries(world_point_clouds, window_name="全局拼接可视化 (所有点云变换到Lidar 1坐标系)")

    if combined_pcd.has_points():
        combined_pcd.paint_uniform_color([0.7, 0.7, 0.7])
        print("\n    - 显示统一颜色的最终合并结果。")
        o3d.visualization.draw_geometries([combined_pcd], window_name="最终合并的点云")

    print("\n可视化完毕。")


if __name__ == "__main__":
    main()