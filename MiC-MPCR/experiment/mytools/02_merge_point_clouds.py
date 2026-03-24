import os
import argparse
import numpy as np


# =========================================================================
#               点云文件合并脚本
# =========================================================================
#
# 功能:
#   - 将指定文件夹下的所有txt点云文件合并为一个txt点云文件。
#   - 支持.xzy格式的点云文件。
#
# 使用方法 (在终端中运行):
#   python 02_merge_point_clouds.py --input_folder "path/to/your/point_cloud_folder" --output_file "merged_points.txt"
#
# =========================================================================

def merge_point_clouds(input_folder, output_file):
    """
    将文件夹下的所有txt点云文件合并为一个txt文件。
    
    Args:
        input_folder: 包含点云文件的文件夹路径
        output_file: 输出合并后点云的文件路径
    """
    # 检查输入文件夹是否存在
    if not os.path.isdir(input_folder):
        print(f"错误: 输入文件夹 '{input_folder}' 不存在。")
        return False
    
    # 获取所有txt文件
    txt_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    if not txt_files:
        print("错误: 未找到任何txt文件。")
        return False
    
    print(f"找到 {len(txt_files)} 个txt文件:")
    for i, txt_file in enumerate(txt_files):
        print(f"  {i+1}. {txt_file}")
    
    # 存储所有点云数据
    all_points = []
    
    # 读取每个txt文件
    for txt_file in txt_files:
        file_path = os.path.join(input_folder, txt_file)
        try:
            # 读取点云数据
            points = np.loadtxt(file_path)
            if points.ndim == 1:
                # 如果只有一个点，添加维度
                points = points.reshape(1, -1)
            
            # 确保点云数据是3维的
            if points.shape[1] >= 3:
                # 只取前3列（x, y, z）
                points = points[:, :3]
                all_points.append(points)
                print(f"成功读取 {txt_file}，包含 {points.shape[0]} 个点。")
            else:
                print(f"警告: {txt_file} 不是有效的点云文件（维度不足）。")
        except Exception as e:
            print(f"错误: 读取 {txt_file} 时出错: {e}")
    
    if not all_points:
        print("错误: 未能读取任何有效的点云数据。")
        return False
    
    # 合并所有点云数据
    merged_points = np.vstack(all_points)
    print(f"\n合并完成，总共包含 {merged_points.shape[0]} 个点。")
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 保存合并后的点云数据
    try:
        np.savetxt(output_file, merged_points, fmt='%.6f')
        print(f"成功保存合并后的点云到: {output_file}")
        return True
    except Exception as e:
        print(f"错误: 保存合并后的点云时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="点云文件合并脚本")
    parser.add_argument('--input_folder', type=str, 
                        default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\data\custom\RawData_world\MiC_1',
                        help='包含点云文件的文件夹路径。')
    parser.add_argument('--output_file', type=str, 
                        default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\data\custom\RawData_world\MiC_1_merged_points.txt',
                        help='输出合并后点云的文件路径。')
    args = parser.parse_args()

    print(f"--- 开始合并点云文件 ---")
    print(f"输入文件夹: {args.input_folder}")
    print(f"输出文件: {args.output_file}")
    print()

    success = merge_point_clouds(args.input_folder, args.output_file)
    
    if success:
        print("\n点云合并成功！")
    else:
        print("\n点云合并失败。")


if __name__ == "__main__":
    main()
