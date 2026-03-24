# -*- coding: utf-8 -*-
"""
打印npy文件内容工具

功能:
  - 读取npy格式的点云文件
  - 打印文件的基本信息和内容

输入:
  - npy文件路径

输出:
  - 文件的基本信息（形状、数据类型等）
  - 文件内容（前N行）

使用方法 (在终端中运行):
  python 08_print_npy_content.py --input path/to/file.npy
  python 08_print_npy_content.py --input path/to/file.npy --lines 20
"""

import argparse
import numpy as np


def print_npy_content(file_path, num_lines=10):
    """
    打印npy文件的内容
    
    参数:
        file_path: npy文件路径
        num_lines: 打印的行数（默认10行）
    """
    try:
        # 加载npy文件
        data = np.load(file_path)
        
        # 打印基本信息
        print("=" * 60)
        print(f"文件路径: {file_path}")
        print(f"数据形状: {data.shape}")
        print(f"数据类型: {data.dtype}")
        print(f"数据大小: {data.size} 个元素")
        print(f"内存占用: {data.nbytes / 1024:.2f} KB")
        print("=" * 60)
        
        # 打印内容
        print(f"\n文件内容（前{min(num_lines, len(data))}行）:")
        print("-" * 60)
        
        if data.ndim == 1:
            # 一维数组
            for i in range(min(num_lines, len(data))):
                print(f"[{i}] {data[i]}")
        elif data.ndim == 2:
            # 二维数组（点云数据）
            for i in range(min(num_lines, len(data))):
                row_str = "  ".join([f"{x:.6f}" for x in data[i]])
                print(f"[{i}] {row_str}")
        else:
            # 其他维度
            print(data[:num_lines])
        
        # 如果数据还有更多行，显示省略信息
        if len(data) > num_lines:
            print(f"\n... 还有 {len(data) - num_lines} 行数据 ...")
        
        print("=" * 60)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{file_path}'")
    except Exception as e:
        print(f"错误: 读取文件失败 - {e}")


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description='打印npy文件内容工具')
    parser.add_argument('--input', type=str, default=r'D:\Application\PycharmProfessional\pycharm\PointCloud_registration\GeoTransformer-main\data\custom\PSG_GT2\transform_3_4.npy',
                        help='npy文件路径')
    parser.add_argument('--lines', type=int, default=10,
                        help='打印的行数（默认10行）')
    
    args = parser.parse_args()
    
    # 打印npy文件内容
    print_npy_content(args.input, args.lines)


if __name__ == '__main__':
    main()
