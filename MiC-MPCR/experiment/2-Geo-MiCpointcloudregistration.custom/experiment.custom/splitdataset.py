import os
import argparse
import random
from collections import defaultdict
import platform

# =========================================================================
#               数据集划分脚本
# =========================================================================
#
# 功能:
#   - 将 random_Dataset 下的所有场景数据集划分为训练集、验证集和测试集
#   - 支持两种划分策略：
#     1. 按变异文件夹划分（全局划分所有 Variation）
#     2. 按场景文件夹划分（整个场景作为一个整体分配）
#   - 生成对应的 train_pairs.txt、val_pairs.txt 和 test_pairs.txt 文件
#
# 使用方法 (在终端中运行):
#   python splitdataset.py --data_root "path/to/random_Dataset" --split_ratio "70/10/20" --split_by "variation"
#
# =========================================================================

def normalize_path(path):
    """
    标准化路径，处理WSL环境下的Windows路径转换
    - 在WSL中，将Windows风格路径(D:\folder)转换为WSL路径(/mnt/d/folder)
    - 统一路径分隔符为当前操作系统的标准分隔符
    """
    # 检查是否为Windows环境下的路径格式
    if '\\' in path or (len(path) >= 2 and path[1] == ':'):
        # 替换反斜杠为正斜杠
        path = path.replace('\\', '/')
        
        # 如果是WSL环境且路径是Windows驱动器格式，进行转换
        if platform.system() == 'Linux' and len(path) >= 2 and path[1] == ':':
            # 将D:/folder转换为/mnt/d/folder
            drive_letter = path[0].lower()
            path = f'/mnt/{drive_letter}/{path[3:]}' if len(path) > 3 else f'/mnt/{drive_letter}'
    
    return os.path.normpath(path)

def parse_args():
    parser = argparse.ArgumentParser(description="划分随机增强数据集为训练集、验证集和测试集")
    # 默认路径使用可在Windows和WSL下都能工作的格式
    default_path = 'd:\\Application\\PycharmProfessional\\pycharm\\PointCloud_registration\\GeoTransformer-main\\data\\custom\\random_Dataset'
    parser.add_argument('--data_root', type=str, 
                        default=default_path,
                        help='random_Dataset 根目录路径')
    parser.add_argument('--split_ratio', type=str, default='70/10/20',
                        help='训练/验证/测试集比例，格式如 "70/10/20"')
    parser.add_argument('--seed', type=int, default=216,
                        help='随机种子，保证可重复性')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='输出划分文件的目录，默认为 data_root')
    parser.add_argument('--split_by', type=str, choices=['variation', 'scene'], default='scene',
                        help='划分方式：variation（按变异文件夹）或 scene（按场景文件夹）')
    parser.add_argument('--verbose', action='store_true',
                        help='是否输出详细信息')
    return parser.parse_args()

def load_manifest_file(manifest_path):
    """
    加载 manifest.txt 文件
    """
    pairs = []
    try:
        with open(manifest_path, 'r') as f:
            for line_num, line in enumerate(f):
                # 跳过注释行和空行
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) != 3:
                    print(f"警告: manifest.txt 第{line_num + 1}行格式不正确: {line}")
                    continue
                
                ref_path, src_path, transform_path = parts
                pairs.append((ref_path, src_path, transform_path))
        return pairs
    except Exception as e:
        print(f"加载 manifest.txt 时出错: {e}")
        return []

def split_by_variation(scenes, args):
    """
    按变异文件夹划分数据
    """
    # 收集所有变异文件夹
    all_variations = []
    for scene_name, scene_data in scenes.items():
        for variation_name in scene_data['variations']:
            all_variations.append((scene_name, variation_name))
    
    # 打乱顺序
    random.shuffle(all_variations)
    
    # 计算划分索引
    train_ratio, val_ratio, test_ratio = map(float, args.split_ratio.split('/'))
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    
    # 计算基础数量
    train_count = int(len(all_variations) * train_ratio)
    val_count = int(len(all_variations) * val_ratio)
    
    # 确保验证集至少有一个变异文件夹（如果总变异数足够）
    if len(all_variations) >= 3 and val_count == 0:
        # 从训练集或测试集中调整一个变异到验证集
        if train_count > 1:
            train_count -= 1
            val_count = 1
        elif len(all_variations) - train_count > 1:  # 测试集变异数
            test_count = len(all_variations) - train_count
            test_count -= 1
            val_count = 1
    
    # 分配变异文件夹到不同集合
    train_variations = set(all_variations[:train_count])
    val_variations = set(all_variations[train_count:train_count + val_count])
    test_variations = set(all_variations[train_count + val_count:])
    
    return train_variations, val_variations, test_variations

def split_by_scene(scenes, args):
    """
    在每个场景内部按比例划分数据
    """
    train_scene_pairs = []
    val_scene_pairs = []
    test_scene_pairs = []
    
    # 计算划分比例
    train_ratio, val_ratio, test_ratio = map(float, args.split_ratio.split('/'))
    total = train_ratio + val_ratio + test_ratio
    train_ratio /= total
    val_ratio /= total
    
    # 对每个场景单独进行划分
    for scene_name, scene_data in scenes.items():
        pairs = scene_data['pairs']
        
        # 打乱场景内的数据对顺序
        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)
        
        # 计算当前场景内的划分索引
        total_pairs = len(shuffled_pairs)
        train_count = int(total_pairs * train_ratio)
        val_count = int(total_pairs * val_ratio)
        
        # 确保每个集合至少有一个数据对（如果数据对足够）
        if total_pairs >= 3:
            # 确保验证集至少有一个数据对
            if val_count == 0:
                val_count = 1
                # 从训练集或测试集中调整
                if train_count > 0:
                    train_count = max(1, train_count - 1)
                else:
                    test_count = total_pairs - train_count - val_count
                    if test_count > 1:
                        test_count -= 1
            
            # 确保训练集至少有一个数据对
            if train_count == 0:
                train_count = 1
                if val_count > 1:
                    val_count -= 1
                else:
                    test_count = total_pairs - train_count - val_count
                    if test_count > 1:
                        test_count -= 1
            
            # 确保测试集至少有一个数据对
            test_count = total_pairs - train_count - val_count
            if test_count == 0 and total_pairs >= 3:
                test_count = 1
                if train_count > 1:
                    train_count -= 1
                elif val_count > 1:
                    val_count -= 1
        
        # 将数据对添加到相应集合
        train_scene_pairs.extend([(scene_name, pair) for pair in shuffled_pairs[:train_count]])
        val_scene_pairs.extend([(scene_name, pair) for pair in shuffled_pairs[train_count:train_count + val_count]])
        test_scene_pairs.extend([(scene_name, pair) for pair in shuffled_pairs[train_count + val_count:]])
    
    return train_scene_pairs, val_scene_pairs, test_scene_pairs

def generate_pairs_files(scenes, train_set, val_set, test_set, output_dir, split_by, args):
    """
    生成训练/验证/测试配对文件
    """
    train_pairs = []
    val_pairs = []
    test_pairs = []
    
    if split_by == 'variation':
        # 按变异划分
        for scene_name, scene_data in scenes.items():
            manifest_pairs = scene_data['pairs']
            for ref_path, src_path, transform_path in manifest_pairs:
                # 从路径中提取变异名称
                variation_name = ref_path.split('/')[0]
                
                # 判断属于哪个集合
                if (scene_name, variation_name) in train_set:
                    # 直接使用原始路径，不添加场景名称前缀，避免路径重复
                    train_pairs.append(f"{ref_path} {src_path} {transform_path}")
                elif (scene_name, variation_name) in val_set:
                    val_pairs.append(f"{ref_path} {src_path} {transform_path}")
                elif (scene_name, variation_name) in test_set:
                    test_pairs.append(f"{ref_path} {src_path} {transform_path}")
    elif split_by == 'scene':
        # 按场景内部划分
        for scene_name, pair in train_set:
            ref_path, src_path, transform_path = pair
            train_pairs.append(f"{ref_path} {src_path} {transform_path}")
        
        for scene_name, pair in val_set:
            ref_path, src_path, transform_path = pair
            val_pairs.append(f"{ref_path} {src_path} {transform_path}")
        
        for scene_name, pair in test_set:
            ref_path, src_path, transform_path = pair
            test_pairs.append(f"{ref_path} {src_path} {transform_path}")
    
    # 写入文件
    os.makedirs(output_dir, exist_ok=True)
    
    # 写入训练集
    train_file = os.path.join(output_dir, 'train_pairs.txt')
    with open(train_file, 'w') as f:
        f.write(f"# 训练集配对文件 - 共{len(train_pairs)}个配对\n")
        for pair in train_pairs:
            f.write(f"{pair}\n")
    print(f"已生成训练集文件: {train_file}, 包含 {len(train_pairs)} 个配对")
    
    # 写入验证集
    val_file = os.path.join(output_dir, 'val_pairs.txt')
    with open(val_file, 'w') as f:
        f.write(f"# 验证集配对文件 - 共{len(val_pairs)}个配对\n")
        for pair in val_pairs:
            f.write(f"{pair}\n")
    print(f"已生成验证集文件: {val_file}, 包含 {len(val_pairs)} 个配对")
    
    # 写入测试集
    test_file = os.path.join(output_dir, 'test_pairs.txt')
    with open(test_file, 'w') as f:
        f.write(f"# 测试集配对文件 - 共{len(test_pairs)}个配对\n")
        for pair in test_pairs:
            f.write(f"{pair}\n")
    print(f"已生成测试集文件: {test_file}, 包含 {len(test_pairs)} 个配对")

def main():
    args = parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    
    # 标准化路径，确保在Windows和WSL环境下都能正常工作
    args.data_root = normalize_path(args.data_root)
    
    # 设置输出目录并标准化
    if args.output_dir:
        output_dir = normalize_path(args.output_dir)
    else:
        output_dir = args.data_root
    
    print(f"=== 开始数据集划分 ===")
    print(f"数据根目录: {args.data_root}")
    print(f"输出目录: {output_dir}")
    print(f"划分方式: {args.split_by}")
    print(f"划分比例: {args.split_ratio}")
    print(f"随机种子: {args.seed}")
    print(f"操作系统: {platform.system()}")
    print()
    
    # 扫描所有场景
    scenes = {}
    total_scenes = 0
    total_variations = 0
    total_pairs = 0
    
    print("--- 扫描场景和数据对 ---\n")
    
    # 获取所有场景文件夹
    for item in os.listdir(args.data_root):
        scene_path = os.path.join(args.data_root, item)
        if not os.path.isdir(scene_path):
            continue
        
        # 读取manifest.txt文件
        manifest_path = os.path.join(scene_path, 'manifest.txt')
        if not os.path.exists(manifest_path):
            print(f"警告: 场景 {item} 缺少 manifest.txt 文件，跳过")
            continue
        
        # 加载配对信息
        pairs = load_manifest_file(manifest_path)
        if not pairs:
            print(f"警告: 场景 {item} 的 manifest.txt 中未找到有效的配对信息，跳过")
            continue
        
        # 收集变异文件夹信息
        variations = set()
        for ref_path, src_path, transform_path in pairs:
            # 从路径中提取变异名称
            variation_name = ref_path.split('/')[0]
            variations.add(variation_name)
        
        scenes[item] = {
            'path': scene_path,
            'variations': sorted(list(variations)),
            'pairs': pairs
        }
        
        total_scenes += 1
        total_variations += len(variations)
        total_pairs += len(pairs)
        
        print(f"场景 {item}: 包含 {len(variations)} 个变异文件夹，{len(pairs)} 个数据对")
    
    print(f"\n总览: {total_scenes} 个场景, {total_variations} 个变异文件夹, {total_pairs} 个数据对")
    
    if not scenes:
        print("错误: 未找到有效的场景数据")
        return
    
    print("\n--- 执行数据划分 ---")
    
    if args.split_by == 'variation':
        # 按变异划分
        train_set, val_set, test_set = split_by_variation(scenes, args)
        print(f"按变异划分: 训练集 {len(train_set)} 个变异, 验证集 {len(val_set)} 个变异, 测试集 {len(test_set)} 个变异")
        
        if args.verbose:
            print("\n训练集变异:")
            for scene_name, variation_name in sorted(train_set):
                print(f"  {scene_name}/{variation_name}")
            
            print("\n验证集变异:")
            for scene_name, variation_name in sorted(val_set):
                print(f"  {scene_name}/{variation_name}")
            
            print("\n测试集变异:")
            for scene_name, variation_name in sorted(test_set):
                print(f"  {scene_name}/{variation_name}")
    else:
        # 按场景内部划分
        train_set, val_set, test_set = split_by_scene(scenes, args)
        print(f"按场景内部划分: 训练集 {len(train_set)} 个数据对, 验证集 {len(val_set)} 个数据对, 测试集 {len(test_set)} 个数据对")
        
        if args.verbose:
            # 统计每个场景在训练集中的数据对数量
            train_scene_counts = {}
            for scene_name, _ in train_set:
                train_scene_counts[scene_name] = train_scene_counts.get(scene_name, 0) + 1
            
            val_scene_counts = {}
            for scene_name, _ in val_set:
                val_scene_counts[scene_name] = val_scene_counts.get(scene_name, 0) + 1
            
            test_scene_counts = {}
            for scene_name, _ in test_set:
                test_scene_counts[scene_name] = test_scene_counts.get(scene_name, 0) + 1
            
            print("\n各场景训练集数据对数量:")
            for scene_name in sorted(train_scene_counts.keys()):
                print(f"  {scene_name}: {train_scene_counts[scene_name]} 个数据对")
            
            print("\n各场景验证集数据对数量:")
            for scene_name in sorted(val_scene_counts.keys()):
                print(f"  {scene_name}: {val_scene_counts[scene_name]} 个数据对")
            
            print("\n各场景测试集数据对数量:")
            for scene_name in sorted(test_scene_counts.keys()):
                print(f"  {scene_name}: {test_scene_counts[scene_name]} 个数据对")
    
    print("\n--- 生成配对文件 ---")
    generate_pairs_files(scenes, train_set, val_set, test_set, output_dir, args.split_by, args)
    
    print("\n=== 数据划分完成 ===")

if __name__ == "__main__":
    main()