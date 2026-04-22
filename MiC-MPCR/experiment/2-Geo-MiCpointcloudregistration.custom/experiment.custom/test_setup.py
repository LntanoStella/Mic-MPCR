import os
import sys

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入配置和数据集模块
from config import make_cfg
from dataset import create_dataset, train_valid_data_loader

def test_config_and_dataset():
    """测试配置和数据集设置"""
    print("开始测试新的实验目录设置...")
    
    # 加载配置
    print("\n1. 加载配置文件...")
    try:
        cfg = make_cfg()
        print(f"✅ 配置加载成功！")
        print(f"  - 实验名称: {cfg.exp_name}")
        print(f"  - 数据集根目录: {cfg.data.dataset_root}")
        print(f"  - 输出目录: {cfg.output_dir}")
    except Exception as e:
        print(f"❌ 配置加载失败: {e}")
        return False
    
    # 检查数据目录结构
    print("\n2. 检查数据目录结构...")
    required_dirs = [
        cfg.data.dataset_root,
        cfg.data.point_cloud_dir,
        cfg.data.transforms_dir
    ]
    
    all_dirs_exist = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ 目录存在: {dir_path}")
        else:
            print(f"❌ 目录不存在: {dir_path}")
            all_dirs_exist = False
    
    # 检查配对文件
    print("\n3. 检查配对文件...")
    pairs_files = [
        cfg.data.pairs_file,
        cfg.train.train_pairs_file,
        cfg.train.val_pairs_file,
        cfg.test.test_pairs_file
    ]
    
    all_files_exist = True
    for file_path in pairs_files:
        if file_path and os.path.exists(file_path):
            print(f"✅ 文件存在: {file_path}")
        else:
            print(f"⚠️ 文件不存在或未设置: {file_path}")
    
    # 尝试创建数据集（只读模式，不加载实际数据）
    print("\n4. 测试数据集创建...")
    try:
        dataset = create_dataset(cfg, 'test', is_train=False)
        print(f"✅ 数据集创建成功！")
        print(f"  - 数据集类型: {type(dataset).__name__}")
        print(f"  - 数据集长度: {len(dataset) if hasattr(dataset, '__len__') else '未知'}")
    except Exception as e:
        print(f"⚠️ 数据集创建可能存在问题: {e}")
    
    print("\n测试完成！")
    print("\n📋 建议下一步操作：")
    print("1. 确保自定义数据集目录结构正确")
    print("2. 准备train_pairs.txt, val_pairs.txt和test_pairs.txt文件")
    print("3. 运行训练命令: python trainval.py --config config.py")
    print("4. 运行评估命令: python eval.py --config config.py --snapshot <模型权重文件>")
    
    return True

if __name__ == "__main__":
    test_config_and_dataset()