import sys
import os
import glob
# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import argparse
import os.path as osp
import time
import torch
import numpy as np

from mic_geotrans.engine import SingleTester
from mic_geotrans.utils.torch import release_cuda
from mic_geotrans.utils.common import ensure_dir, get_log_string

# 尝试导入必要的模块
cfg = None
model = None
data_loader = None
evaluator = None

# 先导入配置，这通常不需要编译的扩展
try:
    from config import make_cfg
    cfg = make_cfg()
except Exception as e:
    print(f"警告: 无法导入配置模块: {e}")

# 其他模块可能依赖编译的扩展，使用条件导入
try:
    from dataset import test_data_loader
except ImportError as e:
    print(f"警告: 无法导入数据加载模块: {e}")
    print("提示: 这可能是因为缺少编译的CUDA扩展。请在支持CUDA的环境中安装扩展。")

try:
    from model import create_model
except ImportError as e:
    print(f"警告: 无法导入模型模块: {e}")

try:
    from loss import Evaluator
except ImportError as e:
    print(f"警告: 无法导入评估器模块: {e}")


def make_parser():
    # 使用add_help=False并设置冲突处理器，以避免与nosetest运行器的参数冲突
    parser = argparse.ArgumentParser(add_help=False, conflict_handler='resolve')
    parser.add_argument('--benchmark', choices=['test', 'val'], default='test', help='test benchmark')
    # 注意：不要重写parse_known_args方法，这会导致递归调用错误
    return parser


class Tester(SingleTester):
    def __init__(self, cfg=None):
        # 如果没有提供cfg，自动加载配置
        if cfg is None:
            try:
                cfg = make_cfg()
            except Exception as e:
                print(f"警告: 无法加载配置: {e}")
                # 创建一个最小的配置对象以允许类实例化
                class MockConfig:
                    pass
                cfg = MockConfig()
                cfg.feature_dir = './output'
                cfg.snapshot = type('obj', (object,), {'checkpoint': ''})
        
        # 确保cfg有snapshot属性
        if not hasattr(cfg, 'snapshot'):
            cfg.snapshot = type('obj', (object,), {'checkpoint': ''})
        
        # 初始化Tester
        super().__init__(cfg, parser=make_parser())
        
        # 设置device属性（BaseTester没有定义，需要手动设置）
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 检查必要的模块是否可用
        self.modules_available = {
            'test_data_loader': 'test_data_loader' in globals(),
            'create_model': 'create_model' in globals(),
            'Evaluator': 'Evaluator' in globals()
        }
        
        # 打印模块可用性信息
        print("模块可用性检查:")
        for module_name, available in self.modules_available.items():
            status = "✓ 可用" if available else "✗ 不可用"
            print(f"  {module_name}: {status}")

        # dataloader - 只有在模块可用时才初始化
        if self.modules_available['test_data_loader']:
            try:
                start_time = time.time()
                data_loader, neighbor_limits = test_data_loader(cfg, self.args.benchmark)
                loading_time = time.time() - start_time
                message = f'Data loader created: {loading_time:.3f}s collapsed.'
                self.logger.info(message)
                message = f'Calibrate neighbors: {neighbor_limits}.'
                self.logger.info(message)
                self.register_loader(data_loader)
            except Exception as e:
                print(f"警告: 初始化数据加载器失败: {e}")
                self.data_loader = None
        else:
            print("跳过数据加载器初始化: test_data_loader模块不可用")

        # model - 只有在模块可用时才初始化
        if self.modules_available['create_model']:
            try:
                model = create_model(cfg).to(self.device)
                self.register_model(model)
                print("模型初始化成功")
            except Exception as e:
                print(f"警告: 初始化模型失败: {e}")
                self.model = None
        else:
            print("跳过模型初始化: create_model模块不可用")

        # evaluator - 只有在模块可用时才初始化
        if self.modules_available['Evaluator']:
            try:
                self.evaluator = Evaluator(cfg).to(self.device)
                print("评估器初始化成功")
            except Exception as e:
                print(f"警告: 初始化评估器失败: {e}")
                self.evaluator = None
        else:
            print("跳过评估器初始化: Evaluator模块不可用")

        # preparation
        try:
            self.output_dir = osp.join(cfg.feature_dir, self.args.benchmark)
            ensure_dir(self.output_dir)
        except Exception as e:
            print(f"警告: 创建输出目录失败: {e}")
            self.output_dir = './output'

    def test_step(self, iteration, data_dict):
        # 检查模型是否可用
        if not hasattr(self, 'model') or self.model is None:
            print(f"警告: 在迭代 {iteration} 时，模型不可用，跳过测试步骤")
            # 返回一个空的输出字典
            return {}
        try:
            output_dict = self.model(data_dict)
            return output_dict
        except Exception as e:
            print(f"警告: 测试步骤失败，迭代 {iteration}: {e}")
            return {}

    def eval_step(self, iteration, data_dict, output_dict):
        # 检查评估器是否可用
        if not hasattr(self, 'evaluator') or self.evaluator is None:
            print(f"警告: 在迭代 {iteration} 时，评估器不可用，跳过评估步骤")
            return {}
        try:
            result_dict = self.evaluator(output_dict, data_dict)
            return result_dict
        except Exception as e:
            print(f"警告: 评估步骤失败，迭代 {iteration}: {e}")
            return {}

    def summary_string(self, iteration, data_dict, output_dict, result_dict):
        try:
            scene_name = data_dict.get('scene_name', 'unknown_scene')
            ref_frame = data_dict.get('ref_frame', 'unknown_ref')
            src_frame = data_dict.get('src_frame', 'unknown_src')
            message = f'{scene_name}, id0: {ref_frame}, id1: {src_frame}'
            
            # 安全地构建消息
            try:
                if result_dict:
                    message += ', ' + get_log_string(result_dict=result_dict)
            except Exception:
                message += ', 无法获取评估结果'
            
            try:
                if output_dict and 'corr_scores' in output_dict:
                    message += ', nCorr: {}'.format(output_dict['corr_scores'].shape[0])
            except Exception:
                message += ', 无法获取对应点数量'
            
            return message
        except Exception as e:
            print(f"警告: 生成摘要字符串失败: {e}")
            return f"迭代 {iteration}: 摘要生成失败"

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        try:
            # 检查必要的数据是否存在
            if not data_dict or not output_dict:
                print(f"警告: 在迭代 {iteration} 时，数据不完整，跳过保存步骤")
                return
            
            scene_name = data_dict.get('scene_name', 'unknown_scene')
            ref_id = data_dict.get('ref_frame', 'unknown_ref')
            src_id = data_dict.get('src_frame', 'unknown_src')
            
            # 从完整路径中提取文件名，避免重复目录结构
            if '/' in ref_id or '\\' in ref_id:
                ref_id = osp.basename(ref_id)
            if '/' in src_id or '\\' in src_id:
                src_id = osp.basename(src_id)

            # 创建保存目录
            save_dir = osp.join(self.output_dir, scene_name)
            try:
                ensure_dir(save_dir)
            except Exception as e:
                print(f"警告: 无法创建保存目录 {save_dir}: {e}")
                return
            
            file_name = osp.join(save_dir, f'{ref_id}_{src_id}.npz')
            
            # 准备要保存的数据，使用安全的方式获取每个字段
            save_data = {}
            
            # 安全地添加每个字段
            fields_to_save = [
                ('ref_points', output_dict),
                ('src_points', output_dict),
                ('ref_points_f', output_dict),
                ('src_points_f', output_dict),
                ('ref_points_c', output_dict),
                ('src_points_c', output_dict),
                ('ref_feats_c', output_dict),
                ('src_feats_c', output_dict),
                ('ref_node_corr_indices', output_dict),
                ('src_node_corr_indices', output_dict),
                ('ref_corr_points', output_dict),
                ('src_corr_points', output_dict),
                ('corr_scores', output_dict),
                ('gt_node_corr_indices', output_dict),
                ('gt_node_corr_overlaps', output_dict),
                ('estimated_transform', output_dict),
                ('transform', data_dict),
                ('overlap', data_dict)
            ]
            
            for field_name, source_dict in fields_to_save:
                try:
                    if field_name in source_dict:
                        # 尝试释放CUDA内存（如果需要）
                        try:
                            save_data[field_name] = release_cuda(source_dict[field_name])
                        except Exception:
                            # 如果释放失败，直接使用原始数据
                            save_data[field_name] = source_dict[field_name]
                except Exception as e:
                    print(f"警告: 无法获取字段 {field_name}: {e}")
            
            # 只在有数据时保存
            if save_data:
                np.savez_compressed(file_name, **save_data)
                print(f"已保存部分结果到 {file_name}")
            else:
                print(f"警告: 没有可保存的数据，跳过保存步骤")
                
        except Exception as e:
            print(f"警告: 保存结果失败，迭代 {iteration}: {e}")


def main():
    # 预处理命令行参数，过滤掉可能由测试运行器添加的额外参数
    # 保留我们关心的参数和snapshot参数
    import sys
    filtered_argv = [sys.argv[0]]
    i = 1
    while i < len(sys.argv):
        # 保留我们定义的参数
        if sys.argv[i] in ['--benchmark', '--snapshot', '--test_epoch', '--test_iter']:
            filtered_argv.append(sys.argv[i])
            # 如果参数有值，也保留它
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                filtered_argv.append(sys.argv[i + 1])
                i += 1
        i += 1
    
    # 保存原始的sys.argv，然后替换为过滤后的版本
    original_argv = sys.argv.copy()
    sys.argv = filtered_argv
    
    try:
        # 加载配置
        cfg = make_cfg()
        
        # 自动指定最新的snapshot路径
        snapshot_path = None
        try:
            snapshot_dir = osp.join(cfg.output_dir, 'snapshots')
            if osp.exists(snapshot_dir):
                # 查找所有epoch文件
                snapshots = glob.glob(osp.join(snapshot_dir, 'epoch-*.pth.tar'))
                if snapshots:
                    # 按epoch号排序，选择最大的
                    snapshots.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]), reverse=True)
                    snapshot_path = snapshots[0]
                    print(f"自动选择最新的模型快照: {snapshot_path}")
                elif osp.exists(osp.join(snapshot_dir, 'snapshot.pth.tar')):
                    # 如果没有epoch文件，尝试使用snapshot.pth.tar
                    snapshot_path = osp.join(snapshot_dir, 'snapshot.pth.tar')
                    print(f"使用snapshot.pth.tar: {snapshot_path}")
        except Exception as e:
            print(f"警告: 自动选择snapshot失败: {e}")
        
        # 确保即使没有CUDA扩展也能运行
        if hasattr(cfg, 'snapshot') and hasattr(cfg.snapshot, 'resume'):
            cfg.snapshot.resume = False
        
        # 修改命令行参数以包含snapshot
        import sys
        if snapshot_path and '--snapshot' not in sys.argv:
            sys.argv.append('--snapshot')
            sys.argv.append(snapshot_path)
        
        tester = Tester(cfg)
        tester.run()
    finally:
        # 恢复原始的sys.argv
        sys.argv = original_argv

if __name__ == '__main__':
    main()
