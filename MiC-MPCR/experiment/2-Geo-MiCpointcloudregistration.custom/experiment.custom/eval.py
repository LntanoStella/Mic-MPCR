import argparse
import os.path as osp
import glob
import os
import torch
import numpy as np
from mic_geotrans.engine import Logger
from mic_geotrans.modules.registration import weighted_procrustes
from mic_geotrans.utils.summary_board import SummaryBoard
from mic_geotrans.utils.open3d import registration_with_ransac_from_correspondences
from mic_geotrans.utils.registration import (
    evaluate_sparse_correspondences,
    evaluate_correspondences,
    compute_registration_error,
)
from mic_geotrans.utils.torch import to_cuda
from config import make_cfg
from dataset import test_data_loader
from model import create_model

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_epoch', default=None, type=int, help='test epoch')
    parser.add_argument('--benchmark', choices=['test', 'val'], default='test', help='test benchmark')
    parser.add_argument('--method', choices=['lgr', 'ransac', 'svd'], default='svd', help='registration method')
    parser.add_argument('--num_corr', type=int, default=None, help='number of correspondences for registration')
    parser.add_argument('--verbose', action='store_true', help='verbose mode')
    return parser

def load_model(cfg, logger):
    """加载训练好的模型"""
    # 创建模型
    model = create_model(cfg)
    
    # 优先使用用户指定的权重目录
    custom_weight_dir = osp.join(cfg.working_dir, 'output', 'custom_geotransformer_01', 'snapshots')
    snapshot_dirs = [custom_weight_dir, cfg.snapshot_dir]
    
    snapshot_path = None
    
    # 尝试在多个目录中查找权重文件
    for dir_path in snapshot_dirs:
        if not osp.exists(dir_path):
            logger.warning(f'Weight directory not found: {dir_path}')
            continue
        
        # 优先查找snapshot.pth.tar
        if osp.exists(osp.join(dir_path, 'snapshot1.pth.tar')):
            # 使用snapshot.pth.tar
            snapshot_path = osp.join(dir_path, 'snapshot.pth.tar')
            break
        else:
            # 查找最大epoch的权重文件
            epoch_files = glob.glob(osp.join(dir_path, 'epoch-*.pth.tar'))
            if epoch_files:
                # 按epoch号排序，选择最大的
                epoch_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))
                snapshot_path = epoch_files[-1]
                break
    
    if snapshot_path is None:
        logger.error('No model weights found in any directory')
        return None
    
    logger.info(f'Loading model from {snapshot_path}')
    try:
        # 加载模型权重
        # 使用map_location参数确保在CPU上也能加载CUDA训练的模型
        checkpoint = torch.load(snapshot_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        logger.info(f'Model loaded successfully, epoch: {checkpoint.get("epoch", "unknown")}')
        return model
    except Exception as e:
        logger.error(f'Failed to load model: {e}')
        import traceback
        logger.error(traceback.format_exc())
        return None

def eval_one_epoch(args, cfg, logger):
    # 初始化评估指标
    registration_meter = SummaryBoard()
    registration_meter.register_meter('recall')
    registration_meter.register_meter('mean_rre')
    registration_meter.register_meter('mean_rte')
    registration_meter.register_meter('median_rre')
    registration_meter.register_meter('median_rte')
    
    # 加载模型
    model = load_model(cfg, logger)
    if model is None:
        return
    
    # 设置为评估模式
    model.eval()
    
    # 检查是否有CUDA设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    logger.info(f'Using device: {device}')
    
    # 创建测试数据加载器
    try:
        test_loader, neighbor_limits = test_data_loader(cfg, benchmark=args.benchmark)
        logger.info(f'Loaded {len(test_loader)} test samples')
    except Exception as e:
        logger.error(f'Failed to create test data loader: {e}')
        import traceback
        logger.error(traceback.format_exc())
        return
    
    # 处理每个测试样本
    total_samples = len(test_loader)
    for i, data_dict in enumerate(test_loader):
        try:
            logger.info(f'Processing sample {i+1}/{total_samples}')
            
            # 获取场景名称（如果可用）
            scene_name = "unknown"
            if 'scene_name' in data_dict:
                scene_name = data_dict['scene_name']
            elif 'ref_frame' in data_dict and 'src_frame' in data_dict:
                scene_name = f"{data_dict['ref_frame']}_{data_dict['src_frame']}"
            
            # 将数据移动到设备
            data_dict = to_cuda(data_dict)
            
            # 模型推理
            with torch.no_grad():
                output_dict = model(data_dict)
            
            # 直接使用模型输出的变换矩阵，而不是重新计算
            # 这样可以保证与demo.py的一致性
            estimated_transform = output_dict['estimated_transform'].detach().cpu().numpy()
            
            # 确保是正确的形状 (处理批次维度)
            if len(estimated_transform.shape) == 3:
                estimated_transform = estimated_transform[0]  # 取第一个批次
                
            # 获取真实变换
            transform = data_dict['transform'].detach().cpu().numpy()
            
            # 计算配准误差
            rre, rte = compute_registration_error(transform, estimated_transform)
            
            # 评估配准结果
            if not hasattr(cfg, 'eval') or not hasattr(cfg.eval, 'rre_threshold') or not hasattr(cfg.eval, 'rte_threshold'):
                logger.warning('Using default thresholds for success evaluation')
                is_success = rre < 15.0 and rte < 0.3  # 默认阈值
            else:
                is_success = rre < cfg.eval.rre_threshold and rte < cfg.eval.rte_threshold
            
            # 更新指标
            registration_meter.update('recall', float(is_success))
            registration_meter.update('mean_rre', rre)
            registration_meter.update('mean_rte', rte)
            registration_meter.update('median_rre', rre)
            registration_meter.update('median_rte', rte)
            
            # 输出单个样本的评估指标
            logger.info(f'  Sample {i+1} ({scene_name})')
            logger.info(f'    RRE: {rre:.3f}°, RTE: {rte:.3f}m, Success: {is_success}')
            logger.info(f'    Transform (ground truth):\n{transform}')
            logger.info(f'    Transform (estimated):\n{estimated_transform}')
            
        except Exception as e:
            logger.error(f'Error processing sample {i+1}: {e}')
            import traceback
            logger.error(traceback.format_exc())
    
    # 记录整体结果
    logger.info('Overall results:')
    logger.info(f'Recall: {registration_meter.meter_dict["recall"].mean():.3f}')
    logger.info(f'Mean RRE: {registration_meter.meter_dict["mean_rre"].mean():.3f}°')
    logger.info(f'Mean RTE: {registration_meter.meter_dict["mean_rte"].mean():.3f}m')
    if len(registration_meter.meter_dict["median_rre"].records) > 0:
        logger.info(f'Median RRE: {np.median(registration_meter.meter_dict["median_rre"].records):.3f}°')
        logger.info(f'Median RTE: {np.median(registration_meter.meter_dict["median_rte"].records):.3f}m')
    else:
        logger.warning('No valid results to compute median values')

def main():
    parser = make_parser()
    args = parser.parse_args()
    
    # 加载配置
    cfg = make_cfg()
    
    # 确保必要的目录存在
    for dirname in [cfg.log_dir]:
        os.makedirs(dirname, exist_ok=True)
    
    # 设置日志
    log_file = osp.join(cfg.log_dir, f'eval_{args.benchmark}_{args.method}.log')
    logger = Logger(log_file)
    logger.info('Start evaluation')
    logger.info(f'Config: {args}')
    
    # 确保数据集路径正确
    if hasattr(cfg.data, 'dataset_root'):
        logger.info(f'Dataset root: {cfg.data.dataset_root}')
        # 检查数据集目录是否存在
        if not osp.exists(cfg.data.dataset_root):
            logger.warning(f'Dataset root directory not found: {cfg.data.dataset_root}')
            # 尝试设置为自定义数据集路径
            custom_data_root = osp.join(cfg.root_dir, 'data', 'custom')
            if osp.exists(custom_data_root):
                logger.info(f'Using custom dataset root: {custom_data_root}')
                # 注意：这里不能直接修改cfg.data.dataset_root，因为它可能是只读的
                # 我们只记录这个警告，依赖dataset.py中的正确实现
    
    # 执行评估
    eval_one_epoch(args, cfg, logger)
    
    logger.info('Evaluation finished')

if __name__ == '__main__':
    main()