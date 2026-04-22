import argparse
import os.path as osp
import glob
import os
import torch
import numpy as np
from datetime import datetime
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


def extract_scene_name(path):
    """从路径中提取场景名称"""
    # 路径格式: MiC_X/Variation_XXXX/point_clouds/lidar_Y.npy
    # 或者: MiC_X/Variation_XXXX/point_clouds/lidar_Y.npy
    # 处理Windows和Unix路径分隔符
    path = path.replace('\\', '/')
    parts = path.split('/')
    for part in parts:
        if part.startswith('MiC_'):
            return part
    return "unknown"


def compute_rmse(gt_transform, est_transform, src_points):
    """
    计算配准的均方根误差(RMSE)
    
    参数:
        gt_transform: 真实变换矩阵 (4x4)
        est_transform: 估计变换矩阵 (4x4)
        src_points: 源点云 (N x 3)
    
    返回:
        rmse: 均方根误差
    """
    # 确保输入是numpy数组
    gt_transform = np.asarray(gt_transform)
    est_transform = np.asarray(est_transform)
    
    # 如果src_points是torch张量，先转换为numpy数组
    if hasattr(src_points, 'detach'):
        src_points = src_points.detach().cpu().numpy()
    else:
        src_points = np.asarray(src_points)
    
    # 如果有批次维度，取第一个
    if len(src_points.shape) == 3:
        src_points = src_points[0]
    
    # 计算重新对齐变换
    realignment_transform = np.linalg.inv(gt_transform) @ est_transform
    
    # 应用变换到源点云
    # 转换为齐次坐标
    src_points_homo = np.hstack([src_points, np.ones((src_points.shape[0], 1))])
    
    # 应用变换
    realigned_src_points = (realignment_transform @ src_points_homo.T).T[:, :3]
    
    # 计算欧氏距离
    distances = np.linalg.norm(realigned_src_points - src_points, axis=1)
    
    # 计算均方根误差
    rmse = np.mean(distances)
    
    return rmse


def eval_one_epoch(args, cfg, logger):
    # 初始化全局评估指标
    global_registration_meter = SummaryBoard()
    global_registration_meter.register_meter('recall')
    global_registration_meter.register_meter('mean_rre')
    global_registration_meter.register_meter('mean_rte')
    global_registration_meter.register_meter('mean_rmse')
    global_registration_meter.register_meter('median_rre')
    global_registration_meter.register_meter('median_rte')
    global_registration_meter.register_meter('median_rmse')
    
    # 初始化场景级评估指标
    scene_results = {}
    
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
            
            # 获取场景名称（数据集现在提供了scene_name字段）
            scene_name = "unknown"
            if 'scene_name' in data_dict:
                scene_name = data_dict['scene_name']
            elif 'ref_frame' in data_dict:
                # 如果没有scene_name，从ref_frame中提取
                scene_name = extract_scene_name(data_dict['ref_frame'])
            
            # 提取MiC场景名称（如MiC_1, MiC_2等）
            mic_scene = extract_scene_name(scene_name)
            
            # 初始化场景级指标（如果不存在）
            if mic_scene not in scene_results:
                scene_results[mic_scene] = {
                    'rre': [],
                    'rte': [],
                    'rmse': [],
                    'success': []
                }
            
            # 将数据移动到设备
            data_dict = to_cuda(data_dict)
            
            # 模型推理
            with torch.no_grad():
                output_dict = model(data_dict)
            
            # 直接使用模型输出的变换矩阵
            estimated_transform = output_dict['estimated_transform'].detach().cpu().numpy()
            
            # 确保是正确的形状 (处理批次维度)
            if len(estimated_transform.shape) == 3:
                estimated_transform = estimated_transform[0]
                
            # 获取真实变换
            transform = data_dict['transform'].detach().cpu().numpy()
            
            # 计算配准误差
            rre, rte = compute_registration_error(transform, estimated_transform)
            
            # 计算RMSE - 从output_dict中获取src_points
            src_points = output_dict['src_points']
            rmse = compute_rmse(transform, estimated_transform, src_points)
            
            # 评估配准结果
            if not hasattr(cfg, 'eval') or not hasattr(cfg.eval, 'rre_threshold') or not hasattr(cfg.eval, 'rte_threshold'):
                logger.warning('Using default thresholds for success evaluation')
                is_success = rre < 15.0 and rte < 0.3
            else:
                is_success = rre < cfg.eval.rre_threshold and rte < cfg.eval.rte_threshold
            
            # 更新全局指标
            global_registration_meter.update('recall', float(is_success))
            global_registration_meter.update('mean_rre', rre)
            global_registration_meter.update('mean_rte', rte)
            global_registration_meter.update('mean_rmse', rmse)
            global_registration_meter.update('median_rre', rre)
            global_registration_meter.update('median_rte', rte)
            global_registration_meter.update('median_rmse', rmse)
            
            # 更新场景级指标
            scene_results[mic_scene]['rre'].append(rre)
            scene_results[mic_scene]['rte'].append(rte)
            scene_results[mic_scene]['rmse'].append(rmse)
            scene_results[mic_scene]['success'].append(is_success)
            
            # 输出单个样本的评估指标
            logger.info(f'  Sample {i+1} ({scene_name}, MiC场景: {mic_scene})')
            logger.info(f'    RRE: {rre:.3f}°, RTE: {rte:.3f}m, RMSE: {rmse:.3f}m, Success: {is_success}')
            
        except Exception as e:
            logger.error(f'Error processing sample {i+1}: {e}')
            import traceback
            logger.error(traceback.format_exc())
    
    # 输出全局统计结果
    logger.info('=' * 80)
    logger.info('全局统计结果 (Global Statistics)')
    logger.info('=' * 80)
    logger.info(f'Recall: {global_registration_meter.meter_dict["recall"].mean():.3f}')
    logger.info(f'Mean RRE: {global_registration_meter.meter_dict["mean_rre"].mean():.3f}°')
    logger.info(f'Mean RTE: {global_registration_meter.meter_dict["mean_rte"].mean():.3f}m')
    logger.info(f'Mean RMSE: {global_registration_meter.meter_dict["mean_rmse"].mean():.3f}m')
    if len(global_registration_meter.meter_dict["median_rre"].records) > 0:
        logger.info(f'Median RRE: {np.median(global_registration_meter.meter_dict["median_rre"].records):.3f}°')
        logger.info(f'Median RTE: {np.median(global_registration_meter.meter_dict["median_rte"].records):.3f}m')
        logger.info(f'Median RMSE: {np.median(global_registration_meter.meter_dict["median_rmse"].records):.3f}m')
    else:
        logger.warning('No valid results to compute median values')
    
    # 输出场景级统计结果
    logger.info('=' * 80)
    logger.info('场景级统计结果 (Scene-level Statistics)')
    logger.info('=' * 80)
    
    # 按场景名称排序
    sorted_scenes = sorted(scene_results.keys())
    
    for scene_name in sorted_scenes:
        scene_data = scene_results[scene_name]
        rre_list = scene_data['rre']
        rte_list = scene_data['rte']
        rmse_list = scene_data['rmse']
        success_list = scene_data['success']
        
        if len(rre_list) > 0:
            mean_rre = np.mean(rre_list)
            mean_rte = np.mean(rte_list)
            mean_rmse = np.mean(rmse_list)
            median_rre = np.median(rre_list)
            median_rte = np.median(rte_list)
            median_rmse = np.median(rmse_list)
            recall = np.mean(success_list)
            
            logger.info(f'场景: {scene_name}')
            logger.info(f'  样本数: {len(rre_list)}')
            logger.info(f'  Recall: {recall:.3f}')
            logger.info(f'  Mean RRE: {mean_rre:.3f}°, Mean RTE: {mean_rte:.3f}m, Mean RMSE: {mean_rmse:.3f}m')
            logger.info(f'  Median RRE: {median_rre:.3f}°, Median RTE: {median_rte:.3f}m, Median RMSE: {median_rmse:.3f}m')
            logger.info(f'  RRE范围: [{np.min(rre_list):.3f}°, {np.max(rre_list):.3f}°]')
            logger.info(f'  RTE范围: [{np.min(rte_list):.3f}m, {np.max(rte_list):.3f}m]')
            logger.info(f'  RMSE范围: [{np.min(rmse_list):.3f}m, {np.max(rmse_list):.3f}m]')
            logger.info('-' * 80)
    
    # 保存结果到文件
    save_results_to_file(global_registration_meter, scene_results, cfg.log_dir, args.benchmark)
    
    logger.info('Evaluation finished')


def save_results_to_file(global_meter, scene_results, log_dir, benchmark):
    """将统计结果保存到文件"""
    # 生成时间戳
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    # 创建结果文件
    result_file = osp.join(log_dir, f'evaluation_results_{benchmark}-{timestamp}.txt')
    
    with open(result_file, 'w', encoding='utf-8') as f:
        f.write('=' * 80 + '\n')
        f.write('全局统计结果 (Global Statistics)\n')
        f.write('=' * 80 + '\n')
        f.write(f'Recall: {global_meter.meter_dict["recall"].mean():.3f}\n')
        f.write(f'Mean RRE: {global_meter.meter_dict["mean_rre"].mean():.3f}°\n')
        f.write(f'Mean RTE: {global_meter.meter_dict["mean_rte"].mean():.3f}m\n')
        f.write(f'Mean RMSE: {global_meter.meter_dict["mean_rmse"].mean():.3f}m\n')
        
        if len(global_meter.meter_dict["median_rre"].records) > 0:
            f.write(f'Median RRE: {np.median(global_meter.meter_dict["median_rre"].records):.3f}°\n')
            f.write(f'Median RTE: {np.median(global_meter.meter_dict["median_rte"].records):.3f}m\n')
            f.write(f'Median RMSE: {np.median(global_meter.meter_dict["median_rmse"].records):.3f}m\n')
        
        f.write('\n')
        f.write('=' * 80 + '\n')
        f.write('场景级统计结果 (Scene-level Statistics)\n')
        f.write('=' * 80 + '\n')
        
        # 按场景名称排序
        sorted_scenes = sorted(scene_results.keys())
        
        for scene_name in sorted_scenes:
            scene_data = scene_results[scene_name]
            rre_list = scene_data['rre']
            rte_list = scene_data['rte']
            rmse_list = scene_data['rmse']
            success_list = scene_data['success']
            
            if len(rre_list) > 0:
                mean_rre = np.mean(rre_list)
                mean_rte = np.mean(rte_list)
                mean_rmse = np.mean(rmse_list)
                median_rre = np.median(rre_list)
                median_rte = np.median(rte_list)
                median_rmse = np.median(rmse_list)
                recall = np.mean(success_list)
                
                f.write(f'场景: {scene_name}\n')
                f.write(f'  样本数: {len(rre_list)}\n')
                f.write(f'  Recall: {recall:.3f}\n')
                f.write(f'  Mean RRE: {mean_rre:.3f}°, Mean RTE: {mean_rte:.3f}m, Mean RMSE: {mean_rmse:.3f}m\n')
                f.write(f'  Median RRE: {median_rre:.3f}°, Median RTE: {median_rte:.3f}m, Median RMSE: {median_rmse:.3f}m\n')
                f.write(f'  RRE范围: [{np.min(rre_list):.3f}°, {np.max(rre_list):.3f}°]\n')
                f.write(f'  RTE范围: [{np.min(rte_list):.3f}m, {np.max(rte_list):.3f}m]\n')
                f.write(f'  RMSE范围: [{np.min(rmse_list):.3f}m, {np.max(rmse_list):.3f}m]\n')
                f.write('-' * 80 + '\n')
    
    print(f'结果已保存到: {result_file}')


def main():
    parser = make_parser()
    args = parser.parse_args()
    
    # 加载配置
    cfg = make_cfg()
    
    # 确保必要的目录存在
    for dirname in [cfg.log_dir]:
        os.makedirs(dirname, exist_ok=True)
    
    # 设置日志
    log_file = osp.join(cfg.log_dir, f'eval_with_statistics_{args.benchmark}_{args.method}.log')
    logger = Logger(log_file)
    logger.info('Start evaluation with statistics')
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
    
    # 执行评估
    eval_one_epoch(args, cfg, logger)
    
    logger.info('Evaluation finished')


if __name__ == '__main__':
    main()
