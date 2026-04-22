import os
import numpy as np
import torch
import random
from torch.utils.data import Dataset, Subset
from mic_geotrans.utils.data import registration_collate_fn_stack_mode
from mic_geotrans.utils.data import build_dataloader_stack_mode
from mic_geotrans.utils.data import calibrate_neighbors_stack_mode


class CustomPairDataset(Dataset):
    def __init__(self, cfg, subset):
        self.cfg = cfg
        self.dataset_root = cfg.data.dataset_root
        self.subset = subset
        
        # 数据加载参数
        if subset == 'train':
            self.point_limit = cfg.train.point_limit
        else:
            self.point_limit = cfg.test.point_limit
            
        self.point_cloud_dir = os.path.join(self.dataset_root, 'point_clouds')
        self.transforms_dir = os.path.join(self.dataset_root, 'transforms')
        
        # 加载配对文件（使用config.py中定义的路径）
        if subset == 'train':
            pair_filename = cfg.train.train_pairs_file
        elif subset == 'val':
            pair_filename = cfg.train.val_pairs_file
        elif subset == 'test':
            pair_filename = cfg.test.test_pairs_file
        else:
            raise ValueError(f'Unknown subset: {subset}')
        
        self.pairs = self._load_pairs(pair_filename)
    
    def _load_pairs(self, pair_filename):
        pairs = []
        with open(pair_filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    ref_filename = parts[0]
                    src_filename = parts[1]
                    transform_filename = parts[2] if len(parts) >= 3 else None
                    pairs.append((ref_filename, src_filename, transform_filename))
        
        if not pairs:
            raise ValueError(f'配对文件 {pair_filename} 不包含有效数据')
        
        return pairs
    
    def _load_point_cloud(self, filename):
        file_path = os.path.join(self.dataset_root, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(self.point_cloud_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f'找不到点云文件: {filename}')
        
        if filename.endswith('.npy'):
            points = np.load(file_path)
        elif filename.endswith('.txt') or filename.endswith('.xyz'):
            points = np.loadtxt(file_path)
        else:
            raise ValueError(f'不支持的点云格式: {os.path.splitext(filename)[1]}')
        
        if points.shape[1] > 3:
            points = points[:, :3]
            
        return points
    
    def _load_transform(self, filename):
        if filename is None:
            return np.eye(4)
        
        file_path = os.path.join(self.dataset_root, filename)
        if not os.path.exists(file_path):
            file_path = os.path.join(self.transforms_dir, filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f'找不到变换矩阵文件: {filename}')
        
        if filename.endswith('.npy'):
            transform = np.load(file_path)
        else:
            transform = np.loadtxt(file_path)
            
        if transform.shape != (4, 4):
            if transform.size == 16:
                transform = transform.reshape(4, 4)
            else:
                raise ValueError(f'变换矩阵格式不正确: {transform.shape}')
                
        return transform
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        ref_filename, src_filename, transform_filename = self.pairs[index]
        
        # 加载点云
        ref_points = self._load_point_cloud(ref_filename)
        src_points = self._load_point_cloud(src_filename)
        
        # 加载变换矩阵
        transform = self._load_transform(transform_filename)
        
        # 随机采样点云
        if self.point_limit is not None and self.point_limit > 0:
            if ref_points.shape[0] > self.point_limit:
                indices = np.random.choice(ref_points.shape[0], self.point_limit, replace=False)
                ref_points = ref_points[indices]
                
            if src_points.shape[0] > self.point_limit:
                indices = np.random.choice(src_points.shape[0], self.point_limit, replace=False)
                src_points = src_points[indices]
        
        # 转换为张量
        ref_points = torch.from_numpy(ref_points).float()
        src_points = torch.from_numpy(src_points).float()
        transform = torch.from_numpy(transform).float()
        
        # 计算特征 (使用1维特征，与3DMatch保持一致)
        ref_feats = torch.ones(ref_points.shape[0], 1).float()
        src_feats = torch.ones(src_points.shape[0], 1).float()
        
        # 提取场景名称
        scene_name = self._extract_scene_name(ref_filename)
        
        # 返回符合collate函数要求的数据格式
        data_dict = {
            'ref_points': ref_points,
            'src_points': src_points,
            'ref_feats': ref_feats,
            'src_feats': src_feats,
            'transform': transform,
            'scene_name': scene_name,
            'ref_frame': ref_filename,
            'src_frame': src_filename,
        }
        
        return data_dict
    
    def _extract_scene_name(self, path):
        """从路径中提取MiC场景名称"""
        # 路径格式: MiC_X/Variation_XXXX/point_clouds/lidar_Y.npy
        # 或者: MiC_X/Variation_XXXX/point_clouds/lidar_Y.npy
        parts = path.replace('\\', '/').split('/')
        for part in parts:
            if part.startswith('MiC_'):
                # 只返回场景名称（如 MiC_3），不包含路径的其他部分
                return part
        return "unknown"


def create_dataset(cfg, subset, is_train=False):
    dataset = CustomPairDataset(cfg, subset)
    return dataset


def train_valid_data_loader(cfg, distributed=False):
    train_dataset = create_dataset(cfg, 'train', is_train=True)
    
    # 动态校准邻居限制，确保与num_stages一致
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        keep_ratio=0.8,
        sample_threshold=100  # 减少采样阈值以适应小数据集
    )
    
    # 创建训练数据加载器
    train_loader = build_dataloader_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        distributed=distributed
    )
    
    # 创建验证数据加载器
    valid_loader = None
    try:
        valid_dataset = create_dataset(cfg, 'val', is_train=False)
        valid_loader = build_dataloader_stack_mode(
            valid_dataset,
            registration_collate_fn_stack_mode,
            cfg.backbone.num_stages,
            cfg.backbone.init_voxel_size,
            cfg.backbone.init_radius,
            neighbor_limits,
            batch_size=cfg.test.batch_size,
            num_workers=cfg.test.num_workers,
            shuffle=False,
            distributed=distributed
        )
    except (ValueError, FileNotFoundError):
        valid_loader = None
    
    return train_loader, valid_loader, neighbor_limits


def test_data_loader(cfg, benchmark='test'):
    test_dataset = create_dataset(cfg, benchmark, is_train=False)
    
    # 使用与训练集相同的邻居限制计算方法
    # 先创建一个小的训练集样本来校准邻居限制
    train_dataset = create_dataset(cfg, 'train', is_train=True)
    neighbor_limits = calibrate_neighbors_stack_mode(
        train_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        keep_ratio=0.8,
        sample_threshold=100  # 减少采样阈值以适应小数据集
    )
    
    test_loader = build_dataloader_stack_mode(
        test_dataset,
        registration_collate_fn_stack_mode,
        cfg.backbone.num_stages,
        cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius,
        neighbor_limits,
        batch_size=cfg.test.batch_size,
        num_workers=cfg.test.num_workers,
        shuffle=False
    )
    
    return test_loader, neighbor_limits