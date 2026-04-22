import os.path as osp
import pickle
import random
import numpy as np
import torch
import torch.utils.data
from typing import Dict, List, Tuple, Optional

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.registration import get_correspondences


class CustomPairDataset(torch.utils.data.Dataset):
    r"""
    自定义点云配对数据集，支持多种格式的点云配准数据。
    支持的配对信息格式：
    1. 配对文件（.pkl或.txt）
    2. 目录中的文件命名约定（例如ref_xxx.pcd和src_xxx.pcd）
    """
    def __init__(
        self,
        dataset_root: str,
        subset: str = 'train',
        point_limit: Optional[int] = None,
        use_augmentation: bool = False,
        augmentation_noise: float = 0.005,
        augmentation_rotation: float = 1.0,
        overlap_threshold: Optional[float] = None,
        return_corr_indices: bool = False,
        matching_radius: Optional[float] = None,
        pair_file: Optional[str] = None,
        point_feature_dim: int = 1,
        file_ext: str = '.pth',  # 支持.pth, .npy, .pcd等
    ):
        super(CustomPairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        
        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation
        
        self.point_feature_dim = point_feature_dim
        self.file_ext = file_ext
        
        # 加载配对信息
        self.pair_file = pair_file
        self.metadata_list = self._load_metadata()
        
        # 根据overlap阈值过滤
        if self.overlap_threshold is not None:
            self.metadata_list = [x for x in self.metadata_list if x.get('overlap', 0) > self.overlap_threshold]
    
    def _load_metadata(self) -> List[Dict]:
        r"""加载配对元数据"""
        metadata_list = []
        
        # 1. 如果提供了配对文件
        if self.pair_file and osp.exists(self.pair_file):
            ext = osp.splitext(self.pair_file)[1].lower()
            if ext == '.pkl':
                with open(self.pair_file, 'rb') as f:
                    metadata_list = pickle.load(f)
            elif ext == '.txt':
                with open(self.pair_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            metadata = {
                                'ref_file': parts[0],
                                'src_file': parts[1],
                                'overlap': float(parts[2]) if len(parts) > 2 else 0.5,
                            }
                            # 如果有变换矩阵信息
                            if len(parts) >= 12:
                                transform = np.array(list(map(float, parts[2:11]))).reshape(4, 4)
                                metadata['transform'] = transform
                            metadata_list.append(metadata)
        
        # 2. 否则，尝试自动检测目录中的配对
        elif osp.isdir(self.dataset_root):
            # 尝试查找ref_xxx和src_xxx格式的配对文件
            files = os.listdir(self.dataset_root)
            ref_files = [f for f in files if f.startswith('ref_') and f.endswith(self.file_ext)]
            
            for ref_file in ref_files:
                # 查找对应的src文件
                base_name = ref_file[4:].split('.')[0]  # 去掉ref_前缀和扩展名
                src_file = f'src_{base_name}{self.file_ext}'
                
                if src_file in files:
                    metadata_list.append({
                        'ref_file': ref_file,
                        'src_file': src_file,
                        'overlap': 0.5,  # 默认重叠率
                    })
        
        if not metadata_list:
            raise ValueError(f'无法在{self.dataset_root}中找到有效的配对数据')
        
        return metadata_list
    
    def _load_point_cloud(self, file_name: str) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        加载点云文件，支持多种格式
        返回: (points, features)
        """
        file_path = osp.join(self.dataset_root, file_name)
        
        # 根据文件扩展名选择加载方法
        ext = osp.splitext(file_name)[1].lower()
        
        if ext == '.pth':
            # PyTorch格式
            data = torch.load(file_path)
            if isinstance(data, np.ndarray):
                points = data
                features = np.ones((points.shape[0], self.point_feature_dim))
            elif isinstance(data, dict):
                points = data['points'] if 'points' in data else data['positions']
                features = data['features'] if 'features' in data else np.ones((points.shape[0], self.point_feature_dim))
            else:
                raise ValueError(f'不支持的.pth文件格式: {file_name}')
                
        elif ext == '.npy':
            # NumPy格式
            data = np.load(file_path)
            points = data[:, :3]  # 假设前3列是坐标
            if data.shape[1] > 3:
                features = data[:, 3:3+self.point_feature_dim]
            else:
                features = np.ones((points.shape[0], self.point_feature_dim))
                
        elif ext == '.pcd':
            # PCD格式（需要open3d）
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points)
                features = np.ones((points.shape[0], self.point_feature_dim))
                # 如果有颜色信息，可以作为特征
                if np.asarray(pcd.colors).shape[0] > 0 and self.point_feature_dim >= 3:
                    features[:, :3] = np.asarray(pcd.colors)
            except ImportError:
                raise ImportError('加载PCD文件需要安装open3d')
                
        else:
            raise ValueError(f'不支持的文件格式: {ext}')
        
        # 限制点云数量
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[:self.point_limit]
            points = points[indices]
            features = features[indices]
            
        return points, features
    
    def _augment_point_cloud(self, ref_points, src_points, rotation, translation):
        r"""增强点云数据"""
        # 随机旋转
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
        
        # 随机噪声
        ref_points += (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) - 0.5) * self.aug_noise
        
        return ref_points, src_points, rotation, translation
    
    def __len__(self):
        return len(self.metadata_list)
    
    def __getitem__(self, index):
        data_dict = {}
        
        # 获取元数据
        metadata = self.metadata_list[index]
        
        # 加载点云
        ref_points, ref_features = self._load_point_cloud(metadata['ref_file'])
        src_points, src_features = self._load_point_cloud(metadata['src_file'])
        
        # 获取或生成变换矩阵
        if 'transform' in metadata:
            transform = metadata['transform']
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
        else:
            # 如果没有变换矩阵，检查是否有单独的旋转和平移信息
            if 'rotation' in metadata and 'translation' in metadata:
                rotation = metadata['rotation']
                translation = metadata['translation']
            else:
                # 否则使用单位变换（用于推理场景）
                rotation = np.eye(3)
                translation = np.zeros(3)
        
        # 数据增强
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation
            )
        
        # 构建4x4变换矩阵
        transform = get_transform_from_rotation_translation(rotation, translation)
        
        # 存储基本信息
        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_features'] = ref_features.astype(np.float32)
        data_dict['src_features'] = src_features.astype(np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['rotation'] = rotation.astype(np.float32)
        data_dict['translation'] = translation.astype(np.float32)
        
        # 存储元数据信息
        data_dict['ref_file'] = metadata['ref_file']
        data_dict['src_file'] = metadata['src_file']
        data_dict['overlap'] = metadata.get('overlap', 0.5)
        
        # 返回对应点索引（如果需要）
        if self.return_corr_indices:
            corr_indices = get_correspondences(
                ref_points, src_points, rotation, translation, self.matching_radius
            )
            data_dict['corr_indices'] = corr_indices
        
        return data_dict