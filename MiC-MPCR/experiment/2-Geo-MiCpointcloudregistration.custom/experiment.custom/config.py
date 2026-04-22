import os
import os.path as osp
import argparse

from easydict import EasyDict as edict

from mic_geotrans.utils.common import ensure_dir

_C = edict()

# common
_C.seed = 728  # 随机种子，用于确保实验可重复性，影响权重初始化和数据加载顺序

# dirs
_C.working_dir = osp.dirname(osp.realpath(__file__))  # 当前工作目录路径
_C.root_dir = osp.dirname(osp.dirname(_C.working_dir))  # 项目根目录路径
_C.exp_name = 'custom_MiCgeotrans'  # 实验名称，用于组织输出文件
_C.output_dir = osp.join(_C.working_dir, 'output', _C.exp_name)  # 输出目录，保存模型、日志等
_C.snapshot_dir = osp.join(_C.output_dir, 'snapshots')  # 模型快照保存目录
_C.log_dir = osp.join(_C.output_dir, 'logs')  # 日志文件保存目录
_C.event_dir = osp.join(_C.output_dir, 'events')  # TensorBoard事件文件保存目录
_C.feature_dir = osp.join(_C.output_dir, 'features')  # 特征文件保存目录
_C.registration_dir = osp.join(_C.output_dir, 'registration')  # 配准结果保存目录

ensure_dir(_C.output_dir)
ensure_dir(_C.snapshot_dir)
ensure_dir(_C.log_dir)
ensure_dir(_C.event_dir)
ensure_dir(_C.feature_dir)
ensure_dir(_C.registration_dir)

# data
_C.data = edict()
_C.data.dataset_root = 'path_to_your_dataset'  # 数据集根目录
_C.data.dataset_type = 'CustomPairDataset'  # 数据集类型
_C.data.pairs_file = 'path_to_manifest_file'  # 数据配对文件路径
_C.data.point_cloud_dir = 'path_pointcloud'  # 点云文件目录
_C.data.transforms_dir = 'path_transforms'  # 变换矩阵文件目录
_C.data.point_scaling = 1.0  # 点云缩放因子，用于归一化坐标，增大会使点云更分散
_C.data.max_points = 100000  # 每个点云的最大点数，控制内存使用量，过小会丢失细节信息

# train data
_C.train = edict()
_C.train.batch_size = 1  # 训练批次大小，增大会提高训练效率但增加内存需求
_C.train.num_workers = 0  # 数据加载线程数，增加可提高数据加载速度但消耗更多CPU资源
_C.train.point_limit = 1800  # 训练时每个点云的采样点数限制，用于控制内存使用
_C.train.use_augmentation = True  # 是否使用数据增强，启用可提高模型泛化能力
_C.train.augmentation_noise = 0.01  # 数据增强噪声强度，增大会增加点云扰动
_C.train.augmentation_rotation = 2.0  # 数据增强旋转角度范围（弧度），增大会增加旋转扰动
_C.train.train_pairs_file = 'path_train_pairs'  # 训练集配对文件
_C.train.val_pairs_file = 'path_val_pairs'  # 验证集配对文件

# test data
_C.test = edict()
_C.test.batch_size = 1  # 测试批次大小，增大会提高测试效率但增加内存需求
_C.test.num_workers = 0  # 测试数据加载线程数
_C.test.point_limit = 3000  # 测试时每个点云的采样点数限制（减小以节省显存）
_C.test.test_pairs_file = 'path_test_pairs'  # 测试集配对文件

# evaluation
_C.eval = edict()
_C.eval.acceptance_overlap = 0.0  # 接受重叠阈值，用于评估粗匹配精度
_C.eval.acceptance_radius = 0.1  # 接受距离阈值（米），用于评估精匹配精度，增大会降低精度要求
_C.eval.inlier_ratio_threshold = 0.05  # 内点比率阈值，用于判断配准是否成功
_C.eval.rmse_threshold = 0.2  # RMSE阈值，用于判断配准是否成功
_C.eval.rre_threshold = 1.5  # 相对旋转误差阈值（度），用于判断配准是否成功，增大会降低精度要求
_C.eval.rte_threshold = 0.1  # 相对平移误差阈值（米），用于判断配准是否成功，增大会降低精度要求

# ransac
_C.ransac = edict()
_C.ransac.distance_threshold = 0.05  # RANSAC距离阈值，用于评估对应点是否匹配，增大会增加误匹配
_C.ransac.num_points = 3  # RANSAC采样点数
_C.ransac.num_iterations = 1000  # RANSAC迭代次数，增加会提高精度但降低速度

# optimizer
_C.optim = edict()
_C.optim.lr = 1e-4  # 初始学习率，增大会加快收敛但可能导致不稳定，减小会稳定但收敛慢
_C.optim.lr_decay = 0.95  # 学习率衰减因子，增大会减缓衰减速度
_C.optim.lr_decay_steps = 1  # 学习率衰减步长（epoch），增大会减缓衰减频率
_C.optim.weight_decay = 1e-6  # 权重衰减系数（L2正则化），增大会增强正则化效果
_C.optim.max_epoch = 40  # 最大训练轮数，增加会延长训练时间
_C.optim.grad_acc_steps = 2  # 梯度累积步数，增加可模拟更大批次训练但会延长训练时间

# backbone
_C.backbone = edict()
_C.backbone.num_stages = 4  # 网络阶段数，增加会提高特征提取能力但增加计算复杂度和内存使用
_C.backbone.init_voxel_size = 0.03  # 初始体素大小，增大会降低点云分辨率但提高计算效率
_C.backbone.kernel_size = 15  # KPConv卷积核大小，增加会扩大感受野但增加计算量
_C.backbone.base_radius = 2.5  # 基础半径，影响点云采样密度
_C.backbone.base_sigma = 2.0  # 基础标准差，影响高斯权重计算
_C.backbone.init_radius = _C.backbone.base_radius * _C.backbone.init_voxel_size  # 初始半径
_C.backbone.init_sigma = _C.backbone.base_sigma * _C.backbone.init_voxel_size  # 初始标准差
_C.backbone.group_norm = 32  # GroupNorm组数，影响归一化效果
_C.backbone.input_dim = 1  # 输入特征维度，通常为1表示仅使用点坐标
_C.backbone.init_dim = 64  # 初始特征维度，增加会提高表达能力但增加内存需求
_C.backbone.output_dim = 256  # 输出特征维度，增加会提高表达能力但增加内存需求

# geotransformer
_C.geotransformer = edict()
_C.geotransformer.input_dim = 1024  # Transformer输入维度，应与backbone输出匹配
_C.geotransformer.hidden_dim = 256  # Transformer隐藏层维度，增加会提高表达能力但增加计算量和内存使用
_C.geotransformer.output_dim = 256  # Transformer输出维度
_C.geotransformer.num_heads = 4  # 注意力头数，增加会提高表达能力但增加计算量和内存使用
_C.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']  # Transformer块配置
_C.geotransformer.sigma_d = 0.5  # 距离嵌入的标准差，影响距离信息编码
_C.geotransformer.sigma_a = 20  # 角度嵌入的标准差，影响角度信息编码
_C.geotransformer.angle_k = 5  # 角度嵌入的近邻点数，影响角度信息计算
_C.geotransformer.reduction_a = 'max'  # 角度嵌入归约方式，'max'或'mean'

# model
_C.model = edict()
_C.model.ground_truth_matching_radius = 0.05  # 地面真值匹配半径，用于生成真实对应点，增大会增加正样本
_C.model.num_points_in_patch = 64  # 每个patch中的点数，影响局部特征提取
_C.model.num_sinkhorn_iterations = 100

# coarse matching
_C.coarse_matching = edict()
_C.coarse_matching.num_targets = 256  # 粗匹配目标数量，增加会提高匹配密度但增加计算量
_C.coarse_matching.overlap_threshold = 0.1  # 重叠阈值，用于生成正样本
_C.coarse_matching.num_correspondences = 512  # 粗匹配对应点数量，增加会提高匹配密度
_C.coarse_matching.dual_normalization = True  # 是否使用双重归一化，有助于提高匹配稳定性

# fine matching
_C.fine_matching = edict()
_C.fine_matching.topk = 5  # 精匹配Top-K值，增加会考虑更多候选点但增加计算量
_C.fine_matching.acceptance_radius = 0.15  # 精匹配接受半径，用于判断匹配是否正确
_C.fine_matching.mutual = True  # 是否使用互近邻匹配，有助于提高匹配精度
_C.fine_matching.confidence_threshold = 0.05  # 置信度阈值，用于过滤低置信度匹配
_C.fine_matching.use_dustbin = False  # 是否使用dustbin机制处理无匹配点
_C.fine_matching.use_global_score = False  # 是否使用全局评分
_C.fine_matching.correspondence_threshold = 3  # 对应点阈值，用于判断是否足够匹配
_C.fine_matching.correspondence_limit = None  # 对应点限制，None表示无限制
_C.fine_matching.num_refinement_steps = 5  # 精匹配优化步数，增加会提高精度但降低速度

# coarse loss
_C.coarse_loss = edict()
_C.coarse_loss.positive_margin = 0.05  # 正样本边界，影响圆形损失计算
_C.coarse_loss.negative_margin = 1.6  # 负样本边界，影响圆形损失计算
_C.coarse_loss.positive_optimal = 0.05  # 正样本最优值，影响圆形损失计算
_C.coarse_loss.negative_optimal = 1.6  # 负样本最优值，影响圆形损失计算
_C.coarse_loss.log_scale = 32  # 对数缩放因子，影响损失幅度
_C.coarse_loss.positive_overlap = 0.1  # 正样本重叠阈值，用于生成正负样本掩码

# fine loss
_C.fine_loss = edict()
_C.fine_loss.positive_radius = 0.08  # 精匹配正样本半径，用于判断点对是否为正样本

# overall loss
_C.loss = edict()
_C.loss.weight_coarse_loss = 1.0  # 粗匹配损失权重，增加会更重视粗匹配
_C.loss.weight_fine_loss = 1.0  # 精匹配损失权重，增加会更重视精匹配

# snapshot
_C.snapshot = edict()
_C.snapshot.resume = True  # 是否从快照恢复训练
_C.snapshot.checkpoint = ''  # 检查点路径
_C.snapshot.pretrained = ''  # 预训练模型路径
_C.snapshot.save_freq = 1  # 快照保存频率（epoch）
_C.snapshot.eval_freq = 1  # 验证频率（epoch）


def make_cfg():
    # 保持配置文件中设置的提示性路径
    return _C


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--link_output', dest='link_output', action='store_true', help='link output dir')
    parser.add_argument('--dataset_root', type=str, default=None, help='数据集根目录')
    parser.add_argument('--dataset_type', type=str, default=None, help='数据集类型')
    parser.add_argument('--batch_size', type=int, default=None, help='batch size')
    parser.add_argument('--num_workers', type=int, default=None, help='number of workers')
    parser.add_argument('--max_epoch', type=int, default=None, help='maximum number of epochs')
    parser.add_argument('--resume', action='store_true', help='resume training')
    parser.add_argument('--checkpoint', default=None, help='checkpoint path')
    parser.add_argument('--pretrained', default=None, help='pretrained model path')
    args = parser.parse_args()
    return args


def update_cfg(cfg, args):
    # 从命令行参数更新配置
    # 处理输出目录参数
    if hasattr(args, 'output_dir') and args.output_dir:
        cfg.output_dir = args.output_dir
        cfg.snapshot_dir = osp.join(cfg.output_dir, 'snapshots')
        cfg.log_dir = osp.join(cfg.output_dir, 'logs')
        cfg.event_dir = osp.join(cfg.output_dir, 'events')
        cfg.feature_dir = osp.join(cfg.output_dir, 'features')
        cfg.registration_dir = osp.join(cfg.output_dir, 'registration')
    
    # 确保输出目录存在
    ensure_dir(cfg.output_dir)
    ensure_dir(cfg.snapshot_dir)
    ensure_dir(cfg.log_dir)
    ensure_dir(cfg.event_dir)
    ensure_dir(cfg.feature_dir)
    ensure_dir(cfg.registration_dir)
    
    # 处理数据集根目录参数
    if hasattr(args, 'dataset_root') and args.dataset_root and args.dataset_root.strip():
        cfg.data.dataset_root = args.dataset_root
        # 只有当配对文件路径仍然是提示性路径时，才更新它们
        if cfg.data.pairs_file == 'path_to_manifest_file':
            cfg.data.pairs_file = osp.join(cfg.data.dataset_root, 'manifest.txt')
        if cfg.train.train_pairs_file == 'path_train_pairs':
            cfg.train.train_pairs_file = osp.join(cfg.data.dataset_root, 'train_pairs.txt')
        if cfg.train.val_pairs_file == 'path_val_pairs':
            cfg.train.val_pairs_file = osp.join(cfg.data.dataset_root, 'val_pairs.txt')
        if cfg.test.test_pairs_file == 'path_test_pairs':
            cfg.test.test_pairs_file = osp.join(cfg.data.dataset_root, 'test_pairs.txt')
    
    if hasattr(args, 'dataset_type') and args.dataset_type:
        cfg.data.dataset_type = args.dataset_type
    
    if hasattr(args, 'batch_size') and args.batch_size:
        cfg.train.batch_size = args.batch_size
        cfg.test.batch_size = args.batch_size
    
    if hasattr(args, 'num_workers') and args.num_workers:
        cfg.train.num_workers = args.num_workers
        cfg.test.num_workers = args.num_workers
    
    if hasattr(args, 'max_epoch') and args.max_epoch:
        cfg.optim.max_epoch = args.max_epoch
    
    if hasattr(args, 'resume'):
        cfg.snapshot.resume = args.resume
    
    if hasattr(args, 'checkpoint') and args.checkpoint:
        cfg.snapshot.checkpoint = args.checkpoint
        cfg.snapshot.resume = True
    
    if hasattr(args, 'pretrained') and args.pretrained:
        cfg.snapshot.pretrained = args.pretrained
    
    return cfg


def main():
    cfg = make_cfg()
    args = parse_args()
    update_cfg(cfg, args)
    
    if args.link_output:
        try:
            # 确保链接到正确的output/custom_geotransformer目录
            if os.path.exists('output'):
                if os.path.islink('output'):
                    os.unlink('output')
                else:
                    import shutil
                    shutil.rmtree('output')
            os.symlink(cfg.output_dir, 'output')
        except Exception as e:
            print(f"创建符号链接失败: {e}")


if __name__ == '__main__':
    main()