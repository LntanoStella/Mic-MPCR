import sys
import os
# 设置PyTorch显存管理参数，减少显存碎片化
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# 优化显存管理：增大分割大小，减少碎片化
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

# 禁用MAGMA库，避免CUDA版本不兼容问题
os.environ['PYTORCH_MAGMA_DISABLE'] = '1'

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import time
import os
import os.path as osp
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 导入必要的模块
from mic_geotrans.engine import EpochBasedTrainer
from mic_geotrans.utils.torch import to_cuda
from config import make_cfg
from dataset import train_valid_data_loader
from model import create_model
from loss import OverallLoss, Evaluator


class Trainer(EpochBasedTrainer):
    def __init__(self, cfg):
        super().__init__(cfg, max_epoch=cfg.optim.max_epoch)

        # dataloader
        start_time = time.time()
        train_loader, val_loader, neighbor_limits = train_valid_data_loader(cfg, self.distributed)
        loading_time = time.time() - start_time
        message = 'Data loader created: {:.3f}s collapsed.'.format(loading_time)
        self.logger.info(message)
        message = 'Calibrate neighbors: {}.'.format(neighbor_limits)
        self.logger.info(message)
        self.register_loader(train_loader, val_loader)

        # model, optimizer, scheduler
        model = create_model(cfg).cuda()  # 添加.cuda()调用
        model = self.register_model(model)
        optimizer = optim.Adam(model.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay)
        self.register_optimizer(optimizer)
        scheduler = optim.lr_scheduler.StepLR(optimizer, cfg.optim.lr_decay_steps, gamma=cfg.optim.lr_decay)
        self.register_scheduler(scheduler)

        # loss function, evaluator
        self.loss_func = OverallLoss(cfg).cuda()  # 添加.cuda()调用
        self.evaluator = Evaluator(cfg).cuda()  # 添加.cuda()调用
        
        # 清理GPU内存缓存
        torch.cuda.empty_cache()
        # 打印GPU内存使用情况
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            cached = torch.cuda.memory_reserved() / 1024**3
            self.logger.info(f'GPU内存使用 - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB')

    def train_step(self, epoch, iteration, data_dict):
        # 将数据移动到GPU
        data_dict = to_cuda(data_dict)
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        # 清理GPU内存缓存
        torch.cuda.empty_cache()
        return output_dict, loss_dict

    def val_step(self, epoch, iteration, data_dict):
        # 将数据移动到GPU
        data_dict = to_cuda(data_dict)
        output_dict = self.model(data_dict)
        loss_dict = self.loss_func(output_dict, data_dict)
        result_dict = self.evaluator(output_dict, data_dict)
        loss_dict.update(result_dict)
        # 清理GPU内存缓存
        torch.cuda.empty_cache()
        return output_dict, loss_dict


def main():
    cfg = make_cfg()
    # 移除错误的目录配置，使用config.py中定义的output目录结构
    
    # 移除cpu设备设置，使用默认的CUDA设备
    
    # 确保目录存在
    for dirname in [cfg.log_dir, cfg.event_dir, cfg.snapshot_dir]:
        os.makedirs(dirname, exist_ok=True)
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == '__main__':
    main()