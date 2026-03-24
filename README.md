# Mic-MPCR
# MiC Multi-View Point Cloud Registration Framework

**Repository for Chapter 3 of the paper**  
《基于视觉-激光紧耦合感知的钢结构MiC尺寸测量与分析系统》

## 简介
本仓库提供针对钢结构模块化集成建筑（MiC）场景的多视图点云配准完整实现，包括：
- 仿真驱动的多视角激光雷达扫描建模与布局优化
- 基于几何注意力与点到面约束的成对配准网络
- 位姿图（Pose Graph Optimization）全局一致性优化

主要目标是为钢结构MiC模块生成高质量、几何完整的融合点云，为后续Scan-BIM对齐与尺寸测量提供可靠数据基础。

## 当前状态

仓库仍在整理中，目前仅包含部分核心模块代码与说明文档。  
后续将陆续上传：

- 完整仿真扫描环境代码（基于光线追踪）
- 数据增强与多视图数据集生成脚本
- 成对配准网络训练与推理完整代码（PyTorch实现）
- 位姿图全局优化模块（g2o / Ceres Solver接口）
- 详细使用教程、配置文件示例与性能评估脚本
