#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GeoTransformer点云配准推理脚本

此脚本实现了基于GeoTransformer的点云配准功能，包括：
- 依赖检查和环境设置
- 模型导入和加载
- 点云数据加载和预处理
- 配准执行和结果保存
- 点云配准结果可视化

支持直接在IDE中运行，也支持命令行参数配置
"""

# 环境检查：在导入其他模块前先检查基本依赖
def check_basic_dependencies():
    """检查基本依赖是否已安装"""
    try:
        import numpy
        import os
        import os.path as osp
        import sys
        return True
    except ImportError as e:
        print(f"缺少基本依赖: {e}")
        return False

# 确保基本依赖可用
if not check_basic_dependencies():
    print("请先安装基本依赖后再运行此脚本")
    import sys
    sys.exit(1)

# 基础导入
import numpy as np
import os
import os.path as osp
import sys
import argparse
from types import SimpleNamespace
import datetime

# 将当前目录和项目根目录添加到Python路径
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, current_dir)
sys.path.insert(0, project_root)
print(f"已添加当前目录到Python路径: {current_dir}")
print(f"已添加项目根目录到Python路径: {project_root}")

# 尝试导入torch
try:
    import torch
    print("✓ torch模块已安装")
except ImportError:
    print("✗ torch模块未安装，部分功能将不可用")
    torch = None

# 尝试导入Open3D并检查版本
try:
    import open3d as o3d
    print(f"✓ open3d模块已安装，版本: {o3d.__version__}")
    has_open3d = True
    # 检查是否支持可视化功能
    try:
        # 测试简单的可视化操作是否会失败
        test_pcd = o3d.geometry.PointCloud()
        print("✓ Open3D点云功能正常")
        has_visualization = True
    except Exception as e:
        print(f"! Open3D可视化功能可能受限: {e}")
        has_visualization = False
except ImportError:
    print("✗ open3d模块未安装，无法进行可视化")
    print("提示: 可以运行 'pip install open3d' 来安装Open3D")
    has_open3d = False
    has_visualization = False

# 函数：检查依赖
def check_dependencies():
    """检查必要的依赖包"""
    missing_packages = []
    
    # 检查numpy
    try:
        import numpy
        print("✓ numpy 已安装")
    except ImportError:
        missing_packages.append("numpy")
        print("✗ numpy 未安装")
    
    # 检查torch
    try:
        import torch
        print("✓ torch 已安装")
    except ImportError:
        missing_packages.append("torch")
        print("✗ torch 未安装")
    
    # 检查mic_geotrans
    try:
        import mic_geotrans
        print("✓ mic_geotrans 已安装")
    except ImportError:
        print("✗ mic_geotrans 未安装或无法导入（这是项目内部模块）")
    
    if missing_packages:
        print(f"\n提示: 缺少以下依赖包: {', '.join(missing_packages)}")
        print("请运行以下命令安装依赖:")
        
        # 构建安装命令
        install_packages = []
        if 'torch' in missing_packages:
            install_packages.append('torch')
            install_packages.append('torchvision')
        
        # 添加其他缺失的包
        for pkg in missing_packages:
            if pkg != 'torch':
                install_packages.append(pkg)
        
        # 打印安装命令
        if install_packages:
            print(f"pip install {' '.join(install_packages)}")
    
    # 返回是否有所有必需的核心依赖
    return 'numpy' not in missing_packages

# 函数：设置随机种子
def set_random_seed(seed=0):
    """设置随机种子以确保结果可复现"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"已设置随机种子: {seed}")

# 函数：确保目录存在
def ensure_dir(path):
    """确保目录存在"""
    if not osp.exists(path):
        os.makedirs(path)
        print(f"创建目录: {path}")
    else:
        print(f"目录已存在: {path}")

# 函数：导入模型模块
def import_model_module(experiment_path):
    """从实验目录导入model模块"""
    print(f"实验目录: {experiment_path}")
    
    # 将相对路径转换为绝对路径
    if not osp.isabs(experiment_path):
        experiment_path = osp.join(os.getcwd(), experiment_path)
    
    # 检查目录是否存在
    if not osp.isdir(experiment_path):
        print(f"错误: 实验目录不存在: {experiment_path}")
        return False, None
    
    # 保存原始的sys.path
    original_path = sys.path.copy()
    
    try:
        # 将实验目录添加到Python路径
        if experiment_path not in sys.path:
            sys.path.insert(0, experiment_path)
        
        # 方法1：直接导入
        try:
            import model
            print("成功导入model模块！")
            return True, model
        except ImportError:
            # 方法2：如果直接导入失败，尝试使用importlib
            print("直接导入失败，尝试使用importlib...")
            import importlib.util
            model_file_path = osp.join(experiment_path, "model.py")
            if not osp.exists(model_file_path):
                print(f"错误: model.py文件不存在: {model_file_path}")
                return False, None
            
            spec = importlib.util.spec_from_file_location("model", model_file_path)
            if spec is None:
                print(f"无法创建model模块规范: {model_file_path}")
                return False, None
            
            model = importlib.util.module_from_spec(spec)
            sys.modules["model"] = model
            
            try:
                spec.loader.exec_module(model)
                print("成功使用importlib导入model模块")
                return True, model
            except Exception as e:
                print(f"执行model模块失败: {e}")
                import traceback
                traceback.print_exc()
                return False, None
    except Exception as e:
        print(f"导入model模块时发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        # 恢复原始的sys.path
        sys.path = original_path
    
    return False, None

# 函数：加载单个点云文件
def load_point_cloud(file_path):
    """加载点云数据"""
    print(f"尝试加载点云: {file_path}")
    
    # 将相对路径转换为绝对路径
    if not osp.isabs(file_path):
        file_path = osp.join(os.getcwd(), file_path)
    
    # 检查文件是否存在
    if not osp.isfile(file_path):
        print(f"错误: 点云文件不存在: {file_path}")
        return None
    
    # 尝试加载点云
    try:
        if file_path.endswith('.npy'):
            points = np.load(file_path)
            print(f"成功加载.npy格式点云，点数: {len(points)}")
            return points
        elif file_path.endswith('.ply'):
            # 如果没有open3d，尝试使用简单的读取方式
            print(f"警告: 点云格式为.ply，需要open3d库")
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points)
                print(f"成功加载.ply格式点云，点数: {len(points)}")
                return points
            except ImportError:
                print("错误: open3d模块未安装，无法读取.ply文件")
                return None
        elif file_path.endswith('.pcd'):
            print(f"警告: 点云格式为.pcd，需要open3d库")
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(file_path)
                points = np.asarray(pcd.points)
                print(f"成功加载.pcd格式点云，点数: {len(points)}")
                return points
            except ImportError:
                print("错误: open3d模块未安装，无法读取.pcd文件")
                return None
        else:
            print(f"错误: 不支持的点云格式: {file_path}")
            return None
    except Exception as e:
        print(f"加载点云失败: {e}")
        import traceback
        traceback.print_exc()
        return None

# 函数：加载两个点云文件
def load_point_clouds(ref_file, src_file):
    """加载参考点云和源点云"""
    ref_points = load_point_cloud(ref_file)
    src_points = load_point_cloud(src_file)
    
    if ref_points is None or src_points is None:
        print("点云加载失败")
        return False, None, None
    
    return True, ref_points, src_points

# 函数：创建和加载模型
def create_and_load_model(model_module, snapshot_path):
    """创建模型并加载自定义训练的权重"""
    # 确保torch可用
    if torch is None:
        print("错误: torch模块未安装，无法创建和加载模型")
        return False, None
    
    # 检查模型文件是否存在
    if not osp.exists(snapshot_path):
        print(f"错误: 模型文件不存在: {snapshot_path}")
        # 尝试使用常见的替代路径
        alt_paths = [
            osp.join(osp.dirname(__file__), 'snapshots', 'model_best.pth.tar'),
            osp.join(osp.dirname(__file__), 'snapshots', 'model_latest.pth.tar'),
            '../../output/geotransformer.custom/snapshots/model_best.pth.tar'
        ]
        
        for alt_path in alt_paths:
            if osp.exists(alt_path):
                print(f"尝试使用替代模型路径: {alt_path}")
                snapshot_path = alt_path
                break
        
        if not osp.exists(snapshot_path):
            print("所有尝试的模型路径都不存在，请检查模型权重文件路径")
            return False, None
    
    # 初始化配置变量
    cfg = None
    
    # 尝试加载配置 - 优先使用实验目录中的config.py
    try:
        print("尝试从实验目录加载配置...")
        # 先将当前目录添加到Python路径
        sys.path.insert(0, osp.dirname(snapshot_path))
        from config import make_cfg
        cfg = make_cfg()
        print("成功从实验目录中的config.py加载配置")
    except Exception as e:
        print(f"无法从实验目录的config.py加载配置: {e}")
        print("使用自定义数据集的模拟配置...")
        
        # 创建一个完整的模拟配置对象，专为自定义数据集优化
        cfg = SimpleNamespace()
        
        # 分别创建各个子配置对象
        cfg.model = SimpleNamespace()
        cfg.geotransformer = SimpleNamespace()
        cfg.coarse_matching = SimpleNamespace()
        cfg.fine_matching = SimpleNamespace()
        cfg.backbone = SimpleNamespace()
        
        # 设置model所需的基本属性 - 为自定义数据集优化
        cfg.model.ground_truth_matching_radius = 0.05  # 根据点云尺度调整
        cfg.model.num_points_in_patch = 64
        cfg.model.num_sinkhorn_iterations = 100
        
        # 设置GeoTransformer所需的属性 - 与训练配置保持一致
        cfg.geotransformer.input_dim = 1024
        cfg.geotransformer.hidden_dim = 256
        cfg.geotransformer.output_dim = 256
        cfg.geotransformer.num_heads = 4
        cfg.geotransformer.blocks = ['self', 'cross', 'self', 'cross', 'self', 'cross']
        cfg.geotransformer.sigma_d = 0.2  # 为自定义点云调整的参数
        cfg.geotransformer.sigma_a = 15   # 旋转参数
        cfg.geotransformer.angle_k = 3    # 与训练时保持一致
        cfg.geotransformer.reduction_a = 'max'
        
        # 设置coarse_matching所需的属性
        cfg.coarse_matching.num_targets = 128
        cfg.coarse_matching.overlap_threshold = 0.1  # 为部分重叠场景调整
        cfg.coarse_matching.num_correspondences = 256
        cfg.coarse_matching.dual_normalization = True
        
        # 设置fine_matching所需的属性 - 为自定义数据集优化
        cfg.fine_matching.topk = 3
        cfg.fine_matching.acceptance_radius = 0.1  # 根据点云尺度调整
        cfg.fine_matching.mutual = True
        cfg.fine_matching.confidence_threshold = 0.05
        cfg.fine_matching.use_dustbin = False
        cfg.fine_matching.use_global_score = False
        cfg.fine_matching.correspondence_threshold = 3
        cfg.fine_matching.correspondence_limit = None
        cfg.fine_matching.num_refinement_steps = 5
        
        # 设置backbone所需的属性 - 与训练配置保持一致
        cfg.backbone.input_dim = 1  # 仅使用xyz坐标
        cfg.backbone.init_dim = 64
        cfg.backbone.output_dim = 256
        cfg.backbone.num_stages = 4  # 与训练时保持一致
        cfg.backbone.init_voxel_size = 0.025  # 根据点云分辨率调整
        cfg.backbone.kernel_size = 15
        cfg.backbone.base_radius = 2.5
        cfg.backbone.base_sigma = 2.0
        # 计算初始化半径和sigma
        cfg.backbone.init_radius = cfg.backbone.base_radius * cfg.backbone.init_voxel_size
        cfg.backbone.init_sigma = cfg.backbone.base_sigma * cfg.backbone.init_voxel_size
        cfg.backbone.group_norm = 32  # 确保与训练配置一致
        
        print("模拟配置已创建，包含完整的属性")
    
    # 检查配置是否成功加载
    if cfg is None:
        print("错误: 无法加载任何配置")
        return False, None
    
    # 创建模型
    print("正在创建模型...")
    try:
        # 尝试使用不同的方式创建模型
        model = None
        
        # 方法1：尝试直接使用GeoTransformer类
        if hasattr(model_module, 'GeoTransformer'):
            print("使用GeoTransformer类创建模型")
            model = model_module.GeoTransformer(cfg)
        
        # 方法2：尝试使用create_model函数
        elif hasattr(model_module, 'create_model'):
            print("使用create_model函数创建模型")
            model = model_module.create_model(cfg)
        
        # 方法3：查找其他可能的模型类
        else:
            print("尝试查找并使用其他模型类...")
            # 获取所有可能的模型类
            model_classes = [name for name in dir(model_module) if 
                           isinstance(getattr(model_module, name), type) and 
                           name not in ['object', 'type']]
            
            for class_name in model_classes:
                class_obj = getattr(model_module, class_name)
                try:
                    print(f"尝试使用 {class_name} 创建模型")
                    model = class_obj(cfg)
                    print(f"成功使用 {class_name} 创建模型")
                    break
                except Exception as e:
                    print(f"使用 {class_name} 创建模型失败: {e}")
        
        if model is None:
            print("无法创建模型，请检查model模块中的类定义")
            return False, None
        
        # 使用可用的设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        print(f"模型已创建并移至 {device}")
        
        # 加载模型权重 - 增强的错误处理
        print(f"正在加载模型权重: {snapshot_path}")
        
        try:
            # 尝试加载checkpoint
            checkpoint = torch.load(snapshot_path, map_location=device)
            print(f"成功加载checkpoint，包含键: {list(checkpoint.keys())}")
            
            # 尝试不同的checkpoint结构
            if 'model' in checkpoint:
                # 标准格式: checkpoint['model'] 包含模型权重
                model.load_state_dict(checkpoint['model'], strict=False)
                print("成功从checkpoint['model']加载权重")
            elif 'state_dict' in checkpoint:
                # 可选格式: checkpoint['state_dict'] 包含模型权重
                model.load_state_dict(checkpoint['state_dict'], strict=False)
                print("成功从checkpoint['state_dict']加载权重")
            elif 'network' in checkpoint:
                # 自定义格式: checkpoint['network'] 包含模型权重
                model.load_state_dict(checkpoint['network'], strict=False)
                print("成功从checkpoint['network']加载权重")
            else:
                # 直接加载checkpoint作为state_dict
                model.load_state_dict(checkpoint, strict=False)
                print("成功直接从checkpoint加载权重")
                
            print("模型权重加载成功！")
        except Exception as e:
            print(f"加载模型权重时发生错误: {e}")
            print("尝试使用更宽松的权重加载方式...")
            
            try:
                # 尝试加载并过滤权重
                checkpoint = torch.load(snapshot_path, map_location=device)
                
                # 提取权重字典
                if isinstance(checkpoint, dict):
                    weight_dict = next((checkpoint[k] for k in ['model', 'state_dict', 'network'] if k in checkpoint), checkpoint)
                else:
                    weight_dict = checkpoint
                
                # 创建模型权重的键映射
                model_dict = model.state_dict()
                filtered_dict = {k: v for k, v in weight_dict.items() if k in model_dict}
                
                print(f"找到 {len(filtered_dict)} 个匹配的权重参数")
                if len(filtered_dict) == 0:
                    # 尝试移除模块前缀
                    filtered_dict = {k.replace('module.', ''): v for k, v in weight_dict.items() if k.replace('module.', '') in model_dict}
                    print(f"尝试移除module.前缀后，找到 {len(filtered_dict)} 个匹配的权重参数")
                
                # 更新模型权重
                model_dict.update(filtered_dict)
                model.load_state_dict(model_dict)
                print("成功使用过滤后的权重加载模型！")
            except Exception as inner_e:
                print(f"宽松加载方式也失败: {inner_e}")
                print("检查点文件可能损坏或与模型结构不兼容")
                import traceback
                traceback.print_exc()
                return False, None
        
        return True, model
    except Exception as e:
        print(f"创建模型失败: {e}")
        print(f"配置对象类型: {type(cfg)}")
        print(f"配置对象属性: {dir(cfg)}")
        if hasattr(cfg, 'backbone'):
            print(f"配置对象的backbone属性: {dir(cfg.backbone)}")
        import traceback
        traceback.print_exc()
        return False, None

# 函数：应用变换矩阵到点云
def apply_transform(points, transform):
    """应用4x4变换矩阵到点云"""
    # 检查输入
    if not isinstance(points, np.ndarray) or len(points.shape) != 2:
        print(f"错误: 点云格式无效，应为2D数组，当前形状: {points.shape}")
        return points
    
    if not isinstance(transform, np.ndarray) or transform.shape != (4, 4):
        print(f"错误: 变换矩阵格式无效，应为4x4数组，当前形状: {transform.shape}")
        return points
    
    # 转换为齐次坐标
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])
    
    # 应用变换
    points_transformed = np.dot(points_homogeneous, transform.T)[:, :3]
    
    return points_transformed

# 函数：设置可视化环境

def setup_visualization_environment():
    """设置适合当前操作系统的可视化环境"""
    import platform
    
    # 根据操作系统类型设置不同的环境变量
    system = platform.system()
    print(f"检测到操作系统: {system}")
    
    # Windows环境下的特殊处理
    if system == 'Windows':
        print("在Windows环境中运行，使用DirectX后端")
        # Windows通常不需要设置PYOPENGL_PLATFORM，让系统自动选择
        # 移除可能干扰的环境变量
        for env_var in ['PYOPENGL_PLATFORM', 'MESA_GL_VERSION_OVERRIDE', 'MESA_GLSL_VERSION_OVERRIDE']:
            if env_var in os.environ:
                del os.environ[env_var]
                print(f"移除可能干扰的环境变量: {env_var}")
    else:
        # Linux/macOS环境
        print("在Linux/macOS环境中运行")
        # 尝试不同的渲染后端，按优先级顺序
        for backend in ['glx', 'egl']:
            try:
                os.environ['PYOPENGL_PLATFORM'] = backend
                print(f"尝试使用{backend}渲染后端")
                # 设置版本兼容性
                os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
                os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'
                break
            except Exception as e:
                print(f"设置{backend}后端失败: {e}")
                continue
    
    print("可视化环境设置完成")

# 函数：可视化点云配准结果
def visualize_point_clouds(ref_points, src_points, transformed_points=None, save_path=None, title=None, show_coordinate_frame=True):
    """可视化点云配准结果 - 为自定义数据集优化
    
    Args:
        ref_points: 参考点云 (N, 3)
        src_points: 源点云 (M, 3)
        transformed_points: 变换后的源点云 (M, 3)
        save_path: 保存可视化结果的路径 (可选)
        title: 窗口标题 (可选)
        show_coordinate_frame: 是否显示坐标系
    """
    global has_open3d, has_visualization
    
    # 先检查点云数据格式
    if not isinstance(ref_points, np.ndarray) or len(ref_points.shape) != 2 or ref_points.shape[1] < 3:
        print(f"错误: 参考点云数据格式无效，需要(N, 3)数组，当前形状: {ref_points.shape}")
        return False
    
    if not isinstance(src_points, np.ndarray) or len(src_points.shape) != 2 or src_points.shape[1] < 3:
        print(f"错误: 源点云数据格式无效，需要(M, 3)数组，当前形状: {src_points.shape}")
        return False
    
    if transformed_points is not None:
        if not isinstance(transformed_points, np.ndarray) or len(transformed_points.shape) != 2 or transformed_points.shape[1] < 3:
            print(f"错误: 变换后点云数据格式无效，需要(M, 3)数组，当前形状: {transformed_points.shape}")
            return False
    
    # 总是尝试保存点云数据，即使没有可视化功能
    if save_path is not None:
        try:
            # 确保保存目录存在
            save_dir = osp.dirname(save_path)
            if save_dir and not osp.exists(save_dir):
                os.makedirs(save_dir)
        except Exception as e:
            print(f"创建保存目录失败: {e}")
    
    # 检查是否有Open3D
    if not has_open3d:
        print("警告: 无法导入open3d库，无法进行可视化")
        print("请安装open3d: pip install open3d")
        # 尝试保存点云数据为npy格式
        if save_path is not None:
            npy_path = save_path.replace('.obj', '.npy').replace('.ply', '.npy')
            try:
                if transformed_points is not None:
                    combined_data = {
                        'ref_points': ref_points,
                        'src_points': src_points,
                        'transformed_points': transformed_points
                    }
                    np.save(npy_path, combined_data)
                    print(f"已保存点云数据为NPY格式: {npy_path}")
                else:
                    combined_data = {
                        'ref_points': ref_points,
                        'src_points': src_points
                    }
                    np.save(npy_path, combined_data)
                    print(f"已保存原始点云数据为NPY格式: {npy_path}")
                return True
            except Exception as save_error:
                print(f"保存NPY格式数据失败: {save_error}")
        return False
    
    try:
        # 先设置可视化环境
        setup_visualization_environment()
        
        import open3d as o3d
        print("正在使用open3d进行可视化...")
        
        # 创建点云对象
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
        ref_pcd.paint_uniform_color([0, 0, 1])  # 蓝色表示参考点云
        
        src_pcd = o3d.geometry.PointCloud()
        src_pcd.points = o3d.utility.Vector3dVector(src_points)
        src_pcd.paint_uniform_color([1, 0, 0])  # 红色表示源点云
        
        # 创建可视化对象列表
        geometries = [ref_pcd, src_pcd]
        
        # 如果有变换后的点云，添加到可视化
        if transformed_points is not None:
            transformed_pcd = o3d.geometry.PointCloud()
            transformed_pcd.points = o3d.utility.Vector3dVector(transformed_points)
            transformed_pcd.paint_uniform_color([0, 1, 0])  # 绿色表示变换后的源点云
            geometries.append(transformed_pcd)
            
            # 可视化标题 - 使用自定义标题或默认标题
            if title is None:
                title = "点云配准结果可视化（自定义数据集）(蓝:参考点云, 红:原始源点云, 绿:变换后源点云)"
        else:
            # 只有两个点云时的标题
            if title is None:
                title = "点云配准前可视化（自定义数据集）(蓝:参考点云, 红:源点云)"
        
        # 创建坐标轴 - 适合自定义数据集的尺度
        if show_coordinate_frame:
            # 自动计算点云尺度以设置合适的坐标系大小
            all_points = np.vstack([ref_points, src_points])
            if transformed_points is not None:
                all_points = np.vstack([all_points, transformed_points])
            
            # 计算点云的边界框以确定坐标系大小
            bbox_max = np.max(all_points, axis=0)
            bbox_min = np.min(all_points, axis=0)
            bbox_size = np.max(bbox_max - bbox_min)
            
            # 根据点云尺度设置坐标系大小，确保在不同尺度的自定义数据集中都有良好显示
            coordinate_size = max(0.1, min(1.0, bbox_size * 0.1))
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=coordinate_size, origin=[0, 0, 0])
            geometries.append(coordinate_frame)
        
        # 创建可视化窗口 - 添加错误处理
        vis = o3d.visualization.Visualizer()
        print("创建Open3D可视化器对象")
        
        try:
            print("尝试创建可视化窗口...")
            # 先检查是否支持可视化
            if not has_visualization:
                print("警告: Open3D可视化功能受限，使用非交互式模式")
                window_created = False
            else:
                # 尝试创建窗口 - 使用更大的窗口以更好地显示自定义数据集
                window_created = vis.create_window(window_name=title, width=1024, height=768)
                print(f"窗口创建结果: {'成功' if window_created else '失败'}")
            
            # 添加所有几何体
            for geometry in geometries:
                vis.add_geometry(geometry)
            
            # 设置渲染选项 - 增强自定义数据集的可视化效果
            opt = vis.get_render_option()
            if opt is not None:
                opt.background_color = np.array([0.95, 0.95, 0.95])  # 浅灰色背景，提高点云可见性
                opt.point_size = 3  # 增大点的大小，更适合自定义数据集
                opt.line_width = 1.0
                opt.light_on = True  # 启用光照以增强3D效果
                print("渲染选项设置完成")
            else:
                print("警告: 无法获取渲染选项")
            
            # 保存图像（如果指定了路径）
            if save_path is not None:
                try:
                    # 保存为PNG
                    png_path = save_path.replace('.obj', '.png')
                    # 渲染多帧以确保正确显示
                    for _ in range(3):
                        vis.poll_events()
                        vis.update_renderer()
                    vis.capture_screen_image(png_path)
                    print(f"可视化结果已保存为: {png_path}")
                except Exception as e:
                    print(f"保存图像时出错: {e}")
                    
                    # 保存为OBJ (仅包含点云)
                    if transformed_points is not None:
                        combined_pcd = o3d.geometry.PointCloud()
                        combined_points = np.vstack([ref_points, transformed_points])
                        combined_colors = np.vstack([
                            np.tile([0, 0, 1], (len(ref_points), 1)),
                            np.tile([0, 1, 0], (len(transformed_points), 1))
                        ])
                        combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
                        combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
                        o3d.io.write_point_cloud(save_path, combined_pcd)
                        print(f"组合点云已保存为: {save_path}")
                except Exception as save_error:
                    print(f"保存可视化结果时发生错误: {save_error}")
            
            # 显示并等待用户交互 - 只有在窗口创建成功时才运行交互
            if window_created:
                # 设置更好的初始视角
                ctr = vis.get_view_control()
                ctr.set_lookat([0, 0, 0])  # 朝向原点
                ctr.set_front([-1, -1, -1])  # 设置相机前面方向
                ctr.set_up([0, 1, 0])  # 设置相机上方方向
                
                print("可视化窗口已打开，按ESC键关闭。操作说明:")
                print("- 鼠标左键: 旋转视角")
                print("- 鼠标滚轮: 缩放")
                print("- 鼠标右键: 平移")
                print("- 按R键: 重置视角")
                
                vis.run()
                vis.destroy_window()
                print("可视化完成")
            else:
                # 如果窗口无法创建，尝试保存图像并清理
                print("警告: 无法创建可视化窗口，可能是在无头环境中运行")
                # 在无头环境中自动尝试保存可视化结果，即使没有指定save_path
                if save_path is None:
                    # 自动生成保存路径
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    auto_save_path = osp.join(osp.dirname(osp.abspath(__file__)), f"visualization_{timestamp}.png")
                    print(f"在无头环境中自动保存可视化结果到: {auto_save_path}")
                    try:
                        vis.capture_screen_image(auto_save_path)
                        print(f"可视化结果已自动保存为: {auto_save_path}")
                    except Exception as auto_save_error:
                        print(f"自动保存可视化结果失败: {auto_save_error}")
                        # 如果捕获屏幕失败，尝试直接保存点云
                        if transformed_points is not None:
                            try:
                                point_cloud_path = auto_save_path.replace('.png', '.ply')
                                combined_pcd = o3d.geometry.PointCloud()
                                combined_points = np.vstack([ref_points, transformed_points])
                                combined_colors = np.vstack([
                                    np.tile([0, 0, 1], (len(ref_points), 1)),
                                    np.tile([0, 1, 0], (len(transformed_points), 1))
                                ])
                                combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
                                combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
                                o3d.io.write_point_cloud(point_cloud_path, combined_pcd)
                                print(f"已保存点云数据到: {point_cloud_path}")
                            except Exception as pc_save_error:
                                print(f"保存点云数据失败: {pc_save_error}")
                vis.destroy_window()
                
            return True
            
        except Exception as window_error:
            print(f"创建可视化窗口时发生错误: {window_error}")
            # 尝试使用Offscreen渲染模式
            try:
                # 使用Offscreen渲染器，避免EGL错误
                # 先确保Open3D版本支持Offscreen
                try:
                    # 针对不同Open3D版本使用不同的Offscreen方法
                    import pkg_resources
                    o3d_version = pkg_resources.get_distribution('open3d').version
                    print(f"Open3D版本: {o3d_version}")
                    
                    # 初始化Offscreen渲染器
                    if hasattr(o3d.visualization, 'OffscreenRenderer'):
                        # 新的Offscreen渲染器API
                        vis_offscreen = o3d.visualization.OffscreenRenderer(width=800, height=600)
                        print("使用OffscreenRenderer API")
                    else:
                        # 旧的Visualizer API
                        vis_offscreen = o3d.visualization.Visualizer()
                        vis_offscreen.create_window(visible=False, width=800, height=600)
                        print("使用Visualizer API (offscreen)")
                except Exception as e:
                    print(f"检测Open3D版本失败: {e}，使用默认渲染器")
                    vis_offscreen = o3d.visualization.Visualizer()
                    vis_offscreen.create_window(visible=False, width=800, height=600)
                
                # 添加几何体
                for geometry in geometries:
                    vis_offscreen.add_geometry(geometry)
                
                # 设置渲染选项
                opt_offscreen = vis_offscreen.get_render_option()
                if opt_offscreen is not None:
                    opt_offscreen.background_color = np.array([0.95, 0.95, 0.95])
                    opt_offscreen.point_size = 2
                
                # 保存图像
                if save_path is not None:
                    png_path = save_path.replace('.obj', '.png')
                    
                    # 针对不同的渲染器类型使用不同的保存方法
                    try:
                        if hasattr(vis_offscreen, 'capture_screen_image'):
                            # Visualizer API
                            vis_offscreen.capture_screen_image(png_path)
                        elif hasattr(vis_offscreen, 'render_to_image'):
                            # OffscreenRenderer API
                            img = vis_offscreen.render_to_image()
                            o3d.io.write_image(png_path, img)
                        print(f"已使用Offscreen模式保存可视化结果: {png_path}")
                    except Exception as e:
                        print(f"保存可视化截图时出错: {e}")
                    
                    # 保存点云数据
                    try:
                        if transformed_points is not None:
                            combined_pcd = o3d.geometry.PointCloud()
                            combined_points = np.vstack([ref_points, transformed_points])
                            combined_colors = np.vstack([
                                np.tile([0, 0, 1], (len(ref_points), 1)),
                                np.tile([0, 1, 0], (len(transformed_points), 1))
                            ])
                            combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
                            combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
                            o3d.io.write_point_cloud(save_path, combined_pcd)
                            print(f"组合点云已保存为: {save_path}")
                    except Exception as e:
                        print(f"保存点云数据时出错: {e}")
                else:
                    print("警告: 在无头环境中无法显示可视化，也未指定保存路径")
                    print("提示: 请使用--save_visualization参数来保存可视化结果")
                
                vis_offscreen.destroy_window()
                return True
                
            except Exception as offscreen_error:
                print(f"Offscreen渲染也失败: {offscreen_error}")
                # 最后的备选方案：直接保存点云数据而不渲染图像
                if save_path is not None:
                    try:
                        # 保存点云数据（优先保存为PLY格式）
                        ply_path = save_path.replace('.obj', '.ply')
                        
                        if transformed_points is not None:
                            # 保存组合点云
                            combined_pcd = o3d.geometry.PointCloud()
                            combined_points = np.vstack([ref_points, transformed_points])
                            combined_colors = np.vstack([
                                np.tile([0, 0, 1], (len(ref_points), 1)),
                                np.tile([0, 1, 0], (len(transformed_points), 1))
                            ])
                            combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
                            combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
                            o3d.io.write_point_cloud(ply_path, combined_pcd)
                        else:
                            # 只保存两个原始点云
                            combined_pcd = o3d.geometry.PointCloud()
                            combined_points = np.vstack([ref_points, src_points])
                            combined_colors = np.vstack([
                                np.tile([0, 0, 1], (len(ref_points), 1)),
                                np.tile([1, 0, 0], (len(src_points), 1))
                            ])
                            combined_pcd.points = o3d.utility.Vector3dVector(combined_points)
                            combined_pcd.colors = o3d.utility.Vector3dVector(combined_colors)
                            o3d.io.write_point_cloud(ply_path, combined_pcd)
                             
                        print(f"已保存点云数据: {ply_path}")
                        return True
                    except Exception as save_error:
                        print(f"保存点云数据时发生错误: {save_error}")
                        # 如果PLY保存失败，尝试保存为OBJ格式
                        try:
                            obj_path = ply_path.replace('.ply', '.obj')
                            o3d.io.write_triangle_mesh(obj_path, o3d.geometry.TriangleMesh.create_coordinate_frame())
                            print(f"已保存基础OBJ文件: {obj_path}")
                        except:
                            pass
                        return False
                return False
    except ImportError:
        print("警告: 无法导入open3d库，无法进行可视化")
        print("请安装open3d: pip install open3d")
        # 即使没有open3d，也尝试保存点云数据为npy格式
        if save_path is not None:
            npy_path = save_path.replace('.obj', '.npy')
            try:
                if transformed_points is not None:
                    combined_data = {
                        'ref_points': ref_points,
                        'src_points': src_points,
                        'transformed_points': transformed_points
                    }
                    np.save(npy_path, combined_data)
                    print(f"已保存点云数据为NPY格式: {npy_path}")
                else:
                    combined_data = {
                        'ref_points': ref_points,
                        'src_points': src_points
                    }
                    np.save(npy_path, combined_data)
                    print(f"已保存原始点云数据为NPY格式: {npy_path}")
            except Exception as save_error:
                print(f"保存NPY格式数据失败: {save_error}")
        return False
    except Exception as e:
        print(f"可视化过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        # 即使发生错误，也尝试保存点云数据
        if save_path is not None:
            try:
                import open3d as o3d
                # 保存最后备选的点云数据
                ply_path = save_path.replace('.obj', '.ply')
                combined_pcd = o3d.geometry.PointCloud()
                combined_pcd.points = o3d.utility.Vector3dVector(ref_points)
                o3d.io.write_point_cloud(ply_path, combined_pcd)
                print(f"已保存参考点云数据: {ply_path}")
            except:
                pass
        return False

# 函数：点云配准
def register_point_clouds(model, ref_points, src_points, device=None, max_points=5000):
    """使用GeoTransformer模型进行点云配准
    
    Args:
        model: 加载好的GeoTransformer模型
        ref_points: 参考点云 (N, 3)
        src_points: 源点云 (M, 3)
        device: 计算设备，默认为cuda(如果可用)或cpu
        max_points: 点云最大点数限制
    
    Returns:
        success: 是否配准成功
        transformed_points: 变换后的源点云
        transform_matrix: 估计的变换矩阵
    """
    try:
        # 确定设备
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"模型运行在设备: {device}")
        
        # 将模型移动到指定设备
        model = model.to(device)
        
        # 对点云进行预处理，严格按照demo.py的方式
        # 1. 点云缩放 - 关键步骤！
        # 根据自定义数据集特点，可能需要调整缩放比例
        # ref_points = ref_points / 10000.0
        # src_points = src_points / 10000.0
        print(f"点云缩放后，参考点云点数: {ref_points.shape[0]}，源点云点数: {src_points.shape[0]}")
        
        # 2. 如果点云太大，进行下采样以减少内存使用
        print(f"最大点数限制: {max_points}")
        if ref_points.shape[0] > max_points:
            ref_points = random_sampling(ref_points, max_points)
            print(f"参考点云下采样至 {ref_points.shape[0]} 点")
        
        if src_points.shape[0] > max_points:
            src_points = random_sampling(src_points, max_points)
            print(f"源点云下采样至 {src_points.shape[0]} 点")
        
        # 3. 创建特征向量 - 与demo.py完全一致
        ref_feats = np.ones_like(ref_points[:, :1])
        src_feats = np.ones_like(src_points[:, :1])
        
        # 4. 创建基础数据字典 - 先使用numpy格式
        data_dict = {
            "ref_points": ref_points.astype(np.float32),
            "src_points": src_points.astype(np.float32),
            "ref_feats": ref_feats.astype(np.float32),
            "src_feats": src_feats.astype(np.float32),
            "transform": np.eye(4, dtype=np.float32)  # 添加单位变换矩阵
        }
        
        # 5. 使用registration_collate_fn_stack_mode处理数据，使用与demo.py相同的参数
        try:
            from mic_geotrans.utils.data import registration_collate_fn_stack_mode
            from mic_geotrans.utils.torch import to_cuda
            
            # 设置neighbor_limits，与3DMatch数据集相同
            neighbor_limits = [38, 36, 36, 38]
            
            # 使用与demo.py相同的参数调用collate函数
            # 使用固定参数，与3DMatch配置一致
            num_stages = 4
            init_voxel_size = 0.025
            init_radius = 0.0625
            
            data_dict = registration_collate_fn_stack_mode(
                [data_dict], 
                num_stages,  
                init_voxel_size,
                init_radius,
                neighbor_limits
            )
            
            print("成功使用registration_collate_fn_stack_mode处理数据")
            
            # 6. 使用to_cuda函数将数据移动到设备，与demo.py保持一致
            data_dict = to_cuda(data_dict)
            
        except Exception as e:
            print(f"使用registration_collate_fn_stack_mode失败: {e}")
            print("尝试使用简化的数据格式...")
            
            # 如果collate函数失败，尝试构建最小化的数据格式
            # 转换为tensor
            ref_points_tensor = torch.tensor(ref_points, dtype=torch.float32)
            src_points_tensor = torch.tensor(src_points, dtype=torch.float32)
            ref_feats_tensor = torch.tensor(ref_feats, dtype=torch.float32)
            src_feats_tensor = torch.tensor(src_feats, dtype=torch.float32)
            
            # 连接点云和特征
            points = torch.cat([ref_points_tensor, src_points_tensor], dim=0)
            features = torch.cat([ref_feats_tensor, src_feats_tensor], dim=0)
            
            # 创建简化的数据字典
            data_dict = {
                'features': features,
                'points': [points],  # 多尺度点云列表（至少需要一层）
                'lengths': [[len(ref_points), len(src_points)]],  # 点云长度
                'transform': torch.eye(4).unsqueeze(0),
            }
            
            # 将数据移动到设备
            def move_to_device(obj, device):
                """递归地将数据结构中的所有张量移动到指定设备"""
                if isinstance(obj, torch.Tensor):
                    return obj.to(device)
                elif isinstance(obj, list):
                    return [move_to_device(item, device) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(move_to_device(item, device) for item in obj)
                elif isinstance(obj, dict):
                    return {key: move_to_device(value, device) for key, value in obj.items()}
                else:
                    return obj
            
            data_dict = move_to_device(data_dict, device)
        
        # 确保模型在评估模式
        model.eval()
        
        # 进行推理，禁用梯度计算
        outputs = None  # 初始化outputs变量
        with torch.no_grad():
            try:
                # 直接调用模型，与demo.py保持一致
                outputs = model(data_dict)
                
                # 检查输出是否包含estimated_transform
                if 'estimated_transform' not in outputs:
                    print("警告: 输出中未找到'estimated_transform'键")
                    # 尝试查找其他可能的变换矩阵键
                    if isinstance(outputs, dict):
                        transform_keys = [key for key in outputs.keys() if 'transform' in key.lower()]
                        if transform_keys:
                            print(f"尝试使用{transform_keys[0]}作为变换矩阵")
                            estimated_transform = outputs[transform_keys[0]]
                        else:
                            print(f"可用的输出键: {list(outputs.keys())}")
                            print("无法找到变换矩阵")
                            return False, None, None
                    else:
                        print("输出不是字典格式")
                        return False, None, None
                else:
                    estimated_transform = outputs['estimated_transform']
                
                # 确保变换矩阵形状正确
                if len(estimated_transform.shape) == 3 and estimated_transform.shape[0] == 1:
                    estimated_transform = estimated_transform[0]  # 移除批次维度
                
                # 转换回numpy数组
                if torch.is_tensor(estimated_transform):
                    # 先移回CPU再转换为numpy
                    estimated_transform = estimated_transform.cpu().numpy()
                
                # 构建完整的4x4变换矩阵
                transform_matrix = np.eye(4)
                if estimated_transform.shape == (3, 4):
                    transform_matrix[:3, :4] = estimated_transform
                else:
                    transform_matrix = estimated_transform
                
                # 应用变换到源点云
                # 注意：需要先对原始点云进行缩放，再应用变换，最后恢复缩放
                src_points_transformed = apply_transform(src_points, transform_matrix)
                
                # 将结果再乘以10000，恢复原始尺度
                # src_points_transformed = src_points_transformed * 10000.0
                
                return True, src_points_transformed, transform_matrix
                
            except RuntimeError as e:
                print(f"CUDA内存错误: {e}")
                print("尝试减小点云大小并重试...")
                # 如果内存不足，尝试更激进的下采样
                smaller_ref = random_sampling(ref_points, 2000)
                smaller_src = random_sampling(src_points, 2000)
                print(f"已使用更小的点云：参考点云{smaller_ref.shape[0]}点，源点云{smaller_src.shape[0]}点")
                
                # 重新创建数据字典并尝试
                small_ref_feats = np.ones_like(smaller_ref[:, :1])
                small_src_feats = np.ones_like(smaller_src[:, :1])
                
                small_data_dict = {
                    "ref_points": smaller_ref.astype(np.float32),
                    "src_points": smaller_src.astype(np.float32),
                    "ref_feats": small_ref_feats.astype(np.float32),
                    "src_feats": small_src_feats.astype(np.float32),
                    "transform": np.eye(4, dtype=np.float32)
                }
                
                try:
                    from mic_geotrans.utils.data import registration_collate_fn_stack_mode
                    from mic_geotrans.utils.torch import to_cuda
                    
                    small_data_dict = registration_collate_fn_stack_mode(
                        [small_data_dict], 
                        4, 
                        0.025,
                        0.0625,
                        [38, 36, 36, 38]
                    )
                    
                    small_data_dict = to_cuda(small_data_dict)
                    
                    # 再次尝试推理
                    small_outputs = model(small_data_dict)
                    
                    # 提取变换矩阵
                    if 'estimated_transform' in small_outputs:
                        small_estimated_transform = small_outputs['estimated_transform']
                        if len(small_estimated_transform.shape) == 3 and small_estimated_transform.shape[0] == 1:
                            small_estimated_transform = small_estimated_transform[0]
                        
                        if torch.is_tensor(small_estimated_transform):
                            small_estimated_transform = small_estimated_transform.cpu().numpy()
                        
                        small_transform_matrix = np.eye(4)
                        if small_estimated_transform.shape == (3, 4):
                            small_transform_matrix[:3, :4] = small_estimated_transform
                        else:
                            small_transform_matrix = small_estimated_transform
                        
                        # 应用变换到完整的源点云
                        original_src_transformed = apply_transform(src_points, small_transform_matrix)
                        # original_src_transformed = original_src_transformed * 10000.0
                        
                        print("使用更小的点云成功完成配准")
                        return True, original_src_transformed, small_transform_matrix
                except Exception as inner_e:
                    print(f"使用更小点云重试也失败: {inner_e}")
                
                # 如果所有尝试都失败
                if outputs is not None:
                    print(f"输出类型: {type(outputs)}")
                    if isinstance(outputs, dict):
                        print(f"输出键: {list(outputs.keys())}")
                import traceback
                traceback.print_exc()
                return False, None, None
            except Exception as e:
                print(f"模型前向传播错误: {e}")
                if outputs is not None:
                    print(f"输出类型: {type(outputs)}")
                    if isinstance(outputs, dict):
                        print(f"输出键: {list(outputs.keys())}")
                    else:
                        print(f"输出形状: {getattr(outputs, 'shape', '未知')}")
                else:
                    print("输出未定义")
                import traceback
                traceback.print_exc()
                return False, None, None
    except Exception as e:
        print(f"点云配准过程中的错误: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None

def random_sampling(points, num_points):
    """随机采样点云，减少点数以节省内存
    
    Args:
        points: 原始点云 (N, 3)
        num_points: 需要保留的点数
    
    Returns:
        采样后的点云 (num_points, 3)
    """
    if points.shape[0] <= num_points:
        return points
    
    # 随机选择索引
    indices = np.random.choice(points.shape[0], num_points, replace=False)
    return points[indices]

# 主函数
def main():
    """主函数 - 执行点云配准推理"""
    print("===== GeoTransformer 点云配准推理 =====")
    print("自定义数据集版本 - 使用训练好的权重进行配准可视化")
    
    # 1. 解析命令行参数 - 同时支持IDE直接运行和命令行执行
    print("\n1. 解析命令行参数:")
    parser = argparse.ArgumentParser(description='点云配准推理', add_help=False)
    parser.add_argument('--ref_file', type=str, default='../../data/custom/random_Dataset/MiC1/Variation_0032/point_clouds/lidar_1.npy', help='参考点云文件路径')
    parser.add_argument('--src_file', type=str, default='../../data/custom/random_Dataset/MiC1/Variation_0032/point_clouds/lidar_4.npy', help='源点云文件路径')
    parser.add_argument('--snapshot', type=str, default='./output/custom_geotransformer/snapshots/snapshot.pth.tar', help='模型权重文件路径')
    parser.add_argument('--exp_dir', type=str, default='.', help='实验目录路径')
    parser.add_argument('--output_dir', type=str, default='output/registration', help='输出目录')
    parser.add_argument('--visualize', type=lambda x: x.lower() == 'true', default=True, help='是否可视化配准结果 (True/False)')
    parser.add_argument('--save_visualization', action='store_true', default=True, help='是否保存可视化结果')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='强制使用CPU运行')
    parser.add_argument('--max_points', type=int, default=1000, help='点云最大点数')
    
    # 使用 parse_known_args() 确保IDE中直接运行时不会因未知参数报错
    args, unknown = parser.parse_known_args()
    
    # 显示参数设置
    print(f"命令行参数设置:")
    print(f"- 实验目录: {args.exp_dir}")
    print(f"- 模型权重: {args.snapshot}")
    print(f"- 参考点云: {args.ref_file}")
    print(f"- 源点云: {args.src_file}")
    print(f"- 输出目录: {args.output_dir}")
    print(f"- 可视化结果: {args.visualize}")
    print(f"- 保存可视化: {args.save_visualization}")
    
    # 设置随机种子确保结果可复现
    set_random_seed()
    
    # 确保输出目录存在
    ensure_dir(args.output_dir)
    
    # 2. 导入模型模块
    print("\n2. 导入模型模块:")
    import_success, model_module = import_model_module(args.exp_dir)
    if not import_success:
        print("无法导入模型模块，显示点云信息后结束")
    else:
        print("模型模块导入成功")
    
    # 3. 加载点云数据
    print("\n3. 加载点云数据:")
    pc_success, ref_points, src_points = load_point_clouds(args.ref_file, args.src_file)
    if pc_success:
        print(f"参考点云点数: {ref_points.shape[0]}")
        print(f"源点云点数: {src_points.shape[0]}")
        
        # 如果启用了可视化，在配准前先可视化原始点云
        if args.visualize and not import_success:
            print("\n由于无法导入模型模块，仅可视化原始点云")
            visualize_point_clouds(ref_points, src_points, title="自定义数据集 - 原始点云", show_coordinate_frame=True)
    else:
        print("无法加载点云数据，程序结束")
        return
    
    # 如果没有torch，只显示点云信息
    if torch is None:
        print("\n警告: torch模块未安装，无法执行模型创建和点云配准")
        print("请安装PyTorch后再运行完整功能")
        print("\n已完成基本功能演示: 参数解析、点云加载")
        
        # 尝试可视化原始点云
        if args.visualize:
            print("\n可视化原始点云")
            visualize_point_clouds(ref_points, src_points, title="自定义数据集 - 原始点云", show_coordinate_frame=True)
        return
    
    # 4. 创建和加载模型
    print("\n4. 创建和加载模型:")
    model_success, net = create_and_load_model(model_module, args.snapshot)
    if not model_success:
        print("无法创建或加载模型，程序结束")
        
        # 尝试可视化原始点云
        if args.visualize:
            print("\n可视化原始点云")
            visualize_point_clouds(ref_points, src_points, title="自定义数据集 - 原始点云", show_coordinate_frame=True)
        return
    
    # 5. 执行点云配准
    print("\n5. 执行点云配准:")
    # 获取设备信息，确保模型和数据在同一设备上
    if args.use_cpu:
        device = torch.device('cpu')
        print("强制使用CPU运行")
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 调用点云配准函数，传入max_points参数
    reg_success, src_points_transformed, estimated_transform = register_point_clouds(
        net, ref_points, src_points, device=device, max_points=args.max_points
    )
    
    if reg_success:
        print("\n✅ 点云配准成功！")
        print(f"变换矩阵:\n{estimated_transform}")
        
        # 6. 保存结果
        print("\n6. 保存配准结果:")
        # 保存变换后的点云
        transformed_file = osp.join(args.output_dir, 'transformed_points.npy')
        np.save(transformed_file, src_points_transformed)
        print(f"已保存变换后的点云: {transformed_file}")
        
        # 保存变换矩阵
        transform_file = osp.join(args.output_dir, 'transform_matrix.txt')
        np.savetxt(transform_file, estimated_transform)
        print(f"已保存变换矩阵: {transform_file}")
        
        # 简单评估
        print("\n配准统计:")
        print(f"参考点云点数: {ref_points.shape[0]}")
        print(f"源点云点数: {src_points.shape[0]}")
        print(f"变换后源点云点数: {src_points_transformed.shape[0]}")
        
        # 7. 可视化配准结果
        if args.visualize:
            print("\n7. 可视化配准结果:")
            save_path = None
            if args.save_visualization:
                save_path = osp.join(args.output_dir, 'registration_result.ply')
            visualize_point_clouds(ref_points, src_points, src_points_transformed, save_path, title="自定义数据集 - 配准成功结果", show_coordinate_frame=True)
    else:
        print("\n❌ 点云配准失败！")
        
        # 尝试可视化原始点云
        if args.visualize:
            print("\n可视化原始点云")
            visualize_point_clouds(ref_points, src_points, title="自定义数据集 - 原始点云", show_coordinate_frame=True)
    
    print("\n配准过程完成！")

# 如果直接运行此脚本
if __name__ == "__main__":
    main()