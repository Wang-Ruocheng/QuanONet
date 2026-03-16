"""
Unified logging and directory management.
"""
import os
import sys
import logging
import json
from datetime import datetime

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    print("Warning: torch.utils.tensorboard not found. TensorBoard logging will be disabled.")
    SummaryWriter = None

class StreamToLogger(object):
    """Redirects stdout/stderr to logging mechanism."""
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup_logger(log_file):
    """Sets up a logger that outputs to both file and console."""
    # Create directory if needed
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger = logging.getLogger('training')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()
        
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    
    # Console handler
    ch = logging.StreamHandler(sys.__stdout__)
    ch.setLevel(logging.INFO)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def get_experiment_id(config):
    """
    Generates a highly detailed, unique descriptor string for filenames.
    Prevents parameter ablation experiments from overwriting each other.
    """
    # 兼容可能存在的不同键名 ('operator' vs 'operator_type')
    op = config.get('operator_type', config.get('operator', 'Unknown'))
    model = config.get('model_type', 'Unknown')
    nt = config.get('num_train', '?')
    np_ = config.get('num_points', '?')
    seed = config.get('random_seed', config.get('seed', 0))
    
    # 1. 基础命名
    exp_id = f"{op}_{model}"
    
    # 2. 网络尺寸 (net_size)
    net = config.get('net_size')
    if isinstance(net, list) and len(net) > 0:
        net_str = "-".join(map(str, net))
        exp_id += f"_Net{net_str}"
    elif net is not None:
        exp_id += f"_Net{net}"
        
    # 3. 量子特有参数 (Quantum Specifics)
    if model in ['QuanONet', 'HEAQNN']:
        # 量子比特数
        nq = config.get('num_qubits', 5)
        exp_id += f"_Q{nq}"
        
        # 是否开启 TF (Trainable Frequency)
        if_tf = str(config.get('if_trainable_freq', 'false')).lower() == 'true'
        exp_id += "_TF" if if_tf else "_FF"
        
        # Scale Coefficient
        scale = config.get('scale_coeff', 0.01)
        exp_id += f"_S{scale}"
        
        # 记录 Pauli 基底 (默认为 Z，如果不是 Z 则显式标出)
        pauli = config.get('ham_pauli', 'Z')
        if pauli != 'Z':
            exp_id += f"_Pauli{pauli}"
            
        # 记录对角谱或边界 (Diag 优先级最高)
        diag = config.get('ham_diag')
        if diag:
            # 如果是自定义谱矩阵
            diag_str = "-".join(map(str, diag))
            exp_id += f"_Diag{diag_str}"
        else:
            # 如果只是普通的 bound
            ham = config.get('ham_bound')
            if ham and isinstance(ham, list) and ham != [-5, 5]: # 仅当非默认值时标出
                ham_str = "-".join(map(str, ham))
                exp_id += f"_Ham{ham_str}"    
                
    # 4. 数据量与随机种子
    exp_id += f"_{nt}x{np_}_Seed{seed}"
    
    return exp_id


class ExperimentLogger:
    """
    Unified manager for directories, TensorBoard, and JSON logging.
    Follows the standard structure: outputs/Operator/Experiment_Name/...
    """
    def __init__(self, config, base_output_dir="outputs"):
        self.config = config
        self.operator_name = config.get('operator_type', config.get('operator', 'Unknown'))
        
        # 自动生成具有唯一标识的实验名称
        self.exp_name = get_experiment_id(config)
        
        # 1. 定义核心层级路径
        self.base_dir = os.path.join(base_output_dir, self.operator_name)
        self.exp_dir = os.path.join(self.base_dir, self.exp_name)
        self.tb_dir = os.path.join(self.base_dir, "tensorboard", self.exp_name)
        
        # 2. 自动创建嵌套文件夹
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        
        # 3. 初始化 TensorBoard Writer (保存到该算子独立的 tb 目录下)
        self.writer = SummaryWriter(log_dir=self.tb_dir) if SummaryWriter else None
        
        # 4. 获取当前实验的专属纯文本 log 路径
        self.text_log_path = os.path.join(self.exp_dir, "train.log")
        
        # 5. 实验初始化时，立刻持久化配置文件
        self.save_args()

    def save_args(self):
        """保存实验配置到 train_args.json"""
        args_path = os.path.join(self.exp_dir, "train_args.json")
        with open(args_path, 'w') as f:
            json.dump(self.config, f, indent=4, default=str)
            
    def log_metric(self, tag, value, step):
        """向 TensorBoard 写入标量数据 (例如 Loss/train, MSE/test)"""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def save_metrics(self, metrics, history=None):
        """保存最终评估结果到 metric.json"""
        metric_path = os.path.join(self.exp_dir, "metric.json")
        data = {
            'metrics': metrics,
        }
        if history is not None:
            data['history'] = history
            
        with open(metric_path, 'w') as f:
            json.dump(data, f, indent=4, default=str)
        print(f"Results saved to {metric_path}")

    def get_ckpt_path(self, iteration=None, is_final=False):
        """获取标准化的 Checkpoint 保存路径"""
        if is_final:
            return os.path.join(self.exp_dir, "final.ckpt")
        if iteration is not None:
            return os.path.join(self.exp_dir, f"iter_{iteration:05d}.ckpt")
        return os.path.join(self.exp_dir, "best_model.ckpt")
        
    def is_completed(self):
        """检查该实验是否已经完成 (用于优雅的断点续跑)"""
        metric_path = os.path.join(self.exp_dir, "metric.json")
        return os.path.exists(metric_path)
        
    def close(self):
        """关闭 TensorBoard Writer"""
        if self.writer:
            self.writer.close()