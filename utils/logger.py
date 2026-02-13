"""
Unified logging and directory management.
"""
import os
import sys
import logging
import json
from datetime import datetime

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
        
        # Hamiltonian Bound (如果修改了哈密顿量边界)
        ham = config.get('ham_bound')
        if ham and isinstance(ham, list):
            ham_str = "-".join(map(str, ham))
            exp_id += f"_Ham{ham_str}"

    # 4. 数据量与随机种子
    exp_id += f"_{nt}x{np_}_Seed{seed}"
    
    return exp_id

def save_results(config, metrics, history, save_dir, filename_suffix="eval"):
    """Saves metrics and config to JSON."""
    exp_id = get_experiment_id(config)
    filename = f"{filename_suffix}_{exp_id}.json"
    filepath = os.path.join(save_dir, filename)
    
    os.makedirs(save_dir, exist_ok=True)
    
    data = {
        'config': config,
        'metrics': metrics,
        'history': history
    }
    
    # Handle int64 serialization error by using default=str
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)
        
    print(f"Results saved to {filepath}")