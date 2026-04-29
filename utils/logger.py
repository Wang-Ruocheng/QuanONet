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
    op = config.get('operator_type', config.get('operator', 'Unknown'))
    model = config.get('model_type', 'Unknown')
    nt = config.get('num_train', '?')
    np_ = config.get('num_points', '?')
    seed = config.get('random_seed', config.get('seed', 0))
    
    # 1. Basic Identifier: Operator + Model
    exp_id = f"{op}_{model}"
    
    # 2. Network Size
    net = config.get('net_size')
    if isinstance(net, list) and len(net) > 0:
        net_str = "-".join(map(str, net))
        exp_id += f"_Net{net_str}"
    elif net is not None:
        exp_id += f"_Net{net}"
        
    # 3. Quantum Specifics
    if model in ['QuanONet', 'HEAQNN']:
        nq = config.get('num_qubits', 5)
        exp_id += f"_Q{nq}"
        
        # Trainable Frequency
        if_tf = str(config.get('if_trainable_freq', 'false')).lower() == 'true'
        exp_id += "_TF" if if_tf else "_FF"
        
        # Scale Coefficient
        scale = config.get('scale_coeff', 0.01)
        exp_id += f"_S{scale}"
        
        pauli = config.get('ham_pauli', 'Z')
        if pauli != 'Z':
            exp_id += f"_Pauli{pauli}"
            
        diag = config.get('ham_diag')
        if diag:
            diag_str = "-".join(map(str, diag))
            exp_id += f"_Diag{diag_str}"
        else:
            ham = config.get('ham_bound')
            if ham and isinstance(ham, list) and ham != [-5, 5]:
                ham_str = "-".join(map(str, ham))
                exp_id += f"_Ham{ham_str}"    
                
    # 4. Backend suffix (only when non-default, to prevent directory collisions)
    qb = config.get('quantum_backend', 'mindquantum') or 'mindquantum'
    if model in ['QuanONet', 'HEAQNN'] and qb != 'mindquantum':
        backend_abbr = {'torchquantum': 'TQ', 'qiskit': 'Qiskit'}.get(qb, qb)
        exp_id += f"_{backend_abbr}"

    cb = config.get('classical_backend', 'pytorch') or 'pytorch'
    if model not in ['QuanONet', 'HEAQNN'] and cb != 'pytorch':
        backend_abbr = {'mindspore': 'MS'}.get(cb, cb)
        exp_id += f"_{backend_abbr}"

    # 5. Data volume and random seed
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
        
        self.exp_name = get_experiment_id(config)
        
        # 1. Compute directory paths
        self.base_dir = os.path.join(base_output_dir, self.operator_name)
        self.exp_dir = os.path.join(self.base_dir, self.exp_name)
        self.tb_dir = os.path.join(self.base_dir, "tensorboard", self.exp_name)
        
        # 2. Auto-create directories
        os.makedirs(self.exp_dir, exist_ok=True)
        os.makedirs(self.tb_dir, exist_ok=True)
        
        # 3. Init logger
        self.writer = SummaryWriter(log_dir=self.tb_dir) if SummaryWriter else None
        
        # 4. Setup file logger
        self.text_log_path = os.path.join(self.exp_dir, "train.log")
        
        # 5. Setup logging to file and console
        self.save_args()

    def save_args(self):
        """Saves the configuration arguments to a JSON file for reproducibility."""
        args_path = os.path.join(self.exp_dir, "train_args.json")
        with open(args_path, 'w') as f:
            json.dump(self.config, f, indent=4, default=str)
            
    def log_metric(self, tag, value, step):
        """Writes a scalar metric to TensorBoard."""
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def save_metrics(self, metrics, history=None):
        """Saves final metrics and optionally training history to a JSON file."""
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
        """Generates checkpoint path based on iteration or final model."""
        if is_final:
            return os.path.join(self.exp_dir, "final.ckpt")
        if iteration is not None:
            return os.path.join(self.exp_dir, f"iter_{iteration:05d}.ckpt")
        return os.path.join(self.exp_dir, "best_model.ckpt")
        
    def is_completed(self):
        """Checks if the experiment has already been completed by looking for metric.json."""
        metric_path = os.path.join(self.exp_dir, "metric.json")
        return os.path.exists(metric_path)
        
    def close(self):
        """Closes the TensorBoard writer if it exists.
        """
        if self.writer:
            self.writer.close()