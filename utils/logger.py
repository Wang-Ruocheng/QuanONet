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
    """Generates a unique descriptor string for filenames."""
    op = config['operator_type']
    model = config['model_type']
    nt = config.get('num_train', '?')
    np_ = config.get('num_points', '?')
    seed = config.get('random_seed', 0)
    
    # Handle net_size which might be a list
    net = config.get('net_size', 'default')
    if isinstance(net, list):
        net = "-".join(map(str, net))
        
    return f"{op}_{model}_{nt}x{np_}_{net}_{seed}"

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
    
    # Handle int64 serialization error by using default=str or custom encoder if needed
    # Here we rely on the fact that metrics are already floats from compute_metrics
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4, default=str)
        
    print(f"Results saved to {filepath}")