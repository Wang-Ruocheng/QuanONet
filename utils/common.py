"""
Common utilities for argument parsing, config loading, and environment setup.
"""
import os
import json
import random
import argparse
import numpy as np

def get_base_parser():
    """
    Defines arguments common to ALL training scripts (MindSpore & PyTorch).
    """
    parser = argparse.ArgumentParser(description='Operator Learning Platform')
    
    # Core Identity
    parser.add_argument('--operator', '-o', type=str, required=True, help='Operator type (e.g., Inverse, Darcy)')
    parser.add_argument('--model_type', '-m', type=str, required=True, help='Model architecture (e.g., DeepONet, QuanONet)')
    parser.add_argument('--config', '-c', type=str, default=None, help='Path to JSON config file')
    
    # Environment
    parser.add_argument('--random_seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', '-g', type=int, default=None, help='CUDA_VISIBLE_DEVICES ID')
    parser.add_argument('--prefix', '-p', type=str, default=None, help='Output root directory prefix')
    
    # Common Overrides (Data)
    parser.add_argument('--num_train', type=int)
    parser.add_argument('--num_test', type=int)
    parser.add_argument('--num_points', type=int, help='Output grid resolution')
    parser.add_argument('--num_points_0', type=int, help='Input branch resolution')
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--learning_rate', type=float)
    
    # Model Specific (Network size)
    # Accepts list of ints: --net_size 10 20 10
    parser.add_argument('--net_size', type=int, nargs='+', help='Network architecture configuration')
    
    return parser

def load_config(args):
    """
    Loads JSON config and overrides it with command line arguments.
    Order of precedence: CLI Args > JSON Config > Defaults (handled in Solvers)
    """
    config = {}
    
    # 1. Load from JSON file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    
    # 2. Override with CLI arguments (only if they are not None)
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    defaults = {
            'num_train': 1000,
            'num_test': 1000,
            'batch_size': 100,
            'num_epochs': 1000,
            'learning_rate': 0.0001,
            'num_points': 100,     # Output resolution
            'num_points_0': 100,  # Input resolution
            'train_sample_num': 100,
            'test_sample_num': 100
        }
    
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val
    # 3. Ensure essential keys exist
    if 'operator_type' not in config:
        config['operator_type'] = args.operator
    if 'model_type' not in config:
        config['model_type'] = args.model_type
        
    return config

def set_random_seed(seed):
    """
    Safely sets random seeds for available backends (Numpy, Torch, MindSpore).
    """
    if seed is None:
        return
        
    print(f"Setting random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # Try setting PyTorch seed
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
    except ImportError:
        pass

    # Try setting MindSpore seed
    try:
        import mindspore as ms
        ms.set_seed(seed)
    except ImportError:
        pass