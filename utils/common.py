"""
Common utilities for argument parsing, config loading, and environment setup.
"""
import os
import json
import random
import argparse
import numpy as np
import argparse

def get_base_parser():
    """
    Defines arguments common to ALL training scripts (MindSpore & PyTorch).
    Centralized configuration to avoid conflict and duplication.
    """
    parser = argparse.ArgumentParser(description='QuanONet / Operator Learning Platform')
    
    # ==========================================
    # 1. Core Identity
    # ==========================================
    parser.add_argument('--operator', '-o', type=str, required=True, help='Operator type (e.g., Inverse, Darcy)')
    parser.add_argument('--model_type', '-m', type=str, required=True, help='Model architecture (e.g., DeepONet, QuanONet)')
    parser.add_argument('--config', '-c', type=str, default=None, help='Path to JSON config file')
    
    # ==========================================
    # 2. Environment & Hardware
    # ==========================================
    parser.add_argument('--seed', '-s', type=int, default=0, help='Random seed')
    parser.add_argument('--gpu', '-g', type=str, default=None, help='CUDA_VISIBLE_DEVICES ID (Default: Auto)')
    parser.add_argument('--prefix', '-p', type=str, default=None, help='Output root directory prefix')
    parser.add_argument('--device_target', type=str, default='CPU', choices=['CPU', 'GPU', 'Ascend'], help='[MS] Device target')

    # ==========================================
    # 3. Data Configuration
    # ==========================================
    parser.add_argument('--num_train', type=int, help='Number of training samples')
    parser.add_argument('--num_test', type=int, help='Number of test samples')
    parser.add_argument('--num_points', type=int, help='Output grid resolution (Trunk/Target)')
    parser.add_argument('--num_points_0', type=int, help='Input branch resolution (Branch/Source)')
    parser.add_argument('--train_sample_num', type=int, default=10, help='[Data] P_train: Points per function for training')
    parser.add_argument('--test_sample_num', type=int, default=100, help='[Data] P_test: Points per function for testing')
    parser.add_argument('--num_cal', type=int, default=None, help='[Data] High-fidelity resolution. Default: 1000(ODE)/100(PDE)')
    
    # ==========================================
    # 4. Training Hyperparameters
    # ==========================================
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--num_epochs', type=int, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    
    # ==========================================
    # 5. Model Specific (Network size & Quantum)
    # ==========================================
    # Classical Net Size: --net_size 10 20 10
    parser.add_argument('--net_size', type=int, nargs='+', help='Network architecture configuration')
    
    # Quantum Specific
    parser.add_argument('--num_qubits', type=int, default=5, help='[Quantum] Number of qubits')
    parser.add_argument('--scale_coeff', type=float, help='[Quantum] Scaling coefficient')
    parser.add_argument('--ham_bound', type=int, nargs='+', help='[Quantum] Hamiltonian bounds')
    parser.add_argument('--if_trainable_freq', type=str, default='true', help='[Quantum] Trainable frequency')
    
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