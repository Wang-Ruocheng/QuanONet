"""
Common utilities for argument parsing, config loading, and environment setup.
"""
import os
import json
import random
import argparse
import numpy as np

# Parameters that have non-None defaults in argparse but should NOT override JSON config
# when the user hasn't explicitly passed them on the command line.
_ARGPARSE_DEFAULTS = {
    'seed': 0,
    'device_target': 'CPU',
    'train_sample_num': 10,
    'test_sample_num': 100,
    'num_qubits': 5,
    'if_trainable_freq': 'true',
    'ham_bound': [-5, 5],
    'ham_pauli': 'Z',
    'quantum_backend': 'mindquantum',
    'classical_backend': 'pytorch',
}

def get_base_parser():
    """
    Defines arguments common to ALL training scripts (MindSpore & PyTorch).
    Centralized configuration to avoid conflict and duplication.
    All parameters with non-None defaults are set to None here so that
    load_config() can correctly distinguish "user explicitly passed" from
    "argparse filled in the default".  The real defaults live in
    _ARGPARSE_DEFAULTS and in the defaults dict inside load_config().
    """
    parser = argparse.ArgumentParser(description='QuanONet / Operator Learning Platform')
    
    # ==========================================
    # 1. Core Identity
    # ==========================================
    parser.add_argument('--operator', '-o', type=str, required=True, help='Operator type (e.g., Antideriv, Darcy)')
    parser.add_argument('--model_type', '-m', type=str, required=True, help='Model architecture (e.g., DeepONet, QuanONet)')
    parser.add_argument('--config', '-c', type=str, default=None, help='Path to JSON config file')
    
    # ==========================================
    # 2. Environment & Hardware
    # ==========================================
    parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed (default: 0)')
    parser.add_argument('--gpu', '-g', type=str, default=None, help='CUDA_VISIBLE_DEVICES ID (Default: Auto)')
    parser.add_argument('--prefix', '-p', type=str, default=None, help='Output root directory prefix')
    parser.add_argument('--device_target', type=str, default=None, choices=['CPU', 'GPU', 'Ascend'], help='[MS] Device target (default: CPU)')

    # ==========================================
    # 3. Data Configuration
    # ==========================================
    parser.add_argument('--num_train', type=int, help='Number of training samples')
    parser.add_argument('--num_test', type=int, help='Number of test samples')
    parser.add_argument('--num_points', type=int, help='Output grid resolution (Trunk/Target)')
    parser.add_argument('--num_points_0', type=int, help='Input branch resolution (Branch/Source)')
    parser.add_argument('--train_sample_num', type=int, default=None, help='[Data] P_train: Points per function for training (default: 10)')
    parser.add_argument('--test_sample_num', type=int, default=None, help='[Data] P_test: Points per function for testing (default: 100)')
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
    parser.add_argument('--num_qubits', type=int, default=None, help='[Quantum] Number of qubits (default: 5)')
    parser.add_argument('--scale_coeff', type=float, help='[Quantum] Scaling coefficient')
    parser.add_argument('--if_trainable_freq', type=str, default=None, help='[Quantum] Trainable frequency (default: true)')
    parser.add_argument('--ham_bound', type=int, nargs='+', default=None,
                    help='[Quantum] Hamiltonian bounds (e.g., --ham_bound -5 5) (default: -5 5)')
    parser.add_argument('--ham_pauli', type=str, default=None, choices=['X', 'Y', 'Z'],
                        help='Pauli observable basis for the Hamiltonian (default: Z).')
    parser.add_argument('--ham_diag', type=float, nargs='+', default=None,
                        help='Manually specify the exact Hamiltonian eigenvalues (e.g., --ham_diag -5 5 5 5). If set, this strictly overrides --ham_bound and --ham_pauli.')

    # ==========================================
    # 6. Backend Selection
    # ==========================================
    parser.add_argument('--quantum_backend', type=str, default=None,
                        choices=['mindquantum', 'torchquantum', 'qiskit'],
                        help='Quantum simulation backend (default: mindquantum)')
    parser.add_argument('--classical_backend', type=str, default=None,
                        choices=['pytorch', 'mindspore'],
                        help='Classical model backend (default: pytorch)')
    return parser

def load_config(args):
    """
    Loads JSON config and overrides it with command line arguments.
    Order of precedence: CLI Args (explicitly set) > JSON Config > Defaults
    
    Parameters with non-None defaults in argparse are treated as "not set"
    when the user hasn't explicitly passed them, so they won't silently
    overwrite values already present in the JSON config file.
    """
    config = {}
    
    # 1. Load from JSON file if provided
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
        print(f"Loaded configuration from {args.config}")
    
    # 2. Override with CLI arguments.
    #    A value is considered "explicitly set by the user" only if it is not None.
    #    Since all parameters with non-None defaults have been changed to default=None
    #    in get_base_parser(), this correctly captures only user-provided values.
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    # 3. Apply built-in defaults for parameters that were not set by CLI or JSON.
    defaults = {
        'seed': 0,
        'device_target': 'CPU',
        'num_train': 1000,
        'num_test': 1000,
        'batch_size': 100,
        'num_epochs': 1000,
        'learning_rate': 0.0001,
        'num_points': 100,
        'num_points_0': 100,
        'train_sample_num': 10,
        'test_sample_num': 100,
        'num_qubits': 5,
        'if_trainable_freq': 'true',
        'ham_bound': [-5, 5],
        'ham_pauli': 'Z',
        'quantum_backend': 'mindquantum',
        'classical_backend': 'pytorch',
    }
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val

    # 4. Ensure essential keys exist
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