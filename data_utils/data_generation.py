"""
Data generation functions for various operator problems.
"""

import numpy as np
import os
import math
import copy
import logging
import sys
import glob
import time
import pickle
import itertools

# MindSpore imports
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore.ops import operations as P
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.nn import Adam, MSELoss, WithLossCell, TrainOneStepCell
from mindspore.common.initializer import Uniform, initializer, One
from mindspore import Parameter

# MindQuantum imports
from mindquantum.core.circuit import Circuit, change_param_name, AP, A, add_prefix, add_suffix
from mindquantum.core.gates import H, RX, RY, RZ, CNOT, Z, BasicGate, X
from mindquantum.core import gates as G
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.framework import MQLayer
from mindquantum.algorithm.nisq._ansatz import Ansatz
from mindquantum.algorithm.nisq.chem.hardware_efficient_ansatz import _check_single_rot_gate_seq
from mindquantum.core.circuit import ReverseAdder, NoiseExcluder, NoiseChannelAdder
from mindquantum.core.circuit.channel_adder import (
    ChannelAdderBase, SequentialAdder, MixerAdder, 
    MeasureAccepter, BitFlipAdder
)

# Scientific computing imports
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

# Progress bar
from tqdm import tqdm

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from data_utils.random_func import *
from data_utils.PDE_SYSTEMS import *

# 数据生成配置常量
DATA_GENERATION_CONFIG = {
    'save_interval': 100,      # Save data every 100 samples
    'backup_interval': 500,    # Create a backup every 500 samples
    'progress_update': 10,     # Update progress every 10 samples
}

def save_data_with_backup(data_path, u_cals, u0_cals, description=None, backup_interval=None):
    """
    Save data to NPZ file with backup functionality.
    """
    if backup_interval is None:
        backup_interval = DATA_GENERATION_CONFIG['backup_interval']

    # Save main file
    np.savez(data_path, u_cals=u_cals, u0_cals=u0_cals, description=description)

    # Create a timestamped backup every backup_interval samples
    if len(u_cals) % backup_interval == 0:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(os.path.dirname(data_path), 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(data_path))[0]
        backup_path = os.path.join(backup_dir, f"{base_name}_backup_{len(u_cals)}samples_{timestamp}.npz")
        np.savez(backup_path, u_cals=u_cals, u0_cals=u0_cals, description=description)
        print(f"Created backup: {backup_path}")


def find_latest_backup(data_path):
    """
    Find the latest backup file for the given data path.
    """
    backup_dir = os.path.join(os.path.dirname(data_path), 'backups')
    if not os.path.exists(backup_dir):
        return None
    
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    backup_files = glob.glob(os.path.join(backup_dir, f"{base_name}_backup_*samples_*.npz"))
    
    if not backup_files:
        return None
    
    # Sort by modification time and return the latest one
    latest_backup = max(backup_files, key=os.path.getmtime)
    return latest_backup


def load_data_with_recovery(data_path):
    """
    Load data from NPZ file with recovery from backup if necessary.
    """
    # First, try to load the main file
    if os.path.exists(data_path):
        try:
            data = np.load(data_path, allow_pickle=True)
            u_cals = list(data['u_cals']) if 'u_cals' in data else []
            u0_cals = list(data['u0_cals']) if 'u0_cals' in data else []
            print(f"Loaded {len(u_cals)} existing samples from main file")
            return u_cals, u0_cals
        except Exception as e:
            print(f"Warning: Failed to load main data file: {e}")
            print("Attempting to recover from backup...")
    
    # If main file loading fails, try to find the latest backup
    latest_backup = find_latest_backup(data_path)
    if latest_backup:
        try:
            data = np.load(latest_backup, allow_pickle=True)
            u_cals = list(data['u_cals']) if 'u_cals' in data else []
            u0_cals = list(data['u0_cals']) if 'u0_cals' in data else []
            print(f"Recovered {len(u_cals)} samples from backup: {latest_backup}")
            return u_cals, u0_cals
        except Exception as e:
            print(f"Warning: Failed to load backup file: {e}")
    
    print("No valid data found, starting from scratch")
    return [], []

# Predefined ODE systems
ODE_SYSTEMS = {
    'Inverse': {
        'description': 'Inverse operator problem: du/dx = u0(x)',
        'ode_func': lambda u0_fn: lambda x, u: u0_fn(x)
    },
    'Homogeneous': {
        'description': 'Homogeneous operator problem: du/dx = u + u0(x)',
        'ode_func': lambda u0_fn: lambda x, u: u + u0_fn(x)
    },
    'Nonlinear': {
        'description': 'Nonlinear operator problem: du/dx = u - u0²(x)',
        'ode_func': lambda u0_fn: lambda x, u: u - u0_fn(x) ** 2
    }
}

def generate_ODE_Operator_data(operator_type, num_train, num_test, num_points, 
                              length_scale=0.2, num_cal=1000, custom_ode_func=None, 
                              custom_name=None):
    """
    Generate data for ODE operator problems.
    """
    # Determine operator name and function
    if operator_type == 'Custom':
        if custom_ode_func is None or custom_name is None:
            raise ValueError("Custom operator requires both custom_ode_func and custom_name")
        operator_name = custom_name
        ode_func_generator = custom_ode_func
        description = f"Custom operator: {custom_name}"
    elif operator_type in ODE_SYSTEMS:
        operator_name = operator_type
        ode_func_generator = ODE_SYSTEMS[operator_type]['ode_func']
        description = ODE_SYSTEMS[operator_type]['description']
    else:
        raise ValueError(f"Unknown operator type: {operator_type}. Available: {list(ODE_SYSTEMS.keys())} or 'Custom'")
    
    # Data path
    data_path = f'data/{operator_name}_Operator_data/{operator_name}_Operator_data_{num_cal}_1.npz'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    x_cal = np.linspace(0, 1, num_cal)
    
    #Try to load existing data, supporting recovery from backup
    u_cals, u0_cals = load_data_with_recovery(data_path)
    
    # If not enough data, generate new samples
    if len(u_cals) < num_train + num_test:
        print(f"Generating {description}")
        total_needed = num_train + num_test - len(u_cals)
        save_interval = DATA_GENERATION_CONFIG['save_interval']  # Use configuration for save interval
        
        for i in tqdm(range(total_needed), desc=f"Generating {operator_name} Data"):
            u0_fn, u0_cal_new = generate_random_gaussian_field(num_cal, length_scale=length_scale)
            
            # Use the ODE function generator to create the ODE system
            ode_system = ode_func_generator(u0_fn)
            
            try:
                sol = solve_ivp(ode_system, [x_cal[0], x_cal[-1]], [0], t_eval=x_cal, method='RK45')
                u_cal_new = sol.y[0]
            except Exception as e:
                print(f"Warning: ODE solving failed for one sample: {e}")
                # Use zero array if ODE solving fails
                u_cal_new = np.zeros_like(x_cal)
            
            u_cals.append(u_cal_new)
            u0_cals.append(u0_cal_new)
            
            # Save data periodically
            if (i + 1) % save_interval == 0 or i == total_needed - 1:
                print(f"Saving intermediate data... Generated {len(u_cals)}/{num_train + num_test} samples")
                save_data_with_backup(data_path, u_cals, u0_cals, description)
        
        # Final save confirmation
        print(f"Final save: Generated {len(u_cals)} samples total")
        save_data_with_backup(data_path, u_cals, u0_cals, description)
        data = np.load(data_path, allow_pickle=True)
    
    # Ensure data is loaded
    u_cals = data['u_cals']
    u0_cals = data['u0_cals']
    
    # Interpolate to target grid
    x = np.linspace(0, 1, num_points)
    us, u0s = [], []
    for u_cal, u0_cal in zip(u_cals, u0_cals):
        u = np.interp(x, x_cal, u_cal)
        u0 = np.interp(x, x_cal, u0_cal)
        us.append(u)
        u0s.append(u0)
    
    # Randomly split into training and testing sets
    train_index = np.random.choice(num_train + num_test, num_train, replace=False)
    test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])
    
    return (ms.Tensor(np.array(u0s)[train_index], ms.float32), 
            ms.Tensor(np.array(us)[train_index], ms.float32), 
            ms.Tensor(np.array(u0s)[test_index], ms.float32), 
            ms.Tensor(np.array(us)[test_index], ms.float32), 
            ms.Tensor(x, ms.float32))

def generate_Inverse_Operator_data(num_train, num_test, num_points, length_scale=0.2, num_cal=1000):
    """Generate data for inverse operator problem: du/dx = u0(x)."""
    return generate_ODE_Operator_data('Inverse', num_train, num_test, num_points, 
                                     length_scale, num_cal)

def generate_Nonlinear_Operator_data(num_train, num_test, num_points, length_scale=0.2, num_cal=1000):
    """Generate data for nonlinear operator problem: du/dx = u - u0²(x)."""
    return generate_ODE_Operator_data('Nonlinear', num_train, num_test, num_points, 
                                     length_scale, num_cal)


def generate_Homogeneous_Operator_data(num_train, num_test, num_points, length_scale=0.2, num_cal=1000):
    """Generate data for homogeneous operator problem: du/dx = u + u0(x)."""
    return generate_ODE_Operator_data('Homogeneous', num_train, num_test, num_points, 
                                     length_scale, num_cal)

def generate_PDE_Operator_data(operator_type, num_train, num_test, num_points, num_points_0=None, 
                              length_scale=0.2, num_cal=100):
    """
    Generate data for PDE operator problems.
    """
    # Data path
    data_path = f'data/{operator_type}_Operator_data/{operator_type}_Operator_data_{num_cal}_1.npz'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Try to load existing data, supporting recovery from backup
    u_cals, u0_cals = load_data_with_recovery(data_path)

    # Generate missing data
    if len(u_cals) < num_train + num_test:
        print(f"Generating {operator_type} Data")
        total_needed = num_train + num_test - len(u_cals)
        save_interval = DATA_GENERATION_CONFIG['save_interval']  # Use configuration for save interval
        
        for i in tqdm(range(total_needed), desc=f"Generating {operator_type} Data"):
            func_operator_type = f"solve_{operator_type.lower()}_pde"
            solve_func = globals()[func_operator_type]
            u_cal_new, u0_cal_new = solve_func(num_cal, length_scale=length_scale)
            u_cals.append(u_cal_new)
            u0_cals.append(u0_cal_new)
            
            # Save data periodically
            if (i + 1) % save_interval == 0 or i == total_needed - 1:
                print(f"Saving intermediate data... Generated {len(u_cals)}/{num_train + num_test} samples")
                save_data_with_backup(data_path, u_cals, u0_cals)
        
        # Final save confirmation
        print(f"Final save: Generated {len(u_cals)} samples total")
        save_data_with_backup(data_path, u_cals, u0_cals)
    
    # Ensure data is loaded
    if isinstance(u_cals, list) and len(u_cals) >= num_train + num_test:
        # If data is already sufficient, do not regenerate
        pass
    else:
        # If data is insufficient, load from file
        data = np.load(data_path, allow_pickle=True)
        u_cals = data['u_cals']
        u0_cals = data['u0_cals']
    
    # Interpolate to target grid
    x = np.linspace(0, 1, num_points)
    t = np.linspace(0, 1, num_points)
    us, u0s = [], []
    if not num_points_0:
        num_points_0 = num_points
    for u_cal, u0_cal in zip(u_cals, u0_cals):
        # Handle different dimensions of u_cal
        if u_cal.ndim == 2:
            # Process 2D data
            num_cal_x, num_cal_t = u_cal.shape
            u = u_cal[::num_cal_x//num_points, ::num_cal_t//num_points]
        else:
            # Process 1D data
            u = np.interp(x, np.linspace(0, 1, len(u_cal)), u_cal)
        
        # Handle different dimensions of u0_cal
        if u0_cal.ndim == 1:
            u0 = np.interp(np.linspace(0, 1, num_points_0), np.linspace(0, 1, len(u0_cal)), u0_cal)
        else:
            u0 = u0_cal[::len(u0_cal)//num_points_0]
        
        us.append(u)
        u0s.append(u0)
    
    # Randomly split into training and testing sets
    train_index = np.random.choice(num_train + num_test, num_train, replace=False)
    test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])
    
    return (ms.Tensor(np.array(u0s)[train_index], ms.float32), 
            ms.Tensor(np.array(us)[train_index], ms.float32), 
            ms.Tensor(np.array(u0s)[test_index], ms.float32), 
            ms.Tensor(np.array(us)[test_index], ms.float32), 
            ms.Tensor(x, ms.float32), 
            ms.Tensor(t, ms.float32))

# def generate_Burgers_Operator_data(num_train, num_test, num_points, length_scale=0.2, 
#                                nu=0.02, num_cal=100):
#     """Generate data for burgers equation: ∂u/∂t + u∂u/∂x = ν∂²u/∂x² + u0(x,t)."""
#     return generate_PDE_Operator_data(
#         'Burgers', num_train, num_test, num_points,
#         length_scale=length_scale, num_cal=num_cal,
#         nu=nu
#     )

# def generate_Identity_Operator_data(num_train, num_test, num_points, length_scale=0.2, 
#                                    num_cal=100):
#     """Generate data for identity operator: u(x,t) = u0(x) for all t."""
#     return generate_PDE_Operator_data(
#         'Identity', num_train, num_test, num_points,
#         length_scale=length_scale, num_cal=num_cal
#     )

# def generate_Schrodinger_Operator_data(num_train, num_test, num_points, length_scale=0.2, 
#                                       num_cal=100, hbar=1.0, m=1.0, sigma=0.05):
#     """
#     Generate data for Schrödinger operator: iℏ∂ψ/∂t = -ℏ²/(2m)∇²ψ + V(x)ψ
    
#     Args:
#         num_train: 训练样本数
#         num_test: 测试样本数  
#         num_points: 网格点数
#         length_scale: 势能函数的长度尺度
#         num_cal: 计算网格分辨率
#         hbar: 约化普朗克常数
#         m: 粒子质量
#         sigma: 初始高斯波包宽度
    
#     Returns:
#         训练和测试数据集
#     """
#     return generate_PDE_Operator_data(
#         'Schrodinger', num_train, num_test, num_points,
#         length_scale=length_scale, num_cal=num_cal,
#         hbar=hbar, m=m, sigma=sigma
#     )

# def generate_Advection_Operator_data(num_train, num_test, num_points, length_scale=0.2, 
#                                     num_cal=100, c=1.0):
#     """
#     Generate data for advection operator: ∂u/∂t + c∇u = 0
    
#     Args:
#         num_train: 训练样本数
#         num_test: 测试样本数
#         num_points: 网格点数
#         length_scale: 初始条件的长度尺度
#         num_cal: 计算网格分辨率
#         c: 对流速度
    
#     Returns:
#         训练和测试数据集，其中u0作为t=0的初始条件
#     """
#     return generate_PDE_Operator_data(
#         'Advection', num_train, num_test, num_points,
#         length_scale=length_scale, num_cal=num_cal,
#         c=c
#     )

# def generate_Darcy_Operator_data(num_train, num_test, num_points, length_scale=0.2, 
#                                  num_cal=25, K=0.1, f=-1.0):
#     """Generate data for Darcy's law: -∇p = μ/κ u + f(x,y)"""
#     return generate_PDE_Operator_data(
#         'Darcy', num_train, num_test, num_points,
#         length_scale=length_scale, num_cal=num_cal,
#         K=K, f=f, num_points_0=4*num_points
#     )