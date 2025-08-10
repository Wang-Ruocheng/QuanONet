"""
Data generation functions for various operator problems.
"""

# Direct dependency imports to avoid relative import issues
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

# Data generation configuration constants
DATA_GENERATION_CONFIG = {
    'save_interval': 100,      # Save data every N generations
    'backup_interval': 500,    # Create backup every N samples
    'progress_update': 10,     # Update progress info every N times
}

def save_data_with_backup(data_path, u_cals, u0_cals, description, backup_interval=None):
    """
    Save data and optionally create backup
    
    Args:
        data_path: Main data file path
        u_cals: u data list
        u0_cals: u0 data list  
        description: Data description
        backup_interval: Interval for creating backups (number of samples), if None use config default
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
    Find the latest backup file
    
    Args:
        data_path: Main data file path
    
    Returns:
        Latest backup file path, returns None if no backup exists
    """
    backup_dir = os.path.join(os.path.dirname(data_path), 'backups')
    if not os.path.exists(backup_dir):
        return None
    
    base_name = os.path.splitext(os.path.basename(data_path))[0]
    backup_files = glob.glob(os.path.join(backup_dir, f"{base_name}_backup_*samples_*.npz"))
    
    if not backup_files:
        return None
    
    # Sort by file modification time, return the latest
    latest_backup = max(backup_files, key=os.path.getmtime)
    return latest_backup


def load_data_with_recovery(data_path):
    """
    Load data, try to recover from backup if main file is corrupted
    
    Args:
        data_path: Main data file path
    
    Returns:
        (u_cals, u0_cals) tuple, returns ([], []) if both loading fails
    """
    # First try to load main file
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
    
    # If main file doesn't exist or is corrupted, try to recover from backup
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


def RBF(x1, x2, params):
    """Radial Basis Function kernel."""
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)  # Calculate squared difference sum
    return output_scale * np.exp(-0.5 * r2)  # Return RBF kernel result


def generate_random_gaussian_field(m, length_scale=0.2):
    """Generate random Gaussian field using Gaussian process."""
    N = 1024
    jitter = 1e-10
    gp_params = (1.0, length_scale)
    X = np.linspace(0, 1, N)[:, None]
    K = RBF(X, X, gp_params)  # RBF is radial basis function (squared exponential kernel)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    key_train = ms.Tensor(np.random.randn(N)).asnumpy()
    gp_sample = np.dot(L, key_train)
    u_fn = lambda x: np.interp(x, X.flatten(), gp_sample)
    x = np.linspace(0, 1, m)
    u = u_fn(x)
    return u_fn, u

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

# Predefined PDE systems
PDE_SYSTEMS = {
    'RDiffusion': {
        'description': 'RDiffusion PDE: ∂u/∂t = α∇²u + k*u² + u0(x)',
        'default_params': {'alpha': 0.01, 'k': 0.01}
    },
    'Heat': {
        'description': 'Heat equation: ∂u/∂t = α∇²u + u0(x)',
        'default_params': {'alpha': 0.01}
    },
    'Wave': {
        'description': 'Wave equation: ∂²u/∂t² = c²∇²u + u0(x,t)',
        'default_params': {'c': 1.0}
    }
}

def generate_ODE_Operator_data(operator_type, num_train, num_test, num_points, 
                              length_scale=0.2, num_cal=1000, custom_ode_func=None, 
                              custom_name=None):
    """
    General ODE operator data generation function
    
    Args:
        operator_type: Operator type, options: 'Inverse', 'Homogeneous', 'Nonlinear' or 'Custom'
        num_train: Number of training samples
        num_test: Number of test samples
        num_points: Number of spatial discrete points
        length_scale: Length scale of Gaussian field
        num_cal: Number of high-precision grid points for calculation
        custom_ode_func: Custom ODE function, format: lambda u0_fn: lambda x, u: your_equation
        custom_name: Custom operator name, used for file path
    
    Returns:
        Training and test data tuple: (train_u0, train_u, test_u0, test_u, x)
    """
    # Determine operator name and ODE function
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
    
    # Try to load existing data, support recovery from backup
    u_cals, u0_cals = load_data_with_recovery(data_path)
    
    # Generate insufficient data
    if len(u_cals) < num_train + num_test:
        print(f"Generating {description}")
        total_needed = num_train + num_test - len(u_cals)
        save_interval = DATA_GENERATION_CONFIG['save_interval']  # Use save interval from config
        
        for i in tqdm(range(total_needed), desc=f"Generating {operator_name} Data"):
            u0_fn, u0_cal_new = generate_random_gaussian_field(num_cal, length_scale)
            
            # Use specified ODE system
            ode_system = ode_func_generator(u0_fn)
            
            try:
                sol = solve_ivp(ode_system, [x_cal[0], x_cal[-1]], [0], t_eval=x_cal, method='RK45')
                u_cal_new = sol.y[0]
            except Exception as e:
                print(f"Warning: ODE solving failed for one sample: {e}")
                # Use backup method or skip
                u_cal_new = np.zeros_like(x_cal)
            
            u_cals.append(u_cal_new)
            u0_cals.append(u0_cal_new)
            
            # Save data every save_interval times or at the last time
            if (i + 1) % save_interval == 0 or i == total_needed - 1:
                print(f"Saving intermediate data... Generated {len(u_cals)}/{num_train + num_test} samples")
                save_data_with_backup(data_path, u_cals, u0_cals, description)
        
        # Final save confirmation
        print(f"Final save: Generated {len(u_cals)} samples total")
        save_data_with_backup(data_path, u_cals, u0_cals, description)
        data = np.load(data_path, allow_pickle=True)
    
    # Load data
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
    
    # Randomly split training and test sets
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

def generate_PDE_Operator_data(pde_type, num_train, num_test, num_points, 
                              length_scale=0.2, num_cal=100, custom_pde_func=None, 
                              custom_name=None, **pde_params):
    """
    General PDE operator data generation function
    
    Args:
        pde_type: PDE type, options: 'RDiffusion', 'Advection', 'Heat', 'Wave' or 'Custom'
        num_train: Number of training samples
        num_test: Number of test samples
        num_points: Number of spatial discrete points
        length_scale: Length scale of Gaussian field
        num_cal: Number of high-precision grid points for calculation
        custom_pde_func: Custom PDE solving function
        custom_name: Custom PDE name, used for file path
        **pde_params: PDE-specific parameters
    
    Returns:
        Training and test data tuple: (train_u0, train_u, test_u0, test_u, x, t)
    """
    # Determine PDE name and parameters
    if pde_type == 'Custom':
        if custom_pde_func is None or custom_name is None:
            raise ValueError("Custom PDE requires both custom_pde_func and custom_name")
        pde_name = custom_name
        pde_func = custom_pde_func
        description = f"Custom PDE: {custom_name}"
    elif pde_type in ['RDiffusion', 'Advection', 'Heat', 'Wave']:
        pde_name = pde_type
        description = f"{pde_type} equation"
        
        # Define default parameters for each PDE type
        default_params = {
            'RDiffusion': {'alpha': 0.01, 'k': 0.01},
            'Advection': {'nu': 0.001},
            'Heat': {'alpha': 0.01},
            'Wave': {'c': 1.0}
        }
        
        # Merge default parameters with user parameters
        final_params = default_params.get(pde_type, {})
        final_params.update(pde_params)
        
        # Select solver based on PDE type
        if pde_type == 'RDiffusion':
            pde_func = lambda: _solve_rdiffusion_pde(num_cal, length_scale, **final_params)
        elif pde_type == 'Heat':
            pde_func = lambda: _solve_heat_pde(num_cal, length_scale, **final_params)
        elif pde_type == 'Wave':
            pde_func = lambda: _solve_wave_pde(num_cal, length_scale, **final_params)
    else:
        raise ValueError(f"Unknown PDE type: {pde_type}")
    
    # Data path
    data_path = f'data/{pde_name}_Operator_data/{pde_name}_Operator_data_{num_cal}_1.npz'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Try to load existing data, support recovery from backup
    u_cals, u0_cals = load_data_with_recovery(data_path)
    
    # Generate insufficient data
    if len(u_cals) < num_train + num_test:
        print(f"Generating {description}")
        total_needed = num_train + num_test - len(u_cals)
        save_interval = DATA_GENERATION_CONFIG['save_interval']  # Use save interval from config
        
        for i in tqdm(range(total_needed), desc=f"Generating {pde_name} Data"):
            # Call solver to get new samples
            u_cal_new, u0_cal_new = pde_func()
            
            u_cals.append(u_cal_new)
            u0_cals.append(u0_cal_new)
            
            # Save data every save_interval times or at the last time
            if (i + 1) % save_interval == 0 or i == total_needed - 1:
                print(f"Saving intermediate data... Generated {len(u_cals)}/{num_train + num_test} samples")
                save_data_with_backup(data_path, u_cals, u0_cals, description)
        
        # Final save confirmation
        print(f"Final save: Generated {len(u_cals)} samples total")
        save_data_with_backup(data_path, u_cals, u0_cals, description)
    
    # Ensure data is loaded (whether newly generated or loaded from existing file)
    # If data is loaded from file in list form, use directly
    # If need to reload from file, then load file
    if isinstance(u_cals, list) and len(u_cals) > 0:
        # Data is already in list form, use directly
        pass
    else:
        # Reload data from file
        data = np.load(data_path, allow_pickle=True)
        u_cals = data['u_cals']
        u0_cals = data['u0_cals']
    
    # Interpolate to target grid
    x = np.linspace(0, 1, num_points)
    t = np.linspace(0, 1, num_points)
    us, u0s = [], []
    for u_cal, u0_cal in zip(u_cals, u0_cals):
        # For 2D data, perform space-time interpolation
        if len(u_cal.shape) == 2:
            # u_cal shape: (num_cal_x, num_cal_t)
            u = np.array([np.interp(x, np.linspace(0, 1, u_cal.shape[0]), u_cal[:, t_idx]) 
                         for t_idx in range(u_cal.shape[1])])  # Shape: (time, space)
        else:
            # 1D data processing
            u = np.interp(x, np.linspace(0, 1, len(u_cal)), u_cal)
        
        # u0 is initial condition, usually 1D
        if len(u0_cal.shape) == 1:
            u0 = np.interp(x, np.linspace(0, 1, len(u0_cal)), u0_cal)
        else:
            # Multi-dimensional initial condition handling
            u0 = np.interp(x, np.linspace(0, 1, u0_cal.shape[0]), u0_cal[:, 0])
        
        us.append(u)
        u0s.append(u0)
    
    # Randomly split training and test sets
    train_index = np.random.choice(num_train + num_test, num_train, replace=False)
    test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])
    
    return (ms.Tensor(np.array(u0s)[train_index], ms.float32), 
            ms.Tensor(np.array(us)[train_index], ms.float32), 
            ms.Tensor(np.array(u0s)[test_index], ms.float32), 
            ms.Tensor(np.array(us)[test_index], ms.float32), 
            ms.Tensor(x, ms.float32),
            ms.Tensor(t, ms.float32))


def _solve_rdiffusion_pde(num_cal, length_scale, alpha=0.01, k=0.01):
    """Solve reaction-diffusion equation"""
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)
    
    # Calculate time step parameters
    dx = x_cal[1] - x_cal[0]
    dt = min(dx**2 / (2 * alpha), t_cal[1] - t_cal[0])  # Ensure numerical stability
    num_cal_t = int(1//dt)
    
    def rdiffusion_step(u, dx, dt, alpha, k, u0):
        u_new = np.zeros_like(u)
        for i in range(1, len(u) - 1):
            u_new[i] = u[i] + dt * (alpha * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2) + k * (u[i]**2) + u0[i])
        u_new[0] = u_new[-1] = 0  # Boundary conditions
        return u_new
    
    # Generate initial conditions
    _, u0_cal = generate_random_gaussian_field(num_cal, length_scale)
    
    # Time evolution
    u_cal = np.zeros((num_cal, num_cal_t))
    for i in range(1, num_cal_t):
        u_cal[:, i] = rdiffusion_step(u_cal[:, i-1], dx, dt, alpha, k, u0_cal)
    
    # Uniformly sample in time dimension, reduce to num_cal
    u_cal_sampled = u_cal[:, ::max(1, num_cal_t//num_cal)][:, :num_cal]
    
    return u_cal_sampled, u0_cal

def _solve_heat_pde(num_cal, length_scale, alpha=0.01):
    """Solve heat conduction equation ∂u/∂t = α∇²u + u0(x)"""
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)
    
    # Calculate time step parameters
    dx = x_cal[1] - x_cal[0]
    dt = min(dx**2 / (2 * alpha), t_cal[1] - t_cal[0])
    num_cal_t = int(1//dt)
    
    def heat_step(u, dx, dt, alpha, u0):
        u_new = np.zeros_like(u)
        for i in range(1, len(u) - 1):
            u_new[i] = u[i] + dt * (alpha * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2) + u0[i])
        u_new[0] = u_new[-1] = 0  # Boundary conditions
        return u_new
    
    # Generate initial conditions and source term
    _, u0_cal = generate_random_gaussian_field(num_cal, length_scale)
    
    # Time evolution
    u_cal = np.zeros((num_cal, num_cal_t))
    u_cal[:, 0] = u0_cal
    for i in range(1, num_cal_t):
        u_cal[:, i] = heat_step(u_cal[:, i-1], dx, dt, alpha, u0_cal)
    
    return u_cal, u0_cal

def _solve_wave_pde(num_cal, length_scale, c=1.0):
    """Solve wave equation ∂²u/∂t² = c²∇²u + u0(x,t)"""
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)
    
    # Calculate time step parameters
    dx = x_cal[1] - x_cal[0]
    dt = 0.5 * dx / abs(c)  # CFL condition
    num_cal_t = int(1//dt)
    
    def wave_step(u_prev, u_curr, dx, dt, c, u0):
        u_new = np.zeros_like(u_curr)
        r = (c * dt / dx) ** 2
        for i in range(1, len(u_curr) - 1):
            u_new[i] = (2 * u_curr[i] - u_prev[i] + 
                       r * (u_curr[i+1] - 2*u_curr[i] + u_curr[i-1]) + 
                       dt**2 * u0[i])
        u_new[0] = u_new[-1] = 0  # Boundary conditions
        return u_new
    
    # Generate initial conditions and source term
    _, u0_cal = generate_random_gaussian_field(num_cal, length_scale)
    
    # Time evolution (requires initial conditions for two time steps)
    u_cal = np.zeros((num_cal, num_cal_t))
    u_cal[:, 0] = u0_cal
    u_cal[:, 1] = u0_cal * 0.9  # Simple second time step initial condition
    
    for i in range(2, num_cal_t):
        u_cal[:, i] = wave_step(u_cal[:, i-2], u_cal[:, i-1], dx, dt, c, u0_cal)
    
    return u_cal, u0_cal

def generate_RDiffusion_Operator_data(num_train, num_test, num_points, length_scale=0.2, 
                                     k=0.01, alpha=0.01, num_cal=100):
    """Generate data for rdiffusion operator problem."""
    return generate_PDE_Operator_data(
        'RDiffusion', num_train, num_test, num_points, 
        length_scale=length_scale, num_cal=num_cal, 
        alpha=alpha, k=k
    )


def generate_Heat_Operator_data(num_train, num_test, num_points, length_scale=0.2, 
                               alpha=0.01, num_cal=100):
    """Generate data for heat equation: ∂u/∂t = α∇²u + u0(x)."""
    return generate_PDE_Operator_data(
        'Heat', num_train, num_test, num_points,
        length_scale=length_scale, num_cal=num_cal,
        alpha=alpha
    )


def generate_Wave_Operator_data(num_train, num_test, num_points, length_scale=0.2, 
                               c=1.0, num_cal=100):
    """Generate data for wave equation: ∂²u/∂t² = c²∇²u + u0(x,t)."""
    return generate_PDE_Operator_data(
        'Wave', num_train, num_test, num_points,
        length_scale=length_scale, num_cal=num_cal,
        c=c
    )


def generate_Custom_PDE_Operator_data(num_train, num_test, num_points, length_scale=0.2, 
                                     num_cal=100, custom_pde_func=None, custom_name=None, 
                                     **custom_params):
    """
    Generate data for custom PDE operator problem.
    
    Args:
        custom_pde_func: Custom PDE solving function, should return (u_cal, u0_cal)
        custom_name: Custom PDE name
        **custom_params: Custom PDE parameters
    """
    if custom_pde_func is None or custom_name is None:
        raise ValueError("Custom PDE requires both custom_pde_func and custom_name")
    
    return generate_PDE_Operator_data(
        'Custom', num_train, num_test, num_points,
        length_scale=length_scale, num_cal=num_cal,
        custom_pde_func=custom_pde_func, custom_name=custom_name,
        **custom_params
    )