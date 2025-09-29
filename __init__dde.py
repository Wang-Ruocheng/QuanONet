import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import sys
import os
import time
if __name__ == "__main__":
    os.environ["DDE_BACKEND"] = "pytorch"
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import glob
from scipy.integrate import solve_ivp
from tqdm import tqdm
import deepxde as dde
import os
from deepxde.data.sampler import BatchSampler
from deepxde.data.data import Data
from deepxde.nn import FNN, activations
from deepxde.nn.deeponet_strategy import (
    SingleOutputStrategy,
    IndependentStrategy,
    SplitBothStrategy,
    SplitBranchStrategy,
    SplitTrunkStrategy,
)

class FNN_re(nn.Module):
    """Fully-connected neural network."""

    def __init__(
        self, layer_sizes, activation, initializer, regularization=None
    ):
        super().__init__()
        self.activation = activation
        self.regularizer = regularization
        self._auxiliary_vars = None
        self._input_transform = None
        self._output_transform = None
        self.linears = torch.nn.ModuleList()
        for i in range(1, len(layer_sizes)):
            self.linears.append(
                torch.nn.Linear(
                    layer_sizes[i - 1], layer_sizes[i], dtype=torch.float32
                )
            )
            initializer(self.linears[-1].weight)
            self.linears[-1].bias.data.fill_(0.0)
    @property
    def auxiliary_vars(self):
        """Tensors: Any additional variables needed."""
        return self._auxiliary_vars

    @auxiliary_vars.setter
    def auxiliary_vars(self, value):
        self._auxiliary_vars = value

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self._output_transform = transform

    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)
    def forward(self, inputs):
        branch_input, trunk_input = inputs
        x = torch.cat((branch_input, trunk_input), dim=1)
        # x = inputs
        for j, linear in enumerate(self.linears[:-1]):
            x = (
                self.activation[j](linear(x))
                if isinstance(self.activation, list)
                else self.activation(linear(x))
            )
        x = self.linears[-1](x)
        return x
def reset_logging():
    logging.shutdown()
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
        
class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass
    
def RBF(x1, x2, params):
    """Radial Basis Function kernel."""
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)  # Calculate squared difference sum
    return output_scale * np.exp(-0.5 * r2)  # Return RBF kernel result

def generate_random_gaussian_field(m, length_scale=0.2):
    """
    Generate random Gaussian field using Gaussian process.
    
    Args:
        m: Number of output points
        length_scale: Length scale parameter for the RBF kernel
    
    Returns:
        u_fn: Interpolation function
        u: Sampled values at m points
    """
    N = 1024
    jitter = 1e-10
    gp_params = (1.0, length_scale)

    X = np.linspace(0, 1, N)[:, None]
    K = RBF(X, X, gp_params)  # RBF is radial basis function (squared exponential kernel)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    key_train = np.random.randn(N)
    gp_sample = np.dot(L, key_train)
    
    u_fn = lambda x: np.interp(x, X.flatten(), gp_sample)
    x = np.linspace(0, 1, m)
    u = u_fn(x)
    
    return u_fn, u


def sample_1D_Operator_data(v, u, x, sample_num):
    num = u.shape[0]
    num_sensors = u.shape[1]
    output_size = 1
    branch_input = np.repeat(v, sample_num, axis=0)
    trunk_input = np.zeros((0, 1))
    output = np.zeros((0, output_size))
    indices = []
    for i in range(num):
        indice = np.random.choice(num_sensors, sample_num, replace=False).reshape(-1, 1)
        indices.append(indice)
        trunk_input_new = np.take_along_axis(np.expand_dims(x, axis=1), indice, axis=0)
        trunk_input = np.concatenate((trunk_input, trunk_input_new), axis=0)
        gathered_output = np.take_along_axis(u[i].reshape(-1, 1), indice, axis=0)
        output = np.concatenate((output, gathered_output), axis=0)
    return branch_input.astype(np.float32), trunk_input.astype(np.float32), output.astype(np.float32)

# def generate_Inverse_Operator_data(num_train, num_test, num_points, length_scale=0.2, num_cal=1000):
#     data_path = f'data/Inverse_Operator_data/Inverse_Operator_data_{num_cal}_1.npz'
#     os.makedirs(os.path.dirname(data_path), exist_ok=True)
#     x_cal = np.linspace(0, 1, num_cal)
#     if os.path.exists(data_path):
#         data = np.load(data_path, allow_pickle=True)
#         u_cals = list(data['u_cals']) if 'u_cals' in data else []
#         v_cals = list(data['v_cals']) if 'v_cals' in data else []
#     else:
#         u_cals = []
#         v_cals = []
#     if len(u_cals) < num_train + num_test:
#         for _ in tqdm(range(num_train + num_test - len(u_cals)), desc="Generating Inverse Data"):
#             v_fn, v_cal_new = generate_random_gaussian_field(num_cal, length_scale)
#             def ode_system(x, u):
#                 return v_fn(x)
#             sol = solve_ivp(ode_system, [x_cal[0], x_cal[-1]], [0], t_eval=x_cal, method='RK45')
#             u_cal_new = sol.y[0]
#             u_cals.append(u_cal_new)
#             v_cals.append(v_cal_new)
#         np.savez(data_path, u_cals=u_cals, v_cals=v_cals)
#         data = np.load(data_path, allow_pickle=True)
#     u_cals = data['u_cals']
#     v_cals = data['v_cals']
#     x = np.linspace(0, 1, num_points)
#     us, vs = [], []
#     for u_cal, v_cal in zip(u_cals, v_cals):
#         u = np.interp(x, x_cal, u_cal)
#         v = np.interp(x, x_cal, v_cal)
#         us.append(u)
#         vs.append(v)
#     train_index = np.random.choice(num_train + num_test, num_train, replace=False)
#     test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])
#     return np.array(vs)[train_index].astype(np.float32), np.array(us)[train_index].astype(np.float32), np.array(vs)[test_index].astype(np.float32), np.array(us)[test_index].astype(np.float32)

# def generate_Homogeneous_Operator_data(num_train, num_test, num_points, length_scale=0.2, num_cal=1000):
#     data_path = f'data/Homogeneous_Operator_data/Homogeneous_Operator_data_{num_cal}_1.npz'
#     os.makedirs(os.path.dirname(data_path), exist_ok=True)
#     x_cal = np.linspace(0, 1, num_cal)
#     if os.path.exists(data_path):
#         data = np.load(data_path, allow_pickle=True)
#         u_cals = list(data['u_cals']) if 'u_cals' in data else []
#         v_cals = list(data['v_cals']) if 'v_cals' in data else []
#     else:
#         u_cals = []
#         v_cals = []
#     if len(u_cals) < num_train + num_test:
#         for _ in tqdm(range(num_train + num_test - len(u_cals)), desc="Generating Inverse Data"):
#             v_fn, v_cal_new = generate_random_gaussian_field(num_cal, length_scale)
#             def ode_system(x, u):
#                 v = v_fn(x)
#                 return u + v
#             sol = solve_ivp(ode_system, [x_cal[0], x_cal[-1]], [0], t_eval=x_cal, method='RK45')
#             u_cal_new = sol.y[0]
#             u_cals.append(u_cal_new)
#             v_cals.append(v_cal_new)
#         np.savez(data_path, u_cals=u_cals, v_cals=v_cals)
#         data = np.load(data_path, allow_pickle=True)
#     u_cals = data['u_cals']
#     v_cals = data['v_cals']
#     x = np.linspace(0, 1, num_points)
#     us, vs = [], []
#     for u_cal, v_cal in zip(u_cals, v_cals):
#         u = np.interp(x, x_cal, u_cal)
#         v = np.interp(x, x_cal, v_cal)
#         us.append(u)
#         vs.append(v)
#     train_index = np.random.choice(num_train + num_test, num_train, replace=False)
#     test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])
#     return np.array(vs)[train_index].astype(np.float32), np.array(us)[train_index].astype(np.float32), np.array(vs)[test_index].astype(np.float32), np.array(us)[test_index].astype(np.float32)

# def generate_Nonlinear_Operator_data(num_train, num_test, num_points, length_scale=0.2, num_cal=1000):
#     data_path = f'data/Nonlinear_Operator_data/Nonlinear_Operator_data_{num_cal}_1.npz'
#     os.makedirs(os.path.dirname(data_path), exist_ok=True)
#     x_cal = np.linspace(0, 1, num_cal)
#     if os.path.exists(data_path):
#         data = np.load(data_path, allow_pickle=True)
#         u_cals = list(data['u_cals']) if 'u_cals' in data else []
#         v_cals = list(data['v_cals']) if 'v_cals' in data else []
#     else:
#         u_cals = []
#         v_cals = []
#     if len(u_cals) < num_train + num_test:
#         for _ in tqdm(range(num_train + num_test - len(u_cals)), desc="Generating Inverse Data"):
#             v_fn, v_cal_new = generate_random_gaussian_field(num_cal, length_scale)
#             def ode_system(x, u):
#                 v = v_fn(x)
#                 return u - v ** 2
#             sol = solve_ivp(ode_system, [x_cal[0], x_cal[-1]], [0], t_eval=x_cal, method='RK45')
#             u_cal_new = sol.y[0]
#             u_cals.append(u_cal_new)
#             v_cals.append(v_cal_new)
#         np.savez(data_path, u_cals=u_cals, v_cals=v_cals)
#         data = np.load(data_path, allow_pickle=True)
#     u_cals = data['u_cals']
#     v_cals = data['v_cals']
#     x = np.linspace(0, 1, num_points)
#     us, vs = [], []
#     for u_cal, v_cal in zip(u_cals, v_cals):
#         u = np.interp(x, x_cal, u_cal)
#         v = np.interp(x, x_cal, v_cal)
#         us.append(u)
#         vs.append(v)
#     train_index = np.random.choice(num_train + num_test, num_train, replace=False)
#     test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])
#     return np.array(vs)[train_index].astype(np.float32), np.array(us)[train_index].astype(np.float32), np.array(vs)[test_index].astype(np.float32), np.array(us)[test_index].astype(np.float32)

def ODE_encode(generate_data, num_train, num_test, num_sensors, train_sample_num, test_sample_num):
    train_v, train_u, test_v, test_u, _ = generate_data(num_train, num_test)
    x = np.linspace(0, 1, num_sensors).astype(np.float32)
    train_branch_input, train_trunk_input, train_output = sample_1D_Operator_data(train_v, train_u, x, train_sample_num)
    test_branch_input, test_trunk_input, test_output = sample_1D_Operator_data(test_v, test_u, x, test_sample_num)
    return train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output

def ODE_fncode(generate_data, num_train, num_test, num_sensors, train_sample_num, test_sample_num):
    train_v, train_u, test_v, test_u = generate_data(num_train, num_test, num_sensors)
    x = np.linspace(0, 1, num_sensors).astype(np.float32)
    def sample_1D_Operator_fndata(v, u, x, sample_num):
        num = u.shape[0]
        num_sensors = u.shape[1]
        indices = np.zeros((0, sample_num))
        output = np.zeros((0, sample_num))
        x = x.reshape(1, -1)
        x = np.repeat(x, num, axis=0)
        x = np.expand_dims(x, axis=2)
        v = np.expand_dims(v, axis=2)
        input = np.concatenate((v, x), axis=2)
        for i in range(num):
            indice = np.array(range(0, num_sensors, num_sensors//sample_num)).reshape(1, -1)
            indices = np.concatenate((indices, indice), axis=0)
            # output_new = np.take_along_axis(u[i].reshape(1, -1), indice, axis=1)
            output_new = u[i].reshape(1, -1)
            output = np.concatenate((output, output_new), axis=0)
        output = np.expand_dims(output, axis=2)
        return input.astype(np.float32), indices.astype(np.int64), output.astype(np.float32)
    train_input, train_indices, train_output = sample_1D_Operator_fndata(train_v, train_u, x, train_sample_num)
    test_input, test_indices, test_output = sample_1D_Operator_fndata(test_v, test_u, x, test_sample_num)
    return train_input, train_indices, train_output, test_input, test_indices, test_output


def sample_2D_Operator_data(u0, u, x, t, sample_num, num_points):
    num = u.shape[0]
    output_size = 1
    branch_input = np.repeat(u0, sample_num, axis=0)
    trunk_input = np.zeros((0, 2))
    output = np.zeros((0, output_size))
    for i in range(num):
        indices = np.random.choice(num_points**2, sample_num, replace=False).reshape(-1, 1)
        x_repeat = np.repeat(x, num_points).reshape(-1, 1)
        t_tile = np.tile(t, num_points).reshape(-1, 1)
        trunk_input_new = np.take_along_axis(np.concatenate((x_repeat, t_tile), axis=1), indices, axis=0)
        trunk_input = np.concatenate((trunk_input, trunk_input_new), axis=0)
        gathered_output = np.take_along_axis(u[i].reshape(-1, 1), indices, axis=0)
        output = np.concatenate((output, gathered_output), axis=0)
    return branch_input.astype(np.float32), trunk_input.astype(np.float32), output.astype(np.float32)

def PDE_encode(generate_data, num_train, num_test, num_points, train_sample_num, test_sample_num):
    train_u0, train_u, test_u0, test_u = generate_data(num_train, num_test)
    x = np.linspace(0, 1, num_points).astype(np.float32)
    t = np.linspace(0, 1, num_points).astype(np.float32)
    train_branch_input, train_trunk_input, train_output = sample_2D_Operator_data(train_u0, train_u, x, t, train_sample_num, num_points)
    test_branch_input, test_trunk_input, test_output = sample_2D_Operator_data(test_u0, test_u, x, t, test_sample_num, num_points)
    return train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output

def generate_RDiffusion_Operator_data(num_train, num_test, num_points, length_scale=0.2, periodicity=1.0, k=0.01, alpha=0.01, num_cal=100):
    num_cal = 100
    data_path = f'data/RDiffusion_Operator_data/RDiffusion_Operator_data_{num_cal}_1.npz'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)
    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        u_cals = list(data['u_cals']) if 'u0_cals' in data else []
        u0_cals = list(data['u0_cals']) if 'u0_cals' in data else []
    else:
        u_cals = []
        u0_cals = []
    u_cals = data['u_cals']
    u0_cals = data['u0_cals']
    x = np.linspace(0, 1, num_points)
    t = np.linspace(0, 1, num_points)
    us, u0s = [], []
    for u_cal, u0_cal in zip(u_cals, u0_cals):
        u = u_cal[::num_cal//num_points, ::num_cal//num_points]
        u0 = u0_cal[::num_cal//num_points]
        us.append(u)
        u0s.append(u0)
    train_index = np.random.choice(num_train + num_test, num_train, replace=False)
    test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])
    return np.array(u0s)[train_index].astype(np.float32), np.array(us)[train_index].astype(np.float32), np.array(u0s)[test_index].astype(np.float32), np.array(us)[test_index].astype(np.float32)

def generate_PDE_Operator_data(num_train, num_test, operator):
    num_cal = 100 if operator != 'Darcy' else 25
    num_points = num_cal
    data_path = f'data/{operator}_Operator_data/{operator}_Operator_data_{num_cal}_1.npz'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)
    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        u_cals = list(data['u_cals']) if 'u0_cals' in data else []
        u0_cals = list(data['u0_cals']) if 'u0_cals' in data else []
    else:
        u_cals = []
        u0_cals = []
    u_cals = data['u_cals']
    u0_cals = data['u0_cals']
    x = np.linspace(0, 1, num_points)
    t = np.linspace(0, 1, num_points)
    us, u0s = [], []
    for u_cal, u0_cal in zip(u_cals, u0_cals):
        us.append(u_cal)
        u0s.append(u0_cal)
    train_index = np.random.choice(num_train + num_test, num_train, replace=False)
    test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])
    return np.array(u0s)[train_index].astype(np.float32), np.array(us)[train_index].astype(np.float32), np.array(u0s)[test_index].astype(np.float32), np.array(us)[test_index].astype(np.float32)


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
        'description': 'Nonlinear operator problem: du/dx = -u0(x)² + u',
        'ode_func': lambda u0_fn: lambda x, u: u - u0_fn(x)**2
    }
}

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

# 数据生成配置常量
DATA_GENERATION_CONFIG = {
    'save_interval': 100,      # Save data every 100 samples
    'backup_interval': 500,    # Create a backup every 500 samples
    'progress_update': 10,     # Update progress every 10 samples
}

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
    
    return  np.array(u0s)[train_index].astype(np.float32), np.array(us)[train_index].astype(np.float32), np.array(u0s)[test_index].astype(np.float32), np.array(us)[test_index].astype(np.float32), x.astype(np.float32)


# def generate_ODE_Operator_data(num_train, num_test, operator):
#     num_cal = 1000
#     num_points = 100
#     data_path = f'data/{operator}_Operator_data/{operator}_Operator_data_{num_cal}_1.npz'
#     os.makedirs(os.path.dirname(data_path), exist_ok=True)
#     x_cal = np.linspace(0, 1, num_cal)
#     if os.path.exists(data_path):
#         data = np.load(data_path, allow_pickle=True)
#         u_cals = list(data['u_cals']) if 'u0_cals' in data else []
#         u0_cals = list(data['u0_cals']) if 'u0_cals' in data else []
#     else:
#         u_cals = []
#         u0_cals = []
#     u_cals = [u_cal[::num_cal//num_points] for u_cal in data['u_cals']]
#     u0_cals = [u0_cal[::num_cal//num_points] for u0_cal in data['u0_cals']]
#     x = np.linspace(0, 1, num_points)
#     us, u0s = [], []
#     for u_cal, u0_cal in zip(u_cals, u0_cals):
#         us.append(u_cal)
#         u0s.append(u0_cal)
#     train_index = np.random.choice(num_train + num_test, num_train, replace=False)
#     test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])
#     return np.array(u0s)[train_index].astype(np.float32), np.array(us)[train_index].astype(np.float32), np.array(u0s)[test_index].astype(np.float32), np.array(us)[test_index].astype(np.float32)

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d, self).__init__()

        """
        1D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  #Number of Fourier modes to multiply, at most floor(N/2) + 1

        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1)//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=x.size(-1))
        return x

class FNO1d(nn.Module):
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the initial condition and location (a(x), x)
        input shape: (batchsize, x=s, c=2)
        output: the solution of a later timestep
        output shape: (batchsize, x=s, c=1)
        """

        self.modes1 = modes
        self.width = width
        self.fc0 = nn.Linear(2, self.width) # input channel is 2: (a(x), x)
        self.regularizer = None
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)


        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, input):
        x = input

        x = self.fc0(x)
        x = x.permute(0, 2, 1)

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.relu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # indice = torch.arange(0, x.size(1), 10).reshape(1, -1, 1).repeat(x.size(0), 1, 1)
        # x = torch.gather(x, 1, indice)
        return x

class Double(Data):
    """Dataset with each data point as a triple.

    The couple of the first two elements are the input, and the third element is the
    output. This dataset can be used with the network ``DeepONet`` for operator
    learning.

    Args:
        X_train: A tuple of two NumPy arrays.
        y_train: A NumPy array.

    References:
        `L. Lu, P. Jin, G. Pang, Z. Zhang, & G. E. Karniadakis. Learning nonlinear
        operators via DeepONet based on the universal approximation theorem of
        operators. Nature Machine Intelligence, 3, 218--229, 2021
        <https://doi.org/10.1038/s42256-021-00302-5>`_.
    """

    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test

        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        if batch_size is None:
            return self.train_x, self.train_y
        indices = self.train_sampler.get_next(batch_size)
        return (
            self.train_x[indices], self.train_y[indices],
        )

    def test(self):
        return self.test_x, self.test_y

class DeepONet_nobias(nn.Module):
    """Deep operator network.

    `Lu et al. Learning nonlinear operators via DeepONet based on the universal
    approximation theorem of operators. Nat Mach Intell, 2021.
    <https://doi.org/10.1038/s42256-021-00302-5>`_

    Args:
        layer_sizes_branch: A list of integers as the width of a fully connected network,
            or `(dim, f)` where `dim` is the input dimension and `f` is a network
            function. The width of the last layer in the branch and trunk net
            should be the same for all strategies except "split_branch" and "split_trunk".
        layer_sizes_trunk (list): A list of integers as the width of a fully connected
            network.
        activation: If `activation` is a ``string``, then the same activation is used in
            both trunk and branch nets. If `activation` is a ``dict``, then the trunk
            net uses the activation `activation["trunk"]`, and the branch net uses
            `activation["branch"]`.
        num_outputs (integer): Number of outputs. In case of multiple outputs, i.e., `num_outputs` > 1,
            `multi_output_strategy` below should be set.
        multi_output_strategy (str or None): ``None``, "independent", "split_both", "split_branch" or
            "split_trunk". It makes sense to set in case of multiple outputs.

            - None
            Classical implementation of DeepONet with a single output.
            Cannot be used with `num_outputs` > 1.

            - independent
            Use `num_outputs` independent DeepONets, and each DeepONet outputs only
            one function.

            - split_both
            Split the outputs of both the branch net and the trunk net into `num_outputs`
            groups, and then the kth group outputs the kth solution.

            - split_branch
            Split the branch net and share the trunk net. The width of the last layer
            in the branch net should be equal to the one in the trunk net multiplied
            by the number of outputs.

            - split_trunk
            Split the trunk net and share the branch net. The width of the last layer
            in the trunk net should be equal to the one in the branch net multiplied
            by the number of outputs.
    """

    def __init__(
        self,
        layer_sizes_branch,
        layer_sizes_trunk,
        activation,
        kernel_initializer,
        num_outputs=1,
        multi_output_strategy=None,
        regularization=None,
    ):
        super().__init__()
        if isinstance(activation, dict):
            self.activation_branch = activation["branch"]
            self.activation_trunk = activations.get(activation["trunk"])
        else:
            self.activation_branch = self.activation_trunk = activations.get(activation)
        self.kernel_initializer = kernel_initializer
        self.regularizer = regularization
        self._auxiliary_vars = None
        self._input_transform = None
        self._output_transform = None
        self.num_outputs = num_outputs
        if self.num_outputs == 1:
            if multi_output_strategy is not None:
                raise ValueError(
                    "num_outputs is set to 1, but multi_output_strategy is not None."
                )
        elif multi_output_strategy is None:
            multi_output_strategy = "independent"
            print(
                f"Warning: There are {num_outputs} outputs, but no multi_output_strategy selected. "
                'Use "independent" as the multi_output_strategy.'
            )
        self.multi_output_strategy = {
            None: SingleOutputStrategy,
            "independent": IndependentStrategy,
            "split_both": SplitBothStrategy,
            "split_branch": SplitBranchStrategy,
            "split_trunk": SplitTrunkStrategy,
        }[multi_output_strategy](self)

        self.branch, self.trunk = self.multi_output_strategy.build(
            layer_sizes_branch, layer_sizes_trunk
        )
        if isinstance(self.branch, list):
            self.branch = torch.nn.ModuleList(self.branch)
        if isinstance(self.trunk, list):
            self.trunk = torch.nn.ModuleList(self.trunk)
        # self.b = torch.nn.ParameterList(
        #     [torch.nn.Parameter(torch.tensor(0.0)) for _ in range(self.num_outputs)]
        # )
        self.regularizer = regularization

    def build_branch_net(self, layer_sizes_branch):
        # User-defined network
        if callable(layer_sizes_branch[1]):
            return layer_sizes_branch[1]
        # Fully connected network
        return FNN(layer_sizes_branch, self.activation_branch, self.kernel_initializer)

    def build_trunk_net(self, layer_sizes_trunk):
        return FNN(layer_sizes_trunk, self.activation_trunk, self.kernel_initializer)

    def merge_branch_trunk(self, x_func, x_loc, index):
        y = torch.einsum("bi,bi->b", x_func, x_loc)
        y = torch.unsqueeze(y, dim=1)
        # y += self.b[index]
        return y

    @staticmethod
    def concatenate_outputs(ys):
        return torch.concat(ys, dim=1)

    @property
    def auxiliary_vars(self):
        """Tensors: Any additional variables needed."""
        return self._auxiliary_vars

    @auxiliary_vars.setter
    def auxiliary_vars(self, value):
        self._auxiliary_vars = value

    def apply_feature_transform(self, transform):
        """Compute the features by appling a transform to the network inputs, i.e.,
        features = transform(inputs). Then, outputs = network(features).
        """
        self._input_transform = transform

    def apply_output_transform(self, transform):
        """Apply a transform to the network outputs, i.e.,
        outputs = transform(inputs, outputs).
        """
        self._output_transform = transform

    def num_trainable_parameters(self):
        """Evaluate the number of trainable parameters for the NN."""
        return sum(v.numel() for v in self.parameters() if v.requires_grad)

    def forward(self, inputs):
        x_func = inputs[0]
        x_loc = inputs[1]
        # Trunk net input transform
        if self._input_transform is not None:
            x_loc = self._input_transform(x_loc)
        x = self.multi_output_strategy.call(x_func, x_loc)
        if self._output_transform is not None:
            x = self._output_transform(inputs, x)
        return x
    # def parameters(self, recurse: bool = True):
    #     """
    #     返回模型的可训练参数。
    #     该方法直接继承自 nn.Module，可不重写，除非你有特殊需求。
    #     """
    #     return super().parameters(recurse)