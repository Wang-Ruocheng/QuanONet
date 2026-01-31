"""
Data generation functions for DeepXDE-based classical operator learning.
This module provides MindSpore-free versions of data generation functions.
"""

import numpy as np
import os
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import json


# RBF kernel for Gaussian process
def RBF(X1, X2, gp_params):
    """
    Radial Basis Function (squared exponential) kernel.

    Args:
        X1: First set of points (N1, D)
        X2: Second set of points (N2, D)
        gp_params: Tuple of (output_scale, length_scale)

    Returns:
        Kernel matrix (N1, N2)
    """
    output_scale, length_scale = gp_params
    diffs = X1[:, None, :] - X2[None, :, :]  # (N1, N2, D)
    r2 = np.sum(diffs**2, axis=2)  # (N1, N2)
    return output_scale * np.exp(-0.5 * r2 / (length_scale**2))


def generate_random_gaussian_field(m, length_scale=0.2, if_period=False):
    """
    Generate random Gaussian field using Gaussian process.

    Args:
        m: Number of output points
        length_scale: Length scale parameter for the RBF kernel
        if_period: Whether to make the field periodic

    Returns:
        u_fn: Interpolation function
        u: Sampled values at m points
    """
    N = 1024
    jitter = 1e-10
    gp_params = (1.0, length_scale)

    X = np.linspace(0, 1, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    key_train = np.random.randn(N)
    gp_sample = np.dot(L, key_train)

    if if_period:
        # Make periodic by averaging with reversed version
        gp_sample = (gp_sample + gp_sample[::-1]) / 2

    u_fn = lambda x: np.interp(x, X.flatten(), gp_sample)
    x = np.linspace(0, 1, m)
    u = u_fn(x)

    return u_fn, u


# ODE Systems definitions
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
        'description': 'Nonlinear operator problem: du/dx = u0(x) - u^3',
        'ode_func': lambda u0_fn: lambda x, u: -u**3 + u0_fn(x)
    },
    'Identity': {
        'description': 'Identity operator problem: u = u0(x)',
        'ode_func': None
    }
}


def generate_ODE_Operator_data(operator_type, num_train, num_test,
                               num_points,      # Controls output u(x) resolution (Trunk/Output)
                               num_points_0,    # Controls input u0(x) resolution (Branch Input)
                               length_scale=0.2,
                               num_cal=1000):
    """
    Generate data for ODE operator problems with decoupled input/output resolutions.
    """
    # 1. Determine operator name and function
    if operator_type in ODE_SYSTEMS:
        operator_name = operator_type
        ode_func_generator = ODE_SYSTEMS[operator_type]['ode_func']
        description = ODE_SYSTEMS[operator_type]['description']
    else:
        raise ValueError(f"Unknown operator type: {operator_type}")

    # 2. Data path and loading strategy
    data_path = f'data/{operator_name}_Operator_data/{operator_name}_Operator_data_{num_cal}_1.npz'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Source computation grid
    x_cal = np.linspace(0, 1, num_cal)

    # Try to load existing data
    if os.path.exists(data_path):
        d = np.load(data_path, allow_pickle=True)
        u_cals = list(d['u_cals']) if 'u_cals' in d else []
        u0_cals = list(d['u0_cals']) if 'u0_cals' in d else []
    else:
        u_cals, u0_cals = [], []

    # 3. Generate missing data
    if len(u_cals) < num_train + num_test:
        print(f"Generating {description} (Calculation Resolution: {num_cal})")
        total_needed = num_train + num_test - len(u_cals)

        for i in range(total_needed):
            # Generate random Gaussian field as u0
            u0_fn, u0_cal_new = generate_random_gaussian_field(num_cal, length_scale=length_scale)

            # Solve ODE
            if operator_name == 'Identity':
                u_cal_new = u0_cal_new.copy()
            else:
                try:
                    ode_system = ode_func_generator(u0_fn)
                    # Solve interval [0, 1], initial value u(0)=0
                    sol = solve_ivp(ode_system, [0, 1], [0], t_eval=x_cal, method='RK45')
                    u_cal_new = sol.y[0]
                except Exception as e:
                    print(f"Warning: ODE solving failed: {e}")
                    continue  # Skip failed samples

            u_cals.append(u_cal_new)
            u0_cals.append(u0_cal_new)

        # Save data
        np.savez(data_path, u_cals=u_cals, u0_cals=u0_cals)

    # 4. Dual-resolution interpolation
    print(f"Interpolating data:")
    print(f"  - Input u0: from {num_cal} to {num_points_0}")
    print(f"  - Output u: from {num_cal} to {num_points}")

    # Target grid - Output / Trunk (based on num_points)
    x_target = np.linspace(0, 1, num_points)

    # Target grid - Input / Branch (based on num_points_0)
    x_target_0 = np.linspace(0, 1, num_points_0)

    us, u0s = [], []

    for u_cal, u0_cal in zip(u_cals, u0_cals):
        # Interpolate u (output) to target resolution
        u_interp = interp1d(x_cal, u_cal, kind='linear', bounds_error=False, fill_value='extrapolate')
        u_new = u_interp(x_target)

        # Interpolate u0 (input) to target resolution
        u0_interp = interp1d(x_cal, u0_cal, kind='linear', bounds_error=False, fill_value='extrapolate')
        u0_new = u0_interp(x_target_0)

        us.append(u_new)
        u0s.append(u0_new)

    # Randomly split into training and testing sets
    train_index = np.random.choice(num_train + num_test, num_train, replace=False)
    test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])

    return np.array(u0s)[train_index].astype(np.float32), np.array(us)[train_index].astype(np.float32), np.array(u0s)[test_index].astype(np.float32), np.array(us)[test_index].astype(np.float32), x_target.astype(np.float32)


def generate_RDiffusion_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=100):
    return generate_PDE_Operator_data('RDiffusion', num_train, num_test, num_points, num_points_0, num_cal, length_scale)


def generate_Advection_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=100):
    return generate_PDE_Operator_data('Advection', num_train, num_test, num_points, num_points_0, num_cal, length_scale)


def generate_Darcy_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=100):
    return generate_PDE_Operator_data('Darcy', num_train, num_test, num_points, num_points_0, num_cal, length_scale)


# PDE Systems (simplified versions)
def solve_darcy_pde_simple(num_cal, length_scale=1.0):
    """Simplified Darcy PDE solver for 1D case"""
    x_cal = np.linspace(0, 1, num_cal)

    # Generate random permeability field
    _, K_field = generate_random_gaussian_field(num_cal, length_scale=length_scale)
    K_field = 0.1 + 0.9 * (K_field - K_field.min()) / (K_field.max() - K_field.min())  # Scale to [0.1, 1.0]

    # Simple 1D Darcy: -d/dx(K du/dx) = f
    # For simplicity, assume f = -1, and solve using finite differences
    f = -np.ones(num_cal)
    dx = x_cal[1] - x_cal[0]

    # Simple finite difference solution (approximate)
    u_cal = np.cumsum(f * dx**2 / K_field)  # Very simplified
    u_cal = u_cal - u_cal.mean()  # Center around zero

    # Generate boundary condition representation
    _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale)

    return u_cal, u0_cal


def solve_advection_pde_simple(num_cal, length_scale=0.2):
    """Simplified advection PDE solver"""
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)

    # Generate initial condition
    _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale)

    # Simple advection: du/dt + a du/dx = 0, with a=1
    # Analytical solution: u(x,t) = u0(x-t)
    u_cal = np.zeros((num_cal, num_cal))
    for i, t in enumerate(t_cal):
        u_cal[i] = np.interp(x_cal, x_cal - t, u0_cal, left=u0_cal[0], right=u0_cal[-1])

    return u_cal, u0_cal


def solve_rdiffusion_pde_simple(num_cal, length_scale=0.2):
    """Simplified reaction-diffusion PDE solver"""
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)

    # Generate initial condition
    _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale)

    # Simple reaction-diffusion: du/dt = D d²u/dx² + k u
    # For simplicity, use a basic exponential decay with diffusion
    u_cal = np.zeros((num_cal, num_cal))
    for i, t in enumerate(t_cal):
        # Very simplified solution
        u_cal[i] = u0_cal * np.exp(-0.1 * t) * np.exp(-0.01 * (x_cal - 0.5)**2 / t) if t > 0 else u0_cal

    return u_cal, u0_cal


def generate_PDE_Operator_data(operator_type, num_train, num_test,
                               num_points, num_points_0,
                               num_cal=100,
                               length_scale=0.2):
    """
    Generate data for PDE operator problems.
    """
    operator_name = operator_type

    # Data path
    data_path = f'data/{operator_name}_Operator_data/{operator_name}_Operator_data_{num_cal}_1.npz'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Try to load existing data
    if os.path.exists(data_path):
        d = np.load(data_path, allow_pickle=True)
        u_cals = list(d['u_cals']) if 'u_cals' in d else []
        u0_cals = list(d['u0_cals']) if 'u0_cals' in d else []
    else:
        u_cals, u0_cals = [], []

    # Generate missing data
    if len(u_cals) < num_train + num_test:
        print(f"Generating {operator_name} PDE data (Resolution: {num_cal})")
        total_needed = num_train + num_test - len(u_cals)

        for i in range(total_needed):
            if operator_name == 'Darcy':
                u_cal, u0_cal = solve_darcy_pde_simple(num_cal, length_scale)
            elif operator_name == 'Advection':
                u_cal, u0_cal = solve_advection_pde_simple(num_cal, length_scale)
            elif operator_name == 'RDiffusion':
                u_cal, u0_cal = solve_rdiffusion_pde_simple(num_cal, length_scale)
            else:
                raise ValueError(f"Unknown PDE operator: {operator_name}")

            u_cals.append(u_cal)
            u0_cals.append(u0_cal)

        # Save data
        np.savez(data_path, u_cals=u_cals, u0_cals=u0_cals)

    # For PDEs, we need to handle 2D data properly
    # For simplicity, flatten or take slices
    us, u0s = [], []

    for u_cal, u0_cal in zip(u_cals, u0_cals):
        if u_cal.ndim == 2:  # 2D PDE solution
            # Take the final time step or a representative slice
            u_final = u_cal[-1] if num_points == 1 else u_cal
            if num_points > 1 and u_cal.shape[0] != num_points:
                # Interpolate in time dimension
                t_orig = np.linspace(0, 1, u_cal.shape[0])
                t_target = np.linspace(0, 1, num_points)
                u_interp = np.zeros((num_points, u_cal.shape[1]))
                for j in range(u_cal.shape[1]):
                    u_interp[:, j] = np.interp(t_target, t_orig, u_cal[:, j])
                u_final = u_interp
        else:  # 1D solution
            u_final = u_cal

        # Handle u0 (input)
        if u0_cal.ndim == 1 and num_points_0 != len(u0_cal):
            # Interpolate u0 to target resolution
            x_orig = np.linspace(0, 1, len(u0_cal))
            x_target_0 = np.linspace(0, 1, num_points_0)
            u0_final = np.interp(x_target_0, x_orig, u0_cal)
        else:
            u0_final = u0_cal

        us.append(u_final.flatten() if u_final.ndim > 1 else u_final)
        u0s.append(u0_final)

    # Randomly split into training and testing sets
    train_index = np.random.choice(num_train + num_test, num_train, replace=False)
    test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])

    return np.array(u0s)[train_index].astype(np.float32), np.array(us)[train_index].astype(np.float32), np.array(u0s)[test_index].astype(np.float32), np.array(us)[test_index].astype(np.float32), np.linspace(0, 1, num_points).astype(np.float32)


def generate_Inverse_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=1000):
    return generate_ODE_Operator_data('Inverse', num_train, num_test, num_points, num_points_0, length_scale, num_cal)


def generate_Homogeneous_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=1000):
    return generate_ODE_Operator_data('Homogeneous', num_train, num_test, num_points, num_points_0, length_scale, num_cal)


def generate_Nonlinear_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=1000):
    return generate_ODE_Operator_data('Nonlinear', num_train, num_test, num_points, num_points_0, length_scale, num_cal)