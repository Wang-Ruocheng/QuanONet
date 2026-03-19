"""
Data generation functions for DeepXDE-based classical operator learning.
This module provides MindSpore-free versions of data generation functions.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
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
    'Antideriv': {
        'description': 'Antideriv operator problem: du/dx = u0(x)',
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

def generate_Antideriv_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=1000):
    return generate_ODE_Operator_data('Antideriv', num_train, num_test, num_points, num_points_0, length_scale, num_cal)


def generate_Homogeneous_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=1000):
    return generate_ODE_Operator_data('Homogeneous', num_train, num_test, num_points, num_points_0, length_scale, num_cal)


def generate_Nonlinear_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=1000):
    return generate_ODE_Operator_data('Nonlinear', num_train, num_test, num_points, num_points_0, length_scale, num_cal)

def generate_RDiffusion_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=100):
    return generate_PDE_Operator_data('RDiffusion', num_train, num_test, num_points, num_points_0, num_cal, length_scale)


def generate_Advection_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=100):
    return generate_PDE_Operator_data('Advection', num_train, num_test, num_points, num_points_0, num_cal, length_scale)


def generate_Darcy_Operator_data(num_train, num_test, num_points, num_points_0, length_scale=0.2, num_cal=100):
    return generate_PDE_Operator_data('Darcy', num_train, num_test, num_points, num_points_0, num_cal, length_scale)


# PDE Systems (simplified versions)
def solve_darcy_pde(num_cal, length_scale=1.0, K=0.1, f=-1.0, u0_cal=None):
    """Solve Darcy flow PDE: -∇(K∇u)=f"""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    nx, ny = num_cal, num_cal
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    if u0_cal is None:
        # Generate initial condition and source term
        _, u0_cal = generate_random_gaussian_field(4*num_cal, length_scale=length_scale, if_period=True)
    def boundary_from_1d_func(u0):
        edge = len(u0)//4
        left = u0[:edge]
        right = u0[2*edge:3*edge][::-1]
        bottom = u0[3*edge:][::-1]
        top = u0[edge:2*edge]
        return left, right, bottom, top
    left, right, bottom, top = boundary_from_1d_func(u0_cal)

    # Construct sparse matrix and right-hand side
    N = nx * ny
    main = np.ones(N) * (-2/dx**2 - 2/dy**2)
    offx = np.ones(N) / dx**2
    offy = np.ones(N) / dy**2
    rhs = -np.ones(N) * f / K
    for i in range(nx):
        for j in range(ny):
            idx = i*ny + j
            if j == 0:
                main[idx]=1; offx[idx]=0; offy[idx]=0; rhs[idx]=bottom[i]
            elif j == ny-1:
                main[idx]=1; offx[idx]=0; offy[idx]=0; rhs[idx]=top[i]
            elif i == 0:
                main[idx]=1; offx[idx]=0; offy[idx]=0; rhs[idx]=left[j]
            elif i == nx-1:
                main[idx]=1; offx[idx]=0; offy[idx]=0; rhs[idx]=right[j]
    A = diags([main, offx[:-1], offx[1:], offx[-1], offx[:1], offy[:(N-ny)], offy[ny:], offy[(N-ny):], offy[:ny]], [0, 1, -1, -N+1, N-1, ny, -ny, -N+ny, N-ny], shape=(N, N))
    u_cal = spsolve(A.tocsr(), rhs).reshape((nx, ny))

    return u_cal, u0_cal

def solve_advection_pde(num_cal, length_scale=0.2, c=1.0, u0_cal=None):
    """
    Solve advection equation: ∂u/∂t + c∇u = 0
    """
    x_cal = np.linspace(0, 1, num_cal)
    dx = x_cal[1] - x_cal[0]
    
    t_final = 1.0
    dt = 0.8 * dx / abs(c) if c != 0 else 0.01
    num_t = int(t_final / dt)
    
    if u0_cal is None:
        # _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale, if_period=True)
        _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale, if_period=False)

    u_cal = np.zeros((num_cal, num_t))
    u_cal[:, 0] = u0_cal

    # Use upwind finite difference scheme
    for j in range(1, num_t):
        u_prev = u_cal[:, j-1].copy()
        u_new = np.zeros_like(u_prev)
        
        if c > 0:
            # Positive advection, use backward difference
            for i in range(num_cal):
                if i == 0:
                    # Periodic boundary condition
                    u_new[i] = u_prev[i] - c * dt / dx * (u_prev[i] - u_prev[-1])
                else:
                    u_new[i] = u_prev[i] - c * dt / dx * (u_prev[i] - u_prev[i-1])
        elif c < 0:
            # Negative advection, use forward difference
            for i in range(num_cal):
                if i == num_cal - 1:
                    # Periodic boundary condition
                    u_new[i] = u_prev[i] - c * dt / dx * (u_prev[0] - u_prev[i])
                else:
                    u_new[i] = u_prev[i] - c * dt / dx * (u_prev[i+1] - u_prev[i])
        else:
            u_new = u_prev
        
        u_cal[:, j] = u_new
    
    if num_t > num_cal:
        time_indices = np.linspace(0, num_t-1, num_cal, dtype=int)
        u_cal_sampled = u_cal[:, time_indices]
    else:
        from scipy.interpolate import interp1d
        t_old = np.linspace(0, 1, num_t)
        t_new = np.linspace(0, 1, num_cal)
        u_cal_sampled = np.zeros((num_cal, num_cal))
        for i in range(num_cal):
            interp_func = interp1d(t_old, u_cal[i, :], kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            u_cal_sampled[i, :] = interp_func(t_new)
    
    return u_cal_sampled, u0_cal

def solve_rdiffusion_pde(num_cal, length_scale, D=0.01, k=0.01, u0_cal=None):
    """Solve rdiffusion PDE ∂u/∂t = α∇²u + k*u² + u0(x)"""
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)
    
    # Calculate time step parameters
    dx = x_cal[1] - x_cal[0]
    dt = min(dx**2 / (2 * D), t_cal[1] - t_cal[0])  # Ensure stability
    num_cal_t = int(1//dt)
    
    def rdiffusion_step(u, dx, dt, D, k, u0):
        u_new = np.zeros_like(u)
        for i in range(1, len(u) - 1):
            u_new[i] = u[i] + dt * (D * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2) + k * (u[i]**2) + u0[i])
        u_new[0] = u_new[-1] = 0  # Boundary conditions
        return u_new
    
    if u0_cal is None:
        # Generate initial condition and source term
        _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale)
    
    # Time evolution
    u_cal = np.zeros((num_cal, num_cal_t))
    for i in range(1, num_cal_t):
        u_cal[:, i] = rdiffusion_step(u_cal[:, i-1], dx, dt, D, k, u0_cal)
    
    # Sample the data to match num_cal
    u_cal_sampled = u_cal[:, ::max(1, num_cal_t//num_cal)][:, :num_cal]
    
    return u_cal_sampled, u0_cal


def generate_PDE_Operator_data(operator_type, num_train, num_test,
                               num_points,      # Controls output u(x,t) resolution (Trunk/Output)
                               num_points_0,    # Controls input u0(x) resolution (Branch Input)
                               length_scale=0.2,
                               num_cal=100):
    """
    Generate data for PDE operator problems with decoupled input/output resolutions.
    Structured identically to generate_ODE_Operator_data.
    """
    operator_name = operator_type

    # 1. Data path and loading strategy
    data_path = f'data/{operator_name}_Operator_data/{operator_name}_Operator_data_{num_cal}_1.npz'
    os.makedirs(os.path.dirname(data_path), exist_ok=True)

    # Try to load existing data
    if os.path.exists(data_path):
        try:
            d = np.load(data_path, allow_pickle=True)
            u_cals = list(d['u_cals']) if 'u_cals' in d else []
            u0_cals = list(d['u0_cals']) if 'u0_cals' in d else []
        except Exception as e:
            print(f"Warning: Failed to load cached data {data_path}: {e}")
            u_cals, u0_cals = [], []
    else:
        u_cals, u0_cals = [], []

    # 2. Generate missing data
    if len(u_cals) < num_train + num_test:
        print(f"Generating {operator_name} Data (Calculation Resolution: {num_cal})")
        total_needed = num_train + num_test - len(u_cals)
        save_interval = 100

        for i in tqdm(range(total_needed), desc=f"Generating {operator_name} Data"):
            try:
                if operator_name == 'Darcy':
                    u_cal_new, u0_cal_new = solve_darcy_pde(num_cal, length_scale=length_scale)
                elif operator_name == 'Advection':
                    u_cal_new, u0_cal_new = solve_advection_pde(num_cal, length_scale=length_scale)
                elif operator_name == 'RDiffusion':
                    u_cal_new, u0_cal_new = solve_rdiffusion_pde(num_cal, length_scale=length_scale)
                else:
                    raise ValueError(f"Unknown PDE operator: {operator_name}")

                if np.isnan(u_cal_new).any():
                    print("Warning: NaN detected in solver output, skipping sample.")
                    continue

                u_cals.append(u_cal_new)
                u0_cals.append(u0_cal_new)
            except Exception as e:
                print(f"Error solving PDE: {e}")
                continue

            # Save data periodically
            if (i + 1) % save_interval == 0 or i == total_needed - 1:
                np.savez(data_path, u_cals=u_cals, u0_cals=u0_cals)

    # 3. Dual-resolution interpolation
    print(f"Interpolating data:")
    print(f"  - Input u0: from calculation resolution to {num_points_0}")
    print(f"  - Output u: from calculation resolution to {num_points}x{num_points}")

    # Target grid - Output / Trunk (based on num_points)
    x_target = np.linspace(0, 1, num_points)
    t_target = np.linspace(0, 1, num_points)

    # Target grid - Input / Branch (based on num_points_0)
    x_target_0 = np.linspace(0, 1, num_points_0)

    us, u0s = [], []

    for u_cal, u0_cal in zip(u_cals, u0_cals):
        # Interpolate u0 (input) to target resolution
        if u0_cal.ndim == 1:
            x_source_0 = np.linspace(0, 1, len(u0_cal))
            u0_new = np.interp(x_target_0, x_source_0, u0_cal)
        else:
            u0_new = u0_cal

        # Interpolate u (output) to target resolution
        if u_cal.ndim == 2:
            curr_x_dim, curr_t_dim = u_cal.shape
            x_src_curr = np.linspace(0, 1, curr_x_dim)
            t_src_curr = np.linspace(0, 1, curr_t_dim)

            interp_func = RegularGridInterpolator((x_src_curr, t_src_curr), u_cal,
                                                  method='linear', bounds_error=False, fill_value=None)
            xg, tg = np.meshgrid(x_target, t_target, indexing='ij')
            u_new = interp_func((xg, tg))
        else:
            u_new = np.interp(x_target, np.linspace(0, 1, len(u_cal)), u_cal)

        us.append(u_new)
        u0s.append(u0_new)

    # Randomly split into training and testing sets (Same logic as ODE)
    train_index = np.random.choice(num_train + num_test, num_train, replace=False)
    test_index = np.array([i for i in range(num_train + num_test) if i not in train_index])

    return (np.array(u0s)[train_index].astype(np.float32), 
            np.array(us)[train_index].astype(np.float32), 
            np.array(u0s)[test_index].astype(np.float32), 
            np.array(us)[test_index].astype(np.float32), 
            x_target.astype(np.float32), 
            t_target.astype(np.float32))