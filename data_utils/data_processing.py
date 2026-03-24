"""
Data processing functions for DeepXDE-based classical operator learning.
This module provides MindSpore-free versions of data processing functions.
"""

import numpy as np
from scipy import interpolate

def ODE_encode(generate_data, num_train, num_test, num_points, num_points_0, train_sample_num, test_sample_num, num_cal=None):
    """
    Encode ODE operator data for DeepONet training.
    """
    # Generate data - call with appropriate parameters
    try:
        # Try calling with operator_type first (for dde_data_generation functions)
        u0_train, u_train, u0_test, u_test, x = generate_data(num_train, num_test, num_points, num_points_0)
    except TypeError:
        # Fall back to original signature with num_cal
        u0_train, u_train, u0_test, u_test, x = generate_data(num_train, num_test, num_points, num_points_0, num_cal=num_cal)

    # For ODEs, trunk input is just spatial coordinates
    x_trunk = x.reshape(-1, 1)  # (num_points, 1)

    # Sample spatial points for training and testing
    train_indices = np.array([np.random.choice(num_points, train_sample_num, replace=False) for _ in range(num_train)])
    test_indices = np.array([np.random.choice(num_points, test_sample_num, replace=False) for _ in range(num_test)])

    # Branch input: u0 values repeated for each sample point
    u_train_flat = u_train.reshape(num_train, -1)
    u_test_flat = u_test.reshape(num_test, -1)

    train_output = u_train_flat[np.arange(num_train)[:, None], train_indices].reshape(-1, 1)
    test_output = u_test_flat[np.arange(num_test)[:, None], test_indices].reshape(-1, 1)

    # Trunk input: spatial coordinates at sampled points
    train_trunk_input = x_trunk[train_indices.flatten()]
    test_trunk_input = x_trunk[test_indices.flatten()]

    # Branch input: u0 values repeated for each sample point
    # Shape: (num_samples * sample_num, num_points)
    train_branch_input = np.repeat(u0_train, train_sample_num, axis=0)  # (num_train * train_sample_num, num_points)
    test_branch_input = np.repeat(u0_test, test_sample_num, axis=0)     # (num_test * test_sample_num, num_points)

    return train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output

def ODE_fncode(generate_data, num_train, num_test, num_points, num_cal=None):
    """
    Specialized data encoding for FNO.
    Vectorized version: strictly assumes full-grid evaluation (no subsampling).
    """
    try:
        train_v, train_u, test_v, test_u, x = generate_data(num_train, num_test, num_points, num_points)
    except TypeError:
        train_v, train_u, test_v, test_u, _ = generate_data(num_train, num_test, num_points, num_points, num_cal=num_cal)
    
    current_dim = train_v.shape[1]
    if current_dim != num_points:
        print(f"FNO Alignment: Interpolating input from {current_dim} to {num_points}")
        x_old = np.linspace(0, 1, current_dim)
        x_new = np.linspace(0, 1, num_points)
        f_train = interpolate.interp1d(x_old, train_v, axis=1, kind='linear')
        train_v = f_train(x_new) 
        f_test = interpolate.interp1d(x_old, test_v, axis=1, kind='linear')
        test_v = f_test(x_new)

    x_grid = np.linspace(0, 1, num_points).astype(np.float32)
    
    x_train_exp = np.tile(x_grid, (num_train, 1))[:, :, np.newaxis]
    x_test_exp  = np.tile(x_grid, (num_test, 1))[:, :, np.newaxis]
    
    train_v_exp = train_v[:, :, np.newaxis]
    test_v_exp  = test_v[:, :, np.newaxis]
    
    train_input = np.concatenate((train_v_exp, x_train_exp), axis=2)
    test_input  = np.concatenate((test_v_exp, x_test_exp), axis=2)
    
    train_output = train_u[:, :, np.newaxis]
    test_output  = test_u[:, :, np.newaxis]

    return train_input.astype(np.float32), None, train_output.astype(np.float32), \
           test_input.astype(np.float32), None, test_output.astype(np.float32)

def PDE_encode(generate_data, num_train, num_test, num_points, num_points_0, train_sample_num, test_sample_num, num_cal=None):
    """
    Encode PDE operator data for DeepONet training.
    Structured identically to ODE_encode.
    """
    # Generate data - call with appropriate parameters
    try:
        # Try calling with operator_type first
        u0_train, u_train, u0_test, u_test, x, t = generate_data(num_train, num_test, num_points, num_points_0)
    except TypeError:
        # Fall back to original signature with num_cal
        u0_train, u_train, u0_test, u_test, x, t = generate_data(num_train, num_test, num_points, num_points_0, num_cal=num_cal)

    # For PDEs, trunk input is spatial-temporal coordinates (X, T)
    x_repeat = np.repeat(x, len(t)).reshape(-1, 1)
    t_tile = np.tile(t, len(x)).reshape(-1, 1)
    grid_coords = np.concatenate((x_repeat, t_tile), axis=1)  # (num_points^2, 2)
    total_points = len(x) * len(t)

    # Sample spatial-temporal points for training and testing
    # Shape: (num_samples, sample_num)
    train_indices = np.array([np.random.choice(total_points, train_sample_num, replace=False) for _ in range(num_train)])
    test_indices = np.array([np.random.choice(total_points, test_sample_num, replace=False) for _ in range(num_test)])

    # Branch input: u0 values repeated for each sample point
    # Shape: (num_samples * sample_num, num_points_0)
    train_branch_input = np.repeat(u0_train, train_sample_num, axis=0)
    test_branch_input = np.repeat(u0_test, test_sample_num, axis=0)

    # Trunk input: spatial-temporal coordinates at sampled points
    # Shape: (num_samples * sample_num, 2)
    train_trunk_input = grid_coords[train_indices.flatten()]
    test_trunk_input = grid_coords[test_indices.flatten()]

    # Output: u values at sampled points
    # Flatten the spatial-temporal dimensions of u to match grid_coords indexing
    u_train_flat = u_train.reshape(num_train, -1)
    u_test_flat = u_test.reshape(num_test, -1)

    train_output = u_train_flat[np.arange(num_train)[:, None], train_indices].reshape(-1, 1)
    test_output = u_test_flat[np.arange(num_test)[:, None], test_indices].reshape(-1, 1)

    return train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output


def PDE_fncode(generate_data, num_train, num_test, num_points, num_cal=None):
    """
    Specialized data encoding for FNO on 2D PDEs.
    Flattens the 2D spatial-temporal grid to match FNO1d expected structure.
    """
    try:
        # Try calling with operator_type first
        train_v, train_u, test_v, test_u, x, t = generate_data(num_train, num_test, num_points, num_points)
    except TypeError:
        # Fall back to original signature with num_cal
        train_v, train_u, test_v, test_u, x, t = generate_data(num_train, num_test, num_points, num_points, num_cal=num_cal)

    batch_train = train_v.shape[0]
    batch_test = test_v.shape[0]

    X, T = np.meshgrid(x, t, indexing='ij') 
    x_flat = X.flatten()
    t_flat = T.flatten() 
    total_points = num_points * num_points

    train_v_2d = np.repeat(train_v[:, :, np.newaxis], num_points, axis=2)
    test_v_2d = np.repeat(test_v[:, :, np.newaxis], num_points, axis=2)
    
    train_v_flat = train_v_2d.reshape(batch_train, total_points)
    test_v_flat = test_v_2d.reshape(batch_test, total_points)

    x_exp_train = np.tile(x_flat, (batch_train, 1))
    t_exp_train = np.tile(t_flat, (batch_train, 1))
    train_input = np.stack((train_v_flat, x_exp_train, t_exp_train), axis=2)

    x_exp_test = np.tile(x_flat, (batch_test, 1))
    t_exp_test = np.tile(t_flat, (batch_test, 1))
    test_input = np.stack((test_v_flat, x_exp_test, t_exp_test), axis=2)

    train_output = train_u.reshape(batch_train, total_points, 1)
    test_output = test_u.reshape(batch_test, total_points, 1)

    return train_input.astype(np.float32), None, train_output.astype(np.float32), \
           test_input.astype(np.float32), None, test_output.astype(np.float32)