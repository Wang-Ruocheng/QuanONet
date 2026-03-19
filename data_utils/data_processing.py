"""
Data processing functions for DeepXDE-based classical operator learning.
This module provides MindSpore-free versions of data processing functions.
"""

import numpy as np
from scipy import interpolate  # <--- 关键修复：添加了这个导入

def ODE_encode(generate_data, num_train, num_test, num_points, num_points_0, train_sample_num, test_sample_num, num_cal):
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
    train_indices = np.random.choice(num_points, size=num_train * train_sample_num, replace=True)
    test_indices = np.random.choice(num_points, size=num_test * test_sample_num, replace=True)

    # Branch input: u0 values repeated for each sample point
    # Shape: (num_samples * sample_num, num_points)
    train_branch_input = np.repeat(u0_train, train_sample_num, axis=0)  # (num_train * train_sample_num, num_points)
    test_branch_input = np.repeat(u0_test, test_sample_num, axis=0)     # (num_test * test_sample_num, num_points)

    # Trunk input: spatial coordinates at sampled points
    # Shape: (num_samples * sample_num, 1)
    train_trunk_input = x_trunk[train_indices]  # (num_train * train_sample_num, 1)
    test_trunk_input = x_trunk[test_indices]    # (num_test * test_sample_num, 1)

    # Output: u values at sampled points
    train_output = u_train[np.repeat(np.arange(num_train), train_sample_num), train_indices].reshape(-1, 1)
    test_output = u_test[np.repeat(np.arange(num_test), test_sample_num), test_indices].reshape(-1, 1)

    return train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output

def ODE_fncode(generate_data, num_train, num_test, num_points, train_sample_num, test_sample_num):
    """
    Specialized data encoding for FNO.
    Includes auto-interpolation to align input resolution with target grid resolution.
    """
    # 1. Generate raw data
    # train_v (Input u0) shape: [Batch, 1000] (usually from num_points_0)
    # train_u (Output u) shape: [Batch, 64] (usually from num_points)
    train_v, train_u, test_v, test_u, _ = generate_data(num_train, num_test)
    
    # 2. Auto-align Resolution (Critical for FNO)
    # FNO requires the input function v(x) and the grid x to have the same resolution.
    current_dim = train_v.shape[1]
    
    if current_dim != num_points:
        print(f"FNO Alignment: Interpolating input from {current_dim} to {num_points}")
        x_old = np.linspace(0, 1, current_dim)
        x_new = np.linspace(0, 1, num_points)
        
        # Use scipy for batch interpolation
        f_train = interpolate.interp1d(x_old, train_v, axis=1, kind='linear')
        train_v = f_train(x_new) # Reshape -> [Batch, num_points]
        
        f_test = interpolate.interp1d(x_old, test_v, axis=1, kind='linear')
        test_v = f_test(x_new)

    # 3. Create Grid
    x = np.linspace(0, 1, num_points).astype(np.float32)
    
    def sample_1D_Operator_fndata(v, u, x, sample_num):
        num = u.shape[0]
        num_sensors = u.shape[1]
        indices = np.zeros((0, sample_num))
        output = np.zeros((0, sample_num))
        
        x = x.reshape(1, -1)
        x = np.repeat(x, num, axis=0)
        x = np.expand_dims(x, axis=2)
        v = np.expand_dims(v, axis=2)
        
        # Concatenate: Input = [Function_Value, Coordinate] -> (Batch, Points, 2)
        input = np.concatenate((v, x), axis=2)
        
        for i in range(num):
            if num_sensors == sample_num:
                indice = np.arange(num_sensors).reshape(1, -1)
                output_new = u[i].reshape(1, -1)
            else:
                stride = num_sensors // sample_num
                indice = np.arange(0, num_sensors, stride)[:sample_num].reshape(1, -1)
                output_new = u[i, indice[0]].reshape(1, -1)
                
            indices = np.concatenate((indices, indice), axis=0)
            output = np.concatenate((output, output_new), axis=0)
            
        output = np.expand_dims(output, axis=2)
        return input.astype(np.float32), indices.astype(np.int64), output.astype(np.float32)

    # Ensure sample_num matches resolution for FNO
    train_input, train_indices, train_output = sample_1D_Operator_fndata(train_v, train_u, x, train_sample_num)
    test_input, test_indices, test_output = sample_1D_Operator_fndata(test_v, test_u, x, test_sample_num)
    
    return train_input, train_indices, train_output, test_input, test_indices, test_output


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