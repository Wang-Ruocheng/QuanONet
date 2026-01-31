"""
Data processing functions for DeepXDE-based classical operator learning.
This module provides MindSpore-free versions of data processing functions.
"""

import numpy as np


def ODE_encode(generate_data, num_train, num_test, num_points, num_points_0, train_sample_num, test_sample_num, num_cal):
    """
    Encode ODE operator data for DeepONet training.

    Args:
        generate_data: Function to generate data
        num_train: Number of training samples
        num_test: Number of test samples
        num_points: Number of spatial points
        train_sample_num: Number of training samples per function
        test_sample_num: Number of test samples per function

    Returns:
        train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output
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

    # Adjust trunk input to match sampled points
    train_trunk_input = x_trunk[train_indices]
    test_trunk_input = x_trunk[test_indices]

    return train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output

def ODE_fncode(generate_data, num_train, num_test, num_points, train_sample_num, test_sample_num):
    train_v, train_u, test_v, test_u, _ = generate_data(num_train, num_test)
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
        input = np.concatenate((v, x), axis=2)
        for i in range(num):
            indice = np.array(range(0, num_sensors, num_sensors//sample_num)).reshape(1, -1)
            indices = np.concatenate((indices, indice), axis=0)
            output_new = u[i].reshape(1, -1)
            output = np.concatenate((output, output_new), axis=0)
        output = np.expand_dims(output, axis=2)
        return input.astype(np.float32), indices.astype(np.int64), output.astype(np.float32)
    train_input, train_indices, train_output = sample_1D_Operator_fndata(train_v, train_u, x, train_sample_num)
    test_input, test_indices, test_output = sample_1D_Operator_fndata(test_v, test_u, x, test_sample_num)
    return train_input, train_indices, train_output, test_input, test_indices, test_output


def PDE_encode(generate_data, num_train, num_test, num_points, num_points_0, train_sample_num, test_sample_num, num_cal=None):
    """
    Encode PDE operator data for DeepONet training.

    Args:
        generate_data: Function to generate data
        num_train: Number of training samples
        num_test: Number of test samples
        num_points: Number of spatial/temporal points
        train_sample_num: Number of training samples per function
        test_sample_num: Number of test samples per function

    Returns:
        train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output
    """
    # Generate data - call with appropriate parameters
    try:
        # Try calling with operator_type first (for dde_data_generation functions)
        result = generate_data(num_train, num_test, num_points, num_points_0)
        if len(result) == 5:  # x not included
            u0_train, u_train, u0_test, u_test, x = result
        else:  # x included
            u0_train, u_train, u0_test, u_test, x = result[:5]
    except TypeError:
        # Fall back to original signature with num_cal
        result = generate_data(num_train, num_test, num_points, num_points_0, num_cal=num_cal)
        if len(result) == 5:
            u0_train, u_train, u0_test, u_test, x = result
        else:
            u0_train, u_train, u0_test, u_test, x = result[:5]

    # For PDEs, we need to handle 2D data (space-time)
    if u_train.ndim == 2:  # (num_samples, num_points) - 1D PDE
        # Trunk input is spatial coordinates
        x_trunk = x.reshape(-1, 1)  # (num_points, 1)

        # Sample spatial points for training and testing
        train_indices = np.random.choice(num_points, size=num_train * train_sample_num, replace=True)
        test_indices = np.random.choice(num_points, size=num_test * test_sample_num, replace=True)

        # Branch input: initial/boundary conditions repeated for each sample
        train_branch_input = np.repeat(u0_train, train_sample_num, axis=0)  # (num_train * train_sample_num, num_points)
        test_branch_input = np.repeat(u0_test, test_sample_num, axis=0)     # (num_test * test_sample_num, num_points)

        # Trunk input: spatial coordinates at sampled points
        train_trunk_input = x_trunk[train_indices]  # (num_train * train_sample_num, 1)
        test_trunk_input = x_trunk[test_indices]    # (num_test * test_sample_num, 1)

        # Output: solution values at sampled spatial points
        train_output = u_train[np.repeat(np.arange(num_train), train_sample_num), train_indices].reshape(-1, 1)
        test_output = u_test[np.repeat(np.arange(num_test), test_sample_num), test_indices].reshape(-1, 1)

    else:  # Higher dimensional PDE data
        # Simplified handling for 2D+ PDEs
        # Flatten spatial dimensions for branch input
        spatial_size = u_train.shape[-1] if u_train.ndim > 2 else u_train.shape[1]

        # Branch input: boundary/initial conditions (flattened)
        train_branch_input = u0_train.reshape(num_train, -1) if u0_train.ndim > 1 else u0_train
        test_branch_input = u0_test.reshape(num_test, -1) if u0_test.ndim > 1 else u0_test

        # Create spatial-temporal grid for trunk input
        if u_train.ndim == 3:  # (num_samples, time_points, space_points)
            time_points, space_points = u_train.shape[1], u_train.shape[2]
            x_grid, t_grid = np.meshgrid(x, np.linspace(0, 1, time_points))
            trunk_coords = np.column_stack([t_grid.ravel(), x_grid.ravel()])  # (time_points * space_points, 2)
        else:
            # Fallback for other cases
            trunk_coords = x.reshape(-1, 1)

        # Sample points
        total_points = trunk_coords.shape[0]
        train_indices = np.random.choice(total_points, size=num_train * train_sample_num, replace=True)
        test_indices = np.random.choice(total_points, size=num_test * test_sample_num, replace=True)

        train_trunk_input = trunk_coords[train_indices]
        test_trunk_input = trunk_coords[test_indices]

        # Output: solution values at sampled points
        if u_train.ndim == 3:
            # For 2D PDEs, flatten and sample
            u_train_flat = u_train.reshape(num_train, -1)
            u_test_flat = u_test.reshape(num_test, -1)

            train_output = u_train_flat[np.repeat(np.arange(num_train), train_sample_num), train_indices].reshape(-1, 1)
            test_output = u_test_flat[np.repeat(np.arange(num_test), test_sample_num), test_indices].reshape(-1, 1)
        else:
            # Fallback
            train_output = u_train.flatten()[train_indices].reshape(-1, 1)
            test_output = u_test.flatten()[test_indices].reshape(-1, 1)

    return train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output