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


def sample_2D_Operator_data(u0, u, x, t, sample_num):
    """Sample data for 2D operator problems (Strictly aligned with original logic)."""
    num = u.shape[0]
    
    # 扩展 u0 作为 branch input
    branch_input = np.repeat(u0, sample_num, axis=0)
    
    # 构建正确的 2D 时空坐标网格 (X, T) -> 严格复刻原脚本的 repeat 和 tile 逻辑
    x_repeat = np.repeat(x, len(t)).reshape(-1, 1)
    t_tile = np.tile(t, len(x)).reshape(-1, 1)
    grid_coords = np.concatenate((x_repeat, t_tile), axis=1) # Shape: (len(x)*len(t), 2)
    
    trunk_input_list = []
    output_list = []
    total_points = len(x) * len(t)
    
    for i in range(num):
        # 随机采样索引
        indices = np.random.choice(total_points, sample_num, replace=False)
        
        # 提取对应的坐标和真解
        trunk_input_list.append(grid_coords[indices])
        output_list.append(u[i].reshape(-1, 1)[indices])
        
    trunk_input = np.concatenate(trunk_input_list, axis=0)
    output = np.concatenate(output_list, axis=0)
    
    return branch_input, trunk_input, output


def PDE_encode(generate_data, num_train, num_test, num_points, num_points_0, train_sample_num, test_sample_num, num_cal=None):
    """Encode PDE data for training (Corrected version)."""
    
    # 1. 完整接收 6 个返回值，绝不丢弃 t！
    try:
        train_u0, train_u, test_u0, test_u, x, t = generate_data(num_train, num_test, num_points, num_points_0)
    except TypeError:
        train_u0, train_u, test_u0, test_u, x, t = generate_data(num_train, num_test, num_points, num_points_0, num_cal=num_cal)
        
    # 2. 调用正确的 2D 采样函数
    train_branch, train_trunk, train_out = sample_2D_Operator_data(
        train_u0, train_u, x, t, train_sample_num
    )
    test_branch, test_trunk, test_out = sample_2D_Operator_data(
        test_u0, test_u, x, t, test_sample_num
    )
    
    return train_branch, train_trunk, train_out, test_branch, test_trunk, test_out