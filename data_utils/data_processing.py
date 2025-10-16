"""
Data processing and sampling functions.
"""

import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from mindspore.ops import operations as P
import mindspore.ops as ops


def sample_2D_Operator_data(u0, u, x, t, sample_num):
    """Sample data for 2D operator problems."""
    num = u.shape[0]
    num_points = u.shape[1]
    output_size = 1
    
    branch_input = mnp.repeat(u0, sample_num, axis=0)
    trunk_input = mnp.zeros((0, 2))
    output = mnp.zeros((0, output_size))
    
    for i in range(num):
        indices = ms.Tensor(np.random.choice(num_points**2, sample_num, replace=False), ms.int32)
        gather = P.Gather()
        x_repeat = ops.expand_dims(mnp.repeat(x, num_points, axis=0), 1)
        t_tile = ops.expand_dims(mnp.tile(t, num_points), 1)
        trunk_input_new = mnp.concatenate((x_repeat, t_tile), axis=1)
        trunk_input_new = gather(trunk_input_new, indices, 0)
        trunk_input = mnp.concatenate((trunk_input, trunk_input_new), axis=0)
        output = mnp.concatenate((output, gather(u[i].reshape(-1, 1), indices, 0)), axis=0)
    
    return branch_input, trunk_input, output

def sample_1D_Operator_data(u0, u, x, sample_num):
    """Sample data for 1D operator problems."""
    num = u.shape[0]
    num_points = u.shape[1]
    output_size = 1
    
    branch_input = mnp.repeat(u0, sample_num, axis=0)
    trunk_input = mnp.zeros((0, 1))
    output = mnp.zeros((0, output_size))
    
    for i in range(num):
        indices = ms.Tensor(np.random.choice(num_points, sample_num, replace=False), ms.int32)
        gather = P.Gather()
        trunk_input_new = gather(ops.expand_dims(x, 1), indices, 0)
        trunk_input = mnp.concatenate((trunk_input, trunk_input_new), axis=0)
        output = mnp.concatenate((output, ops.expand_dims(gather(u[i], indices, 0), 1)), axis=0)
    
    return branch_input, trunk_input, output


def ODE_encode(generate_data, num_train, num_test, num_points, train_sample_num, test_sample_num):
    """Encode ODE data for training."""
    train_u0, train_u, test_u0, test_u, x = generate_data(num_train, num_test, num_points)
    train_branch_input, train_trunk_input, train_output = sample_1D_Operator_data(
        train_u0, train_u, x, train_sample_num
    )
    test_branch_input, test_trunk_input, test_output = sample_1D_Operator_data(
        test_u0, test_u, x, test_sample_num
    )
    return (train_branch_input, train_trunk_input, train_output, 
            test_branch_input, test_trunk_input, test_output)


def PDE_encode(generate_data, num_train, num_test, num_points, train_sample_num, test_sample_num):
    """Encode PDE data for training."""
    train_u0, train_u, test_u0, test_u, x, t = generate_data(num_train, num_test, num_points)
    train_branch_input, train_trunk_input, train_output = sample_2D_Operator_data(
        train_u0, train_u, x, t, train_sample_num
    )
    test_branch_input, test_trunk_input, test_output = sample_2D_Operator_data(
        test_u0, test_u, x, t, test_sample_num
    )
    return (train_branch_input, train_trunk_input, train_output, 
            test_branch_input, test_trunk_input, test_output)
