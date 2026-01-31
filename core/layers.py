"""
Neural network layers for QuanONet.
"""

# This file contains various neural network layers used in QuanONet, including spectral convolution layers, feedforward layers, and utility layers for operations like summation and coefficient scaling.
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Parameter

class LinearLayer(nn.Cell):
    """Linear transformation layer with bias."""
    
    def __init__(self, input_size, initial_weight, initial_bias_range=np.pi):
        super(LinearLayer, self).__init__()
        self.input_size = input_size
        self.weights = Parameter(
            ms.Tensor(initial_weight * np.ones(input_size), dtype=ms.float32), 
            name="weights"
        )
        self.bias = Parameter(
            ms.Tensor(np.random.uniform(-initial_bias_range, initial_bias_range, input_size), dtype=ms.float32), 
            name="bias"
        )

    def construct(self, x):
        return x * self.weights + self.bias


class SumLayer(nn.Cell):
    """Sum reduction layer."""
    
    def __init__(self):
        super(SumLayer, self).__init__()
        self.sum = ops.ReduceSum()

    def construct(self, x):
        return self.sum(x, 1)


class CoeffLayer(nn.Cell):
    """Coefficient scaling layer."""
    
    def __init__(self, input_size, coeff):
        super(CoeffLayer, self).__init__()
        self.input_size = input_size
        self.coeff = ms.Tensor(coeff * np.ones(input_size), dtype=ms.float32)
    
    def construct(self, x):
        return x * self.coeff

class PeriodicEmbedding(nn.Cell):
    def __init__(self, domain_length=1.0):
        super().__init__()
        self.two_pi = 2 * np.pi
        self.period = domain_length

    def construct(self, x):
        """
        x: shape (N, d_in)
        By default, only perform periodic feature mapping on the first dimension (dim=0)
        """
        # 1. Extract first dimension (x coordinate)
        x_coord = x[:, 0:1]
        
        # 2. Apply periodic mapping -> (N, 1)
        cos_x = mnp.cos(self.two_pi * x_coord / self.period)
        sin_x = mnp.sin(self.two_pi * x_coord / self.period)
        
        # 3. Concatenate result
        if x.shape[1] > 1:
            # If there are other dimensions (e.g., t), keep them and concatenate after
            rest_coords = x[:, 1:]
            # Result shape: (N, 2 + d_rest)
            return mnp.concatenate([cos_x, sin_x, rest_coords], axis=-1)
        else:
            # Result shape: (N, 2)
            return mnp.concatenate([cos_x, sin_x], axis=-1)

class FNNLayer(nn.Cell):
    """Feedforward Neural Network layer."""
    
    def __init__(self, input_size, output_size, width, depth, activation=nn.Tanh()):
        super(FNNLayer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.width = width
        self.depth = depth
        self.activation = activation
        
        self.fc0 = nn.Dense(input_size, width)
        self.hidden_layers = nn.CellList([nn.Dense(width, width) for _ in range(depth)])
        self.fc_out = nn.Dense(width, output_size)

    def construct(self, x):
        x = self.fc0(x)
        x = self.activation(x)
        
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.activation(x)
        
        x = self.fc_out(x)
        return x


class CombinedNet(nn.Cell):
    """Combines two networks sequentially."""
    
    def __init__(self, Net1, Net2):
        super(CombinedNet, self).__init__()
        self.Net1 = Net1
        self.Net2 = Net2
    
    def construct(self, x):
        x = self.Net1(x)
        x = self.Net2(x)
        return x


class RepeatLayer(nn.Cell):
    """Repeats input to match desired size."""
    
    def __init__(self, n):
        super(RepeatLayer, self).__init__()
        self.n = n

    def construct(self, x):
        m = x.shape[1]
        repeat_times = self.n // m
        repeated_x = ops.tile(x, (1, repeat_times + 1))
        return repeated_x[:, :self.n]
