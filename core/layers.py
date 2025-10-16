"""
Neural network layers for QuanONet.
"""

# This file contains various neural network layers used in QuanONet, including spectral convolution layers, feedforward layers, and utility layers for operations like summation and coefficient scaling.
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Parameter

class SpectralConv1d(nn.Cell):
    """Spectral convolution layer for Fourier Neural Operator."""
    
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.scale = (1 / (in_channels * out_channels))
        self.weights = Parameter(
            self.scale * ms.Tensor(np.random.rand(in_channels, out_channels, modes), ms.float32)
        )

    def compl_mul1d(self, input, weights):
        """Complex multiplication in 1D."""
        return ops.einsum("bix,iox->box", input, weights)

    def construct(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients
        x_np = x.asnumpy()
        x_ft_np = np.fft.fft(x_np, axis=-1)
        x_ft = ms.Tensor(x_ft_np, ms.complex64)

        # Multiply relevant Fourier modes
        out_ft = ops.Zeros()((batchsize, self.out_channels, x.shape[-1])//2 + 1, ms.complex64)
        out_ft[:, :, :self.modes] = self.compl_mul1d(
            x_ft[:, :, :self.modes], 
            ops.cast(self.weights, ms.complex64)
        )

        # Return to physical space
        out_ft_np = out_ft.asnumpy()
        x_np = np.fft.ifft(out_ft_np, axis=-1)
        x = ms.Tensor(x_np, ms.float32)
        return x.real


class FNO1d(nn.Cell):
    """1D Fourier Neural Operator."""
    
    def __init__(self, modes, width):
        super(FNO1d, self).__init__()
        self.modes = modes
        self.width = width
        self.fc0 = nn.Dense(2, self.width)

        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv2 = SpectralConv1d(self.width, self.width, self.modes)
        self.conv3 = SpectralConv1d(self.width, self.width, self.modes)
        
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)

        self.fc1 = nn.Dense(self.width, 128)
        self.fc2 = nn.Dense(128, 1)

    def construct(self, x):
        grid = self.get_grid(x.shape)
        x = ops.Concat(-1)((x, grid))
        x = self.fc0(x)
        x = ops.Transpose()(x, (0, 2, 1))
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = ops.GeLU()(x)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = ops.GeLU()(x)
        
        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = ops.GeLU()(x)
        
        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2
        
        x = ops.Transpose()(x, (0, 2, 1))
        x = self.fc1(x)
        x = ops.GeLU()(x)
        x = self.fc2(x)
        return x

    def get_grid(self, shape):
        """Generate coordinate grid."""
        batchsize, size_x = shape[0], shape[1]
        gridx = np.linspace(0, 1, size_x)
        gridx = np.tile(gridx.reshape(1, size_x, 1), (batchsize, 1, 1))
        return ms.Tensor(gridx, ms.float32)


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
