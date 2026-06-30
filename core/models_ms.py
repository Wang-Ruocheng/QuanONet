"""
MindSpore model definitions for QuanONet.

Classical models:  FNNMS, DeepONetMS, SpectralConv1dMS, FNOMS  (nn.Cell)
Quantum models:    QuanONetMS, HEAQNNMS                       (nn.Cell)
"""

import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Parameter, Tensor
import mindspore.common.dtype as mstype

try:
    from layers import *
    from quantum_circuits_ms import circuit2network, quanonet_build, heaqnnwork_build
except ImportError:
    from .layers import *
    from .quantum_circuits_ms import circuit2network, quanonet_build, heaqnnwork_build

try:
    from mindspore.ops import fft_rfft, fft_irfft
    _has_fft_ops = True
except ImportError:
    _has_fft_ops = False
    try:
        from mindspore.ops import FFTWithSize as _FFTWithSize
        _has_fft_class = True
    except ImportError:
        _has_fft_class = False


# ── Quantum models ────────────────────────────────────────────────────────────

class QuanONetMS(nn.Cell):
    """Quantum Operator Network with optional trainable frequency support."""

    def __init__(self, num_qubits, branch_input_size, trunk_input_size, net_size,
                 ham, scale_coeff=1, if_trainable_freq=False):
        super(QuanONetMS, self).__init__()
        (self.branch_depth, self.branch_linear_depth,
         self.trunk_depth, self.trunk_linear_depth) = net_size

        self.if_trainable_freq = if_trainable_freq

        QuanONet_circuit, trunk_circuit, branch_circuit = quanonet_build(
            num_qubits, branch_input_size, trunk_input_size, net_size
        )
        self.circuit = QuanONet_circuit
        self.trunk_circuit = trunk_circuit
        self.branch_circuit = branch_circuit
        self.QuanONet = circuit2network(QuanONet_circuit, ham)

        if self.if_trainable_freq:
            self.branch_LinearLayer = CombinedNet(
                RepeatLayer(self.branch_depth * num_qubits),
                LinearLayer(self.branch_depth * num_qubits, scale_coeff)
            )
            self.trunk_LinearLayer = CombinedNet(
                RepeatLayer(self.trunk_depth * num_qubits),
                LinearLayer(self.trunk_depth * num_qubits, scale_coeff)
            )
        else:
            self.branch_ScaleLayer = CombinedNet(
                CoeffLayer(branch_input_size, scale_coeff),
                RepeatLayer(self.branch_depth * num_qubits)
            )
            self.trunk_ScaleLayer = CombinedNet(
                CoeffLayer(trunk_input_size, scale_coeff),
                RepeatLayer(self.trunk_depth * num_qubits)
            )
        self.bias = ms.Parameter(ms.Tensor(0.0, dtype=ms.float32), name="bias")

    def construct(self, input):
        branch_input = input[0]
        trunk_input = input[1]

        if self.if_trainable_freq:
            branch_input = self.branch_LinearLayer(branch_input)
            trunk_input = self.trunk_LinearLayer(trunk_input)
        else:
            branch_input = self.branch_ScaleLayer(branch_input)
            trunk_input = self.trunk_ScaleLayer(trunk_input)

        input = mnp.concatenate((trunk_input, branch_input), axis=1)
        output = self.QuanONet(input) + self.bias
        return output


class HEAQNNMS(nn.Cell):
    """Hardware Efficient Ansatz Quantum Neural Network with optional trainable frequency support."""

    def __init__(self, num_qubits, input_size, net_size,
                 ham, scale_coeff=1, if_trainable_freq=False):
        super(HEAQNNMS, self).__init__()
        self.depth, self.linear_depth = net_size[0], net_size[1]

        self.if_trainable_freq = if_trainable_freq
        self.input_size = input_size

        HEAQNN_circuit = heaqnnwork_build(
            num_qubits, self.input_size, self.depth, self.linear_depth
        )
        self.HEAQNN = circuit2network(HEAQNN_circuit, ham)

        if self.if_trainable_freq:
            self.LinearLayer = CombinedNet(
                RepeatLayer(self.depth * num_qubits),
                LinearLayer(self.depth * num_qubits, scale_coeff)
            )
        else:
            self.CoeffLayer = CoeffLayer(self.input_size, scale_coeff)
            self.RepeatLayer = RepeatLayer(self.depth * num_qubits)

    def construct(self, input):
        if self.if_trainable_freq:
            input = self.LinearLayer(input)
        else:
            input = self.CoeffLayer(input)
            input = self.RepeatLayer(input)

        return self.HEAQNN(input)


# ── Classical models ──────────────────────────────────────────────────────────

class FNNMS(nn.Cell):
    """Feedforward Neural Network for comparison."""

    def __init__(self, input_size, output_size, net_size,
                 activation=nn.Tanh()):
        super(FNNMS, self).__init__()
        self.hidden_layer_depth, self.hidden_layer_width = net_size[0], net_size[1]

        self.FNN = FNNLayer(
            input_size, output_size,
            self.hidden_layer_width, self.hidden_layer_depth, activation
        )

    def construct(self, input):
        return self.FNN(input)


class DeepONetMS(nn.Cell):
    """MindSpore implementation of Deep Operator Network."""

    def __init__(self, branch_input_size, trunk_input_size, net_size,
                 activation=nn.Tanh()):
        super(DeepONetMS, self).__init__()

        (self.branch_depth, self.branch_width,
         self.trunk_depth, self.trunk_width) = net_size

        self.activation = activation  # applied to trunk output (paper convention)
        self.branch_net = FNNLayer(
            branch_input_size, self.branch_width,
            self.branch_width, self.branch_depth - 2, activation
        )
        self.trunk_net = FNNLayer(
            trunk_input_size, self.trunk_width,
            self.trunk_width, self.trunk_depth - 2, activation
        )

        self.sum_layer = SumLayer()
        self.bias = ms.Parameter(ms.Tensor(0.0, dtype=ms.float32), name="bias")

    def construct(self, input_tuple):
        branch_input = input_tuple[0]
        trunk_input = input_tuple[1]

        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        trunk_output = self.activation(trunk_output)  # paper convention: basis functions σ(trunk)

        output = branch_output * trunk_output
        output = self.sum_layer(output).expand_dims(1) + self.bias
        return output


# ── FNO (MindSpore) ───────────────────────────────────────────────────────────

def _rfft(x):
    """1D real FFT along last dimension."""
    if _has_fft_ops:
        return fft_rfft(x)
    elif _has_fft_class:
        op = _FFTWithSize(signal_ndim=1, inverse=False, real=True,
                          norm='backward', onesided=True)
        return op(x)
    else:
        raise RuntimeError(
            "No FFT op found in this MindSpore version. "
            "Please upgrade to MindSpore >= 2.1."
        )


def _irfft(x, n):
    """1D inverse real FFT along last dimension. n = output signal length."""
    if _has_fft_ops:
        return fft_irfft(x, n=n)
    elif _has_fft_class:
        op = _FFTWithSize(signal_ndim=1, inverse=True, real=True,
                          norm='backward', onesided=True, signal_sizes=(n,))
        return op(x)
    else:
        raise RuntimeError(
            "No FFT op found in this MindSpore version. "
            "Please upgrade to MindSpore >= 2.1."
        )


def _compl_mul1d(x, w):
    """Complex batched matmul: einsum('bix,iox->box', x, w)."""
    x_t = x.transpose(2, 0, 1)   # (freq, batch, in_ch)
    w_t = w.transpose(2, 0, 1)   # (freq, in_ch, out_ch)
    out = ops.bmm(x_t, w_t)      # (freq, batch, out_ch)
    return out.transpose(1, 2, 0)


class SpectralConv1dMS(nn.Cell):
    """1D Fourier spectral convolution layer (MindSpore). Mirrors SpectralConv1dPT in models_pt.py."""

    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1dMS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        scale = 1.0 / (in_channels * out_channels)
        # Store real and imaginary parts as last dim to match PyTorch complex numel count:
        # PT: cfloat param shape (in, out, modes) → numel = in*out*modes (complex count)
        # MS: float32 param shape (in, out, modes, 2) → numel = in*out*modes*2, so we
        #     halve in count_parameters to align. Functionally identical to split params.
        w_init = np.random.uniform(-scale, scale,
                                   (in_channels, out_channels, modes1, 2)).astype(np.float32)
        self.weight = Parameter(ms.Tensor(w_init), name='weight')

    def construct(self, x):
        signal_length = x.shape[-1]
        x_ft = _rfft(x)

        w = ops.Complex()(self.weight[..., 0], self.weight[..., 1])
        freq_size = x_ft.shape[-1]
        batch = x_ft.shape[0]

        modes_slice = x_ft[:, :, :self.modes1]
        modes_out = _compl_mul1d(modes_slice, w)

        if self.modes1 < freq_size:
            pad_size = freq_size - self.modes1
            zeros_pad = ops.zeros((batch, self.out_channels, pad_size), mstype.complex64)
            out_ft = ops.concat([modes_out, zeros_pad], axis=2)
        else:
            out_ft = modes_out

        return _irfft(out_ft, n=signal_length)


class FNOMS(nn.Cell):
    """
    Fourier Neural Operator (MindSpore). Mirrors FNOPT in models_pt.py.

    Input  format: (batch, n_points, in_channels).
    Output format: (batch, n_points, 1).
    Requires MindSpore >= 2.1 for fft_rfft / fft_irfft ops.
    """

    def __init__(self, modes, width, layers=1, fc_hidden=32, in_channels=2):
        super(FNOMS, self).__init__()
        self.modes = modes
        self.width = width
        self.layers = layers

        self.fc0 = nn.Dense(in_channels, width)
        self.convs = nn.CellList(
            [SpectralConv1dMS(width, width, modes) for _ in range(layers)]
        )
        self.ws = nn.CellList(
            [nn.Conv1d(width, width, kernel_size=1, has_bias=True, weight_init='HeUniform')
             for _ in range(layers)]
        )
        self.fc1 = nn.Dense(width, fc_hidden)
        self.fc2 = nn.Dense(fc_hidden, 1)
        self.relu = nn.ReLU()

        self.regularizer = None

    def construct(self, x):
        # x: (batch, n_points, in_channels)
        x = self.fc0(x)               # (batch, n_points, width)
        x = x.transpose(0, 2, 1)     # (batch, width, n_points)

        for i in range(self.layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = self.relu(x1 + x2)

        x = x.transpose(0, 2, 1)     # (batch, n_points, width)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x                       # (batch, n_points, 1)
