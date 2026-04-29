"""
MindSpore FNO (Fourier Neural Operator) — port of core/dde_models.py.

Implements the same SpectralConv1d + FNO architecture using MindSpore ops.
Requires MindSpore >= 2.1 for fft_rfft / fft_irfft ops.
"""
import numpy as np

try:
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.ops as ops
    from mindspore import Parameter, Tensor
    import mindspore.common.dtype as mstype
except ImportError:
    ms = None

try:
    # MindSpore 2.1+: functional FFT ops
    from mindspore.ops import fft_rfft, fft_irfft
    _has_fft_ops = True
except ImportError:
    _has_fft_ops = False
    try:
        # Fallback: FFTWithSize class (older MindSpore)
        from mindspore.ops import FFTWithSize as _FFTWithSize
        _has_fft_class = True
    except ImportError:
        _has_fft_class = False


def _rfft(x):
    """1D real FFT along last dimension. Returns complex tensor."""
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
    """
    Complex batched matrix multiply equivalent to einsum('bix,iox->box', x, w).

    x: (batch, in_channels, freq) complex
    w: (in_channels, out_channels, freq) complex
    Returns: (batch, out_channels, freq) complex
    """
    # Rearrange: (freq, batch, in_ch) × (freq, in_ch, out_ch) → (freq, batch, out_ch)
    # MindSpore bmm handles 3D tensors: (b, n, m) × (b, m, k)
    freq = x.shape[2]
    # x: (batch, in_ch, freq) → (freq, batch, in_ch)
    x_t = x.transpose(2, 0, 1)
    # w: (in_ch, out_ch, freq) → (freq, in_ch, out_ch)
    w_t = w.transpose(2, 0, 1)
    out = ops.bmm(x_t, w_t)   # (freq, batch, out_ch)
    return out.transpose(1, 2, 0)  # (batch, out_ch, freq)


class SpectralConv1d_MS(nn.Cell):
    """
    1D Fourier spectral convolution layer (MindSpore).

    Mirrors SpectralConv1d in core/dde_models.py.
    """
    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1d_MS, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        scale = 1.0 / (in_channels * out_channels)
        # Complex weight tensor: real and imaginary parts stored separately
        # Shape: (in_channels, out_channels, modes1, 2) where last dim = [real, imag]
        w_real = ms.Tensor(
            np.random.uniform(-scale, scale,
                              (in_channels, out_channels, modes1)).astype(np.float32)
        )
        w_imag = ms.Tensor(
            np.random.uniform(-scale, scale,
                              (in_channels, out_channels, modes1)).astype(np.float32)
        )
        self.weight_real = Parameter(w_real, name='weight_real')
        self.weight_imag = Parameter(w_imag, name='weight_imag')

    def construct(self, x):
        # x: (batch, in_channels, signal_length) float32
        signal_length = x.shape[-1]

        # rfft → (batch, in_channels, freq) complex
        x_ft = _rfft(x)

        # Build complex weight
        w = ops.Complex()(self.weight_real, self.weight_imag)
        # (in_channels, out_channels, modes1) complex

        freq_size = x_ft.shape[-1]
        batch = x_ft.shape[0]

        # Allocate output frequency tensor (zeros)
        out_ft = ops.zeros(
            (batch, self.out_channels, freq_size), mstype.complex64
        )

        # Multiply relevant modes
        modes_slice = x_ft[:, :, :self.modes1]         # (batch, in_ch, modes1)
        modes_out = _compl_mul1d(modes_slice, w)        # (batch, out_ch, modes1)

        # Scatter into output (use concat with zero padding for remaining modes)
        if self.modes1 < freq_size:
            pad_size = freq_size - self.modes1
            zeros_pad = ops.zeros(
                (batch, self.out_channels, pad_size), mstype.complex64
            )
            out_ft = ops.concat([modes_out, zeros_pad], axis=2)
        else:
            out_ft = modes_out

        # irfft → (batch, out_channels, signal_length)
        x_out = _irfft(out_ft, n=signal_length)
        return x_out


class FNO_MS(nn.Cell):
    """
    Fourier Neural Operator (MindSpore).

    Mirrors FNO in core/dde_models.py. Input format: (batch, n_points, in_channels).

    Args:
        modes: number of Fourier modes to keep
        width: channel width (hidden dimension)
        layers: number of Fourier layers
        fc_hidden: hidden size of the projection MLP
        in_channels: number of input channels (default 2: position + function value)
    """
    def __init__(self, modes, width, layers=1, fc_hidden=32, in_channels=2):
        super(FNO_MS, self).__init__()
        self.modes = modes
        self.width = width
        self.layers = layers

        # Initial projection: (n_pts, in_channels) → (n_pts, width)
        self.fc0 = nn.Dense(in_channels, width)

        # Fourier layers
        self.convs = nn.CellList(
            [SpectralConv1d_MS(width, width, modes) for _ in range(layers)]
        )
        self.ws = nn.CellList(
            [nn.Conv1d(width, width, kernel_size=1, has_bias=True)
             for _ in range(layers)]
        )

        # Projection MLP
        self.fc1 = nn.Dense(width, fc_hidden)
        self.fc2 = nn.Dense(fc_hidden, 1)
        self.relu = nn.ReLU()

        self.regularizer = None

    def construct(self, x):
        # x: (batch, n_points, in_channels)
        x = self.fc0(x)                  # (batch, n_points, width)
        x = x.transpose(0, 2, 1)        # (batch, width, n_points)

        for i in range(self.layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = self.relu(x1 + x2)

        x = x.transpose(0, 2, 1)        # (batch, n_points, width)
        x = self.relu(self.fc1(x))       # (batch, n_points, fc_hidden)
        x = self.fc2(x)                  # (batch, n_points, 1)
        return x.squeeze(-1)             # (batch, n_points)
