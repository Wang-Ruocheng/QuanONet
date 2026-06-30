"""
PyTorch nn.Module variants of QuanONet/HEAQNN and classical models (FNO).

Quantum models:   QuanONetPT, HEAQNNPT — mirror core/models_ms.py using
                  TorchQuantum / Qiskit / PennyLane backends.
Classical models: SpectralConv1dPT, FNOPT  — PyTorch Fourier Neural Operator.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _TiledElementWise(nn.Module):
    """
    Trainable frequency layer that exactly mirrors MindSpore
    CombinedNet(RepeatLayer(out_size), LinearLayer(out_size, init_scale)):
      1. Tile x along dim-1: (batch, in) → (batch, out)  [same as RepeatLayer]
      2. Element-wise: y = x_tiled * weights + bias       [same as LinearLayer]

    Parameters are directly mappable from MindSpore .npz:
      branch_LinearLayer.Net2.weights → self.weights
      branch_LinearLayer.Net2.bias   → self.bias

    Args:
        in_features: input dimension
        out_features: total output dimension (= depth * num_qubits)
        init_scale: initial value for weights (matches MindSpore init)
    """
    def __init__(self, in_features, out_features, init_scale=0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.repeats = int(np.ceil(out_features / in_features))
        self.weights = nn.Parameter(torch.full((out_features,), init_scale))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        # Tile: (batch, in_features) → (batch, out_features)
        tiled = x.repeat(1, self.repeats)[:, :self.out_features]
        return tiled * self.weights + self.bias


class _ScaleRepeat(nn.Module):
    """
    Fixed-scale layer: y = (x * scale).repeat_interleave until size out_features.

    Mirrors MindSpore CombinedNet(CoeffLayer(in, scale), RepeatLayer(out_size)).

    Args:
        in_features: input dimension
        out_features: total output dimension (= depth * num_qubits)
        scale: fixed scaling coefficient
    """
    def __init__(self, in_features, out_features, scale=0.01):
        super().__init__()
        self.scale = scale
        self.in_features = in_features
        self.out_features = out_features
        repeats = int(np.ceil(out_features / in_features))
        self.repeats = repeats

    def forward(self, x):
        # x: (batch, in_features)
        scaled = x * self.scale
        # Tile to cover out_features then slice
        repeated = scaled.repeat(1, self.repeats)[:, :self.out_features]
        return repeated


def _build_quantum_layer(quantum_backend, num_qubits, total_input_size,
                         net_size, ham_bound, ham_diag,
                         branch_input_size=None, trunk_input_size=None):
    """Dispatch to TQ or Qiskit builder."""
    if quantum_backend == 'torchquantum':
        from core.quantum_circuits_tq import build_quanonet_tq, build_heaqnn_tq
        if branch_input_size is not None:
            return build_quanonet_tq(num_qubits, branch_input_size, trunk_input_size,
                                     net_size, ham_bound=ham_bound, ham_diag=ham_diag)
        else:
            return build_heaqnn_tq(num_qubits, total_input_size, net_size,
                                   ham_bound=ham_bound, ham_diag=ham_diag)
    elif quantum_backend == 'qiskit':
        from core.quantum_circuits_qiskit import build_quanonet_qiskit, build_heaqnn_qiskit
        if branch_input_size is not None:
            return build_quanonet_qiskit(num_qubits, branch_input_size, trunk_input_size,
                                         net_size, ham_bound=ham_bound, ham_diag=ham_diag)
        else:
            return build_heaqnn_qiskit(num_qubits, total_input_size, net_size,
                                       ham_bound=ham_bound, ham_diag=ham_diag)
    elif quantum_backend == 'pennylane':
        from core.quantum_circuits_pl import build_quanonet_pl, build_heaqnn_pl
        if branch_input_size is not None:
            return build_quanonet_pl(num_qubits, branch_input_size, trunk_input_size,
                                     net_size, ham_bound=ham_bound, ham_diag=ham_diag)
        else:
            return build_heaqnn_pl(num_qubits, total_input_size, net_size,
                                   ham_bound=ham_bound, ham_diag=ham_diag)
    else:
        raise ValueError(f"Unknown quantum_backend for PyTorch models: '{quantum_backend}'")


class QuanONetPT(nn.Module):
    """
    PyTorch QuanONet with TorchQuantum or Qiskit quantum backend.

    Mirrors core/models_ms.py QuanONet (MindSpore nn.Cell) in architecture:
      - Trainable-freq mode: branch/trunk inputs go through separate Linear layers
        that map to (batch, depth*num_qubits), then are concatenated and fed to
        the quantum circuit.
      - Fixed-scale mode: inputs are scaled by scale_coeff and tiled.

    Args:
        num_qubits: number of qubits
        branch_input_size: number of branch input features
        trunk_input_size: number of trunk input features
        net_size: (branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth)
        scale_coeff: initial scale for the frequency mapping
        if_trainable_freq: whether to use trainable (Linear) frequency mapping
        quantum_backend: 'torchquantum' or 'qiskit'
        ham_bound: (lower, upper) for the simple Hamiltonian
        ham_diag: optional explicit Hamiltonian diagonal (overrides ham_bound)
    """
    def __init__(self, num_qubits, branch_input_size, trunk_input_size, net_size,
                 scale_coeff=1.0, if_trainable_freq=False,
                 quantum_backend='torchquantum',
                 ham_bound=(-5.0, 5.0), ham_diag=None):
        super().__init__()
        branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth = net_size
        self.if_trainable_freq = if_trainable_freq
        self.branch_enc_size = branch_depth * num_qubits
        self.trunk_enc_size = trunk_depth * num_qubits

        if if_trainable_freq:
            self.branch_freq = _TiledElementWise(branch_input_size,
                                                 self.branch_enc_size, scale_coeff)
            self.trunk_freq = _TiledElementWise(trunk_input_size,
                                                self.trunk_enc_size, scale_coeff)
        else:
            self.branch_freq = _ScaleRepeat(branch_input_size,
                                            self.branch_enc_size, scale_coeff)
            self.trunk_freq = _ScaleRepeat(trunk_input_size,
                                           self.trunk_enc_size, scale_coeff)

        self.quantum_layer = _build_quantum_layer(
            quantum_backend, num_qubits,
            total_input_size=self.trunk_enc_size + self.branch_enc_size,
            net_size=net_size, ham_bound=ham_bound, ham_diag=ham_diag,
            branch_input_size=branch_input_size, trunk_input_size=trunk_input_size,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch_input, trunk_input):
        """
        Args:
            branch_input: (batch, branch_input_size)
            trunk_input:  (batch, trunk_input_size)
        Returns:
            (batch, 1) output
        """
        branch_enc = self.branch_freq(branch_input)  # (batch, branch_depth*n_qubits)
        trunk_enc = self.trunk_freq(trunk_input)      # (batch, trunk_depth*n_qubits)
        # Concatenate trunk first then branch, matching MindQuantum circuit order
        x = torch.cat([trunk_enc, branch_enc], dim=1)
        out = self.quantum_layer(x)                   # (batch, 1)
        return out + self.bias


class HEAQNNPT(nn.Module):
    """
    PyTorch HEAQNN with TorchQuantum or Qiskit quantum backend.

    Mirrors core/models_ms.py HEAQNN (MindSpore nn.Cell).

    Args:
        num_qubits: number of qubits
        input_size: number of input features
        net_size: (depth, linear_depth, _, _)
        scale_coeff: initial scale for the frequency mapping
        if_trainable_freq: whether to use trainable frequency mapping
        quantum_backend: 'torchquantum' or 'qiskit'
        ham_bound: (lower, upper) for the simple Hamiltonian
        ham_diag: optional explicit Hamiltonian diagonal
    """
    def __init__(self, num_qubits, input_size, net_size,
                 scale_coeff=1.0, if_trainable_freq=False,
                 quantum_backend='torchquantum',
                 ham_bound=(-5.0, 5.0), ham_diag=None):
        super().__init__()
        depth = net_size[0]
        enc_size = depth * num_qubits
        self.if_trainable_freq = if_trainable_freq

        if if_trainable_freq:
            self.freq = _TiledElementWise(input_size, enc_size, scale_coeff)
        else:
            self.freq = _ScaleRepeat(input_size, enc_size, scale_coeff)

        self.quantum_layer = _build_quantum_layer(
            quantum_backend, num_qubits,
            total_input_size=enc_size,
            net_size=net_size, ham_bound=ham_bound, ham_diag=ham_diag,
        )

    def forward(self, x):
        """
        Args:
            x: (batch, input_size)
        Returns:
            (batch, 1)
        """
        enc = self.freq(x)
        return self.quantum_layer(enc)


# ── FNO (PyTorch) ─────────────────────────────────────────────────────────────

class SpectralConv1dPT(nn.Module):
    """1D Fourier spectral convolution layer (PyTorch). Mirrors SpectralConv1dMS in models_ms.py."""

    def __init__(self, in_channels, out_channels, modes1):
        super(SpectralConv1dPT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1,
                                    dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-1) // 2 + 1,
                             device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        return torch.fft.irfft(out_ft, n=x.size(-1))


class FNOPT(nn.Module):
    """
    Fourier Neural Operator (PyTorch). Mirrors FNOMS in models_ms.py.

    Input format: (batch, n_points, in_channels).
    """

    def __init__(self, modes, width, layers=1, fc_hidden=32, in_channels=2):
        super(FNOPT, self).__init__()
        self.modes1 = modes
        self.width = width
        self.layers = layers

        self.fc0 = nn.Linear(in_channels, self.width)
        self.convs = nn.ModuleList(
            [SpectralConv1dPT(self.width, self.width, self.modes1) for _ in range(layers)]
        )
        self.ws = nn.ModuleList(
            [nn.Conv1d(self.width, self.width, 1) for _ in range(layers)]
        )
        self.fc1 = nn.Linear(self.width, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, 1)

        self.regularizer = None

    def forward(self, x):
        # x: (batch, n_points, in_channels)
        x = self.fc0(x)
        x = x.permute(0, 2, 1)   # (batch, width, n_points)

        for i in range(self.layers):
            x1 = self.convs[i](x)
            x2 = self.ws[i](x)
            x = F.relu(x1 + x2)

        x = x.permute(0, 2, 1)   # (batch, n_points, width)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
