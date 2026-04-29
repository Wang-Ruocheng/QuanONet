"""
PyTorch nn.Module variants of QuanONet and HEAQNN.

These mirror the MindSpore models in core/models.py but use PyTorch and
delegate quantum circuit construction to either TorchQuantum or Qiskit
backends (quantum_circuits_tq.py / quantum_circuits_qiskit.py).

The trainable-frequency (TF) logic is replicated using nn.Linear and
nn.Parameter, matching the MindSpore CombinedNet(RepeatLayer, LinearLayer).
"""
import numpy as np
import torch
import torch.nn as nn


class _RepeatLinear(nn.Module):
    """
    Trainable frequency layer: y = Linear(x).repeat_interleave(repeat_factor, dim=1)

    Mirrors MindSpore CombinedNet(RepeatLayer(out_size), LinearLayer(out_size, init_scale)).

    Args:
        in_features: input dimension
        out_features: total output dimension (= depth * num_qubits)
        init_scale: initial scale for the linear weight
    """
    def __init__(self, in_features, out_features, init_scale=0.1):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        nn.init.constant_(self.linear.weight, init_scale)

    def forward(self, x):
        return self.linear(x)


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
    else:
        raise ValueError(f"Unknown quantum_backend for PyTorch models: '{quantum_backend}'")


class QuanONetPT(nn.Module):
    """
    PyTorch QuanONet with TorchQuantum or Qiskit quantum backend.

    Mirrors core/models.py QuanONet (MindSpore nn.Cell) in architecture:
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
            self.branch_freq = _RepeatLinear(branch_input_size,
                                             self.branch_enc_size, scale_coeff)
            self.trunk_freq = _RepeatLinear(trunk_input_size,
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

    Mirrors core/models.py HEAQNN (MindSpore nn.Cell).

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
            self.freq = _RepeatLinear(input_size, enc_size, scale_coeff)
        else:
            self.freq = _ScaleRepeat(input_size, enc_size, scale_coeff)

        self.quantum_layer = _build_quantum_layer(
            quantum_backend, num_qubits,
            total_input_size=enc_size,
            net_size=net_size, ham_bound=ham_bound, ham_diag=ham_diag,
        )
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: (batch, input_size)
        Returns:
            (batch, 1)
        """
        enc = self.freq(x)
        out = self.quantum_layer(enc)
        return out + self.bias
