"""
TorchQuantum circuit builders for QuanONet/HEAQNN.

Implements Hardware Efficient Ansatz (HEA) circuits mirroring the MindQuantum
structure in quantum_circuits.py: data re-uploading encoding (RX) + alternating
RY/RZ/RY ansatz sublayers + CNOT ring entanglement.
"""
import numpy as np
import torch
import torch.nn as nn

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
except ImportError:
    tq = None
    tqf = None


class _TQHEACircuit(nn.Module):
    """
    A parametric HEA quantum circuit implemented with TorchQuantum.

    The circuit consists of a sequence of blocks, each containing:
      1. Encoding layer: RX(x_i) on each qubit (data re-uploading)
      2. `linear_depth` ansatz sublayers: RY/RZ/RY per qubit + CNOT ring

    The circuit operates on a single quantum register of `n_wires` qubits,
    matching the trunk+branch concatenated structure of MindQuantum QuanONet.

    Args:
        n_wires: number of qubits
        block_configs: list of (n_encode_params, linear_depth) tuples, one per block
        ham_offset: constant offset of the Hamiltonian (scalar)
        ham_coeff_per_qubit: coefficient for each Z_i term; used when ham_diag is None
        ham_diag: full diagonal of the Hamiltonian in the computational basis (2^n_wires,);
                  if provided, overrides ham_offset / ham_coeff_per_qubit
    """
    def __init__(self, n_wires, block_configs, ham_offset=0.0,
                 ham_coeff_per_qubit=0.0, ham_diag=None):
        super().__init__()
        if tq is None:
            raise ImportError("torchquantum is required but not installed. "
                              "Install with: pip install torchquantum")
        self.n_wires = n_wires
        self.block_configs = block_configs

        # Trainable ansatz weights: one (3, n_wires) block per ansatz sublayer
        total_ansatz_blocks = sum(ld for _, ld in block_configs)
        self.ansatz_weights = nn.Parameter(
            torch.empty(total_ansatz_blocks, 3, n_wires)
        )
        nn.init.uniform_(self.ansatz_weights, -np.pi, np.pi)

        # Hamiltonian specification
        if ham_diag is not None:
            self.register_buffer('ham_diag',
                                 torch.tensor(ham_diag, dtype=torch.float32))
            self.use_full_ham = True
        else:
            self.ham_offset = float(ham_offset)
            self.ham_coeff = float(ham_coeff_per_qubit)
            self.use_full_ham = False

    def forward(self, x):
        """
        Args:
            x: (batch, total_encode_params) float tensor — pre-processed
               concatenated [trunk_enc | branch_enc] from models_pt.py
        Returns:
            (batch, 1) Hamiltonian expectation value
        """
        batch = x.shape[0]
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=batch, device=x.device)

        param_col = 0
        ansatz_block = 0

        for n_encode, linear_depth in self.block_configs:
            # --- Encoding: RX(x_i) on each qubit ---
            for j in range(n_encode):
                wire = j % self.n_wires
                if param_col < x.shape[1]:
                    tqf.rx(qdev, wires=wire,
                           params=x[:, param_col].unsqueeze(-1))
                param_col += 1

            # --- Ansatz: linear_depth sublayers of RY/RZ/RY + CNOT ring ---
            for _ in range(linear_depth):
                w = self.ansatz_weights[ansatz_block]  # (3, n_wires)
                for i in range(self.n_wires):
                    tqf.ry(qdev, wires=i,
                           params=w[0, i].unsqueeze(0).expand(batch).unsqueeze(-1))
                    tqf.rz(qdev, wires=i,
                           params=w[1, i].unsqueeze(0).expand(batch).unsqueeze(-1))
                    tqf.ry(qdev, wires=i,
                           params=w[2, i].unsqueeze(0).expand(batch).unsqueeze(-1))
                for i in range(self.n_wires):
                    tqf.cnot(qdev, wires=[i, (i + 1) % self.n_wires])
                ansatz_block += 1

        return self._measure(qdev, batch, x.device)

    def _measure(self, qdev, batch, device):
        """Compute Hamiltonian expectation from the current quantum state."""
        # Get statevector: (batch, 2^n_wires) complex
        states = qdev.get_states_1d()
        probs = states.abs().pow(2)  # (batch, 2^n_wires)

        if self.use_full_ham:
            ham_d = self.ham_diag.to(device)
            expval = (probs * ham_d.unsqueeze(0)).sum(dim=1, keepdim=True)
        else:
            # H = offset + coeff_per_qubit * sum_i Z_i
            # <Z_i> = sum_k (-1)^{bit_i(k)} * prob_k
            n = self.n_wires
            k = torch.arange(2 ** n, device=device)
            z_sum = torch.zeros(batch, 1, device=device)
            for i in range(n):
                # bit i of k (little-endian: qubit 0 = bit 0)
                sign = (1 - 2 * ((k >> i) & 1).float())  # (2^n,)
                z_i = (probs * sign.unsqueeze(0)).sum(dim=1, keepdim=True)
                z_sum = z_sum + z_i
            expval = self.ham_offset + self.ham_coeff * z_sum
        return expval  # (batch, 1)


def _make_block_configs(num_qubits, trunk_depth, trunk_linear_depth,
                        branch_depth, branch_linear_depth):
    """Build block_configs list matching MindQuantum QuanONet_build layout."""
    configs = []
    for _ in range(trunk_depth):
        configs.append((num_qubits, trunk_linear_depth))
    for _ in range(branch_depth):
        configs.append((num_qubits, branch_linear_depth))
    return configs


def _ham_params(num_qubits, lower_bound=-5.0, upper_bound=5.0):
    """Return (offset, coeff_per_qubit) for generate_simple_hamiltonian equivalent."""
    coff = upper_bound - lower_bound
    offset = lower_bound + coff / 2.0
    coeff_per_qubit = coff / 2.0 / num_qubits
    return offset, coeff_per_qubit


def build_quanonet_tq(num_qubits, branch_input_size, trunk_input_size, net_size,
                      ham_bound=(-5.0, 5.0), ham_diag=None):
    """
    Build a TorchQuantum QuanONet circuit.

    Mirrors QuanONet_build() + circuit2network() from quantum_circuits.py.

    Args:
        num_qubits: number of qubits
        branch_input_size: size of branch encoder input (not used directly; net_size governs capacity)
        trunk_input_size: size of trunk encoder input
        net_size: (branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth)
        ham_bound: (lower, upper) for the simple Z-sum Hamiltonian
        ham_diag: optional explicit Hamiltonian diagonal (overrides ham_bound)
    Returns:
        _TQHEACircuit instance (nn.Module)
    """
    branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth = net_size
    block_configs = _make_block_configs(num_qubits,
                                        trunk_depth, trunk_linear_depth,
                                        branch_depth, branch_linear_depth)
    if ham_diag is not None:
        return _TQHEACircuit(num_qubits, block_configs, ham_diag=ham_diag)

    offset, coeff = _ham_params(num_qubits, ham_bound[0], ham_bound[1])
    return _TQHEACircuit(num_qubits, block_configs,
                         ham_offset=offset, ham_coeff_per_qubit=coeff)


def build_heaqnn_tq(num_qubits, input_size, net_size,
                    ham_bound=(-5.0, 5.0), ham_diag=None):
    """
    Build a TorchQuantum HEAQNN circuit.

    Mirrors HEAQNNwork_build() from quantum_circuits.py.

    Args:
        num_qubits: number of qubits
        input_size: size of input (not used directly; net_size governs capacity)
        net_size: (depth, linear_depth, _, _) — only first two elements used
        ham_bound: (lower, upper) for the simple Z-sum Hamiltonian
        ham_diag: optional explicit Hamiltonian diagonal
    Returns:
        _TQHEACircuit instance (nn.Module)
    """
    depth = net_size[0]
    linear_depth = net_size[1]
    block_configs = [(num_qubits, linear_depth)] * depth
    if ham_diag is not None:
        return _TQHEACircuit(num_qubits, block_configs, ham_diag=ham_diag)

    offset, coeff = _ham_params(num_qubits, ham_bound[0], ham_bound[1])
    return _TQHEACircuit(num_qubits, block_configs,
                         ham_offset=offset, ham_coeff_per_qubit=coeff)
