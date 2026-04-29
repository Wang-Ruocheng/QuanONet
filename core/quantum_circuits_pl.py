"""
PennyLane circuit builders for QuanONet/HEAQNN.

Implements the same Hardware Efficient Ansatz (HEA) as quantum_circuits_tq.py:
data re-uploading RX encoding + alternating RY/RZ/RY ansatz + CNOT ring.
Uses PennyLane default.qubit with PyTorch interface and backprop differentiation.
"""
import numpy as np
import torch
import torch.nn as nn

try:
    import pennylane as qml
except ImportError:
    qml = None

from core.quantum_circuits_tq import _make_block_configs, _ham_params


class _PLHEACircuit(nn.Module):
    """
    A parametric HEA quantum circuit implemented with PennyLane.

    Mirrors _TQHEACircuit exactly: same gate sequence, same CNOT direction,
    same Hamiltonian. Uses default.qubit with backprop for gradient computation.

    Args:
        n_wires: number of qubits
        block_configs: list of (n_encode_params, linear_depth) tuples
        ham_offset: constant offset of the Hamiltonian
        ham_coeff_per_qubit: coefficient for each Z_i term
        ham_diag: full diagonal of the Hamiltonian (2^n_wires,); overrides above
    """
    def __init__(self, n_wires, block_configs, ham_offset=0.0,
                 ham_coeff_per_qubit=0.0, ham_diag=None):
        super().__init__()
        if qml is None:
            raise ImportError(
                "pennylane is required but not installed. "
                "Install with: pip install pennylane"
            )
        self.n_wires = n_wires
        self.block_configs = block_configs

        total_ansatz_blocks = sum(ld for _, ld in block_configs)
        self.ansatz_weights = nn.Parameter(
            torch.empty(total_ansatz_blocks, 3, n_wires)
        )
        nn.init.uniform_(self.ansatz_weights, -np.pi, np.pi)

        if ham_diag is not None:
            self.register_buffer('ham_diag',
                                 torch.tensor(ham_diag, dtype=torch.float32))
            self.use_full_ham = True
        else:
            self.ham_offset = float(ham_offset)
            self.ham_coeff = float(ham_coeff_per_qubit)
            self.use_full_ham = False

        # Build QNode once at construction time
        dev = qml.device('default.qubit', wires=n_wires)

        @qml.qnode(dev, interface='torch', diff_method='backprop')
        def _circuit(x, weights):
            """
            x:       (total_encode_params,) — single-sample encoding input
            weights: (total_ansatz_blocks, 3, n_wires) — trainable parameters
            Returns: state vector (2^n_wires,) complex
            """
            param_col = 0
            ansatz_block = 0
            for n_encode, linear_depth in block_configs:
                for j in range(n_encode):
                    wire = j % n_wires
                    qml.RX(x[param_col], wires=wire)
                    param_col += 1
                for _ in range(linear_depth):
                    w = weights[ansatz_block]
                    for i in range(n_wires):
                        qml.RY(w[0, i], wires=i)
                        qml.RZ(w[1, i], wires=i)
                        qml.RY(w[2, i], wires=i)
                    for i in range(n_wires):
                        # MindQuantum: CNOT.on(target=i, control=(i+1)%n)
                        qml.CNOT(wires=[(i + 1) % n_wires, i])
                    ansatz_block += 1
            return qml.state()

        self._circuit = _circuit

    def forward(self, x):
        """
        Args:
            x: (batch, total_encode_params) float tensor
        Returns:
            (batch, 1) Hamiltonian expectation value
        """
        results = []
        for b in range(x.shape[0]):
            state = self._circuit(x[b], self.ansatz_weights)
            probs = state.abs().pow(2)          # (2^n,) real
            expval = self._ham_expval(probs, x.device)   # scalar ()
            results.append(expval)
        return torch.stack(results, dim=0).unsqueeze(-1)  # (batch, 1)

    def _ham_expval(self, probs, device):
        if self.use_full_ham:
            return (probs * self.ham_diag.to(device)).sum()
        n = self.n_wires
        k = torch.arange(2 ** n, device=device)
        z_sum = torch.zeros(1, device=device)
        for i in range(n):
            sign = (1 - 2 * ((k >> i) & 1).float())
            z_sum = z_sum + (probs * sign).sum()
        return (self.ham_offset + self.ham_coeff * z_sum).squeeze(0)


def build_quanonet_pl(num_qubits, branch_input_size, trunk_input_size, net_size,
                      ham_bound=(-5.0, 5.0), ham_diag=None):
    """
    Build a PennyLane QuanONet circuit.

    Mirrors build_quanonet_tq() from quantum_circuits_tq.py.

    Args:
        num_qubits: number of qubits
        branch_input_size: branch encoder input size
        trunk_input_size: trunk encoder input size
        net_size: (branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth)
        ham_bound: (lower, upper) for the simple Z-sum Hamiltonian
        ham_diag: optional explicit Hamiltonian diagonal
    Returns:
        _PLHEACircuit instance (nn.Module)
    """
    branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth = net_size
    block_configs = _make_block_configs(num_qubits,
                                        trunk_depth, trunk_linear_depth,
                                        branch_depth, branch_linear_depth)
    if ham_diag is not None:
        return _PLHEACircuit(num_qubits, block_configs, ham_diag=ham_diag)
    offset, coeff = _ham_params(num_qubits, ham_bound[0], ham_bound[1])
    return _PLHEACircuit(num_qubits, block_configs,
                         ham_offset=offset, ham_coeff_per_qubit=coeff)


def build_heaqnn_pl(num_qubits, input_size, net_size,
                    ham_bound=(-5.0, 5.0), ham_diag=None):
    """
    Build a PennyLane HEAQNN circuit.

    Mirrors build_heaqnn_tq() from quantum_circuits_tq.py.

    Args:
        num_qubits: number of qubits
        input_size: input size
        net_size: (depth, linear_depth, _, _)
        ham_bound: (lower, upper) for the simple Z-sum Hamiltonian
        ham_diag: optional explicit Hamiltonian diagonal
    Returns:
        _PLHEACircuit instance (nn.Module)
    """
    depth = net_size[0]
    linear_depth = net_size[1]
    block_configs = [(num_qubits, linear_depth)] * depth
    if ham_diag is not None:
        return _PLHEACircuit(num_qubits, block_configs, ham_diag=ham_diag)
    offset, coeff = _ham_params(num_qubits, ham_bound[0], ham_bound[1])
    return _PLHEACircuit(num_qubits, block_configs,
                         ham_offset=offset, ham_coeff_per_qubit=coeff)
