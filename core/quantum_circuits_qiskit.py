"""
Qiskit circuit builders for QuanONet/HEAQNN.

Uses qiskit-machine-learning EstimatorQNN + TorchConnector to produce
torch.nn.Module wrappers around parametric quantum circuits.

Requirements:
    pip install qiskit qiskit-machine-learning

Compatible with qiskit >= 1.0 and qiskit-machine-learning >= 0.7.
"""
import numpy as np

try:
    from qiskit.circuit import QuantumCircuit, ParameterVector
    from qiskit.quantum_info import SparsePauliOp
    _qiskit_available = True
except ImportError:
    _qiskit_available = False

try:
    from qiskit_machine_learning.neural_networks import EstimatorQNN
    from qiskit_machine_learning.connectors import TorchConnector
    _qiskit_ml_available = True
except ImportError:
    _qiskit_ml_available = False


def _check_imports():
    if not _qiskit_available:
        raise ImportError("qiskit is required. Install with: pip install qiskit")
    if not _qiskit_ml_available:
        raise ImportError(
            "qiskit-machine-learning is required. "
            "Install with: pip install qiskit-machine-learning"
        )


def _build_hea_circuit(n_qubits, n_encode_params, n_ansatz_params):
    """
    Build a parametric HEA circuit with separate input and weight parameters.

    Structure per block:
        RX(x_i) encoding on each qubit (data re-uploading)
        + linear_depth × [RY(θ)/RZ(θ)/RY(θ) per qubit + CNOT ring]

    Args:
        n_qubits: number of qubits
        n_encode_params: total number of encoding (input) parameters
        n_ansatz_params: total number of trainable weight parameters
    Returns:
        (qc, input_params, weight_params)
    """
    input_params = ParameterVector('x', n_encode_params)
    weight_params = ParameterVector('θ', n_ansatz_params)
    qc = QuantumCircuit(n_qubits)
    return qc, input_params, weight_params


def _fill_hea_circuit(qc, input_params, weight_params, n_qubits, block_configs):
    """
    Fill the quantum circuit with HEA gates according to block_configs.

    Args:
        qc: QuantumCircuit to fill
        input_params: ParameterVector for encoding
        weight_params: ParameterVector for ansatz
        n_qubits: number of qubits
        block_configs: list of (n_encode_per_block, linear_depth) tuples
    """
    enc_idx = 0
    ans_idx = 0

    for n_encode, linear_depth in block_configs:
        # Encoding: RX on each qubit up to n_encode params
        for j in range(n_encode):
            wire = j % n_qubits
            qc.rx(input_params[enc_idx], wire)
            enc_idx += 1

        # Ansatz: linear_depth sublayers of RY/RZ/RY + CNOT ring
        for _ in range(linear_depth):
            for i in range(n_qubits):
                qc.ry(weight_params[ans_idx], i); ans_idx += 1
                qc.rz(weight_params[ans_idx], i); ans_idx += 1
                qc.ry(weight_params[ans_idx], i); ans_idx += 1
            for i in range(n_qubits):
                qc.cx(i, (i + 1) % n_qubits)


def _build_hamiltonian_op(n_qubits, lower_bound=-5.0, upper_bound=5.0, ham_diag=None):
    """
    Build a Qiskit SparsePauliOp for the Hamiltonian.

    For ham_diag=None: H = offset*I + coeff_per_qubit * sum_i Z_i
    For ham_diag: reconstruct from diagonal via Walsh-Hadamard (calls ham_diag_to_operator logic).

    Note: Qiskit Pauli string convention is big-endian (qubit n-1 ... qubit 0,
    left to right). Z on qubit i → 'I'*(n-1-i) + 'Z' + 'I'*i.
    """
    n = n_qubits
    if ham_diag is None:
        coff = upper_bound - lower_bound
        offset = lower_bound + coff / 2.0
        coeff_per_qubit = coff / 2.0 / n
        terms = []
        if abs(offset) > 1e-12:
            terms.append(('I' * n, offset))
        for i in range(n):
            # qubit i in Qiskit big-endian: Z at position (n-1-i) from left
            pauli = 'I' * (n - 1 - i) + 'Z' + 'I' * i
            terms.append((pauli, coeff_per_qubit))
        return SparsePauliOp.from_list(terms)
    else:
        # Walsh-Hadamard transform to get Pauli coefficients (same as ham_diag_to_operator)
        diag = np.asarray(ham_diag, dtype=float)
        wh = np.array([[(-1) ** bin(i & j).count('1') for j in range(2 ** n)]
                       for i in range(2 ** n)])
        coeffs = wh @ diag / 2 ** n
        terms = []
        for idx, c in enumerate(coeffs):
            if abs(c) < 1e-12:
                continue
            pauli_chars = ['I'] * n
            for q in range(n):
                if (idx >> q) & 1:
                    # qubit q → position (n-1-q) in Qiskit big-endian string
                    pauli_chars[n - 1 - q] = 'Z'
            terms.append((''.join(pauli_chars), c))
        if not terms:
            terms = [('I' * n, 0.0)]
        return SparsePauliOp.from_list(terms)


def _make_torch_connector(n_qubits, block_configs, ham_op, initial_weights=None):
    """
    Build an EstimatorQNN and wrap it with TorchConnector.

    Returns a torch.nn.Module whose forward(x) computes the Hamiltonian
    expectation value for a batch of input encodings x of shape
    (batch, n_encode_params).
    """
    n_encode_params = sum(n_enc for n_enc, _ in block_configs)
    n_ansatz_params = sum(3 * n_qubits * ld for _, ld in block_configs)

    qc, input_params, weight_params = _build_hea_circuit(
        n_qubits, n_encode_params, n_ansatz_params
    )
    _fill_hea_circuit(qc, input_params, weight_params, n_qubits, block_configs)

    qnn = EstimatorQNN(
        circuit=qc,
        observables=ham_op,
        input_params=list(input_params),
        weight_params=list(weight_params),
    )

    if initial_weights is None:
        initial_weights = (np.random.uniform(-np.pi, np.pi, n_ansatz_params)
                           .astype(np.float32))

    return TorchConnector(qnn, initial_weights=initial_weights)


def build_quanonet_qiskit(num_qubits, branch_input_size, trunk_input_size, net_size,
                          ham_bound=(-5.0, 5.0), ham_diag=None):
    """
    Build a Qiskit-based QuanONet quantum layer.

    Mirrors QuanONet_build() + circuit2network() from quantum_circuits.py,
    returning a torch.nn.Module via TorchConnector.

    Args:
        num_qubits: number of qubits
        branch_input_size: branch encoder input size (determines circuit capacity check)
        trunk_input_size: trunk encoder input size
        net_size: (branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth)
        ham_bound: (lower, upper) for simple Z-sum Hamiltonian (ignored if ham_diag set)
        ham_diag: optional explicit Hamiltonian diagonal of length 2^num_qubits
    Returns:
        torch.nn.Module (TorchConnector wrapping EstimatorQNN)
    """
    _check_imports()
    branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth = net_size

    # Block configs: trunk blocks first, then branch blocks (matches MindQuantum order)
    block_configs = (
        [(num_qubits, trunk_linear_depth)] * trunk_depth +
        [(num_qubits, branch_linear_depth)] * branch_depth
    )

    ham_op = _build_hamiltonian_op(num_qubits, ham_bound[0], ham_bound[1], ham_diag)
    return _make_torch_connector(num_qubits, block_configs, ham_op)


def build_heaqnn_qiskit(num_qubits, input_size, net_size,
                        ham_bound=(-5.0, 5.0), ham_diag=None):
    """
    Build a Qiskit-based HEAQNN quantum layer.

    Args:
        num_qubits: number of qubits
        input_size: input size (determines circuit capacity check)
        net_size: (depth, linear_depth, _, _) — first two elements used
        ham_bound: (lower, upper) for simple Z-sum Hamiltonian
        ham_diag: optional explicit Hamiltonian diagonal
    Returns:
        torch.nn.Module (TorchConnector wrapping EstimatorQNN)
    """
    _check_imports()
    depth = net_size[0]
    linear_depth = net_size[1]
    block_configs = [(num_qubits, linear_depth)] * depth
    ham_op = _build_hamiltonian_op(num_qubits, ham_bound[0], ham_bound[1], ham_diag)
    return _make_torch_connector(num_qubits, block_configs, ham_op)
