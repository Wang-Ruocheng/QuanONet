"""
Quantum circuit construction for QuanONet.
"""

# This file contains functions to generate quantum circuits, including encoding layers, entanglement layers, ansatz layers, and Hamiltonian generation for use in QuanONet. It also includes utility functions for parameter management and circuit construction.
import numpy as np
import itertools

# MindQuantum imports
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import RX, RY, RZ, CNOT, Z
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.simulator import Simulator
from mindquantum.framework import MQLayer
from mindspore.common.initializer import Uniform

def zero_state_hamiltonian(num_qubits, lower_bound=0, upper_bound=1):
    """Create a Hamiltonian for zero state preparation."""
    coff = upper_bound - lower_bound
    zero_state_op = QubitOperator('', lower_bound)
    for ops_seq in itertools.product(['I', 'Z'], repeat=num_qubits):
        term = ' '.join(f'{op}{i}' for i, op in enumerate(ops_seq) if op != 'I')
        zero_state_op += QubitOperator(term, 1/2**num_qubits*coff)
    ham = Hamiltonian(zero_state_op)
    return ham


def generate_simple_hamiltonian(num_qubits, lower_bound=None, upper_bound=None, pauli='Z'):
    """Generate a simple Hamiltonian with Pauli operators."""
    if lower_bound is None:
        lower_bound = -num_qubits
    if upper_bound is None:
        upper_bound = num_qubits
    coff = upper_bound - lower_bound
    ham = QubitOperator('', lower_bound + coff/2)
    for i in range(num_qubits):
        ham += QubitOperator(f'{pauli}{i}', coff/2/num_qubits)
    ham = Hamiltonian(ham)
    return ham


def ham_diag_to_operator(diag_elements, num_qubits):
    """
    将32维对角矩阵(5比特系统)展开为Pauli-Z项的QubitOperator。
    diag_vec: 长度为32的对角线值
    返回: QubitOperator
    """
    n = num_qubits
    diag_elements = np.asarray(diag_elements)
    # Walsh-Hadamard变换
    coeffs = np.dot(np.array([[(-1)**(bin(i & j).count('1')) for j in range(2**n)] for i in range(2**n)]), diag_elements) / 2**n
    op = QubitOperator('', 0)
    for idx, c in enumerate(coeffs):
        if abs(c) < 1e-12:
            continue
        # idx转Pauli字符串
        pauli_str = []
        for q in range(n):
            if (idx >> q) & 1:
                pauli_str.append(f'Z{q}')
        term = ' '.join(pauli_str)
        op += QubitOperator(term, c)
    op = Hamiltonian(op)
    return op


def generate_ham_diag(num_qubits, rank, seed=None):
    """
    Generate a 1D array of length 2**num_qubit with 'rank' nonzero entries.
    The number of 1's and -1's are both rank//2 (rank must be even).
    """
    length = 2 ** num_qubits
    assert rank <= length, "rank cannot be greater than array length"
    assert rank % 2 == 0, "rank must be even for equal number of 1's and -1's"
    if seed is not None:
        np.random.seed(seed)
    arr = np.zeros(length)
    # Randomly select 'rank' unique positions
    idx = np.random.choice(length, rank, replace=False)
    num_ones = rank // 2
    num_neg_ones = rank // 2
    values = np.array([1] * num_ones + [-1] * num_neg_ones)
    np.random.shuffle(values)
    arr[idx] = values
    return 5*arr


def generate_ham_diag_rank1(num_qubits, seed=None):
    """
    随机在2**num_qubits个位置中产生一个1,其余全为0,返回该数组乘10减5
    """
    length = 2 ** num_qubits
    if seed is not None:
        np.random.seed(seed)
    arr = np.zeros(length)
    idx = np.random.choice(length, 1, replace=False)  # 随机选一个位置
    arr[idx[0]] = 1
    return arr * 10 - 5

def generate_ham_diag_diffspectrum(num_qubits, rank, seed=None):
    """
    若rank为1,随机在一个位置生成1,其余全为0,返回数组*10-5;
    若rank为2;随机选两个位置分别为5和-5,其余为0;
    其他rank,随机选rank个位置,前两个分别为5和-5,剩下的为-5到5的随机值,其余为0。
    """
    length = 2 ** num_qubits
    assert rank <= length, "rank cannot be greater than array length"
    if seed is not None:
        np.random.seed(seed)
    arr = np.zeros(length)
    idx = np.random.choice(length, rank, replace=False)
    if rank == 1:
        arr[idx[0]] = 1
        return arr * 10 - 5
    elif rank == 2:
        arr[idx[0]] = 5
        arr[idx[1]] = -5
        return arr
    else:
        arr[idx[0]] = 5
        arr[idx[1]] = -5
        # 剩下的rank-2个位置赋值为-5到5的随机值（不含5和-5）
        arr[idx[2:]] = np.random.uniform(-5, 5, rank - 2)
        return arr


def generate_diag_from_rank(rank, num_qubits):
    """
    根据秩生成对角哈密顿量:一半为1,一半为-1,其余为0
    rank: 非零元素个数(必须为偶数且不超过2^num_qubits)
    num_qubits: 比特数
    返回:长度为2^num_qubits的对角线数组
    """
    dim = 2 ** num_qubits
    if rank > dim:
        raise ValueError("rank不能大于哈密顿量维度")
    if rank % 2 != 0:
        raise ValueError("rank必须为偶数")
    diag = [0] * dim
    half = rank // 2
    diag[:half] = [5] * half
    diag[half:rank] = [-5] * half
    # 其余为0
    return diag


def Encode_layer(num_qubits, input_size, e_name_list, PauliRotGate=RX):
    """Create encoding layer for quantum circuit."""
    circ = Circuit()
    if e_name_list != []:
        for i in range(num_qubits):
            circ += PauliRotGate(f'{e_name_list[i%len(e_name_list)]}_q{i}').on(i)
    circ.as_encoder()
    return circ
    

def Entangle_layer(num_qubits):
    """Create entanglement layer with CNOT gates."""
    circ = Circuit()
    if num_qubits > 1:
        for i in range(num_qubits):
            circ += CNOT.on(i, (i+1)%num_qubits)
    circ.as_ansatz()
    return circ


def Ansatz_layer(num_qubits, a_name_list, PauliRotGate=RY):
    """Create ansatz layer with parameterized rotations."""
    circ = Circuit()
    for i in range(num_qubits):
        circ += PauliRotGate(a_name_list[i]).on(i)
    circ.as_ansatz()
    return circ


def params_update(a_num_list, num_para, add_num):
    """Update parameter list."""
    for i in range(add_num):
        a_num_list.append(f'para{num_para}')
        num_para += 1
    return a_num_list, num_para


def add_parameter(num_list, num_para, num):
    """Add parameters to list."""
    for _ in range(num):
        num_list.append(f'para{num_para}')
        num_para += 1
    return num_list, num_para


def QuanONet_build(num_qubits, branch_input_size, trunk_input_size, net_size, if_print_circuit=False):
    """Build quantum circuit for QuanONet."""
    circuit = Circuit()
    (branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth) = net_size
    a_num_list = []
    trunk = Circuit()
    branch = Circuit()
    num_para = 0
    
    # Build branch network
    for j in range(branch_depth):
        e_num_list = [f"xi_l{j}" for _ in range(num_qubits)]
        branch += Encode_layer(num_qubits, num_qubits, e_num_list, PauliRotGate=RX)
        for _ in range(branch_linear_depth):
            a_num_list, num_para = params_update(a_num_list, num_para, 3*num_qubits)
            branch += Ansatz_layer(num_qubits, a_num_list[-3*num_qubits:-2*num_qubits], RY)
            branch += Ansatz_layer(num_qubits, a_num_list[-2*num_qubits:-num_qubits], RZ)
            branch += Ansatz_layer(num_qubits, a_num_list[-num_qubits:], RY)
            branch += Entangle_layer(num_qubits)
    
    # Build trunk network
    for j in range(trunk_depth):
        e_num_list = [f"nu_l{j}" for _ in range(num_qubits)]
        for _ in range(trunk_linear_depth):
            a_num_list, num_para = params_update(a_num_list, num_para, 3*num_qubits)
            trunk += Entangle_layer(num_qubits)
            trunk += Ansatz_layer(num_qubits, a_num_list[-3*num_qubits:-2*num_qubits], RY)
            trunk += Ansatz_layer(num_qubits, a_num_list[-2*num_qubits:-num_qubits], RZ)
            trunk += Ansatz_layer(num_qubits, a_num_list[-num_qubits:], RY)
        trunk += Encode_layer(num_qubits, num_qubits, e_num_list)
    
    circuit = branch + trunk
    
    if num_qubits * branch_depth < branch_input_size or num_qubits * trunk_depth < trunk_input_size:
        print("The number of encoder params is not enough for the input size.")
    
    if if_print_circuit:
        circuit.summary()
    
    return circuit   


def HEAQNNwork_build(num_qubits, input_size, depth, linear_depth):
    """Build Hardware Efficient Ansatz Quantum Neural Network."""
    circuit = Circuit()
    a_num_list = []
    num_para = 0
    
    for j in range(depth):
        e_num_list = [f"x_l{j}" for _ in range(num_qubits)]
        circuit += Encode_layer(num_qubits, num_qubits, e_num_list)
        for _ in range(linear_depth):
            a_num_list, num_para = params_update(a_num_list, num_para, 3*num_qubits)
            circuit += Ansatz_layer(num_qubits, a_num_list[-3*num_qubits:-2*num_qubits], RY)
            circuit += Ansatz_layer(num_qubits, a_num_list[-2*num_qubits:-num_qubits], RZ)
            circuit += Ansatz_layer(num_qubits, a_num_list[-num_qubits:], RY)
            circuit += Entangle_layer(num_qubits)
    
    if num_qubits * depth < input_size:
        print("The number of encoder params is not enough for the input size.")
    
    return circuit  


def circuit2network(circuit, ham):
    """Convert quantum circuit to MindQuantum network layer."""
    sim = Simulator('mqvector', circuit.n_qubits)
    grad_ops = sim.get_expectation_with_grad(ham, circuit)
    return MQLayer(grad_ops, Uniform(np.pi))
