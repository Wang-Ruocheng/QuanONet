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
