"""
Main model definitions for QuanONet.
"""

# This file contains the main model definitions for QuanONet, including the QuanONet and HEAQNN classes, which implement quantum neural networks with optional trainable frequency support.
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Parameter

# Import local modules (must be in the same directory)
try:
    from layers import *
    from quantum_circuits import circuit2network, QuanONet_build, HEAQNNwork_build
except ImportError:
    # If direct import fails, try relative import
    from .layers import *
    from .quantum_circuits import circuit2network, QuanONet_build, HEAQNNwork_build


class QuanONet(nn.Cell):
    """Quantum Operator Network with optional trainable frequency support."""
    
    def __init__(self, num_qubits, branch_input_size, trunk_input_size, net_size, 
                 ham, scale_coeff=1, if_trainable_freq=False):
        super(QuanONet, self).__init__()
        (self.branch_depth, self.branch_linear_depth, 
         self.trunk_depth, self.trunk_linear_depth) = net_size
        
        self.if_trainable_freq = if_trainable_freq
        
        QuanONet_circuit = QuanONet_build(
            num_qubits, branch_input_size, trunk_input_size, net_size
        )
        self.circuit = QuanONet_circuit
        self.QuanONet = circuit2network(QuanONet_circuit, ham)
        
        if self.if_trainable_freq:
            # Using trainable linear layers
            self.branch_LinearLayer = CombinedNet(
                RepeatLayer(self.branch_depth * num_qubits), 
                LinearLayer(self.branch_depth * num_qubits, scale_coeff)
            )
            self.trunk_LinearLayer = CombinedNet(
                RepeatLayer(self.trunk_depth * num_qubits), 
                LinearLayer(self.trunk_depth * num_qubits, scale_coeff)
            )
        else:
            # Using fixed scaling layers
            self.branch_ScaleLayer = CombinedNet(
                CoeffLayer(branch_input_size, scale_coeff), 
                RepeatLayer(self.branch_depth * num_qubits)
            )
            self.trunk_ScaleLayer = CombinedNet(
                CoeffLayer(trunk_input_size, scale_coeff), 
                RepeatLayer(self.trunk_depth * num_qubits)
            )
        
    def construct(self, input):
        branch_input = input[0]
        trunk_input = input[1]
        
        if self.if_trainable_freq:
            branch_input = self.branch_LinearLayer(branch_input)
            trunk_input = self.trunk_LinearLayer(trunk_input)
        else:
            # Original QuanONet processing method
            branch_input = self.branch_ScaleLayer(branch_input)
            trunk_input = self.trunk_ScaleLayer(trunk_input)
        
        input = mnp.concatenate((branch_input, trunk_input), axis=1)
        return self.QuanONet(input) 


class HEAQNN(nn.Cell):
    """Hardware Efficient Ansatz Quantum Neural Network with optional trainable frequency support."""
    
    def __init__(self, num_qubits, branch_input_size, trunk_input_size, net_size, 
                 ham, scale_coeff=1, if_trainable_freq=False):
        super(HEAQNN, self).__init__()
        (self.depth, self.linear_depth) = net_size
        
        self.if_trainable_freq = if_trainable_freq
        self.total_input_size = branch_input_size + trunk_input_size
        
        HEAQNN_circuit = HEAQNNwork_build(
            num_qubits, self.total_input_size, self.depth, self.linear_depth
        )
        self.HEAQNN = circuit2network(HEAQNN_circuit, ham)
        
        if self.if_trainable_freq:
            # Using trainable linear layers
            self.LinearLayer = CombinedNet(
                RepeatLayer(self.depth * num_qubits), 
                LinearLayer(self.depth * num_qubits, scale_coeff)
            )
        else:
            # Using fixed scaling layers
            self.CoeffLayer = CoeffLayer(self.total_input_size, scale_coeff)
            self.RepeatLayer = RepeatLayer(self.depth * num_qubits)
    
    def construct(self, input):
        branch_input = input[0]
        trunk_input = input[1]
        input = mnp.concatenate((branch_input, trunk_input), axis=1)
        
        if self.if_trainable_freq:
            input = self.LinearLayer(input)
        else:
            # Original HEAQNN processing method
            input = self.CoeffLayer(input)
            input = self.RepeatLayer(input)
        
        return self.HEAQNN(input)


class FNN(nn.Cell):
    """Feedforward Neural Network for comparison."""
    
    def __init__(self, branch_input_size, trunk_input_size, output_size, net_size, 
                 activation=nn.Tanh()):
        super(FNN, self).__init__()
        (self.hidden_layer_depth, self.hidden_layer_width) = net_size
        
        self.FNN = FNNLayer(
            branch_input_size + trunk_input_size, output_size, 
            self.hidden_layer_width, self.hidden_layer_depth, activation
        )
    
    def construct(self, input):
        branch_input = input[0]
        trunk_input = input[1]
        input = mnp.concatenate((branch_input, trunk_input), axis=1)
        return self.FNN(input)


class DeepONet(nn.Cell):
    """Deep Operator Network."""
    
    def __init__(self, branch_input_size, trunk_input_size, net_size, activation=nn.Tanh()):
        super(DeepONet, self).__init__()
        (self.branch_depth, self.branch_width, 
         self.trunk_depth, self.trunk_width) = net_size
        
        self.branch_net = FNNLayer(
            branch_input_size, self.branch_width, 
            self.branch_width, self.branch_depth + 1, activation
        )
        self.trunk_net = FNNLayer(
            trunk_input_size, self.trunk_width, 
            self.trunk_width, self.trunk_depth + 1, activation
        )
        
        self.sum_layer = SumLayer()
        self.bias = Parameter(ms.Tensor(0.0, dtype=ms.float32), name="bias")
    
    def construct(self, input):
        branch_input = input[0]
        trunk_input = input[1]
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        output = branch_output * trunk_output
        output = self.sum_layer(output).unsqueeze(1) + self.bias
        return output
