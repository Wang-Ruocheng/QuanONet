"""
Main model definitions for QuanONet.
"""

# This file contains the main model definitions for QuanONet, including the QuanONet and HEAQNN classes, which implement quantum neural networks with optional trainable frequency support.
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore import Parameter
from scipy.interpolate import interp1d

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

        self.PostprocessLayer = LinearLayer(1, 1, initial_bias_range=0)
        
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
        output = self.QuanONet(input) 
        output = self.PostprocessLayer(output)
        return output


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


class PINN(nn.Cell):
    """Physics-Informed wrapper around HEAQNN that computes second derivative
    with respect to the last input dimension and returns PI losses.

    This version creates a zero `branch` input internally so we can wrap
    HEAQNN as a function of only the trunk input and compute derivatives
    w.r.t. the last trunk dimension.

    For robustness across MindSpore versions we approximate the second
    derivative numerically using a central finite difference. This avoids
    nested GradOperation issues ("bprop_cut's bprop not defined").
    """

    def __init__(self, model: nn.Cell, trunk_input_size: int, operator: str):
        super(PINN, self).__init__()
        self.model = model
        self.trunk_input_size = trunk_input_size
        self.operator = operator


        # MSE
        self.loss = nn.MSELoss()
    def piloss(self, model_output, first_grad, second_grad, branch_value_at_trunks):
        if self.operator.lower() == 'inverse':
            return self.loss(first_grad[0], branch_value_at_trunks)
        elif self.operator.lower() == 'homogeneous':
            return self.loss(first_grad[0], branch_value_at_trunks + model_output)
        elif self.operator.lower() == 'nonlinear':
            return self.loss(first_grad[0], branch_value_at_trunks**3 + model_output)
        elif self.operator.lower() == 'rdiffusion':
            alpha = 0.01
            k = 0.01
            return self.loss(first_grad[1], alpha * second_grad[0] + k * model_output**2 + branch_value_at_trunks)
        elif self.operator.lower() == 'advection':
            c = 1.0
            return self.loss(first_grad[1], -c * first_grad[0])
        elif self.operator.lower() == 'darcy':
            K = 0.1
            f = -1.0
            return self.loss(-K * (second_grad[0] + second_grad[1]), f)
        else:
            raise ValueError(f"Unknown operator {self.operator}")
            

    def _first_derivative_dim(self, x, dim=-1, h=1e-3):
        branch_input, trunk_input = x
        n, d = trunk_input.shape[0], trunk_input.shape[1]
        if dim < 0:
            dim = d + dim
        delta = mnp.zeros((n, d), dtype=trunk_input.dtype)
        delta[:, dim] = h
        delta = ms.Tensor(delta.asnumpy())
        trunk_plus = trunk_input + delta
        trunk_minus = trunk_input - delta
        x_plus = (branch_input, trunk_plus)
        x_minus = (branch_input, trunk_minus)
        u_plus = self.model(x_plus)
        u_minus = self.model(x_minus)
        h2 = ms.Tensor(np.array(2.0 * h, dtype=np.float32))
        u_x = (u_plus - u_minus) / h2
        return u_x

    def _second_derivative_dim(self, x, dim=-1, h=1e-3):
        branch_input, trunk_input = x
        n, d = trunk_input.shape[0], trunk_input.shape[1]
        if dim < 0:
            dim = d + dim
        delta = mnp.zeros((n, d), dtype=trunk_input.dtype)
        delta[:, dim] = h
        delta = ms.Tensor(delta.asnumpy())
        trunk_plus = trunk_input + delta
        trunk_minus = trunk_input - delta
        x_plus = (branch_input, trunk_plus)
        x_minus = (branch_input, trunk_minus)
        u = self.model(x)
        u_plus = self.model(x_plus)
        u_minus = self.model(x_minus)
        h2 = ms.Tensor(np.array(h * h, dtype=np.float32))
        u_xx = (u_plus - 2.0 * u + u_minus) / h2
        return u_xx
    
    def net_first_grad_last_trunk_size(self, x, h=1e-3):
        return mnp.stack([self._first_derivative_dim(x, dim=i - self.trunk_input_size, h=h) for i in range(self.trunk_input_size)])

    def net_second_grad_last_trunk_size(self, x, h=1e-3):
        return mnp.stack([self._second_derivative_dim(x, dim=i-self.trunk_input_size, h=h) for i in range(self.trunk_input_size)])

    def construct(self, x, target):
        branch_input, trunk_input = x

        model_output = self.model(x)

        mse = self.loss(model_output, target)
        first_grad = self.net_first_grad_last_trunk_size(x).squeeze(-1)
        second_grad = self.net_second_grad_last_trunk_size(x).squeeze(-1) if self.operator.lower() in ['rdiffusion', 'darcy'] else None
        trunk_values = trunk_input.asnumpy().flatten() 
        branch_values = branch_input.asnumpy()
        locations = np.linspace(0, 1, num=branch_values.shape[1])

        branch_value_at_trunks = []
        for i in range(branch_values.shape[0]):
            interp_func = interp1d(locations, branch_values[i], fill_value="extrapolate")
            branch_value_at_trunk = interp_func(trunk_values[i])
            branch_value_at_trunks.append(branch_value_at_trunk)
        
        branch_value_at_trunks = ms.Tensor(np.array(branch_value_at_trunks, dtype=np.float32))
        grad_diff = self.piloss(model_output, first_grad, second_grad, branch_value_at_trunks)

        total_loss = mse + grad_diff*0.01
        return total_loss