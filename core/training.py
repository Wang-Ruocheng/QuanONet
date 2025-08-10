"""
Training utilities and network processing functions.
"""

# This file contains utilities for training neural networks, including logging, checkpoint management, and network processing.
import numpy as np
import os
import glob
import logging
import sys
import mindspore as ms
import mindspore.nn as nn
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.nn import MSELoss, WithLossCell, TrainOneStepCell
from tqdm import tqdm


class StreamToLogger(object):
    """Redirect stdout/stderr to logger."""
    
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


class NetworkProcess:
    """Network processing utilities."""
    
    def __init__(self, network):
        self.Network = network

    def train(self, learning_rate, num_epochs, train_input, train_output, 
              target_error=1e-10, loss_show=False):
        """Train the network."""
        opti = nn.Adam(self.Network.trainable_params(), learning_rate=learning_rate)
        net_with_loss = nn.WithLossCell(self.Network, self.loss)
        train_net = TrainOneStepCell(net_with_loss, opti)
        
        for i in tqdm(range(num_epochs), desc="Training Progress"):
            res = train_net(train_input, train_output)
            if loss_show and i % 20 == 0:
                print(f"Epoch {i + 1}/{num_epochs}, Loss {res}")
            if res < target_error:
                break
        
        print(f"Training finished, train loss: {res}")

    def get_net(self):
        """Get the network."""
        return self.Network
    
    def get_params_num(self):
        """Get number of trainable parameters."""
        total_params = 0
        for param in self.Network.trainable_params():
            total_params += np.prod(param.shape)
        return total_params


def TrainNetwork(Network, train_input, train_output, test_input, test_output, 
                num_epochs, lr_schedule, batch_size, output_file, problem, port, 
                device_id, name, num_qubits, net_size, scale_coeff, seed_num, 
                if_save, if_keep, if_batch, checkpoint_file_name, device_type, 
                loss_fn=MSELoss()):
    """
    Comprehensive training function for networks.
    
    Args:
        Network: The neural network to train
        train_input: Training input data (branch, trunk)
        train_output: Training output data
        test_input: Test input data (branch, trunk)
        test_output: Test output data
        num_epochs: Number of training epochs
        lr_schedule: Learning rate schedule
        batch_size: Batch size for training
        output_file: Output directory for logs
        problem: Problem type string
        port: Port number
        device_id: Device ID
        name: Model name
        num_qubits: Number of qubits (for quantum models)
        net_size: Network architecture size
        scale_coeff: Scaling coefficient
        seed_num: Random seed number
        if_save: Whether to save logs
        if_keep: Whether to keep existing checkpoints
        if_batch: Whether to use batch training
        checkpoint_file_name: Checkpoint file path
        device_type: Device type ("quantum" or "classical")
        loss_fn: Loss function to use
    """
    # Setup logging
    if device_type == "quantum":
        output_path = f"{output_file}/{problem}/{port}_{device_id}_{name}_{num_qubits}_{net_size}_{scale_coeff}_seed{seed_num}_loss.log"
    else:
        output_path = f"{output_file}/{problem}/{port}_{device_id}_{name}_{num_qubits}_{net_size}_seed{seed_num}_loss.log"
    
    if if_save:
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.basicConfig(filename=output_path, level=logging.INFO, format='%(message)s')
            logger = logging.getLogger()
            sys.stdout = StreamToLogger(logger, logging.INFO)
            sys.stderr = StreamToLogger(logger, logging.ERROR)
        except Exception as e:
            print(f"Failed to set up logging: {e}")
    
    # Load checkpoint if exists
    if os.path.exists(checkpoint_file_name) and if_keep:
        param_dict = load_checkpoint(checkpoint_file_name)
        load_param_into_net(Network, param_dict)
    
    # Setup training
    opti = nn.Adam(Network.trainable_params(), learning_rate=lr_schedule)
    net_with_loss = nn.WithLossCell(Network, loss_fn)
    train_net = TrainOneStepCell(net_with_loss, opti)
    
    NP = NetworkProcess(Network)
    num_batches_train = max(len(train_input[0]) // batch_size, 1)
    batch_size_test = min(batch_size, len(test_input[0]))
    num_batches_test = max(len(test_input[0]) // batch_size_test, 1)
    
    # Check for existing training logs
    path_begin = f"{output_file}/{problem}/"
    if device_type == "quantum":
        path_end = f"{name}_{num_qubits}_{net_size}_{scale_coeff}_seed{seed_num}_loss.log"
    else:
        path_end = f"{name}_{num_qubits}_{net_size}_seed{seed_num}_loss.log"
    
    log_files = glob.glob(f"{path_begin}*{path_end}")
    train_losses = []
    
    for path in log_files:
        with open(path, 'r') as file:
            for line in file:
                if 'Epoch' in line:
                    parts = line.strip().split(':')
                    if len(parts) == 2:
                        losses = parts[1].split(',')
                        if len(losses) == 2:
                            train_loss = float(losses[0].strip())
                            test_loss = float(losses[1].strip())
                            train_losses.append(train_loss)
    
    start_epoch = len(train_losses) if if_keep else 0
    
    if start_epoch == 0:
        print(f"Training {name} with size {net_size}, params num: {NP.get_params_num()}")
    
    # Training loop
    for epoch in range(start_epoch, num_epochs):
        ave_res = 0
        
        if if_batch:
            permutation = np.random.permutation(train_input[0].shape[0])
        else:
            permutation = np.array(range(train_input[0].shape[0]))
        
        permutation = ms.Tensor(permutation, dtype=ms.int32)
        
        # Training batches
        for batch in range(num_batches_train):
            start = batch * batch_size
            end = start + batch_size
            batch_indices = permutation[start:end]
            batch_indices = ms.Tensor(batch_indices, dtype=ms.int32)
            
            branch_batch_input = train_input[0][batch_indices]
            trunk_batch_input = train_input[1][batch_indices]
            batch_input = (branch_batch_input, trunk_batch_input)
            batch_output = train_output[batch_indices]
            
            res = train_net(batch_input, batch_output)
            ave_res += res.asnumpy()
        
        # Test evaluation
        total_test_loss = 0
        for batch in range(num_batches_test):
            start = batch * batch_size_test
            end = start + batch_size_test
            test_branch_batch_input = test_input[0][start:end]
            test_trunk_batch_input = test_input[1][start:end]
            test_batch_input = (test_branch_batch_input, test_trunk_batch_input)
            test_batch_output = test_output[start:end]
            
            test_loss = loss_fn(Network(test_batch_input), test_batch_output)
            total_test_loss += test_loss.asnumpy()
        
        average_test_loss = total_test_loss / num_batches_test
        average_train_loss = ave_res / num_batches_train
        
        print(f"Epoch {epoch}: {average_train_loss}, {average_test_loss}")
        
        if if_save:
            save_checkpoint(Network, checkpoint_file_name)
