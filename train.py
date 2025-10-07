#!/usr/bin/env python3
"""
General ODE Operator QuanONet Training Script

This script supports multiple operator problems:
1. Inverse: du/dx = u0(x) (Integral operator)
2. Homogeneous: du/dx = u + u0(x) (Homogeneous operator)
3. Nonlinear: du/dx = u - u0²(x) (Nonlinear operator)
4. Custom: Custom operator (user-defined ODE equation)

Usage:
python train_ODE.py --operator Inverse
python train_ODE.py --operator Homogeneous
nohup python -u train_ODE.py --operator Nonlinear > training_Nonlinear.log 2>&1 &
nohup python -u train_ODE.py --operator Custom --custom_ode "u + 2*u0" --custom_name "MyOperator" > training_Custom.log 2>&1 &

Custom operator examples:
- Linear combination: --custom_ode "0.5*u + u0"
- Quadratic term: --custom_ode "u**2 + u0"
- Trigonometric function: --custom_ode "sin(u) + u0"
- Exponential function: --custom_ode "exp(u) - u0"
"""

import sys
import stat
import os
import json
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.train.serialization import save_checkpoint, load_checkpoint
from tqdm import tqdm
import argparse
from datetime import datetime

# Import necessary modules
from data_utils.data_generation import (
    generate_Inverse_Operator_data,
    generate_Homogeneous_Operator_data,
    generate_Nonlinear_Operator_data,
    generate_ODE_Operator_data
)
from data_utils.data_processing import ODE_encode
from core.models import QuanONet, HEAQNN, FNN, DeepONet
from core.quantum_circuits import generate_simple_hamiltonian, ham_diag_to_operator, generate_diag_from_rank, generate_ham_diag, generate_ham_diag_diffspectrum
from utils.utils import count_parameters

# Set MindSpore context
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

def set_random_seed(seed=None):
    """
    Set random seed to ensure reproducibility.

    Args:
        seed: Random seed value. If None, random seed is not set.
    """
    if seed is not None:
        print(f"Setting random seed: {seed}")
        np.random.seed(seed)
        ms.set_seed(seed)
        # If there are other random libraries, they can be set here
        import random
        random.seed(seed)
    else:
        print("Random seed not set, using system default randomness.")

# Global variable: Supported operator types
OPERATOR_TYPES = {
    'Inverse': generate_Inverse_Operator_data,
    'Homogeneous': generate_Homogeneous_Operator_data,
    'Nonlinear': generate_Nonlinear_Operator_data,
    'Custom': 'custom'  # Custom operator will be handled separately
}

def parse_custom_ode_function(ode_string):
    """
    Parse custom ODE function string.

    Args:
        ode_string: ODE function string, e.g. "u + 2*u0" or "u*u0 - u0**2"

    Returns:
        lambda function
    """
    try:
        # Safe namespace, only allow basic math operations
        safe_namespace = {
            '__builtins__': {},
            'exp': np.exp,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'log': np.log,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'pi': np.pi,
            'e': np.e
        }

        # Create function: lambda u0_fn: lambda x, u: your_equation
        ode_func = lambda u0_fn: lambda x, u: eval(ode_string.replace('u0', 'u0_fn(x)'), safe_namespace, {'u': u, 'u0_fn': u0_fn, 'x': x})
        
        return ode_func
    except Exception as e:
        raise ValueError(f"Unable to parse ODE function '{ode_string}': {e}")

class ODEOperatorSolver:
    """ODE operator problem solver """

    def __init__(self, operator_type='Inverse', config_file=None, custom_ode_func=None, custom_name=None, prefix=None):
        """
        Initialize the solver

        Args:
            operator_type: Operator type ('Inverse', 'Homogeneous', 'Nonlinear', 'Custom')
        Args:
            operator_type: Operator type ('Inverse', 'Homogeneous', 'Nonlinear', 'Custom')
            config_file: Configuration file path. If None, default configuration is used.
            custom_ode_func: Custom ODE function (only used when operator_type='Custom')
            custom_name: Custom operator name (only used when operator_type='Custom')
        """
        if operator_type not in OPERATOR_TYPES:
            raise ValueError(f"Unsupported operator type: {operator_type}. Supported types: {list(OPERATOR_TYPES.keys())}")
        
        self.operator_type = operator_type
        self.custom_ode_func = custom_ode_func
        self.custom_name = custom_name
        
        # If operator_type is 'Custom', use custom_ode_func and custom_name
        if operator_type == 'Custom':
            if custom_ode_func is None or custom_name is None:
                raise ValueError("Custom operator requires both custom_ode_func and custom_name to be specified.")
            self.data_generator = lambda *args, **kwargs: generate_ODE_Operator_data(
                'Custom', *args, custom_ode_func=custom_ode_func, custom_name=custom_name, **kwargs
            )
        else:
            self.data_generator = OPERATOR_TYPES[operator_type]
        
        self.config = self.load_config(config_file)
        self.config['operator_type'] = operator_type  # Store operator type in config
        self.model = None
        self.data = {}
        self.training_history = []
        
        # Weight management
        self.best_model_path = None
        self.best_loss = float('inf')
        self.checkpoint_interval = 50  # Save checkpoint every 50 epochs
        self.saved_checkpoints = []
        self.prefix = prefix

        # Create necessary directories
        self.logs_dir = os.path.join(self.prefix, "logs") if self.prefix else "logs"
        self.checkpoints_dir = os.path.join(self.prefix, "checkpoints") if self.prefix else "checkpoints"
        self.data_dir = os.path.join(self.prefix, "data") if self.prefix else "data"
        self.dairy_dir = os.path.join(self.prefix, "dairy") if self.prefix else "dairy"
        print(f"Logs directory: {self.logs_dir}")
        print(f"Checkpoints directory: {self.checkpoints_dir}")
        print(f"Data directory: {self.data_dir}")


        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.dairy_dir, exist_ok=True)


    def load_config(self, config_file=None):
        """ Load configuration from file or use default settings."""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_file}")
        else:
            # Default configuration
            config = {
                "num_train": 100,
                "num_test": 100,
                "num_points": 100,
                "train_sample_num": 10,
                "test_sample_num": 10,
                "branch_input_size": 100,
                "trunk_input_size": 1,
                "output_size": 1,
                "num_qubits": 3,
                "net_size": [2, 1, 2, 1],
                "scale_coeff": 1.0,
                "learning_rate": 0.01,
                "num_epochs": 50,
                "target_error": 1e-4,
                "model_type": "QuanONet",
                "if_trainable_freq": False,
                "batch_size": 32,
                "validation_split": 0.2,
                "random_seed": None  # Random seed, None means not set
            }
            print("Using default configuration")

        # Note: Random seed will be set uniformly after command line argument processing
        # Do not set random seed here to avoid repeated setting
        
        return config
    
    def load_or_generate_data(self):
        """Load or generate operator data"""
        print(f"\n=== {self.operator_type} Operator Data Preparation ===")
        
        # Determine data file name
        if self.operator_type == 'Custom':
            operator_name = self.custom_name
        else:
            operator_name = self.operator_type

        # Data file path - includes all parameters affecting data size
        rank_folder = self.get_rank_folder()
        qubits_folder = self.get_qubits_folder()
        data_file = f"{self.data_dir}/{operator_name}/{rank_folder}/{qubits_folder}/{operator_name}_Operator_dataset_{self.config['num_qubits']}_{self.config['num_train']}_{self.config['num_test']}_{self.config['num_points']}_{self.config['train_sample_num']}_{self.config['test_sample_num']}.npz"
        config_file = f"{self.data_dir}/{operator_name}/{rank_folder}/{qubits_folder}/{operator_name}_Operator_config_{self.config['num_qubits']}_{self.config['num_train']}_{self.config['num_test']}_{self.config['num_points']}_{self.config['train_sample_num']}_{self.config['test_sample_num']}.json"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        if os.path.exists(data_file):
            print(f"Loading existing data from {data_file}  ...")
            try:
                data = np.load(data_file)
                
                self.data = {
                    'train_input': ms.Tensor(data['train_input'], ms.float32),
                    'train_output': ms.Tensor(data['train_output'], ms.float32),
                    'test_input': ms.Tensor(data['test_input'], ms.float32),
                    'test_output': ms.Tensor(data['test_output'], ms.float32)
                }
                
                # If branch and trunk inputs are available, load them separately
                if 'train_branch_input' in data:
                    self.data.update({
                        'train_branch_input': ms.Tensor(data['train_branch_input'], ms.float32),
                        'train_trunk_input': ms.Tensor(data['train_trunk_input'], ms.float32),
                        'test_branch_input': ms.Tensor(data['test_branch_input'], ms.float32),
                        'test_trunk_input': ms.Tensor(data['test_trunk_input'], ms.float32)
                    })
                
                print(f"Data loaded successfully from {data_file}")
                
            except Exception as e:
                print(f"Failed to load data: {e}")
                print("Generating new data...")
                self.generate_data()
        else:
            print("Generating new data...")
            self.generate_data()
        
        self.print_data_info()
        
        # Split test set into validation and test sets
        self.split_validation_test()
        
    def split_validation_test(self):
        """Split test set into validation and test sets"""
        validation_split = self.config.get('validation_split', 0.2)
        
        if validation_split <= 0 or validation_split >= 1:
            print(f"Warning: validation_split={validation_split} is invalid, not using validation set")
            # Mark not using validation set
            self.use_validation = False
            return
        
        # Mark using validation set
        self.use_validation = True
        
        # Get test set size
        if 'test_branch_input' in self.data:
            test_size = self.data['test_branch_input'].shape[0]
        else:
            test_size = self.data['test_input'].shape[0]
        
        # Calculate validation set size
        val_size = int(test_size * validation_split)
        
        # Randomly select validation set indices
        np.random.seed(42)  # Set random seed for reproducibility
        indices = np.random.permutation(test_size)
        val_indices = indices[:val_size]
        test_indices = indices[val_size:]
        
        # Convert indices to MindSpore tensors
        val_indices = ms.Tensor(val_indices, dtype=ms.int32)
        test_indices = ms.Tensor(test_indices, dtype=ms.int32)
        
        if 'test_branch_input' in self.data:
            # Branch and trunk format data
            # Create validation set
            self.data['val_branch_input'] = self.data['test_branch_input'][val_indices]
            self.data['val_trunk_input'] = self.data['test_trunk_input'][val_indices]
            self.data['val_output'] = self.data['test_output'][val_indices]
            
            # Update test set
            self.data['test_branch_input'] = self.data['test_branch_input'][test_indices]
            self.data['test_trunk_input'] = self.data['test_trunk_input'][test_indices]
            self.data['test_output'] = self.data['test_output'][test_indices]
            
            # If branch and trunk inputs are available, update them
            self.data['test_input'] = self.data['test_input'][test_indices]
        else:
            # Merge format data
            # Create validation set
            self.data['val_input'] = self.data['test_input'][val_indices]
            self.data['val_output'] = self.data['test_output'][val_indices]

            # Update test set
            self.data['test_input'] = self.data['test_input'][test_indices]
            self.data['test_output'] = self.data['test_output'][test_indices]

        print(f"\n=== Validation Set Split ===")
        print(f"Original Test Set Size: {test_size}")
        print(f"Validation Set Ratio: {validation_split}")
        print(f"Validation Set Size: {val_size}")
        print(f"New Test Set Size: {len(test_indices)}")
        print(f"During training, the validation set will be used for model selection, and the final evaluation will be on the test set.")
        
    def generate_data(self):
        """Generate new operator data"""
        if self.operator_type == 'Custom':
            print(f"Generating custom operator data: {self.custom_name}...")
        else:
            print(f"Generating {self.operator_type} operator data...")

        # Generate raw data
        # Use ODE encoding
        train_branch_input, train_trunk_input, train_output, \
        test_branch_input, test_trunk_input, test_output = ODE_encode(
            self.data_generator,
            self.config['num_train'],
            self.config['num_test'], 
            self.config['num_points'],
            self.config['train_sample_num'],
            self.config['test_sample_num']
        )

        # Merge branch and trunk inputs
        train_input = mnp.concatenate((train_branch_input, train_trunk_input), axis=1)
        test_input = mnp.concatenate((test_branch_input, test_trunk_input), axis=1)

        # Store data
        self.data = {
            'train_input': train_input,
            'train_output': train_output,
            'test_input': test_input,
            'test_output': test_output,
            'train_branch_input': train_branch_input,
            'train_trunk_input': train_trunk_input,
            'test_branch_input': test_branch_input,
            'test_trunk_input': test_trunk_input
        }

        # Save data
        rank_folder = self.get_rank_folder()
        if self.operator_type == 'Custom':
            operator_name = self.custom_name
        else:
            operator_name = self.operator_type

        qubits_folder = self.get_qubits_folder()
        data_file = f"{self.data_dir}/{operator_name}/{rank_folder}/{qubits_folder}/{operator_name}_Operator_dataset_{self.config['num_qubits']}_{self.config['num_train']}_{self.config['num_test']}_{self.config['num_points']}_{self.config['train_sample_num']}_{self.config['test_sample_num']}.npz"
        config_file = f"{self.data_dir}/{operator_name}/{rank_folder}/{qubits_folder}/{operator_name}_Operator_config_{self.config['num_qubits']}_{self.config['num_train']}_{self.config['num_test']}_{self.config['num_points']}_{self.config['train_sample_num']}_{self.config['test_sample_num']}.json"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        print(f"Saving data to {data_file}...")
        if self.config['if_save']:
            np.savez_compressed(
                data_file,
                train_input=train_input.asnumpy(),
                train_output=train_output.asnumpy(),
                test_input=test_input.asnumpy(),
                test_output=test_output.asnumpy(),
                train_branch_input=train_branch_input.asnumpy(),
                train_trunk_input=train_trunk_input.asnumpy(),
                test_branch_input=test_branch_input.asnumpy(),
                test_trunk_input=test_trunk_input.asnumpy()
            )
        
        # Save configuration
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        if self.config['if_save']:
            with open(config_file, 'w') as f:
                json.dump(self.config, f, indent=2)

        print("Data generation and saving completed!")

    def get_operator_description(self):
        """Get operator description"""
        descriptions = {
            'Inverse': 'Inverse operator (du/dx = u0(x))',
            'Homogeneous': 'Homogeneous operator (du/dx = u + u0(x))',
            'Nonlinear': 'Nonlinear operator (du/dx = u - u0²(x))'
        }
        
        if self.operator_type == 'Custom':
            return f"Custom operator: {self.custom_name}"
        else:
            return descriptions.get(self.operator_type, f"{self.operator_type} operator")
    
    def print_data_info(self):
        """Print data information"""
        print(f"\n=== Data Information ===")
        print(f"Operator Type: {self.operator_type}")
        print(f"Operator Description: {self.get_operator_description()}")
        print(f"Number of Training Samples: {self.data['train_input'].shape[0]}")
        print(f"Number of Test Samples: {self.data['test_input'].shape[0]}")
        print(f"Input Dimension: {self.data['train_input'].shape[1]}")
        print(f"Output Dimension: {self.data['train_output'].shape[1]}")
        
        if 'train_branch_input' in self.data:
            print(f"Branch Input Dimension: {self.data['train_branch_input'].shape[1]}")
            print(f"Trunk Input Dimension: {self.data['train_trunk_input'].shape[1]}")
    def get_ham_rank_str(self):
        """返回哈密顿秩字符串，格式如 _rank2,没有则空字符串"""
        rank = self.config.get('ham_rank', None)
        return f"_rank{rank}" if rank is not None else ""
    
    def get_rank_folder(self):
        """返回如 rank2,没有则空字符串"""
        rank = self.config.get('ham_rank', None)
        return f"rank{rank}" if rank is not None else ""
    
    def get_qubits_folder(self):
        """返回如 qubits3"""
        return f"qubits{self.config['num_qubits']}"
    
    def create_model(self):
        print("\n=== Model Creation ===")

        # Create Hamiltonian
        if self.config.get('ham_rank', None) is None:
            ham = generate_simple_hamiltonian(self.config['num_qubits'], lower_bound=-5, upper_bound=5)
        else:
            
            diag = generate_ham_diag_diffspectrum(self.config['num_qubits'], self.config['ham_rank'], self.config['random_seed'])
            ham = ham_diag_to_operator(diag, self.config['num_qubits'])
        print(f"Using Hamiltonian: {ham.hamiltonian.matrix()}")
        # Get input dimensions
        if 'train_branch_input' in self.data:
            branch_input_size = self.data['train_branch_input'].shape[1]
            trunk_input_size = self.data['train_trunk_input'].shape[1]
        else:
            # Infer from config
            branch_input_size = self.config['branch_input_size']
            trunk_input_size = self.config['trunk_input_size']

        # Create model
        model_type = self.config.get('model_type', 'QuanONet')
        if_trainable_freq = self.config.get('if_trainable_freq', False)
        
        if model_type == 'QuanONet':
            self.model = QuanONet(
                num_qubits=self.config['num_qubits'],
                branch_input_size=branch_input_size,
                trunk_input_size=trunk_input_size,
                net_size=tuple(self.config['net_size']),
                ham=ham,
                scale_coeff=self.config['scale_coeff'],
                if_trainable_freq=if_trainable_freq
            )
        elif model_type == 'HEAQNN':
            self.model = HEAQNN(
                num_qubits=self.config['num_qubits'],
                branch_input_size=branch_input_size,
                trunk_input_size=trunk_input_size,
                net_size=tuple(self.config['net_size']),
                ham=ham,
                scale_coeff=self.config['scale_coeff'],
                if_trainable_freq=if_trainable_freq
            )
        elif model_type == 'FNN':
            self.model = FNN(
                branch_input_size=branch_input_size,
                trunk_input_size=trunk_input_size,
                output_size = 1,
                net_size=tuple(self.config['net_size']),
            )
        elif model_type == 'DeepONet':
            self.model = DeepONet(
                branch_input_size=branch_input_size,
                trunk_input_size=trunk_input_size,
                net_size=tuple(self.config['net_size']),
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Print model summary
        total_params = count_parameters(self.model)
        print(f"Model Type: {model_type}")
        if hasattr(self.model, 'circuit'):
            print(f"Circuit Parameters: ")
            self.model.circuit.summary()
        print(f"Trainable Frequency: {'Enabled' if if_trainable_freq else 'Disabled'}")
        print(f"Network Structure: {self.config['net_size']}")
        print(f"Total Parameters: {total_params:,}")

        return self.model
    
    def train_model(self):
        """Train model"""
        print("\n=== Model Training ===")

        if self.model is None:
            raise ValueError("Please create a model first")

        # Set up training components
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.config['learning_rate'])
        loss_fn = nn.MSELoss()

        # Create training network
        net_with_loss = nn.WithLossCell(self.model, loss_fn)
        train_net = nn.TrainOneStepCell(net_with_loss, optimizer)
        
        # Set random seed for reproducibility
        if 'train_branch_input' in self.data:
            train_input = (self.data['train_branch_input'], self.data['train_trunk_input'])
            if hasattr(self, 'use_validation') and self.use_validation:
                val_input = (self.data['val_branch_input'], self.data['val_trunk_input'])
            else:
                val_input = None
        else:
            train_input = self.data['train_input']
            if hasattr(self, 'use_validation') and self.use_validation:
                val_input = self.data['val_input']
            else:
                val_input = None
        
        train_output = self.data['train_output']
        if hasattr(self, 'use_validation') and self.use_validation:
            val_output = self.data['val_output']
        else:
            val_output = None

        # Get batch size
        batch_size = self.config.get('batch_size', len(train_output))  # Default to all data
        if isinstance(train_input, tuple):
            train_size = train_input[0].shape[0]
            if val_input is not None:
                val_size = val_input[0].shape[0]
            else:
                val_size = 0
        else:
            train_size = train_input.shape[0]
            if val_input is not None:
                val_size = val_input.shape[0]
            else:
                val_size = 0
        
        # Calculate batch numbers
        num_batches_train = max(train_size // batch_size, 1)
        if val_input is not None:
            batch_size_val = min(batch_size, val_size)
            num_batches_val = max(val_size // batch_size_val, 1)
        else:
            batch_size_val = 0
            num_batches_val = 0
        
        print(f"Training Parameters:")
        print(f"  Learning Rate: {self.config['learning_rate']}")
        print(f"  Max Epochs: {self.config['num_epochs']}")
        print(f"  Target Error: {self.config['target_error']}")
        print(f"  Batch Size: {batch_size}")
        print(f"  Training Batches: {num_batches_train}")
        print(f"  Use Validation Set: {'Yes' if val_input is not None else 'No'}")
        print(f"  Checkpoint Interval: {self.checkpoint_interval} epochs")
        
        # Training loop
        best_test_loss = float('inf')
        best_epoch = 0
        patience = 50  # Early stopping patience
        no_improve = 0
        
        print("\nStarting training...")
        for epoch in tqdm(range(self.config['num_epochs']), desc="Training Progress"):
            # Training step - supports batch training
            epoch_train_loss = 0.0
            
            # Shuffle training data randomly
            if isinstance(train_input, tuple):
                permutation = np.random.permutation(train_input[0].shape[0])
                permutation = ms.Tensor(permutation, dtype=ms.int32)
            else:
                permutation = np.random.permutation(train_input.shape[0])
                permutation = ms.Tensor(permutation, dtype=ms.int32)
            
            # Train all batches
            for batch_idx in range(num_batches_train):
                start = batch_idx * batch_size
                end = min(start + batch_size, train_size)
                batch_indices = permutation[start:end]
                
                # Prepare batch data
                if isinstance(train_input, tuple):
                    batch_branch_input = train_input[0][batch_indices]
                    batch_trunk_input = train_input[1][batch_indices]
                    batch_input = (batch_branch_input, batch_trunk_input)
                else:
                    batch_input = train_input[batch_indices]
                
                batch_output = train_output[batch_indices]
                
                # Execute training step
                train_loss = train_net(batch_input, batch_output)
                epoch_train_loss += float(train_loss)
            
            # Calculate average training loss
            avg_train_loss = epoch_train_loss / num_batches_train
            
            # Evaluate every 10 epochs (if validation set is available)
            if epoch % 10 == 0:
                if val_input is not None:
                    # Validation evaluation - supports batch processing
                    total_val_loss = 0.0
                    for batch_idx in range(num_batches_val):
                        start = batch_idx * batch_size_val
                        end = min(start + batch_size_val, val_size)
                        
                        # Prepare validation batch data
                        if isinstance(val_input, tuple):
                            val_batch_branch_input = val_input[0][start:end]
                            val_batch_trunk_input = val_input[1][start:end]
                            val_batch_input = (val_batch_branch_input, val_batch_trunk_input)
                        else:
                            val_batch_input = val_input[start:end]
                        
                        val_batch_output = val_output[start:end]
                        
                        # Calculate validation loss
                        val_pred = self.model(val_batch_input)
                        val_loss_batch = loss_fn(val_pred, val_batch_output)
                        total_val_loss += float(val_loss_batch)
                    
                    # Average validation loss
                    avg_val_loss = total_val_loss / num_batches_val
                    val_loss = ms.Tensor(avg_val_loss, ms.float32)
                    
                    self.training_history.append({
                        'epoch': epoch,
                        'train_loss': avg_train_loss,
                        'val_loss': float(val_loss)
                    })
                    
                    # Save best model (based on validation loss)
                    if val_loss < best_test_loss - 1e-6:
                        print(f"  Found better model: Training loss: {float(avg_train_loss):.6f}, Validation loss improved from {float(best_test_loss):.6f} to {float(val_loss):.6f}")
                        best_test_loss = val_loss
                        best_epoch = epoch
                        no_improve = 0
                        if self.config['if_save']:
                            self.save_model("best", overwrite=True)
                    else:
                        no_improve += 1
                else:
                    # No validation set, save model based on training loss
                    self.training_history.append({
                        'epoch': epoch,
                        'train_loss': avg_train_loss
                    })
                    
                    # Save best model (based on training loss)
                    if avg_train_loss < best_test_loss - 1e-6:
                        print(f"  Found better model: Training loss improved from {float(best_test_loss):.6f} to {float(avg_train_loss):.6f}")
                        best_test_loss = avg_train_loss
                        best_epoch = epoch
                        no_improve = 0
                        if self.config['if_save']:
                            self.save_model("best", overwrite=True)
                    else:
                        no_improve += 1
                
                # Periodically save checkpoints
                if epoch > 0 and epoch % self.checkpoint_interval == 0:
                    if self.config['if_save']:
                        self.save_model(f"checkpoint_epoch_{epoch}", overwrite=False)
                
                # Print progress every 100 epochs
                if epoch % 100 == 0:
                    if val_input is not None:
                        print(f"Epoch {epoch}: Train={avg_train_loss:.6f}, Val={float(val_loss):.6f}, Best={'Val' if val_input is not None else 'Train'}={float(best_test_loss):.6f}")
                    else:
                        print(f"Epoch {epoch}: Train={avg_train_loss:.6f}, Best Train={float(best_test_loss):.6f}")
                
                # Check convergence
                if avg_train_loss < self.config['target_error']:
                    print(f"Reached target error {self.config['target_error']} at epoch {epoch}")
                    break
                
                # Early stopping
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}: {patience} epochs without improvement")
                    break
        
        # Final evaluation on complete test set - using batch processing
        if 'test_branch_input' in self.data:
            test_input = (self.data['test_branch_input'], self.data['test_trunk_input'])
        else:
            test_input = self.data['test_input']
        test_output = self.data['test_output']
        
        # Calculate test set batch parameters
        if isinstance(test_input, tuple):
            test_size = test_input[0].shape[0]
        else:
            test_size = test_input.shape[0]
        
        batch_size_test = min(batch_size, test_size)
        num_batches_test = max(test_size // batch_size_test, 1)
        
        print(f"Final test evaluation: {test_size} samples, {num_batches_test} batches")
        
        final_test_loss = 0.0
        for batch_idx in range(num_batches_test):
            start = batch_idx * batch_size_test
            end = min(start + batch_size_test, test_size)
            
            if isinstance(test_input, tuple):
                test_batch_branch_input = test_input[0][start:end]
                test_batch_trunk_input = test_input[1][start:end]
                test_batch_input = (test_batch_branch_input, test_batch_trunk_input)
            else:
                test_batch_input = test_input[start:end]
            
            test_batch_output = test_output[start:end]
            test_pred = self.model(test_batch_input)
            batch_loss = loss_fn(test_pred, test_batch_output)
            final_test_loss += float(batch_loss) * (end - start)
        
        final_test_loss = final_test_loss / test_size
        
        print(f"\nTraining completed!")
        print(f"Final training loss: {avg_train_loss:.6f}")
        if val_input is not None:
            print(f"Best validation loss: {float(best_test_loss):.6f} (epoch {best_epoch})")
        else:
            print(f"Best training loss: {float(best_test_loss):.6f} (epoch {best_epoch})")
        print(f"Final test loss: {final_test_loss:.6f}")
        
        # Save final model
        if self.config['if_save']:
            self.save_model("final", overwrite=False)
        
        # Display saved models summary
            self.print_saved_models_summary()
        
        return final_test_loss
    
    def save_model(self, suffix="", overwrite=False):
        """
        Improved model saving method
        
        Args:
            suffix: Filename suffix
            overwrite: Whether to overwrite save (for best model)
        """
        rank_folder = self.get_rank_folder()
        if suffix == "best" and overwrite:
            # Best model: fixed filename, overwrite save
            if self.operator_type == 'Custom':
                operator_name = self.custom_name
            else:
                operator_name = self.operator_type
            qubits_folder = self.get_qubits_folder()
            if self.config['if_trainable_freq']:
                filename = f"{operator_name}/{rank_folder}/{qubits_folder}/best_{operator_name}_TF-{self.config['model_type']}_{self.config['net_size']}_{self.config['random_seed']}.ckpt"
            else:
                filename = f"{operator_name}/{rank_folder}/{qubits_folder}/best_{operator_name}_{self.config['model_type']}_{self.config['net_size']}_{self.config['random_seed']}.ckpt"
            filepath = os.path.join(self.checkpoints_dir, filename)
            dir_name = os.path.dirname(filepath)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            # Delete old best model file
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    os.chmod(self.best_model_path, stat.S_IWRITE)
                    os.remove(self.best_model_path)
                    # print(f"    Deleted old best model: {os.path.basename(self.best_model_path)}")
                except Exception as e:
                    print(f"    Failed to delete old model: {e}")
            
            self.best_model_path = filepath
            if self.config['if_save']:
                save_checkpoint(self.model, filepath)
            # print(f"    Saved best model: {os.path.basename(filepath)}")
            
        else:
            # Other models: unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if self.operator_type == 'Custom':
                operator_name = self.custom_name
            else:
                operator_name = self.operator_type
            qubits_folder = self.get_qubits_folder()
            if self.config['if_trainable_freq']:
                filename = f"{operator_name}/{rank_folder}/{qubits_folder}/{operator_name}_TF-{self.config['model_type']}_{suffix}_{timestamp}.ckpt"
            else:
                filename = f"{operator_name}/{rank_folder}/{qubits_folder}/{operator_name}_{self.config['model_type']}_{suffix}_{timestamp}.ckpt"
            filepath = os.path.join(self.checkpoints_dir, filename)
            dir_name = os.path.dirname(filepath)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
            if self.config['if_save']:
                save_checkpoint(self.model, filepath)
            # print(f"    Saved model: {os.path.basename(filepath)}")
            
            # Record checkpoint (for management)
            if suffix.startswith("checkpoint"):
                self.saved_checkpoints.append(filepath)
                # Keep only the latest 3 checkpoints
                if len(self.saved_checkpoints) > 3:
                    old_checkpoint = self.saved_checkpoints.pop(0)
                    try:
                        os.remove(old_checkpoint)
                        # print(f"    Deleted old checkpoint: {os.path.basename(old_checkpoint)}")
                    except Exception as e:
                        print(f"    Failed to delete checkpoint: {e}")
        
        return filepath
    
    def print_saved_models_summary(self):
        """Print saved models summary"""
        print(f"\n=== Saved Models Summary ===")

        if not os.path.exists(self.checkpoints_dir):
            print("No saved models")
            return

        rank_folder = self.get_rank_folder()
        qubits_folder = self.get_qubits_folder()  
        operator_folder = os.path.join(self.checkpoints_dir, self.operator_type, rank_folder, qubits_folder)
        saved_files = []
        if os.path.exists(operator_folder):
            saved_files = [f for f in os.listdir(operator_folder) if f.endswith('.ckpt')]
        
        best_files = [f for f in saved_files if 'best' in f]
        final_files = [f for f in saved_files if 'final' in f]
        checkpoint_files = [f for f in saved_files if 'checkpoint' in f]
        
        print(f"📍 Best models: {len(best_files)} (should be only 1)")
        for f in best_files:
            print(f"   {f}")
        
        print(f"🏁 Final models: {len(final_files)}")
        for f in final_files:
            print(f"   {f}")
        
        print(f"⏰ Checkpoints: {len(checkpoint_files)} (max 3 kept)")
        for f in checkpoint_files:
            print(f"   {f}")
        
        print(f"\nModel saving strategy:")
        print(f"  - Checkpoints: Save every {self.checkpoint_interval} epochs, keep max 3")
    
    def evaluate_model(self):
        """Evaluate model performance - supports batch processing"""
        print("\n=== Model Evaluation ===")
        
        if self.model is None:
            raise ValueError("Please train the model first")
        
        rank_folder = self.get_rank_folder()
        
        # Prepare test data
        if 'test_branch_input' in self.data:
            test_input = (self.data['test_branch_input'], self.data['test_trunk_input'])
            test_size = test_input[0].shape[0]
        else:
            test_input = self.data['test_input']
            test_size = test_input.shape[0]
        
        test_output = self.data['test_output']
        
        # Get batch size for batch evaluation
        batch_size = self.config.get('batch_size', 32)
        batch_size_eval = min(batch_size, test_size)
        num_batches_eval = max(test_size // batch_size_eval, 1)
        
        print(f"Test set size: {test_size}")
        print(f"Evaluation batch size: {batch_size_eval}")
        print(f"Number of evaluation batches: {num_batches_eval}")
        
        # Batch prediction and metric calculation
        all_predictions = []
        all_targets = []
        total_mse = 0.0
        total_mae = 0.0
        max_error = 0.0
        
        for batch_idx in range(num_batches_eval):
            start = batch_idx * batch_size_eval
            end = min(start + batch_size_eval, test_size)
            
            # Prepare test batch data
            if isinstance(test_input, tuple):
                test_batch_branch_input = test_input[0][start:end]
                test_batch_trunk_input = test_input[1][start:end]
                test_batch_input = (test_batch_branch_input, test_batch_trunk_input)
            else:
                test_batch_input = test_input[start:end]
            
            test_batch_output = test_output[start:end]
            
            # Prediction
            test_batch_pred = self.model(test_batch_input)
            
            # Calculate batch metrics
            batch_mse = nn.MSELoss()(test_batch_pred, test_batch_output)
            batch_mae = mnp.mean(mnp.abs(test_batch_pred - test_batch_output))
            batch_max_error = mnp.max(mnp.abs(test_batch_pred - test_batch_output))
            
            # Accumulate metrics
            total_mse += float(batch_mse) * (end - start)
            total_mae += float(batch_mae) * (end - start)
            max_error = max(max_error, float(batch_max_error))
            
            # Collect prediction results (optional, for further analysis)
            all_predictions.append(test_batch_pred.asnumpy())
            all_targets.append(test_batch_output.asnumpy())
        
        # Calculate average metrics
        mse = total_mse / test_size
        mae = total_mae / test_size
        
        results = {
            'MSE': mse,
            'MAE': mae,
            'Max_Error': max_error,
        }
        
        print("Evaluation results:")
        for metric, value in results.items():
            print(f"  {metric}: {value:.6f}")
        
        # Save evaluation results
        if self.config['if_trainable_freq']:
            results_file = os.path.join(self.logs_dir, f"{self.operator_type}/{rank_folder}/training_{self.operator_type}_TF-{self.config['model_type']}_{self.config['num_qubits']}_{self.config['net_size']}_seed{self.config['random_seed']}_.json")
        else:
            qubits_folder = self.get_qubits_folder()
            results_file = os.path.join(
                self.logs_dir,
                f"{self.operator_type}/{rank_folder}/{qubits_folder}/training_{self.operator_type}_{self.config['model_type']}_{self.config['num_qubits']}_{self.config['net_size']}_seed{self.config['random_seed']}_.json"
            )
        dir_name = os.path.dirname(results_file)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)
        if self.config['if_save']:
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            
            log_dict = {
            'config': self.config,  # 保存本次配置
            'training_history': self.training_history,  # 保存训练过程损失（每10个epoch一次）
            'results': results  # 保存最终评估结果
            }
            with open(results_file, 'w') as f:
                json.dump(log_dict, f, indent=2)
            print(f"Evaluation results saved to: {results_file}")
            
        return results
    
    def run_complete_pipeline(self):
        """Run complete training pipeline"""
        print(f"=== {self.config['model_type']} {self.operator_type} Operator Solving Pipeline ===")
        print(f"Operator Description: {self.get_operator_description()}")
        print(f"Start Time: {datetime.now()}")

        results = None  # 初始化，避免未赋值报错

        try:
            # 1. Data preparation
            self.load_or_generate_data()
            
            # 2. Model creation
            self.create_model()
            
            if self.config['if_train']:
                # 3. Model training
                final_loss = self.train_model()
            
            # 4. Model evaluation
            for init_checkpoint in self.config.get("init_checkpoints", []):
                print(f"\nLoading initial checkpoint from {init_checkpoint} for evaluation...")
                load_checkpoint(init_checkpoint, self.model)
                results = self.evaluate_model()
            
            # 如果没有init_checkpoints，也要评估一次
            if not self.config.get("init_checkpoints", []):
                results = self.evaluate_model()
            
            print(f"\n=== Complete Pipeline Execution Successful ===")
            print(f"End Time: {datetime.now()}")

            return results

        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='ODE Operator Solving')
    parser.add_argument('--operator', type=str, choices=list(OPERATOR_TYPES.keys()), help='Operator type')
    parser.add_argument('--config', type=str, default='configs/config_ODE.json', help='Configuration file path')
    parser.add_argument('--model_type', type=str, choices=['QuanONet', 'HEAQNN', 'FNN', 'DeepONet'], help='Model type')
    parser.add_argument('--net_size', type=int, nargs='+', help='Network size')
    parser.add_argument('--num_qubits', type=int, help='Number of qubits')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--validation_split', type=float, help='Validation set ratio')
    parser.add_argument('--random_seed', type=int, help='Random seed for reproducible results')
    parser.add_argument('--scale_coeff', type=float, help='Scale coefficient for loss function')
    parser.add_argument('--if_trainable_freq', type=str, help='Whether to use trainable frequency (true/false)')
    parser.add_argument('--prefix', type=str, help='Prefix of outputs')
    parser.add_argument('--ham_diag', type=int, nargs='+', help='Ham diagonal elements')
    parser.add_argument('--if_save', type=str, help='Whether to save (true/false)')
    parser.add_argument('--if_keep', type=str, help='Whether to keep (true/false)')
    parser.add_argument('--if_train', type=str, help='Whether to train (true/false)')
    parser.add_argument('--init_checkpoints', type=list, help='Initial checkpoint file path')
    # Custom operator parameters
    parser.add_argument('--custom_ode', type=str, default=None, 
                       help='Custom ODE function string (only used when operator=Custom)')
    parser.add_argument('--custom_name', type=str, default=None, 
                       help='Custom operator name (only used when operator=Custom)')
    parser.add_argument('--ham_rank', type=int, help='Hamiltonian rank (only used when operator=Inverse(ODE))')
    args = parser.parse_args()
    
    print(f"=== ODE Operator {args.model_type} Training System ===")
    print(f"Operator type: {args.operator}")
    
    # Handle custom operator
    custom_ode_func = None
    if args.operator == 'Custom':
        if args.custom_ode is None or args.custom_name is None:
            print("Error: Custom operator requires both --custom_ode and --custom_name to be specified")
            print("Example: --operator Custom --custom_ode 'u + 2*u0' --custom_name 'MyOperator'")
            return
        
        try:
            custom_ode_func = parse_custom_ode_function(args.custom_ode)
            print(f"Custom ODE function: {args.custom_ode}")
            print(f"Custom name: {args.custom_name}")
        except ValueError as e:
            print(f"Error: {e}")
            return
    
    # Create solver
    solver = ODEOperatorSolver(
        operator_type=args.operator, 
        config_file=args.config,
        custom_ode_func=custom_ode_func,
        custom_name=args.custom_name,
        prefix=args.prefix
    )
    
    # Override configuration with command line arguments (only when command line arguments exist)
    if args.model_type:
        solver.config['model_type'] = args.model_type
    if args.num_qubits is not None:
        solver.config['num_qubits'] = args.num_qubits
    if args.net_size:
        solver.config['net_size'] = args.net_size
    if args.learning_rate is not None:
        solver.config['learning_rate'] = args.learning_rate
    if args.num_epochs is not None:
        solver.config['num_epochs'] = args.num_epochs
    if args.batch_size is not None:
        solver.config['batch_size'] = args.batch_size
    if args.validation_split is not None:
        solver.config['validation_split'] = args.validation_split
    if args.random_seed is not None:
        print("Command line argument overrides random seed setting")
        solver.config['random_seed'] = args.random_seed
    if args.scale_coeff is not None:
        solver.config['scale_coeff'] = args.scale_coeff
    if args.if_trainable_freq is not None:
        solver.config['if_trainable_freq'] = args.if_trainable_freq.lower() == 'true'
    if args.if_save is not None:
        solver.config['if_save'] = args.if_save.lower() == 'true'
    if args.if_keep is not None:
        solver.config['if_keep'] = args.if_keep.lower() == 'true'
    if args.ham_rank is not None:
        solver.config['ham_rank'] = args.ham_rank
    if args.if_train is not None:
        solver.config['if_train'] = args.if_train.lower() == 'true'
    if args.init_checkpoints is not None:
        solver.config['init_checkpoints'] = args.init_checkpoints
    
    
        
    # Uniformly set random seed (considering command line override and config file)
    final_seed = solver.config.get('random_seed', None)
    set_random_seed(final_seed)
    
    print(f"Using configuration: {solver.config}")
    
    # Run complete pipeline
    results = solver.run_complete_pipeline()
    
    if results:
        print("\n=== Final Results Summary ===")
        print(f"Operator type: {solver.operator_type}")
        print(f"Operator description: {solver.get_operator_description()}")
        print(f"Model type: {solver.config['model_type']}")
        print(f"MSE: {results['MSE']:.6f}")
        print(f"MAE: {results['MAE']:.6f}")
        print(f"Max Error: {results['Max_Error']:.6f}")
    else:
        print("Training failed")


if __name__ == "__main__":
    main()
