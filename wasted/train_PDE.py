#!/usr/bin/env python3
"""
Universal PDE Operator Training Script

This script supports:
1. RDiffusion: Reaction-diffusion equation (∂u/∂t = α∇²u + k*u² + u0(x))
2. Burgers: Burgers conduction equation (∂u/∂t = α∇²u + u0(x))
3. Identity: Identity operator (u(x,t) = u0(x) for all t)
4. Schrodinger: Schrödinger equation (iℏ∂ψ/∂t = -ℏ²/(2m)∇²ψ + V(x)ψ)
5. Advection: Advection equation (∂u/∂t + c∇u = 0, with u0 as initial condition)
6. Darcy: Darcy flow (−∇(K∇u)=f)

Usage:
python train_PDE.py --operator RDiffusion
python train_PDE.py --operator Advection --model_type QuanONet --net_size 40 2 20 2 --scale_coeff 0.001 --random_seed 0 --if_trainable_freq true
nohup python -u train_PDE.py --operator Burgers > training_Burgers.log 2>&1 &
nohup python -u train_PDE.py --config configs/config_Darcy.json --operator Darcy > training_Darcy.log 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python -u train_PDE.py --operator RDiffusion --scale_coeff 0.1 --random_seed 0 --if_trainable_freq false --net_size 40 2 40 2 > training_RDiffusion_QuanONet_0.1_0.log 2>&1 &
nohup python -u train_PDE.py --operator RDiffusion --model_type HEAQNN --scale_coeff 0.001 --random_seed 0 --if_trainable_freq false --net_size 40 2 > training_RDiffusion_HEAQNN_0.001_0.log 2>&1 &
!!! Note that Darcy operator requires the unique config file, due to the resolution difference between u0 and u!!!
"""

import sys
import os
import json
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
from tqdm import tqdm
from mindspore.train.serialization import save_checkpoint, load_checkpoint
import argparse
from datetime import datetime

# Import local modules
from data_utils.data_generation import generate_PDE_Operator_data
from data_utils import PDE_SYSTEMS
from data_utils.data_processing import PDE_encode
from core.models import QuanONet, HEAQNN, FNN, DeepONet
from core.quantum_circuits import generate_simple_hamiltonian
from utils.utils import count_parameters

# Set MindSpore environment
ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")

def set_random_seed(seed=None):
    """
    Set random seed for reproducible results
    
    Args:
        seed: Random seed value, if None then no random seed is set
    """
    if seed is not None:
        print(f"Setting random seed: {seed}")
        np.random.seed(seed)
        ms.set_seed(seed)
        # Set other random libraries if available
        import random
        random.seed(seed)
    else:
        print("Random seed not set, using system default randomness")


class PDEOperatorSolver:
    """PDE operator problem solver"""
    

    def __init__(self, operator_type, config_file):
        """
        Initialize solver
        
        Args:
            operator_type: PDE Operator Type ('RDiffusion', 'Advection', ...)
            config_file: Configuration file path, if None then use default configuration
        """

        self.operator_type = operator_type
        self.config = self.load_config(config_file)
        self.config['operator_type'] = operator_type
        self.model = None
        self.data = {}
        self.training_history = []
        
        # Weight saving management
        self.best_model_path = None
        self.best_loss = float('inf')
        self.checkpoint_interval = 50  # Save checkpoints every 50 epochs
        self.saved_checkpoints = []
        
        # Create necessary directories
        os.makedirs("/mnt/nas-new/home/yange/wangruocheng/QON/data", exist_ok=True)
        os.makedirs("/mnt/nas-new/home/yange/wangruocheng/QON/checkpoints", exist_ok=True)
        os.makedirs("/mnt/nas-new/home/yange/wangruocheng/QON/logs", exist_ok=True)

        self.data_generator = lambda *args, **kwargs: generate_PDE_Operator_data(
            self.operator_type, *args, **kwargs, num_points_0=self.config['num_points_0']
        )
        
    def load_config(self, config_file=None):
        """Load configuration"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Loading configuration from {config_file}")
        else:
            # Default configuration
            config = {
                "num_train": 100,
                "num_test": 100,
                "num_points": 100,
                "train_sample_num": 10,
                "test_sample_num": 10,
                "branch_input_size": 100,
                "trunk_input_size": 2,  # PDE requires (x,t) two inputs
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
                "random_seed": None,  # Random seed, None means not set
                # Schrödinger equation specific parameters
                "hbar": 1.0,
                "m": 1.0,
                "sigma": 0.05
            }
            print("Using default configuration")
        
        # Note: Random seed will be set uniformly after command line argument processing
        # Do not set random seed here to avoid repeated setting
        
        return config
    
    def load_or_generate_data(self):
        """Load or generate operator data"""
        print(f"\n=== {self.operator_type} Operator Data Preparation ===")
        
        # Determine data file name
        operator_name = self.operator_type
        
        # Data file path - includes all parameters affecting data size
        log_subdir = "/mnt/nas-new/home/yange/wangruocheng/QON/logs"

        data_file = os.path.join(log_subdir, f"{operator_name}_Operator_dataset_{self.config['num_train']}_{self.config['num_test']}_{self.config['num_points']}_{self.config['train_sample_num']}_{self.config['test_sample_num']}.npz")
        # _{self.config['if_trainable_freq']}_{self.config['num_qubits']}_{self.config['net_size']}_{self.config['random_seed']}
        
        if os.path.exists(data_file):
            print(f"Loading existing data from {data_file}...")
            try:
                data = np.load(data_file)
                
                self.data = {
                    'train_input': ms.Tensor(data['train_input'], ms.float32),
                    'train_output': ms.Tensor(data['train_output'], ms.float32),
                    'test_input': ms.Tensor(data['test_input'], ms.float32),
                    'test_output': ms.Tensor(data['test_output'], ms.float32)
                }
                
                # If separate format data is available, load it too
                if 'train_branch_input' in data:
                    self.data.update({
                        'train_branch_input': ms.Tensor(data['train_branch_input'], ms.float32),
                        'train_trunk_input': ms.Tensor(data['train_trunk_input'], ms.float32),
                        'test_branch_input': ms.Tensor(data['test_branch_input'], ms.float32),
                        'test_trunk_input': ms.Tensor(data['test_trunk_input'], ms.float32)
                    })
                
                print("Data loaded successfully!")
                
            except Exception as e:
                print(f"Data loading failed: {e}")
                print("Regenerating data...")
                self.generate_data()
        else:
            print("Data file does not exist, generating new data...")
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
        
        # Convert to MindSpore tensors
        val_indices = ms.Tensor(val_indices, dtype=ms.int32)
        test_indices = ms.Tensor(test_indices, dtype=ms.int32)
        
        if 'test_branch_input' in self.data:
            # Separate format data
            # Create validation set
            self.data['val_branch_input'] = self.data['test_branch_input'][val_indices]
            self.data['val_trunk_input'] = self.data['test_trunk_input'][val_indices]
            self.data['val_output'] = self.data['test_output'][val_indices]
            
            # Update test set
            self.data['test_branch_input'] = self.data['test_branch_input'][test_indices]
            self.data['test_trunk_input'] = self.data['test_trunk_input'][test_indices]
            self.data['test_output'] = self.data['test_output'][test_indices]
            
            # Update merged format test set
            self.data['test_input'] = self.data['test_input'][test_indices]
        else:
            # Merged format data
            # Create validation set
            self.data['val_input'] = self.data['test_input'][val_indices]
            self.data['val_output'] = self.data['test_output'][val_indices]
            
            # Update test set
            self.data['test_input'] = self.data['test_input'][test_indices]
            self.data['test_output'] = self.data['test_output'][test_indices]
        
        print(f"\n=== Validation Set Split ===")
        print(f"Original test set size: {test_size}")
        print(f"Validation set ratio: {validation_split}")
        print(f"Validation set size: {val_size}")
        print(f"New test set size: {len(test_indices)}")
        print(f"During training, validation set will be used for model selection, final evaluation on test set")
        
    def generate_data(self):
        """Generate new operator data"""
        print(f"Generating {self.operator_type} operator data...")
        
        # Generate raw data
        # Use PDE encoding
        train_branch_input, train_trunk_input, train_output, \
        test_branch_input, test_trunk_input, test_output = PDE_encode(
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
        operator_name = self.operator_type
            
        data_file = f"/mnt/nas-new/home/yange/wangruocheng/QON/data/{operator_name}_Operator_dataset_{self.config['num_train']}_{self.config['num_test']}_{self.config['num_points']}_{self.config['train_sample_num']}_{self.config['test_sample_num']}.npz"
        
        print(f"Saving data to {data_file}...")
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
        
        # Also save configuration
        config_file = f"/mnt/nas-new/home/yange/wangruocheng/QON/data/{operator_name}_Operator_config_{self.config['num_train']}_{self.config['num_test']}_{self.config['num_points']}_{self.config['train_sample_num']}_{self.config['test_sample_num']}.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        print("Data generation and saving completed!")
    
    def print_data_info(self):
        """Print data information"""
        print(f"\n=== Data Information ===")
        print(f"Operator Type: {self.operator_type}")
        print(f"Training samples: {self.data['train_input'].shape[0]}")
        print(f"Test samples: {self.data['test_input'].shape[0]}")
        print(f"Input dimension: {self.data['train_input'].shape[1]}")
        print(f"Output dimension: {self.data['train_output'].shape[1]}")
        
        if 'train_branch_input' in self.data:
            print(f"Branch input dimension: {self.data['train_branch_input'].shape[1]}")
            print(f"Trunk input dimension: {self.data['train_trunk_input'].shape[1]}")
    
    def create_model(self):
        """Create model"""
        print("\n=== Model Creation ===")
        
        # Create Hamiltonian
        ham = generate_simple_hamiltonian(self.config['num_qubits'], lower_bound=-5, upper_bound=5)
        print(f"Using Hamiltonian: {ham}")
        # Get input dimensions
        if 'train_branch_input' in self.data:
            branch_input_size = self.data['train_branch_input'].shape[1]
            trunk_input_size = self.data['train_trunk_input'].shape[1]
        else:
            # Infer from configuration
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
        
        # Calculate parameter count
        total_params = count_parameters(self.model)
        print(f"Model type: {model_type}")
        if hasattr(self.model, 'circuit'):
            print(f"Circuit parameters: ")
            self.model.circuit.summary()
        print(f"Trainable frequency: {'Enabled' if if_trainable_freq else 'Disabled'}")
        print(f"Network structure: {self.config['net_size']}")
        print(f"Trainable parameters: {total_params:,}")
        
        return self.model
    
    def train_model(self):
        """Train model"""
        print("\n=== Model Training ===")
        
        if self.model is None:
            raise ValueError("Please create model first")
        
        # Set up training components
        optimizer = nn.Adam(self.model.trainable_params(), learning_rate=self.config['learning_rate'])
        loss_fn = nn.MSELoss()
        
        # Create training network
        net_with_loss = nn.WithLossCell(self.model, loss_fn)
        train_net = nn.TrainOneStepCell(net_with_loss, optimizer)
        
        # Prepare training data
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
        
        # Calculate number of batches
        num_batches_train = max(train_size // batch_size, 1)
        if val_input is not None:
            batch_size_val = min(batch_size, val_size)
            num_batches_val = max(val_size // batch_size_val, 1)
        else:
            batch_size_val = 0
            num_batches_val = 0
        
        print(f"Training parameters:")
        print(f"  Learning rate: {self.config['learning_rate']}")
        print(f"  Max epochs: {self.config['num_epochs']}")
        print(f"  Target error: {self.config['target_error']}")
        print(f"  Batch size: {batch_size}")
        print(f"  Training batches: {num_batches_train}")
        print(f"  Use validation set: {'Yes' if val_input is not None else 'No'}")
        print(f"  Checkpoint interval: {self.checkpoint_interval} epochs")
        
        # Training loop
        best_test_loss = float('inf')
        best_epoch = 0
        patience = 50  # Early stopping patience
        no_improve = 0
        
        print("\nStarting training...")
        for epoch in tqdm(range(self.config['num_epochs']), desc="Training Progress"):
            # Training step - support batch training
            epoch_train_loss = 0.0
            
            # Randomly shuffle training data
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
            
            # Every 10 epochs evaluation (if validation set exists)
            if epoch % 10 == 0:
                if val_input is not None:
                    # Validation evaluation - support batch processing
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
                    
                    # Save best models (based on validation loss)
                    if val_loss < best_test_loss - 1e-6:
                        print(f"  Found better model: training loss: {float(avg_train_loss):.6f}, validation loss from {float(best_test_loss):.6f} to {float(val_loss):.6f}")
                        best_test_loss = val_loss
                        best_epoch = epoch
                        no_improve = 0
                        self.save_model("best", overwrite=True)
                    else:
                        no_improve += 1
                else:
                    # No validation set, save model based on training loss
                    self.training_history.append({
                        'epoch': epoch,
                        'train_loss': avg_train_loss
                    })
                    
                    # Save best models (based on training loss)
                    if avg_train_loss < best_test_loss - 1e-6:
                        print(f"  Found better model: training loss decreased from {float(best_test_loss):.6f} to {float(avg_train_loss):.6f}")
                        best_test_loss = avg_train_loss
                        best_epoch = epoch
                        no_improve = 0
                        self.save_model("best", overwrite=True)
                    else:
                        no_improve += 1
                
                # Regularly save checkpoints
                if epoch > 0 and epoch % self.checkpoint_interval == 0:
                    self.save_model(f"checkpoint_epoch_{epoch}", overwrite=False)
                
                # Print progress every 100 epochs
                if epoch % 100 == 0:
                    if val_input is not None:
                        print(f"Epoch {epoch}: train={avg_train_loss:.6f}, validation={float(val_loss):.6f}, best={'validation' if val_input is not None else 'training'}={float(best_test_loss):.6f}")
                    else:
                        print(f"Epoch {epoch}: train={avg_train_loss:.6f}, best training={float(best_test_loss):.6f}")
                
                # Check convergence
                if avg_train_loss < self.config['target_error']:
                    print(f"Target error {self.config['target_error']} reached at epoch {epoch}")
                    break
                
                # Early stopping
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch}: no improvement for {patience} epochs")
                    break
        
        # Final evaluation on complete test set - use batch processing
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
        
        # Save final models
        self.save_model("final", overwrite=False)
        
        # Display saved models summary
        self.print_saved_models_summary()
        
        return final_test_loss
    
    def save_model(self, suffix="", overwrite=False):
        """
        Improved model saving method
        
        Args:
            suffix: filename suffix
            overwrite: whether to overwrite save (for best models)
        """
        if suffix == "best" and overwrite:
            # Best models: fixed filename, overwrite save
            operator_name = self.operator_type
            if self.config['if_trainable_freq']:
                filename = f"best_{operator_name}_TF-{self.config['model_type']}_{self.config['num_qubits']}_{self.config['net_size']}.ckpt"
            else:
                filename = f"best_{operator_name}_{self.config['model_type']}_{self.config['num_qubits']}_{self.config['net_size']}_seed{self.config['random_seed']}.ckpt"

            # subdir = os.path.join("checkpoints", operator_name)
            subdir = os.path.join("/mnt/nas-new/home/yange/wangruocheng/QON/checkpoints", operator_name)
            os.makedirs(subdir, exist_ok=True)
            filepath = os.path.join(subdir, filename)
            # Delete old best model file
            if self.best_model_path and os.path.exists(self.best_model_path):
                try:
                    os.remove(self.best_model_path)
                    # print(f"    Deleted old best model: {os.path.basename(self.best_model_path)}")
                except Exception as e:
                    print(f"    Failed to delete old model: {e}")
            
            self.best_model_path = filepath
            save_checkpoint(self.model, filepath)
            # print(f"    Saved best model: {os.path.basename(filepath)}")
            
        else:
            # Other models: unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            operator_name = self.operator_type
            filename = f"{operator_name}_{self.config['model_type']}_{suffix}_{timestamp}.ckpt"
            subdir = os.path.join("/mnt/nas-new/home/yange/wangruocheng/QON/checkpoints", operator_name)
            os.makedirs(subdir, exist_ok=True)
            filepath = os.path.join(subdir, filename)

            save_checkpoint(self.model, filepath)
            # print(f"    Saved model: {os.path.basename(filepath)}")
            
            # Record checkpoints (for management)
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
        print(f"\n=== Saved Model Summary ===")
        
        checkpoint_dir = "/mnt/nas-new/home/yange/wangruocheng/QON/checkpoints"
        if not os.path.exists(checkpoint_dir):
            print("No saved models")
            return
        
        saved_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
        
        best_files = [f for f in saved_files if 'best' in f]
        final_files = [f for f in saved_files if 'final' in f]
        checkpoint_files = [f for f in saved_files if 'checkpoint' in f]
        
        print(f"📍 Best models: {len(best_files)} files (should be only 1)")
        for f in best_files:
            print(f"   {f}")
        
        print(f"🏁 Final models: {len(final_files)} files")
        for f in final_files:
            print(f"   {f}")
        
        print(f"⏰ Checkpoints: {len(checkpoint_files)} files (maximum 3 kept)")
        for f in checkpoint_files:
            print(f"   {f}")
        
        print(f"\nModel saving strategy:")
        print(f"  - Checkpoints: saved every {self.checkpoint_interval} epochs, maximum 3 kept")
    
    def evaluate_model(self):
        """Evaluate model performance - support batch processing"""
        print("\n=== Model Evaluation ===")
        
        if self.model is None:
            raise ValueError("Please train model first")
        
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
        print(f"Evaluation batches: {num_batches_eval}")
        
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
        log_subdir = os.path.join("/mnt/nas-new/home/yange/wangruocheng/QON/logs", self.operator_type)
        os.makedirs(log_subdir, exist_ok=True)
        # results_file = f"logs/evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        # if self.config['if_trainable_freq']:
        #     results_file = os.path.join(log_subdir, f"training_{self.operator_type}_TF-{self.config['model_type']}_{self.config['num_qubits']}_{self.config['net_size']}.json")
        # else:
        #     results_file = os.path.join(log_subdir, f"training_{self.operator_type}_{self.config['model_type']}_{self.config['num_qubits']}_{self.config['net_size']}_{self.config['random_seed']}.json")
        results['config'] = self.config
        results['training_history'] = self.training_history
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Evaluation results saved to: {results_file}")
        
        return results
    
    def run_complete_pipeline(self):
        """Run complete training pipeline"""
        print(f"=== {self.config['model_type']} {self.operator_type} PDE Operator Solving Pipeline ===")
        print(f"Start time: {datetime.now()}")
        
        try:
            # 1. Data preparation
            self.load_or_generate_data()
            
            # 2. Model creation
            self.create_model()
            
            # # 3. Model training
            # final_loss = self.train_model()
            
            # 4. Model evaluation
            if self.config['if_trainable_freq']:
                self.best_model_path = f"checkpoints/{self.operator_type}/best_TF-{self.operator_type}_{self.config['model_type']}_{self.config['scale_coeff']}_seed{self.config['random_seed']}.ckpt"
            else:
                self.best_model_path = f"checkpoints/{self.operator_type}/best_{self.operator_type}_{self.config['model_type']}_{self.config['num_qubits']}_seed{self.config['random_seed']}.ckpt"
            print(f"Best model path set to: {self.best_model_path}")
            best_checkpoint = self.best_model_path
            if best_checkpoint and os.path.exists(best_checkpoint):
                print(f"\nLoading best model from {best_checkpoint} for final evaluation...")
                load_checkpoint(best_checkpoint, self.model)
            results = self.evaluate_model()
            
            print(f"\n=== Complete Pipeline Execution Successful ===")
            print(f"End time: {datetime.now()}")
            
            return results
            
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='PDE operator solving')
    parser.add_argument('--operator', type=str, help='PDE Operator Type')
    parser.add_argument('--config', type=str, default='configs/config_PDE.json', help='Configuration file path')
    parser.add_argument('--model_type', type=str, choices=['QuanONet', 'HEAQNN', 'FNN', 'DeepONet'], help='Model type')
    parser.add_argument('--net_size', type=int, nargs='+', help='Network size')
    parser.add_argument('--num_qubits', type=int, help='Number of qubits')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--validation_split', type=float, help='Validation set proportion')
    parser.add_argument('--random_seed', type=int, help='Random seed for reproducible results')
    parser.add_argument('--scale_coeff', type=float, help='Scaling coefficient for the input data')
    parser.add_argument('--if_trainable_freq', type=str, help='Whether to use trainable frequency (true/false)')

    
    args = parser.parse_args()

    print(f"=== PDE Operator {args.model_type} Training System ===")
    print(f"Operator Type: {args.operator}")
    
    
    # Create solver
    solver = PDEOperatorSolver(
        operator_type=args.operator, 
        config_file=args.config,
    )
    
    # Override configuration with command line arguments (only when command line arguments exist)
    if args.model_type:
        solver.config['model_type'] = args.model_type
    if args.net_size:
        solver.config['net_size'] = args.net_size
    if args.num_qubits is not None:
        solver.config['num_qubits'] = args.num_qubits
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

    # Uniformly set random seed (considering command line override and config file)
    final_seed = solver.config.get('random_seed', None)
    set_random_seed(final_seed)
    
    print(f"Using configuration: {solver.config}")
    
    # Run complete pipeline
    results = solver.run_complete_pipeline()
    
    if results:
        print("\n=== Final Result Summary ===")
        print(f"PDE Operator Type: {solver.operator_type}")
        print(f"Model type: {solver.config['model_type']}")
        print(f"MSE: {results['MSE']:.6f}")
        print(f"MAE: {results['MAE']:.6f}")
    else:
        print("Training failed")


if __name__ == "__main__":
    main()
