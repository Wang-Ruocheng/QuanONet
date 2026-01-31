#!/usr/bin/env python3
"""
DeepXDE-based Classical Operator Learning Training Script

This script provides classical baselines (DeepONet, FNN) for operator learning tasks.
It aligns 'num_epochs' with standard deep learning definitions (1 Epoch = 1 pass over all data).

Usage:
# DeepONet (100 epochs, standard batch training)
python train_dde.py --operator Inverse --model DeepONet --num_epochs 100 --batch_size 100

# FNN
python train_dde.py --operator Inverse --model FNN
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import torch.nn as nn
import argparse
from datetime import datetime
from torch.utils.data import BatchSampler, SequentialSampler, RandomSampler

# Set DeepXDE backend
os.environ["DDE_BACKEND"] = "pytorch"

# Compatibility patch for older PyTorch versions
if not hasattr(torch, 'get_default_device'):
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    torch.get_default_device = get_default_device

import deepxde as dde
from deepxde.nn import FNN
from deepxde.data import Data

# Import shared utilities from the main codebase
from data_utils.data_generation import (
    generate_Inverse_Operator_data,
    generate_Homogeneous_Operator_data,
    generate_Nonlinear_Operator_data,
    generate_ODE_Operator_data,
    generate_RDiffusion_Operator_data,
    generate_Advection_Operator_data,
    generate_Darcy_Operator_data,
    generate_PDE_Operator_data
)
from data_utils.data_processing import ODE_encode, PDE_encode

DDE_MODEL_TYPES = ['DeepONet', 'FNN']

# Operator types supported by DeepXDE framework
DDE_OPERATOR_TYPES = {
    'Inverse': generate_Inverse_Operator_data,
    'Homogeneous': generate_Homogeneous_Operator_data,
    'Nonlinear': generate_Nonlinear_Operator_data,
    'RDiffusion': generate_RDiffusion_Operator_data,
    'Advection': generate_Advection_Operator_data,
    'Darcy': generate_Darcy_Operator_data
}

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

class Double(Data):
    """Dataset with each data point as a triple."""
    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        return self.train_x, self.train_y

    def test(self):
        return self.test_x, self.test_y


class DDEOperatorSolver:
    """DeepXDE-based operator learning solver"""

    def __init__(self, operator_type='Inverse', model_type='DeepONet', config_file=None, prefix=None):
        if operator_type not in DDE_OPERATOR_TYPES:
            raise ValueError(f"Unsupported operator type: {operator_type}. Supported types: {list(DDE_OPERATOR_TYPES.keys())}")

        if model_type not in DDE_MODEL_TYPES:
            raise ValueError(f"Unsupported model type: {model_type}. Supported types: {DDE_MODEL_TYPES}")

        self.operator_type = operator_type
        self.model_type = model_type
        self.data_generator = DDE_OPERATOR_TYPES[operator_type]

        self.config = self.load_config(config_file)
        self.config['operator_type'] = operator_type
        self.config['model_type'] = model_type

        self.model = None
        self.data = {}
        self.training_history = []
        self.param_num = 0

        # Create necessary directories
        self.prefix = prefix
        self.logs_dir = os.path.join(self.prefix, "logs") if self.prefix else "logs"
        self.checkpoints_dir = os.path.join(self.prefix, "checkpoints") if self.prefix else "checkpoints"
        self.data_dir = os.path.join(self.prefix, "data") if self.prefix else "data"
        self.dairy_dir = os.path.join(self.prefix, "dairy") if self.prefix else "dairy"

        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.dairy_dir, exist_ok=True)

    def load_config(self, config_file=None):
        """Load configuration from file or use default settings."""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_file}")
        else:
            # Determine default config based on operator type
            if self.operator_type in ['Inverse', 'Homogeneous', 'Nonlinear']:
                default_config_file = 'configs/config_ODE.json'
            elif self.operator_type in ['RDiffusion', 'Advection']:
                default_config_file = 'configs/config_PDE.json'
            elif self.operator_type in ['Darcy']:
                default_config_file = 'configs/config_Darcy.json'
            else:
                default_config_file = 'configs/config_ODE.json' # Fallback

            if os.path.exists(default_config_file):
                with open(default_config_file, 'r') as f:
                    config = json.load(f)
                print(f"Loaded default configuration from {default_config_file}")
            else:
                print(f"Warning: Default config {default_config_file} not found. Using minimal defaults.")
                config = {
                    'num_train': 1000, 'num_test': 1000, 'num_points': 100,
                    'num_points_0': 1000, 'train_sample_num': 10, 'test_sample_num': 100,
                    'num_cal': 1000, 'learning_rate': 0.001, 'batch_size': 100, 'num_epochs': 1000,
                    'branch_input_size': 32, 'trunk_input_size': 32, 'output_size': 1,
                    'net_size': [20, 2, 10, 2] # Default for FNN/DeepONet
                }
        return config

    def load_or_generate_data(self):
        """Load or generate operator data"""
        print(f"\n=== {self.operator_type} Operator Data Preparation ===")
        operator_name = self.operator_type
        
        # Use .get() with defaults for safety
        num_train = self.config.get('num_train', 1000)
        num_test = self.config.get('num_test', 1000)
        num_points = self.config.get('num_points', 100)
        num_points_0 = self.config.get('num_points_0', 1000)
        train_sample_num = self.config.get('train_sample_num', 10)
        test_sample_num = self.config.get('test_sample_num', 100)

        # Include num_points_0 in filename
        data_file = f"{self.data_dir}/{operator_name}/{operator_name}_Operator_dataset_{num_train}_{num_test}_{num_points}_{num_points_0}_{train_sample_num}_{test_sample_num}.npz"

        if os.path.exists(data_file):
            print(f"Loading existing data from {data_file}...")
            try:
                data = np.load(data_file)
                self.data = {
                    'train_branch_input': data['train_branch_input'],
                    'train_trunk_input': data['train_trunk_input'],
                    'train_output': data['train_output'],
                    'test_branch_input': data['test_branch_input'],
                    'test_trunk_input': data['test_trunk_input'],
                    'test_output': data['test_output']
                }
                print(f"Data loaded successfully.")
            except Exception as e:
                print(f"Failed to load data: {e}. Generating new data...")
                self.generate_data()
        else:
            print("Generating new data...")
            self.generate_data()

        self.print_data_info()

    def generate_data(self):
        """Generate new operator data"""
        # Ensure config has required keys
        num_train = self.config.get('num_train', 1000)
        num_test = self.config.get('num_test', 1000)
        num_points = self.config.get('num_points', 100)
        num_points_0 = self.config.get('num_points_0', 1000)
        train_sample_num = self.config.get('train_sample_num', 10)
        test_sample_num = self.config.get('test_sample_num', 100)
        num_cal = self.config.get('num_cal', 1000)

        if self.operator_type in ['Inverse', 'Homogeneous', 'Nonlinear']:
            train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output = ODE_encode(
                self.data_generator, num_train, num_test, num_points, num_points_0, train_sample_num, test_sample_num, num_cal
            )
        else:
            train_branch_input, train_trunk_input, train_output, test_branch_input, test_trunk_input, test_output = PDE_encode(
                self.data_generator, num_train, num_test, num_points, num_points_0, train_sample_num, test_sample_num, num_cal
            )

        self.data = {
            'train_branch_input': train_branch_input,
            'train_trunk_input': train_trunk_input,
            'train_output': train_output,
            'test_branch_input': test_branch_input,
            'test_trunk_input': test_trunk_input,
            'test_output': test_output
        }

        operator_name = self.operator_type
        # Include num_points_0 in filename
        data_file = f"{self.data_dir}/{operator_name}/{operator_name}_Operator_dataset_{num_train}_{num_test}_{num_points}_{num_points_0}_{train_sample_num}_{test_sample_num}.npz"
        
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        np.savez(data_file, **self.data)
        print(f"Data saved to {data_file}")

    def print_data_info(self):
        print("Data shapes:")
        for key, value in self.data.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: {value.shape}")

    def create_model(self):
        """Create DeepXDE model"""
        print(f"\n=== Creating {self.model_type} Model ===")

        # Prepare Dataset
        if self.model_type == 'DeepONet':
            X_train = (self.data['train_branch_input'], self.data['train_trunk_input'])
            y_train = self.data['train_output']
            X_test = (self.data['test_branch_input'], self.data['test_trunk_input'])
            y_test = self.data['test_output']
            
        elif self.model_type == 'FNN':
            X_train = np.concatenate([self.data['train_branch_input'], self.data['train_trunk_input']], axis=1)
            y_train = self.data['train_output']
            X_test = np.concatenate([self.data['test_branch_input'], self.data['test_trunk_input']], axis=1)
            y_test = self.data['test_output']

        self.dataset = Double(X_train, y_train, X_test, y_test)

        # Build Network
        if self.model_type == 'DeepONet':
            m = self.data['train_branch_input'].shape[1] 
            dim_x = 1 if self.operator_type in ['Inverse', 'Homogeneous', 'Nonlinear'] else 2
            
            # --- Dynamic Net Structure for DeepONet ---
            net_size = self.config.get('net_size')
            if net_size and len(net_size) >= 4:
                bd, bw, td, tw = net_size[:4]
                layer_size_branch = [m] + [bw] * bd
                layer_size_trunk = [dim_x] + [tw] * td
                # Force matching latent dim
                if layer_size_branch[-1] != layer_size_trunk[-1]:
                    print(f"Warning: Forcing Trunk latent dim to match Branch: {layer_size_branch[-1]}")
                    layer_size_trunk[-1] = layer_size_branch[-1]
            else:
                print("Using default [20, 32, 20, 32] DeepONet structure.")
                layer_size_branch = [m] + [32] * 20
                layer_size_trunk = [dim_x] + [32] * 20

            print(f"DeepONet Branch: {layer_size_branch}")
            print(f"DeepONet Trunk:  {layer_size_trunk}")

            net = dde.nn.DeepONet(
                layer_size_branch,
                layer_size_trunk,
                "relu",
                "Glorot normal",
            )
            # ------------------------------------------

        elif self.model_type == 'FNN':
            input_size = self.data['train_branch_input'].shape[1] + self.data['train_trunk_input'].shape[1]
            output_size = self.config.get('output_size', 1)
            # FNN uses net_size as hidden layers list
            hidden_layers = self.config.get('net_size', [20, 2, 10, 2])
            
            net = FNN([input_size] + hidden_layers + [output_size], "relu", "Glorot normal")

        self.model = dde.Model(self.dataset, net)
        self.param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Model created with {self.param_num} trainable parameters")

    def train_model(self):
        """Train the model with Epoch Alignment"""
        print(f"\n=== Training {self.model_type} on {self.operator_type} Operator ===")
        print(f"Start Time: {datetime.now()}")

        lr = self.config.get('learning_rate', 0.001)
        epochs = self.config.get('num_epochs', 1000) # This is TARGET EPOCHS
        batch_size = self.config.get('batch_size', 32)

        # --- FIX: Calculate total iterations (steps) ---
        num_samples = self.data['train_output'].shape[0]
        # Calculate steps per epoch (rounding up)
        steps_per_epoch = int(np.ceil(num_samples / batch_size))
        total_iterations = epochs * steps_per_epoch
        
        print(f"Training Config:")
        print(f"  Total Samples:   {num_samples}")
        print(f"  Batch Size:      {batch_size}")
        print(f"  Steps per Epoch: {steps_per_epoch}")
        print(f"  Target Epochs:   {epochs}")
        print(f"  Total Steps:     {total_iterations}")
        # -----------------------------------------------

        self.model.compile("adam", lr=lr, loss="mse")

        log_file = os.path.join(self.dairy_dir, f"{self.operator_type}/train_{self.operator_type}_{self.model_type}_{self.config.get('num_train')}*{self.config.get('train_sample_num')}_{self.config.get('net_size')}_{self.config.get('random_seed')}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        # Logging setup
        logger = logging.getLogger('training')
        logger.setLevel(logging.INFO)
        if logger.hasHandlers():
            logger.handlers.clear()
            
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        
        # Output to console
        ch = logging.StreamHandler(sys.__stdout__)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

        old_stdout = sys.stdout
        sys.stdout = StreamToLogger(logger, logging.INFO)

        try:
            # Use total_iterations instead of epochs
            losshistory, train_state = self.model.train(iterations=total_iterations, batch_size=batch_size)
            sys.stdout = old_stdout

            self.training_history = {
                'loss': losshistory.loss_train,
                'loss_test': losshistory.loss_test if hasattr(losshistory, 'loss_test') else None,
                'iterations': list(range(len(losshistory.loss_train)))
            }

            final_loss = losshistory.loss_train[-1]
            if isinstance(final_loss, (list, np.ndarray)):
                final_loss = float(final_loss[0]) if len(final_loss) > 0 else float(final_loss)
            print(f"Training completed. Final loss: {final_loss:.6f}")
            return final_loss

        except Exception as e:
            sys.stdout = old_stdout
            raise e

    def evaluate_model(self):
        """Evaluate the model"""
        print(f"\n=== Evaluating {self.model_type} ===")
        
        # 预测
        y_pred = self.model.predict(self.dataset.test_x)
        y_true = self.data['test_output']
        
        # --- FIX: 确保两者都是一维向量，避免 (N,1)-(N,) 产生 (N,N) 矩阵 ---
        y_pred_flat = y_pred.flatten()
        y_true_flat = y_true.flatten()
        
        mse = np.mean((y_true_flat - y_pred_flat)**2)
        mae = np.mean(np.abs(y_true_flat - y_pred_flat))
        max_error = np.max(np.abs(y_true_flat - y_pred_flat))
        # ----------------------------------------------------------------

        metrics = {'MSE': float(mse), 'MAE': float(mae), 'Max_Error': float(max_error)}
        print(f"Metrics: {metrics}")

        # Save JSON results
        results_file = os.path.join(self.logs_dir, f"{self.operator_type}/eval_{self.describe}.json")
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        results = {
            'config': self.config,
            'metrics': metrics,
            # 'history': ... (omitted for brevity)
        }
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save Checkpoint
        ckpt_path = os.path.join(self.checkpoints_dir, f"{self.operator_type}/{self.operator_type}_{self.model_type}_{self.config.get('num_train')}*{self.config.get('num_points')}_{self.config.get('net_size')}_{self.config.get('random_seed')}/best_{self.operator_type}_{self.model_type}_{self.config.get('num_train')}*{self.config.get('num_points')}_{self.config.get('net_size')}_{self.config.get('random_seed')}.ckpt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(self.model.net.state_dict(), ckpt_path)
        
        return metrics

    def run(self):
        # Description string for filenames
        self.describe = f"{self.operator_type}_{self.model_type}_{self.config.get('num_train')}*{self.config.get('num_points')}_{self.config.get('net_size', 'default')}_{self.config.get('random_seed')}"

        try:
            self.load_or_generate_data()
            self.create_model()
            self.train_model()
            metrics = self.evaluate_model()
            return metrics
        except Exception as e:
            print(f"Pipeline execution failed: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description='DeepXDE Operator Learning')
    
    # Required args
    parser.add_argument('--operator', '-o', type=str, required=True, choices=list(DDE_OPERATOR_TYPES.keys()))
    parser.add_argument('--model_type', '-m', type=str, required=True, choices=DDE_MODEL_TYPES)
    
    # Optional configs
    parser.add_argument('--config', '-c', type=str, default=None)
    parser.add_argument('--prefix', '-p', type=str, default=None)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--gpu', '-g', type=int, default=None)
    
    # Overrides
    parser.add_argument('--num_train', type=int)
    parser.add_argument('--num_test', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--learning_rate', type=float)
    
    parser.add_argument('--num_points', type=int, help='Number of sensors/grid points (e.g. 100)')
    parser.add_argument('--num_points_0', type=int, help='High resolution grid points (e.g. 1000)')
    parser.add_argument('--num_cal', type=int, help='Calculation grid resolution')
    parser.add_argument('--train_sample_num', type=int, help='Points per sample for training')
    parser.add_argument('--test_sample_num', type=int, help='Points per sample for testing')
    
    # Unified net_size argument
    parser.add_argument('--net_size', type=int, nargs='+', 
                       help='Network architecture list. For FNN: [hidden1, hidden2, ...].')

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.seed != 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    solver = DDEOperatorSolver(args.operator, args.model_type, args.config, args.prefix)

    # Apply overrides
    if args.num_train: solver.config['num_train'] = args.num_train
    if args.num_test: solver.config['num_test'] = args.num_test
    if args.num_epochs: solver.config['num_epochs'] = args.num_epochs
    if args.batch_size: solver.config['batch_size'] = args.batch_size
    if args.learning_rate: solver.config['learning_rate'] = args.learning_rate
    
    if args.num_points: solver.config['num_points'] = args.num_points
    if args.num_points_0: solver.config['num_points_0'] = args.num_points_0
    if args.num_cal: solver.config['num_cal'] = args.num_cal
    if args.train_sample_num: solver.config['train_sample_num'] = args.train_sample_num
    if args.test_sample_num: solver.config['test_sample_num'] = args.test_sample_num
    
    # Apply net_size override
    if args.net_size:
        solver.config['net_size'] = args.net_size

    solver.config['random_seed'] = args.seed


    print(f"Config: {solver.config}")
    solver.run()

if __name__ == "__main__":
    sys.exit(main())