#!/usr/bin/env python3
"""
DeepXDE-based FNO (Fourier Neural Operator) Training Script

This script trains FNO models for operator learning tasks.
It enforces train_sample_num = num_points to avoid dimension mismatch errors.

Usage:
# Train FNO on Inverse operator
python train_fno.py --operator Inverse --net_size 16 32 3 32 --num_epochs 100
"""

import os
import sys
import json
import logging
import numpy as np
import torch
import argparse
import importlib.util
from datetime import datetime

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
from deepxde.data import Data

# Import shared utilities
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

# --- FIX: Dynamic Import for FNO Model ---
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "core", "dde_models.py")

if os.path.exists(model_path):
    spec = importlib.util.spec_from_file_location("dde_models", model_path)
    dde_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(dde_models)
    FNO1d = dde_models.FNO1d
else:
    raise FileNotFoundError(f"FNO model definition not found at {model_path}")
# ------------------------------------------

# Operator types
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
    """Dataset wrapper for FNO (Input, Output pair)."""
    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x, self.train_y = X_train, y_train
        self.test_x, self.test_y = X_test, y_test

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        return self.train_x, self.train_y

    def test(self):
        return self.test_x, self.test_y

def ODE_fncode(generate_data, num_train, num_test, num_points, train_sample_num, test_sample_num):
    """
    Specialized data encoding for FNO. 
    Reshapes data into (Batch, Spatial_Dim, Channels).
    """
    train_v, train_u, test_v, test_u, _ = generate_data(num_train, num_test)
    
    # --- Interpolation Logic (Ensure input matches num_points) ---
    current_dim = train_v.shape[1]
    if current_dim != num_points:
        print(f"Interpolating data: Input dim {current_dim} -> Target {num_points}")
        
        # Simple slicing if integer multiple
        if current_dim % num_points == 0:
            stride = current_dim // num_points
            train_v = train_v[:, ::stride]
            test_v = test_v[:, ::stride]
            if train_u.shape[1] == current_dim:
                train_u = train_u[:, ::stride]
                test_u = test_u[:, ::stride]
        else:
            # Linear interpolation
            old_x = np.linspace(0, 1, current_dim)
            new_x = np.linspace(0, 1, num_points)
            def interp_batch(data, old_x, new_x):
                return np.array([np.interp(new_x, old_x, d) for d in data]).astype(np.float32)

            train_v = interp_batch(train_v, old_x, new_x)
            test_v = interp_batch(test_v, old_x, new_x)
            if train_u.shape[1] != num_points:
                train_u = interp_batch(train_u, old_x, new_x)
                test_u = interp_batch(test_u, old_x, new_x)
    # -------------------------------------------------------------

    x = np.linspace(0, 1, num_points).astype(np.float32)
    
    def sample_1D_Operator_fndata(v, u, x, sample_num):
        num = u.shape[0]
        num_sensors = u.shape[1]
        
        # FNO expects consistent grid size, usually sample_num should == num_sensors here
        # But keeping logic for flexibility if needed, though strictly used as full grid below
        
        # Reshape x for broadcasting
        x = x.reshape(1, -1) # (1, num_points)
        x = np.repeat(x, num, axis=0) # (num, num_points)
        x = np.expand_dims(x, axis=2) # (num, num_points, 1)
        v = np.expand_dims(v, axis=2) # (num, num_points, 1)
        
        # Input: [Function_Value, Coordinate] -> (Batch, Points, 2)
        input = np.concatenate((v, x), axis=2)
        
        # Output processing
        # Ensure output matches sample_num (which is forced to num_points in Solver)
        if u.shape[1] != sample_num:
             # If strictly forced, this shouldn't happen often if interpolation worked, 
             # but as a safeguard we take first sample_num points or interpolate
             if u.shape[1] > sample_num:
                 u = u[:, :sample_num]
        
        output = np.expand_dims(u, axis=2) # (Batch, Sample_Num, 1)
        
        # Indices are not strictly used by standard FNO1d but kept for compatibility
        indices = np.zeros((num, sample_num), dtype=np.int64) 
        
        return input.astype(np.float32), indices, output.astype(np.float32)

    train_input, train_indices, train_output = sample_1D_Operator_fndata(train_v, train_u, x, train_sample_num)
    test_input, test_indices, test_output = sample_1D_Operator_fndata(test_v, test_u, x, test_sample_num)
    
    return train_input, train_indices, train_output, test_input, test_indices, test_output


class FNOOperatorSolver:
    """Solver specialized for FNO models"""

    def __init__(self, operator_type='Inverse', config_file=None, prefix=None):
        if operator_type not in DDE_OPERATOR_TYPES:
            raise ValueError(f"Unsupported operator type: {operator_type}")

        self.operator_type = operator_type
        self.model_type = "FNO"
        
        self.config = self.load_config(config_file)
        self.config['operator_type'] = operator_type
        self.config['model_type'] = "FNO"

        # --- FIX: Setup Data Generator with correct args ---
        num_points = self.config.get('num_points', 100)
        num_points_0 = self.config.get('num_points_0', 1000)
        
        if operator_type in ['Inverse', 'Homogeneous', 'Nonlinear']:
            self.data_generator = lambda n_tr, n_te: DDE_OPERATOR_TYPES[operator_type](operator_type, n_tr, n_te, num_points, num_points_0)
        else:
            self.data_generator = lambda n_tr, n_te: DDE_OPERATOR_TYPES[operator_type](n_tr, n_te, operator=operator_type)
        # --------------------------------------------------

        self.model = None
        self.data = {}
        self.training_history = []
        self.param_num = 0

        # Directories
        self.prefix = prefix
        self.logs_dir = os.path.join(self.prefix, "logs") if self.prefix else "logs"
        self.checkpoints_dir = os.path.join(self.prefix, "checkpoints") if self.prefix else "checkpoints"
        self.data_dir = os.path.join(self.prefix, "data") if self.prefix else "data"
        self.dairy_dir = os.path.join(self.prefix, "dairy") if self.prefix else "dairy"

        for d in [self.logs_dir, self.checkpoints_dir, self.data_dir, self.dairy_dir]:
            os.makedirs(d, exist_ok=True)

    def load_config(self, config_file=None):
        """Load configuration"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            print(f"Loaded configuration from {config_file}")
        else:
            default_config_file = 'configs/config_ODE.json'
            if os.path.exists(default_config_file):
                with open(default_config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {
                    'num_train': 1000, 'num_test': 1000, 'num_points': 100,
                    'train_sample_num': 100, 'test_sample_num': 100,
                    'learning_rate': 0.001, 'batch_size': 32, 'num_epochs': 1000,
                    'net_size': [16, 32, 3, 32]
                }
        return config

    def load_or_generate_data(self):
        """Load or generate FNO specific data"""
        print(f"\n=== {self.operator_type} FNO Data Preparation ===")
        operator_name = self.operator_type
        
        num_train = self.config.get('num_train', 1000)
        num_test = self.config.get('num_test', 1000)
        num_points = self.config.get('num_points', 100)
        
        # --- CRITICAL FIX: Force sample nums to match num_points for FNO ---
        # FNO requires the output grid to match the model's grid resolution
        train_sample_num = num_points
        test_sample_num = num_points
        
        # Update config to match reality
        self.config['train_sample_num'] = num_points
        self.config['test_sample_num'] = num_points
        print(f"Note: For FNO, train/test_sample_num forced to num_points ({num_points})")
        # -------------------------------------------------------------------

        data_file = f"{self.data_dir}/{operator_name}/{operator_name}_FNO_dataset_{num_train}_{num_test}_{num_points}_{train_sample_num}_{test_sample_num}.npz"

        if os.path.exists(data_file):
            print(f"Loading existing FNO data from {data_file}...")
            try:
                data = np.load(data_file)
                self.data = {
                    'train_input': data['train_input'],
                    'train_output': data['train_output'],
                    'test_input': data['test_input'],
                    'test_output': data['test_output']
                }
                print(f"Data loaded successfully.")
            except Exception as e:
                print(f"Failed to load data: {e}. Generating new data...")
                self.generate_data()
        else:
            print("Generating new FNO data...")
            self.generate_data()

        print(f"Data shapes: Train X {self.data['train_input'].shape}, Y {self.data['train_output'].shape}")

    def generate_data(self):
        """Generate new FNO data using ODE_fncode"""
        num_train = self.config.get('num_train', 1000)
        num_test = self.config.get('num_test', 1000)
        num_points = self.config.get('num_points', 100)
        
        # --- CRITICAL FIX: Ensure generation uses forced sample nums ---
        train_sample_num = num_points
        test_sample_num = num_points
        # ---------------------------------------------------------------

        # For ODE generation, we need to pass correct arguments
        # Re-wrapping generator to match ODE_fncode expectation
        num_points_0 = self.config.get('num_points_0', 1000)
        
        if self.operator_type in ['Inverse', 'Homogeneous', 'Nonlinear']:
             generator_func = lambda n_tr, n_te: generate_ODE_Operator_data(
                 self.operator_type, n_tr, n_te, num_points, num_points_0
             )
        else:
             generator_func = lambda n_tr, n_te: generate_PDE_Operator_data(n_tr, n_te, operator=self.operator_type)

        train_in, _, train_out, test_in, _, test_out = ODE_fncode(
            generator_func, num_train, num_test, num_points, train_sample_num, test_sample_num
        )

        self.data = {
            'train_input': train_in,
            'train_output': train_out,
            'test_input': test_in,
            'test_output': test_out
        }

        operator_name = self.operator_type
        data_file = f"{self.data_dir}/{operator_name}/{operator_name}_FNO_dataset_{num_train}_{num_test}_{num_points}_{train_sample_num}_{test_sample_num}.npz"
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        np.savez(data_file, **self.data)
        print(f"Data saved to {data_file}")

    def create_model(self):
        """Create FNO model"""
        print(f"\n=== Creating FNO Model ===")
        
        # Dataset
        X_train = self.data['train_input'].astype(np.float32)
        y_train = self.data['train_output'].astype(np.float32)
        X_test = self.data['test_input'].astype(np.float32)
        y_test = self.data['test_output'].astype(np.float32)
        
        self.dataset = Double(X_train, y_train, X_test, y_test)

        # Network Config
        net_config = self.config.get('net_size', [16, 32, 3, 32])
        if not isinstance(net_config, (list, tuple)): net_config = [16, 32, 3, 32]
        
        defaults = [16, 32, 3, 32] 
        if len(net_config) < 4:
            net_config = list(net_config) + defaults[len(net_config):]

        modes = int(net_config[0])
        width = int(net_config[1])
        layers = int(net_config[2])
        fc_hidden = int(net_config[3])

        print(f"Initializing FNO: modes={modes}, width={width}, layers={layers}, fc={fc_hidden}")
        
        # Initialize FNO1d
        net = FNO1d(modes=modes, width=width, layers=layers, fc_hidden=fc_hidden)
        
        self.model = dde.Model(self.dataset, net)
        self.param_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f"Model params: {self.param_num}")

    def train_model(self):
        """Train"""
        print(f"\n=== Training FNO on {self.operator_type} ===")
        print(f"Start Time: {datetime.now()}")

        lr = self.config.get('learning_rate', 0.001)
        epochs = self.config.get('num_epochs', 1000)
        batch_size = self.config.get('batch_size', 32)

        self.model.compile("adam", lr=lr, loss="mse")

        # Log file naming
        log_file = os.path.join(self.dairy_dir, f"{self.operator_type}/train_{self.operator_type}_FNO_{self.config['num_train']}*{self.config['num_points']}_{self.config.get('net_size')}_{self.config['random_seed']}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # Logger setup
        logger = logging.getLogger('training_fno')
        logger.setLevel(logging.INFO)
        if logger.hasHandlers(): logger.handlers.clear()
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)

        old_stdout = sys.stdout
        sys.stdout = StreamToLogger(logger, logging.INFO)

        try:
            losshistory, train_state = self.model.train(iterations=epochs, batch_size=batch_size)
            sys.stdout = old_stdout

            # Save history
            self.training_history = {
                'loss': [float(x[0]) for x in losshistory.loss_train],
                'loss_test': [float(x[0]) for x in losshistory.loss_test],
                'iterations': list(range(len(losshistory.loss_train)))
            }
            
            final_loss = losshistory.loss_train[-1][0]
            print(f"Training finished. Final Loss: {final_loss}")
            return final_loss

        except Exception as e:
            sys.stdout = old_stdout
            raise e

    def evaluate_model(self):
        """Evaluate"""
        print(f"\n=== Evaluating FNO ===")
        y_pred = self.model.predict(self.dataset.test_x)
        y_true = self.data['test_output']

        mse = np.mean((y_true - y_pred)**2)
        mae = np.mean(np.abs(y_true - y_pred))
        max_error = np.max(np.abs(y_true - y_pred))

        metrics = {'MSE': float(mse), 'MAE': float(mae), 'Max_Error': float(max_error)}
        print(f"Metrics: {metrics}")

        # Save JSON
        json_file = os.path.join(self.logs_dir, f"{self.operator_type}/eval_{self.operator_type}_FNO_{self.config['num_train']}*{self.config['num_points']}_{self.config.get('net_size')}_{self.config['random_seed']}.json")
        os.makedirs(os.path.dirname(json_file), exist_ok=True)
        
        results = {
            'config': self.config,
            'metrics': metrics,
            'param_num': self.param_num
        }
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Saved results to {json_file}")
        
        # Save Checkpoint
        ckpt_path = os.path.join(self.checkpoints_dir, f"{self.operator_type}/{self.operator_type}_FNO_{self.config.get('num_train')}*{self.config.get('num_points')}_{self.config.get('net_size')}_{self.config.get('random_seed')}/best_{self.operator_type}_FNO_{self.config.get('num_train')}*{self.config.get('num_points')}_{self.config.get('net_size')}_{self.config.get('random_seed')}.ckpt")
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        torch.save(self.model.net.state_dict(), ckpt_path)
        
        return metrics

    def run(self):
        try:
            self.load_or_generate_data()
            self.create_model()
            self.train_model()
            metrics = self.evaluate_model()
            return metrics
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    parser = argparse.ArgumentParser(description='DeepXDE FNO Training')
    
    parser.add_argument('--operator', '-o', type=str, required=True, choices=list(DDE_OPERATOR_TYPES.keys()))
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
    
    parser.add_argument('--net_size', type=int, nargs='+', help='FNO Config: [modes, width, layers, fc_hidden]')

    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if args.seed != 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    solver = FNOOperatorSolver(args.operator, args.config, args.prefix)

    # Apply overrides
    if args.num_train: solver.config['num_train'] = args.num_train
    if args.num_test: solver.config['num_test'] = args.num_test
    if args.num_epochs: solver.config['num_epochs'] = args.num_epochs
    if args.batch_size: solver.config['batch_size'] = args.batch_size
    if args.learning_rate: solver.config['learning_rate'] = args.learning_rate
    if args.net_size: solver.config['net_size'] = args.net_size
    solver.config['random_seed'] = args.seed

    print(f"Config: {solver.config}")
    solver.run()

if __name__ == "__main__":
    sys.exit(main())