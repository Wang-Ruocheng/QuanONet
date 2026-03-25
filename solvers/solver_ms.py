"""
MindSpore Solver for Quantum/Classical Models (QuanONet, HEAQNN).
Refactored from original train.py.
"""
import os
import sys
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

try:
    import mindspore as ms
    import mindspore.nn as nn
    import mindspore.numpy as mnp
    from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
except ImportError:
    ms = None

from data_utils.data_manager import DataManager
from utils.logger import ExperimentLogger, setup_logger, StreamToLogger
from utils.metrics import compute_metrics
from utils.utils import count_parameters 

from core.models import QuanONet, HEAQNN, FNN, DeepONet
from core.quantum_circuits import generate_simple_hamiltonian, ham_diag_to_operator

class MSSolver:
    def __init__(self, config):
        if ms is None:
            raise ImportError("MindSpore is required for MSSolver but not found.")
            
        self.config = config
        self.operator_type = config['operator_type']
        self.model_type = config['model_type']
        
        prefix = config.get('prefix') or "outputs"
        self.exp_logger = ExperimentLogger(config, base_output_dir=prefix)
        self.run_id = self.exp_logger.exp_name
        self.config['run_id'] = self.run_id 

        # 1. Setup Logger
        self.logger = setup_logger(self.exp_logger.text_log_path)
        sys.stdout = StreamToLogger(self.logger) 
        
        # Context Setup
        device_target = "GPU" if config.get('gpu') is not None else "CPU"
        mode = ms.PYNATIVE_MODE 
        gpu_val = config.get('gpu')
        if gpu_val is not None:
             ms.context.set_context(device_id=int(gpu_val))
        ms.context.set_context(mode=mode, device_target=device_target)

        self.logger.info(f"Initialized MSSolver (Quantum) for {self.model_type}")
        self.logger.info(f"Context: {device_target}, Mode: PyNative")

        # 2. Load Data (Unified Manager)
        self.dm = DataManager(config, data_dir=os.path.join(prefix, "..", "data"), logger=self.logger)
        self.data_dict_np = self.dm.get_data() 
        self._convert_data_to_ms() 
        
        # 3. Build Model
        self.model = self._create_model()
        
        # Optimization Setup
        self.optimizer = nn.Adam(self.model.trainable_params(), learning_rate=config['learning_rate'])
        self.loss_fn = nn.MSELoss()
        
        self.best_loss = float('inf')

    def _convert_data_to_ms(self):
        """Converts numpy data from DataManager to MindSpore Tensors."""
        self.logger.info("Routing data for MindSpore models...")
        self.data = self.data_dict_np 
                
        if self.model_type in ['HEAQNN', 'FNN']:
            self.train_input = self.data['train_input']
            self.test_input = self.data['test_input']
        else:
            self.train_input = (self.data['train_branch_input'], self.data['train_trunk_input'])
            self.test_input = (self.data['test_branch_input'], self.data['test_trunk_input'])
        self.train_output = self.data['train_output']
        self.test_output = self.data['test_output']

    def _create_model(self):
        self.logger.info("Creating Quantum Model...")
        ham_bound = self.config.get('ham_bound')
        if not ham_bound: ham_bound = [-5, 5] 
        
        if self.config.get('ham_diag') is None:
            ham = generate_simple_hamiltonian(
                self.config['num_qubits'], 
                lower_bound=ham_bound[0], 
                upper_bound=ham_bound[1], 
                pauli=self.config.get('ham_pauli', 'Z')
            )
        else:
            ham = ham_diag_to_operator(self.config['ham_diag'], self.config['num_qubits'])
            
        if hasattr(ham, 'hamiltonian'):
            self.logger.info(f"Hamiltonian Formula:\n{ham.hamiltonian}")
        else:
            self.logger.info(f"Hamiltonian:\n{ham}")

        net_size = tuple(self.config.get('net_size', [20, 2, 10, 2]))
        if_trainable_freq = str(self.config.get('if_trainable_freq', 'true')).lower() == 'true'

        if self.model_type == 'QuanONet':
            branch_in = self.data['train_branch_input'].shape[1]
            trunk_in = self.data['train_trunk_input'].shape[1]
            model = QuanONet(self.config['num_qubits'], branch_in, trunk_in, net_size, ham, 
                             self.config.get('scale_coeff', 0.01), if_trainable_freq)
        elif self.model_type == 'HEAQNN':
            input_size = self.data['train_input'].shape[1]
            model = HEAQNN(self.config['num_qubits'], input_size, net_size, ham,
                           self.config.get('scale_coeff', 0.01), if_trainable_freq)
        elif self.model_type == 'DeepONet': 
            branch_in = self.data['train_branch_input'].shape[1]
            trunk_in = self.data['train_trunk_input'].shape[1]
            model = DeepONet(branch_in, trunk_in, net_size)
        elif self.model_type == 'FNN': 
            input_size = self.data['train_input'].shape[1]
            model = FNN(input_size, 1, net_size)
        else:
            raise ValueError(f"Unknown MS model: {self.model_type}")
            
        self.logger.info(f"Model Parameters: {count_parameters(model)}")
        return model

    def train(self):
        if self.exp_logger.is_completed():
            print(f"⏩ [Resume] The experiment has been completed and the existing result file has been detected. Skip the training directly.")
            sys.exit(0)
            
        self.logger.info("Starting Training...")
        net_with_loss = nn.WithLossCell(self.model, self.loss_fn)
        train_net = nn.TrainOneStepCell(net_with_loss, self.optimizer)
        train_net.set_train()
        
        epochs = self.config['num_epochs']
        total_samples = len(self.train_output)
        if self.model_type == 'FNO':
            total_samples = self.train_output.shape[0]
        else:
            total_samples = self.config['num_train'] * self.config.get('train_sample_num', 10)

        current_bs = self.config.get('batch_size', 100)
        if total_samples < current_bs:
            self.logger.warning(f"⚠️ Batch size {current_bs} > total samples {total_samples}. Reducing to {total_samples}.")
            self.config['batch_size'] = total_samples
        
        batch_size = self.config.get('batch_size', 100)
        
        if isinstance(self.train_input, tuple):
            num_samples = self.train_input[0].shape[0]
        else:
            num_samples = self.train_input.shape[0]
            
        num_batches = max(1, num_samples // batch_size)
        history = {'loss_train': [], 'loss_test': []}
        
        if self.config.get('init_checkpoint'):
             load_checkpoint(self.config['init_checkpoint'], self.model)
             self.logger.info(f"Loaded init checkpoint: {self.config['init_checkpoint']}")

        if not str(self.config.get('if_train', 'true')).lower() == 'true':
            self.logger.info("Skipping training (if_train=false)")
            return history

        for epoch in tqdm(range(epochs)):
            epoch_loss = 0
            epoch_rel_err = 0
            indices = np.random.permutation(num_samples)
            
            for i in range(num_batches):
                idx = indices[i * batch_size : (i+1) * batch_size]
                
                if isinstance(self.train_input, tuple):
                    batch_in = (ms.Tensor(self.train_input[0][idx], ms.float32), 
                                ms.Tensor(self.train_input[1][idx], ms.float32))
                else:
                    batch_in = ms.Tensor(self.train_input[idx], ms.float32)
                
                batch_out = ms.Tensor(self.train_output[idx], ms.float32)
                
                loss = train_net(batch_in, batch_out)
                loss_val = float(loss.asnumpy())
                epoch_loss += loss_val
                
                n_elements = batch_out.size
                l2_diff = np.sqrt(loss_val * n_elements)
                l2_true = np.linalg.norm(batch_out.asnumpy())
                batch_rel = l2_diff / (l2_true + 1e-8)
                epoch_rel_err += batch_rel
            
            avg_loss = epoch_loss / num_batches
            avg_rel_err = epoch_rel_err / num_batches 
            history['loss_train'].append(avg_loss)
            
            self.exp_logger.log_metric("Loss/train", avg_loss, epoch)
            self.exp_logger.log_metric("Error/rel_l2", avg_rel_err, epoch)
            
            # Save Best
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                if self.config.get('if_save', True):
                    self.best_model_path = self.exp_logger.get_ckpt_path()
                    save_checkpoint(self.model, self.best_model_path)
                    npz_path = self.best_model_path.replace('.ckpt', '.npz')
                    param_dict_np = {}
                    for param in self.model.get_parameters():
                        param_dict_np[param.name] = param.asnumpy()
                    np.savez(npz_path, **param_dict_np)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | MSE: {avg_loss:.6e} | Rel_L2: {avg_rel_err:.4%}")

            if self.config.get('if_save', True):
                final_model_path = self.exp_logger.get_ckpt_path(is_final=True)
                save_checkpoint(self.model, final_model_path)
                
                npz_path = final_model_path.replace('.ckpt', '.npz')
                param_dict_np = {param.name: param.asnumpy() for param in self.model.get_parameters()}
                np.savez(npz_path, **param_dict_np)
                self.logger.info(f"Saved FINAL model to {final_model_path}")   
                
        return history

    def evaluate(self, history=None):
        self.logger.info("Evaluating...")
        if hasattr(self, 'best_model_path') and self.best_model_path and os.path.exists(self.best_model_path):
            param_dict = load_checkpoint(self.best_model_path)
            load_param_into_net(self.model, param_dict)
            self.logger.info(f"Loaded best model from {self.best_model_path}")
        elif self.config.get('ckpt_path') and os.path.exists(self.config['ckpt_path']):
            param_dict = load_checkpoint(self.config['ckpt_path'])
            load_param_into_net(self.model, param_dict)
            self.logger.info(f"Loaded evaluation model from {self.config['ckpt_path']}")
            
        self.model.set_train(False)
        
        batch_size = self.config.get('batch_size', 100)
        preds = []
        
        if isinstance(self.test_input, tuple):
            num_samples = self.test_input[0].shape[0]
        else:
            num_samples = self.test_input.shape[0]
            
        num_batches = int(np.ceil(num_samples / batch_size))
        for i in range(num_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, num_samples)
            
            if isinstance(self.test_input, tuple):
                batch_in = (ms.Tensor(self.test_input[0][start:end], ms.float32), 
                            ms.Tensor(self.test_input[1][start:end], ms.float32))
            else:
                batch_in = ms.Tensor(self.test_input[start:end], ms.float32)
            
            batch_pred = self.model(batch_in)
            preds.append(batch_pred.asnumpy()) 
            
        y_pred_np = np.concatenate(preds, axis=0)
        
        y_true_np = self.test_output 

        l2_diff = np.linalg.norm(y_pred_np - y_true_np)
        l2_true = np.linalg.norm(y_true_np)
        rel_error = l2_diff / (l2_true + 1e-8)
        
        self.logger.info(f"⚡ Test Relative L2 Error: {rel_error:.6f} ({rel_error:.2%})")
        
        metrics = compute_metrics(y_true_np, y_pred_np)
        metrics['rel_l2'] = rel_error 
        self.logger.info(f"Metrics: {metrics}")
        
        self.exp_logger.save_metrics(metrics, history)
        self.exp_logger.close()
        
        return metrics