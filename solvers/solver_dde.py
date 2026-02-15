"""
Unified Solver for PyTorch/DeepXDE based models (DeepONet, FNN, FNO).
"""
import os
import sys
import torch
import numpy as np
import deepxde as dde
from deepxde.data import Data
# 关键修复：必须导入 BatchSampler
from deepxde.data.sampler import BatchSampler 

from data_utils.data_manager import DataManager
from utils.logger import setup_logger, StreamToLogger, save_results, get_experiment_id
from utils.metrics import compute_metrics
from utils.common import set_random_seed

# ==========================================
# Helper Class: DeepXDE Dataset Wrapper
# ==========================================
class Double(Data):
    """
    A simple DeepXDE dataset wrapper that passes pre-loaded numpy arrays.
    Supports correct Batch Sampling for both single input (FNN) and tuple input (DeepONet).
    """
    def __init__(self, X_train, y_train, X_test, y_test):
        self.train_x = X_train
        self.train_y = y_train
        self.test_x = X_test
        self.test_y = y_test
        
        # Initialize BatchSampler
        self.train_sampler = BatchSampler(len(self.train_y), shuffle=True)

    def losses(self, targets, outputs, loss_fn, inputs, model, aux=None):
        return loss_fn(targets, outputs)

    def train_next_batch(self, batch_size=None):
        # [Fix] Use Sampler to return mini-batches
        if batch_size is None:
            return self.train_x, self.train_y
        
        indices = self.train_sampler.get_next(batch_size)
        
        if isinstance(self.train_x, tuple):
            # 如果是 (Branch, Trunk)，则分别对两个输入进行切片
            X_batch = (self.train_x[0][indices], self.train_x[1][indices])
        else:
            # 如果是普通 Tensor (FNN/FNO)，直接切片
            X_batch = self.train_x[indices]
            
        return X_batch, self.train_y[indices]

    def test(self):
        return self.test_x, self.test_y
    
# ==========================================
# Main Solver Class
# ==========================================
class DDESolver:
    def __init__(self, config):
        self.config = config
        self.model_type = config['model_type']
        self.operator_type = config['operator_type']
        
        # 1. Setup Directories & Logger
        prefix = config.get('prefix') or "."
        self.logs_dir = os.path.join(prefix, "logs", self.operator_type)
        self.ckpt_dir = os.path.join(prefix, "checkpoints", self.operator_type)
        self.dairy_dir = os.path.join(prefix, "dairy", self.operator_type)
        
        net_info = ""
        if config.get('net_size'):
            net_info = "_Net" + "-".join(map(str, config.get('net_size')))
            
        self.run_id = get_experiment_id(config)
        self.config['run_id'] = self.run_id
        
        log_path = os.path.join(self.dairy_dir, f"train_{self.run_id}.log")
        self.logger = setup_logger(log_path)
        
        # Redirect stdout to logger
        sys.stdout = StreamToLogger(self.logger)
        self.logger.info(f"Initialized DDESolver for {self.model_type} on {self.operator_type}")
        self.logger.info(f"Config: {config}")

        # 2. Load Data
        self.dm = DataManager(config, data_dir=os.path.join(prefix, "data"), logger=self.logger)
        self.data_dict = self.dm.get_data()
        
        # 3. Build Model
        self.model = self._build_model()
        
    def _build_model(self):
        """Constructs the DeepXDE Model based on model_type."""
        self.logger.info(f"Building {self.model_type} model...")
        
        # --- Prepare Data Inputs based on Model Type ---
        if self.model_type == 'DeepONet':
            X_train = (self.data_dict['train_branch_input'], self.data_dict['train_trunk_input'])
            X_test  = (self.data_dict['test_branch_input'], self.data_dict['test_trunk_input'])
            y_train = self.data_dict['train_output']
            y_test  = self.data_dict['test_output']
            
            # Architecture Dimensions
            net_config = self.config.get('net_size')
            m = X_train[0].shape[1]      # Branch input dim
            dim_x = X_train[1].shape[1]  # Trunk input dim
            b_depth, b_width = 2, 10
            t_depth, t_width = 2, 10
            last_layer_size = None  # p (latent dimension)

            if not net_config:
                pass # Use defaults
                
            elif len(net_config) == 2:
                b_depth = t_depth = net_config[0]
                b_width = t_width = net_config[1]
                last_layer_size = b_width 
                
            elif len(net_config) == 4:
                b_depth, b_width = net_config[0], net_config[1]
                t_depth, t_width = net_config[2], net_config[3]
                if b_width == t_width:
                    last_layer_size = b_width
                else:
                    last_layer_size = min(b_width, t_width)
                    self.logger.warning(f"⚠️ Width mismatch ({b_width} vs {t_width}). Auto-setting output dim p={last_layer_size}.")

            elif len(net_config) == 5:
                b_depth, b_width = net_config[0], net_config[1]
                t_depth, t_width = net_config[2], net_config[3]
                last_layer_size = net_config[4]
            else:
                self.logger.warning(f"Net size {net_config} not recognized. Using default [2, 32].")

            
            # Branch Net Construction
            if last_layer_size and len(net_config) == 5:
                layer_size_branch = [m] + [b_width] * (b_depth - 1) + [last_layer_size]
                layer_size_trunk  = [dim_x] + [t_width] * (t_depth - 1) + [last_layer_size]
            else:
                layer_size_branch = [m] + [b_width] * b_depth
                layer_size_trunk  = [dim_x] + [t_width] * t_depth
            
            self.logger.info(f"DeepONet Config: Branch({b_depth}x{b_width}), Trunk({t_depth}x{t_width}), P={last_layer_size if last_layer_size else 'Auto'}")
            self.logger.info(f"  - Branch Layers: {layer_size_branch}")
            self.logger.info(f"  - Trunk Layers:  {layer_size_trunk}")
            
            net = dde.nn.DeepONet(layer_size_branch, layer_size_trunk, "relu", "Glorot normal")
        elif self.model_type == 'FNO':
            X_train = self.data_dict['train_input'].astype(np.float32)
            y_train = self.data_dict['train_output'].astype(np.float32)
            X_test  = self.data_dict['test_input'].astype(np.float32)
            y_test  = self.data_dict['test_output'].astype(np.float32)

            from core.dde_models import FNO1d
            
            # 1. 获取用户输入，如果没有则使用空列表
            user_cfg = self.config.get('net_size', [])
            
            # 2. 智能解析参数 (Modes, Width, Depth, FC_Hidden)
            # 默认值: modes=15, width=14, depth=3, fc_hidden=32
            modes = user_cfg[0] if len(user_cfg) > 0 else 15
            width = user_cfg[1] if len(user_cfg) > 1 else 14
            depth = user_cfg[2] if len(user_cfg) > 2 else 3
            fc_hidden = user_cfg[3] if len(user_cfg) > 3 else 32  # 如果没传第4个，就用默认32

            self.logger.info(f"FNO Config: modes={modes}, width={width}, depth={depth}, fc_hidden={fc_hidden}")
            
            # 3. 传入解析后的变量
            net = FNO1d(modes=modes, width=width, layers=depth, fc_hidden=fc_hidden)

        elif self.model_type == 'FNN':
            X_train = self.data_dict['train_input']
            X_test  = self.data_dict['test_input']
            y_train = self.data_dict['train_output']
            y_test  = self.data_dict['test_output']
            
            input_dim = X_train.shape[1]
            output_dim = 1
            hidden_layers = self.config.get('net_size', [20, 20, 20])
            
            layer_sizes = [input_dim] + hidden_layers + [output_dim]
            self.logger.info(f"FNN Structure: {layer_sizes}")
            net = dde.nn.FNN(layer_sizes, "relu", "Glorot normal")
            
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        self.logger.info(f"Model Parameters: {pytorch_total_params}")
        # ------------------------------------

        # Wrap in DDE Model
        dataset = Double(X_train, y_train, X_test, y_test)
        model = dde.Model(dataset, net)
        return model

    def train(self):
        """Execute training loop."""
        json_path = os.path.join(self.logs_dir, f"eval_{self.run_id}.json")
        if os.path.exists(json_path):
            print(f"⏩ [Resume] The experiment has been completed and {json_path} has been detected. Skip the training directly.")
            sys.exit(0)
        lr = self.config.get('learning_rate', 0.0001)
        epochs = self.config.get('num_epochs', 1000)
        batch_size = self.config.get('batch_size', 100)
        
        # Calculate iterations
        num_samples = self.data_dict['train_output'].shape[0]
        # Avoid division by zero if batch_size > num_samples
        steps_per_epoch = max(1, int(np.ceil(num_samples / batch_size)))
        total_iterations = epochs * steps_per_epoch
        
        self.logger.info(f"Training Start. Epochs: {epochs}, BS: {batch_size}, Total Iter: {total_iterations}")
        
        self.model.compile(
                            "adam", 
                            lr=lr, 
                            metrics=["l2 relative error"]  # <--- 核心修改：添加这个列表
                        )
        # Note: model.train uses 'iterations', not epochs
        losshistory, train_state = self.model.train(iterations=total_iterations, batch_size=batch_size)
        
        history = {
            'loss_train': [float(x) for x in losshistory.loss_train],
            'loss_test': [float(x) for x in losshistory.loss_test],
            'steps': [int(x) for x in losshistory.steps]
        }
        return history

    def evaluate(self, history=None):
        """Evaluate and Save."""
        self.logger.info("Evaluating model...")
        
        # 1. 获取测试数据和预测结果
        X_test, y_true = self.model.data.test()
        y_pred = self.model.predict(X_test)
        
        # --- 🔴 新增：计算 Relative L2 Error ---
        # 显式计算相对误差，确保与 MS 版本逻辑一致
        l2_diff = np.linalg.norm(y_pred - y_true)
        l2_true = np.linalg.norm(y_true)
        rel_error = l2_diff / (l2_true + 1e-8)
        
        self.logger.info(f"⚡ Test Relative L2 Error: {rel_error:.6f} ({rel_error:.2%})")
        # -------------------------------------
        
        # 2. 计算其他指标并将 rel_l2 加入字典
        metrics = compute_metrics(y_true, y_pred)
        metrics['rel_l2'] = float(rel_error)  # <--- 将其存入 metrics 字典
        
        self.logger.info(f"Final Metrics: {metrics}")
        
        # 3. 保存结果
        save_results(self.config, metrics, history, self.logs_dir, filename_suffix="eval")
        
        # 4. 保存 Checkpoint
        ckpt_name = f"best_{self.run_id}.ckpt"
        ckpt_path = os.path.join(self.ckpt_dir, ckpt_name)
        os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
        
        # 处理可能的 DataParallel 包装
        if hasattr(self.model.net, 'module'):
            torch.save(self.model.net.module.state_dict(), ckpt_path)
        elif hasattr(self.model.net, 'state_dict'):
            torch.save(self.model.net.state_dict(), ckpt_path)
            
        self.logger.info(f"Model checkpoint saved to {ckpt_path}")
        
        return metrics