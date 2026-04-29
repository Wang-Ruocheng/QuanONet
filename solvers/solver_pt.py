"""
Standalone PyTorch solver for TorchQuantum / Qiskit quantum models.

No DeepXDE dependency. Mirrors MSSolver structure: DataManager, ExperimentLogger,
Adam+MSELoss, tqdm, best-checkpoint saved as .pt + .npz.
"""
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from data_utils.data_manager import DataManager
from utils.logger import ExperimentLogger, setup_logger, StreamToLogger
from utils.metrics import compute_metrics
from utils.utils import count_parameters


class PTSolver:
    """
    PyTorch training loop for QuanONetPT / HEAQNNPT models.

    Used when quantum_backend is 'torchquantum' or 'qiskit'.
    """
    def __init__(self, config):
        self.config = config
        self.model_type = config['model_type']
        self.operator_type = config['operator_type']
        self.quantum_backend = config.get('quantum_backend', 'torchquantum')

        prefix = config.get('prefix') or "outputs"
        self.exp_logger = ExperimentLogger(config, base_output_dir=prefix)
        self.run_id = self.exp_logger.exp_name
        self.config['run_id'] = self.run_id

        self.logger = setup_logger(self.exp_logger.text_log_path)
        sys.stdout = StreamToLogger(self.logger)

        # Device
        if config.get('gpu') is not None:
            self.device = torch.device(f"cuda:{config['gpu']}")
        elif torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.logger.info(f"Initialized PTSolver ({self.quantum_backend}) for "
                         f"{self.model_type} on {self.operator_type}")
        self.logger.info(f"Device: {self.device}")

        # Data
        self.dm = DataManager(config, data_dir=os.path.join(prefix, "..", "data"),
                              logger=self.logger)
        self.data_dict = self.dm.get_data()
        self._setup_data()

        # Model
        self.model = self._create_model().to(self.device)
        self.logger.info(f"Model Parameters: {count_parameters(self.model)}")

        # Optimisation
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=config['learning_rate']
        )
        self.loss_fn = nn.MSELoss()
        self.best_loss = float('inf')
        self.best_model_path = None

    # ------------------------------------------------------------------
    # Data setup
    # ------------------------------------------------------------------
    def _setup_data(self):
        d = self.data_dict
        if self.model_type in ['HEAQNN']:
            self.train_input = d['train_input']       # (N, input_dim)
            self.test_input  = d['test_input']
        else:  # QuanONet
            self.train_input = (d['train_branch_input'], d['train_trunk_input'])
            self.test_input  = (d['test_branch_input'], d['test_trunk_input'])
        self.train_output = d['train_output']         # (N, out_dim) or (N,)
        self.test_output  = d['test_output']

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------
    def _create_model(self):
        self.logger.info("Creating PyTorch quantum model...")
        ham_bound = tuple(self.config.get('ham_bound', [-5, 5]))
        ham_diag  = self.config.get('ham_diag', None)
        net_size  = tuple(self.config.get('net_size', [20, 2, 10, 2]))
        if_tf     = str(self.config.get('if_trainable_freq', 'true')).lower() == 'true'
        scale     = float(self.config.get('scale_coeff', 0.01))
        n_qubits  = int(self.config['num_qubits'])

        from core.models_pt import QuanONetPT, HEAQNNPT

        if self.model_type == 'QuanONet':
            branch_in = self.data_dict['train_branch_input'].shape[1]
            trunk_in  = self.data_dict['train_trunk_input'].shape[1]
            model = QuanONetPT(
                num_qubits=n_qubits,
                branch_input_size=branch_in,
                trunk_input_size=trunk_in,
                net_size=net_size,
                scale_coeff=scale,
                if_trainable_freq=if_tf,
                quantum_backend=self.quantum_backend,
                ham_bound=ham_bound,
                ham_diag=ham_diag,
            )
        elif self.model_type == 'HEAQNN':
            input_size = self.data_dict['train_input'].shape[1]
            model = HEAQNNPT(
                num_qubits=n_qubits,
                input_size=input_size,
                net_size=net_size,
                scale_coeff=scale,
                if_trainable_freq=if_tf,
                quantum_backend=self.quantum_backend,
                ham_bound=ham_bound,
                ham_diag=ham_diag,
            )
        else:
            raise ValueError(f"PTSolver does not support model_type='{self.model_type}'")
        return model

    # ------------------------------------------------------------------
    # Helper: convert batch to tensors on device
    # ------------------------------------------------------------------
    def _to_tensor(self, arr):
        if isinstance(arr, np.ndarray):
            return torch.tensor(arr, dtype=torch.float32, device=self.device)
        return arr.to(self.device)

    def _batch_input(self, inp, idx):
        if isinstance(inp, tuple):
            return tuple(self._to_tensor(x[idx]) for x in inp)
        return self._to_tensor(inp[idx])

    def _model_forward(self, inp_batch):
        if isinstance(inp_batch, tuple):
            return self.model(inp_batch[0], inp_batch[1])
        return self.model(inp_batch)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self):
        if self.exp_logger.is_completed():
            print("⏩ [Resume] Experiment already completed. Skipping training.")
            sys.exit(0)

        self.logger.info("Starting Training...")

        if isinstance(self.train_input, tuple):
            num_samples = self.train_input[0].shape[0]
        else:
            num_samples = self.train_input.shape[0]

        batch_size = self.config.get('batch_size', 100)
        if num_samples < batch_size:
            self.logger.warning(
                f"⚠️ Batch size {batch_size} > total samples {num_samples}. "
                f"Reducing to {num_samples}."
            )
            batch_size = num_samples

        epochs = self.config['num_epochs']
        num_batches = max(1, num_samples // batch_size)
        history = {'loss_train': [], 'loss_test': []}

        ckpt_path = self.exp_logger.get_ckpt_path(is_final=False)
        self.best_model_path = ckpt_path.replace('.ckpt', '.pt')

        for epoch in tqdm(range(epochs)):
            self.model.train()
            indices = np.random.permutation(num_samples)
            epoch_loss = 0.0
            epoch_rel_err = 0.0

            for i in range(num_batches):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                inp_batch = self._batch_input(self.train_input, idx)
                out_batch = self._to_tensor(self.train_output[idx])
                if out_batch.dim() == 1:
                    out_batch = out_batch.unsqueeze(-1)

                self.optimizer.zero_grad()
                pred = self._model_forward(inp_batch)
                loss = self.loss_fn(pred, out_batch)
                loss.backward()
                self.optimizer.step()

                loss_val = loss.item()
                epoch_loss += loss_val
                rel = (np.sqrt(loss_val * out_batch.numel()) /
                       (out_batch.norm().item() + 1e-8))
                epoch_rel_err += rel

            avg_loss = epoch_loss / num_batches
            avg_rel  = epoch_rel_err / num_batches
            history['loss_train'].append(avg_loss)

            self.exp_logger.log_metric("Loss/train", avg_loss, epoch)
            self.exp_logger.log_metric("Error/rel_l2", avg_rel, epoch)

            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                if self.config.get('if_save', True):
                    state_dict = self.model.state_dict()
                    torch.save(state_dict, self.best_model_path)
                    npz_path = self.best_model_path.replace('.pt', '.npz')
                    np.savez(npz_path,
                             **{k: v.cpu().numpy() for k, v in state_dict.items()})

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | MSE: {avg_loss:.6e} | Rel_L2: {avg_rel:.4%}")

        # Save final checkpoint
        if self.config.get('if_save', True):
            final_path = self.exp_logger.get_ckpt_path(is_final=True).replace('.ckpt', '.pt')
            state_dict = self.model.state_dict()
            torch.save(state_dict, final_path)
            np.savez(final_path.replace('.pt', '.npz'),
                     **{k: v.cpu().numpy() for k, v in state_dict.items()})
            self.logger.info(f"Saved FINAL model to {final_path}")

        return history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, history=None):
        self.logger.info("Evaluating...")

        if (self.best_model_path and os.path.exists(self.best_model_path)):
            state_dict = torch.load(self.best_model_path,
                                    map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.logger.info(f"Loaded best model from {self.best_model_path}")

        self.model.eval()
        preds = []

        if isinstance(self.test_input, tuple):
            num_test = self.test_input[0].shape[0]
        else:
            num_test = self.test_input.shape[0]

        batch_size = self.config.get('batch_size', 100)
        num_batches = int(np.ceil(num_test / batch_size))

        with torch.no_grad():
            for i in range(num_batches):
                start = i * batch_size
                end   = min((i + 1) * batch_size, num_test)
                if isinstance(self.test_input, tuple):
                    idx = slice(start, end)
                    inp = tuple(self._to_tensor(x[idx]) for x in self.test_input)
                    pred = self.model(inp[0], inp[1])
                else:
                    inp = self._to_tensor(self.test_input[start:end])
                    pred = self.model(inp)
                preds.append(pred.cpu().numpy())

        y_pred = np.concatenate(preds, axis=0)
        y_true = self.test_output
        if y_true.ndim == 1:
            y_true = y_true[:, None]

        l2_diff  = np.linalg.norm(y_pred - y_true)
        l2_true  = np.linalg.norm(y_true)
        rel_error = l2_diff / (l2_true + 1e-8)

        self.logger.info(f"⚡ Test Relative L2 Error: {rel_error:.6f} ({rel_error:.2%})")

        metrics = compute_metrics(y_true, y_pred)
        metrics['rel_l2'] = float(rel_error)
        self.logger.info(f"Metrics: {metrics}")

        self.exp_logger.save_metrics(metrics, history)
        self.exp_logger.close()
        return metrics
