"""
Backend Manager: Handles dynamic imports and backend detection.
"""
import sys
import importlib.util

class BackendManager:
    def __init__(self):
        pass

    @property
    def is_mindspore_available(self):
        return importlib.util.find_spec("mindspore") is not None

    @property
    def is_torch_available(self):
        return importlib.util.find_spec("torch") is not None

    @property
    def is_torchquantum_available(self):
        return importlib.util.find_spec("torchquantum") is not None

    @property
    def is_qiskit_ml_available(self):
        return importlib.util.find_spec("qiskit_machine_learning") is not None

    def check_compatibility(self, model_type, quantum_backend='mindquantum', classical_backend='pytorch'):
        """
        Determines the required backend based on model type and backend selections.

        Returns one of:
          'mindspore'           — QuanONet/HEAQNN with MindQuantum
          'pytorch_quantum'     — QuanONet/HEAQNN with TorchQuantum or Qiskit
          'pytorch'             — DeepONet/FNN/FNO with PyTorch+DeepXDE
          'mindspore_classical' — DeepONet/FNN/FNO with MindSpore
        """
        quantum_models = ['QuanONet', 'HEAQNN']
        classical_models = ['DeepONet', 'FNN', 'FNO']

        if model_type in quantum_models:
            if quantum_backend == 'mindquantum':
                if not self.is_mindspore_available:
                    raise ImportError(
                        f"Model '{model_type}' with mindquantum backend requires MindSpore, "
                        "but it is not installed."
                    )
                return 'mindspore'
            elif quantum_backend == 'torchquantum':
                if not self.is_torch_available:
                    raise ImportError(
                        f"Model '{model_type}' with torchquantum backend requires PyTorch, "
                        "but it is not installed."
                    )
                if not self.is_torchquantum_available:
                    raise ImportError(
                        f"Model '{model_type}' with torchquantum backend requires torchquantum, "
                        "but it is not installed. Install with: pip install torchquantum"
                    )
                return 'pytorch_quantum'
            elif quantum_backend == 'qiskit':
                if not self.is_torch_available:
                    raise ImportError(
                        f"Model '{model_type}' with qiskit backend requires PyTorch, "
                        "but it is not installed."
                    )
                if not self.is_qiskit_ml_available:
                    raise ImportError(
                        f"Model '{model_type}' with qiskit backend requires qiskit-machine-learning, "
                        "but it is not installed. Install with: pip install qiskit-machine-learning"
                    )
                return 'pytorch_quantum'
            else:
                raise ValueError(f"Unknown quantum_backend: '{quantum_backend}'. "
                                 "Choose from: mindquantum, torchquantum, qiskit")

        elif model_type in classical_models:
            if classical_backend == 'pytorch':
                if not self.is_torch_available:
                    raise ImportError(
                        f"Model '{model_type}' requires PyTorch/DeepXDE, but it is not installed."
                    )
                return 'pytorch'
            elif classical_backend == 'mindspore':
                if not self.is_mindspore_available:
                    raise ImportError(
                        f"Model '{model_type}' with mindspore backend requires MindSpore, "
                        "but it is not installed."
                    )
                return 'mindspore_classical'
            else:
                raise ValueError(f"Unknown classical_backend: '{classical_backend}'. "
                                 "Choose from: pytorch, mindspore")

        else:
            return 'unknown'

# Singleton instance
backend = BackendManager()
