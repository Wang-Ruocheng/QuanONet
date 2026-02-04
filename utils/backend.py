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
        
    def check_compatibility(self, model_type):
        """
        Determines the required backend based on model type.
        """
        # MindSpore Models
        ms_models = ['QuanONet', 'HEAQNN']
        
        # PyTorch/DeepXDE Models
        pt_models = ['DeepONet', 'FNN', 'FNO']
        
        if model_type in ms_models:
            if not self.is_mindspore_available:
                raise ImportError(f"Model '{model_type}' requires MindSpore, but it is not installed.")
            return 'mindspore'
            
        elif model_type in pt_models:
            if not self.is_torch_available:
                raise ImportError(f"Model '{model_type}' requires PyTorch/DeepXDE, but it is not installed.")
            return 'pytorch'
            
        else:
            # Fallback/Unknown
            return 'unknown'

# Singleton instance
backend = BackendManager()