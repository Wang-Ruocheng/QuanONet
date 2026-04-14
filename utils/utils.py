"""
Utility functions for the QuanONet library.
"""

import numpy as np
import os
import logging



def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    """
    try:
        # 1. Try MindSpore
        import mindspore.nn as nn
        if isinstance(model, nn.Cell):
            total_params = 0
            for param in model.trainable_params():
                total_params += np.prod(param.shape)
            return int(total_params)
    except ImportError:
        pass

    try:
        # 2. Try PyTorch
        import torch
        if isinstance(model, torch.nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except ImportError:
        pass

    # 3. Fallback: Try to get length if possible
    try:
        return len(model)
    except:
        return "Unknown"