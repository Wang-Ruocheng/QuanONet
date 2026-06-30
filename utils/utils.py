"""
Utility functions for the QuanONet library.
"""

import numpy as np
import os
import logging



def count_parameters(model):
    """
    Count the number of trainable real-valued parameters in a model.

    Complex parameters (SpectralConv weights) are counted as 2 real numbers each:
    - PyTorch: cfloat tensors → numel * 2
    - MindSpore: SpectralConv1dMS stores (in, out, modes, 2) float32 → numel as-is
    """
    try:
        # 1. Try MindSpore
        import mindspore.nn as nn
        if isinstance(model, nn.Cell):
            total_params = 0
            for param in model.trainable_params():
                total_params += int(np.prod(param.shape))
            return total_params
    except Exception:
        pass

    try:
        # 2. Try PyTorch
        import torch
        if isinstance(model, torch.nn.Module):
            total = 0
            for p in model.parameters():
                if p.requires_grad:
                    total += p.numel() * 2 if p.is_complex() else p.numel()
            return total
    except Exception:
        pass

    try:
        return len(model)
    except Exception:
        return -1
