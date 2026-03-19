"""
Utility functions for the QuanONet library.
"""

import numpy as np
import os
import pickle
import logging


class StreamToLogger(object):
    """Redirect stdout/stderr to logger."""
    
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


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