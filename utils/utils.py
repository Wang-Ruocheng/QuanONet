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
    通用参数计数器，支持 MindSpore (Cell) 和 PyTorch (nn.Module)
    """
    try:
        # 1. 尝试 MindSpore 方式
        import mindspore.nn as nn
        if isinstance(model, nn.Cell):
            total_params = 0
            for param in model.trainable_params():
                total_params += np.prod(param.shape)
            return int(total_params)
    except ImportError:
        pass

    try:
        # 2. 尝试 PyTorch 方式 (适用于 DeepXDE 的 net)
        import torch
        if isinstance(model, torch.nn.Module):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)
    except ImportError:
        pass

    # 3. 如果都不是，尝试打印 len (比如简单的 list)
    try:
        return len(model)
    except:
        return "Unknown"