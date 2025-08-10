"""
Loss functions for QuanONet.
"""

import numpy as np
import mindspore.nn as nn
import mindspore.ops as ops


class ProportionalLoss(nn.Cell):
    """Proportional loss function that handles zero targets."""
    
    def __init__(self):
        super(ProportionalLoss, self).__init__()
        self.mean = ops.ReduceMean()
        self.abs = ops.Abs()
        self.epsilon = 1e-8  # Small constant to avoid division by zero

    def construct(self, predicted, target):
        target = np.where(target == 0, self.epsilon, target)  # Avoid division by zero
        ratio = predicted / target
        mean_ratio = self.mean(ratio)
        proportional_error = self.abs(ratio - mean_ratio)
        return self.mean(proportional_error)
