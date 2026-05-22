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


def get_trunk_fn(model):
    """
    Return the trunk callable for a given operator-learning model.

    The returned object is directly callable: ``trunk_fn(x) -> encoded``.

    Model → returned trunk callable
    ────────────────────────────────────────────────────────────────────
    MS DeepONet        (models_ms.DeepONet)
        model.trunk_net                          FNNLayer, x → basis

    DDE DeepONet       (dde.nn.DeepONet, PyTorch)
        activation_trunk ∘ trunk                 FNN + activation, x → basis

    MS QuanONet  (TF)  (if_trainable_freq=True)
        model.trunk_LinearLayer                  CombinedNet(Repeat, Linear)
                                                 x → (trunk_depth × n_qubits,)

    MS QuanONet (non-TF) (if_trainable_freq=False)
        model.trunk_ScaleLayer                   CombinedNet(Coeff, Repeat)
                                                 x → (trunk_depth × n_qubits,)

    PT QuanONetPT      (models_pt.QuanONetPT)
        model.trunk_freq                         _TiledElementWise or _ScaleRepeat
                                                 x → (trunk_enc_size,)

    Args:
        model: a trained model instance (any of the above).

    Returns:
        callable: the trunk sub-network / layer.

    Raises:
        ValueError: if the model type is not recognised.
    """
    # MS DeepONet
    if hasattr(model, 'trunk_net'):
        return model.trunk_net

    # DDE / PyTorch DeepONet — trunk MLP + per-element activation
    if hasattr(model, 'trunk') and hasattr(model, 'activation_trunk'):
        class _DDETrunk:
            def __init__(self, net, act):
                self._net = net
                self._act = act
            def __call__(self, x):
                return self._act(self._net(x))
        return _DDETrunk(model.trunk, model.activation_trunk)

    # MS QuanONet — trainable-frequency variant
    if hasattr(model, 'trunk_LinearLayer'):
        return model.trunk_LinearLayer

    # MS QuanONet — fixed-scale variant
    if hasattr(model, 'trunk_ScaleLayer'):
        return model.trunk_ScaleLayer

    # PT QuanONetPT (TF or fixed-scale both stored as trunk_freq)
    if hasattr(model, 'trunk_freq'):
        return model.trunk_freq

    raise ValueError(
        f"Cannot determine trunk function for model of type '{type(model).__name__}'. "
        "Expected one of: MS DeepONet, DDE DeepONet, MS QuanONet, PT QuanONetPT."
    )