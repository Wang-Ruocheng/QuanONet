"""
Data utilities for QuanONet
"""

# Only import data processing functions to avoid MindSpore dependency
from .data_processing import ODE_encode, PDE_encode

__all__ = [
    'ODE_encode', 'PDE_encode'
]
