"""
Data utilities for QuanONet
"""

from .data_generation import (
    generate_ODE_Operator_data,
    generate_PDE_Operator_data,
)
from .data_processing import ODE_encode, PDE_encode

__all__ = [
    'generate_ODE_Operator_data',
    'generate_PDE_Operator_data',
    'ODE_encode', 'PDE_encode'
]
