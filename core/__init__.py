"""
Core modules for QuanONet - core functionalities including model definitions, network layers, quantum circuits, and training processes.
"""

# Conditional imports to handle different environments
try:
    from .models import QuanONet, HEAQNN
    _mindspore_available = True
except ImportError:
    _mindspore_available = False
    QuanONet = None
    HEAQNN = None

from .layers import *
from .quantum_circuits import generate_simple_hamiltonian

__all__ = [
    'generate_simple_hamiltonian', 'FNN', 'DeepONet'
]

if _mindspore_available:
    __all__.extend(['QuanONet', 'HEAQNN'])
