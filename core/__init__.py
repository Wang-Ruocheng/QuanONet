"""
Core modules for QuanONet - core functionalities including model definitions, network layers, quantum circuits, and training processes.
"""

# Conditional imports to handle different environments
try:
    from .models import QuanONet, HEAQNN
    _mindspore_available = True
    from .layers import *
    from .quantum_circuits import generate_simple_hamiltonian
except ImportError:
    _mindspore_available = False
    QuanONet = None
    HEAQNN = None

__all__ = [
    'FNN', 'DeepONet'
]

if _mindspore_available:
    __all__.extend(['QuanONet', 'HEAQNN', 'generate_simple_hamiltonian'])
