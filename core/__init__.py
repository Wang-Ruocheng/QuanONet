"""
Core modules for QuanONet - core functionalities including model definitions, network layers, quantum circuits, and training processes.
"""

# Conditional imports to handle different environments
try:
    from .models_ms import QuanONetMS, HEAQNNMS
    from .layers import *
    from .quantum_circuits_ms import generate_simple_hamiltonian
    _mindspore_available = True
except ImportError:
    _mindspore_available = False
    QuanONetMS = None
    HEAQNNMS = None
    generate_simple_hamiltonian = None

__all__ = [
    'FNNMS', 'DeepONetMS'
]

if _mindspore_available:
    __all__.extend(['QuanONetMS', 'HEAQNNMS', 'generate_simple_hamiltonian'])
