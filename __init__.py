"""
===================================

A machine learning framework for operator learning.
"""

from .models import QuanONet, HEAQNN, FNN, HEAQNN
from .quantum_circuits import QuanONet_build, HEAQNNwork_build, circuit2network
from .data_generation import generate_ODE_Operator_data, generate_PDE_Operator_data
from .data_processing import ODE_encode, PDE_encode, sample_1D_Operator_data, sample_2D_Operator_data
from .layers import SpectralConv1d, FNO1d, LinearLayer, SumLayer, CoeffLayer, FNNLayer, CombinedNet, RepeatLayer
from .loss_functions import ProportionalLoss
from .utils import StreamToLogger
from .config import load_config, get_problem_config, list_available_problems

__version__ = "1.0.1"
__author__ = "QuanONet Team"

__all__ = [
    # Models
    'QuanONet', 'HEAQNN', 'FNN', 'DeepONet'
    # Quantum Circuits
    'QuanONet_build', 'HEAQNNwork_build', 'circuit2network',
    # Data Generation
    'generate_ODE_Operator_data', 'generate_PDE_Operator_data',
    # Data Processing
    'ODE_encode', 'PDE_encode', 'sample_1D_Operator_data', 'sample_2D_Operator_data',
    # Layers
    'SpectralConv1d', 'FNO1d', 'LinearLayer', 'SumLayer', 'CoeffLayer', 
    'FNNLayer', 'CombinedNet', 'RepeatLayer',
    # Loss Functions
    'ProportionalLoss',
    # Utils
    'StreamToLogger',
    # Config
    'load_config', 'get_problem_config', 'list_available_problems'
]
