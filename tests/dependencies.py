"""
Common imports and dependencies for the QuanONet library.
"""

import numpy as np
import os
import math
import copy
import logging
import sys
import glob
import time
import pickle
import itertools

# MindSpore imports
import mindspore as ms
import mindspore.numpy as mnp
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.dataset as ds
from mindspore.ops import operations as P
from mindspore.train.serialization import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.nn import Adam, MSELoss, WithLossCell, TrainOneStepCell
from mindspore.common.initializer import Uniform, initializer, One
from mindspore import Parameter

# MindQuantum imports
from mindquantum.core.circuit import Circuit, change_param_name, AP, A, add_prefix, add_suffix
from mindquantum.core.gates import H, RX, RY, RZ, CNOT, Z, BasicGate, X
from mindquantum.core.gates import gates as G
from mindquantum.core.parameterresolver import PRGenerator
from mindquantum.simulator import Simulator
from mindquantum.core.operators import QubitOperator, Hamiltonian
from mindquantum.framework import MQLayer
from mindquantum.algorithm.nisq._ansatz import Ansatz
from mindquantum.algorithm.nisq.chem.hardware_efficient_ansatz import _check_single_rot_gate_seq
from mindquantum.core.circuit import ReverseAdder, NoiseExcluder, NoiseChannelAdder
from mindquantum.core.circuit.channel_adder import (
    ChannelAdderBase, SequentialAdder, MixerAdder, 
    MeasureAccepter, BitFlipAdder
)

# Scientific computing imports
from scipy.integrate import solve_ivp
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

# Progress bar
from tqdm import tqdm

# Plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# IPython display (for Jupyter notebooks)
try:
    from IPython.display import display_svg
except ImportError:
    display_svg = None
