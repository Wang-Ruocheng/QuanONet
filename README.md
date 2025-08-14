# QuanONet Project Documentation

## Project Overview
This is an operator learning project based on Quantum Neural Networks (QuanONet), supporting the solution of ODE and PDE operator problems.

## Project Structure
```
QON/
├── configs/                    # Configuration files
│   ├── config_ODE.json        # General ODE operator configuration
│   ├── config_PDE.json        # General PDE operator configuration
|   └── config_Darcy.json      # Specific operator configuration
├── core/                       # Core modules
│   ├── models.py              # QuanONet model implementation
│   ├── quantum_circuits.py    # Quantum circuits
│   ├── layers.py              # Neural network layers
│   └── training.py            # Training utilities
├── data_utils/                 # Data utilities
│   ├── data_generation.py     # Data generation
│   ├── data_processing.py     # Data processing
│   ├── random_func.py         # RGF function
│   └── PDE_SYSTEMS.py         # PDE solving
├── utils/                      # Utility functions
├── train_ODE.py      # ODE training script
├── train_PDE.py      # PDE training script
└──...
```

## Supported Operator Types

### ODE Operators
- **Inverse**: Inverse operator problem (du/dx = u0(x))
- **Homogeneous**: Homogeneous operator problem (du/dx = u + u0(x))
- **Nonlinear**: Nonlinear operator problem (du/dx = u - u0²(x))
- **Custom**: Custom ODE operator

### PDE Operators
- **RDiffusion**: Reaction-diffusion equation (∂u/∂t = α∇²u + k*u² + u0(x))
- **Burgers**: Burgers equation (∂u/∂t + u∂u/∂x = ν∂²u/∂x²)
- **Advection**: Advection equation (∂u/∂t + c∇u = 0)
- **Identity**: Identity operator (u(x,t) = u0(x))
- **Darcy**: Darcy flow equation (-∇(K∇u)=f)

## Usage

### Training ODE Operators
```bash
# Train Inverse operator with default configuration
python train_ODE.py --operator Inverse

# Set random seed
python train_ODE.py --operator Homogeneous --random_seed 42

# Custom parameters
python train_ODE.py --operator Nonlinear --learning_rate 0.001 --num_epochs 500

# Custom ODE operator
python train_ODE.py --operator Custom --custom_ode "u + 2*u0" --custom_name "MyOperator"
```

### Training PDE Operators
```bash
# Train Burgers equation with default configuration
python train_PDE.py --operator Burgers --config configs/config_PDE.json

# Custom parameters
python train_PDE.py --operator Darcy --config configs/Darcy.json --learning_rate 0.001 --num_epochs 500
```

## Configuration File Description

### ODE Configuration (config_ODE.json)
- `trunk_input_size`: 1 (ODE has only x variable)
- `num_cal`: 1000 (High precision calculation grid points)
- `num_epochs`: 1000

### PDE Configuration (config_PDE.json)
- `trunk_input_size`: 2 (PDE has x,t two variables)
- `num_cal`: 100 (PDE computation is intensive, use fewer grid points)
- `num_epochs`: 100

## Main Parameter Description
- `num_train/num_test`: Number of training/testing samples
- `num_points`: Number of points (spatial discrete points of u)
- `num_qubits`: Number of qubits
- `net_size`: Network structure [branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth] for QuanONet and [depth, linear_depth] for HEAQNN
- `if_trainable_freq`: Whether to enable trainable frequency
- `random_seed`: Random seed (None means not set)

## Model Saving
After training is completed, models are automatically saved to the `checkpoints/` directory:
- `best_*.ckpt`: Best model (based on validation/training loss)
- `*_final_*.ckpt`: Final model
- `*_checkpoint_*.ckpt`: Regular checkpoints (maximum 3 kept)

## Log Files
- Standard output: Training progress and loss information
- `logs/`: JSON files of evaluation results

## Custom Operators
```bash
# Custom ODE operator
python train_ODE.py --operator Custom --custom_ode "u + 2*u0" --custom_name "MyOperator"

# Custom PDE operator
For PDE operators, there are many ways to map the initial function to the solution (including initial conditions, driving terms, and boundary conditions). If you want to add a new PDE operator, Add the solver function def solve_myoperator_pde(num_cal, length_scale, **params, u0_cal=None) to data_utils/PDE_SYSTEMS.py, Add the global variable with python train_PDE.py --operator Myoperator. If the resolution of the initial and solution functions is different, use different num_points_0 and num_points in the config file (see configs/config_Darcy.json).
```

## 🚀 Features

- **Multiple Operator Support**: ODE operators (inverse problems, homogeneous, nonlinear and custom) and PDE operators (reaction-diffusion, Burgers equation, Advection equation, Identity equation and Darcy flow)
- **Quantum Neural Networks**: QuanONet and HEAQNN model based on MindQuantum
- **Flexible Configuration**: Support for JSON configuration files and command line parameters
- **Visualization**: Built-in result visualization tools

## 📋 Supported Operator Types

### ODE Operators
- **Inverse**: Inverse operator problem (`du/dx = u0(x)`)
- **Homogeneous**: Homogeneous operator problem (`du/dx = u + u0(x)`)
- **Nonlinear**: Nonlinear operator problem (`du/dx = u - u0²(x)`)
- **Custom** :Custom operator probelm

### PDE Operators
Mapping from driving terms::
- **RDiffusion**: Reaction-diffusion equation (`∂u/∂t = α∇²u + k*u² + u0(x)`)
Mapping from initial conditions:
- **Identity**: Identity operator: (`u(x,t) = u0(x)`)
- **Advection**: Advection equation: (`∂u/∂t + c∇u = 0`)
- **Burgers**: Burgers equation (`∂u/∂t + u∂u/∂x = ν∂²u/∂x²`)
Mapping from boundary conditions:
- **Darcy**: Darcy flow PDE: (`-∇(K∇u)=f`)

## 🎯 Visualization

visualize_ODE_results.ipynb and visualize_PDE_results.ipynb provide a visualization of the model results, which are stored in visualization_results.