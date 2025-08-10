# QuanONet Project Documentation

## Project Overview
This is an operator learning project based on Quantum Neural Networks (QuanONet), supporting the solution of ODE and PDE operator problems.

## Project Structure
```
QON/
├── configs/                    # Configuration files
│   ├── config_ODE.json        # General ODE operator configuration
│   └── config_PDE.json        # General PDE operator configuration
├── core/                       # Core modules
│   ├── models.py              # QuanONet model implementation
│   ├── quantum_circuits.py    # Quantum circuits
│   └── layers.py              # Neural network layers
├── data_utils/                 # Data utilities
│   ├── data_generation.py     # Data generation
│   └── data_processing.py     # Data processing
├── utils/                      # Utility functions
├── examples/                   # Example scripts
├── tests/                      # Test scripts
├── train_ODE_quanonet.py      # ODE training script
└── train_PDE_quanonet.py      # PDE training script
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
- **Heat**: Heat conduction equation (∂u/∂t = α∇²u + u0(x))
- **Wave**: Wave equation (∂²u/∂t² = c²∇²u + u0(x,t))
- **Custom**: Custom PDE operator

## Usage

### Training ODE Operators
```bash
# Train Inverse operator with default configuration
python train_ODE_quanonet.py --operator Inverse --config configs/config_ODE.json

# Set random seed
python train_ODE_quanonet.py --operator Homogeneous --config configs/config_ODE.json --random_seed 42

# Custom parameters
python train_ODE_quanonet.py --operator Nonlinear --config configs/config_ODE.json --learning_rate 0.001 --num_epochs 500
```

### Training PDE Operators
```bash
# Train Burgers equation with default configuration
python train_PDE_quanonet.py --operator Burgers --config configs/config_PDE.json

# Train reaction-diffusion equation
python train_PDE_quanonet.py --operator RDiffusion --config configs/config_PDE.json --random_seed 0

# Background training with log saving
CUDA_VISIBLE_DEVICES=0 nohup python -u train_PDE_quanonet.py --operator Heat --config configs/config_PDE.json --random_seed 123 > training_heat.log 2>&1 &
```

## Configuration File Description

### ODE Configuration (config_ODE.json)
- `trunk_input_size`: 1 (ODE has only x variable)
- `num_cal`: 1000 (High precision calculation grid points)
- `num_epochs`: 1000 (ODE usually requires more epochs)

### PDE Configuration (config_PDE.json)
- `trunk_input_size`: 2 (PDE has x,t two variables)
- `num_cal`: 100 (PDE computation is intensive, use fewer grid points)
- `num_epochs`: 100 (PDE converges faster)

## Main Parameter Description
- `num_train/num_test`: Number of training/testing samples
- `num_sensors`: Number of sensors (spatial discrete points)
- `num_qubits`: Number of qubits
- `net_size`: Network structure [input layer, hidden layer1, hidden layer2, output layer]
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
python train_ODE_quanonet.py --operator Custom --custom_ode "custom_equation" --custom_name "MyODE"

# Custom PDE operator
python train_PDE_quanonet.py --operator Custom --custom_pde "custom_equation" --custom_name "MyPDE"
```

## Troubleshooting
1. **CUDA Error**: Check GPU availability and MindSpore version
2. **Out of Memory**: Reduce `batch_size` or `num_train`
3. **Slow Convergence**: Adjust `learning_rate` or increase `num_epochs`
4. **Data Generation Failure**: Check `num_cal` and numerical stability parameters# QuanONet - Quantum Operator Network

QON (Quantum Operator Network) is a quantum neural network-based operator learning framework that supports solving various ODE and PDE operator problems.

## 🚀 Features

- **Multiple Operator Support**: ODE operators (inverse problems, homogeneous, nonlinear) and PDE operators (reaction-diffusion, Burgers equation, heat conduction, wave equation)
- **Quantum Neural Networks**: QuanONet model based on MindQuantum
- **Flexible Configuration**: Support for JSON configuration files and command line parameters
- **Batch Processing**: Support for multi-problem parallel data generation and training
- **Reproducibility**: Complete random seed support
- **Visualization**: Built-in result visualization tools

## 📋 Supported Operator Types

### ODE Operators
- **Inverse**: Inverse operator problem (`du/dx = u0(x)`)
- **Homogeneous**: Homogeneous operator problem (`du/dx = u + u0(x)`)
- **Nonlinear**: Nonlinear operator problem (`du/dx = u - u0²(x)`)

### PDE Operators
- **RDiffusion**: Reaction-diffusion equation (`∂u/∂t = α∇²u + k*u² + u0(x)`)
- **Burgers**: Burgers equation (`∂u/∂t + u∂u/∂x = ν∂²u/∂x²`)
- **Heat**: Heat conduction equation (`∂u/∂t = α∇²u + u0(x)`)
- **Wave**: Wave equation (`∂²u/∂t² = c²∇²u + u0(x,t)`)

## 🛠️ Installation

### Requirements
- Python 3.7+
- MindSpore 2.0+
- MindQuantum 0.9+
- NumPy, SciPy, Matplotlib

### Quick Installation
```bash
# Clone repository
git clone <repository-url>
cd QON

# Install dependencies
pip install mindspore mindquantum scipy matplotlib tqdm
```

## 🎯 Quick Start

### 1. ODE Operator Training
```bash
# Train inverse operator problem
python train_ODE_quanonet.py --operator Inverse --config configs/config_ODE.json --random_seed 42

# Custom parameters
python train_ODE_quanonet.py --operator Nonlinear --num_epochs 200 --learning_rate 0.001
```

### 2. PDE Operator Training
```bash
# Train Burgers equation
python train_PDE_quanonet.py --operator Burgers --config configs/config_Burgers.json --random_seed 0

# Train reaction-diffusion equation
python train_PDE_quanonet.py --operator RDiffusion --config configs/config_PDE.json
```

## 📁 Project Structure

```
QON/
├── configs/                    # Configuration files
│   ├── config_ODE.json        # ODE operator configuration
│   ├── config_PDE.json        # PDE operator configuration
│   └── config_Burgers.json    # Burgers equation specific configuration
├── core/                      # Core modules
│   ├── models.py              # QuanONet model definition
│   ├── quantum_circuits.py    # Quantum circuits
│   └── layers.py              # Network layer definition
├── data_utils/                # Data processing
│   ├── data_generation.py     # Data generation functions
│   └── data_processing.py     # Data preprocessing
├── utils/                     # Utility functions
│   ├── visualization.py       # Visualization tools
│   ├── loss_functions.py      # Loss functions
│   └── utils.py               # General utilities
├── examples/                  # Example scripts
│   ├── batch_generate.py      # Batch data generation
│   └── examples.py            # Usage examples
├── tests/                     # Test scripts
├── docs/                      # Documentation
├── train_ODE_quanonet.py      # ODE training script
└── train_PDE_quanonet.py      # PDE training script
```

## ⚙️ Configuration Files

### Basic Configuration Example
```json
{
  "num_train": 1000,
  "num_test": 1000,
  "num_sensors": 100,
  "num_qubits": 5,
  "net_size": [40, 2, 20, 2],
  "learning_rate": 0.0001,
  "num_epochs": 100,
  "batch_size": 100,
  "model_type": "QuanONet",
  "if_trainable_freq": true,
  "random_seed": 42
}
```

### Operator-Specific Parameters
- **Burgers equation**: `"nu": 0.05` (viscosity coefficient)
- **Reaction-diffusion**: `"alpha": 0.01, "k": 0.01`
- **Heat conduction**: `"alpha": 0.01`
- **Wave equation**: `"c": 1.0`

## 🎨 Output Directory Structure

### Standard Training Scripts
```
checkpoints/          # Model checkpoints
logs/                # Training logs
data/                # Generated data
```

### Research Version (train_PDE_quanonet_paper.py)
```
checkpoints_paper/
├── QuanONet_42/     # Regular model, seed 42
├── TFQuanONet_123/  # Trainable frequency model, seed 123
└── CustomNet_999/   # Custom model, seed 999

logs_paper/
├── QuanONet_42/
├── TFQuanONet_123/
└── CustomNet_999/
```

## 📊 Visualization

The project includes multiple Jupyter Notebooks for result visualization:
- `visualize_Inverse_results.ipynb` - Inverse operator results
- `visualize_Nonlinear_results.ipynb` - Nonlinear operator results
- `visualize_RDiffusion_results.ipynb` - Reaction-diffusion results
- `dde_deeponet.ipynb` - DeepONet comparison experiments

## 🔧 Advanced Usage

### Batch Data Generation
```bash
python examples/batch_generate.py --problems Inverse_Operator Burgers_Operator --parallel --max_workers 4
```

### Custom Operators
```bash
python train_PDE_quanonet.py --operator Custom --custom_pde "your_pde_function" --custom_name "MyPDE"
```

### GPU Training
```bash
CUDA_VISIBLE_DEVICES=0 python train_PDE_quanonet.py --operator Burgers --config configs/config_Burgers.json
```

### Background Training
```bash
nohup python -u train_PDE_quanonet.py --operator Burgers --random_seed 0 > training.log 2>&1 &
```

## 📚 Documentation

For detailed documentation, please refer to:
- [Data Generation Guide](docs/DATA_GENERATION_GUIDE.md)
- [ODE Training Guide](docs/ODE_TRAINING_GUIDE.md)
- [PDEBench Data Download](docs/PDEBENCH_DOWNLOAD_README.md)

## 🤝 Contributing

Welcome to submit Issues and Pull Requests to improve the project.

## 📄 License

[Add your license information]

## 🙏 Acknowledgments

This project is developed based on MindSpore and MindQuantum frameworks.
