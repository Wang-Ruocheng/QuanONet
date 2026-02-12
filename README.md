
# QuanONet: Quantum Neural Operators with Adaptive Frequency Strategy

**Official Implementation**

## 📖 Introduction

**QuanONet** is a pure quantum neural operator framework designed for the NISQ era to solve partial differential equations (PDEs). Unlike hybrid architectures that rely on classical post-processing, QuanONet performs end-to-end learning within the quantum Hilbert space.

**Key Theoretical Insights:**

- **Implicit Quadratic Frame ($\mathcal{O}(p^2)$)**: We prove that the trace-based measurement protocol constructs an implicit quadratic frame from a latent dimension of $p$, strictly exceeding the representational capacity of classical linear frames ($\mathcal{O}(p)$).
- **Trainable Frequency (TF) Strategy**: By treating embedding frequencies as continuous learnable parameters, **TF-QuanONet** breaks spectral symmetries, preventing dimensional collapse and unlocking the full expressivity of the density matrix.

<p align="center">
<img src="image/qon_circ2.png" alt="QuanONet Architecture" width="800">
</p>

## 📂 Project Structure

The repository adopts a unified solver architecture handling both Quantum (MindSpore) and Classical (PyTorch/DeepXDE) backends:

```text
.
├── main.py                # Unified Entry Point (Auto-backend selection)
├── scripts/               # Automated reproduction scripts (Table 4, 5, 7)
├── configs/               # Configuration files
├── core/                  # Model Definitions (QuanONet, DeepONet, FNO)
├── solvers/               # Solver implementations (Quantum & Classical)
├── data_utils/            # Data generation and processing
├── dairy/                 # Training logs and history
├── logs/                  # Evaluation metrics (.json)
├── data/                  # Dataset storage
└── requirements.txt       # Project dependencies

```

## 🛠️ Installation

The project requires **MindSpore** (for Quantum models) and **PyTorch** + **DeepXDE** (for Classical baselines).

```bash
pip install -r requirements.txt

```

## 🚀 Quick Start

All models are trained using the unified `main.py` entry point.

### Smart Device Selection 🤖

You don't need to manually specify the device. The system automatically assigns:

* **Quantum Models (QuanONet/HEAQNN)**: Run on **CPU** (default) to save GPU resources.
* **Classical Models (DeepONet/FNN/FNO)**: Run on **GPU** (if available).

To force a specific GPU, use `--gpu <ID>`.

### 1. Train TF-QuanONet (Ours)

Train the quantum model on the Inverse ODE problem using the Trainable Frequency strategy:

```bash
python main.py \
  --operator Inverse \
  --model_type QuanONet \
  --if_trainable_freq true \
  --num_qubits 5 \
  --num_epochs 1000

```

### 2. Train Classical Baseline (DeepONet)

Train a DeepONet with asymmetric Branch/Trunk widths (automatically handled):

```bash
# Branch: 100 width, Trunk: 100 width
python main.py \
  --operator Inverse \
  --model_type DeepONet \
  --net_size 3 100 3 100 \
  --num_epochs 2000

```

---

## 📊 Reproducing Paper Results

We provide automated bash scripts in the `scripts/` directory to reproduce the experiments presented in the manuscript.

### Usage

```bash
# Run on default devices (Quantum->CPU, Classical->GPU)
./scripts/script_name.sh
```

### Available Experiments

| Script | Description | Corresponding Table/Fig |
| --- | --- | --- |
| **`reproduce_table4.sh`** | **General Benchmarks**: Compares QuanONet, HEAQNN, DeepONet, and FNN across 6 operators (Inverse, Darcy, etc.). Iterates over scales and seeds. | **Table 4** |
| **`reproduce_table5.sh`** | **Small Data Regime**: Evaluation on small training sets (). Uses fixed scales for TF-QuanONet. | **Table 5** |
| **`reproduce_table7.sh`** | **Architecture Search**: Grid search for TF-QuanONet ( vs. Qubits) and DeepONet (Depth vs. Width). | **Table 7** |

---

## ⚙️ Configuration & Parameters

The `main.py` script supports the following arguments:

### 1. Task & Data Setup

| Argument | Description | Default |
| --- | --- | --- |
| `--operator` | Problem type: `Inverse`, `Homogeneous`, `Nonlinear`, `RDiffusion`, `Advection`, `Darcy`. | **Required** |
| `--num_train` / `--num_test` | Number of function samples for training/testing. | `1000` / `1000` |
| `--train_sample_num` | Points sampled per function for training (). | `10` |
| `--test_sample_num` | Points sampled per function for testing (). | `100` |
| `--num_points` | **Output** resolution (Trunk/Target grid size). | `100` |
| `--num_points_0` | **Input** resolution (Branch/Source function size). | `100` (PDE) / `1000` (ODE) |
| `--num_cal` | **High-Fidelity Resolution** for data generation (Ground Truth). | `1000` (ODE) / `100` (PDE) |
| `--seed` | Random seed for reproducibility. | `0` |

### 2. Model Architecture (`--net_size`)

| Model | Format | Example |
| :--- | :--- | :--- |
| **QuanONet** | `[b_depth, b_ansatz, t_depth, t_ansatz]` | `20 2 10 2` |
| **DeepONet** | `[b_depth, b_width, t_depth, t_width]` <br> *Optional 5th arg for output dim:* `[... p]` | `3 100 3 100`<br>`3 100 3 50 10` |
| **FNO** | `[modes, width, layers, fc_hidden]` | `16 32 3 32` |

### 3. Quantum Specifics

| Argument | Description | Default |
| --- | --- | --- |
| `--num_qubits` | Number of qubits. Defines latent dimension . | `5` |
| `--if_trainable_freq` | Enable Trainable Frequency (TF) strategy (`true`/`false`). | `false` |
| `--scale_coeff` | Scaling coefficient for encoding. | `0.01` |
| `--ham_bound` | Hamiltonian eigenvalue range (e.g., `5 5` for ). | `[-5, 5]` |

### 4. Training & System

| Argument | Description | Default |
| --- | --- | --- |
| `--batch_size` | Size of mini-batches. | `100` |
| `--learning_rate` | Initial learning rate. | `0.001` |
| `--num_epochs` | Number of training epochs. | `1000` |
| `--gpu` | GPU ID (e.g., `0`). If unspecified, uses **Smart Auto-Select**. | `None` (Auto) |
| `--prefix` | Prefix for output directories (logs/checkpoints). | `None` |

