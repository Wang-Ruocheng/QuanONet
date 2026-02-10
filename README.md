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


```

.
├── main.py                # Unified Entry Point for training and evaluation
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

Create a virtual environment and install the dependencies. The project requires **MindSpore** (for Quantum models) and **PyTorch** + **DeepXDE** (for Classical baselines).

```bash
pip install -r requirements.txt

```

## 🚀 Quick Start

All models are trained using the unified `main.py` entry point.

### 1. Train TF-QuanONet (Ours)

Train the quantum model on the Inverse ODE problem using the Trainable Frequency strategy:

```bash
python main.py \
  --operator Inverse \
  --model_type QuanONet \
  --if_trainable_freq true \
  --num_qubits 5 \
  --num_epochs 100

```

### 2. Train Classical Baseline (DeepONet)

Train the classical DeepONet benchmark:

```bash
python main.py \
  --operator Inverse \
  --model_type DeepONet \
  --net_size 3 100 3 100 \
  --num_epochs 2000

```

> **Note on FNO**: FNO is fully supported. Use `--net_size modes width layers` to configure the architecture (e.g., `--net_size 15 14 3 32`).

---

## 📊 Reproducing Paper Results

This section provides the commands to reproduce the experiments presented in the manuscript.

### Experiment 1: General Benchmarks (Table 4)

*Comparison of different models across various ODE and PDE operators.*

**1. ODE Benchmarks (e.g., Homogeneous Operator)**

```bash
# TF-QuanONet
python main.py --operator Homogeneous --model_type QuanONet --if_trainable_freq true

```

**2. PDE Benchmarks (e.g., Reaction-Diffusion)**

```bash
# TF-QuanONet
python main.py --operator RDiffusion --model_type QuanONet --if_trainable_freq true

```

### Experiment 2: Dimensionality Scaling Analysis (Fig. 9)

*Investigating model performance scaling with respect to the latent dimension .*

Control the latent dimension by changing the number of qubits ().

* : `--num_qubits 2`
* : `--num_qubits 5`
* : `--num_qubits 8`

```bash
# TF-QuanONet (p=4 example)
python main.py --operator Inverse --model_type QuanONet --if_trainable_freq true --num_qubits 2 --net_size 100 2 100 2

```

---

### Experiment 3: Hamiltonian Ablation Studies (Fig. 10 & 11)

*Analyzing the impact of spectral properties on expressivity.*

**1. Spectral Radius Control (`--ham_bound`)**

Test how the magnitude of eigenvalues affects the unitary orbit volume.

```bash
python main.py --operator Inverse --model_type QuanONet --if_trainable_freq true --ham_bound 10

```

**2. Pauli Basis Selection (`--ham_pauli`)**

Test invariance to the choice of Pauli operator basis.

```bash
python main.py --operator Inverse --model_type QuanONet --if_trainable_freq true --ham_pauli X

```

**3. Spectral Degeneracy (`--ham_diag`)**

Manually specify eigenvalues to test the effect of manifold dimension .

```bash
python main.py --operator Inverse --model_type QuanONet --if_trainable_freq true --num_qubits 2 --net_size 50 2 50 2 --ham_diag -5 5 5 5

```

## ⚙️ Configuration & Parameters

The `main.py` script supports the following arguments:

| Category                         | Argument              | Description                                                  | Default/Example |
| :------------------------------- | :-------------------- | :----------------------------------------------------------- | :-------------- |
| **Task Setup** | `--operator`          | git add .Problem type: `Inverse`, `Homogeneous`, `Nonlinear`, `RDiffusion`, `Advection`, `Darcy`. | -               |
|                                  | `--num_points`        | **Output** resolution (Trunk/Target grid size).              | `100`           |
|                                  | `--num_points_0`      | **Input** resolution (Branch/Source function size).          | `100`           |
|                                  | `--train_sample_num`  | Number of sampling points per function for training (P_train). | `10`            |
|                                  | `--test_sample_num`   | Number of sampling points per function for testing (P_test). | `100`           |
| **Model** | `--model_type`        | Architecture to train: `QuanONet`, `HEAQNN`, `DeepONet`, `FNN`, `FNO`. | `QuanONet`      |
|                                  | `--net_size`          | Network structure configuration.<br>• **QuanONet**: `[branch_depth, branch_ansatz_depth, trunk_depth, trunk_ansatz_depth]`<br>• **DeepONet**: `[branch_depth, branch_width, trunk_depth, trunk_width]`<br>• **FNO**: `[modes, width, layers, fc_hidden]` | `3 100 3 100`   |
| **Quantum**<br>*(QuanONet only)* | `--num_qubits`        | Number of qubits. Defines latent dimension $p=2^n$.          | `5` ($p=32$)    |
|                                  | `--if_trainable_freq` | Enable Trainable Frequency (TF) strategy (`true`/`false`).   | `false`         |
|                                  | `--ham_bound`         | Hamiltonian eigenvalue range (e.g., `[-5, 5]`).              | `[-5, 5]`       |
| **Training** | `--num_epochs`        | Number of training epochs.                                   | `1000`          |
|                                  | `--batch_size`        | Size of mini-batches.                                        | `100`           |
|                                  | `--learning_rate`     | Initial learning rate.                                       | `0.001`         |
|                                  | `--num_train`         | Number of function samples for training.                     | `1000`          |