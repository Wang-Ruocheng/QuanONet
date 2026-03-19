# QuanONet: Quantum Neural Operators with Adaptive Frequency Strategy

**Official Implementation**

## Introduction

QuanONet is a pure quantum neural operator framework designed for the Noisy Intermediate-Scale Quantum (NISQ) era to solve partial differential equations (PDEs). Unlike hybrid architectures that rely on classical post-processing, QuanONet performs end-to-end learning within the quantum Hilbert space.

## Repository Structure

The repository utilizes a unified solver architecture that handles both Quantum (MindSpore) and Classical (PyTorch/DeepXDE) backends:

```text
.
├── main.py                # Unified Entry Point (Auto-backend selection)
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
│
├── scripts/               # Automated reproduction bash scripts
│   ├── reproduce_table4.sh  # General Benchmarks (ODE & PDE)
│   ├── reproduce_table5.sh  # Asymmetric Parameterization (vs. FNO)
│   ├── reproduce_table7.sh  # Implicit Frame Capacity Search
│   ├── reproduce_fig9_scaling.sh # High-Dimensional Scaling Limit
│   ├── reproduce_table8.sh  # Circuit Architecture Ablation
│   └── reproduce_sec54.sh   # Hamiltonian Design Ablation
│
├── core/                  # Core model architectures
│   ├── models.py            # Unified model wrapper
│   ├── quantum_circuits.py  # QuanONet & HEAQNN circuits (MindSpore)
│   ├── dde_models.py        # Classical baselines (DeepONet, FNO, FNN)
│   └── layers.py            # Custom neural network layers
│
├── solvers/               # Training & Evaluation solvers
│   ├── solver_ms.py         # Quantum solver (MindSpore backend)
│   └── solver_dde.py        # Classical solver (DeepXDE/PyTorch backend)
│
├── data_utils/            # Data pipelines and generation
│   ├── data_generation.py   # Dataset generation scripts
│   ├── data_manager.py      # Data loading, formatting, and batching
│   └── random_func.py       # Gaussian Random Field (GRF) generators
│
├── utils/                 # Utilities and helpers
│   ├── common.py            # Argument parsing (hyperparameters)
│   ├── logger.py            # Unified logging, tracking, and JSON saving
│   ├── metrics.py           # Evaluation metrics computation (L2, MSE)
│   └── backend.py           # Hardware backend management
│
├── hardware_deployment/        # Real-device deployment on IBM Quantum
│   ├── requirements_qiskit.txt # Standalone Qiskit environment dependencies
│   ├── ibm_inference.py        # Transpilation, execution, profiling, and plotting
│   └── best_Antideriv_QuanONet_Net5-1-5-1_Q2_TF_S0.001_1000x100_Seed0
│      └── best_model.npz # Pre-trained weights
│
├── configs/               # Configuration presets
└── image/                 # Architectural diagrams
```

## Installation

The framework requires MindSpore (for Quantum models) and PyTorch + DeepXDE (for Classical baselines).

We recommend creating and managing your Python virtual environment with [uv](https://github.com/astral-sh/uv):

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Usage and Execution

All model training and evaluation are executed through the unified `main.py` entry point.

**Data Generation:** Training and evaluation datasets based on Gaussian Random Fields (GRF) do not require manual download. The framework utilizes an on-the-fly data generation pipeline. Upon the first execution of a specific task, the system automatically generates, solves, and caches the corresponding `.npz` files.

**Device Allocation:** The framework automatically routes quantum models (QuanONet/HEAQNN) to the CPU to optimize resource allocation, while classical baselines (DeepONet/FNN/FNO) are routed to the GPU. To manually override this behavior, append `--gpu <ID>`.

### Example Commands

**1. Train TF-QuanONet**

Execute the quantum model on the Antiderivative ODE problem utilizing the Trainable Frequency strategy:

```bash
python main.py \
  --operator Antideriv \
  --model_type QuanONet \
  --if_trainable_freq true \
  --num_qubits 5 \
  --num_epochs 1000
```

**2. Train Classical Baseline (DeepONet)**

*Note on Asymmetric Parameterization:* To rigorously evaluate parameter efficiency, the classical baselines in our experiments are intentionally parameterized to ~10,000 parameters, granting them nearly an order of magnitude advantage over the compact TF-QuanONet (~1,200 parameters).

```bash
python main.py \
  --operator Antideriv \
  --model_type DeepONet \
  --net_size 3 100 3 100 \
  --num_epochs 2000
```

## Reproducing Paper Results

The `scripts/` directory contains automated bash scripts to reproduce the primary experimental results reported in the manuscript.

```bash
./scripts/reproduce_table4.sh
```

| **Script**              | **Description**                                                                                                                                        | **Relevant Section** |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `reproduce_table4.sh`       | **General Benchmarks**: Evaluates TF-QuanONet against Quantum (HEA, TF-HEA) and Classical (DeepONet, FNN) baselines across ODE and PDE operator tasks. | Table 4 (Sec 5.2.2)        |
| `reproduce_table5.sh`       | **Asymmetric Parameterization**: Compares compact TF-QuanONet (~1.2k params) against over-parameterized FNO and DeepONet (~10k params).                | Table 5 & 6 (Sec 5.2.3)    |
| `reproduce_table7.sh`       | **Implicit Frame Capacity**: Grid search over network width and depth to verify the $\mathcal{O}(p^2)$ implicit frame and analyze error saturation.  | Table 7 (Sec 5.3.1)        |
| `reproduce_fig9_scaling.sh` | **High-Dimensional Scaling Limit**: Sweeps the latent dimension $p$ from 4 to 256. Demonstrates optimization stability in high dimensions.           | Fig 9 (Sec 5.3.1)          |
| `reproduce_table8.sh`       | **Circuit Architecture Ablation**: Investigates the trade-off between circuit width (qubits) and depth regarding expressivity vs. trainability.        | Table 8 (Sec 5.3.2)        |
| `reproduce_sec54.sh`        | **Hamiltonian Ablation**: Evaluates the impact of Hamiltonian design (Pauli basis, spectral radii, and degeneracy) on model expressivity.              | Fig 10 & 11 (Sec 5.4)      |

## Configuration & Parameters

The `main.py` script accepts the following primary configurations:

### 1. Task & Data Setup

| **Argument**               | **Description**                                                                                 | **Default**              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------ |
| `--operator`                   | Problem type:`Antideriv`, `Homogeneous`, `Nonlinear`, `RDiffusion`, `Advection`, `Darcy`. | **Required**             |
| `--num_train` / `--num_test` | Number of function samples for training/testing.                                                      | `1000` / `1000`            |
| `--train_sample_num`           | Points sampled per function for training.                                                             | `10`                         |
| `--test_sample_num`            | Points sampled per function for testing.                                                              | `100`                        |
| `--num_points`                 | Output resolution (Trunk/Target grid size).                                                           | `100`                        |
| `--num_points_0`               | Input resolution (Branch/Source function size).                                                       | `100` (PDE) / `1000` (ODE) |
| `--num_cal`                    | High-Fidelity resolution for Ground Truth data generation.                                            | `1000` (ODE) / `100` (PDE) |

### 2. Model Architecture (`--net_size`)

| **Model**    | **Format**                                                                          | **Example**                 |
| ------------------ | ----------------------------------------------------------------------------------------- | --------------------------------- |
| **QuanONet** | `[b_depth, b_ansatz, t_depth, t_ansatz]`                                                | `20 2 10 2`                     |
| **DeepONet** | `[b_depth, b_width, t_depth, t_width]` *Optional 5th arg for output dim:* `[... p]` | `3 100 3 100` `3 100 3 50 10` |
| **FNO**      | `[modes, width, layers, fc_hidden]`                                                     | `16 32 3 32`                    |

### 3. Quantum Specifics

| **Argument**      | **Description**                                                 | **Default** |
| ----------------------- | --------------------------------------------------------------------- | ----------------- |
| `--num_qubits`        | Defines latent dimension$p=2^n$.                                    | `5`             |
| `--if_trainable_freq` | Enable Trainable Frequency (TF) strategy (`true`/`false`).        | `false`         |
| `--scale_coeff`       | Scaling coefficient for encoding.                                     | `0.01`          |
| `--ham_bound`         | Hamiltonian eigenvalue range.                                         | `[-5, 5]`       |
| `--ham_pauli`         | Pauli basis for the Hamiltonian (`X`, `Y`, or `Z`).             | `Z`             |
| `--ham_diag`          | Manually specify exact eigenvalues. Overrides bounds and Pauli basis. | `None`          |

## Real-Device Deployment (IBM Quantum)

The `hardware_deployment/` directory isolates the deployment and evaluation of QuanONet on real superconducting quantum processors (e.g., `ibm_fez`) utilizing a standalone Qiskit environment to prevent dependency conflicts.

**Setup:**

```bash
pip install -r requirements_qiskit.txt
export QISKIT_IBM_TOKEN="your_token_here"
```

**Execution (`ibm_inference.py`):**

This unified script automatically analyzes the physical chip topology, routes optimal qubits based on calibration data, transpiles the circuit, and executes inference. It supports evaluating both linear (`x`) and trigonometric (`cos\pi x`) initial conditions via the `--input_func` argument.

- **Mode 1: Ideal Simulation (Local)**

  Verify logical depth without a server connection.

  ```bash
  python ibm_inference.py --simulator_only --input_func linear
  ```
- **Mode 2: Real Hardware Execution**

  Automatically profiles the least busy QPU and submits a new physical execution job.

  ```bash
  python ibm_inference.py --input_func cos
  ```
- **Mode 3: Fetch Existing Results**

  Bypass queue times by fetching completed job data directly.

  ```bash
  python ibm_inference.py --job_id <YOUR_JOB_ID_HERE>
  ```

**Using Custom Weights:**

The repository includes a pre-trained checkpoint (`best_Antideriv_QuanONet_Net5-1-5-1_Q2_TF_S0.001_1000x100_Seed0/best_model.npz`) matching the lightweight configuration. To evaluate custom models:

1. Execute with the `.npz` weight path argument: `python ibm_inference.py --weight_path YOUR_WEIGHT_FILE.npz`.
2. The script automatically parses the network dimensions from the filename. If a custom naming convention is used, dimensions must be passed manually (e.g., `--n_qubits 4 --n_branch 5 --n_trunk 5 --n_hidden 1`).

## Citation

If you utilize this framework, the implicit quadratic frame theory, or the real-device deployment pipeline in your research, please cite our primary manuscript:

```bibtex
@article{wang2026quantum,
      author={Wang, Ruocheng and Xia, Zhuo and Zhong, Xiaoqiu and Yan, Junchi},
      title={Quantum Neural Operators: Implicit Quadratic Frame and Expressivity Advantages with Adaptive Frequency Strategy},
      journal={Nature Machine Intelligence (Under Review)},
      year={2026}
}
```

For the preliminary conference version establishing the foundational QuanONet architecture, please cite:

```bibtex
@inproceedings{wang2025quanonet,
  title={QuanONet: Quantum Neural Operator with Application to Differential Equation},
  author={Wang, Ruocheng and Xia, Zhuo and Yan, Ge and Yan, Junchi},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
