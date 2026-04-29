# QuanONet: Quantum Neural Operators with Adaptive Frequency Strategy

**Official Implementation**

## Introduction

QuanONet is a pure quantum neural operator framework designed for the Noisy Intermediate-Scale Quantum (NISQ) era to solve partial differential equations (PDEs).

## Repository Structure

```text
.
├── main.py                # Unified Entry Point (auto-backend selection)
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
├── test_backends.py       # Backend integration test script
│
├── scripts/               # Automated reproduction bash scripts
│   ├── reproduce_table4.sh
│   ├── reproduce_table5.sh
│   ├── reproduce_table7.sh
│   ├── reproduce_fig9_scaling.sh
│   ├── reproduce_table8.sh
│   └── reproduce_sec54.sh
│
├── visualization.ipynb    # Jupyter notebook for PDE results visualization
├── pretrained_weights/    # Pre-trained checkpoints (.ckpt)
│
├── core/                  # Core model architectures
│   ├── models.py               # MindSpore QuanONet / HEAQNN / FNN / DeepONet
│   ├── models_pt.py            # PyTorch QuanONetPT / HEAQNNPT (TQ or Qiskit)
│   ├── quantum_circuits.py     # HEA circuits — MindQuantum backend
│   ├── quantum_circuits_tq.py  # HEA circuits — TorchQuantum backend
│   ├── quantum_circuits_qiskit.py # HEA circuits — Qiskit EstimatorQNN backend
│   ├── dde_models.py           # PyTorch FNO (used by DeepXDE solver)
│   ├── ms_fno.py               # MindSpore FNO
│   └── layers.py               # Custom MindSpore layers
│
├── solvers/               # Training & Evaluation solvers
│   ├── solver_ms.py         # MindSpore solver (QuanONet/HEAQNN/FNN/DeepONet/FNO)
│   ├── solver_pt.py         # PyTorch solver (QuanONet/HEAQNN via TQ or Qiskit)
│   └── solver_dde.py        # DeepXDE/PyTorch solver (DeepONet/FNN/FNO)
│
├── data_utils/            # Data pipelines and generation
│   ├── data_generation.py
│   ├── data_manager.py
│   └── random_func.py
│
├── utils/                 # Utilities and helpers
│   ├── common.py            # Argument parsing (all hyperparameters)
│   ├── backend.py           # Backend routing (5-way dispatch)
│   ├── logger.py            # Logging and JSON metrics
│   └── metrics.py           # L2 / MSE evaluation
│
└── hardware_deployment/        # Real-device deployment on IBM Quantum
    ├── requirements_qiskit.txt
    ├── ibm_inference.py
    └── Antideriv/.../best_model.npz
```

## Installation

The framework supports three quantum simulation backends and two classical backends. All can coexist in a single environment.

**Tested environment:** Python 3.9, conda.

```bash
conda create -n quanode python=3.9
conda activate quanode
pip install -r requirements.txt
```

> **TorchQuantum compatibility note:** `torchquantum==0.1.8` has import-time incompatibilities with qiskit 1.x. Apply the following one-time patches after installation:
>
> ```python
> # torchquantum/plugin/__init__.py  — wrap import in try/except
> # torchquantum/plugin/qiskit/__init__.py  — wrap imports in try/except
> # torchquantum/util/utils.py  — wrap qiskit_ibm_runtime import in try/except
> ```
>
> These patches disable only the optional IBM-Q connectivity plugin; local statevector simulation is unaffected. See `test_backends.py` for a ready-to-run verification.

### Backend Matrix

| Backend flag | Library required | Notes |
|---|---|---|
| `--quantum_backend mindquantum` (default) | `mindspore`, `mindquantum` | Original QuanONet backend |
| `--quantum_backend torchquantum` | `torchquantum` | PyTorch-native statevector sim |
| `--quantum_backend qiskit` | `qiskit`, `qiskit-machine-learning` | EstimatorQNN + TorchConnector |
| `--classical_backend pytorch` (default) | `torch`, `DeepXDE` | DeepONet / FNN / FNO |
| `--classical_backend mindspore` | `mindspore` | MindSpore FNO / FNN / DeepONet |

## Usage and Execution

All model training and evaluation are executed through the unified `main.py` entry point.

**Data Generation:** Training and evaluation datasets based on Gaussian Random Fields (GRF) do not require manual download. The framework utilizes an on-the-fly data generation pipeline. Upon the first execution of a specific task, the system automatically generates, solves, and caches the corresponding `.npz` files.

**Device Allocation:** The framework automatically routes quantum models (QuanONet/HEAQNN) to the CPU to optimize resource allocation, while classical baselines (DeepONet/FNN/FNO) are routed to the GPU when available. To manually override, append `--gpu <ID>`.

### Example Commands

**1. Train TF-QuanONet (original MindQuantum backend)**

```bash
python main.py \
  --operator Antideriv \
  --model_type QuanONet \
  --if_trainable_freq true \
  --num_qubits 5 \
  --num_epochs 1000
```

**2. Train TF-QuanONet with TorchQuantum backend**

```bash
python main.py \
  --operator Antideriv \
  --model_type QuanONet \
  --quantum_backend torchquantum \
  --if_trainable_freq true \
  --num_qubits 5 \
  --num_epochs 1000
```

**3. Train TF-QuanONet with Qiskit backend**

```bash
python main.py \
  --operator Antideriv \
  --model_type QuanONet \
  --quantum_backend qiskit \
  --if_trainable_freq true \
  --num_qubits 3 \
  --num_epochs 200
```

**4. Train FNO with MindSpore backend**

```bash
python main.py \
  --operator Darcy \
  --model_type FNO \
  --classical_backend mindspore \
  --net_size 16 32 3 32 \
  --num_epochs 1000
```

**5. Train Classical Baseline (DeepONet, PyTorch)**

```bash
python main.py \
  --operator Antideriv \
  --model_type DeepONet \
  --net_size 3 100 3 100 \
  --num_epochs 1000
```

**6. Verify all backends**

```bash
python test_backends.py
```

## Reproducing Paper Results

For quick qualitative evaluation, we provide a Jupyter Notebook `visualization.ipynb`. It automatically loads the PDE pre-trained weights (MindSpore `.ckpt`) from the `pretrained_weights/` directory.

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
| `--model_type` | Neural operator type:`QuanONet`, `HEAQNN`,`DeepONet`, `FNN`, `FNO`. | **Required** |
| `--num_train` / `--num_test` | Number of function samples for training/testing.                                                      | `1000` / `1000`            |
| `--train_sample_num`           | Points sampled per function for training.                                                             | `10`                         |
| `--test_sample_num`            | Points sampled per function for testing.                                                              | `100`                        |
| `--num_points`                 | Output resolution (Trunk/Target grid size).                                                           | `100`                        |
| `--num_points_0`               | Input resolution (Branch/Source function size).                                                       | `100` (PDE) / `1000` (ODE) |
| `--num_cal`                    | High-Fidelity resolution for Ground Truth data generation.                                            | `1000` (ODE) / `100` (PDE) |
| `--num_epoch` | Model training rounds. | `1000` |
| `--learning_rate` | Model training learning rate. | `0.0001` |

### 2. Model Architecture (`--net_size`)

| **Model**    | **Format**                                                                          | **Example**                 |
| ------------------ | ----------------------------------------------------------------------------------------- | --------------------------------- |
| **QuanONet** | `[b_depth, b_ansatz, t_depth, t_ansatz]`                                                | `20 2 10 2`                     |
| **HEAQNN** | `[depth, ansatz]` | `32 2` |
| **DeepONet** | `[b_depth, b_width, t_depth, t_width]` *Optional 5th arg for output dim:* `[... p]` | `3 10 3 10` `3 20 3 30 10` |
| **FNN** | `[depth, width]` | `2 20` |
| **FNO**      | `[modes, width, layers, fc_hidden]`                                                     | `16 32 3 32`                    |

### 3. Quantum Specifics

| **Argument**      | **Description**                                                 | **Default** |
| ----------------------- | --------------------------------------------------------------------- | ----------------- |
| `--num_qubits`        | Defines latent dimension $p=2^n$.                                    | `5`             |
| `--if_trainable_freq` | Enable Trainable Frequency (TF) strategy (`true`/`false`).        | `false`         |
| `--scale_coeff`       | Scaling coefficient for encoding.                                     | `0.01`          |
| `--ham_bound`         | Hamiltonian eigenvalue range.                                         | `[-5, 5]`       |
| `--ham_pauli`         | Pauli basis for the Hamiltonian (`X`, `Y`, or `Z`).             | `Z`             |
| `--ham_diag`          | Manually specify exact eigenvalues. Overrides bounds and Pauli basis. | `None`          |

### 4. Backend Selection

| **Argument**           | **Choices**                              | **Default**      |
| ---------------------- | ---------------------------------------- | ---------------- |
| `--quantum_backend`    | `mindquantum`, `torchquantum`, `qiskit`  | `mindquantum`    |
| `--classical_backend`  | `pytorch`, `mindspore`                   | `pytorch`        |

The `--quantum_backend` flag applies to `QuanONet` and `HEAQNN` models.
The `--classical_backend` flag applies to `DeepONet`, `FNN`, and `FNO` models.

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

The repository includes a pre-trained checkpoint (`Antideriv/Antideriv_QuanONet_Net5-1-5-1_Q2_TF_S0.001_1000x100_Seed0/best_model.npz`) matching the lightweight configuration. To evaluate custom models:

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
  booktitle={Forty-second International Conference on Machine Learning},
  year={2025}
}
```
