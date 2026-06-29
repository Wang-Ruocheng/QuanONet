# QuanONet: Quantum Neural Operators with Adaptive Frequency Strategy

**Official Implementation**

## Introduction

QuanONet is a pure quantum neural operator framework designed for the Noisy Intermediate-Scale Quantum (NISQ) era to solve partial differential equations (PDEs).

## Repository Structure

```text
.
├── main.py                # Unified entry point (auto-backend selection)
├── infer.py               # Standalone inference — Python API and CLI
├── ibm_inference.py       # Real-device deployment on IBM Quantum hardware
├── compare_backends.py    # Cross-backend consistency check (all models × all backends)
├── CITATION.cff           # Citation metadata
├── requirements.txt       # Project dependencies
│
├── scripts/               # Automated reproduction bash scripts
│   ├── reproduce_benchmarks1.sh
│   ├── reproduce_benchmarks2.sh
│   ├── reproduce_capacity.sh
│   ├── reproduce_scaling.sh
│   ├── reproduce_circuit.sh
│   └── reproduce_hamiltonian.sh
│
├── visualization.ipynb    # Jupyter notebook for PDE results visualization
│
├── pretrained_weights/    # Pre-trained checkpoints
│   ├── Advection/         # MindSpore .ckpt — Net40-2-20-2, Q5
│   ├── Darcy/             # MindSpore .ckpt — Net40-2-20-2, Q5
│   ├── RDiffusion/        # MindSpore .ckpt — Net40-2-20-2, Q5
│   │                      #             .npz — Net2-2-2-2,  Q5 (hardware)
│   └── Antideriv/         # NumPy   .npz  — Net5-1-5-1,  Q2 (hardware)
│
├── core/                  # Core model architectures
│   ├── models_ms.py            # MindSpore QuanONetMS / HEAQNNMS / FNNMS / DeepONetMS / FNOMS
│   ├── models_pt.py            # PyTorch QuanONetPT / HEAQNNPT (TQ or Qiskit) / FNOPT
│   ├── quantum_circuits_ms.py  # HEA circuits — MindQuantum backend
│   ├── quantum_circuits_tq.py  # HEA circuits — TorchQuantum backend
│   ├── quantum_circuits_qiskit.py # HEA circuits — Qiskit EstimatorQNN backend
│   ├── quantum_circuits_pl.py  # HEA circuits — PennyLane backend
│   └── layers.py               # Custom MindSpore layers
│
├── solvers/               # Training & evaluation solvers
│   ├── solver_ms.py         # MindSpore solver (QuanONet/HEAQNN/FNN/DeepONet/FNO)
│   ├── solver_pt.py         # PyTorch solver (QuanONet/HEAQNN via TQ or Qiskit)
│   └── solver_dde.py        # DeepXDE/PyTorch solver (DeepONet/FNN/FNO)
│
├── data_utils/            # Data pipelines and generation
│   ├── data_generation.py
│   ├── data_manager.py
│   └── data_processing.py
│
└── utils/                 # Utilities and helpers
    ├── common.py            # Argument parsing (all hyperparameters)
    ├── backend.py           # Backend routing (5-way dispatch)
    ├── weight_transfer.py   # MindSpore .npz → PyTorch state_dict conversion
    ├── logger.py            # Logging and JSON metrics
    ├── metrics.py           # L2 / MSE evaluation
    └── utils.py             # Miscellaneous helpers (parameter counting, etc.)
```

## System Requirements

### Software Dependencies
See `requirements.txt` for the full list of dependencies with pinned versions. Key packages:

| Package | Version | Purpose |
|---|---|---|
| Python | 3.9 | Runtime |
| mindspore | 2.8.0 | MindQuantum backend training |
| mindquantum | 0.11.0 | Quantum circuit simulation |
| torch | 2.8.0 | PyTorch backends (TQ, Qiskit, DeepXDE) |
| DeepXDE | 1.15.0 | Classical operator learning |
| pennylane | 0.38.0 | PennyLane backend |
| qiskit | 1.4.5 | Qiskit backend |

**Tested on:** Ubuntu 20.04 / 22.04 / 24.04, x86\_64. MindSpore does not support Windows.

### Hardware
- **CPU:** Any modern x86\_64 processor. Quantum simulation (MindQuantum/TorchQuantum/PennyLane) runs on CPU only. Multi-core CPUs benefit parallel seed experiments.
- **RAM:** ≥ 16 GB recommended.
- **GPU:** Optional. Used only for classical baselines (DeepONet/FNN/FNO via PyTorch/DeepXDE).
- **Non-standard hardware (optional):** IBM Quantum processor — required only for `ibm_inference.py` (real-device deployment). All simulations run on standard hardware without a quantum device.

## Installation

The framework supports three quantum simulation backends and two classical backends. All can coexist in a single environment.

**Tested environment:** Python 3.9, conda, Ubuntu 22.04 x86\_64.

**Typical install time:** 20–30 minutes on a standard desktop with a broadband connection (dominated by MindSpore and Qiskit package downloads).

```bash
conda create -n quanode python=3.9
conda activate quanode

# Install most packages (skip torch if the instance has a pre-installed NVIDIA build)
grep -v -E '^(torch(vision|audio)?|torchquantum|qiskit-machine-learning)==' requirements.txt | pip install -r /dev/stdin

# Install packages with conflicting transitive dependencies individually
pip install torchquantum==0.1.8 --no-deps          # dill==0.3.4 vs mindspore dill>=0.3.7
pip install qiskit-machine-learning==0.8.4 --no-deps  # numpy>=2.0 vs mindspore numpy<2.0
```

> **GPU instance note:** NVIDIA-provisioned instances often ship with a custom `torch` build (e.g. `torch==2.8.0a0+...nv25.6`). Reinstalling torch from PyPI will break it. Use the `grep` command above to skip torch/torchvision, and install them manually only if the instance has no pre-installed version.

> **TorchQuantum compatibility note:** `torchquantum==0.1.8` has import-time incompatibilities with qiskit 1.x. Apply the following one-time patches after installation:
>
> ```python
> # torchquantum/plugin/__init__.py  — wrap import in try/except
> # torchquantum/plugin/qiskit/__init__.py  — wrap imports in try/except
> # torchquantum/util/utils.py  — wrap qiskit_ibm_runtime import in try/except
> ```
>
> These patches disable only the optional IBM-Q connectivity plugin; local statevector simulation is unaffected.

> **PennyLane compatibility note:** `pennylane==0.38` requires `autoray<0.7`. `autoray>=0.7` removes an internal attribute used by PennyLane 0.38, causing an import error. The pin is included in `requirements.txt`.

### Backend Matrix

| Backend flag | Library required | Notes |
|---|---|---|
| `--quantum_backend mindquantum` (default) | `mindspore`, `mindquantum` | Original QuanONet backend |
| `--quantum_backend torchquantum` | `torchquantum` | PyTorch-native statevector sim |
| `--quantum_backend qiskit` | `qiskit`, `qiskit-machine-learning` | EstimatorQNN + TorchConnector |
| `--quantum_backend pennylane` | `pennylane>=0.38` | PennyLane default.qubit, backprop diff |
| `--classical_backend pytorch` (default) | `torch`, `DeepXDE` | DeepONet / FNN / FNO |
| `--classical_backend mindspore` | `mindspore` | MindSpore FNO / FNN / DeepONet |

## Demo

The fastest way to verify the installation is to run inference on the included pre-trained weights — no training required.

```bash
python infer.py \
  --ckpt pretrained_weights/Antideriv/Antideriv_QuanONet_Net5-1-5-1_Q2_TF_S0.001_1000x100_Seed0/best_model.npz
```

**Expected output:**
```
Model : QuanONet  backend=mindspore
Config: net_size=[5, 1, 5, 1]  num_qubits=2
Output: (1000, 100)
Rel-L2 : 0.0312  (3.12%)
MSE    : 0.000285
MAE    : 0.010341
```

**Expected run time:** ~1 minute on a standard desktop CPU (model loading + forward pass over 1000 test samples).

> The exact metric values will vary slightly from the example above depending on the platform and NumPy version, but should remain in the same order of magnitude.

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

Expected run time: ~80 minutes on a server-class CPU (e.g., Intel Xeon); ~90–120 minutes on a typical desktop CPU.

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

**4. Train TF-QuanONet with PennyLane backend**

```bash
python main.py \
  --operator Antideriv \
  --model_type QuanONet \
  --quantum_backend pennylane \
  --if_trainable_freq true \
  --num_qubits 5 \
  --num_epochs 1000
```

**5. Train FNO with MindSpore backend**

```bash
python main.py \
  --operator Darcy \
  --model_type FNO \
  --classical_backend mindspore \
  --net_size 16 32 3 32 \
  --num_epochs 1000
```

**6. Train Classical Baseline (DeepONet, PyTorch)**

```bash
python main.py \
  --operator Antideriv \
  --model_type DeepONet \
  --net_size 3 100 3 100 \
  --num_epochs 1000
```

Expected run time: ~5 minutes on a standard desktop CPU; ~2 minutes with GPU.

**8. Cross-backend consistency check**

Verifies that all 5 backends (mindquantum, torchquantum, qiskit, pennylane, mindspore-classical) produce identical outputs for the same weights across all model types (QuanONet, HEAQNN, FNN, DeepONet, FNO):

```bash
python compare_backends.py
```

## Inference on Pre-trained Weights

`infer.py` provides a unified inference interface for both MindSpore (`.ckpt`) and PyTorch (`.npz`) checkpoints. Model hyper-parameters are automatically parsed from the checkpoint directory name.

**CLI:**

```bash
# Evaluate on a data file and print metrics
python infer.py \
  --ckpt pretrained_weights/RDiffusion/RDiffusion_QuanONet_Net40-2-20-2_Q5_TF_S0.1_1000x100_Seed0/best_model.ckpt \
  --data data/RDiffusion/RDiffusion_100_100_100_100_100_100.npz

# Run on raw arrays and save predictions
python infer.py --ckpt best_model.ckpt \
                --branch branch.npy --trunk trunk.npy \
                --output preds.npy
```

**Python API:**

```python
from infer import load_model, predict, evaluate

model, cfg = load_model(
    'pretrained_weights/RDiffusion/.../best_model.ckpt',
    branch_in=100, trunk_in=2,
)
preds   = predict(model, branch_input, trunk_input, cfg=cfg)
metrics = evaluate(preds, y_true)   # {'rel_l2', 'mse', 'mae'}
```

## Reproducing Paper Results

For quick qualitative evaluation, we provide a Jupyter Notebook `visualization.ipynb`. It automatically loads the PDE pre-trained weights from the `pretrained_weights/` directory.

The `scripts/` directory contains automated bash scripts to reproduce the primary experimental results reported in the manuscript.

```bash
./scripts/reproduce_benchmarks1.sh
```

| **Script**              | **Description**                                                                                                                                        | **Relevant Section** |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------- |
| `reproduce_benchmarks1.sh`  | **General Benchmarks**: Evaluates TF-QuanONet against Quantum (HEA, TF-HEA) and Classical (DeepONet, FNN) baselines across ODE and PDE operator tasks. | Table 4 (Sec 5.2.2)        |
| `reproduce_benchmarks2.sh`  | **Aligned Parameter Comparison**: Compares TF-QuanONet (net_size 160-2-90-2) against DeepONet and FNO at matched parameter counts.                    | Table 5 & 6 (Sec 5.2.3)    |
| `reproduce_capacity.sh`     | **Implicit Frame Capacity**: Grid search over network width and depth to verify the $\mathcal{O}(p^2)$ implicit frame and analyze error saturation.  | Table 7 (Sec 5.3.1)        |
| `reproduce_scaling.sh`      | **High-Dimensional Scaling Limit**: Sweeps the latent dimension $p$ from 4 to 256. Demonstrates optimization stability in high dimensions.           | Fig 9 (Sec 5.3.1)          |
| `reproduce_circuit.sh`      | **Circuit Architecture Ablation**: Investigates the trade-off between circuit width (qubits) and depth regarding expressivity vs. trainability.        | Table 8 (Sec 5.3.2)        |
| `reproduce_hamiltonian.sh`  | **Hamiltonian Ablation**: Evaluates the impact of Hamiltonian design (Pauli basis, spectral radii, and degeneracy) on model expressivity.              | Fig 10 & 11 (Sec 5.4)      |

## Configuration & Parameters

The `main.py` script accepts the following primary configurations:

### 1. Task & Data Setup

| **Argument**               | **Description**                                                                                 | **Default**              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------ |
| `--operator`                   | Problem type:`Identity`, `Antideriv`, `Homogeneous`, `Nonlinear`, `RDiffusion`, `Advection`, `Darcy`. | **Required**             |
| `--model_type` | Neural operator type:`QuanONet`, `HEAQNN`,`DeepONet`, `FNN`, `FNO`. | **Required** |
| `--num_train` / `--num_test` | Number of function samples for training/testing.                                                      | `1000` / `1000`            |
| `--train_sample_num`           | Points sampled per function for training.                                                             | `10`                         |
| `--test_sample_num`            | Points sampled per function for testing.                                                              | `100`                        |
| `--num_points`                 | Output resolution (Trunk/Target grid size).                                                           | `100`                        |
| `--num_points_0`               | Input resolution (Branch/Source function size).                                                       | `100` (PDE) / `1000` (ODE) |
| `--num_cal`                    | High-Fidelity resolution for Ground Truth data generation.                                            | `1000` (ODE) / `100` (PDE) |
| `--num_epochs` | Model training rounds. | `1000` |
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
| `--quantum_backend`    | `mindquantum`, `torchquantum`, `qiskit`, `pennylane` | `mindquantum`    |
| `--classical_backend`  | `pytorch`, `mindspore`                   | `pytorch`        |

The `--quantum_backend` flag applies to `QuanONet` and `HEAQNN` models.
The `--classical_backend` flag applies to `DeepONet`, `FNN`, and `FNO` models.

## Real-Device Deployment (IBM Quantum)

`ibm_inference.py` deploys and evaluates QuanONet on real superconducting quantum processors (e.g., `ibm_fez`). It automatically analyzes the physical chip topology, routes optimal qubits based on calibration data, and transpiles the circuit for hardware execution.

**Setup:** No separate environment is required. Set your IBM Quantum token as an environment variable:

```bash
export QISKIT_IBM_TOKEN="your_token_here"
```

**Mode 1: Ideal Simulation (local, no token needed)**

```bash
python ibm_inference.py --simulator_only --input_func cos
```

**Mode 2: Real Hardware Execution**

Automatically profiles the least busy QPU and submits a new job:

```bash
python ibm_inference.py --input_func cos
```

**Mode 3: Fetch Existing Results**

Retrieve a completed job without re-queuing:

```bash
python ibm_inference.py --job_id <YOUR_JOB_ID>
```

**Using Custom Weights:**

The default checkpoint is `pretrained_weights/Antideriv/Antideriv_QuanONet_Net5-1-5-1_Q2_TF_S0.001_1000x100_Seed0/best_model.npz`. To evaluate a different model:

```bash
python ibm_inference.py --weight_path pretrained_weights/RDiffusion/.../best_model.npz
```

Architecture dimensions are auto-parsed from the filename. Override manually if needed:

```bash
python ibm_inference.py --weight_path custom.npz --n_qubits 4 --n_branch 5 --n_trunk 5 --n_hidden 1
```

## Citation

If you utilize this framework, the implicit quadratic frame theory, or the real-device deployment pipeline in your research, please cite our primary manuscript:

```bibtex
@article{wang2026quantum,
      author={Wang, Ruocheng and Zhong, Xiaoqiu and Xia, Zhuo and Yan, Junchi},
      title={Quantum Neural Operators with Implicit Quadratic Frame and Expressivity Advantages},
      journal={Nature Machine Intelligence},
      year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
