# Real-Device Deployment on IBM Quantum

This folder contains the scripts used to deploy and evaluate **QuanONet** on real superconducting quantum processors (e.g., `ibm_fez`, `ibm_torino`), as reported in **Section 5.5** of our manuscript.

Due to potential dependency conflicts between different quantum frameworks, we provide a standalone environment for real-device inference using Qiskit.

## Setup

1. Create a fresh virtual environment and install the required packages:

   ```bash
pip install -r requirements_qiskit.txt
   ```

2. Set your IBM Quantum token as an environment variable:
   
   ```bash
   export QISKIT_IBM_TOKEN="your_token_here"
   ```

## Workflow

* **`1_backend_analysis.py`** : Scans the physical coupling map of the target IBM backend, evaluates T1/T2 relaxation times, and calculates ECR/CZ gate fidelities to find the optimal physical qubit pairs for circuit routing.
* **`2_ibm_inference.py`** : Constructs the logical QuanONet circuit, transpiles it to the native basis gates, executes it on the noisy QPU with Measurement Error Mitigation (Resilience Level 1/2), and plots the comparison against the ideal statevector simulation.
