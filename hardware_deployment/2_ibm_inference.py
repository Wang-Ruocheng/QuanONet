import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator
from qiskit_ibm_runtime import QiskitRuntimeService

import warnings
warnings.filterwarnings("ignore")

def create_circuit(branch_inputs, trunk_inputs, weights, coefficients, n_branch_layers, n_trunk_layers):
    """Constructs the native QuanONet circuit architecture."""
    def entangle(qc, wires):
        n = len(wires)
        for i in range(n): qc.cx(wires[(i+1) % n], wires[i])

    def ansatz(qc, weights_layer, wires):
        for j in wires:
            qc.ry(weights_layer[0][j], j)
            qc.rz(weights_layer[1][j], j)
            qc.ry(weights_layer[2][j], j)

    def encode(qc, coeffs, features, wires):
        for j in wires:
            angle = features[j] * float(coeffs[0][j]) + float(coeffs[1][j])
            qc.rx(angle, j)

    branch_size, trunk_size = len(branch_inputs), len(trunk_inputs)
    n_hidden_layers, n_wires = weights.shape[1], weights.shape[-1]
    qc = QuantumCircuit(n_wires)

    # Trunk Encoding & Evolution
    for i in range(n_trunk_layers):
        trunk = [trunk_inputs[(i * n_wires % trunk_size + j) % trunk_size] for j in range(n_wires)]
        encode(qc, coefficients[i], trunk, range(n_wires))
        for h in range(n_hidden_layers):
            ansatz(qc, weights[i][h], range(n_wires))
            entangle(qc, range(n_wires))

    # Branch Encoding & Evolution
    for i in range(n_branch_layers):
        branch = [branch_inputs[(i * n_wires % branch_size + j) % branch_size] for j in range(n_wires)]
        encode(qc, coefficients[n_trunk_layers + i], branch, range(n_wires))
        for h in range(n_hidden_layers):
            ansatz(qc, weights[n_trunk_layers + i][h], range(n_wires))
            entangle(qc, range(n_wires))
    return qc

def calculate_metrics(pred, true, name=""):
    mse = np.mean((pred - true) ** 2)
    l2_rel = np.linalg.norm(pred - true) / np.linalg.norm(true)
    print(f"[{name}] MSE: {mse:.2e} | Relative L2: {l2_rel:.2%}")
    return mse, l2_rel

def main():
    # 1. Load Pre-trained Weights (Assuming file is in the parent directory or specify path)
    weight_path = "best_Inverse_QuanONet_Net5-1-5-1_Q2_TF_S0.01_1000x100_Seed0.npz"
    if not os.path.exists(weight_path):
        print(f"Warning: Pre-trained weights {weight_path} not found. Using random weights for demonstration.")
        # [Fallback logic for random weights can be added here if needed]
        return

    data = np.load(weight_path)
    n_qubits, n_branch_layers, n_trunk_layers, n_hidden_layers = 2, 5, 5, 1
    
    raw_weights = data["QuanONet.weight"]
    weights = raw_weights.reshape(n_branch_layers + n_trunk_layers, n_hidden_layers, 3, n_qubits)
    t_w = data["trunk_LinearLayer.Net2.weights"].reshape(n_trunk_layers, n_qubits)
    t_b = data["trunk_LinearLayer.Net2.bias"].reshape(n_trunk_layers, n_qubits)
    b_w = data["branch_LinearLayer.Net2.weights"].reshape(n_branch_layers, n_qubits)
    b_b = data["branch_LinearLayer.Net2.bias"].reshape(n_branch_layers, n_qubits)

    coefficients = np.stack([np.concatenate([t_w, b_w], axis=0), np.concatenate([t_b, b_b], axis=0)], axis=1)
    global_bias = float(data["bias"])

    # 2. Prepare Inputs
    num_points_0, num_points = 10, 100
    branch_vec = np.cos(np.pi * np.linspace(0, 1, num_points_0))
    trunk_vec = np.linspace(0, 1, num_points)
    inputs = np.hstack([np.tile(branch_vec, (num_points, 1)), trunk_vec.reshape(-1, 1)])
    true_solution = np.sin(np.pi * trunk_vec) / np.pi

    # 3. Construct Logical Circuit
    branch_param = ParameterVector('branch', num_points_0)
    trunk_param = ParameterVector('trunk', 1)
    qc = create_circuit(branch_param, trunk_param, weights, coefficients, n_branch_layers, n_trunk_layers)
    
    # Save Logical Circuit Diagram
    try:
        qc.draw(output='mpl', filename='logical_circuit.pdf', style='iqx', fold=-1)
        print("-> Logical circuit saved to logical_circuit.pdf")
    except Exception as e:
        print(f"Could not draw circuit (pylatexenc missing?): {e}")

    # 4. Ideal Simulation
    print("\n--- Running Ideal Simulation ---")
    hamiltonian = SparsePauliOp.from_sparse_list([("Z", [i], 1.0) for i in range(n_qubits)], num_qubits=n_qubits)
    pm_ideal = generate_preset_pass_manager(optimization_level=3, backend=None)
    isa_circuit_ideal = pm_ideal.run(qc)
    
    estimator_ideal = StatevectorEstimator()
    job_ideal = estimator_ideal.run([(isa_circuit_ideal, [hamiltonian], inputs)])
    ideal_pred = job_ideal.result()[0].data.evs * 5 / n_qubits + global_bias
    calculate_metrics(ideal_pred, true_solution, name="Ideal Simulator")

    # 5. Fetch Real Device Results (From existing job to avoid queue waiting)
    print("\n--- Fetching Real Hardware Results ---")
    token = os.getenv("QISKIT_IBM_TOKEN")
    if token:
        service = QiskitRuntimeService(channel="ibm_cloud", token=token)
        # Note: Replace with the actual Job ID you ran and recorded
        job_id = 'd58m4h3ht8fs73a3tapg' 
        print(f"Fetching Job ID: {job_id}...")
        try:
            job = service.job(job_id)
            backend_name = job.backend().name
            noisy_evs = job.result()[0].data.evs
            noisy_pred = noisy_evs * 5 / n_qubits + global_bias
            calculate_metrics(noisy_pred, true_solution, name=f"IBM QPU ({backend_name})")
        except Exception as e:
            print(f"Could not fetch hardware job. Error: {e}")
            noisy_pred, backend_name = None, "Unknown"
    else:
        print("IBM Token not found. Skipping real hardware fetch.")
        noisy_pred = None

    # 6. Plotting
    plt.figure(figsize=(8, 5))
    plt.plot(trunk_vec, true_solution, 'k-', linewidth=2, label='Ground Truth')
    plt.plot(trunk_vec, ideal_pred, 'b--', linewidth=2, label='Ideal QuanONet')
    if noisy_pred is not None:
        plt.plot(trunk_vec, noisy_pred, 'r-', linewidth=1.5, alpha=0.7, label=f'Noisy QuanONet ({backend_name})')
    
    plt.title("QuanONet Inference: Ideal vs. Real Superconducting Hardware", fontsize=12)
    plt.xlabel("Spatial Coordinate x")
    plt.ylabel("Operator Output u(x)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("hardware_inference_comparison.pdf", dpi=300)
    print("\n-> Inference plot saved to hardware_inference_comparison.pdf")

if __name__ == "__main__":
    main()