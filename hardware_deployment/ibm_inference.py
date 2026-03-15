import os
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.primitives import StatevectorEstimator
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator

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

def profile_hardware(backend):
    """Analyzes the hardware to recommend the best physical qubit pairs and their status."""
    print(f"\n--- Hardware Profiling: {backend.name} ---")
    props = backend.properties()
    conf = backend.configuration()
    basis_gates = conf.basis_gates
    gate_name = 'ecr' if 'ecr' in basis_gates else ('cz' if 'cz' in basis_gates else 'cx')

    scored_pairs = []
    processed_pairs = set()

    for pair in conf.coupling_map:
        q1, q2 = pair
        pair_tuple = tuple(sorted((q1, q2)))
        if pair_tuple in processed_pairs:
            continue
        processed_pairs.add(pair_tuple)

        try:
            gate_param = props.gate_property(gate_name, [q1, q2]) or props.gate_property(gate_name, [q2, q1])
            if not gate_param or 'gate_error' not in gate_param:
                continue

            gate_err = gate_param['gate_error'][0]
            ro_err_1 = props.qubit_property(q1, 'readout_error')[0]
            ro_err_2 = props.qubit_property(q2, 'readout_error')[0]
            
            t1_1 = props.qubit_property(q1, 'T1')[0] * 1e6
            t2_1 = props.qubit_property(q1, 'T2')[0] * 1e6
            t1_2 = props.qubit_property(q2, 'T1')[0] * 1e6
            t2_2 = props.qubit_property(q2, 'T2')[0] * 1e6

            score = gate_err + ro_err_1 + ro_err_2
            scored_pairs.append({
                'pair': [q1, q2], 'score': score, 'gate_err': gate_err, 
                'ro_avg': (ro_err_1 + ro_err_2) / 2,
                't1_avg': (t1_1 + t1_2) / 2, 't2_avg': (t2_1 + t2_2) / 2
            })
        except Exception:
            continue

    scored_pairs.sort(key=lambda x: x['score'])
    best = scored_pairs[0]
    
    print(f"Recommended Best Pair : {best['pair']}")
    print(f"  -> Avg T1 Time      : {best['t1_avg']:.1f} µs")
    print(f"  -> Avg T2 Time      : {best['t2_avg']:.1f} µs")
    print(f"  -> Gate Error ({gate_name.upper()}) : {best['gate_err']:.4%}")
    print(f"  -> Avg Readout Error: {best['ro_avg']:.4%}")
    
    return best['pair']

def main():
    # Setup Argument Parser
    parser = argparse.ArgumentParser(description="QuanONet Hardware Inference Script")
    parser.add_argument('--job_id', type=str, default=None, help="Fetch results from an existing IBM Quantum Job ID.")
    parser.add_argument('--weight_path', type=str, default="best_Antideriv_QuanONet_Net5-1-5-1_Q2_TF_S0.001_1000x100_Seed0.npz", help="Path to the pre-trained weights.")
    
    # Optional arguments to manually override network dimensions
    parser.add_argument('--n_qubits', type=int, default=None)
    parser.add_argument('--n_branch', type=int, default=None)
    parser.add_argument('--n_trunk', type=int, default=None)
    parser.add_argument('--n_hidden', type=int, default=None)
    args = parser.parse_args()

    weight_path = args.weight_path
    if not os.path.exists(weight_path):
        print(f"Warning: Pre-trained weights {weight_path} not found.")
        return

    # Default fallbacks
    parsed_q, parsed_b, parsed_t, parsed_h = 2, 5, 5, 1 
    
    #  Net[branch]-[hidden]-[trunk]-[hidden]_Q[qubits]
    match = re.search(r"Net(\d+)-(\d+)-(\d+)-(\d+)_Q(\d+)", weight_path)
    if match:
        parsed_b = int(match.group(1))
        parsed_h = int(match.group(2))
        parsed_t = int(match.group(3))
        parsed_q = int(match.group(5))
        print(f"-> Automatically parsed architecture from filename: {parsed_b}-{parsed_h}-{parsed_t}-{parsed_h} (Qubits: {parsed_q})")

    n_qubits = args.n_qubits if args.n_qubits is not None else parsed_q
    n_branch_layers = args.n_branch if args.n_branch is not None else parsed_b
    n_trunk_layers = args.n_trunk if args.n_trunk is not None else parsed_t
    n_hidden_layers = args.n_hidden if args.n_hidden is not None else parsed_h

    # 1. Load Pre-trained Weights
    data = np.load(weight_path)
    
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

    # 3. Construct Logical Circuit & Hamiltonian
    branch_param = ParameterVector('branch', num_points_0)
    trunk_param = ParameterVector('trunk', 1)
    qc = create_circuit(branch_param, trunk_param, weights, coefficients, n_branch_layers, n_trunk_layers)
    hamiltonian = SparsePauliOp.from_sparse_list([("Z", [i], 1.0) for i in range(n_qubits)], num_qubits=n_qubits)
    
    try:
        qc.draw(output='mpl', filename='logical_circuit.pdf', style='iqx', fold=-1)
        print("-> Logical circuit saved to logical_circuit.pdf")
    except Exception as e:
        print(f"Could not draw circuit: {e}")

    # 4. Ideal Simulation
    print("\n--- Running Ideal Simulation ---")
    pm_ideal = generate_preset_pass_manager(optimization_level=3, backend=None)
    isa_circuit_ideal = pm_ideal.run(qc)
    
    estimator_ideal = StatevectorEstimator()
    job_ideal = estimator_ideal.run([(isa_circuit_ideal, [hamiltonian], inputs)])
    ideal_pred = job_ideal.result()[0].data.evs * 5 / n_qubits + global_bias
    calculate_metrics(ideal_pred, true_solution, name="Ideal Simulator")

    # 5. Real Device Execution or Fetching
    token = os.getenv("QISKIT_IBM_TOKEN")
    if token:
        try:
            service = QiskitRuntimeService(channel="ibm_cloud", token=token)
        except:
            service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            
        if args.job_id:
            # ==========================================
            # MODE: FETCH EXISTING JOB
            # ==========================================
            print(f"\n--- Fetching Existing Job: {args.job_id} ---")
            try:
                job = service.job(args.job_id)
                status = job.status()
                print(f"Job Status: {status}")
                
                if status == "DONE":
                    backend_name = job.backend().name if job.backend() else "Unknown"
                    result = job.result()[0]
                    noisy_evs = result.data.evs
                    noisy_pred = noisy_evs * 5 / n_qubits + global_bias
                    
                    print(f"\n✅ Successfully fetched results!")
                    calculate_metrics(noisy_pred, true_solution, name=f"IBM QPU ({backend_name})")
                else:
                    print(f"⚠️ Job is currently {status}. Cannot fetch results yet.")
                    noisy_pred = None
                    backend_name = "Unknown"
            except Exception as e:
                print(f"❌ Failed to fetch job. Error: {e}")
                noisy_pred = None
                backend_name = "Unknown"

        else:
            # ==========================================
            # MODE: SUBMIT NEW JOB
            # ==========================================
            print("\n--- Searching for the least busy IBM Quantum backend ---")
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
            backend_name = backend.name
            
            # Hardware profiling
            best_pair = profile_hardware(backend)

            # Transpile for physical hardware
            print("\n--- Transpiling Circuit for Real Hardware ---")
            pm_physical = generate_preset_pass_manager(optimization_level=3, backend=backend, initial_layout=best_pair)
            isa_circuit_physical = pm_physical.run(qc)
            
            # Apply physical layout to Hamiltonian
            isa_hamiltonian = hamiltonian.apply_layout(isa_circuit_physical.layout)
            
            ops = isa_circuit_physical.count_ops()
            depth = isa_circuit_physical.depth()
            num_2q = ops.get('ecr', 0) + ops.get('cz', 0) + ops.get('cx', 0)
            
            print(f"Logical Depth (Before)  : {qc.depth()}")
            print(f"Physical Depth (After)  : {depth}")
            print(f"2-Qubit Gates (ECR/CZ)  : {num_2q}")
            
            # Submit the job
            print(f"\n--- Submitting New Job to {backend_name} ---")
            print("Note: This may take some time depending on the IBM Quantum queue...")
            estimator = Estimator(mode=backend)
            estimator.options.default_shots = 10000
            
            try:
                job = estimator.run([(isa_circuit_physical, [isa_hamiltonian], inputs)])
                print(f"Job ID: {job.job_id()} - Waiting for completion...")
                
                # This blocks until the job is done
                result = job.result()[0]
                noisy_evs = result.data.evs
                noisy_pred = noisy_evs * 5 / n_qubits + global_bias
                
                print(f"\n✅ Job Finished!")
                calculate_metrics(noisy_pred, true_solution, name=f"IBM QPU ({backend_name})")
                
            except Exception as e:
                print(f"Hardware execution failed. Error: {e}")
                noisy_pred = None
                backend_name = "Unknown"
    else:
        print("\nIBM Token not found. Skipping real hardware execution.")
        noisy_pred = None
        backend_name = "Unknown"

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