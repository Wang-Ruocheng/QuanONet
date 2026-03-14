import os
from qiskit_ibm_runtime import QiskitRuntimeService

def get_best_pairs(backend, top_k=5):
    """Scans the backend's coupling map to find the qubit pairs with the lowest error rates."""
    props = backend.properties()
    conf = backend.configuration()
    basis_gates = conf.basis_gates
    gate_name = 'ecr' if 'ecr' in basis_gates else ('cz' if 'cz' in basis_gates else 'cx')

    print(f"Target Backend: {backend.name}")
    print(f"Native 2-Qubit Gate: {gate_name.upper()}")
    print("-" * 65)
    print(f"{'Rank':<5} | {'Pair (Q1-Q2)':<15} | {'Total Score':<12} | {'Gate Err':<10} | {'Readout (Avg)':<15}")
    print("-" * 65)

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
            
            # Simple heuristic score: Gate Error + Average Readout Error
            score = gate_err + ro_err_1 + ro_err_2

            scored_pairs.append({
                'pair': [q1, q2], 'score': score, 'gate_err': gate_err, 'ro_avg': (ro_err_1 + ro_err_2) / 2
            })
        except Exception:
            continue

    scored_pairs.sort(key=lambda x: x['score'])
    for i, item in enumerate(scored_pairs[:top_k]):
        p = item['pair']
        print(f"{i+1:<5} | {str(p):<15} | {item['score']:.4%}      | {item['gate_err']:.4%}   | {item['ro_avg']:.4%}")
    
    return scored_pairs[0]['pair']

if __name__ == "__main__":
    token = os.getenv("QISKIT_IBM_TOKEN")
    if not token:
        raise ValueError("Please set the QISKIT_IBM_TOKEN environment variable.")
    
    service = QiskitRuntimeService(channel="ibm_cloud", token=token)
    backend = service.least_busy(operational=True, simulator=False, min_num_qubits=127)
    
    best_pair = get_best_pairs(backend)
    print("-" * 65)
    print(f"[Recommendation] Optimal physical qubit layout for routing: initial_layout={best_pair}")