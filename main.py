#!/usr/bin/env python3
"""
QuanONet Main Entry Point.
Universal launcher for both Quantum (MindSpore) and Classical (PyTorch) models.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common import get_base_parser, load_config, set_random_seed
from utils.backend import backend

def main():
    # 1. Parse Arguments (Common + Quantum Specific)
    parser = get_base_parser()
    
    # Quantum specific args (added here so they show up in --help)
    parser.add_argument('--num_qubits', type=int, help='[Quantum] Number of qubits')
    parser.add_argument('--scale_coeff', type=float, help='[Quantum] Scaling coefficient')
    parser.add_argument('--ham_bound', type=int, nargs='+', help='[Quantum] Hamiltonian bounds')
    parser.add_argument('--if_trainable_freq', type=str, default='false', help='[Quantum] Trainable frequency')
    parser.add_argument('--device_target', type=str, default='CPU', choices=['CPU', 'GPU', 'Ascend'], help='[MS] Device target')
    
    args = parser.parse_args()
    config = load_config(args)

    # 2. Determine Backend
    model_type = config['model_type']
    target_backend = backend.check_compatibility(model_type)
    
    print(f"\n===========================================================")
    print(f" QuanONet Launcher | Model: {model_type} | Operator: {config['operator_type']}")
    print(f"===========================================================")

    solver = None

    # 3. Dynamic Dispatch
    if target_backend == 'mindspore':
        print(f"🌊 Backend: MindSpore (Quantum Mode)")
        try:
            from solvers.solver_ms import MSSolver
            solver = MSSolver(config)
        except Exception as e:
            print(f"❌ Initialization Failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    elif target_backend == 'pytorch':
        print(f"🔥 Backend: PyTorch (Classical Mode)")
        try:
            # Enforce DDE Backend
            os.environ["DDE_BACKEND"] = "pytorch"
            from solvers.solver_dde import DDESolver
            solver = DDESolver(config)
        except Exception as e:
            print(f"❌ Initialization Failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
            
    else:
        print(f"❌ Error: Unknown model type '{model_type}'.")
        sys.exit(1)

    # 4. Run Pipeline
    try:
        set_random_seed(config.get('random_seed', 0))
        
        history = solver.train()
        metrics = solver.evaluate(history)
        
        print("\n✅ Execution Finished Successfully.")
        
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()