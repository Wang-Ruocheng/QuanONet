#!/usr/bin/env python3
"""
QuanONet Main Entry Point.
Universal launcher for Quantum (MindSpore/TorchQuantum/Qiskit) and Classical
(PyTorch/MindSpore) models.
"""
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.common import get_base_parser, load_config, set_random_seed
from utils.backend import backend

def main():
    # 1. Parse Arguments
    parser = get_base_parser()
    args = parser.parse_args()
    config = load_config(args)

    # 2. Determine Backend
    model_type       = config['model_type']
    quantum_backend  = config.get('quantum_backend', 'mindquantum')
    classical_backend = config.get('classical_backend', 'pytorch')
    target_backend = backend.check_compatibility(
        model_type, quantum_backend, classical_backend
    )

    print(f"\n===========================================================")
    print(f" QuanONet Launcher | Model: {model_type} | Operator: {config['operator_type']}")
    print(f" Quantum backend: {quantum_backend} | Classical backend: {classical_backend}")
    print(f" Resolved target: {target_backend}")
    print(f"===========================================================")

    # 3. Set random seed
    set_random_seed(config.get('seed', 0))

    # 4. Configure GPU / Device
    if args.gpu is not None:
        gpu_id = int(args.gpu)
        print(f"[Manual] User specified GPU: {gpu_id}")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config['gpu'] = gpu_id
        config['device_target'] = "GPU"
    else:
        if target_backend in ('pytorch', 'pytorch_quantum'):
            try:
                import torch
                if torch.cuda.is_available():
                    print("[Auto] PyTorch Backend -> Found CUDA, defaulting to GPU 0")
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    config['gpu'] = 0
                else:
                    print("[Auto] PyTorch Backend -> No CUDA, using CPU")
                    config['gpu'] = None
            except ImportError:
                config['gpu'] = None
        elif target_backend in ('mindspore', 'mindspore_classical'):
            print("[Auto] MindSpore Backend -> Defaulting to CPU")
            config['device_target'] = "CPU"
            config['gpu'] = None

    solver = None

    # 5. Dynamic Dispatch
    if target_backend == 'mindspore':
        print(f"Backend: MindSpore + MindQuantum (Quantum)")
        try:
            from solvers.solver_ms import MSSolver
            solver = MSSolver(config)
        except Exception as e:
            print(f"Initialization Failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    elif target_backend == 'pytorch_quantum':
        print(f"Backend: PyTorch + {quantum_backend} (Quantum)")
        try:
            from solvers.solver_pt import PTSolver
            solver = PTSolver(config)
        except Exception as e:
            print(f"Initialization Failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    elif target_backend == 'pytorch':
        print(f"Backend: PyTorch + DeepXDE (Classical)")
        try:
            os.environ["DDE_BACKEND"] = "pytorch"
            from solvers.solver_dde import DDESolver
            solver = DDESolver(config)
        except Exception as e:
            print(f"Initialization Failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    elif target_backend == 'mindspore_classical':
        print(f"Backend: MindSpore (Classical)")
        try:
            from solvers.solver_ms import MSSolver
            solver = MSSolver(config)
        except Exception as e:
            print(f"Initialization Failed: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    else:
        print(f"Error: Unknown model type '{model_type}'.")
        sys.exit(1)

    # 6. Run Pipeline
    try:
        history = solver.train()
        metrics = solver.evaluate(history)
        print("\nExecution Finished Successfully.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nExecution Failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
