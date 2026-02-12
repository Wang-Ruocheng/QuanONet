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

    if args.gpu is not None:
        # 【场景 1】用户手动指定了 GPU (例如 --gpu 4)
        print(f"🔧 [Manual] User specified GPU: {args.gpu}")
        
        # 设置可见设备，仅暴露用户指定的 GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        
        config['gpu'] = 0  
        config['device_target'] = "GPU" # 确保 MS 知道要用 GPU

    else:
        # 【场景 2】用户未指定 GPU (自动模式)
        if target_backend == 'pytorch':
            # DDE/PyTorch: 优先使用 GPU
            try:
                import torch
                if torch.cuda.is_available():
                    print("🚀 [Auto] PyTorch Backend -> Found CUDA, defaulting to GPU 0")
                    # 默认使用第一块卡
                    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                    config['gpu'] = 0
                else:
                    print("🐢 [Auto] PyTorch Backend -> No CUDA, using CPU")
                    config['gpu'] = None
            except ImportError:
                config['gpu'] = None

        elif target_backend == 'mindspore':
            # MindSpore: 默认使用 CPU (如您所愿)
            print("🤖 [Auto] MindSpore Backend -> Defaulting to CPU")
            config['device_target'] = "CPU"
            config['gpu'] = None
    # 4. Run Pipeline
    try:
        set_random_seed(config.get('seed', 0))
        
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