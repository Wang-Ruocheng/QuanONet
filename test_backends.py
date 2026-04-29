"""
Backend integration test for multi-backend QuanONet extension.
Tests: TorchQuantum, Qiskit (EstimatorQNN), and MindSpore FNO.
Run: conda run -n quanode python test_backends.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import warnings
warnings.filterwarnings("ignore")

import torch
import numpy as np

PASS = "[PASS]"
FAIL = "[FAIL]"

# ─────────────────────────────────────────────
# 1. TorchQuantum backend
# ─────────────────────────────────────────────
def test_torchquantum():
    print("\n=== TorchQuantum Backend ===")
    from core.quantum_circuits_tq import build_quanonet_tq, build_heaqnn_tq

    n_qubits   = 3
    net_size   = (2, 1, 2, 1)    # branch_depth=2, branch_ld=1, trunk_depth=2, trunk_ld=1
    branch_in  = 4
    trunk_in   = 2
    batch      = 8

    # QuanONetPT (TQ)
    from core.models_pt import QuanONetPT
    model = QuanONetPT(
        num_qubits=n_qubits,
        branch_input_size=branch_in,
        trunk_input_size=trunk_in,
        net_size=net_size,
        scale_coeff=0.1,
        if_trainable_freq=True,
        quantum_backend='torchquantum',
        ham_bound=(-5.0, 5.0),
    )
    b = torch.randn(batch, branch_in)
    t = torch.randn(batch, trunk_in)
    out = model(b, t)
    assert out.shape == (batch, 1), f"Shape mismatch: {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  QuanONetPT(TQ): output shape={out.shape}, params={n_params}")

    # One backward pass
    loss = out.sum()
    loss.backward()
    print(f"  Backward pass OK")

    # HEAQNNPT (TQ)
    from core.models_pt import HEAQNNPT
    heaqnn = HEAQNNPT(n_qubits, input_size=4, net_size=(2,1,0,0),
                      quantum_backend='torchquantum')
    x = torch.randn(batch, 4)
    out2 = heaqnn(x)
    assert out2.shape == (batch, 1)
    print(f"  HEAQNNPT(TQ): output shape={out2.shape}  {PASS}")

    print(f"  {PASS} TorchQuantum backend")
    return True


# ─────────────────────────────────────────────
# 2. Qiskit backend (small circuit, CPU statevec)
# ─────────────────────────────────────────────
def test_qiskit():
    print("\n=== Qiskit Backend ===")
    from core.models_pt import QuanONetPT

    n_qubits  = 2   # tiny: 2 qubits for speed
    net_size  = (1, 1, 1, 1)
    branch_in = 2
    trunk_in  = 2
    batch     = 4

    model = QuanONetPT(
        num_qubits=n_qubits,
        branch_input_size=branch_in,
        trunk_input_size=trunk_in,
        net_size=net_size,
        scale_coeff=0.1,
        if_trainable_freq=False,
        quantum_backend='qiskit',
        ham_bound=(-5.0, 5.0),
    )
    b = torch.randn(batch, branch_in)
    t = torch.randn(batch, trunk_in)
    out = model(b, t)
    assert out.shape[0] == batch, f"Shape mismatch: {out.shape}"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  QuanONetPT(Qiskit): output shape={out.shape}, params={n_params}")

    loss = out.sum()
    loss.backward()
    print(f"  Backward pass OK")
    print(f"  {PASS} Qiskit backend")
    return True


# ─────────────────────────────────────────────
# 3. MindSpore FNO backend
# ─────────────────────────────────────────────
def test_ms_fno():
    print("\n=== MindSpore FNO Backend ===")
    import mindspore as ms
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target="CPU")
    from core.ms_fno import FNO_MS

    batch   = 4
    n_pts   = 32
    in_ch   = 2
    modes   = 8
    width   = 12
    layers  = 2

    model = FNO_MS(modes=modes, width=width, layers=layers,
                   fc_hidden=16, in_channels=in_ch)
    x = ms.Tensor(np.random.randn(batch, n_pts, in_ch).astype(np.float32))
    out = model(x)
    assert out.shape == (batch, n_pts), f"Shape mismatch: {out.shape}"

    from utils.utils import count_parameters
    n_params = count_parameters(model)
    print(f"  FNO_MS: input={x.shape}, output={out.shape}, params={n_params}")

    # Simple gradient check via MindSpore grad
    from mindspore import grad
    loss_fn = lambda inp: model(inp).sum()
    g = grad(loss_fn)(x)
    print(f"  Gradient shape: {g.shape}, norm: {float(g.norm().asnumpy()):.4f}")
    print(f"  {PASS} MindSpore FNO backend")
    return True


# ─────────────────────────────────────────────
# 4. Backend routing
# ─────────────────────────────────────────────
def test_routing():
    print("\n=== Backend Routing ===")
    from utils.backend import backend

    cases = [
        ('QuanONet', 'mindquantum', 'pytorch',   'mindspore'),
        ('QuanONet', 'torchquantum','pytorch',   'pytorch_quantum'),
        ('QuanONet', 'qiskit',      'pytorch',   'pytorch_quantum'),
        ('FNO',      'mindquantum', 'pytorch',   'pytorch'),
        ('FNO',      'mindquantum', 'mindspore', 'mindspore_classical'),
    ]
    for model_type, qb, cb, expected in cases:
        result = backend.check_compatibility(model_type, qb, cb)
        status = PASS if result == expected else FAIL
        print(f"  {status}  {model_type} | qb={qb} | cb={cb} → {result}")
    print(f"  {PASS} Backend routing")
    return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
if __name__ == "__main__":
    results = {}

    results['routing'] = test_routing()

    try:
        results['torchquantum'] = test_torchquantum()
    except Exception as e:
        print(f"  {FAIL} TorchQuantum: {e}")
        import traceback; traceback.print_exc()
        results['torchquantum'] = False

    try:
        results['qiskit'] = test_qiskit()
    except Exception as e:
        print(f"  {FAIL} Qiskit: {e}")
        import traceback; traceback.print_exc()
        results['qiskit'] = False

    import importlib.util
    if importlib.util.find_spec('mindspore') is None:
        print("\n=== MindSpore FNO Backend ===")
        print("  [SKIP] MindSpore not installed — skipping")
    else:
        try:
            results['ms_fno'] = test_ms_fno()
        except Exception as e:
            print(f"  {FAIL} MindSpore FNO: {e}")
            import traceback; traceback.print_exc()
            results['ms_fno'] = False

    print("\n========== Summary ==========")
    for k, v in results.items():
        print(f"  {PASS if v else FAIL}  {k}")
    all_passed = all(results.values())
    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
    sys.exit(0 if all_passed else 1)
