"""
infer.py — Standalone inference for QuanONet / HEAQNN / DeepONet / FNN / FNO.

Supports MindSpore (.ckpt) and PyTorch (.pt / .npz) checkpoints.
Model hyper-parameters are auto-parsed from the standard checkpoint directory
naming convention; individual fields can be overridden via keyword args or CLI.

Python API
----------
    from infer import load_model, predict, evaluate

    model, cfg = load_model(
        'pretrained_weights/Advection/.../best_model.ckpt',
        branch_in=100, trunk_in=2,
    )
    preds   = predict(model, branch_input, trunk_input, cfg=cfg)
    metrics = evaluate(preds, y_true)

CLI
---
    # Evaluate on a data file with ground truth
    python infer.py --ckpt pretrained_weights/Advection/.../best_model.ckpt \\
                    --data data/Advection/Advection_1000_1000_100_100_10_100.npz

    # Run on raw numpy arrays, save output
    python infer.py --ckpt best_model.ckpt \\
                    --branch branch.npy --trunk trunk.npy \\
                    --output preds.npy
"""
import os
import re
import argparse
import numpy as np

# ── Config parsing from checkpoint path ──────────────────────────────────────

_NET_RE   = re.compile(r'Net(\d+)-(\d+)-(\d+)-(\d+)')
_Q_RE     = re.compile(r'_Q(\d+)')
_S_RE     = re.compile(r'_S([\d.]+)')
_TF_RE    = re.compile(r'_(TF|NTF)_')
_MODEL_RE = re.compile(r'_(QuanONet|HEAQNN|DeepONet|FNN|FNO)_')

_DEFAULTS = {
    'model_type':        'QuanONet',
    'num_qubits':        5,
    'net_size':          [40, 2, 20, 2],
    'scale_coeff':       0.1,
    'if_trainable_freq': True,
    'ham_bound':         [-5.0, 5.0],
    'ham_diag':          None,
    'quantum_backend':   'mindquantum',
    'batch_size':        128,
}


def _parse_path(ckpt_path: str) -> dict:
    """Extract hyper-parameters encoded in the checkpoint directory name."""
    name = os.path.basename(os.path.dirname(os.path.abspath(ckpt_path)))
    cfg  = {}
    m = _MODEL_RE.search(name)
    if m:
        cfg['model_type'] = m.group(1)
    m = _NET_RE.search(name)
    if m:
        cfg['net_size'] = [int(m.group(i)) for i in range(1, 5)]
    m = _Q_RE.search(name)
    if m:
        cfg['num_qubits'] = int(m.group(1))
    m = _S_RE.search(name)
    if m:
        cfg['scale_coeff'] = float(m.group(1))
    m = _TF_RE.search(name)
    if m:
        cfg['if_trainable_freq'] = (m.group(1) == 'TF')
    return cfg


def _resolve_config(ckpt_path: str, overrides: dict) -> dict:
    cfg = {**_DEFAULTS, **_parse_path(ckpt_path)}
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


# ── Backend detection ─────────────────────────────────────────────────────────

def _detect_backend(ckpt_path: str, cfg: dict) -> str:
    ext = os.path.splitext(ckpt_path)[1].lower()
    if ext == '.ckpt':
        return 'mindspore'
    # .pt / .npz → PyTorch path; quantum_backend selects TQ or Qiskit
    return cfg.get('quantum_backend', 'torchquantum')


# ── Model construction ────────────────────────────────────────────────────────

def _build_ms_model(cfg: dict, branch_in: int, trunk_in: int):
    import mindspore as ms
    from core.models import QuanONet, HEAQNN, FNN, DeepONet
    from core.quantum_circuits import generate_simple_hamiltonian, ham_diag_to_operator

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')

    mt        = cfg['model_type']
    net_size  = tuple(cfg['net_size'])
    n_q       = int(cfg['num_qubits'])
    scale     = float(cfg['scale_coeff'])
    if_tf     = bool(cfg['if_trainable_freq'])
    ham_bound = cfg['ham_bound']

    if mt in ('QuanONet', 'HEAQNN'):
        if cfg.get('ham_diag') is not None:
            ham = ham_diag_to_operator(cfg['ham_diag'], n_q)
        else:
            ham = generate_simple_hamiltonian(n_q, lower_bound=ham_bound[0],
                                              upper_bound=ham_bound[1])
        if mt == 'QuanONet':
            return QuanONet(n_q, branch_in, trunk_in, net_size, ham, scale, if_tf)
        else:
            return HEAQNN(n_q, branch_in, net_size, ham, scale, if_tf)

    if mt == 'DeepONet':
        return DeepONet(branch_in, trunk_in, net_size)
    if mt == 'FNN':
        return FNN(branch_in, 1, net_size)
    if mt == 'FNO':
        from core.ms_fno import FNO_MS
        ns = list(cfg['net_size'])
        modes, width = int(ns[0]), int(ns[1])
        depth = int(ns[2]) if len(ns) > 2 else 3
        fc_h  = int(ns[3]) if len(ns) > 3 else 32
        return FNO_MS(modes=modes, width=width, layers=depth,
                      fc_hidden=fc_h, in_channels=branch_in)
    raise ValueError(f"Unknown model_type: {mt}")


def _build_pt_model(cfg: dict, branch_in: int, trunk_in: int):
    from core.models_pt import QuanONetPT, HEAQNNPT

    mt       = cfg['model_type']
    net_size = tuple(cfg['net_size'])
    n_q      = int(cfg['num_qubits'])
    scale    = float(cfg['scale_coeff'])
    if_tf    = bool(cfg['if_trainable_freq'])
    hb       = tuple(cfg['ham_bound'])
    qb       = cfg.get('quantum_backend', 'torchquantum')

    if mt == 'QuanONet':
        return QuanONetPT(n_q, branch_in, trunk_in, net_size,
                          scale_coeff=scale, if_trainable_freq=if_tf,
                          quantum_backend=qb, ham_bound=hb,
                          ham_diag=cfg.get('ham_diag'))
    if mt == 'HEAQNN':
        return HEAQNNPT(n_q, branch_in, net_size,
                        scale_coeff=scale, if_trainable_freq=if_tf,
                        quantum_backend=qb, ham_bound=hb,
                        ham_diag=cfg.get('ham_diag'))
    raise NotImplementedError(
        f"PyTorch inference is only implemented for QuanONet and HEAQNN. "
        f"For {mt}, use the MindSpore .ckpt checkpoint."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def load_model(ckpt_path: str, branch_in: int, trunk_in: int = 0, **overrides):
    """
    Load a model from a checkpoint file.

    Args:
        ckpt_path:  Path to checkpoint (.ckpt for MindSpore, .pt/.npz for PyTorch).
        branch_in:  Branch input dimension (number of sensor points / features).
        trunk_in:   Trunk input dimension (coordinate dimension). 0 for HEAQNN/FNN.
        **overrides: Override any auto-parsed config field, e.g. num_qubits=3,
                     quantum_backend='torchquantum', net_size=[20,2,10,2].

    Returns:
        (model, cfg): model is ready for inference; cfg is the resolved config dict.
    """
    cfg     = _resolve_config(ckpt_path, overrides)
    backend = _detect_backend(ckpt_path, cfg)
    cfg['_backend'] = backend

    if backend == 'mindspore':
        from mindspore.train.serialization import load_checkpoint, load_param_into_net
        model = _build_ms_model(cfg, branch_in, trunk_in)
        load_param_into_net(model, load_checkpoint(ckpt_path))
        model.set_train(False)
    else:
        import torch
        model = _build_pt_model(cfg, branch_in, trunk_in)
        if ckpt_path.endswith('.npz'):
            d = np.load(ckpt_path)
            sd = {k: torch.tensor(d[k]) for k in d.files}
        else:
            sd = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(sd)
        model.eval()

    return model, cfg


def predict(model, branch_input: np.ndarray, trunk_input: np.ndarray = None,
            cfg: dict = None, batch_size: int = 128) -> np.ndarray:
    """
    Run batched inference.

    Args:
        model:        Model returned by load_model().
        branch_input: numpy array (N, branch_features).
        trunk_input:  numpy array (N, trunk_features). None for HEAQNN / FNN.
        cfg:          Config dict from load_model() (needed to choose forward signature).
        batch_size:   Number of samples per forward call.

    Returns:
        Predictions as numpy array (N, 1).
    """
    backend    = (cfg or {}).get('_backend', 'mindspore')
    model_type = (cfg or {}).get('model_type', 'QuanONet')
    has_trunk  = trunk_input is not None and model_type in ('QuanONet', 'DeepONet')
    n          = branch_input.shape[0]
    preds      = []

    if backend == 'mindspore':
        import mindspore as ms
        for s in range(0, n, batch_size):
            b = ms.Tensor(branch_input[s:s+batch_size].astype(np.float32))
            if has_trunk:
                t   = ms.Tensor(trunk_input[s:s+batch_size].astype(np.float32))
                out = model((b, t))
            else:
                out = model(b)
            preds.append(out.asnumpy())
    else:
        import torch
        device = next(model.parameters()).device
        with torch.no_grad():
            for s in range(0, n, batch_size):
                b = torch.tensor(branch_input[s:s+batch_size].astype(np.float32), device=device)
                if has_trunk:
                    t   = torch.tensor(trunk_input[s:s+batch_size].astype(np.float32), device=device)
                    out = model(b, t)
                else:
                    out = model(b)
                preds.append(out.cpu().numpy())

    return np.concatenate(preds, axis=0)


def evaluate(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Compute Rel-L2, MSE, MAE."""
    diff   = y_pred - y_true
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(y_true) + 1e-8))
    return {
        'rel_l2': rel_l2,
        'mse':    float(np.mean(diff ** 2)),
        'mae':    float(np.mean(np.abs(diff))),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parser():
    p = argparse.ArgumentParser(
        description='QuanONet inference',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--ckpt',    required=True,
                   help='Checkpoint path (.ckpt / .pt / .npz)')
    p.add_argument('--data',    default=None,
                   help='.npz data file with test_branch_input / test_trunk_input / test_output')
    p.add_argument('--branch',  default=None, help='Branch input .npy (alternative to --data)')
    p.add_argument('--trunk',   default=None, help='Trunk input .npy (optional)')
    p.add_argument('--output',  default=None, help='Save predictions to .npy or .npz')
    p.add_argument('--batch_size', type=int, default=128)
    # Model config overrides (all optional — auto-parsed from path if omitted)
    p.add_argument('--model_type',       default=None)
    p.add_argument('--num_qubits',       type=int, default=None)
    p.add_argument('--net_size',         type=int, nargs='+', default=None)
    p.add_argument('--scale_coeff',      type=float, default=None)
    p.add_argument('--quantum_backend',  default=None,
                   choices=['mindquantum', 'torchquantum', 'qiskit'])
    p.add_argument('--ham_bound',        type=float, nargs=2, default=None)
    return p


def main():
    args = _parser().parse_args()

    # ── Load input data ───────────────────────────────────────────────────────
    y_true = None
    if args.data:
        d      = np.load(args.data)
        branch = d['test_branch_input']
        trunk  = d['test_trunk_input'] if 'test_trunk_input' in d.files else None
        if 'test_output' in d.files:
            y_true = d['test_output']
    elif args.branch:
        branch = np.load(args.branch)
        trunk  = np.load(args.trunk) if args.trunk else None
    else:
        raise SystemExit("Provide --data <file.npz> or --branch <file.npy>.")

    branch_in = branch.shape[1]
    trunk_in  = trunk.shape[1] if trunk is not None else 0

    # ── Load model ────────────────────────────────────────────────────────────
    overrides = dict(
        model_type      = args.model_type,
        num_qubits      = args.num_qubits,
        net_size        = args.net_size,
        scale_coeff     = args.scale_coeff,
        quantum_backend = args.quantum_backend,
        ham_bound       = args.ham_bound,
    )
    model, cfg = load_model(args.ckpt, branch_in=branch_in, trunk_in=trunk_in, **overrides)
    print(f"Model : {cfg['model_type']}  backend={cfg['_backend']}")
    print(f"Config: net_size={cfg['net_size']}  num_qubits={cfg.get('num_qubits', '-')}")

    # ── Inference ─────────────────────────────────────────────────────────────
    preds = predict(model, branch, trunk, cfg=cfg, batch_size=args.batch_size)
    print(f"Output: {preds.shape}")

    # ── Metrics ───────────────────────────────────────────────────────────────
    if y_true is not None:
        m = evaluate(preds, y_true)
        print(f"Rel-L2 : {m['rel_l2']:.4f}  ({m['rel_l2']:.2%})")
        print(f"MSE    : {m['mse']:.6f}")
        print(f"MAE    : {m['mae']:.6f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    if args.output:
        if args.output.endswith('.npz'):
            np.savez(args.output, predictions=preds,
                     **(evaluate(preds, y_true) if y_true is not None else {}))
        else:
            np.save(args.output, preds)
        print(f"Saved  : {args.output}")

    return preds


if __name__ == '__main__':
    main()
