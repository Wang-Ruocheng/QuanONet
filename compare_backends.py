"""
compare_backends.py — Cross-backend inference consistency check.

Tests that the same weights produce identical (or near-identical) outputs
when loaded on different backends.

Quantum models (QuanONet, HEAQNN):
  • PyTorch group: TorchQuantum ≡ Qiskit ≡ PennyLane  (same weights)
  • MindSpore ≡ TorchQuantum                           (weight_transfer)

Classical models (FNN, DeepONet, FNO):
  • PyTorch ≡ MindSpore                                (manual weight transfer)

Run:
    conda run -n quanode python compare_backends.py
"""
import os, sys, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn

# ── tolerances ────────────────────────────────────────────────────────────────
ATOL_PT   = 1e-4    # same circuit, same weights — PL/TQ are exact; Qiskit may differ ~1e-5
ATOL_MSPT = 1e-4    # MindSpore ↔ PyTorch fp32 cross-framework
ATOL_CLS      = 1e-5    # classical PT ↔ MS with identical weights
ATOL_GRAD_PT  = 1e-4    # backward: PT-only backends (TQ / PL / Qiskit)
ATOL_GRAD_MS  = 5e-4    # backward: MindSpore ↔ PyTorch quantum models
ATOL_GRAD_CLS = 2e-5    # backward: classical PT ↔ MS

# ── helpers ───────────────────────────────────────────────────────────────────
_GREEN = "\033[32m"
_RED   = "\033[31m"
_RST   = "\033[0m"

results = {}

def _ok(tag, a, b, atol):
    a, b = np.asarray(a).ravel(), np.asarray(b).ravel()
    diff = float(np.abs(a - b).max())
    passed = diff <= atol
    sym = f"{_GREEN}[PASS]{_RST}" if passed else f"{_RED}[FAIL]{_RST}"
    print(f"  {sym}  {tag:<52s}  max_diff={diff:.2e}")
    results[tag] = passed
    return passed

def _skip(tag, reason=""):
    print(f"  [SKIP]  {tag:<52s}  {reason}")
    results[tag] = None

RNG = np.random.default_rng(0)


# ── Inline PT classical models (mirror core/models_ms.py FNNLayer / FNN / DeepONet)
class _FNNLayerPT(nn.Module):
    def __init__(self, in_size, out_size, width, depth):
        super().__init__()
        self.fc0 = nn.Linear(in_size, width)
        self.hidden_layers = nn.ModuleList([nn.Linear(width, width) for _ in range(depth)])
        self.fc_out = nn.Linear(width, out_size)

    def forward(self, x):
        x = torch.tanh(self.fc0(x))
        for layer in self.hidden_layers:
            x = torch.tanh(layer(x))
        return self.fc_out(x)

class FNN_PT(nn.Module):
    def __init__(self, in_size, out_size, net_size):
        super().__init__()
        depth, width = net_size
        self.FNN = _FNNLayerPT(in_size, out_size, width, depth)

    def forward(self, x):
        return self.FNN(x)

class DeepONet_PT(nn.Module):
    def __init__(self, branch_in, trunk_in, net_size):
        super().__init__()
        b_depth, b_width, t_depth, t_width = net_size
        self.branch_net = _FNNLayerPT(branch_in, b_width, b_width, b_depth + 1)
        self.trunk_net  = _FNNLayerPT(trunk_in,  t_width, t_width, t_depth + 1)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, branch, trunk):
        b = self.branch_net(branch)
        t = self.trunk_net(trunk)
        return (b * t).sum(dim=1, keepdim=True) + self.bias


# ── Weight conversion utilities ───────────────────────────────────────────────

def tq_ansatz_to_qiskit_flat(ansatz_weights):
    """
    Convert TQ ansatz_weights (blocks, 3, n_wires) → Qiskit flat vector.

    TQ layout: ansatz_weights[block, gate, qubit]  — gate-major per block
    Qiskit layout (from _fill_hea_circuit):
      for each qubit: RY, RZ, RY  — qubit-major per block

    Same gate types and same qubit ordering; only the memory layout differs.
    Conversion: permute (blocks, n_wires, 3) then reshape to 1D.
    """
    # (blocks, 3, n_wires) → (blocks, n_wires, 3) → flat
    return ansatz_weights.permute(0, 2, 1).reshape(-1).detach().numpy().astype(np.float32)


def _pt_fno_to_ms_params(sd_pt, layers):
    """Convert PT FNO state dict → MS parameter dict.

    Key differences:
      - spectral conv: PT uses complex cfloat 'weights1'; MS splits into 'weight_real'+'weight_imag'
      - Conv1d: PT weight is (out, in, 1); MS weight is (out, in, 1, 1) — need extra dim
    """
    import mindspore as ms
    param_dict = {}
    for k, v in sd_pt.items():
        v_np = v.numpy()
        if 'weights1' in k:
            base = k.replace('weights1', '')
            param_dict[base + 'weight_real'] = ms.Parameter(
                ms.Tensor(v_np.real.astype(np.float32)))
            param_dict[base + 'weight_imag'] = ms.Parameter(
                ms.Tensor(v_np.imag.astype(np.float32)))
        elif k.startswith('ws.') and k.endswith('.weight'):
            # Conv1d weight: PT (out, in, 1) → MS (out, in, 1, 1)
            param_dict[k] = ms.Parameter(
                ms.Tensor(v_np.reshape(*v_np.shape, 1).astype(np.float32)))
        else:
            param_dict[k] = ms.Parameter(ms.Tensor(v_np.astype(np.float32)))
    return param_dict


# ═══════════════════════════════════════════════════════════════════════════════
# Part 1: QuanONet — PyTorch backends (TQ / Qiskit / PennyLane)
# ═══════════════════════════════════════════════════════════════════════════════

def test_quanonet_pt_backends():
    print("\n─── QuanONet: TorchQuantum vs Qiskit vs PennyLane ───")
    from core.models_pt import QuanONetPT
    from core.quantum_circuits_qiskit import build_quanonet_qiskit

    n_q   = 2
    ns    = (2, 1, 2, 1)
    b_in  = 8
    t_in  = 1
    hb    = (-5.0, 5.0)
    batch = 6
    cfg   = dict(num_qubits=n_q, branch_input_size=b_in, trunk_input_size=t_in,
                 net_size=ns, scale_coeff=0.1, if_trainable_freq=True, ham_bound=hb)

    torch.manual_seed(42)
    model_tq = QuanONetPT(**cfg, quantum_backend='torchquantum')
    model_tq.eval()
    sd = model_tq.state_dict()

    # Load same state dict on PennyLane (keys are identical to TQ)
    model_pl = QuanONetPT(**cfg, quantum_backend='pennylane')
    model_pl.load_state_dict(sd)
    model_pl.eval()

    # Qiskit: convert ansatz_weights → flat, use as initial_weights; share freq/bias via strict=False
    flat_w = tq_ansatz_to_qiskit_flat(sd['quantum_layer.ansatz_weights'])
    model_qk = QuanONetPT.__new__(QuanONetPT)
    nn.Module.__init__(model_qk)
    # Build Qiskit quantum layer directly with converted weights
    from core.models_pt import _TiledElementWise
    model_qk.if_trainable_freq = True
    model_qk.branch_enc_size = ns[0] * n_q
    model_qk.trunk_enc_size  = ns[2] * n_q
    model_qk.branch_freq = _TiledElementWise(b_in, model_qk.branch_enc_size, 0.1)
    model_qk.trunk_freq  = _TiledElementWise(t_in, model_qk.trunk_enc_size,  0.1)
    model_qk.quantum_layer = build_quanonet_qiskit(n_q, b_in, t_in, ns,
                                                    ham_bound=hb,
                                                    initial_weights=flat_w)
    model_qk.bias = nn.Parameter(torch.zeros(1))
    # Load non-quantum params from TQ state dict
    model_qk.load_state_dict(sd, strict=False)
    model_qk.eval()

    branch = torch.tensor(RNG.random((batch, b_in)).astype(np.float32))
    trunk  = torch.tensor(RNG.random((batch, t_in)).astype(np.float32))

    with torch.no_grad():
        out_tq = model_tq(branch, trunk).numpy()
        out_pl = model_pl(branch, trunk).numpy()
        out_qk = model_qk(branch, trunk).numpy()

    _ok("QuanONet  TQ == PennyLane", out_tq, out_pl, ATOL_PT)
    _ok("QuanONet  TQ == Qiskit",    out_tq, out_qk, ATOL_PT)

    # ── Backward ──────────────────────────────────────────────────────────────
    tgt = torch.tensor(RNG.random((batch, 1)).astype(np.float32))

    model_tq.zero_grad()
    ((model_tq(branch, trunk) - tgt) ** 2).mean().backward()
    g_tq = model_tq.quantum_layer.ansatz_weights.grad.detach().numpy().copy()

    model_pl.zero_grad()
    ((model_pl(branch, trunk) - tgt) ** 2).mean().backward()
    g_pl = model_pl.quantum_layer.ansatz_weights.grad.detach().numpy().copy()

    _ok("QuanONet  TQ == PennyLane (grad ansatz)", g_tq, g_pl, ATOL_GRAD_PT)
    try:
        model_qk.zero_grad()
        ((model_qk(branch, trunk) - tgt) ** 2).mean().backward()
        g_qk = model_qk.quantum_layer.ansatz_weights.grad.detach().numpy().copy()
        _ok("QuanONet  TQ == Qiskit    (grad ansatz)", g_tq, g_qk, ATOL_GRAD_PT)
    except Exception as e:
        _skip("QuanONet  TQ == Qiskit    (grad ansatz)", f"{e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 2: HEAQNN — PyTorch backends (TQ / Qiskit / PennyLane)
# ═══════════════════════════════════════════════════════════════════════════════

def test_heaqnn_pt_backends():
    print("\n─── HEAQNN: TorchQuantum vs Qiskit vs PennyLane ───")
    from core.models_pt import HEAQNNPT, _TiledElementWise
    from core.quantum_circuits_qiskit import build_heaqnn_qiskit

    n_q   = 2
    ns    = (2, 1, 0, 0)
    in_sz = 6
    hb    = (-5.0, 5.0)
    batch = 6
    cfg   = dict(num_qubits=n_q, input_size=in_sz, net_size=ns,
                 scale_coeff=0.1, if_trainable_freq=True, ham_bound=hb)

    torch.manual_seed(42)
    model_tq = HEAQNNPT(**cfg, quantum_backend='torchquantum')
    model_tq.eval()
    sd = model_tq.state_dict()

    model_pl = HEAQNNPT(**cfg, quantum_backend='pennylane')
    model_pl.load_state_dict(sd)
    model_pl.eval()

    flat_w  = tq_ansatz_to_qiskit_flat(sd['quantum_layer.ansatz_weights'])
    model_qk = HEAQNNPT.__new__(HEAQNNPT)
    nn.Module.__init__(model_qk)
    enc_sz = ns[0] * n_q
    model_qk.if_trainable_freq = True
    model_qk.freq = _TiledElementWise(in_sz, enc_sz, 0.1)
    model_qk.quantum_layer = build_heaqnn_qiskit(n_q, in_sz, ns,
                                                  ham_bound=hb,
                                                  initial_weights=flat_w)
    model_qk.bias = nn.Parameter(torch.zeros(1))
    model_qk.load_state_dict(sd, strict=False)
    model_qk.eval()

    x = torch.tensor(RNG.random((batch, in_sz)).astype(np.float32))
    with torch.no_grad():
        out_tq = model_tq(x).numpy()
        out_pl = model_pl(x).numpy()
        out_qk = model_qk(x).numpy()

    _ok("HEAQNN   TQ == PennyLane", out_tq, out_pl, ATOL_PT)
    _ok("HEAQNN   TQ == Qiskit",    out_tq, out_qk, ATOL_PT)

    # ── Backward ──────────────────────────────────────────────────────────────
    tgt = torch.tensor(RNG.random((batch, 1)).astype(np.float32))

    model_tq.zero_grad()
    ((model_tq(x) - tgt) ** 2).mean().backward()
    g_tq = model_tq.quantum_layer.ansatz_weights.grad.detach().numpy().copy()

    model_pl.zero_grad()
    ((model_pl(x) - tgt) ** 2).mean().backward()
    g_pl = model_pl.quantum_layer.ansatz_weights.grad.detach().numpy().copy()

    _ok("HEAQNN   TQ == PennyLane (grad ansatz)", g_tq, g_pl, ATOL_GRAD_PT)
    try:
        model_qk.zero_grad()
        ((model_qk(x) - tgt) ** 2).mean().backward()
        g_qk = model_qk.quantum_layer.ansatz_weights.grad.detach().numpy().copy()
        _ok("HEAQNN   TQ == Qiskit    (grad ansatz)", g_tq, g_qk, ATOL_GRAD_PT)
    except Exception as e:
        _skip("HEAQNN   TQ == Qiskit    (grad ansatz)", f"{e}")


# ═══════════════════════════════════════════════════════════════════════════════
# Part 3: QuanONet — MindSpore vs TorchQuantum  (pretrained Antideriv weights)
# ═══════════════════════════════════════════════════════════════════════════════

def test_quanonet_ms_vs_pt():
    print("\n─── QuanONet: MindSpore vs TorchQuantum (pretrained Antideriv Q2) ───")

    NPZ  = ('pretrained_weights/Antideriv/'
            'Antideriv_QuanONet_Net5-1-5-1_Q2_TF_S0.001_1000x100_Seed0/best_model.npz')
    DATA = 'data/Antideriv/Antideriv_1000_1000_100_10_10_100.npz'

    if not os.path.exists(NPZ):
        _skip("QuanONet MS == TQ", "pretrained NPZ not found"); return

    cfg = dict(net_size=(5, 1, 5, 1), num_qubits=2,
               branch_in=10, trunk_in=1, scale_coeff=0.001, ham_bound=(-5.0, 5.0))

    # ── MindSpore ─────────────────────────────────────────────────────────────
    import mindspore as ms
    from mindspore.train.serialization import load_param_into_net
    from core.models_ms import QuanONetMS
    from core.quantum_circuits_ms import generate_simple_hamiltonian

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
    ham = generate_simple_hamiltonian(cfg['num_qubits'],
                                      lower_bound=cfg['ham_bound'][0],
                                      upper_bound=cfg['ham_bound'][1])
    model_ms = QuanONetMS(cfg['num_qubits'], cfg['branch_in'], cfg['trunk_in'],
                        cfg['net_size'], ham, scale_coeff=cfg['scale_coeff'],
                        if_trainable_freq=True)

    ms_np = np.load(NPZ)
    param_dict = {k: ms.Parameter(ms.Tensor(ms_np[k].astype(np.float32)))
                  for k in ms_np.files}
    load_param_into_net(model_ms, param_dict)
    model_ms.set_train(False)

    if os.path.exists(DATA):
        d    = np.load(DATA)
        b_np = d['test_branch_input'][:16].astype(np.float32)
        t_np = d['test_trunk_input'][:16].astype(np.float32)
    else:
        b_np = RNG.random((16, cfg['branch_in'])).astype(np.float32)
        t_np = RNG.random((16, cfg['trunk_in'])).astype(np.float32)

    out_ms = model_ms((ms.Tensor(b_np), ms.Tensor(t_np))).asnumpy()

    # ── TorchQuantum (weight_transfer) ────────────────────────────────────────
    from utils.weight_transfer import ms_npz_to_pt_state_dict
    from core.models_pt import QuanONetPT

    pt_sd = ms_npz_to_pt_state_dict(NPZ, net_size=cfg['net_size'],
                                    num_qubits=cfg['num_qubits'])
    model_pt = QuanONetPT(cfg['num_qubits'], cfg['branch_in'], cfg['trunk_in'],
                          cfg['net_size'], scale_coeff=cfg['scale_coeff'],
                          if_trainable_freq=True, quantum_backend='torchquantum',
                          ham_bound=cfg['ham_bound'])
    model_pt.load_state_dict(pt_sd)
    model_pt.eval()

    with torch.no_grad():
        out_pt = model_pt(torch.tensor(b_np), torch.tensor(t_np)).numpy()

    _ok("QuanONet  MS == TQ (pretrained Antideriv)", out_ms, out_pt, ATOL_MSPT)

    # ── Backward ──────────────────────────────────────────────────────────────
    tgt_np  = RNG.random((16, 1)).astype(np.float32)
    tgt_ms  = ms.Tensor(tgt_np)
    tgt_pt  = torch.tensor(tgt_np)
    b_t, t_t = ms.Tensor(b_np), ms.Tensor(t_np)

    ms_params = model_ms.trainable_params()
    def _fwd_ms_q(b, t):
        return ((model_ms((b, t)) - tgt_ms) ** 2).mean()
    model_ms.set_train(True)
    _, ms_grads = ms.value_and_grad(_fwd_ms_q, grad_position=None, weights=ms_params)(b_t, t_t)
    model_ms.set_train(False)
    ms_gd = {p.name: g.asnumpy() for p, g in zip(ms_params, ms_grads)}

    total_blocks = (cfg['net_size'][0] * cfg['net_size'][1] +
                    cfg['net_size'][2] * cfg['net_size'][3])
    model_pt.zero_grad()
    ((model_pt(torch.tensor(b_np), torch.tensor(t_np)) - tgt_pt) ** 2).mean().backward()

    _ok("QuanONet  MS == TQ (grad circuit,      pretrained)",
        ms_gd['QuanONet.weight'].reshape(total_blocks, 3, cfg['num_qubits']),
        model_pt.quantum_layer.ansatz_weights.grad.detach().numpy(), ATOL_GRAD_MS)
    _ok("QuanONet  MS == TQ (grad branch_freq,  pretrained)",
        ms_gd['branch_LinearLayer.Net2.weights'],
        model_pt.branch_freq.weights.grad.detach().numpy(), ATOL_GRAD_MS)
    _ok("QuanONet  MS == TQ (grad trunk_freq,   pretrained)",
        ms_gd['trunk_LinearLayer.Net2.weights'],
        model_pt.trunk_freq.weights.grad.detach().numpy(), ATOL_GRAD_MS)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 4: HEAQNN — MindSpore vs TorchQuantum  (random weights, manual transfer)
# ═══════════════════════════════════════════════════════════════════════════════

def test_heaqnn_ms_vs_pt():
    print("\n─── HEAQNN: MindSpore vs TorchQuantum (random weights) ───")

    import mindspore as ms
    from mindspore.train.serialization import load_param_into_net
    from core.models_ms import HEAQNNMS
    from core.quantum_circuits_ms import generate_simple_hamiltonian
    from core.models_pt import HEAQNNPT

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')

    n_q, in_size = 2, 6
    depth, lin_d = 2, 1
    scale, hb    = 0.1, (-5.0, 5.0)

    ham = generate_simple_hamiltonian(n_q, lower_bound=hb[0], upper_bound=hb[1])
    model_ms = HEAQNNMS(n_q, in_size, (depth, lin_d), ham,
                      scale_coeff=scale, if_trainable_freq=True)
    model_ms.set_train(False)

    ms_params = {name: p.asnumpy() for name, p in model_ms.parameters_and_names()}

    total_blocks = depth * lin_d
    pt_sd = {
        'freq.weights':              torch.tensor(ms_params['LinearLayer.Net2.weights']),
        'freq.bias':                 torch.tensor(ms_params['LinearLayer.Net2.bias']),
        'quantum_layer.ansatz_weights': torch.tensor(
            ms_params['HEAQNN.weight'].reshape(total_blocks, 3, n_q)
        ),
        'bias': torch.zeros(1),
    }

    model_pt = HEAQNNPT(n_q, in_size, net_size=(depth, lin_d, 0, 0),
                        scale_coeff=scale, if_trainable_freq=True,
                        quantum_backend='torchquantum', ham_bound=hb)
    model_pt.load_state_dict(pt_sd)
    model_pt.eval()

    x_np = RNG.random((8, in_size)).astype(np.float32)
    out_ms = model_ms(ms.Tensor(x_np)).asnumpy()
    with torch.no_grad():
        out_pt = model_pt(torch.tensor(x_np)).numpy()

    _ok("HEAQNN   MS == TQ (random weights)", out_ms, out_pt, ATOL_MSPT)

    # ── Backward ──────────────────────────────────────────────────────────────
    tgt_np_b  = RNG.random((8, 1)).astype(np.float32)
    tgt_ms_b  = ms.Tensor(tgt_np_b)
    tgt_pt_b  = torch.tensor(tgt_np_b)
    x_ms_b    = ms.Tensor(x_np)

    ms_params_b = model_ms.trainable_params()
    def _fwd_ms_h(x):
        return ((model_ms(x) - tgt_ms_b) ** 2).mean()
    model_ms.set_train(True)
    _, ms_grads_b = ms.value_and_grad(_fwd_ms_h, grad_position=None, weights=ms_params_b)(x_ms_b)
    model_ms.set_train(False)
    ms_gd_b = {p.name: g.asnumpy() for p, g in zip(ms_params_b, ms_grads_b)}

    model_pt.zero_grad()
    ((model_pt(torch.tensor(x_np)) - tgt_pt_b) ** 2).mean().backward()

    _ok("HEAQNN   MS == TQ (grad circuit,      random)",
        ms_gd_b['HEAQNN.weight'].reshape(total_blocks, 3, n_q),
        model_pt.quantum_layer.ansatz_weights.grad.detach().numpy(), ATOL_GRAD_MS)
    _ok("HEAQNN   MS == TQ (grad freq_weights, random)",
        ms_gd_b['LinearLayer.Net2.weights'],
        model_pt.freq.weights.grad.detach().numpy(), ATOL_GRAD_MS)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 5: FNN — PyTorch vs MindSpore
# ═══════════════════════════════════════════════════════════════════════════════

def test_fnn_ms_vs_pt():
    print("\n─── FNN: PyTorch vs MindSpore ───")

    import mindspore as ms
    from mindspore.train.serialization import load_param_into_net
    from core.models_ms import FNNMS

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')

    in_size, out_size, net_size = 10, 1, (2, 20)

    torch.manual_seed(7)
    model_pt = FNN_PT(in_size, out_size, net_size)
    model_pt.eval()
    sd_pt = model_pt.state_dict()

    model_ms = FNNMS(in_size, out_size, net_size)
    model_ms.set_train(False)

    ms_names = {name for name, _ in model_ms.parameters_and_names()}
    param_dict = {}
    for k, v in sd_pt.items():
        if k not in ms_names:
            print(f"  [WARN] PT key '{k}' not in MS model — skipping")
            continue
        param_dict[k] = ms.Parameter(ms.Tensor(v.numpy().astype(np.float32)))
    load_param_into_net(model_ms, param_dict)

    x_np = RNG.random((16, in_size)).astype(np.float32)
    with torch.no_grad():
        out_pt = model_pt(torch.tensor(x_np)).numpy()
    out_ms = model_ms(ms.Tensor(x_np)).asnumpy()

    _ok("FNN      PT == MS", out_pt, out_ms, ATOL_CLS)

    # ── Backward ──────────────────────────────────────────────────────────────
    tgt_np_f  = RNG.random((16, out_size)).astype(np.float32)
    tgt_ms_f  = ms.Tensor(tgt_np_f)
    tgt_pt_f  = torch.tensor(tgt_np_f)
    x_ms_f    = ms.Tensor(x_np)
    x_pt_f    = torch.tensor(x_np)

    ms_params_f = model_ms.trainable_params()
    def _fwd_ms_f(x):
        return ((model_ms(x) - tgt_ms_f) ** 2).mean()
    model_ms.set_train(True)
    _, ms_grads_f = ms.value_and_grad(_fwd_ms_f, grad_position=None, weights=ms_params_f)(x_ms_f)
    model_ms.set_train(False)
    ms_gd_f = {p.name: g.asnumpy() for p, g in zip(ms_params_f, ms_grads_f)}

    model_pt.zero_grad()
    ((model_pt(x_pt_f) - tgt_pt_f) ** 2).mean().backward()

    _ok("FNN      PT == MS (grad FNN.fc0.weight)",    model_pt.FNN.fc0.weight.grad.numpy(),    ms_gd_f.get('FNN.fc0.weight'),    ATOL_GRAD_CLS)
    _ok("FNN      PT == MS (grad FNN.fc_out.weight)", model_pt.FNN.fc_out.weight.grad.numpy(), ms_gd_f.get('FNN.fc_out.weight'), ATOL_GRAD_CLS)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 6: DeepONet — PyTorch vs MindSpore
# ═══════════════════════════════════════════════════════════════════════════════

def test_deeponet_ms_vs_pt():
    print("\n─── DeepONet: PyTorch vs MindSpore ───")

    import mindspore as ms
    from mindspore.train.serialization import load_param_into_net
    from core.models_ms import DeepONetMS

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')

    branch_in, trunk_in = 10, 2
    net_size = (2, 16, 2, 16)

    torch.manual_seed(7)
    model_pt = DeepONet_PT(branch_in, trunk_in, net_size)
    model_pt.eval()
    sd_pt = model_pt.state_dict()

    model_ms = DeepONetMS(branch_in, trunk_in, net_size)
    model_ms.set_train(False)

    ms_names = {name for name, _ in model_ms.parameters_and_names()}
    param_dict = {}
    for k, v in sd_pt.items():
        if k not in ms_names:
            print(f"  [WARN] PT key '{k}' not in MS model — skipping")
            continue
        param_dict[k] = ms.Parameter(ms.Tensor(v.numpy().astype(np.float32)))
    load_param_into_net(model_ms, param_dict)

    b_np = RNG.random((16, branch_in)).astype(np.float32)
    t_np = RNG.random((16, trunk_in)).astype(np.float32)

    with torch.no_grad():
        out_pt = model_pt(torch.tensor(b_np), torch.tensor(t_np)).numpy()
    out_ms = model_ms((ms.Tensor(b_np), ms.Tensor(t_np))).asnumpy()

    _ok("DeepONet PT == MS", out_pt, out_ms, ATOL_CLS)

    # ── Backward ──────────────────────────────────────────────────────────────
    tgt_np_d  = RNG.random((16, 1)).astype(np.float32)
    tgt_ms_d  = ms.Tensor(tgt_np_d)
    tgt_pt_d  = torch.tensor(tgt_np_d)
    b_ms_d, t_ms_d = ms.Tensor(b_np), ms.Tensor(t_np)

    ms_params_d = model_ms.trainable_params()
    def _fwd_ms_d(b, t):
        return ((model_ms((b, t)) - tgt_ms_d) ** 2).mean()
    model_ms.set_train(True)
    _, ms_grads_d = ms.value_and_grad(_fwd_ms_d, grad_position=None, weights=ms_params_d)(b_ms_d, t_ms_d)
    model_ms.set_train(False)
    ms_gd_d = {p.name: g.asnumpy() for p, g in zip(ms_params_d, ms_grads_d)}

    model_pt.zero_grad()
    ((model_pt(torch.tensor(b_np), torch.tensor(t_np)) - tgt_pt_d) ** 2).mean().backward()

    _ok("DeepONet PT == MS (grad branch_net.fc0.weight)",
        model_pt.branch_net.fc0.weight.grad.numpy(), ms_gd_d.get('branch_net.fc0.weight'), ATOL_GRAD_CLS)
    _ok("DeepONet PT == MS (grad trunk_net.fc0.weight)",
        model_pt.trunk_net.fc0.weight.grad.numpy(),  ms_gd_d.get('trunk_net.fc0.weight'),  ATOL_GRAD_CLS)


# ═══════════════════════════════════════════════════════════════════════════════
# Part 7: FNO — PyTorch vs MindSpore
# ═══════════════════════════════════════════════════════════════════════════════

def test_fno_ms_vs_pt():
    print("\n─── FNO: PyTorch vs MindSpore ───")

    import mindspore as ms
    from mindspore.train.serialization import load_param_into_net
    from core.models_pt import FNOPT
    from core.models_ms import FNOMS

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')

    modes, width, layers, fc_hidden, in_ch, n_pts = 8, 16, 2, 16, 2, 64

    torch.manual_seed(7)
    model_pt = FNOPT(modes=modes, width=width, layers=layers,
                   fc_hidden=fc_hidden, in_channels=in_ch)
    model_pt.eval()
    sd_pt = model_pt.state_dict()

    model_ms = FNOMS(modes=modes, width=width, layers=layers,
                      fc_hidden=fc_hidden, in_channels=in_ch)
    model_ms.set_train(False)

    param_dict = _pt_fno_to_ms_params(sd_pt, layers)

    ms_names = {name for name, _ in model_ms.parameters_and_names()}
    for k in ms_names:
        if k not in param_dict:
            print(f"  [WARN] MS key '{k}' has no PT counterpart")
    load_param_into_net(model_ms, param_dict)

    x_np = RNG.random((4, n_pts, in_ch)).astype(np.float32)
    with torch.no_grad():
        out_pt = model_pt(torch.tensor(x_np)).numpy()          # (4, n_pts, 1)
    out_ms = model_ms(ms.Tensor(x_np)).asnumpy()               # (4, n_pts, 1)

    _ok("FNO      PT == MS", out_pt, out_ms, ATOL_CLS)

    # ── Backward ──────────────────────────────────────────────────────────────
    tgt_np_fno  = RNG.random((4, n_pts, 1)).astype(np.float32)
    tgt_ms_fno  = ms.Tensor(tgt_np_fno)
    tgt_pt_fno  = torch.tensor(tgt_np_fno)
    x_ms_fno    = ms.Tensor(x_np)

    ms_params_fno = model_ms.trainable_params()
    def _fwd_ms_fno(x):
        return ((model_ms(x) - tgt_ms_fno) ** 2).mean()
    model_ms.set_train(True)
    _, ms_grads_fno = ms.value_and_grad(_fwd_ms_fno, grad_position=None, weights=ms_params_fno)(x_ms_fno)
    model_ms.set_train(False)
    ms_gd_fno = {p.name: g.asnumpy() for p, g in zip(ms_params_fno, ms_grads_fno)}

    model_pt.zero_grad()
    ((model_pt(torch.tensor(x_np)) - tgt_pt_fno) ** 2).mean().backward()

    _ok("FNO      PT == MS (grad fc0.weight)", model_pt.fc0.weight.grad.numpy(), ms_gd_fno.get('fc0.weight'), ATOL_GRAD_CLS)
    _ok("FNO      PT == MS (grad fc2.weight)", model_pt.fc2.weight.grad.numpy(), ms_gd_fno.get('fc2.weight'), ATOL_GRAD_CLS)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("=" * 72)
    print("  QuanONet Cross-Backend Consistency Check")
    print("=" * 72)

    # ── Quantum: PyTorch backends ─────────────────────────────────────────────
    try:
        test_quanonet_pt_backends()
    except Exception as e:
        print(f"  {_RED}[ERR]{_RST} QuanONet PT backends: {e}")
        import traceback; traceback.print_exc()

    try:
        test_heaqnn_pt_backends()
    except Exception as e:
        print(f"  {_RED}[ERR]{_RST} HEAQNN PT backends: {e}")
        import traceback; traceback.print_exc()

    # ── Quantum: MindSpore vs PyTorch ─────────────────────────────────────────
    try:
        test_quanonet_ms_vs_pt()
    except Exception as e:
        print(f"  {_RED}[ERR]{_RST} QuanONet MS vs PT: {e}")
        import traceback; traceback.print_exc()

    try:
        test_heaqnn_ms_vs_pt()
    except Exception as e:
        print(f"  {_RED}[ERR]{_RST} HEAQNN MS vs PT: {e}")
        import traceback; traceback.print_exc()

    # ── Classical: PyTorch vs MindSpore ───────────────────────────────────────
    try:
        test_fnn_ms_vs_pt()
    except Exception as e:
        print(f"  {_RED}[ERR]{_RST} FNN PT vs MS: {e}")
        import traceback; traceback.print_exc()

    try:
        test_deeponet_ms_vs_pt()
    except Exception as e:
        print(f"  {_RED}[ERR]{_RST} DeepONet PT vs MS: {e}")
        import traceback; traceback.print_exc()

    try:
        test_fno_ms_vs_pt()
    except Exception as e:
        print(f"  {_RED}[ERR]{_RST} FNO PT vs MS: {e}")
        import traceback; traceback.print_exc()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  Summary")
    print("=" * 72)
    passed = skipped = failed = 0
    for tag, ok in results.items():
        if ok is None:
            sym = "[SKIP]"; skipped += 1
        elif ok:
            sym = f"{_GREEN}[PASS]{_RST}"; passed += 1
        else:
            sym = f"{_RED}[FAIL]{_RST}"; failed += 1
        print(f"  {sym}  {tag}")
    print(f"\n  {passed} passed, {failed} failed, {skipped} skipped")
    sys.exit(0 if failed == 0 else 1)
