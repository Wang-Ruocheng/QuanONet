"""
MindSpore .npz → PyTorch state_dict weight transfer for QuanONet.

.. warning::
    **Work In Progress — Do Not Use In Production**

    Weight transfer has been verified to be incorrect: with the same .npz
    weights loaded into QuanONetPT (TorchQuantum), inference outputs differ
    from the MindSpore model by > 0.8 on normalised LWR test data.
    The root cause (gate rotation-angle convention or parameter-ordering
    mismatch between MindQuantum and TorchQuantum) has not yet been
    identified.  Use the MindSpore backend (MSSolver / core/models.py)
    for all production inference until this is resolved.

Parameter mapping (names are correct; numerical equivalence is not):
  MindSpore key                        → PyTorch key
  MindSpore key                        → PyTorch key
  ─────────────────────────────────────────────────────────
  bias                                 → bias
  branch_LinearLayer.Net2.weights      → branch_freq.weights
  branch_LinearLayer.Net2.bias         → branch_freq.bias
  trunk_LinearLayer.Net2.weights       → trunk_freq.weights
  trunk_LinearLayer.Net2.bias          → trunk_freq.bias
  QuanONet.weight  (1800,)             → quantum_layer.ansatz_weights (120,3,5)

Ansatz weight reshape rationale:
  MindSpore stores params in circuit construction order:
    trunk_depth=20 blocks × trunk_linear_depth=2 sublayers = 40 trunk sublayers
    branch_depth=40 blocks × branch_linear_depth=2 sublayers = 80 branch sublayers
    Total = 120 sublayers, each with 3×5=15 params [RY[0..4], RZ[0..4], RY'[0..4]]
  → flat vector (1800,).reshape(120, 3, 5) maps directly to
    TorchQuantum ansatz_weights[block, gate, qubit]
"""
import numpy as np
import torch


_MS_TO_PT = {
    'bias':                              'bias',
    'branch_LinearLayer.Net2.weights':   'branch_freq.weights',
    'branch_LinearLayer.Net2.bias':      'branch_freq.bias',
    'trunk_LinearLayer.Net2.weights':    'trunk_freq.weights',
    'trunk_LinearLayer.Net2.bias':       'trunk_freq.bias',
}


def ms_npz_to_pt_state_dict(npz_path, net_size=(40, 2, 20, 2), num_qubits=5):
    """
    Load a MindSpore .npz checkpoint and return a PyTorch state_dict.

    Args:
        npz_path: path to .npz file produced by MSSolver
        net_size: (branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth)
        num_qubits: number of qubits

    Returns:
        state_dict: dict[str, torch.Tensor] ready for model.load_state_dict()
    """
    branch_depth, branch_ld, trunk_depth, trunk_ld = net_size
    total_ansatz_blocks = branch_depth * branch_ld + trunk_depth * trunk_ld

    ms = np.load(npz_path)
    state_dict = {}

    # ── Simple key renames ────────────────────────────────────────────────────
    for ms_key, pt_key in _MS_TO_PT.items():
        if ms_key not in ms:
            raise KeyError(f"Expected key '{ms_key}' not found in {npz_path}. "
                           f"Available: {list(ms.files)}")
        val = ms[ms_key].astype(np.float32)
        # MindSpore bias is scalar (); PyTorch bias is shape (1,)
        if pt_key == 'bias':
            val = val.reshape(1)
        state_dict[pt_key] = torch.tensor(val)

    # ── Ansatz weights: (1800,) → (total_blocks, 3, num_qubits) ─────────────
    raw = ms['QuanONet.weight'].astype(np.float32)
    expected = total_ansatz_blocks * 3 * num_qubits
    if raw.size != expected:
        raise ValueError(
            f"QuanONet.weight has {raw.size} elements but expected "
            f"{expected} ({total_ansatz_blocks}×3×{num_qubits}). "
            "Check net_size and num_qubits."
        )
    state_dict['quantum_layer.ansatz_weights'] = torch.tensor(
        raw.reshape(total_ansatz_blocks, 3, num_qubits)
    )

    return state_dict


def load_quanonet_pt(npz_path, quantum_backend='torchquantum',
                     branch_input_size=100, trunk_input_size=2,
                     net_size=(40, 2, 20, 2), num_qubits=5,
                     scale_coeff=0.1, ham_bound=(-5.0, 5.0)):
    """
    Convenience: build QuanONetPT, transfer weights, set eval mode.

    Args:
        npz_path: path to MindSpore best_model.npz
        quantum_backend: 'torchquantum' or 'qiskit'
        branch_input_size: branch input features (100 for LWR)
        trunk_input_size: trunk input features (2 for LWR: x, t)
        net_size: (branch_depth, branch_linear_depth, trunk_depth, trunk_linear_depth)
        num_qubits: number of qubits
        scale_coeff: must match training config
        ham_bound: must match training config

    Returns:
        model: QuanONetPT ready for inference
    """
    from core.models_pt import QuanONetPT

    model = QuanONetPT(
        num_qubits=num_qubits,
        branch_input_size=branch_input_size,
        trunk_input_size=trunk_input_size,
        net_size=net_size,
        scale_coeff=scale_coeff,
        if_trainable_freq=True,
        quantum_backend=quantum_backend,
        ham_bound=ham_bound,
    )
    state_dict = ms_npz_to_pt_state_dict(npz_path, net_size=net_size, num_qubits=num_qubits)
    model.load_state_dict(state_dict)
    model.eval()
    return model


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    NPZ = 'pretrained_weights/LWR_QuanONet_Net40-2-20-2_Q5_TF_S0.1_1000x100_Seed0_best_model.npz'
    DATA = 'data/LWR/LWR_1000_1000_100_100_100_1000.npz'

    print("Loading PyTorch model from MindSpore weights...")
    model_pt = load_quanonet_pt(NPZ, quantum_backend='torchquantum')
    n_params = sum(p.numel() for p in model_pt.parameters())
    print(f"PyTorch model: {n_params} parameters")

    # Sample 8 test points
    data = np.load(DATA)
    b = torch.tensor(data['test_branch_input'][:8].astype(np.float32))
    t = torch.tensor(data['test_trunk_input'][:8].astype(np.float32))
    y = data['test_output'][:8, 0]

    with torch.no_grad():
        pred_pt = model_pt(b, t).numpy()[:, 0]

    print(f"PT predictions:  {pred_pt[:4].round(4)}")
    print(f"True values:     {y[:4].round(4)}")
    print(f"Max abs error:   {np.abs(pred_pt - y).max():.6f}")
    print("Weight transfer OK" if np.abs(pred_pt - y).max() < 0.3 else "Check failed")
