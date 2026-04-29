import sys, os, numpy as np, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(__file__))

CKPT = 'pretrained_weights/RDiffusion/RDiffusion_QuanONet_Net40-2-20-2_Q5_TF_S0.1_1000x100_Seed0/best_model.ckpt'
DATA = 'data/RDiffusion/RDiffusion_100_100_100_100_100_100.npz'
N    = 8

data   = np.load(DATA)
branch = data['test_branch_input'][:N].astype(np.float32)
trunk  = data['test_trunk_input'][:N].astype(np.float32)
y_true = data['test_output'][:N]

# ── 1. MindSpore ──────────────────────────────────────────────────────────────
print("=== 1. MindSpore 推理 ===")
import mindspore as ms
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from core.models import QuanONet
from core.quantum_circuits import generate_simple_hamiltonian

ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')
ham    = generate_simple_hamiltonian(5, lower_bound=-5.0, upper_bound=5.0)
net_ms = QuanONet(5, 100, 2, (40,2,20,2), ham, 0.1, if_trainable_freq=True)
load_param_into_net(net_ms, load_checkpoint(CKPT))
net_ms.set_train(False)

pred_ms = net_ms((ms.Tensor(branch), ms.Tensor(trunk))).asnumpy()
rel_ms  = np.linalg.norm(pred_ms - y_true) / (np.linalg.norm(y_true) + 1e-8)
print("preds :", pred_ms[:,0].round(4).tolist())
print("truth :", y_true[:,0].round(4).tolist())
print(f"Rel-L2: {rel_ms:.4f}  ({rel_ms:.2%})")

# 导出参数
npz_buf = {k: v.asnumpy() for k, v in load_checkpoint(CKPT).items()}
np.savez('/tmp/rdiff_w.npz', **npz_buf)

# ── 2. PyTorch / TorchQuantum ────────────────────────────────────────────────
print()
print("=== 2. PyTorch (TorchQuantum) 推理 ===")
import torch
from utils.weight_transfer import ms_npz_to_pt_state_dict
from core.models_pt import QuanONetPT

sd = ms_npz_to_pt_state_dict('/tmp/rdiff_w.npz', net_size=(40,2,20,2), num_qubits=5)
net_pt = QuanONetPT(5, 100, 2, (40,2,20,2), scale_coeff=0.1, if_trainable_freq=True,
                    quantum_backend='torchquantum', ham_bound=(-5.0, 5.0))
net_pt.load_state_dict(sd)
net_pt.eval()

with torch.no_grad():
    pred_pt = net_pt(torch.tensor(branch), torch.tensor(trunk)).numpy()

rel_pt = np.linalg.norm(pred_pt - y_true) / (np.linalg.norm(y_true) + 1e-8)
print("preds :", pred_pt[:,0].round(4).tolist())
print(f"Rel-L2: {rel_pt:.4f}  ({rel_pt:.2%})")

# ── 3. 对比 ───────────────────────────────────────────────────────────────────
print()
print("=== 3. 对比 ===")
diff = np.abs(pred_ms - pred_pt)
print("逐样本 |MS - PT|:", diff[:,0].round(4).tolist())
print(f"最大绝对误差 : {diff.max():.6f}")
print(f"平均绝对误差 : {diff.mean():.6f}")

bw_ms = npz_buf['branch_LinearLayer.Net2.weights']
bw_pt = sd['branch_freq.weights'].numpy()
print(f"\n频率层权重 max_diff : {np.abs(bw_ms - bw_pt).max():.6f}")
print(f"bias         diff   : {abs(float(npz_buf['bias']) - float(sd['bias'].numpy())):.6f}")
