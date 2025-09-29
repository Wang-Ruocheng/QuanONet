import os
import glob
import mindspore as ms
import hashlib
import re
from collections import defaultdict

ckpt_dir = "melt_quanonet_dim4/checkpoints/Inverse"
pattern = os.path.join(ckpt_dir, "final_Inverse_TF-QuanONet_[*.ckpt")

def get_ckpt_hash(ckpt_path):
    ckpt = ms.load_checkpoint(ckpt_path)
    branch = ckpt["branch_LinearLayer.Net2.weights"].asnumpy().tobytes()
    trunk = ckpt["trunk_LinearLayer.Net2.weights"].asnumpy().tobytes()
    return hashlib.md5(branch + trunk).hexdigest()

grouped = defaultdict(list)

for f in glob.glob(pattern):
    basename = os.path.basename(f)
    m = re.search(r"\[(\d+), 2, (\d+), 2\]_(\d+)\.ckpt", basename)
    if m:
        branch_depth = int(m.group(1))
        trunk_depth = int(m.group(2))
        seed = int(m.group(3))
        h = get_ckpt_hash(f)
        grouped[(branch_depth, trunk_depth)].append((seed, f, h))

for key, ckpt_list in grouped.items():
    hash_to_seeds = defaultdict(list)
    for seed, path, h in ckpt_list:
        hash_to_seeds[h].append(seed)
    for h, seeds in hash_to_seeds.items():
        if len(seeds) > 1:
            print(f"branch_depth={key[0]}, trunk_depth={key[1]} 存在相同权重的seeds: {sorted(seeds)}")
