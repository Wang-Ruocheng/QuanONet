import os
import glob
import re
from collections import defaultdict

ckpt_dir = "melt_quanonet_dim4/checkpoints/Inverse"
pattern = os.path.join(ckpt_dir, "final_Inverse_TF-QuanONet_[*.ckpt")

seeds_dict = defaultdict(list)

for f in glob.glob(pattern):
    basename = os.path.basename(f)
    m = re.search(r"\[(\d+), 2, (\d+), 2\]_(\d+)\.ckpt", basename)
    if m:
        branch_depth = int(m.group(1))
        trunk_depth = int(m.group(2))
        seed = int(m.group(3))
        seeds_dict[(branch_depth, trunk_depth)].append(seed)

for (branch_depth, trunk_depth), seeds in sorted(seeds_dict.items()):
    seeds.sort()
    print(f"branch_depth={branch_depth}, trunk_depth={trunk_depth} 已有 seeds: {seeds}")
