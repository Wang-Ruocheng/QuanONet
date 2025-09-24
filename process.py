import os
import glob
import re

ckpt_dir = "/mnt/nas-new/home/yange/wangruocheng/QON_wrc/melt_quanonet_dim4/checkpoints/Inverse"
patterns = [
    os.path.join(ckpt_dir, "final_Inverse_QuanONet_[*.ckpt"),
    os.path.join(ckpt_dir, "best_Inverse_QuanONet_[*.ckpt"),
]

for pattern in patterns:
    for filepath in glob.glob(pattern):
        dirname, basename = os.path.split(filepath)
        # 替换QuanONet为TF-QuanONet
        new_basename = re.sub(r'QuanONet', 'TF-QuanONet', basename)
        new_filepath = os.path.join(dirname, new_basename)
        if filepath != new_filepath:
            os.rename(filepath, new_filepath)
            print(f"✅ {basename} -> {new_basename}")