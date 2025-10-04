import os
import shutil

src_dir = "melt_quanonet_dim4/checkpoints/Inverse"
dst_dir = "melt_quanonet_dim4/checkpoints/Inverse/Inverse_TF-QuanONet_[200, 2, 20, 2]_0"

# 创建目标文件夹（如果不存在）
os.makedirs(dst_dir, exist_ok=True)

# 遍历源目录下所有文件（不包括子目录）
for filename in os.listdir(src_dir):
    src_file = os.path.join(src_dir, filename)
    dst_file = os.path.join(dst_dir, filename)
    if os.path.isfile(src_file):
        shutil.move(src_file, dst_file)
