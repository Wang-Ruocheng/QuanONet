import os
import glob
import subprocess
import re

ckpt_dir = "/mnt/nas-new/home/yange/wangruocheng/QON_wrc/melt_quanonet_dim4/checkpoints/Inverse"
output_prefix = "melt_quanonet_dim4_expdamp"
log_dir = f"/mnt/nas-new/home/yange/wangruocheng/QON_wrc/{output_prefix}/dairy"
os.makedirs(log_dir, exist_ok=True)

ckpt_files = glob.glob(os.path.join(ckpt_dir, "final_Inverse_TF-QuanONet_[*.ckpt"))

for ckpt in ckpt_files:
    basename = os.path.basename(ckpt)
    m = re.search(r"\[(\d+), 2, (\d+), 2\]_(\d+)", basename)
    if not m:
        print(f"跳过未匹配的文件: {basename}")
        continue
    branch_depth = m.group(1)
    trunk_depth = m.group(2)
    seed = m.group(3)
    log_file = os.path.join(log_dir, f"train_QuanONet_netsize_{branch_depth}_2_{trunk_depth}_2_seed_{seed}.log")
    if os.path.exists(log_file):
        print(f"已存在日志文件，跳过: {log_file}")
        continue
    cmd = [
        "nohup", "python", "-u", "train_ODE.py",
        "--operator", "Inverse",
        "--model_type", "QuanONet",
        "--scale_coeff", "0.001",
        "--if_trainable_freq", "true",
        "--num_qubits", "2",
        "--net_size", branch_depth, "2", trunk_depth, "2",
        "--if_train", "true",
        "--if_keep", "true",
        "--if_save", "true",
        "--if_adjust_lr", "true",
        "--init_checkpoint", ckpt,
        "--random_seed", seed,
        "--prefix", output_prefix,
        "--num_epochs", "1000"
    ]
    print(f"运行命令: {' '.join(cmd)}")
    with open(log_file, "w") as f:
        process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        process.wait()
    print(f"完成: {log_file}")
