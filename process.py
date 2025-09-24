import os
import glob
import subprocess
import re
import shutil

ckpt_root = "/mnt/nas-new/home/yange/wangruocheng/QON_wrc/melt_quanonet_dim4/checkpoints/Inverse"
log_root = "/mnt/nas-new/home/yange/wangruocheng/QON_wrc/ongoing_wrc"
py_script = "/mnt/nas-new/home/yange/wangruocheng/QON_wrc/train_ODE.py"

for folder in glob.glob(f"{ckpt_root}/*_*"):
    base = os.path.basename(folder)
    m = re.match(r"^(\d+)_(\d+)$", base)
    if not m:
        continue  # 跳过不符合格式的文件夹
    branch_depth, trunk_depth = map(int, m.groups())
    ckpt_files = glob.glob(os.path.join(folder, "Inverse_TF-QuanONet_final_*.ckpt"))
    for random_seed in range(5):
        print(f"Processing: branch_depth={branch_depth}, trunk_depth={trunk_depth}, seed={random_seed}, folder={folder}")
        log_file = os.path.join(
            log_root,
            f"train_QuanONet_netsize_{branch_depth}_2_{trunk_depth}_2_seed_{random_seed}.log"
        )
        if not os.path.exists(log_file):
            continue
        # 读取目标log的评估结果
        with open(log_file) as f:
            log_content = f.read()
        match = re.search(
            r"Evaluation results:\s*MSE:\s*([0-9.eE+-]+)\s*MAE:\s*([0-9.eE+-]+)\s*Max_Error:\s*([0-9.eE+-]+)",
            log_content
        )
        if not match:
            continue
        target_mse, target_mae, target_maxerr = match.groups()
        found = False
        print(f" Target MSE: {target_mse}, MAE: {target_mae}, Max_Error: {target_maxerr}")
        for ckpt in ckpt_files:
            print(f"  Checking checkpoint: {ckpt}")
            cmd = [
                "python", py_script,
                "--operator", "Inverse",
                "--model_type", "QuanONet",
                "--scale_coeff", "0.001",
                "--if_trainable_freq", "true",
                "--num_qubits", "2",
                "--net_size", str(branch_depth), "2", str(trunk_depth), "2",
                "--if_train", "false",
                "--if_keep", "true",
                "--if_save", "false",
                "--init_checkpoint", ckpt,
                "--random_seed", str(random_seed)
            ]
            # 运行模型评估
            result = subprocess.run(cmd, capture_output=True, text=True)
            output = result.stdout + result.stderr
            eval_match = re.search(
                r"MSE:\s*([0-9.eE+-]+)\s*MAE:\s*([0-9.eE+-]+)\s*Max[_ ]Error:\s*([0-9.eE+-]+)",
                output
            )
            print(f"   Eval output: {output.strip()}")
            if eval_match:
                mse, mae, maxerr = eval_match.groups()
                # 比较结果
                if (abs(float(mse) - float(target_mse)) < 1e-6 and
                    abs(float(mae) - float(target_mae)) < 1e-6 and
                    abs(float(maxerr) - float(target_maxerr)) < 1e-6):
                    # 重命名ckpt
                    new_ckpt = os.path.join(
                        ckpt_root,
                        f"final_Inverse_QuanONet_[{branch_depth}, 2, {trunk_depth}, 2]_{random_seed}.ckpt"
                    )
                    shutil.move(ckpt, new_ckpt)
                    print(f"✅ {ckpt} -> {new_ckpt}")
                    found = True
                    break
        if found:
            break  # 该seed已找到，跳过同目录下其他ckpt