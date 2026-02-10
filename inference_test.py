import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# 确保能导入 core
sys.path.append(os.getcwd())

# ==========================================
# 1. 修正导入路径
# ==========================================
try:
    # 尝试导入 FNO，如果名字不对，请根据 grep 结果修改这里
    from core.dde_models import FNO 
except ImportError:
    try:
        from core.dde_models import FNO1d as FNO
    except ImportError:
        print("❌ 错误：无法在 core/dde_models.py 中找到 FNO 或 FNO1d 类。")
        print("请运行 'grep class core/dde_models.py' 查看正确的类名，并修改脚本第15行。")
        sys.exit(1)

def run_inference():
    # ---------------------------------------------------------
    # 2. 配置参数 (对应 --net_size 16 64 4 32)
    # ---------------------------------------------------------
    # 你的 net_size 是: [modes, width, depth, something_else]
    # 通常 FNO 只需要前三个。第4个参数(32)可能是 padding 或 fc_dim
    modes = 15
    width = 14
    depth = 3
    
    # 这一步很关键：DeepXDE 的 FNO 构造函数参数可能不一致
    # 我们先定义好参数，下面自动尝试
    
    checkpoint_path = "./checkpoints/Inverse/best_FNO_100x100_None.ckpt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Loading weights from {checkpoint_path}...")

    # ---------------------------------------------------------
    # 3. 初始化模型 (自动适配参数)
    # ---------------------------------------------------------
    try:
        # 优先尝试传入 4 个参数 (modes, width, depth, fc_dim/padding)
        print(f"Attempting to initialize FNO with extra dim: {32}...")
        model = FNO(modes, width, depth, 32)
        print("✅ Success: Model initialized with (modes, width, depth, 32)")
    except TypeError:
        try:
            # 如果上面失败，尝试把 32 当作 padding 关键字参数
            model = FNO(modes, width, depth, padding=32)
            print("✅ Success: Model initialized with (modes, width, depth, padding=32)")
        except TypeError:
            print("⚠️ Warning: Could not pass 32. Fallback to default (this may fail load_state_dict).")
            model = FNO(modes, width, depth)

    # 加载权重
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(device)
    model.eval()

    # ---------------------------------------------------------
    # 4. 构造输入数据 u(x)=x
    # ---------------------------------------------------------
    num_points = 100
    # x 从 0 到 1
    x = torch.linspace(0, 1, num_points).to(device)
    
    # 通道 1: u(x) = x
    # Shape: (Batch=1, Grid=100, Channel=1)
    u_channel = x.clone().reshape(1, num_points, 1)
    
    # 通道 2: grid = x
    # Shape: (Batch=1, Grid=100, Channel=1)
    grid_channel = x.clone().reshape(1, num_points, 1)
    
    # 拼接！形成 2 个通道的输入 (u, x)
    # Shape: (1, 100, 2)
    input_tensor = torch.cat([u_channel, grid_channel], dim=-1)

    print(f"Input shape: {input_tensor.shape} (Should be [1, 100, 2])")

    # ---------------------------------------------------------
    # 5. 推理 (传入 input_tensor 而不是 u_channel)
    # ---------------------------------------------------------
    with torch.no_grad():
        pred_y = model(input_tensor)

    # ---------------------------------------------------------
    # 5. 推理
    # ---------------------------------------------------------
    with torch.no_grad():
        pred_y = model(input_tensor)

    # 转为 numpy
    x_np = x.cpu().numpy()
    pred_y_np = pred_y.squeeze().cpu().numpy()
    
    # ---------------------------------------------------------
    # 6. 真解 y = x^2 / 2
    # ---------------------------------------------------------
    true_y_np = 0.5 * (x_np ** 2)

    # ---------------------------------------------------------
    # 7. 绘图
    # ---------------------------------------------------------
    mse = np.mean((pred_y_np - true_y_np)**2)
    print(f"Inference MSE on 'y=x': {mse:.2e}")

    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_np, x.cpu().numpy(), 'g--', label='Input: u=x', alpha=0.3)
    plt.plot(x_np, true_y_np, 'k-', linewidth=2, label='Ground Truth: x^2/2')
    plt.plot(x_np, pred_y_np, 'r--', linewidth=2, label='Prediction')
    plt.title(f"Integration Operator (MSE: {mse:.2e})")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(x_np, np.abs(pred_y_np - true_y_np), 'b.-')
    plt.title("Absolute Error")
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    save_path = "inference_result_y_eq_x.png"
    plt.savefig(save_path, dpi=150)
    print(f"\n✅ Result saved to {save_path}")

if __name__ == "__main__":
    run_inference()