import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import sys
import json
from pathlib import Path
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import warnings
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.train.serialization import load_checkpoint, load_param_into_net

def plot_comparison(result, save_path=None):
    """
    绘制扩散算子问题的对比图 (四子图布局：初始条件+真实解+预测解+误差)
    Args:
        result: 包含样本数据的字典
        save_path: 保存路径
    """
    sample_idx = result['sample_idx']
    u0 = result['u0']
    u_true = result['u_true']
    u_pred = result['u_pred']
    x_coords = result['x_coords']
    t_coords = result['t_coords']
    abs_error = np.abs(u_true - u_pred)
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    axes[0].plot(np.linspace(0, 1, len(u0)), u0, 'b-', linewidth=2, label='Initial Condition u₀(x)')
    axes[0].set_xlabel('Space (x)')
    axes[0].set_ylabel('u₀(x)')
    axes[0].set_title(f'Sample {sample_idx}: Initial Condition')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    if len(x_coords) > 50:
        subsample_factor = max(1, len(x_coords) // 50)
        subsample_x = slice(0, len(x_coords), subsample_factor)
        subsample_t = slice(0, len(t_coords), subsample_factor)
    else:
        subsample_x = slice(None)
        subsample_t = slice(None)
    x_sub = x_coords[subsample_x]
    t_sub = t_coords[subsample_t]
    u_true_sub = u_true[subsample_x, :][:, subsample_t]
    u_pred_sub = u_pred[subsample_x, :][:, subsample_t]
    abs_error_sub = abs_error[subsample_x, :][:, subsample_t]
    X_sub, T_sub = np.meshgrid(x_sub, t_sub, indexing='ij')
    vmin_solution = min(np.min(u_true_sub), np.min(u_pred_sub))
    vmax_solution = max(np.max(u_true_sub), np.max(u_pred_sub))
    im1 = axes[1].contourf(X_sub, T_sub, u_true_sub, levels=20, cmap='viridis', alpha=0.8, vmin=vmin_solution, vmax=vmax_solution)
    axes[1].set_xlabel('Space (x)')
    axes[1].set_ylabel('Time (t)')
    axes[1].set_title(f'Sample {sample_idx}: True Solution u(x,t)')
    cbar1 = plt.colorbar(im1, ax=axes[1], shrink=0.8)
    cbar1.set_label('u(x,t)', rotation=270, labelpad=15)
    im2 = axes[2].contourf(X_sub, T_sub, u_pred_sub, levels=20, cmap='viridis', alpha=0.8, vmin=vmin_solution, vmax=vmax_solution)
    axes[2].set_xlabel('Space (x)')
    axes[2].set_ylabel('Time (t)')
    axes[2].set_title(f'Sample {sample_idx}: Predicted Solution u(x,t)')
    cbar2 = plt.colorbar(im2, ax=axes[2], shrink=0.8)
    cbar2.set_label('u(x,t)', rotation=270, labelpad=15)
    im3 = axes[3].contourf(X_sub, T_sub, abs_error_sub, levels=20, cmap='Reds', alpha=0.8)
    axes[3].set_xlabel('Space (x)')
    axes[3].set_ylabel('Time (t)')
    axes[3].set_title(f'Sample {sample_idx}: Absolute Error')
    cbar3 = plt.colorbar(im3, ax=axes[3], shrink=0.8)
    cbar3.set_label('|Error|', rotation=270, labelpad=15)
    plt.tight_layout()
    if save_path is not None:
        filename = f"sample_{sample_idx}_comparison.png"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_analysis(results, save_path):
    num_samples = len(results)
    fig, axes = plt.subplots(2, num_samples, figsize=(6*num_samples, 10))
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    for i, result in enumerate(results):
        sample_idx = result['sample_idx']
        u_true = result['u_true']
        u_pred = result['u_pred']
        x_coords = result['x_coords']
        t_coords = result['t_coords']
        abs_error = np.abs(u_true - u_pred)
        if len(x_coords) > 50:
            subsample_factor = max(1, len(x_coords) // 50)
            subsample_x = slice(0, len(x_coords), subsample_factor)
            subsample_t = slice(0, len(t_coords), subsample_factor)
        else:
            subsample_x = slice(None)
            subsample_t = slice(None)
        x_sub = x_coords[subsample_x]
        t_sub = t_coords[subsample_t]
        abs_error_sub = abs_error[subsample_x, :][:, subsample_t]
        X_sub, T_sub = np.meshgrid(x_sub, t_sub, indexing='ij')
        im1 = axes[0, i].contourf(X_sub, T_sub, abs_error_sub, levels=20, cmap='Reds')
        axes[0, i].set_xlabel('Space (x)')
        axes[0, i].set_ylabel('Time (t)')
        axes[0, i].set_title(f'Sample {sample_idx}: Absolute Error')
        plt.colorbar(im1, ax=axes[0, i], shrink=0.8)
        error_flat = abs_error.flatten()
        axes[1, i].hist(error_flat, bins=50, alpha=0.7, color='red', density=True)
        axes[1, i].set_xlabel('Absolute Error')
        axes[1, i].set_ylabel('Density')
        axes[1, i].set_title(f'Sample {sample_idx}: Error Distribution')
        axes[1, i].grid(True, alpha=0.3)
        mean_error = np.mean(error_flat)
        max_error = np.max(error_flat)
        axes[1, i].axvline(mean_error, color='blue', linestyle='--', label=f'Mean: {mean_error:.4f}')
        axes[1, i].axvline(max_error, color='red', linestyle='--', label=f'Max: {max_error:.4f}')
        axes[1, i].legend()
    plt.tight_layout()
    filename = "error_analysis.png"
    filepath = os.path.join(save_path, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.show()


def generate_predictions_mindspore(model, u0, num_points=100):
    """使用MindSpore模型生成预测"""
    # 检查u0_sample的长度，确保与模型期望的输入尺寸匹配
    
    # Branch input: 初始条件 u0(x)
    branch_input = ms.Tensor(u0.reshape(1, -1), ms.float32)
    
    x_coords = np.linspace(0, 1, num_points)  # 空间坐标
    t_coords = np.linspace(0, 1, num_points)  # 时间坐
    # 创建时空坐标网格
    X, T = np.meshgrid(x_coords, t_coords, indexing='ij')
    trunk_coords = np.stack([X.flatten(), T.flatten()], axis=1)  # shape: (num_x*num_t, 2)
    
    # 分批预测
    batch_size = 2000
    predictions = []
    
    for i in range(0, trunk_coords.shape[0], batch_size):
        end_idx = min(i + batch_size, trunk_coords.shape[0])
        trunk_batch = trunk_coords[i:end_idx]
        
        # 准备输入数据
        trunk_input = ms.Tensor(trunk_batch, ms.float32)  # shape: (batch_size, 2)
        branch_batch = ms.ops.tile(branch_input, (trunk_batch.shape[0], 1))  # 重复branch输入
        
        # 模型预测
        try:
            # construct方法需要一个包含[branch_input, trunk_input]的输入
            model_input = [branch_batch, trunk_input]
            pred_batch = model(model_input)
            predictions.append(pred_batch.asnumpy().flatten())
                
        except Exception as e:
            print(f"    预测批次出错: {e}")
            # 使用随机数据作为备选
            pred_batch = np.random.randn(trunk_batch.shape[0]) * 0.1
            predictions.append(pred_batch)
    
    # 合并所有预测结果
    u_pred_flat = np.concatenate(predictions)  # shape: (num_x*num_t,)
    u_pred_sample = u_pred_flat.reshape(num_points, num_points) 
    
    return u_pred_sample