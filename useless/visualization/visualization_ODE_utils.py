from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# MindSpore related imports
import mindspore as ms
import mindspore.nn as nn
import mindspore.numpy as mnp
from mindspore.train.serialization import load_checkpoint, load_param_into_net

# Local module imports - 修复导入路径
from core.models import QuanONet, HEAQNN
from core.quantum_circuits import generate_simple_hamiltonian
from data_utils.data_generation import *
from data_utils.data_processing import ODE_encode

# Configuration parameters
def load_latest_config(operator_type):
    """Load the latest configuration file"""
    config_files = list(Path("data").glob(f"{operator_type}_Operator_config_*.json"))
    if not config_files:
        print("❌ 未找到配置文件!")
        return None
    
    latest_config = max(config_files, key=os.path.getctime)
    with open(latest_config, 'r') as f:
        config = json.load(f)
    
    print(f"✅ 已加载配置文件: {latest_config}")
    return config

def load_data(config):
    # Build data file path
    data_file = f"data/{config['operator_type']}_Operator_dataset_{config['num_train']}_{config['num_test']}_{config['num_points']}_{config['train_sample_num']}_{config['test_sample_num']}.npz"
    
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在，正在生成: {data_file}")
        
        # First generate the raw data to get original functions
        train_u0, train_u, test_u0, test_u, x = generate_ODE_Operator_data(config['operator_type'], config['num_train'], config['num_test'], config['num_points'], config['length_scale'], config['num_cal'])

        # Convert to numpy for saving
        test_u0_np = test_u0.asnumpy()  # Original function u0(x) (input)
        test_u_np = test_u.asnumpy()    # Solution function u(x) satisfying du/dx = u - u0²(x) (output/target)
        x_points = x.asnumpy()
        func_operator_type = f"generate_{config['operator_type']}_Operator_data"
        generate_Operator_data = globals()[func_operator_type]  # Get the function dynamically
        # Use ODE encoding for network training
        train_branch_input, train_trunk_input, train_output, \
        test_branch_input, test_trunk_input, test_output = ODE_encode(
            generate_Operator_data,
            config['num_train'], config['num_test'], config['num_points'],
            config['train_sample_num'], config['test_sample_num']
        )
        
        # Save data including original functions
        np.savez_compressed(
            data_file,
            train_branch_input=train_branch_input.asnumpy(),
            train_trunk_input=train_trunk_input.asnumpy(),
            train_output=train_output.asnumpy(),
            test_branch_input=test_branch_input.asnumpy(),
            test_trunk_input=test_trunk_input.asnumpy(),
            test_output=test_output.asnumpy(),
            x_points=x_points
        )
        print(f"✅ 数据已生成并保存")
    
    # Load data
    data = np.load(data_file, allow_pickle=True)
    
    return {
        'test_branch_input': ms.Tensor(data['test_branch_input'], ms.float32),
        'test_trunk_input': ms.Tensor(data['test_trunk_input'], ms.float32), 
        'test_output': ms.Tensor(data['test_output'], ms.float32),
        'x_points': data['x_points'] if 'x_points' in data else None
    }

def load_trained_model(config, data):
    """Load the trained model"""
    # Find the best model file
    best_model_path = f"checkpoints/best_{config['operator_type']}_{config['model_type']}.ckpt"
    
    if not os.path.exists(best_model_path):
        print(f"❌ 模型文件不存在: {best_model_path}")
        return None
    
    # Create model
    ham = generate_simple_hamiltonian(config['num_qubits'])
    branch_input_size = data['test_branch_input'].shape[1]
    trunk_input_size = data['test_trunk_input'].shape[1]
    
    if config['model_type'] == 'QuanONet':
        model = QuanONet(
            num_qubits=config['num_qubits'],
            branch_input_size=branch_input_size,
            trunk_input_size=trunk_input_size,
            net_size=tuple(config['net_size']),
            ham=ham,
            scale_coeff=config['scale_coeff'],
            if_trainable_freq=config.get('if_trainable_freq', False)
        )
    elif config['model_type'] == 'HEAQNN':
        model = QuanONet(
            num_qubits=config['num_qubits'],
            branch_input_size=branch_input_size,
            trunk_input_size=trunk_input_size,
            net_size=tuple(config['net_size']),
            ham=ham,
            scale_coeff=config['scale_coeff'],
            if_trainable_freq=config.get('if_trainable_freq', False)
        )
    else:
        raise ValueError(f"Unknown model: {config['model_type']}")

    # Load model weights
    param_dict = load_checkpoint(best_model_path)
    load_param_into_net(model, param_dict)
    model.set_train(False)  # Set to evaluation mode
    
    print(f"✅ 模型加载成功: {config['model_type']}")
    
    return model

# 工具函数：求解 ODE 系统
def solve_ode_system(u0_func, x_points, ode_system, initial_condition=0):
    """
    通用 ODE 求解工具函数。
    参数：
        u0_func: 可为函数或 numpy 数组
        x_points: 求解区间的采样点
        initial_condition: 初始条件
        ode_system: ODE 系统（需外部传入）
    """
    # 如果 u0_func 是数组，创建插值函数
    if isinstance(u0_func, np.ndarray):
        u0_interp = interp1d(x_points, u0_func, kind='cubic', fill_value='extrapolate')
    else:
        u0_interp = u0_func

    # 用 lambda 包装，确保 ode_func 只接收 (t, y) 两个参数
    ode_func = ode_system(u0_interp)
    ode_func_wrapped = lambda t, y: ode_func(t, y)

    # 求解微分方程
    solution = solve_ivp(
        ode_func_wrapped,
        [x_points[0], x_points[-1]],
        [initial_condition],
        t_eval=x_points,
        method='RK45',
        rtol=1e-8,
        atol=1e-10
    )
    return solution.y[0]

def generate_predictions(model, data, batch_size=100):
    """Generate model predictions"""
    if model is None:
        print("❌ 模型未加载!")
        return None
    
    test_input = (data['test_branch_input'], data['test_trunk_input'])
    test_output = data['test_output']
    
    # Batch prediction to avoid memory issues
    test_size = test_input[0].shape[0]
    predictions = []
    
    print(f"🔮 开始预测 {test_size} 个测试样本...")
    
    for start in range(0, test_size, batch_size):
        end = min(start + batch_size, test_size)
        
        batch_branch = test_input[0][start:end]
        batch_trunk = test_input[1][start:end]
        batch_input = (batch_branch, batch_trunk)
        
        # Model prediction
        batch_pred = model(batch_input)
        predictions.append(batch_pred.asnumpy())
    
    # Merge all prediction results
    predictions = np.vstack(predictions)
    true_values = test_output.asnumpy()
    
    # Calculate error metrics
    mse = np.mean((predictions - true_values) ** 2)
    mae = np.mean(np.abs(predictions - true_values))
    max_error = np.max(np.abs(predictions - true_values))
    rel_error = np.mean(np.abs(predictions - true_values) / (np.abs(true_values) + 1e-8))
    
    print(f"✅ 预测完成!")
    print(f"📈 评估指标 - MSE: {mse:.6f}, MAE: {mae:.6f}")
    
    return {
        'predictions': predictions,
        'true_values': true_values,
        'mse': mse,
        'mae': mae,
        'max_error': max_error,
        'rel_error': rel_error
    }

def plot_single_sample_comparison(sample_idx, data, results, config, save_path=None):
    """Plot comparison for a single sample with horizontal layout: original function, true vs predicted, error"""
    
    # Get prediction data
    samples_per_func = config['test_sample_num']
    start_idx = sample_idx * samples_per_func
    end_idx = start_idx + samples_per_func
    
    pred_u = results['predictions'][start_idx:end_idx].flatten()
    true_u = results['true_values'][start_idx:end_idx].flatten()
    pred_x = data['test_trunk_input'].asnumpy()[start_idx:end_idx].flatten()
    
    samples_per_func = config['test_sample_num']
    start_idx = sample_idx * samples_per_func
    branch_input = data['test_branch_input'].asnumpy()[start_idx]  # Take first sample of this function
    
    # Create x coordinates for branch input (assumed to be evenly spaced from 0 to 1)
    num_points = len(branch_input)
    x_points = np.linspace(0, 1, num_points)
    
    # Sort points for interpolation
    sort_indices = np.argsort(pred_x)
    sorted_x = pred_x[sort_indices]
    sorted_true_u = true_u[sort_indices]
    sorted_pred_u = pred_u[sort_indices]
    
    # Create interpolated curves for visualization
    from scipy.interpolate import interp1d
    
    # Create finer x grid for smooth curves
    x_fine = np.linspace(sorted_x.min(), sorted_x.max(), 200)
    
    # Interpolate true and predicted values
    f_true = interp1d(sorted_x, sorted_true_u, kind='cubic', fill_value='extrapolate')
    f_pred = interp1d(sorted_x, sorted_pred_u, kind='cubic', fill_value='extrapolate')
    
    true_u_interp = f_true(x_fine)
    pred_u_interp = f_pred(x_fine)
    
    # Interpolate branch input (original function) to sampling points
    f_branch = interp1d(x_points, branch_input, kind='cubic', fill_value='extrapolate')
    branch_at_pred_x = f_branch(pred_x)
    
    # Create horizontal layout with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # Subplot 1: Original function reconstructed from branch_input
    ax1.plot(x_points, branch_input, 'b-', linewidth=2, label='Original Function u0(x) (from branch_input)')
    ax1.scatter(pred_x, branch_at_pred_x, c='blue', s=50, alpha=0.7, label='Sampling Points')
    ax1.set_xlabel('x')
    ax1.set_ylabel('u0(x)')
    ax1.set_title(f'Sample {sample_idx}: Original Function u0(x)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: True vs Predicted solution function (interpolated)
    ax2.plot(x_fine, true_u_interp, 'g-', linewidth=2, alpha=0.8, label='True Solution Function (Interpolated)')
    ax2.plot(x_fine, pred_u_interp, 'r-', linewidth=2, alpha=0.8, label='Predicted Solution Function (Interpolated)')
    ax2.scatter(sorted_x, sorted_true_u, c='green', s=50, alpha=0.8, label='True Values', zorder=5)
    ax2.scatter(sorted_x, sorted_pred_u, c='red', s=50, alpha=0.8, label='Predicted Values', marker='^', zorder=5)
    ax2.set_xlabel('x')
    ax2.set_ylabel('u(x)')
    ax2.set_title('True vs Predicted Solution Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Error analysis
    errors = np.abs(sorted_pred_u - sorted_true_u)
    ax3.scatter(sorted_x, errors, c='purple', s=50, alpha=0.7)
    ax3.plot(sorted_x, errors, 'purple', alpha=0.5, linewidth=1)
    ax3.set_xlabel('x')
    ax3.set_ylabel('|Predicted - True|')
    ax3.set_title(f'Absolute Error (MAE: {np.mean(errors):.6f})')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(f"{save_path}/sample_{sample_idx}_comparison.png", dpi=300, bbox_inches='tight')
        print(f"📊 图表已保存: {save_path}")
    
    plt.show()

def plot_special_functions_comparison(model, config, save_path=None):
    """绘制特定函数对比图：u0=x 和 u0=sin(πx)，使用微分方程数值解"""
    
    print("🔧 构造特定函数 u0=x 和 u0=sin(πx) 进行预测...")
    
    # 获取特定函数的预测结果
    special_results = predict_special_functions(model, config)
    
    for row, func_result in enumerate(special_results):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        func_operator_type = func_result['operator_type']
        u0 = func_result['u0']
        u_true_full = func_result['u_true_full']
        x_points = func_result['x_points']
        sample_x = func_result['sample_x']
        pred_u = func_result['pred_u']
        true_u = func_result['true_u']
        
        # 子图 1：原始函数
        axes[0].plot(x_points, u0, 'b-', linewidth=2, label='Original Function u0(x)')
        axes[0].scatter(sample_x, u0[np.searchsorted(x_points, sample_x)], 
                           c='blue', s=50, alpha=0.7, label='Sampling Points')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('u0(x)')
        axes[0].set_title(f'{func_operator_type}: Original Function')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 子图 2：真实值 vs 预测值
        axes[1].plot(x_points, u_true_full, 'g-', linewidth=2, alpha=0.8, label='True Solution (ODE)')
        axes[1].scatter(sample_x, true_u, c='green', s=50, alpha=0.7, label='True Values (Sampled)')
        axes[1].scatter(sample_x, pred_u, c='red', s=50, alpha=0.7, label='Predicted Values', marker='^')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('u(x)')
        axes[1].set_title('True vs Predicted Solution')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 子图 3：误差分析
        errors = np.abs(pred_u - true_u)
        axes[2].scatter(sample_x, errors, c='purple', s=50, alpha=0.7)
        axes[2].plot(sample_x, errors, 'purple', alpha=0.5, linewidth=1)
        axes[2].set_xlabel('x')
        axes[2].set_ylabel('|Predicted - True|')
        mae = np.mean(errors)
        axes[2].set_title(f'Absolute Error (MAE: {mae:.6f})')
        axes[2].grid(True, alpha=0.3)
        
        print(f"  {func_operator_type}: MAE = {mae:.6f}")
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/sample_{func_operator_type}_comparison.png", dpi=300, bbox_inches='tight')

        plt.tight_layout()
        plt.show() 

    print(f"📊 特定函数比较图已保存: {save_path}")
    

    
    return special_results

def predict_special_functions(model, config, num_points=100, num_samples=100):
    """构造并预测特定函数：u0=x 和 u0=sin(πx)，通过求解微分方程获得真解"""
    
    # 创建 x 坐标范围
    x_points = np.linspace(0, 1, num_points)
    
    # 定义特定函数
    u0_linear = x_points  # u0 = x
    u0_sin = np.sin(np.pi * x_points)  # u0 = sin(πx)
    
    # 通过求解微分方程 du/dx = u0(x) 获得数值解
    print("   正在求解微分方程 du/dx = u0(x)...")
    
    # 对于 u0 = x，求解 du/dx = x，初始条件 u(0) = 0
    u_linear_true = solve_ode_system(u0_linear, x_points, config['ode_system'], initial_condition=0)

    # 对于 u0 = sin(πx)，求解 du/dx = sin(πx)，初始条件 u(0) = 0
    u_sin_true = solve_ode_system(u0_sin, x_points, config['ode_system'], initial_condition=0)

    # 准备网络预测数据，使用与训练数据相同的编码方式
    special_functions = [
        {'operator_type': 'Linear Function (u0=x)', 'u0': u0_linear, 'u_true': u_linear_true},
        {'operator_type': 'Sine Function (u0=sin(πx))', 'u0': u0_sin, 'u_true': u_sin_true}
    ]
    
    predictions_results = []
    
    for func_data in special_functions:
        # 创建 branch input（传感器点处的函数值）
        branch_input = ms.Tensor(func_data['u0'].reshape(1, -1), ms.float32)
        
        # 创建 trunk input（预测的采样点）
        sample_x = np.random.uniform(0, 1, num_samples)  # 随机采样点
        sample_x = np.sort(sample_x)  # 排序以便可视化
        trunk_input = ms.Tensor(sample_x.reshape(-1, 1), ms.float32)
        
        # 为每个采样点重复 branch input
        branch_input_expanded = ms.ops.tile(branch_input, (num_samples, 1))
        
        # 网络预测
        network_input = (branch_input_expanded, trunk_input)
        pred_u = model(network_input).asnumpy().flatten()
        
        # 通过插值获得采样点处的真实值
        from scipy.interpolate import interp1d
        f_true = interp1d(x_points, func_data['u_true'], kind='cubic', fill_value='extrapolate')
        true_u = f_true(sample_x)
        
        predictions_results.append({
            'operator_type': func_data['operator_type'],
            'u0': func_data['u0'],
            'u_true_full': func_data['u_true'],
            'x_points': x_points,
            'sample_x': sample_x,
            'pred_u': pred_u,
            'true_u': true_u
        })
    
    return predictions_results