"""
Data generation script - Generate training and test data for various operator problems

Usage:
    python data_utils/generate_data.py --problem Inverse_Operator --num_train 1000 --num_test 1000
    python data_utils/generate_data.py --problem Diffusion_Operator --num_train 100 --num_test 100
"""

import sys
import os
import argparse
import time

# Add parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import modules from current directory
from data_utils.data_generation import *
from data_utils.data_processing import *
from core.quantum_circuits import *
from core.models import *
from core.layers import *
from utils.loss_functions import *
from utils.visualization import *
from utils.utils import *
import mindspore as ms
import mindspore.numpy as mnp
import numpy as np


def setup_mindspore():
    """Set up MindSpore environment"""
    ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="CPU")


def get_problem_config():
    """Get problem configuration"""
    # Data generation function mapping
    generate_data_dict = {
        "Inverse_Operator": generate_Inverse_Operator_data, 
        "Homogeneous_Operator": generate_Homogeneous_Operator_data, 
        "Nonlinear_Operator": generate_Nonlinear_Operator_data, 
        "Diffusion_Operator": generate_Diffusion_Operator_data, 
        "Advection_Operator": generate_Advection_Operator_data
    }
    
    # Problem type mapping (ODE or PDE)
    DE_dict = {
        "Inverse_Operator": "ODE", 
        "Nonlinear_Operator": "ODE", 
        "Homogeneous_Operator": "ODE",
        "Diffusion_Operator": "PDE", 
        "Advection_Operator": "PDE"
    }
    
    return generate_data_dict, DE_dict


def generate_dataset(problem, num_train=10000, num_test=1000, num_sensors=100, 
                    train_sample_num=100, test_sample_num=100, save_data=True):
    """
    Generate dataset for specified problem
    
    Args:
        problem: Problem type
        num_train: Number of training samples
        num_test: Number of test samples
        num_sensors: Number of sensors
        train_sample_num: Training sample count
        test_sample_num: Test sample count
        save_data: Whether to save data
    
    Returns:
        Dictionary containing training and test data
    """
    print(f"Starting to generate data for {problem} problem...")
    print(f"Training samples: {num_train}, Test samples: {num_test}, Sensors: {num_sensors}")
    
    # Get configuration
    generate_data_dict, DE_dict = get_problem_config()
    
    if problem not in generate_data_dict:
        raise ValueError(f"Unsupported problem type: {problem}")
    
    # Calculate network input dimensions
    branch_input_size = num_sensors
    trunk_input_size = 1 if DE_dict[problem] == "ODE" else 2
    output_size = 1
    
    # Ensure sample count does not exceed sensor count
    train_sample_num = min(train_sample_num, num_sensors)
    test_sample_num = min(test_sample_num, num_sensors)
    
    # Select data generation function and encoding function
    generate_data = generate_data_dict[problem]
    encode = PDE_encode if DE_dict[problem] == "PDE" else ODE_encode
    
    print(f"Problem type: {DE_dict[problem]}")
    print(f"Branch network input dimension: {branch_input_size}")
    print(f"Trunk network input dimension: {trunk_input_size}")
    print(f"Training sample count: {train_sample_num}")
    print(f"Test sample count: {test_sample_num}")
    
    # Record start time
    start_time = time.time()
    
    # Generate and encode data
    print("Generating data...")
    train_branch_input, train_trunk_input, train_output, \
    test_branch_input, test_trunk_input, test_output = encode(
        generate_data, num_train, num_test, num_sensors, 
        train_sample_num, test_sample_num
    )
    
    # Data statistics
    print(f"\nData generation completed! Time taken: {time.time() - start_time:.2f} seconds")
    print(f"Training data shapes:")
    print(f"  Branch input: {train_branch_input.shape}")
    print(f"  Trunk input: {train_trunk_input.shape}")
    print(f"  Output: {train_output.shape}")
    print(f"Test data shapes:")
    print(f"  Branch input: {test_branch_input.shape}")
    print(f"  Trunk input: {test_trunk_input.shape}")
    print(f"  Output: {test_output.shape}")
    
    # Organize data
    data_dict = {
        'problem': problem,
        'problem_type': DE_dict[problem],
        'config': {
            'num_train': num_train,
            'num_test': num_test,
            'num_sensors': num_sensors,
            'train_sample_num': train_sample_num,
            'test_sample_num': test_sample_num,
            'branch_input_size': branch_input_size,
            'trunk_input_size': trunk_input_size,
            'output_size': output_size
        },
        'train_data': {
            'branch_input': train_branch_input,
            'trunk_input': train_trunk_input,
            'output': train_output
        },
        'test_data': {
            'branch_input': test_branch_input,
            'trunk_input': test_trunk_input,
            'output': test_output
        }
    }
    
    # 保存数据
    if save_data:
        save_path = f"data/{problem}_dataset_{num_train}_{num_test}_{num_sensors}.npz"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        print(f"正在保存数据到: {save_path}")
        np.savez_compressed(
            save_path,
            **{k: v.asnumpy() if hasattr(v, 'asnumpy') else v 
               for k, v in data_dict.items() if k not in ['config']}
        )
        
        # 保存配置信息
        config_path = f"data/{problem}_config_{num_train}_{num_test}_{num_sensors}.json"
        import json
        with open(config_path, 'w') as f:
            json.dump(data_dict['config'], f, indent=2)
        
        print(f"数据已保存到: {save_path}")
        print(f"配置已保存到: {config_path}")
    
    return data_dict


def generate_concatenated_data(problem, num_train=10000, num_test=1000, num_sensors=100, 
                              train_sample_num=100, test_sample_num=100):
    """
    生成拼接格式的数据（兼容原始脚本格式）
    
    Returns:
        train_input: 拼接后的训练输入 (branch + trunk)
        train_output: 训练输出
        test_input: 拼接后的测试输入 (branch + trunk)
        test_output: 测试输出
    """
    # 生成数据
    data_dict = generate_dataset(
        problem, num_train, num_test, num_sensors, 
        train_sample_num, test_sample_num, save_data=False
    )
    
    # 拼接分支和主干输入
    train_input = mnp.concatenate((
        data_dict['train_data']['branch_input'], 
        data_dict['train_data']['trunk_input']
    ), axis=1)
    
    test_input = mnp.concatenate((
        data_dict['test_data']['branch_input'], 
        data_dict['test_data']['trunk_input']
    ), axis=1)
    
    train_output = data_dict['train_data']['output']
    test_output = data_dict['test_data']['output']
    
    print(f"拼接后的数据形状:")
    print(f"  训练输入: {train_input.shape}")
    print(f"  训练输出: {train_output.shape}")
    print(f"  测试输入: {test_input.shape}")
    print(f"  测试输出: {test_output.shape}")
    
    return train_input, train_output, test_input, test_output


def check_data_quality(data_dict):
    """检查数据质量"""
    print("\n=== 数据质量检查 ===")
    
    train_branch = data_dict['train_data']['branch_input'].asnumpy()
    train_trunk = data_dict['train_data']['trunk_input'].asnumpy()
    train_output = data_dict['train_data']['output'].asnumpy()
    
    # 检查 NaN 和 Inf
    def check_nan_inf(data, name):
        has_nan = np.isnan(data).any()
        has_inf = np.isinf(data).any()
        print(f"{name}: NaN={has_nan}, Inf={has_inf}")
        return has_nan or has_inf
    
    has_issues = False
    has_issues |= check_nan_inf(train_branch, "训练分支输入")
    has_issues |= check_nan_inf(train_trunk, "训练主干输入")
    has_issues |= check_nan_inf(train_output, "训练输出")
    
    if not has_issues:
        print("✓ 数据质量检查通过，无异常值")
    
    # 数据范围统计
    print(f"\n数据范围统计:")
    print(f"  分支输入: [{train_branch.min():.4f}, {train_branch.max():.4f}]")
    print(f"  主干输入: [{train_trunk.min():.4f}, {train_trunk.max():.4f}]")
    print(f"  输出: [{train_output.min():.4f}, {train_output.max():.4f}]")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='生成算子学习问题的数据集')
    parser.add_argument('--problem', type=str, default='Inverse_Operator',
                       choices=['Inverse_Operator', 'Homogeneous_Operator', 
                               'Nonlinear_Operator', 'Diffusion_Operator', 
                               'Advection_Operator'],
                       help='问题类型')
    parser.add_argument('--num_train', type=int, default=1000, help='训练样本数量')
    parser.add_argument('--num_test', type=int, default=1000, help='测试样本数量')
    parser.add_argument('--num_sensors', type=int, default=100, help='传感器数量')
    parser.add_argument('--train_sample_num', type=int, default=100, help='训练采样数量')
    parser.add_argument('--test_sample_num', type=int, default=100, help='测试采样数量')
    parser.add_argument('--save_data', action='store_true', default=True, help='保存数据')
    parser.add_argument('--check_quality', action='store_true', help='检查数据质量')
    parser.add_argument('--concatenated', action='store_true', help='生成拼接格式数据')
    
    args = parser.parse_args()
    
    # 设置 MindSpore
    setup_mindspore()
    
    print("=== 数据生成器 ===")
    print(f"问题类型: {args.problem}")
    
    try:
        if args.concatenated:
            # 生成拼接格式数据
            train_input, train_output, test_input, test_output = generate_concatenated_data(
                args.problem, args.num_train, args.num_test, args.num_sensors,
                args.train_sample_num, args.test_sample_num
            )
        else:
            # 生成标准格式数据
            data_dict = generate_dataset(
                args.problem, args.num_train, args.num_test, args.num_sensors,
                args.train_sample_num, args.test_sample_num, args.save_data
            )
            
            if args.check_quality:
                check_data_quality(data_dict)
        
        print("\n✓ 数据生成完成！")
        
    except Exception as e:
        print(f"❌ 数据生成失败: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
