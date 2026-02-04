"""
Unified Data Manager for Operator Learning.
Handles data loading, generation, caching, and model-specific formatting (e.g., FNO).
"""

import os
import numpy as np
import logging
from .data_generation import (
    generate_Inverse_Operator_data,
    generate_Homogeneous_Operator_data,
    generate_Nonlinear_Operator_data,
    generate_RDiffusion_Operator_data,
    generate_Advection_Operator_data,
    generate_Darcy_Operator_data,
    generate_ODE_Operator_data,
    generate_PDE_Operator_data
)
from .data_processing import ODE_encode, PDE_encode, ODE_fncode

# Map operator names to their generator functions
GENERATOR_MAP = {
    'Inverse': generate_Inverse_Operator_data,
    'Homogeneous': generate_Homogeneous_Operator_data,
    'Nonlinear': generate_Nonlinear_Operator_data,
    'RDiffusion': generate_RDiffusion_Operator_data,
    'Advection': generate_Advection_Operator_data,
    'Darcy': generate_Darcy_Operator_data,
    # Generic fallbacks
    'ODE': generate_ODE_Operator_data,
    'PDE': generate_PDE_Operator_data
}

class DataManager:
    def __init__(self, config, data_dir="data", logger=None):
        """
        Args:
            config (dict): Configuration dictionary containing:
                - operator_type, model_type, num_train, num_test
                - num_points, num_points_0, train_sample_num, test_sample_num
                - num_cal (optional)
            data_dir (str): Root directory for data storage.
            logger: Optional logger instance.
        """
        self.config = config
        self.data_dir = data_dir
        self.logger = logger or logging.getLogger(__name__)
        
        self.operator_type = config['operator_type']
        self.model_type = config.get('model_type', 'DeepONet')
        
        # Ensure critical parameters exist with defaults
        self.num_points = config.get('num_points', 100)
        self.num_points_0 = config.get('num_points_0', 1000)
        self.num_cal = config.get('num_cal', 1000)
        
    def get_data(self):
        """
        Main entry point.
        Returns a dictionary containing:
            - train_input, train_output
            - test_input, test_output
            - (Optional) Branch/Trunk splits if applicable
        """
        # 1. Determine filenames based on config
        filename = self._get_filename()
        filepath = os.path.join(self.data_dir, self.operator_type, filename)
        
        # 2. Try loading cached data
        if os.path.exists(filepath):
            self.logger.info(f"Loading cached data from {filepath}")
            try:
                data = np.load(filepath)
                # Convert to dict
                return {k: data[k] for k in data.files}
            except Exception as e:
                self.logger.warning(f"Failed to load cached data: {e}. Regenerating...")
        
        # 3. Generate new data
        self.logger.info(f"Generating new data for {self.operator_type}...")
        data_dict = self._generate_and_process()
        
        # 4. Save to cache
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.logger.info(f"Saving data to {filepath}")
        np.savez_compressed(filepath, **data_dict)
        
        return data_dict

    def _get_filename(self):
        """Generate unique filename based on data parameters."""
        c = self.config
        # Base name: Operator_Train_Test_OutRes_InRes
        base = f"{self.operator_type}_{c['num_train']}_{c['num_test']}_{self.num_points}_{self.num_points_0}"
        
        # If model is FNO, the data format is fundamentally different (Grid vs Coordinate)
        if self.model_type == 'FNO':
            base += "_FNO"
        else:
            # For DeepONet/FNN, sample_nums matter
            base += f"_{c.get('train_sample_num', 10)}_{c.get('test_sample_num', 10)}"
            
        return f"{base}.npz"

    def _generate_and_process(self):
        """
        Orchestrates generation -> encoding -> formatting.
        """
        # A. Select Generator
        generator = GENERATOR_MAP.get(self.operator_type)
        if not generator:
            raise ValueError(f"Unknown operator type: {self.operator_type}")
            
        # Helper to call generator with standardized args
        # Note: We bind the static args here so the Encoders just call gen_func(nt, nte)
        def gen_func_wrapper(nt, nte, *args, **kwargs):
                    # *args 会吞掉传入的 num_points 和 num_points_0
                    # 我们直接使用 self.num_points 以保证配置的一致性
                    return generator(
                        nt, nte, 
                        self.num_points, 
                        self.num_points_0, 
                        num_cal=self.num_cal
                    )

        # B. Encode Data based on Model Type
        c = self.config
        
        if self.model_type == 'FNO':
            # === FNO Specific Path ===
            # FNO requires grid-structured data. We use ODE_fncode from data_processing.py
            # Note: ODE_fncode expects a generator that returns (v_train, u_train, v_test, u_test, x)
            # Your generate_* functions return exactly this 5-tuple.
            
            # FNO enforces sample_num == num_points usually
            train_sample = self.num_points
            test_sample = self.num_points
            
            train_in, _, train_out, test_in, _, test_out = ODE_fncode(
                gen_func_wrapper, 
                c['num_train'], c['num_test'], 
                self.num_points, 
                train_sample, test_sample
            )
            
            return {
                'train_input': train_in,
                'train_output': train_out,
                'test_input': test_in,
                'test_output': test_out
            }
            
        else:
            # === DeepONet / FNN / QuanONet Path ===
            # Use ODE_encode or PDE_encode based on operator category
            is_pde = self.operator_type in ['RDiffusion', 'Advection', 'Darcy']
            encoder = PDE_encode if is_pde else ODE_encode
            
            train_branch, train_trunk, train_out, test_branch, test_trunk, test_out = encoder(
                gen_func_wrapper,
                c['num_train'], c['num_test'],
                self.num_points, self.num_points_0,
                c.get('train_sample_num', 10),
                c.get('test_sample_num', 10),
                self.num_cal
            )
            
            # Combine for saving (Standard format)
            return {
                'train_branch_input': train_branch,
                'train_trunk_input': train_trunk,
                'train_output': train_out,
                'test_branch_input': test_branch,
                'test_trunk_input': test_trunk,
                'test_output': test_out,
                # Also save combined input for FNN convenience
                'train_input': np.concatenate([train_branch, train_trunk], axis=1),
                'test_input': np.concatenate([test_branch, test_trunk], axis=1)
            }