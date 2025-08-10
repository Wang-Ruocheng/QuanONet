"""
Utility functions for the QuanONet library.
"""

import numpy as np
import os
import pickle
import logging


class StreamToLogger(object):
    """Redirect stdout/stderr to logger."""
    
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass


def setup_logging(output_path):
    """Setup logging configuration."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        logging.basicConfig(
            filename=output_path, 
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        return True
    except Exception as e:
        print(f"Failed to set up logging: {e}")
        return False


def count_parameters(network):
    """Count the number of trainable parameters in a network."""
    total_params = 0
    for param in network.trainable_params():
        total_params += np.prod(param.shape)
    return total_params


def save_results(results, filepath):
    """Save results to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(results, f)


def load_results(filepath):
    """Load results from file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def create_checkpoint_path(base_path, model_name, config):
    """Create standardized checkpoint path."""
    config_str = "_".join([str(v) for v in config.values()])
    filename = f"{model_name}_{config_str}.ckpt"
    return os.path.join(base_path, filename)


def print_model_info(model, model_name):
    """Print model information."""
    param_count = count_parameters(model)
    print(f"Model: {model_name}")
    print(f"Total parameters: {param_count:,}")
    print("-" * 50)
