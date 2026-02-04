"""
Unified metrics calculation.
"""
import numpy as np

def compute_metrics(y_true, y_pred):
    """
    Computes MSE, MAE, and Max Error.
    Automatically flattens inputs to avoid shape mismatch (e.g. (N,1) vs (N,)).
    """
    # Convert to numpy if they are tensors (MindSpore/Torch)
    if hasattr(y_true, 'asnumpy'): y_true = y_true.asnumpy()
    if hasattr(y_pred, 'asnumpy'): y_pred = y_pred.asnumpy()
    if hasattr(y_true, 'detach'): y_true = y_true.detach().cpu().numpy()
    if hasattr(y_pred, 'detach'): y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten to 1D arrays to ensure safe broadcasting
    t = np.ravel(y_true)
    p = np.ravel(y_pred)
    
    mse = np.mean((t - p)**2)
    mae = np.mean(np.abs(t - p))
    max_error = np.max(np.abs(t - p))
    
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'Max_Error': float(max_error)
    }