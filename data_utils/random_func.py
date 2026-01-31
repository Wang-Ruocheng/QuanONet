import numpy as np
import mindspore as ms

def RBF(x1, x2, params, if_period=False):
    """
    Radial Basis Function kernel (supports both Standard and Periodic).
    
    Args:
        x1, x2: Input tensors of shape (N, d)
        params: Tuple (output_scale, lengthscales)
        if_period: Boolean, if True uses Periodic Kernel, else Standard RBF
    """
    output_scale, lengthscales = params
    
    if not if_period:
        # --- Standard RBF Kernel ---
        # Formula: exp( - ||x1 - x2||^2 / (2 * l^2) )
        # We calculate (x1/l - x2/l)^2 which equals (x1-x2)^2 / l^2
        diffs = np.expand_dims(x1 / lengthscales, 1) - \
                np.expand_dims(x2 / lengthscales, 0)
        r2 = np.sum(diffs**2, axis=2)
        return output_scale * np.exp(-0.5 * r2)
        
    else:
        # --- Periodic Kernel ---
        # Formula: exp( - 2 * sin^2(pi * ||x1 - x2||) / l^2 )
        # Note: We need the raw difference first, then apply sin(pi * diff)
        diffs_raw = np.expand_dims(x1, 1) - np.expand_dims(x2, 0)
        sin_term = np.sin(np.pi * np.abs(diffs_raw))
        sin_sq = np.sum(sin_term**2, axis=2)
        return output_scale * np.exp(-2 * sin_sq / (lengthscales**2))

def generate_random_gaussian_field(m, length_scale=0.2, if_period=False):
    """
    Generate random Gaussian field using Gaussian process.
    
    Args:
        m: Number of output points (resolution of u)
        length_scale: Length scale parameter for the kernel
        if_period: Whether to enforce periodic boundary conditions u(0)=u(1)
    
    Returns:
        u_fn: Interpolation function
        u: Sampled values at m points
    """
    N = 1024
    jitter = 1e-10
    gp_params = (1.0, length_scale)

    # Generate grid for GP prior
    X = np.linspace(0, 1, N)[:, None]
    
    # --- FIX: Pass if_period flag to RBF ---
    K = RBF(X, X, gp_params, if_period=if_period)
    
    # Cholesky decomposition for sampling
    try:
        L = np.linalg.cholesky(K + jitter * np.eye(N))
    except np.linalg.LinAlgError:
        # Fallback if jitter is insufficient (rare but possible)
        L = np.linalg.cholesky(K + 1e-6 * np.eye(N))
        
    key_train = np.random.randn(N) # Standard normal noise
    gp_sample = np.dot(L, key_train)
    
    # Create interpolation function
    u_fn = lambda x: np.interp(x, X.flatten(), gp_sample)
    
    # Sample at requested resolution
    x_query = np.linspace(0, 1, m)
    u = u_fn(x_query)
    
    return u_fn, u

# --- Test Code ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 1. Generate Non-Periodic Field
    _, u_rbf = generate_random_gaussian_field(m=100, length_scale=0.2, if_period=False)
    
    # 2. Generate Periodic Field
    _, u_per = generate_random_gaussian_field(m=100, length_scale=0.2, if_period=True)
    
    print(f"Non-Periodic Boundary: u(0)={u_rbf[0]:.4f}, u(1)={u_rbf[-1]:.4f}")
    print(f"Periodic Boundary:     u(0)={u_per[0]:.4f}, u(1)={u_per[-1]:.4f}")
    
    # Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, 1, 100), u_rbf, label='Standard RBF')
    plt.plot(np.linspace(0, 1, 100), u_per, label='Periodic Kernel')
    plt.legend()
    plt.title("Gaussian Random Fields")
    plt.show()