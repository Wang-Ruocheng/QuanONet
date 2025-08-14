import numpy as np
import mindspore as ms

def RBF(x1, x2, params):
    """Radial Basis Function kernel."""
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs**2, axis=2)  # Calculate squared difference sum
    return output_scale * np.exp(-0.5 * r2)  # Return RBF kernel result

def generate_random_gaussian_field(m, length_scale=0.2):
    """
    Generate random Gaussian field using Gaussian process.
    
    Args:
        m: Number of output points
        length_scale: Length scale parameter for the RBF kernel
    
    Returns:
        u_fn: Interpolation function
        u: Sampled values at m points
    """
    N = 1024
    jitter = 1e-10
    gp_params = (1.0, length_scale)

    X = np.linspace(0, 1, N)[:, None]
    K = RBF(X, X, gp_params)  # RBF is radial basis function (squared exponential kernel)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    key_train = ms.Tensor(np.random.randn(N)).asnumpy()
    gp_sample = np.dot(L, key_train)
    
    u_fn = lambda x: np.interp(x, X.flatten(), gp_sample)
    x = np.linspace(0, 1, m)
    u = u_fn(x)
    
    return u_fn, u
