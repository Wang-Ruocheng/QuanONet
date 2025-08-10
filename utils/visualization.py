"""
Visualization functions for different operator problems.
"""

import numpy as np
import mindspore as ms
import mindspore.numpy as mnp
from scipy.integrate import solve_ivp


def inverse_plot(v_cal, num_sensors, Network, num_cal=1000):
    """Plot results for inverse operator problem."""
    x_cal = np.linspace(0, 1, num_cal)
    u_cal = np.cumsum(v_cal) * (x_cal[1] - x_cal[0])
    v_cal = ms.Tensor(v_cal, ms.float32)
    u_cal = ms.Tensor(u_cal, ms.float32)
    x_cal = ms.Tensor(x_cal, ms.float32)
    
    u = u_cal[::int(num_cal/num_sensors)]
    x = x_cal[::int(num_cal/num_sensors)]
    v = v_cal[::int(num_cal/num_sensors)]
    
    branch_input = mnp.repeat(ms.ops.expand_dims(v, 0), num_sensors, axis=0)
    trunk_input = ms.ops.expand_dims(x, 1)
    print(branch_input.shape, trunk_input.shape)
    output = Network((branch_input, trunk_input))
    
    return branch_input, trunk_input, output, u.reshape(-1, 1)


def homogeneous_plot(v_cal, num_sensors, Network, num_cal=1000):
    """Plot results for homogeneous operator problem."""
    x_cal = np.linspace(0, 1, num_cal)
    
    def ode_system(x, u):
        v = np.interp(x, x_cal, v_cal)
        return u + v
    
    sol = solve_ivp(ode_system, [x_cal[0], x_cal[-1]], [0], t_eval=x_cal)
    u_cal = sol.y[0]
    
    v_cal = ms.Tensor(v_cal, ms.float32)
    u_cal = ms.Tensor(u_cal, ms.float32)
    x_cal = ms.Tensor(x_cal, ms.float32)
    
    u = u_cal[::int(num_cal/num_sensors)]
    x = x_cal[::int(num_cal/num_sensors)]
    v = v_cal[::int(num_cal/num_sensors)]
    
    branch_input = mnp.repeat(ms.ops.expand_dims(v, 0), num_sensors, axis=0)
    trunk_input = ms.ops.expand_dims(x, 1)
    output = Network((branch_input, trunk_input))
    
    return branch_input, trunk_input, output, u.reshape(-1, 1)


def nonlinear_plot(v_cal, num_sensors, Network, num_cal=1000):
    """Plot results for nonlinear operator problem."""
    x_cal = np.linspace(0, 1, num_cal)
    
    def ode_system(x, u):
        v = np.interp(x, x_cal, v_cal)
        return u - v ** 2
    
    sol = solve_ivp(ode_system, [x_cal[0], x_cal[-1]], [0], t_eval=x_cal, method='RK45')
    u_cal = sol.y[0]
    
    v_cal = ms.Tensor(v_cal, ms.float32)
    u_cal = ms.Tensor(u_cal, ms.float32)
    x_cal = ms.Tensor(x_cal, ms.float32)
    
    u = u_cal[::int(num_cal/num_sensors)]
    x = x_cal[::int(num_cal/num_sensors)]
    v = v_cal[::int(num_cal/num_sensors)]
    
    branch_input = mnp.repeat(ms.ops.expand_dims(v, 0), num_sensors, axis=0)
    trunk_input = ms.ops.expand_dims(x, 1)
    print(branch_input.shape, trunk_input.shape)
    output = Network((branch_input, trunk_input))
    
    return branch_input, trunk_input, output, u.reshape(-1, 1)


def diffusion_plot(u0_cal, num_sensors, Network, num_cal=1000):
    """Plot results for diffusion operator problem."""
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)
    
    def diffusion_step(u, dx, dt, alpha, k, u0):
        u_new = np.zeros_like(u)
        for i in range(1, len(u) - 1):
            u_new[i] = u[i] + dt * (alpha * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2) + k * (u[i]**2) + u0[i])
        u_new[0] = u_new[-1] = 0  # Boundary conditions
        return u_new
    
    dx_cal = x_cal[1] - x_cal[0]
    dt_cal = min(dx_cal**2 / (2 * 0.01), t_cal[1] - t_cal[0])  # Ensure numerical stability
    u_cal = np.zeros((num_cal, num_cal))
    
    for i in range(1, num_cal):
        u_cal[:, i] = diffusion_step(u_cal[:, i-1], dx_cal, dt_cal, 0.01, 0.01, u0_cal)
    
    u0_cal = ms.Tensor(u0_cal, ms.float32)
    u_cal = ms.Tensor(u_cal, ms.float32)
    x_cal = ms.Tensor(x_cal, ms.float32)
    t_cal = ms.Tensor(t_cal, ms.float32)
    
    u = u_cal[::int(num_cal/num_sensors), ::int(num_cal/num_sensors)]
    x = x_cal[::int(num_cal/num_sensors)]
    t = t_cal[::int(num_cal/num_sensors)]
    u0 = u0_cal[::int(num_cal/num_sensors)]
    
    branch_input = mnp.repeat(ms.ops.expand_dims(u0, 0), num_sensors**2, axis=0)
    x_repeat = mnp.repeat(x, num_sensors, axis=0).expand_dims(1)
    t_tile = mnp.tile(t, num_sensors).expand_dims(1)
    trunk_input = mnp.concatenate((x_repeat, t_tile), axis=1)
    output = Network((branch_input, trunk_input)).reshape(num_sensors, num_sensors)
    
    return branch_input, trunk_input, output, u
