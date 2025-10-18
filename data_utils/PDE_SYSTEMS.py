from data_utils.random_func import *

def solve_darcy_pde(num_cal, length_scale, K=0.1, f=-1.0, u0_cal=None):
    """Solve Darcy flow PDE: -∇(K∇u)=f"""
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    nx, ny = num_cal, num_cal
    Lx, Ly = 1.0, 1.0
    dx, dy = Lx/(nx-1), Ly/(ny-1)
    if u0_cal is None:
        # Generate initial condition and source term
        _, u0_cal = generate_random_gaussian_field(4*num_cal, length_scale=length_scale)
    def boundary_from_1d_func(u0):
        edge = len(u0)//4
        left = u0[:edge]
        right = u0[2*edge:3*edge][::-1]
        bottom = u0[3*edge:][::-1]
        top = u0[edge:2*edge]
        return left, right, bottom, top
    left, right, bottom, top = boundary_from_1d_func(u0_cal)

    # Construct sparse matrix and right-hand side
    N = nx * ny
    main = np.ones(N) * (-2/dx**2 - 2/dy**2)
    offx = np.ones(N) / dx**2
    offy = np.ones(N) / dy**2
    rhs = -np.ones(N) * f / K
    for i in range(nx):
        for j in range(ny):
            idx = i*ny + j
            if j == 0:
                main[idx]=1; offx[idx]=0; offy[idx]=0; rhs[idx]=bottom[i]
            elif j == ny-1:
                main[idx]=1; offx[idx]=0; offy[idx]=0; rhs[idx]=top[i]
            elif i == 0:
                main[idx]=1; offx[idx]=0; offy[idx]=0; rhs[idx]=left[j]
            elif i == nx-1:
                main[idx]=1; offx[idx]=0; offy[idx]=0; rhs[idx]=right[j]
    A = diags([main, offx[:-1], offx[1:], offx[-1], offx[:1], offy[:(N-ny)], offy[ny:], offy[(N-ny):], offy[:ny]], [0, 1, -1, -N+1, N-1, ny, -ny, -N+ny, N-ny], shape=(N, N))
    u_cal = spsolve(A.tocsr(), rhs).reshape((nx, ny))

    return u_cal, u0_cal

def solve_rdiffusion_pde(num_cal, length_scale, D=0.01, k=0.01, u0_cal=None):
    """Solve rdiffusion PDE ∂u/∂t = α∇²u + k*u² + u0(x)"""
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)
    
    # Calculate time step parameters
    dx = x_cal[1] - x_cal[0]
    dt = min(dx**2 / (2 * D), t_cal[1] - t_cal[0])  # Ensure stability
    num_cal_t = int(1//dt)
    
    def rdiffusion_step(u, dx, dt, D, k, u0):
        u_new = np.zeros_like(u)
        for i in range(1, len(u) - 1):
            u_new[i] = u[i] + dt * (D * (u[i+1] - 2*u[i] + u[i-1]) / (dx**2) + k * (u[i]**2) + u0[i])
        u_new[0] = u_new[-1] = 0  # Boundary conditions
        return u_new
    
    if u0_cal is None:
        # Generate initial condition and source term
        _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale)
    
    # Time evolution
    u_cal = np.zeros((num_cal, num_cal_t))
    for i in range(1, num_cal_t):
        u_cal[:, i] = rdiffusion_step(u_cal[:, i-1], dx, dt, D, k, u0_cal)
    
    # Sample the data to match num_cal
    u_cal_sampled = u_cal[:, ::max(1, num_cal_t//num_cal)][:, :num_cal]
    
    return u_cal_sampled, u0_cal

def solve_burgers_pde(num_cal, length_scale, nu=0.02, u0_cal=None):
    """Solve Burgers PDE ∂u/∂t + u∂u/∂x = ν∂²u/∂x² + u0(x,t)"""
    x_cal = np.linspace(0, 1, num_cal)
    t_cal = np.linspace(0, 1, num_cal)
    
    # Calculate time step parameters
    dx = x_cal[1] - x_cal[0]
    dt = min(dx**2 / (2 * nu), t_cal[1] - t_cal[0])  # Ensure stability
    num_cal_t = int(1//dt)
    
    def rdiffusion_step(u, dx, dt, nu):
        u_new = np.zeros_like(u)
        for i in range(0, len(u)):
            u_new[i] = u[i] + dt * (nu * (u[(i+1)%len(u)] - 2*u[i] + u[i-1]) / (dx**2) - 0.5 * u[i] * (u[(i+1)%len(u)] - u[i-1]))
        return u_new
    
    if u0_cal is None:
        # Generate initial condition and source term
        _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale)
    
    # Time evolution
    u_cal = np.zeros((num_cal, num_cal_t))
    u_cal[:, 0] = u0_cal
    for i in range(1, num_cal_t):
        u_cal[:, i] = rdiffusion_step(u_cal[:, i-1], dx, dt, nu)

    # Sample the data to match num_cal
    u_cal_sampled = u_cal[:, ::max(1, num_cal_t//num_cal)][:, :num_cal]
    
    return u_cal_sampled, u0_cal

def solve_identity_pde(num_cal, length_scale, u0_cal=None):
    """Solve identity operator: u(x,t) = u0(x) for all t"""
    # Generate initial condition
    if u0_cal is None:
        _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale)

    # For identity operator, u(x,t) = u0(x) for all t
    # Create a 2D array where each time slice is identical to u0
    u_cal = np.tile(u0_cal[:, np.newaxis], (1, num_cal))
    
    return u_cal, u0_cal

def solve_advection_pde(num_cal, length_scale, c=1.0, u0_cal=None):
    """
    Solve advection equation: ∂u/∂t + c∇u = 0
    """
    x_cal = np.linspace(0, 1, num_cal)
    dx = x_cal[1] - x_cal[0]
    
    t_final = 1.0
    dt = 0.8 * dx / abs(c) if c != 0 else 0.01
    num_t = int(t_final / dt)
    
    if u0_cal is None:
        _, u0_cal = generate_random_gaussian_field(num_cal, length_scale=length_scale)

    u_cal = np.zeros((num_cal, num_t))
    u_cal[:, 0] = u0_cal

    # Use upwind finite difference scheme
    for j in range(1, num_t):
        u_prev = u_cal[:, j-1].copy()
        u_new = np.zeros_like(u_prev)
        
        if c > 0:
            # Positive advection, use backward difference
            for i in range(num_cal):
                if i == 0:
                    # Periodic boundary condition
                    u_new[i] = u_prev[i] - c * dt / dx * (u_prev[i] - u_prev[-1])
                else:
                    u_new[i] = u_prev[i] - c * dt / dx * (u_prev[i] - u_prev[i-1])
        elif c < 0:
            # Negative advection, use forward difference
            for i in range(num_cal):
                if i == num_cal - 1:
                    # Periodic boundary condition
                    u_new[i] = u_prev[i] - c * dt / dx * (u_prev[0] - u_prev[i])
                else:
                    u_new[i] = u_prev[i] - c * dt / dx * (u_prev[i+1] - u_prev[i])
        else:
            u_new = u_prev
        
        u_cal[:, j] = u_new
    
    if num_t > num_cal:
        time_indices = np.linspace(0, num_t-1, num_cal, dtype=int)
        u_cal_sampled = u_cal[:, time_indices]
    else:
        from scipy.interpolate import interp1d
        t_old = np.linspace(0, 1, num_t)
        t_new = np.linspace(0, 1, num_cal)
        u_cal_sampled = np.zeros((num_cal, num_cal))
        for i in range(num_cal):
            interp_func = interp1d(t_old, u_cal[i, :], kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            u_cal_sampled[i, :] = interp_func(t_new)
    
    return u_cal_sampled, u0_cal
