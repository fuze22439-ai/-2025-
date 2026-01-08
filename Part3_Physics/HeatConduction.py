import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Parameters
L = 1.0
alpha = 0.01
nx = 51 # Number of spatial points
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

# Stability condition r <= 0.5
r = 0.4
dt = r * dx**2 / alpha
print(f"dx: {dx:.4f}, dt: {dt:.4f}, r: {r}")

def heat_simulation(t_max):
    # Initial condition
    T = np.sin(np.pi * x)
    nt = int(t_max / dt)
    
    # Store history for visualization
    history = [T.copy()]
    times = [0]
    
    T_new = T.copy()
    for n in range(nt):
        # Boundary conditions T(0,t) = T(L,t) = 0
        T_new[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[0:-2])
        T[:] = T_new[:]
        if (n + 1) * dt in [0, 0.1, 0.5, 1, 5] or n == nt - 1:
             pass # Will handle specific times manually
        
    return T

def analytical_solution(x, t):
    return np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

def task_1_2_simulation():
    target_times = [0, 0.1, 1, 5, 20] # Using 20 to see more decay
    plt.figure(figsize=(10, 6))
    
    T = np.sin(np.pi * x)
    current_t = 0
    
    for t_target in target_times:
        nt_steps = int((t_target - current_t) / dt)
        for _ in range(nt_steps):
            T[1:-1] = T[1:-1] + r * (T[2:] - 2*T[1:-1] + T[0:-2])
            current_t += dt
        
        plt.plot(x, T, label=f'Numerical t={t_target}s')
        plt.plot(x, analytical_solution(x, t_target), 'k--', alpha=0.3)
        
    plt.title("Heat Conduction (FTCS Method)")
    plt.xlabel("Position x (m)")
    plt.ylabel("Temperature T")
    plt.legend()
    plt.grid(True)
    plt.savefig("heat_conduction_decay.png")
    plt.show()

def task_3_visualization():
    # Run full simulation to get 3D data
    t_end = 10
    nt = int(t_end / dt)
    T_all = np.zeros((nt + 1, nx))
    T_all[0, :] = np.sin(np.pi * x)
    
    for n in range(nt):
        T_all[n+1, 1:-1] = T_all[n, 1:-1] + r * (T_all[n, 2:] - 2*T_all[n, 1:-1] + T_all[n, 0:-2])
        T_all[n+1, 0] = 0
        T_all[n+1, -1] = 0
        
    t_vals = np.linspace(0, t_end, nt + 1)
    X, T_grid = np.meshgrid(x, t_vals)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, T_grid, T_all, cmap=cm.coolwarm, antialiased=False)
    
    ax.set_xlabel('Position x')
    ax.set_ylabel('Time t')
    ax.set_zlabel('Temperature T')
    plt.title("Temperature Evolution (3D Surface)")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.savefig("heat_3d_surface.png")
    plt.show()

def instability_demo():
    # Demonstate instability with r > 0.5
    r_bad = 0.55
    dt_bad = r_bad * dx**2 / alpha
    T = np.sin(np.pi * x)
    for _ in range(200):
        T[1:-1] = T[1:-1] + r_bad * (T[2:] - 2*T[1:-1] + T[0:-2])
    
    plt.figure()
    plt.plot(x, T)
    plt.title(f"Numerical Instability (r={r_bad})")
    plt.savefig("heat_instability.png")
    plt.show()

if __name__ == "__main__":
    task_1_2_simulation()
    task_3_visualization()
    instability_demo()
