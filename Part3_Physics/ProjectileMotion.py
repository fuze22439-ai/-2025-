import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Parameters
g = 9.8
v0 = 20.0
m = 1.0

def projectile_no_drag(theta_deg):
    theta = np.deg2rad(theta_deg)
    t_flight = 2 * v0 * np.sin(theta) / g
    t = np.linspace(0, t_flight, 200)
    x = v0 * np.cos(theta) * t
    y = v0 * np.sin(theta) * t - 0.5 * g * t**2
    return x, y

def task_1_no_drag():
    thetas = [15, 30, 45, 60, 75]
    plt.figure(figsize=(10, 6))
    max_range = 0
    best_theta = 0
    
    for th in thetas:
        x, y = projectile_no_drag(th)
        plt.plot(x, y, label=f'theta={th}°')
        if x[-1] > max_range:
            max_range = x[-1]
            best_theta = th
            
    plt.axhline(0, color='black', lw=1)
    plt.title("Projectile Motion without Air Resistance")
    plt.xlabel("Range (m)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid(True)
    plt.annotate(f'Max Range: {max_range:.2f}m at {best_theta}°', 
                 xy=(max_range, 0), xytext=(max_range-10, 5),
                 arrowprops=dict(facecolor='black', shrink=0.05))
    plt.savefig("projectile_no_drag.png")
    plt.show()
    print(f"Max range is {max_range:.4f}m at theta={best_theta}°")

def projectile_with_drag_ode(t, state, b_m):
    x, y, vx, vy = state
    dxdt = vx
    dydt = vy
    dvxdt = -(b_m) * vx
    dvydt = -g - (b_m) * vy
    return [dxdt, dydt, dvxdt, dvydt]

def solve_projectile_with_drag(v0, theta_deg, b_m):
    theta = np.deg2rad(theta_deg)
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    initial_state = [0, 0, vx0, vy0]
    
    # Event function to stop when y < 0
    def hit_ground(t, state, b_m):
        return state[1]
    hit_ground.terminal = True
    hit_ground.direction = -1
    
    sol = solve_ivp(projectile_with_drag_ode, [0, 10], initial_state, 
                     args=(b_m,), events=hit_ground, max_step=0.05)
    return sol.y[0], sol.y[1], sol.t, sol.y[2], sol.y[3]

def task_2_with_drag():
    b_ms = [0, 0.1, 0.3, 0.5]
    plt.figure(figsize=(10, 6))
    for b_m in b_ms:
        x, y, t, vx, vy = solve_projectile_with_drag(v0, 45, b_m)
        plt.plot(x, y, label=f'b/m={b_m}')
    
    plt.title("Projectile Motion with Air Resistance (theta=45°)")
    plt.xlabel("Range (m)")
    plt.ylabel("Height (m)")
    plt.legend()
    plt.grid(True)
    plt.savefig("projectile_with_drag.png")
    plt.show()

def task_3_energy_analysis():
    # No drag case
    x, y, t, vx, vy = solve_projectile_with_drag(v0, 45, 0)
    v2 = vx**2 + vy**2
    energy = 0.5 * m * v2 + m * g * y
    
    # Drag case (b/m = 0.3)
    xd, yd, td, vxd, vyd = solve_projectile_with_drag(v0, 45, 0.3)
    v2d = vxd**2 + vyd**2
    energy_d = 0.5 * m * v2d + m * g * yd
    
    plt.figure(figsize=(10, 5))
    plt.plot(t, energy, label='No Drag (Total Energy)')
    plt.plot(td, energy_d, label='With Drag (b/m=0.3)')
    plt.title("Mechanical Energy over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("Energy (J)")
    plt.legend()
    plt.grid(True)
    plt.savefig("energy_analysis.png")
    plt.show()

def best_angle_analysis():
    # Find best angle with drag b/m = 0.3
    angles = np.linspace(20, 70, 51)
    ranges = []
    b_m = 0.3
    for ang in angles:
        x, y, t, vx, vy = solve_projectile_with_drag(v0, ang, b_m)
        ranges.append(x[-1])
    
    plt.figure()
    plt.plot(angles, ranges)
    plt.title(f"Range vs Angle (b/m={b_m})")
    plt.xlabel("Angle (deg)")
    plt.ylabel("Range (m)")
    best_ang = angles[np.argmax(ranges)]
    plt.axvline(best_ang, color='r', linestyle='--', label=f'Optimal Angle: {best_ang:.1f}°')
    plt.legend()
    plt.grid(True)
    plt.savefig("optimal_angle_drag.png")
    plt.show()
    print(f"Optimal angle with drag (b/m=0.3) is {best_ang:.2f}°")

if __name__ == "__main__":
    task_1_no_drag()
    task_2_with_drag()
    task_3_energy_analysis()
    best_angle_analysis()
