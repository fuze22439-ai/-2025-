import numpy as np
import matplotlib.pyplot as plt

def trapezoidal_rule(f, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (0.5 * y[0] + 0.5 * y[-1] + np.sum(y[1:-1]))

def simpson_rule(f, a, b, n):
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule.")
    x = np.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h / 3 * (y[0] + y[-1] + 4 * np.sum(y[1:-1:2]) + 2 * np.sum(y[2:-1:2]))

def task_1_basic_integration():
    f = np.sin
    exact = 2.0
    ns = [4, 8, 16, 32, 64]
    err_trap = []
    err_simp = []
    
    for n in ns:
        i_trap = trapezoidal_rule(f, 0, np.pi, n)
        i_simp = simpson_rule(f, 0, np.pi, n)
        err_trap.append(abs(i_trap - exact))
        err_simp.append(abs(i_simp - exact))
    
    plt.figure(figsize=(8, 6))
    plt.loglog(ns, err_trap, 'o-', label='Trapezoidal Rule')
    plt.loglog(ns, err_simp, 's-', label="Simpson's Rule")
    
    # helper lines for O(h^2) and O(h^4)
    # h = pi/n => error proportional to n^-2 and n^-4
    plt.loglog(ns, [err_trap[0]*(ns[0]/n)**2 for n in ns], 'k--', alpha=0.3, label='O(n^-2)')
    plt.loglog(ns, [err_simp[0]*(ns[0]/n)**4 for n in ns], 'k:', alpha=0.3, label='O(n^-4)')
    
    plt.title("Error Scaling of Numerical Integration")
    plt.xlabel("Number of intervals (n)")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.grid(True, which="both", ls="-")
    plt.savefig("integration_error.png")
    plt.show()

def gaussian_integral_truncated():
    # Integral of exp(-x^2) from -inf to inf is sqrt(pi)
    f = lambda x: np.exp(-x**2)
    exact = np.sqrt(np.pi)
    Ls = np.arange(1, 6, 0.5)
    errors = []
    for L in Ls:
        val = trapezoidal_rule(f, -L, L, 1000)
        errors.append(abs(val - exact))
    
    plt.figure()
    plt.semilogy(Ls, errors, 'o-')
    plt.title("Accuracy vs Truncation Limit L")
    plt.xlabel("L")
    plt.ylabel("Error")
    plt.grid(True)
    plt.savefig("gaussian_truncation.png")
    plt.show()

def gaussian_integral_transformed():
    # Transformation t = x / sqrt(1-x^2) => x = t / sqrt(1-t^2) ?? 
    # Usually we use x = t / (1-t^2) or x = tan(t)
    # The prompt says t = x / sqrt(1-x^2) which maps x in [-inf, inf] to ... wait.
    # If x -> inf, t -> x/|x| = 1. If x -> -inf, t -> -1.
    # Inverse: x = t / sqrt(1-t^2)
    # dx = [sqrt(1-t^2) - t * (1/2)(1-t^2)^-1/2 * (-2t)] / (1-t^2) dt
    # dx = [ (1-t^2) + t^2 ] / (1-t^2)^3/2 dt = (1-t^2)^-3/2 dt
    
    def integrand_transformed(t):
        # Clip t to avoid division by zero near boundaries
        t = np.clip(t, -1 + 1e-15, 1 - 1e-15)
        x = t / np.sqrt(1 - t**2)
        dx_dt = (1 - t**2)**(-1.5)
        return np.exp(-x**2) * dx_dt

    val = trapezoidal_rule(integrand_transformed, -0.999, 0.999, 2000)
    print(f"Transformed Integral result: {val}, Error: {abs(val - np.sqrt(np.pi))}")

def monte_carlo_integration(N):
    # Integral of exp(-x^2) from 0 to 1
    x = np.random.uniform(0, 1, N)
    f_vals = np.exp(-x**2)
    est = np.mean(f_vals)
    # Exact value of erf(1)*sqrt(pi)/2 is roughly 0.7468241328
    exact = 0.746824132812427
    return est, abs(est - exact)

def task_3_monte_carlo():
    Ns = np.logspace(2, 6, 10, dtype=int)
    errors = []
    for N in Ns:
        _, err = monte_carlo_integration(N)
        errors.append(err)
    
    plt.figure()
    plt.loglog(Ns, errors, 'o-', label='Monte Carlo Error')
    plt.loglog(Ns, [errors[0]*(Ns[0]/n)**0.5 for n in Ns], 'k--', alpha=0.3, label='O(1/sqrt(N))')
    plt.title("Monte Carlo Error Scaling")
    plt.xlabel("N")
    plt.ylabel("Error")
    plt.legend()
    plt.grid(True)
    plt.savefig("monte_carlo_error.png")
    plt.show()

if __name__ == "__main__":
    task_1_basic_integration()
    gaussian_integral_truncated()
    gaussian_integral_transformed()
    task_3_monte_carlo()
