import numpy as np
import matplotlib.pyplot as plt

def square_wave_fourier(x, N):
    """
    Fourier series for square wave f(x) = 1 for 0 < x < pi, -1 for pi < x < 2pi.
    f(x) $\approx \sum_{n=1,3,5,...}^{N} (4 / (n * pi)) * sin(n * x)$
    """
    sum_f = np.zeros_like(x)
    for n in range(1, N + 1, 2):
        sum_f += (4 / (n * np.pi)) * np.sin(n * x)
    return sum_f

def triangle_wave_fourier(x, N):
    """
    Fourier series for triangle wave g(x) = |x| for -pi < x < pi.
    a0 = pi
    an = 2/(pi * n^2) * ((-1)^n - 1)
    g(x) $\approx pi/2 + \sum_{n=1}^{N} an * cos(n * x)$
    """
    sum_g = np.full_like(x, np.pi / 2)
    for n in range(1, N + 1):
        an = (2 / (np.pi * n**2)) * ((-1)**n - 1)
        sum_g += an * np.cos(n * x)
    return sum_g

def task_1_square_wave():
    x = np.linspace(0, 2 * np.pi, 1000)
    target = np.where(x < np.pi, 1, -1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, target, 'k--', label='True Square Wave', alpha=0.5)
    
    for N in [3, 5, 11, 51]:
        plt.plot(x, square_wave_fourier(x, N), label=f'N={N}')
    
    plt.title("Fourier Series Approximation of Square Wave")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("square_wave_fourier.png")
    plt.show()

def task_2_triangle_wave():
    x = np.linspace(-np.pi, np.pi, 1000)
    target = np.abs(x)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, target, 'k--', label='True Triangle Wave (|x|)', alpha=0.5)
    
    for N in [1, 3, 5, 10]:
        plt.plot(x, triangle_wave_fourier(x, N), label=f'N={N}')
    
    plt.title("Fourier Series Approximation of Triangle Wave")
    plt.xlabel("x")
    plt.ylabel("g(x)")
    plt.legend()
    plt.grid(True)
    plt.savefig("triangle_wave_fourier.png")
    plt.show()

def task_3_fft():
    # s(t) = sin(2pi * 3t) + 0.5 * sin(2pi * 7t) + 0.3 * sin(2pi * 11t)
    fs = 100 # Sampling frequency
    t = np.arange(0, 2, 1/fs) # t in [0, 2]
    s = np.sin(2 * np.pi * 3 * t) + 0.5 * np.sin(2 * np.pi * 7 * t) + 0.3 * np.sin(2 * np.pi * 11 * t)
    
    # Plot waveform
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(t, s)
    plt.title("Synthesized Signal s(t)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # FFT
    n = len(s)
    f = np.fft.fftfreq(n, d=1/fs)
    s_fft = np.fft.fft(s)
    
    # Display positive frequencies only
    plt.subplot(1, 2, 2)
    plt.plot(f[:n//2], 2.0/n * np.abs(s_fft[:n//2]))
    plt.title("Frequency Spectrum (FFT)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Magnitude")
    plt.xlim(0, 15)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("signal_fft.png")
    plt.show()

if __name__ == "__main__":
    task_1_square_wave()
    task_2_triangle_wave()
    task_3_fft()
