import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def interactive_wave_interference():
    """
    Interactive Visualization of Wave Superposition and Interference.
    Two waves: y1 = A1 * sin(k1*x - omega1*t + phi1)
               y2 = A2 * sin(k2*x - omega2*t + phi2)
    Users can adjust Phase and Frequency of the second wave.
    """
    
    # Initialize parameters
    x = np.linspace(0, 10, 1000)
    a1 = 1.0
    f1 = 1.0
    phi1 = 0
    
    a2 = 1.0
    f2 = 1.0
    phi2 = 0
    
    # Create the figure and the line that we will manipulate
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    plt.subplots_adjust(bottom=0.25)
    
    y1 = a1 * np.sin(2 * np.pi * f1 * x + phi1)
    y2 = a2 * np.sin(2 * np.pi * f2 * x + phi2)
    y_sum = y1 + y2
    
    line1, = ax1.plot(x, y1, label='Wave 1', alpha=0.5)
    line2, = ax1.plot(x, y2, label='Wave 2', alpha=0.5)
    line_sum, = ax2.plot(x, y_sum, lw=2, color='red', label='Superposition')
    
    ax1.set_title("Individual Waves")
    ax1.legend()
    ax1.grid(True)
    ax1.set_ylim(-2.2, 2.2)
    
    ax2.set_title("Resultant Wave (Interference)")
    ax2.set_xlabel("Position x")
    ax2.legend()
    ax2.grid(True)
    ax2.set_ylim(-4.2, 4.2)
    
    # Define Sliders
    ax_freq = plt.axes([0.25, 0.1, 0.65, 0.03])
    ax_phase = plt.axes([0.25, 0.05, 0.65, 0.03])
    
    s_freq = Slider(ax_freq, 'Freq 2', 0.1, 5.0, valinit=f2)
    s_phase = Slider(ax_phase, 'Phase 2', 0, 2*np.pi, valinit=phi2)
    
    def update(val):
        f2_new = s_freq.val
        phi2_new = s_phase.val
        
        y2_new = a2 * np.sin(2 * np.pi * f2_new * x + phi2_new)
        y_sum_new = y1 + y2_new
        
        line2.set_ydata(y2_new)
        line_sum.set_ydata(y_sum_new)
        fig.canvas.draw_idle()
        
    s_freq.on_changed(update)
    s_phase.on_changed(update)
    
    print("--------------------------------------------------")
    print("使用说明 (Usage Instructions):")
    print("1. 调节 Freq 2: 观察当 Freq 2 趋近于 1.0 时出现的 '拍' (Beats) 现象。")
    print("2. 调节 Phase 2: 设为 3.14 (PI) 观察相消干涉 (Destructive Interference)。")
    print("3. 调节 Phase 2: 设为 0 或 6.28 观察相长干涉 (Constructive Interference)。")
    print("4. 关闭窗口即可保存当前视图为 wave_interference.png。")
    print("--------------------------------------------------")
    
    plt.savefig("wave_interference.png")
    plt.show()

if __name__ == "__main__":
    interactive_wave_interference()
