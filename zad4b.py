import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def lpf_impulse_response(n, wc):
    """Generate the ideal low-pass filter impulse response"""
    h = np.zeros_like(n, dtype=float)
    h[n == 0] = wc / np.pi
    mask = n != 0
    h[mask] = np.sin(wc * n[mask]) / (np.pi * n[mask])
    return h

def lms_filter(x, d, filter_order, mu):
    """LMS adaptive filter implementation"""
    N = len(x)
    h = np.zeros(filter_order)
    y = np.zeros(N)
    e = np.zeros(N)
    h_history = np.zeros((N, filter_order))
    
    for n in range(filter_order, N):
        x_window = x[n:n-filter_order:-1]
        y[n] = np.dot(h, x_window)
        e[n] = d[n] - y[n]
        h += mu * e[n] * x_window
        h_history[n] = h
        
    return y, e, h_history

def rls_filter(x, d, filter_order, delta=0.01, lambda_=0.99):
    """RLS adaptive filter implementation"""
    N = len(x)
    h = np.zeros(filter_order)
    P = np.eye(filter_order) / delta
    y = np.zeros(N)
    e = np.zeros(N)
    h_history = np.zeros((N, filter_order))
    
    for n in range(filter_order, N):
        x_window = x[n:n-filter_order:-1]
        y[n] = np.dot(h, x_window)
        e[n] = d[n] - y[n]
        
        # RLS update equations
        k = np.dot(P, x_window) / (lambda_ + np.dot(x_window, np.dot(P, x_window)))
        h += k * e[n]
        P = (P - np.outer(k, np.dot(x_window, P))) / lambda_
        
        h_history[n] = h
        
    return y, e, h_history

# Simulation parameters
np.random.seed(42)
N = 2000  # Number of samples
M = 50    # Filter order (should be 2M+1 for full symmetric response)
wc1 = np.pi/2  # Initial cutoff frequency
wc2 = 1.2*np.pi/2  # Changed cutoff frequency
change_point = N//2  # Point where cutoff frequency changes

# Generate time indices for the ideal filter
n_ideal = np.arange(-M, M+1)

# Generate the ideal filter responses
h_ideal1 = lpf_impulse_response(n_ideal, wc1)
h_ideal2 = lpf_impulse_response(n_ideal, wc2)

# Generate input signal (Gaussian white noise)
x = np.random.normal(0, 1, N)

# Generate desired signal (filtered input + noise)
d1 = np.convolve(x, h_ideal1, mode='same')
d2 = np.convolve(x, h_ideal2, mode='same')
d = np.concatenate((d1[:change_point], d2[change_point:]))
d += np.random.normal(0, 0.1, N)  # Add measurement noise

# Adaptive filter parameters
filter_order = 2*M + 1
mu = 0.01  # LMS step size
delta = 0.01  # RLS initialization parameter
lambda_ = 0.99  # RLS forgetting factor

# Run adaptive filters
y_lms, e_lms, h_lms_history = lms_filter(x, d, filter_order, mu)
y_rls, e_rls, h_rls_history = rls_filter(x, d, filter_order, delta, lambda_)

# Plot results
plt.figure(figsize=(15, 10))

# Plot impulse response adaptation
plt.subplot(3, 1, 1)
plt.plot(n_ideal, h_ideal1, 'k--', linewidth=2, label='Ideal (1st half)')
plt.plot(n_ideal, h_ideal2, 'r--', linewidth=2, label='Ideal (2nd half)')
plt.plot(n_ideal, h_lms_history[-1], 'b-', label='LMS final estimate')
plt.plot(n_ideal, h_rls_history[-1], 'g-', label='RLS final estimate')
plt.title('Impulse Response Comparison')
plt.xlabel('Time index n')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# Plot error convergence
plt.subplot(3, 1, 2)
plt.semilogy(e_lms**2, 'b-', label='LMS error')
plt.semilogy(e_rls**2, 'g-', label='RLS error')
plt.axvline(change_point, color='r', linestyle='--', label='Cutoff change')
plt.title('Error Convergence')
plt.xlabel('Time index n')
plt.ylabel('Squared error (log scale)')
plt.legend()
plt.grid(True)

# Plot filter coefficient evolution
plt.subplot(3, 1, 3)
plt.imshow(h_lms_history.T, aspect='auto', cmap='jet', 
           extent=[0, N, -M, M], vmin=-0.2, vmax=0.2)
plt.colorbar(label='Coefficient value')
plt.axvline(change_point, color='w', linestyle='--')
plt.title('LMS Coefficient Adaptation Over Time')
plt.xlabel('Time index n')
plt.ylabel('Coefficient index')
plt.tight_layout()
plt.show()

# Animation function (optional - uncomment to use)
def animate_coefficients(h_history, ideal1, ideal2, change_point):
    from matplotlib.animation import FuncAnimation
    
    fig, ax = plt.subplots(figsize=(10, 6))
    n_coeff = np.arange(-M, M+1)
    line_est, = ax.plot(n_coeff, h_history[0], 'b-', label='Estimated')
    line_ideal1, = ax.plot(n_coeff, ideal1, 'k--', label='Ideal (1st half)')
    line_ideal2, = ax.plot(n_coeff, ideal2, 'r--', label='Ideal (2nd half)')
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlabel('Coefficient index')
    ax.set_ylabel('Value')
    ax.set_title('Adaptive Filter Coefficient Evolution')
    ax.legend()
    ax.grid(True)
    
    def update(frame):
        line_est.set_ydata(h_history[frame*10])
        return line_est,
    
    ani = FuncAnimation(fig, update, frames=len(h_history)//10, 
                        interval=50, blit=True)
    plt.close()
    return ani

# Uncomment to create animations
# ani_lms = animate_coefficients(h_lms_history, h_ideal1, h_ideal2, change_point)
# ani_rls = animate_coefficients(h_rls_history, h_ideal1, h_ideal2, change_point)