import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

# Parameters
L = 1.0  # Length of the string
c = 1.0  # Wave speed
T = 2.0  # Total time for simulation
dx = 0.01  # Spatial step
dt = 0.01  # Time step
x = np.arange(0, L + dx, dx)
t = np.arange(0, T + dt, dt)

# Initial condition functions
def f1(x, L):
    return np.abs(np.sin(np.pi * x / L))

def f2(x, L):
    return 2 * np.abs(np.abs(x - L / 2) - L / 2)

# Compute b_n coefficients
def compute_bn(fx, n, L):
    integrand = lambda x: fx(x, L) * np.sin(n * np.pi * x / L)
    return (2 / L) * np.trapezoid(integrand(x), x)

# Compute u(x, t)
def compute_u(x, t, L, c, fx, num_terms=50):
    u = np.zeros((len(t), len(x)))
    for n in range(1, num_terms + 1):
        bn = compute_bn(fx, n, L)
        u += bn * np.sin(n * np.pi * x / L)[None, :] * np.cos(n * np.pi * c * t / L)[:, None]
    return u

# Select initial condition
fx = f1  # Change to f2 for the second example

# Compute the solution
u = compute_u(x, t, L, c, fx)

# Update parameters
c = 10.0  # Wave speed
dx = 0.001  # Spatial step
x = np.arange(0, L + dx, dx)
t = np.arange(0, 0.5 + 0.005, 0.005)  # Time range from 0 to 0.5 with step 0.005

# Compute the solution for updated parameters
u = compute_u(x, t, L, c, fx)

# Visualization for updated parameters (2D plot with all lines and 3D surface plot)
fig = plt.figure(figsize=(14, 6))

# 2D plot
ax1 = fig.add_subplot(1, 2, 1)
for i, ti in enumerate(t):  # Iterate through all time steps
    ax1.plot(x, u[i, :], label=f't={ti:.3f}')
ax1.set_xlabel('x')
ax1.set_ylabel('u(x, t)')
ax1.set_title('2D Plot: Wave Equation Solution\n(c=10, L=1, dx=0.001)')
ax1.legend(loc='upper right', fontsize='small', ncol=1)  # Add legend
ax1.grid()

# 3D surface plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
X, T = np.meshgrid(x, t)
ax2.plot_surface(X, T, u, cmap='viridis', edgecolor='none')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x, t)')
ax2.set_title('3D Surface Plot: Wave Equation Solution\n(c=10, L=1, dx=0.001)')

plt.tight_layout()
plt.show()

# Compute the solution for fx = 2 * abs(abs(x - L/2) - L/2)
fx = f2  # Update initial condition
u_f2 = compute_u(x, t, L, c, fx)

# Visualization for fx = 2 * abs(abs(x - L/2) - L/2) (2D plot with all lines and 3D surface plot)
fig = plt.figure(figsize=(14, 6))

# 2D plot
ax1 = fig.add_subplot(1, 2, 1)
for i, ti in enumerate(t):  # Iterate through all time steps
    ax1.plot(x, u_f2[i, :], label=f't={ti:.3f}')
ax1.set_xlabel('x')
ax1.set_ylabel('u(x, t)')
ax1.set_title('2D Plot: Wave Equation Solution (fx = 2 * |x - L/2|)\n(c=10, L=1, dx=0.001)')
ax1.legend(loc='upper right', fontsize='small', ncol=1)  # Add legend
ax1.grid()

# 3D surface plot
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
X, T = np.meshgrid(x, t)
ax2.plot_surface(X, T, u_f2, cmap='plasma', edgecolor='none')
ax2.set_xlabel('x')
ax2.set_ylabel('t')
ax2.set_zlabel('u(x, t)')
ax2.set_title('3D Surface Plot: Wave Equation Solution (fx = 2 * |x - L/2|)\n(c=10, L=1, dx=0.001)')

plt.tight_layout()
plt.show()