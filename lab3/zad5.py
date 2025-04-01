import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 50, 50
lx, ly = 1.0, 1.0
dx, dy = lx / (nx - 1), ly / (ny - 1)
tolerance = 1e-6

# Initialize the grid with a square in the center
u = np.zeros((ny, nx))
center_x, center_y = nx // 2, ny // 2
square_size = min(nx, ny) // 4  # Size of the square
u[center_y - square_size:center_y + square_size, center_x - square_size:center_x + square_size] = 1

u[0, :] = 0 
u[:, 0] = 0 
u[:, -1] = 0 


def solve_laplace_time(u, dx, dy, tolerance, time_steps):
    results = [(0, u.copy())]  # Include the initial condition for t=0
    error = 1.0
    step = 0
    while error > tolerance and step < max(time_steps):
        u_new = u.copy()
        u_new[1:-1, 1:-1] = 0.25 * (u[1:-1, :-2] + u[1:-1, 2:] + u[:-2, 1:-1] + u[2:, 1:-1])
        error = np.max(np.abs(u_new - u))
        u = u_new
        step += 1
        if step in time_steps:
            results.append((step, u.copy()))
    return results

# Time steps
time_steps = [0, 1, 5, 10, 20, 30, 130, 170, 500]

results = solve_laplace_time(u, dx, dy, tolerance, time_steps)

while len(results) < 9:
    results.append((None, np.zeros_like(u))) 

# Visualization
x = np.linspace(0, lx, nx)
y = np.linspace(0, ly, ny)
X, Y = np.meshgrid(x, y)

fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for ax, (step, u_snapshot) in zip(axes.flat, results):
    if step is not None:
        contour = ax.contourf(X, Y, u_snapshot, 50, cmap='viridis')
        ax.set_title(f't={step * tolerance:.6f}')
    else:
        ax.set_title('Empty')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    cbar = plt.colorbar(contour, ax=ax, orientation='vertical')
    cbar.set_label('Potential')

plt.tight_layout(rect=[0, 0, 1, 0.96]) 
plt.show()
