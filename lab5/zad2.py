import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return x**2 - 2**x

def f_prime(x):
    return 2 * x - 2**x * np.log(2)

# Newton's method implementation
def newton_method(x0, max_iter=100, tol=1e-6):
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        fpx = f_prime(x)
        if abs(fx) < tol:
            break
        x -= fx / fpx
    return x


x_vals = np.linspace(-1, 4, 500)
y_vals = f(x_vals)


plots = [
    (0, -0.7666666),
    (1, 2),
    (3, 4),
    (1, 0.48509),
    (3, 3.2124)
]


fig, axes = plt.subplots(3, 2, figsize=(12, 10))  
axes = axes.flatten()


for i, (x0, solution) in enumerate(plots):
    ax = axes[i]
    ax.plot(x_vals, y_vals, label='f(x)')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.scatter([solution], [f(solution)], color='red', label='Solution', zorder=5)
    ax.set_title(f"f(x) = x^2 - 2^x, x0 = {x0}, x = {solution:.5f}")
    ax.set_xlim(-1, 4)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xticks(np.arange(-1, 5, 1))
    ax.set_yticks(np.arange(-1.5, 1.5, 0.5))
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)


fig.delaxes(axes[-1])  

plt.tight_layout()
plt.show()
