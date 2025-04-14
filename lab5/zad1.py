import numpy as np
import matplotlib.pyplot as plt

# Funkcja logistyczna
def logistic_map(x, lam):
    return lam * x * (1 - x)

# Parametry iteracji
N_iter = 1000
x0 = 0.1
lambdas = [2.5, 1.5, 3.2, 3.9]

# Przygotowanie wykresów
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

x = np.linspace(0, 1, 500)

for i, lam in enumerate(lambdas):
    ax = axes[i]
    
    ax.plot(x, logistic_map(x, lam), 'k')  # wykres funkcji
    ax.plot(x, x, 'k--')                   # przekątna y = x

    xn = x0
    for _ in range(N_iter):
        x_next = logistic_map(xn, lam)
        # pionowa linia
        ax.plot([xn, xn], [xn, x_next], 'b', linewidth=0.5)
        # pozioma linia
        ax.plot([xn, x_next], [x_next, x_next], 'b', linewidth=0.5)
        xn = x_next

    ax.set_title(f'λ={lam}, x0={x0}, N_$iter$={N_iter}')
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)

plt.tight_layout()
plt.show()
