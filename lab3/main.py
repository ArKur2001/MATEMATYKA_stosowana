import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import trapezoid

#ZAD1

# Definicja parametrów
L = 1.0      
D = 0.25     
N = 100      
x = np.arange(0, L+0.01, 0.01)  
t_values = [0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 0.040, 0.059, 0.085, 0.122, 0.174, 0.247, 0.351, 0.498, 0.706, 1.000] 

def f1(x):
    return np.abs(np.sin(3 * np.pi * x / L))  

def f2(x):
    return 2 * np.abs(np.abs(x - L/2) - L/2)  

def f3(x):
    return np.where((x >= 0.4) & (x <= 0.6), 1, 0) 

def compute_bn(n, f):
    return (2 / L) * trapezoid(f(x) * np.sin(n * np.pi * x / L), x)

def u_xt(x, t, f):
    sum_series = np.zeros_like(x)
    for n in range(1, N+1):
        bn = compute_bn(n, f)
        sum_series += bn * np.sin(n * np.pi * x / L) * np.exp(-n**2 * np.pi**2 * D * t / L**2)
    return sum_series

def plot_results(f, title):
    # Wykres 2D
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in t_values:
        ax.plot(x, u_xt(x, t, f), label=f't={t:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f'2D Rozwiązanie {title}')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()  

    # Wykres 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x, t_values)
    U = np.array([u_xt(x, t, f) for t in t_values])
    ax.plot_surface(X, T, U, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title(f'3D Rozwiązanie {title}')
    ax.set_xlim(np.max(x), np.min(x)) 
    plt.tight_layout()
    plt.show()  

    # Wykres współczynników Fouriera
    fig, ax = plt.subplots(figsize=(10, 5))
    n_values = np.arange(1, N+1)
    bn_values = np.array([compute_bn(n, f) for n in n_values])
    ax.stem(n_values, bn_values)
    ax.set_xlabel('n')
    ax.set_ylabel('b_n')
    ax.set_title(f'Współczynniki Fouriera {title}')
    ax.grid()
    plt.tight_layout()
    plt.show()  

# Generowanie wykresów
plot_results(f1, 'dla fx = |sin(3πx/L)|')
plot_results(f2, 'dla fx = 2| |x-L/2| - L/2 |')
plot_results(f3, 'dla fx = Prostokąt (0.4 ≤ x ≤ 0.6)')

#ZAD2

# Parametry
L = 1.0      
D = 0.25     
N = 100      
x = np.arange(0, L+0.01, 0.01)  
t_values = [0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 0.040, 0.059, 0.085, 0.122, 0.174, 0.247, 0.351, 0.498, 0.706, 1.000]  

def compute_an(n, f):
    return (2 / L) * trapezoid(f(x) * np.cos(n * np.pi * x / L), x)

def u_xt(x, t, f):
    sum_series = np.zeros_like(x)
    
    a_0 = (2 / L) * trapezoid(f(x), x)
    sum_series += a_0 / 2
    
    for n in range(1, N+1):
        an = compute_an(n, f)
        sum_series += an * np.cos(n * np.pi * x / L) * np.exp(-n**2 * np.pi**2 * D * t / L**2)
        
    return sum_series

def plot_results(f, title):
    # Wykres 2D
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in t_values:
        ax.plot(x, u_xt(x, t, f), label=f't={t:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f'2D Rozwiązanie {title}')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()  

    # Wykres 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x, t_values)
    U = np.array([u_xt(x, t, f) for t in t_values])
    ax.plot_surface(X, T, U, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title(f'3D Rozwiązanie {title}')
    ax.set_xlim(np.max(x), np.min(x)) 
    plt.tight_layout()
    plt.show()  

    # Wykres współczynników Fouriera
    fig, ax = plt.subplots(figsize=(10, 5))
    n_values = np.arange(1, N+1)
    an_values = np.array([compute_an(n, f) for n in n_values])
    ax.stem(n_values, an_values)
    ax.set_xlabel('n')
    ax.set_ylabel('a_n')
    ax.set_title(f'Współczynniki Fouriera {title}')
    ax.grid()
    plt.tight_layout()
    plt.show() 

# Generowanie wykresów 
plot_results(f1, 'dla fx = |sin(3πx/L)|')
plot_results(f2, 'dla fx = 2| |x-L/2| - L/2 |')
plot_results(f3, 'dla fx = Prostokąt (0.4 ≤ x ≤ 0.6)')

#ZAD3

def f1(x, L):
    return np.abs(np.sin(3 * np.pi * x / L))

def f2(x, L):
    return 2 * np.abs(np.abs(x - L/2) - L/2)

def f3(x, L):
    return np.where((x >= 0.4 * L) & (x <= 0.6 * L), 1, 0)

L = 1
D = 0.25
dx = 0.01
dt = 0.0001  
x = np.arange(0, L + dx, dx)
t_values = [0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 0.040, 0.059, 
            0.085, 0.122, 0.174, 0.247, 0.351, 0.498, 0.706, 1.000]
C1_C2_cases = [(0.6, 0.1), (0.2, 0.7), (0.1, 0.4)]

def solve_diffusion(f, C1, C2, L, D, dx, dt, N):
    u = f(x, L)
    v = C1 + (C2 - C1) * x / L
    u = u + v 
    r = D * dt / dx**2
    
    for _ in range(N):
        u_new = u.copy()
        u_new[1:-1] = u[1:-1] + r * (u[2:] - 2*u[1:-1] + u[:-2])
        u_new[0] = C1
        u_new[-1] = C2
        u = u_new
    
    return u

# Funkcja do rysowania wykresów 2D i 3D
def plot_results(f, title, C1, C2, L, D, dx, dt, t_values, x):
    # Wykres 2D
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in t_values:
        N_t = int(t / dt)
        u_sol = solve_diffusion(f, C1, C2, L, D, dx, dt, N_t)
        ax.plot(x, u_sol, label=f't={t:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f'2D Rozwiązanie {title}')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()  # Wyświetlenie wykresu 2D

    # Wykres 3D
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    X, T = np.meshgrid(x, t_values)
    U = np.array([solve_diffusion(f, C1, C2, L, D, dx, dt, int(t/dt)) for t in t_values])
    ax.plot_surface(X, T, U, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('u(x,t)')
    ax.set_title(f'3D Rozwiązanie {title}')
    ax.set_xlim(np.max(x), np.min(x))  # Odwrócenie osi X
    plt.tight_layout()
    plt.show()  # Wyświetlenie wykresu 3D

# Generowanie wykresów 
plot_results(f1, 'dla fx = |sin(3πx/L)|, C1 = 0.6, C2 = 0.1', 0.6, 0.1, L, D, dx, dt, t_values, x)
plot_results(f2, 'dla fx = 2| |x-L/2| - L/2 |, C1 = 0.2, C2 = 0.7', 0.2, 0.7, L, D, dx, dt, t_values, x)
plot_results(f3, 'dla fx = Prostokąt (0.4 ≤ x ≤ 0.6), C1 = 0.1, C2 = 0.4', 0.1, 0.4, L, D, dx, dt, t_values, x)
