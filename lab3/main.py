import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import trapezoid

#ZAD1

# Definicja parametrów
L = 1.0      # Długość rury
D = 0.25     # Współczynnik dyfuzji
N = 100      # Liczba składników w szeregu Fouriera
x = np.arange(0, L+0.01, 0.01)  # Punkty przestrzenne z krokiem dx=0.01
t_values = [0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 0.040, 0.059, 0.085, 0.122, 0.174, 0.247, 0.351, 0.498, 0.706, 1.000]  # Nowe wartości czasowe

def f1(x):
    return np.abs(np.sin(3 * np.pi * x / L))  # Pierwsza funkcja początkowa

def f2(x):
    return 2 * np.abs(np.abs(x - L/2) - L/2)  # Druga funkcja początkowa

def f3(x):
    return np.where((x >= 0.4) & (x <= 0.6), 1, 0)  # Prostokąt o amplitudzie 1

def compute_bn(n, f):
    """ Oblicza współczynniki Fouriera bn dla danej funkcji f """
    return (2 / L) * trapezoid(f(x) * np.sin(n * np.pi * x / L), x)

def u_xt(x, t, f):
    """ Oblicza rozwinięcie szeregu Fouriera dla danej funkcji f """
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
    ax.set_title(f'Rozwiązanie {title}')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()  # Wyświetlenie wykresu 2D

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
    plt.tight_layout()
    plt.show()  # Wyświetlenie wykresu 3D

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
    plt.show()  # Wyświetlenie wykresu współczynników Fouriera

# Generowanie wykresów
plot_results(f1, 'dla fx = |sin(3πx/L)|')
plot_results(f2, 'dla fx = 2| |x-L/2| - L/2 |')
plot_results(f3, 'dla fx = Prostokąt o amplitudzie 1 (0.4 ≤ x ≤ 0.6)')

#ZAD2

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid

# Parametry
L = 1.0      # Długość rury
D = 0.25     # Współczynnik dyfuzji
N = 100      # Liczba składników w szeregu Fouriera
x = np.arange(0, L+0.01, 0.01)  # Punkty przestrzenne
t_values = [0.000, 0.002, 0.004, 0.007, 0.012, 0.018, 0.027, 0.040, 0.059, 0.085, 0.122, 0.174, 0.247, 0.351, 0.498, 0.706, 1.000]  # Punkty czasowe

# Funkcje początkowe
def f1(x):
    return np.abs(np.sin(3 * np.pi * x / L))  # Funkcja sinusoidalna

def f2(x):
    return 2 * np.abs(np.abs(x - L/2) - L/2)  # Funkcja prostokątna

def f3(x):
    return np.where((x >= 0.4) & (x <= 0.6), 1, 0)  # Prostokąt o amplitudzie 1

# Obliczanie współczynników Fouriera a_n (kosinus)
def compute_an(n, f):
    """ Oblicza współczynniki Fouriera a_n dla danej funkcji f z użyciem cosinusów """
    return (2 / L) * trapezoid(f(x) * np.cos(n * np.pi * x / L), x)

# Obliczanie rozwiązania równania dyfuzji
def u_xt(x, t, f):
    """ Oblicza rozwinięcie szeregu Fouriera dla funkcji f """
    sum_series = np.zeros_like(x)
    
    # Obliczamy a_0 (średnia funkcji f(x))
    a_0 = (2 / L) * trapezoid(f(x), x)
    sum_series += a_0 / 2
    
    for n in range(1, N+1):
        an = compute_an(n, f)
        sum_series += an * np.cos(n * np.pi * x / L) * np.exp(-n**2 * np.pi**2 * D * t / L**2)
        
    return sum_series

# Funkcja do rysowania wykresów
def plot_results(f, title):
    # Wykres 2D
    fig, ax = plt.subplots(figsize=(10, 5))
    for t in t_values:
        ax.plot(x, u_xt(x, t, f), label=f't={t:.3f}')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')
    ax.set_title(f'Rozwiązanie {title}')
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.show()  # Wyświetlenie wykresu 2D

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
    plt.tight_layout()
    plt.show()  # Wyświetlenie wykresu 3D

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
    plt.show()  # Wyświetlenie wykresu współczynników Fouriera

# Generowanie wykresów dla różnych funkcji początkowych
plot_results(f1, 'dla fx = |sin(3πx/L)|')
plot_results(f2, 'dla fx = 2| |x-L/2| - L/2 |')
plot_results(f3, 'dla fx = Prostokąt o amplitudzie 1 (0.4 ≤ x ≤ 0.6)')

#ZAD3


