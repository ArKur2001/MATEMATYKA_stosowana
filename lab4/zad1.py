import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad

# Funkcja e^x
def f(x):
    return np.exp(x)

# Przybliżenie Taylora dla e^x (n = 5)
def taylor_approximation(x, n):
    approximation = np.zeros_like(x)
    for i in range(n+1):
        approximation += (x**i) / math.factorial(i)
    return approximation

# Aproksymacja szeregiem Fouriera
def fourier_series_approximation(func, n_terms, samples=1000, interval=(-np.pi, np.pi)):
    a, b = interval
    L = (b - a) / 2
    x_values = np.linspace(a, b, samples)

    a0 = (1/L) * quad(lambda x: func(x), a, b)[0]

    an = []
    bn = []
    for n in range(1, n_terms+1):
        an_coeff = (1/L) * quad(lambda x: func(x) * np.cos(n*np.pi*x/L), a, b)[0]
        bn_coeff = (1/L) * quad(lambda x: func(x) * np.sin(n*np.pi*x/L), a, b)[0]
        an.append(an_coeff)
        bn.append(bn_coeff)

    approximation = a0/2 * np.ones_like(x_values)
    for n in range(1, n_terms+1):
        approximation += an[n-1] * np.cos(n*np.pi*x_values/L) + bn[n-1] * np.sin(n*np.pi*x_values/L)

    return x_values, approximation

# Wielomiany Czebyszewa
def chebyshev_polynomials(x, degree):
    T = np.zeros((degree + 1, len(x)))
    T[0] = np.ones_like(x)
    if degree > 0:
        T[1] = x
    for n in range(2, degree + 1):
        T[n] = 2 * x * T[n - 1] - T[n - 2]
    return T

# WLS z Czebyszewem
def wls_approximation_chebyshev(x, y, degree):
    T = chebyshev_polynomials(x, degree).T
    W = np.diag(np.ones_like(x))
    A = T.T @ W @ T
    b = T.T @ W @ y
    coeffs = np.linalg.solve(A, b)
    y_approx = T @ coeffs
    return y_approx

# WLS z klasycznymi wielomianami
def wls_approximation_polynomial(x, y, degree):
    X = np.vander(x, degree + 1, increasing=True)
    W = np.diag(np.ones_like(x))
    A = X.T @ W @ X
    b = X.T @ W @ y
    coeffs = np.linalg.solve(A, b)
    y_approx = X @ coeffs
    return y_approx


# Funkcje wagowe
def constant_weight(x):
    return np.ones_like(x)

def triangle_weight(x):
    return (1 - np.abs(x)) * 1000

def v_shape_weight(x):
    return 1000 - (1 - np.abs(x)) * 1000

def wls_with_weight(x, y, degree, weight_func, basis='polynomial'):
    W = np.diag(weight_func(x))

    if basis == 'polynomial':
        X = np.vander(x, degree + 1, increasing=True)
    elif basis == 'chebyshev':
        X = chebyshev_polynomials(x, degree).T
    else:
        raise ValueError("basis must be 'polynomial' or 'chebyshev'")

    A = X.T @ W @ X
    b = X.T @ W @ y
    coeffs = np.linalg.solve(A, b)
    y_approx = X @ coeffs
    return y_approx

weights = {
    "Stała": constant_weight,
    "Trójkątna": triangle_weight,
    "Trójkątna odwrotna": v_shape_weight
}

# Zakres x w [-π, π]
x = np.linspace(-np.pi, np.pi, 400)

# Obliczenie funkcji i przybliżeń
y_exact = f(x)
y_taylor = taylor_approximation(x, 5)

# Fourier
n_terms = 10
samples = 1000
a, b = -np.pi, np.pi
x_fourier, y_fourier = fourier_series_approximation(f, n_terms, samples, (a, b))

# WLS – przeskalowanie x do [-1, 1]
x_scaled = x / np.pi
y_scaled = f(x)

# WLS: Chebyshev i klasyczny wielomian
y_wls_chebyshev = wls_approximation_chebyshev(x_scaled, y_scaled, degree=5)
y_wls_poly = wls_approximation_polynomial(x_scaled, y_scaled, degree=5)

# Błędy
error_taylor = np.abs(y_exact - y_taylor)
error_fourier = np.abs(y_exact - np.interp(x, x_fourier, y_fourier))
error_wls_chebyshev = np.abs(y_exact - y_wls_chebyshev)
error_wls_poly = np.abs(y_exact - y_wls_poly)

# Wykres funkcji i aproksymacji
plt.figure(figsize=(10, 6))
plt.plot(x, y_exact, label='e^x', linewidth=2)
plt.plot(x, y_taylor, label="Taylor (n=5)", color="red", linestyle="--", linewidth=2)
plt.plot(x_fourier, y_fourier, label=f"Fourier (n = {n_terms})", color="green", linestyle=":", linewidth=2)
plt.plot(x, y_wls_poly, label="$x^n$ (n=5)", color="orange", linestyle="dashdot", linewidth=2)
plt.plot(x, y_wls_chebyshev, label="Chebyshev (n=5)", color="purple", linestyle="-.", linewidth=2)
plt.title("Funkcja $e^x$ i jej aproksymacje")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()

# --- Wykresy błędów na wspólnym rysunku z 2 subplotami ---
fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# Pełny zakres błędów
axs[0].plot(x, error_taylor, label="Taylor (n=5)", color="red", linestyle="--")
axs[0].plot(x, error_fourier, label="Fourier (n=10)", color="green", linestyle=":")
axs[0].plot(x, error_wls_poly, label="$x^n$ (n=5)", color="orange", linestyle="dashdot")
axs[0].plot(x, error_wls_chebyshev, label="Chebyshev (n=5)", color="purple", linestyle="-.")
axs[0].set_title("Błąd aproksymacji – pełny zakres")
axs[0].set_xlabel("x")
axs[0].set_ylabel("Błąd bezwzględny")
axs[0].legend()
axs[0].grid(True)

# Ograniczony zakres błędów
axs[1].plot(x, error_taylor, label="Taylor (n=5)", color="red", linestyle="--")
axs[1].plot(x, error_fourier, label="Fourier (n=10)", color="green", linestyle=":")
axs[1].plot(x, error_wls_poly, label="$x^n$ (n=5)", color="orange", linestyle="dashdot")
axs[1].plot(x, error_wls_chebyshev, label="Chebyshev (n=5)", color="purple", linestyle="-.")
axs[1].set_title("Błąd aproksymacji – przycięty do [-1, 1]")
axs[1].set_xlabel("x")
axs[1].set_ylim(-1, 1)
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()

# --- Wykresy wielomianów bazowych w jednej figurze z 2 subplotami ---
x_base = np.linspace(-1, 1, 400)
degree = 4

fig, axs = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

# x^n
for i in range(degree + 1):
    axs[0].plot(x_base, x_base**i, label=f"$x^{i}$")
axs[0].set_title("Bazowe wielomiany potęgowe $x^n$")
axs[0].set_xlabel("x")
axs[0].set_ylabel("Wartość")
axs[0].grid(True)

# Czebyszewa
T_vals = chebyshev_polynomials(x_base, degree)
for i in range(degree + 1):
    axs[1].plot(x_base, T_vals[i], label=f"T_{i}(x)")
axs[1].set_title("Wielomiany Czebyszewa")
axs[1].set_xlabel("x")
axs[1].grid(True)

plt.tight_layout()
plt.show()

fig, axs = plt.subplots(len(weights), 2, figsize=(12, 12))

for idx, (name, weight_func) in enumerate(weights.items()):
    w_vals = weight_func(x_scaled)
    
    # Rysuj funkcję wagową
    axs[idx, 0].plot(x_scaled, w_vals)
    axs[idx, 0].set_title(f"Funkcja wagowa: {name}")
    axs[idx, 0].set_xlabel("x")
    axs[idx, 0].set_ylabel("W(x)")
    axs[idx, 0].grid(True)

    # WLS z daną wagą (na wielomianach potęgowych)
    y_wls_weighted = wls_with_weight(x_scaled, y_scaled, degree=5, weight_func=weight_func, basis='polynomial')
    error_weighted = np.abs(y_exact - y_wls_weighted)

    # Wykres błędu
    axs[idx, 1].plot(x, error_weighted, label="WLS z wagą", color="blue")
    axs[idx, 1].plot(x, error_taylor, label="Taylor (n=5)", color="red", linestyle="--")
    axs[idx, 1].plot(x, error_fourier, label="Fourier (n=10)", color="green", linestyle=":")
    axs[idx, 1].plot(x, error_wls_poly, label="$x^n$ (n=5)", color="orange", linestyle="dashdot")
    axs[idx, 1].plot(x, error_wls_chebyshev, label="Chebyshev (n=5)", color="purple", linestyle="-.")
    axs[idx, 1].set_title(f"Funkcja $e^x$ – błąd aproksymacji z wagą: {name}")
    axs[idx, 1].set_xlabel("x")
    axs[idx, 1].set_ylabel("Błąd")
    axs[idx, 1].set_ylim(-1, 1)  
    axs[idx, 1].legend()
    axs[idx, 1].grid(True)

plt.tight_layout()
plt.show()