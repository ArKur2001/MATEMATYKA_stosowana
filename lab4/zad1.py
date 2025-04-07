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

# Funkcja obliczająca aproksymację szeregiem Fouriera
def fourier_series_approximation(func, n_terms, samples=1000, interval=(-np.pi, np.pi)):
    a, b = interval
    L = (b - a) / 2
    x_values = np.linspace(a, b, samples)
    
    # Oblicz współczynniki a0, an, bn
    a0 = (1/L) * quad(lambda x: func(x), a, b)[0]
    
    an = []
    bn = []
    for n in range(1, n_terms+1):
        an_coeff = (1/L) * quad(lambda x: func(x) * np.cos(n*np.pi*x/L), a, b)[0]
        bn_coeff = (1/L) * quad(lambda x: func(x) * np.sin(n*np.pi*x/L), a, b)[0]
        an.append(an_coeff)
        bn.append(bn_coeff)
    
    # Zbuduj aproksymację
    approximation = a0/2 * np.ones_like(x_values)
    for n in range(1, n_terms+1):
        approximation += an[n-1] * np.cos(n*np.pi*x_values/L) + bn[n-1] * np.sin(n*np.pi*x_values/L)
    
    return x_values, approximation

# Przybliżenie za pomocą WLS z Chebyshev
def chebyshev_polynomials(x, degree):
    T = np.zeros((degree + 1, len(x)))
    T[0] = np.ones_like(x)
    T[1] = x
    for n in range(2, degree + 1):
        T[n] = 2 * x * T[n - 1] - T[n - 2]
    return T.T  # Zwracamy macierz wielomianów Czebyszewa

def wls_approximation_chebyshev(x, y, degree):
    # Generowanie wielomianów Czebyszewa
    T = chebyshev_polynomials(x, degree)
    
    # Obliczanie wag, przyjmujemy wagę 1 dla wszystkich punktów
    w = np.ones_like(x)
    
    # Wagi i obliczanie macierzy A
    W = np.diag(w)
    A = T.T @ W @ T
    
    # Wektor b
    b = T.T @ W @ y
    
    # Rozwiązywanie układu równań normalnych A * coeffs = b
    coeffs = np.linalg.solve(A, b)
    
    # Obliczanie dopasowanego wykresu
    y_approx = T @ coeffs
    return y_approx

# Tworzenie wektora wartości x w zakresie [-3, 3]
x = np.linspace(-3, 3, 400)

# Obliczenie wartości funkcji e^x, przybliżenia Taylora i Fouriera
y_exact = f(x)
y_taylor = taylor_approximation(x, 5)

# Parametry aproksymacji Fouriera
n_terms = 10  # Liczba wyrazów w szeregu Fouriera
samples = 1000
a, b = -np.pi, np.pi

# Obliczanie aproksymacji szeregiem Fouriera
x_fourier, y_fourier = fourier_series_approximation(f, n_terms, samples, (a, b))

# Wykorzystanie WLS do przybliżenia funkcji e^x
# Zmieniamy zakres x na [-1, 1] do przybliżenia
x_scaled = 2 * (x - min(x)) / (max(x) - min(x)) - 1  # Przeskalowanie do [-1, 1]
y_scaled = f(x)

# Przybliżenie za pomocą WLS z wielomianem Czebyszewa stopnia 5
y_wls_chebyshev = wls_approximation_chebyshev(x_scaled, y_scaled, degree=5)

# Rysowanie wykresów
plt.figure(figsize=(10, 6))
plt.plot(x, y_exact, label='e^x (dokładna)', linewidth=2)
plt.plot(x, y_taylor, label="Przybliżenie Taylora (n=5)", color="red", linestyle="--", linewidth=2)
plt.plot(x_fourier, y_fourier, label=f"Aproksymacja Fouriera ({n_terms} wyrazów)", color="green", linestyle=":", linewidth=2)
plt.plot(x, y_wls_chebyshev, label="WLS z Chebyshev (stopień=5)", color="purple", linestyle="-.", linewidth=2)
plt.title("Funkcja e^x, Przybliżenie Taylora, Fouriera i WLS z Chebyshev")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
