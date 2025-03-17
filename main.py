import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

## Zadanie 1 ##

def euler_method(x0, r, K, E, dt, T):
    N = int(T / dt)
    t_vals = np.linspace(0, T, N)
    x_vals = np.zeros(N)
    x_vals[0] = x0
    
    for i in range(1, N):
        x_vals[i] = x_vals[i-1] + dt * (r * x_vals[i-1] * (1 - x_vals[i-1] / K) - E * x_vals[i-1])
    
    return t_vals, x_vals

def analytical_solution(x0, r, K, t_vals):
    return (K * x0 * np.exp(r * t_vals)) / (K - x0 + x0 * np.exp(r * t_vals))

# Parametry
r = 10
K = 1e7
dt = 0.001
T = 1
x0 = 5e6

# Obliczenia 
E_values = range(0, 9)
plt.figure(figsize=(10, 6))

for E in E_values:
    t_vals, x_vals = euler_method(x0, r, K, E, dt, T)
    x_analytic = analytical_solution(x0, r, K, t_vals)
    plt.plot(t_vals, x_vals, label=f'Euler E={E}', linewidth=1.5)
    plt.plot(t_vals, x_analytic, linestyle='dashed', linewidth=1.5, label=f'Analityczne E={E}')

# Wykres 1
for E in range(0, 9):
    x_fixed_t = K * (1 - E / r) * np.ones_like(t_vals)
    plt.plot(t_vals, x_fixed_t, linestyle='dotted', linewidth=2, label=f'Asymptota E={E}')

plt.text(0.01 * T, 0.85 * K, f"r = {r}\nK = {K:.0f}\ndt = {dt}\nT = {T}\nx0 = {x0:.0f}", 
         fontsize=10, bbox=dict(facecolor='white', alpha=0.8))

plt.xlabel('Czas')
plt.ylabel('Populacja x(t)')
plt.title('Zadanie 1 x(0)=5000000,K=10000000')
plt.legend()
plt.grid()

def calc_bifurk():
  min_r = 0
  max_r = 4
  step_r = 0.001
  max_iterations = 10000
  skip_iterations = 500
  max_counter = int((max_iterations - skip_iterations) * (max_r - min_r) / step_r)
  result_x = np.zeros(max_counter)
  result_r = np.zeros(max_counter)
  i = 0
  for r in np.arange(min_r, max_r, step_r):
    x = 0.1
    for it in range(max_iterations):
      x = r * x * (1-x)
      if it > skip_iterations:
        result_x[i] = x
        result_r[i] = r
        i += 1
  result_x = result_x[result_r != 0].copy()
  result_r = result_r[result_r != 0].copy()
  return result_x, result_r

result_x, result_r = calc_bifurk()

# Wykres 2 
plt.figure(figsize=(5, 3), dpi=200)
plt.plot(result_r, result_x, ",", color='k')
plt.title('Zadanie 1 Porównanie metod numerycznych, analitycznych i asymptot dla różnych wartości E')
plt.show()

## Zadanie 2 ##

P = 0.01   # Roczna wpłata
r = 0.05   # Roczna stopa oprocentowania (5%)
target = 10**6  # Cel: 1 milion

# Obliczenie
T = np.log((target * r / P) + 1) / r
print(f"Czas potrzebny na zgromadzenie 1 miliona: {T:.2f} lat")

## Zadanie 3 ##

x = np.array(range(1790, 2000, 10))
y = np.array([3900000, 5300000, 7200000, 9600000, 12900000, 17100000, 23100000, 31400000, 38600000, 50200000, 62900000, 76000000, 92000000, 105700000, 122800000, 131700000, 150700000, 179000000, 205000000, 226500000, 248700000])

plt.scatter(x, y)
plt.xlabel("Year")
plt.ylabel("Population")
plt.grid(True)


def exponential_func(t, x_0, r):
    return x_0*np.exp(r*t)

popt, _ = curve_fit(exponential_func, x, y, p0=(1, 1e-8))

y_fit = exponential_func(x, *popt)

# Plot the dataset and the fitted curve
plt.scatter(x, y, label="Data points", color="blue")
plt.plot(x, y_fit, label=f"Best fit: x(t) = {popt[0]:.5f} * e^({popt[1]:.5f}t)", color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Exponential Fit")
plt.legend()
plt.grid()
plt.show()

# Print the fitted parameters
print(f"Best fit equation: x(t) = {popt[0]:.5f} * e^({popt[1]:.5f}*t)")

## Zadanie 4 ##

def mandelbrot(c, max_iter):
    """Oblicza liczbę iteracji zanim z przekroczy próg dla danej liczby zespolonej c."""
    z = 0
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z**2 - c
    return max_iter

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """Generuje zbiór Mandelbrota na siatce kompleksowych liczb."""
    x = np.linspace(xmin, xmax, width)
    y = np.linspace(ymin, ymax, height)
    fractal = np.zeros((height, width))
    
    for i in range(height):
        for j in range(width):
            fractal[i, j] = mandelbrot(complex(x[j], y[i]), max_iter)
    
    return fractal

# Parametry obrazu
xmin, xmax, ymin, ymax = -2, 1, -1.5, 1.5
width, height = 800, 800
max_iter = 100

# Generowanie zbioru Mandelbrota
fractal = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

# Rysowanie
plt.figure(figsize=(10, 10))
plt.imshow(fractal, extent=(xmin, xmax, ymin, ymax), cmap='inferno')
plt.colorbar(label='Liczba iteracji')
plt.title('Zbiór Mandelbrota')
plt.xlabel('Re(c)')
plt.ylabel('Im(c)')
plt.show()
