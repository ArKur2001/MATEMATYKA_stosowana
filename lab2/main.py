import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.special import jn

#ZADANIE_1#

import numpy as np
import matplotlib.pyplot as plt

r = 0 
F0 = 1
omega = 7 
omega0 = 5
u0 = 1
x0 = 1

def forced_damped_oscillator(t, y):
    x, v = y
    dxdt = v
    dvdt = (F0 * np.cos(omega * t) - r * v / omega - omega0 * omega0 * x)
    return np.array([dxdt, dvdt])

def euler_method(f, y0, t0, tf, dt):
    t = np.arange(t0, tf, dt)
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(1, len(t)):
        y[i] = y[i-1] + dt * f(t[i-1], y[i-1])
    return t, y

def analytic_solution(t):
    # Homogeneous solution
    x_h = x0 * np.cos(omega0 * t) + (u0 * np.sin(omega0 * t)) / omega0

    if omega0 != omega:
        # Particular solution
        x_p = (F0 * np.cos(omega * t) - np.cos(omega0 * t) ) / (omega0**2 - omega**2)
    else:
        x_p = -(np.cos(omega0 * t) * (F0 - 1))
    
    return x_h + x_p, x_h, x_p

y0 = [0.0, 0.0]
t0 = 0.0
tf = 20 
dt = 0.001 

t, y_numerical = euler_method(forced_damped_oscillator, y0, t0, tf, dt)

x_analytic, x_h, x_p = analytic_solution(t)

error = np.sum(np.abs(y_numerical[:, 0] - x_analytic))

# Create a figure with three subplots
fig, axs = plt.subplots(2, 2, figsize=(15, 10))

# Plot the numerical and analytic solutions
axs[0, 0].plot(t, y_numerical[:, 0], label='Numerical Solution')
axs[0, 0].plot(t, x_analytic, label='Analytic Solution', linestyle='dashed')
axs[0, 0].set_xlabel('t')
axs[0, 0].set_ylabel('x(t)')
axs[0, 0].set_title(f'x0={x0}, u0={u0}, r={r}, F0={F0}, ω={omega}, ω0={omega0}')
axs[0, 0].legend()
axs[0, 0].grid(True)
axs[0, 0].text(0.25, 0.95, f'Total error: {error:.2e}', transform=axs[0, 0].transAxes, fontsize=12,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Plot homogeneous and particular solutions
axs[1, 0].plot(t, x_h, label='Homogeneous Solution')
axs[1, 0].plot(t, x_p, label='Particular Solution', linestyle='dashed')
axs[1, 0].set_xlabel('t')
axs[1, 0].set_ylabel('x(t)')
axs[1, 0].set_title('Homogeneous and Particular Solutions')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Plot the steady-state variation of amplitude
relative_frequencies = np.linspace(0.1, 2.0, 500)
r_values = [0.1] + list(range(1, 21))
amplitudes = {r: [] for r in r_values}

for r in r_values:
    for omega_rel in relative_frequencies:
        omega = omega_rel * omega0
        if omega0 != omega:
            A = F0 / np.sqrt((omega0**2 - omega**2)**2 + (r * omega)**2)
        else:
            A = F0 / (r * omega)
        amplitudes[r].append(A)

for r in r_values:
    axs[0, 1].plot(relative_frequencies, amplitudes[r], label=f'r = {r}')
axs[0, 1].set_xlabel('Relative Frequency (ω/ω0)')
axs[0, 1].set_ylabel('Amplitude')
axs[0, 1].set_title('Steady-State Amplitude vs. Relative Frequency for Different r Values')
axs[0, 1].legend()
axs[0, 1].grid(True)

axs[1, 1].axis('off')

plt.tight_layout()
plt.show()

print(f"Total error: {error}")

#ZADANIE_2#

print("ZADANIE 2")

T_max = 100
dt = 0.01

ro = 10
r = 28
b = 8/3

x = np.zeros(int(T_max/dt))
y = np.zeros(int(T_max/dt))
z = np.zeros(int(T_max/dt))

x[0] = 10
y[0] = 10
z[0] = 30

def x_n(x, y, n):
    return x[n-1] + dt * ro * (y[n-1] - x[n-1])

def y_n(x, y, z, n):
    return y[n-1] + dt * (((r - z[n-1]) * x[n-1]) - y[n-1])

def z_n(x, y, z, n):
    return z[n-1] + dt * (x[n-1] * y[n-1] - b * z[n-1])

for n in range(1, int(T_max/dt)):
    x[n] = x_n(x, y, n)
    y[n] = y_n(x, y, z, n)
    z[n] = z_n(x, y, z, n)


fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(3, 2)

ax = fig.add_subplot(gs[:,1], projection='3d')
ax.set_title(f'Metoda Eulera, T_max = {T_max}, dt = {dt}, x_0={int(x[0]), int(y[0]), int(z[0])}')
ax.plot(x, y, z)

bx = fig.add_subplot(gs[0,0])
bx.set_title(f'Metoda Eulera, dt={dt}, T_max={T_max}')
bx.plot(range(0, int(T_max/dt)), x)

by = fig.add_subplot(gs[1,0])
by.plot(range(0, int(T_max/dt)), y)

bz = fig.add_subplot(gs[2,0])
bz.plot(range(0, int(T_max/dt)), z)

plt.show()

#ZADANIE_3#

print("ZADANIE 3")

# Definicja równania Bessela
def bessel_ode(x, Y, n):
    y, dy = Y
    d2y = -(x * dy + (x**2 - n**2) * y) / x**2
    return [dy, d2y]

# Rozwiązanie równania dla n=0 i n=1
x_span = (0.01, 10) 
x_eval = np.linspace(*x_span, 100)

# Warunki początkowe dla funkcji Bessela
init_conditions = { 
    0: [1, 0],  # J_0(0) = 1, J_0'(0) = 0
    1: [0, 1]  # J_1(0) = 0, J_1'(0) = 1
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, n in enumerate([0, 1]):
    # Użycie metody RK45 z odpowiednią tolerancją
    sol = solve_ivp(bessel_ode, x_span, init_conditions[n], t_eval=x_eval, args=(n,), method='RK45', rtol=1e-10, atol=1e-12)
    
    # Wykresy
    axes[i].plot(x_eval, sol.y[0], label=f'Numeryczne J_{n}(x)')
    axes[i].plot(x_eval, jn(n, x_eval), '--', label=f'Analityczne J_{n}(x)')
    axes[i].set_xlabel("x")
    axes[i].set_ylabel(f"J_{n}(x)")
    axes[i].set_title(f"Funkcja Bessela J_{n}(x)")
    axes[i].legend()
    axes[i].grid()

plt.tight_layout()
plt.show()