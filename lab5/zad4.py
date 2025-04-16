import numpy as np
import matplotlib.pyplot as plt

# Parametry
M = 40                  # Długość filtra
N = 50000               # Liczba próbek
sigma_x = 1             # Odchylenie standardowe sygnału x[n]
sigma_v = 0.5           # Odchylenie standardowe szumu
omega_c1 = np.pi / 2    # Wyjściowa częstotliwość odcięcia

# Tworzenie odpowiedzi impulsowej filtru dolnoprzepustowego
def lowpass_impulse_response(omega_c, M):
    h = np.zeros(M)
    for n in range(-M//2, M//2):
        if n == 0:
            h[n+M//2] = omega_c / np.pi
        else:
            h[n+M//2] = np.sin(omega_c * n) / (np.pi * n)
    return h

h_true = lowpass_impulse_response(omega_c1, M)

# Generowanie sygnału wejściowego x[n]
x = np.random.normal(0, sigma_x, N)

# Filtracja sygnału przez układ (d[n])
d_clean = np.convolve(x, h_true, mode='full')[:N]
v = np.random.normal(0, sigma_v, N)
d = d_clean + v

# Funkcja LMS
def LMS(x, d, M, mu):
    N = len(x)
    h = np.zeros(M)
    e = np.zeros(N)
    H = np.zeros((N, M))
    for n in range(M, N):
        x_vec = x[n:n-M:-1]
        y = np.dot(h, x_vec)
        e[n] = d[n] - y
        h = h + 2 * mu * e[n] * x_vec
        H[n, :] = h
    return e, H

# Poprawiona funkcja RLS
def RLS(x, d, M, lam, delta=0.01):
    N = len(x)
    h = np.zeros(M)
    e = np.zeros(N)
    P = (1/delta) * np.eye(M)
    H = np.zeros((N, M))
    for n in range(M, N):
        x_vec = x[n:n-M:-1]
        pi = np.dot(P, x_vec)
        g = 1.0 / (lam + np.dot(x_vec, pi))
        k = pi * g
        y = np.dot(h, x_vec)
        e[n] = d[n] - y
        h = h + k * e[n]
        P = (P - np.outer(k, np.dot(x_vec, P))) / lam
        H[n, :] = h
    return e, H

# Symulacja
mu = 0.001
lam1 = 1
lam2 = 0.999

e_LMS, H_LMS = LMS(x, d, M, mu)
e_RLS_1, H_RLS_1 = RLS(x, d, M, lam1)
e_RLS_0999, H_RLS_0999 = RLS(x, d, M, lam2)

# --- Wykresy ---

# Impulse Response Estimate oraz błąd estymacji
plt.figure(figsize=(14,6))
n = np.arange(-M//2, M//2)

# Estymacje h[n]
plt.subplot(1, 2, 1)
plt.plot(n, h_true, 'ko-', label='True')
plt.plot(n, H_LMS[-1,:], 'r*-', label='LMS, µ=0.001')
plt.plot(n, H_RLS_1[-1,:], 'b^-', label='RLS, λ=1')
plt.plot(n, H_RLS_0999[-1,:], 'gs-', label='RLS, λ=0.999')
plt.xlabel('n')
plt.ylabel('h[n]')
plt.title('Disturbed desired signal d[n]')
plt.grid()
plt.legend()

# Błąd estymacji współczynników: v_est[n] - h[n]
plt.subplot(1, 2, 2)
plt.plot(n, H_LMS[-1,:] - h_true, 'r*-', label='LMS, µ=0.001')
plt.plot(n, H_RLS_1[-1,:] - h_true, 'b^-', label='RLS, λ=1')
plt.plot(n, H_RLS_0999[-1,:] - h_true, 'gs-', label='RLS, λ=0.999')
plt.xlabel('n')
plt.ylabel('Error')
plt.title('Impulse Response Estimation Error')
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# Error signal
plt.figure(figsize=(15, 4))
plt.subplot(1,3,1)
plt.plot(e_LMS, 'r')
plt.title('LMS, µ=0.001')
plt.grid()

plt.subplot(1,3,2)
plt.plot(e_RLS_1, 'b')
plt.title('RLS, λ=1')
plt.grid()

plt.subplot(1,3,3)
plt.plot(e_RLS_0999, 'g')
plt.title('RLS, λ=0.999')
plt.grid()
plt.tight_layout()
plt.show()

# Ewolucja współczynników filtru
plt.figure(figsize=(15, 4))
plt.subplot(1,3,1)
plt.plot(H_LMS)
plt.title('LMS, µ=0.001')
plt.grid()

plt.subplot(1,3,2)
plt.plot(H_RLS_1)
plt.title('RLS, λ=1')
plt.grid()

plt.subplot(1,3,3)
plt.plot(H_RLS_0999)
plt.title('RLS, λ=0.999')
plt.grid()
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

# Parametry
M = 40                  # Długość filtra
N = 50000               # Liczba próbek
sigma_x = 1             # Odchylenie standardowe sygnału x[n]
sigma_v = 0.5           # Odchylenie standardowe szumu
omega_c1 = np.pi / 2    # Wyjściowa częstotliwość odcięcia
omega_c2 = 0.8 * np.pi / 2  # Po zmianie

# Tworzenie odpowiedzi impulsowej filtru dolnoprzepustowego
def lowpass_impulse_response(omega_c, M):
    h = np.zeros(M)
    for n in range(-M//2, M//2):
        if n == 0:
            h[n+M//2] = omega_c / np.pi
        else:
            h[n+M//2] = np.sin(omega_c * n) / (np.pi * n)
    return h

h_true1 = lowpass_impulse_response(omega_c1, M)
h_true2 = lowpass_impulse_response(omega_c2, M)

# Generowanie sygnału wejściowego x[n]
x = np.random.normal(0, sigma_x, N)

# Filtracja sygnału przez układ (d[n]) z podmianą w połowie
d_clean = np.zeros(N)
for n in range(N):
    if n < N//2:
        h_true = h_true1
    else:
        h_true = h_true2
    if n >= M:
        d_clean[n] = np.dot(h_true, x[n:n-M:-1])

v = np.random.normal(0, sigma_v, N)
d = d_clean + v

# Funkcja LMS
def LMS(x, d, M, mu):
    N = len(x)
    h = np.zeros(M)
    e = np.zeros(N)
    H = np.zeros((N, M))
    for n in range(M, N):
        x_vec = x[n:n-M:-1]
        y = np.dot(h, x_vec)
        e[n] = d[n] - y
        h = h + 2 * mu * e[n] * x_vec
        H[n, :] = h
    return e, H

# Funkcja RLS
def RLS(x, d, M, lam, delta=0.01):
    N = len(x)
    h = np.zeros(M)
    e = np.zeros(N)
    P = (1/delta) * np.eye(M)
    H = np.zeros((N, M))
    for n in range(M, N):
        x_vec = x[n:n-M:-1]
        pi = np.dot(P, x_vec)
        g = 1.0 / (lam + np.dot(x_vec, pi))
        k = pi * g
        y = np.dot(h, x_vec)
        e[n] = d[n] - y
        h = h + k * e[n]
        P = (P - np.outer(k, np.dot(x_vec, P))) / lam
        H[n, :] = h
    return e, H

# Symulacja
mu = 0.001
lam1 = 1
lam2 = 0.999

e_LMS, H_LMS = LMS(x, d, M, mu)
e_RLS_1, H_RLS_1 = RLS(x, d, M, lam1)
e_RLS_0999, H_RLS_0999 = RLS(x, d, M, lam2)

# --- Wykresy ---

n_axis = np.arange(-M//2, M//2)

# 1. Disturbed desired signal + Estymacja błędu filtru w jednym oknie
plt.figure(figsize=(14,5))

# 1a. Estymacja h[n]
plt.subplot(1,2,1)
plt.plot(n_axis, h_true2, 'g-o', label='True')
plt.plot(n_axis, H_LMS[-1,:], 'r*-', label='LMS µ=0.001')
plt.plot(n_axis, H_RLS_1[-1,:], 'b^-', label='RLS λ=1')
plt.plot(n_axis, H_RLS_0999[-1,:], 'ms-', label='RLS λ=0.999')
plt.xlabel('n')
plt.title("Disturbed desired signal d[n]")
plt.grid()
plt.legend()

# 1b. Estymacja błędu filtru: v_est[n] - h[n]
plt.subplot(1,2,2)
plt.plot(n_axis, H_LMS[-1,:] - h_true2, 'r*-', label='LMS µ=0.001')
plt.plot(n_axis, H_RLS_1[-1,:] - h_true2, 'b^-', label='RLS λ=1')
plt.plot(n_axis, H_RLS_0999[-1,:] - h_true2, 'ms-', label='RLS λ=0.999')
plt.xlabel('n')
plt.title("Impulse Response Estimation Error]")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()

# 2. Błąd estymacji w czasie
plt.figure(figsize=(15, 4))
plt.subplot(1,3,1)
plt.plot(e_LMS, 'r')
plt.title('LMS µ=0.001')
plt.grid()
plt.subplot(1,3,2)
plt.plot(e_RLS_1, 'b')
plt.title('RLS λ=1')
plt.grid()
plt.subplot(1,3,3)
plt.plot(e_RLS_0999, 'g')
plt.title('RLS λ=0.999')
plt.grid()
plt.tight_layout()
plt.show()

# 3. Ewolucja współczynników
plt.figure(figsize=(15, 4))
plt.subplot(1,3,1)
plt.plot(H_LMS)
plt.title('LMS µ=0.001')
plt.grid()
plt.subplot(1,3,2)
plt.plot(H_RLS_1)
plt.title('RLS λ=1')
plt.grid()
plt.subplot(1,3,3)
plt.plot(H_RLS_0999)
plt.title('RLS λ=0.999')
plt.grid()
plt.tight_layout()
plt.show()
