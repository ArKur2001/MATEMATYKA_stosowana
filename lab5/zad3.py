import numpy as np
import matplotlib.pyplot as plt

# Funkcje wielomianowe i ich pochodne
functions = [
    (lambda z: z**3 - 1, lambda z: 3 * z**2, "f(z) = z^3 - 1"),
    (lambda z: z**4 - 1, lambda z: 4 * z**3, "f(z) = z^4 - 1"),
    (lambda z: z**5 - 1, lambda z: 5 * z**4, "f(z) = z^5 - 1"),
    (lambda z: z**3 - z, lambda z: 3 * z**2 - 1, "f(z) = z^3 - z")
]

# Metoda Newtona
def newton_method(f, f_prime, z, max_iter=50, tol=1e-6):
    for i in range(max_iter):
        dz = f(z) / f_prime(z)
        z -= dz
        if abs(dz) < tol:
            break
    return z, i

# Siatka punktów w płaszczyźnie zespolonej
x = np.linspace(-2, 2, 800)
y = np.linspace(-2, 2, 800)
X, Y = np.meshgrid(x, y)
Z = X + 1j * Y

# Generowanie wykresów dla każdej funkcji
plt.figure(figsize=(16, 12))

for idx, (f, f_prime, title) in enumerate(functions):
    # Iteracje Newtona
    roots = np.zeros(Z.shape, dtype=complex)
    iterations = np.zeros(Z.shape, dtype=int)

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            roots[i, j], iterations[i, j] = newton_method(f, f_prime, Z[i, j])

    # Unikalne pierwiastki
    unique_roots = np.unique(np.round(roots, decimals=6))

    # Mapowanie pierwiastków na kolory
    root_colors = {root: idx for idx, root in enumerate(unique_roots)}
    colors = np.vectorize(lambda z: root_colors[np.round(z, decimals=6)])(roots)

    # Rysowanie wykresów
    plt.subplot(len(functions), 2, 2 * idx + 1)
    plt.imshow(iterations, extent=(-2, 2, -2, 2), cmap='Blues') 
    plt.colorbar(label='Number of iterations')
    plt.title(f'{title} - Iterations')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')

    plt.subplot(len(functions), 2, 2 * idx + 2)
    plt.imshow(colors, extent=(-2, 2, -2, 2), cmap='hsv')
    plt.colorbar(label='Root index')
    plt.title(f'{title} - Roots')
    plt.xlabel('Re(z)')
    plt.ylabel('Im(z)')

plt.tight_layout()
plt.show()