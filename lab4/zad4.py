import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

def irwls(A, x, p, maxiter=200, stopeps=1e-7):
    """
    Computes the minimum solution c to ||x - Ac||_p using
    iteratively reweighted least squares (IRLS) algorithm.

    Parameters:
        A (ndarray): System matrix.
        x (ndarray): Right-hand side of the equation.
        p (float): L_p norm.
        maxiter (int): Maximum number of iterations.
        stopeps (float): Convergence threshold.

    Returns:
        c (ndarray): Solution vector.
    """
    pk = 2  # Starting value of p
    c = np.linalg.lstsq(A, x, rcond=None)[0]  # Initial LS solution
    xhat = A @ c
    gamma = 1.5

    for k in range(maxiter):
        pk = min(p, gamma * pk)  # Update p for this iteration
        e = x - xhat  # Estimation error
        s = np.abs(e) ** ((pk - 2) / 2)  # New weights
        WA = np.diag(s) @ A  # Weighted matrix
        chat = np.linalg.lstsq(WA, s * x, rcond=None)[0]  # Weighted LS solution
        lambda_ = 1 / (pk - 1)
        cnew = lambda_ * chat + (1 - lambda_) * c

        if np.linalg.norm(c - cnew) < stopeps:
            c = cnew
            break

        c = cnew
        xhat = A @ c

    return c

def design_fir_filter(N, wc, p, transition_width):
    M = 512
    omega = np.linspace(0, np.pi, M)

    # Desired frequency response
    Hd = np.zeros(M)
    Hd[omega <= wc] = 1 

    # Transition band
    transition_start = wc
    transition_end = min(np.pi, wc + transition_width)
    transition_indices = (omega >= transition_start) & (omega <= transition_end)
    Hd[transition_indices] = np.linspace(1, 0, np.sum(transition_indices))

    # Construct the system matrix A
    k = np.arange(N)
    A = np.cos(np.outer(omega, k))

    # Solve for filter coefficients using IRLS
    h = irwls(A, Hd, p)

    return h, Hd, omega

def plot_results(h, Hd, omega, wc, transition_width, title_suffix):
    # Time-domain filter coefficients (symmetry applied)
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    h_symmetric = np.concatenate((h[::-1], h)) 
    plt.stem(h_symmetric, basefmt=" ", markerfmt="o") 
    plt.title(f"Filter Coefficients (Time Domain, Symmetrical) - {title_suffix}")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.grid()

    # Desired vs actual frequency response
    fft_size = 4096 
    H = np.fft.fft(h, fft_size)
    H = np.abs(H[:fft_size // 2])  
    omega_fine = np.linspace(0, np.pi, fft_size // 2)
    plt.subplot(2, 2, 2)
    plt.plot(omega, Hd, label="H ideal")
    plt.plot(omega_fine, H, label="H IRLS")
    plt.title(f"Frequency Response - {title_suffix}")
    plt.xlabel("Frequency (rad)")
    plt.ylabel("Magnitude")
    plt.legend()
    plt.grid()

    # Error between ideal and actual response
    plt.subplot(2, 2, 3)
    Hd_interp = np.interp(omega_fine, omega, Hd) 
    plt.plot(omega_fine, Hd_interp - H)
    plt.title(f"Error - {title_suffix}")
    plt.xlabel("Frequency (rad)")
    plt.ylabel("Error")
    plt.grid()

    # Log-magnitude response
    plt.subplot(2, 2, 4)
    plt.plot(omega_fine, 20 * np.log10(np.maximum(H, 1e-10)))
    plt.title(f"Log-Magnitude Response - {title_suffix}")
    plt.xlabel("Frequency (rad)")
    plt.ylabel("Magnitude (dB)")
    plt.grid()

    plt.tight_layout()
    plt.show()

# Parameters
N = 51
wc = np.pi / 2
p = 30

transition_width_narrow = 0.05 * np.pi
h_narrow, Hd_narrow, omega = design_fir_filter(N, wc, p, transition_width_narrow)
plot_results(h_narrow, Hd_narrow, omega, wc, transition_width_narrow, "Narrow Transition Band")

transition_width_wide = 0.2 * np.pi
h_wide, Hd_wide, omega = design_fir_filter(N, wc, p, transition_width_wide)
plot_results(h_wide, Hd_wide, omega, wc, transition_width_wide, "Wide Transition Band")