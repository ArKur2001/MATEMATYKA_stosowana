import numpy as np
import matplotlib.pyplot as plt

def h_lp(n, wc=np.pi/2):
    if n == 0:
        return wc / np.pi
    else:
        return np.sin(wc * n) / (np.pi * n)

def generate_impulse_response(M, wc=np.pi/2):
    return np.array([h_lp(n, wc) for n in range(-M, M + 1)])

def generate_gaussian_noise(length, mean=0, std=1):
    return np.random.normal(mean, std, length)

def identify_impulse_response(f, d_noisy, M):
    N = len(f)
    L = 2 * M + 1
    X = np.zeros((N, L))
    for i in range(N):
        for j in range(L):
            if 0 <= i - M + j < N:
                X[i, j] = f[i - M + j]
    h_estimated = np.linalg.pinv(X) @ d_noisy
    return h_estimated

def main():
    configurations = [(20, 20), (20, 18), (18, 20), (22, 19)]
    N_true = 5000  
    N_estimated = 5000  
    wc = np.pi / 2  

    plt.figure(figsize=(18, 12))  

    for idx, (M_true, M_estimated) in enumerate(configurations):
        h_true = generate_impulse_response(M_true, wc)

        f_true = generate_gaussian_noise(N_true)

        d_true = np.convolve(f_true, h_true, mode='same')

        h_estimated_true = generate_impulse_response(M_estimated, wc)

        f_estimated = generate_gaussian_noise(N_estimated)

        noise = generate_gaussian_noise(N_estimated, mean=0, std=0.1)
        d_noisy = np.convolve(f_estimated, h_estimated_true, mode='same') + noise

        h_estimated = identify_impulse_response(f_estimated, d_noisy, M_estimated)

        n_true = np.arange(-M_true, M_true + 1)
        n_estimated = np.arange(-M_estimated, M_estimated + 1)

        plt.subplot(4, 2, 2 * idx + 1)
        plt.title(f"Ideal vs Estimated Impulse Response\n(M_true={M_true}, M_estimated={M_estimated})")
        plt.stem(n_true, h_true, linefmt='k-', markerfmt='ko', basefmt='k', label="True")
        plt.plot(n_estimated, h_estimated, 'r-', label="Estimated")
        plt.xlabel("n")
        plt.ylabel("h[n]")
        plt.legend()
        plt.grid()

        plt.subplot(4, 2, 2 * idx + 2)
        plt.title(f"Error (Noisy Output)\n(M_true={M_true}, M_estimated={M_estimated})")

        min_len = min(len(h_true), len(h_estimated))
        error = h_true[:min_len] - h_estimated[:min_len]
        n_error = np.arange(-min_len // 2, -min_len // 2 + min_len)

        plt.plot(n_error, error, 'r-')
        plt.xlabel("n")
        plt.ylabel("Error")
        plt.grid()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()