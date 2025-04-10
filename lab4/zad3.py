import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import freqz

# Parametry filtru
M = 20
N_freqs = 512
w = np.linspace(0, np.pi, N_freqs)

# Macierz A
A = np.zeros((N_freqs, M+1))
for i in range(M+1):
    A[:, i] = 2 * np.cos(w * i)
A[:, 0] = 1

def ideal_response(w, typ='wide'):
    if typ == 'wide':
        wc = np.pi / 3  # Szerokie pasmo przejściowe
    elif typ == 'narrow':
        wc = 3 * np.pi / 4  # Wąskie pasmo przejściowe
    else:
        raise ValueError("Unknown type")
    return np.where(w <= wc, 1, 0), wc

def weight_function(w, typ='S1'):
    if typ == 'S1':
        return np.ones_like(w)
    elif typ == 'S2':
        return np.where(w < 1.5, 20.0, 0.0)
    elif typ == 'S3':
        return np.where(w < 1.5, 0.0, 20.0)
    else:
        raise ValueError("Unknown weighting function")

def plot_filter_response(w, Hd, A, S, title):
    W = np.diag(S)
    c = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ Hd

    # Odpowiedź impulsowa h[n]
    h = np.zeros(2*M+1)
    h[M] = c[0]
    for n in range(1, M+1):
        h[M+n] = h[M-n] = c[n]/2

    # Odpowiedź częstotliwościowa
    w_out, H_out = freqz(h, worN=w)
    H_mag = np.abs(H_out)
    error = np.abs(Hd - H_mag)

    # Rysowanie wykresów
    fig, axs = plt.subplots(1, 5, figsize=(22, 4))
    fig.suptitle(f"{title}", fontsize=14)

    axs[0].plot(w, S)
    axs[0].set_title("S(e^jω)")
    axs[0].set_xlabel("ω [rad]")
    axs[0].grid()

    axs[1].stem(np.arange(-M, M+1), h, basefmt=" ")
    axs[1].set_title("h[n]")
    axs[1].set_xlabel("n")
    axs[1].grid()

    axs[2].plot(w_out, H_mag, label='|H(e^jω)|')
    axs[2].plot(w, Hd, '--', label='Hd(ω)', linewidth=1)
    axs[2].set_title("Porównanie |H(e^jω)| i Hd(ω)")
    axs[2].set_xlabel("ω [rad]")
    axs[2].legend()
    axs[2].grid()

    axs[3].plot(w, error)
    axs[3].set_title("Błąd aproksymacji |Hd - H|")
    axs[3].set_xlabel("ω [rad]")
    axs[3].grid()

    axs[4].plot(w_out, 20 * np.log10(H_mag + 1e-10))
    axs[4].set_title("|H(e^jω)| [dB]")
    axs[4].set_xlabel("ω [rad]")
    axs[4].grid()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# === S₁: flat weight ===
S1 = weight_function(w, 'S1')
Hd_wide, _ = ideal_response(w, 'wide')
Hd_narrow, _ = ideal_response(w, 'narrow')
plot_filter_response(w, Hd_wide, A, S1, "Wideband $S_1$")
plot_filter_response(w, Hd_narrow, A, S1, "Narrowband $S_1$")

# === S₂: większa waga w dolnym paśmie ===
S2 = weight_function(w, 'S2')
plot_filter_response(w, Hd_wide, A, S2, "Wideband $S_2$")
plot_filter_response(w, Hd_narrow, A, S2, "Narrowband $S_2$")

# === S₃: większa waga w górnym paśmie ===
S3 = weight_function(w, 'S3')
plot_filter_response(w, Hd_wide, A, S3, "Wideband $S_3$")
plot_filter_response(w, Hd_narrow, A, S3, "Narrowband $S_3$")

def plot_combined_responses(w, Hd, A, weights, labels, title):
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(title, fontsize=14)

    for S, label in zip(weights, labels):
        W = np.diag(S)
        c = np.linalg.inv(A.T @ W @ A) @ A.T @ W @ Hd

        # Tworzenie odpowiedzi impulsowej
        h = np.zeros(2*M+1)
        h[M] = c[0]
        for n in range(1, M+1):
            h[M+n] = h[M-n] = c[n]/2

        # Odpowiedź częstotliwościowa
        w_out, H_out = freqz(h, worN=w)
        H_mag = np.abs(H_out)
        error = np.abs(Hd - H_mag)

        axs[0].plot(w_out, H_mag, label=label)
        axs[1].plot(w, error, label=label)

    axs[0].plot(w, Hd, 'k--', linewidth=1, label='Hd(ω)')
    axs[0].set_title("Porównanie |H(e^jω)|")
    axs[0].set_xlabel("ω [rad]")
    axs[0].legend()
    axs[0].grid()

    axs[1].set_title("Błąd aproksymacji |Hd - H|")
    axs[1].set_xlabel("ω [rad]")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()

# Przygotowanie wag
S1 = weight_function(w, 'S1')
S2 = weight_function(w, 'S2')
S3 = weight_function(w, 'S3')

# Zbiorcze porównanie: wideband
plot_combined_responses(
    w, Hd_wide, A,
    weights=[S1, S2, S3],
    labels=["$S_1$", "$S_2$", "$S_3$"],
    title="Zbiorcze porównanie filtrów – Wideband"
)

# Zbiorcze porównanie: narrowband
plot_combined_responses(
    w, Hd_narrow, A,
    weights=[S1, S2, S3],
    labels=["$S_1$", "$S_2$", "$S_3$"],
    title="Zbiorcze porównanie filtrów – Narrowband"
)
