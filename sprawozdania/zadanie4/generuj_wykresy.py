"""
Skrypt generujący wykresy do sprawozdania z Zadania 4.

Generuje:
  - s2_sygnal.png          -- sygnał testowy S2
  - s2_dft.png             -- widmo DFT (amplituda + faza)
  - s2_fft.png             -- widmo FFT (amplituda + faza)
  - s2_rekonstrukcja_fourier.png -- rekonstrukcja sygnału po IDFT/IFFT
  - s2_wht.png             -- WHT macierzowa
  - s2_fwht.png            -- szybka WHT
  - porownanie_czasow.png  -- wykres porównania czasów
"""

import sys
import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Dodaj główny katalog projektu do ścieżki
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.Transforms import dft, idft, fft, ifft, wht, iwht, fast_wht, ifast_wht

OUT_DIR = os.path.join(os.path.dirname(__file__), "wykresy")
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Parametry
# ---------------------------------------------------------------------------
N = 16
F_PR = 16.0


def generate_S2(N: int = 16, f_pr: float = 16.0):
    """S2(t) = 2·sin(2π/2·t) + sin(2π/1·t) + 5·sin(2π/0.5·t)"""
    t = np.arange(N) / f_pr
    s = (2.0 * np.sin(2 * np.pi / 2.0 * t)
         + np.sin(2 * np.pi / 1.0 * t)
         + 5.0 * np.sin(2 * np.pi / 0.5 * t))
    return t, s


# ---------------------------------------------------------------------------
# 1. Sygnał S2
# ---------------------------------------------------------------------------
def plot_s2_signal():
    t, s = generate_S2(N, F_PR)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.stem(t, s, linefmt="C0-", markerfmt="C0o", basefmt="gray")
    ax.set_title("Sygnał testowy S2  (N=16, $f_{pr}$=16 Hz)", fontsize=11)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s2_sygnal.png"), dpi=150)
    plt.close(fig)
    print("[OK] s2_sygnal.png")


# ---------------------------------------------------------------------------
# 2. DFT
# ---------------------------------------------------------------------------
def plot_dft():
    t, s = generate_S2(N, F_PR)
    X = dft(s)
    freqs = np.arange(N) * F_PR / N
    amp = np.abs(X)
    phase = np.angle(X)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax1.stem(freqs, amp, linefmt="C1-", markerfmt="C1o", basefmt="gray")
    ax1.set_title("DFT -- widmo amplitudowe $|X(m)|$", fontsize=11)
    ax1.set_ylabel("$|X(m)|$")
    ax1.grid(True, alpha=0.4)

    ax2.stem(freqs, phase, linefmt="C2-", markerfmt="C2o", basefmt="gray")
    ax2.set_title("DFT -- widmo fazowe $\\angle X(m)$", fontsize=11)
    ax2.set_xlabel("Częstotliwość [Hz]")
    ax2.set_ylabel("Faza [rad]")
    ax2.grid(True, alpha=0.4)

    fig.tight_layout(pad=2.5)
    fig.savefig(os.path.join(OUT_DIR, "s2_dft.png"), dpi=150)
    plt.close(fig)
    print("[OK] s2_dft.png")


# ---------------------------------------------------------------------------
# 3. FFT
# ---------------------------------------------------------------------------
def plot_fft():
    t, s = generate_S2(N, F_PR)
    X = fft(s)
    freqs = np.arange(N) * F_PR / N
    amp = np.abs(X)
    phase = np.angle(X)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax1.stem(freqs, amp, linefmt="C1-", markerfmt="C1o", basefmt="gray")
    ax1.set_title("FFT (DIT) -- widmo amplitudowe $|X(m)|$", fontsize=11)
    ax1.set_ylabel("$|X(m)|$")
    ax1.grid(True, alpha=0.4)

    ax2.stem(freqs, phase, linefmt="C2-", markerfmt="C2o", basefmt="gray")
    ax2.set_title("FFT (DIT) -- widmo fazowe $\\angle X(m)$", fontsize=11)
    ax2.set_xlabel("Częstotliwość [Hz]")
    ax2.set_ylabel("Faza [rad]")
    ax2.grid(True, alpha=0.4)

    fig.tight_layout(pad=2.5)
    fig.savefig(os.path.join(OUT_DIR, "s2_fft.png"), dpi=150)
    plt.close(fig)
    print("[OK] s2_fft.png")


# ---------------------------------------------------------------------------
# 4. Rekonstrukcja Fouriera
# ---------------------------------------------------------------------------
def plot_reconstruction():
    t, s = generate_S2(N, F_PR)
    X_dft = dft(s)
    s_rec = idft(X_dft).real

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.stem(t, s_rec, linefmt="C3-", markerfmt="C3o", basefmt="gray",
            label="Sygnał odtworzony (IDFT)")
    ax.plot(t, s, "C0--", alpha=0.6, label="Oryginał")
    ax.set_title("Rekonstrukcja sygnału S2 po IDFT", fontsize=11)
    ax.set_xlabel("Czas [s]")
    ax.set_ylabel("Amplituda")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.4)

    err = np.max(np.abs(s_rec - s))
    ax.text(0.02, 0.95, f"Maks. błąd: {err:.2e}",
            transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "s2_rekonstrukcja_fourier.png"), dpi=150)
    plt.close(fig)
    print("[OK] s2_rekonstrukcja_fourier.png")


# ---------------------------------------------------------------------------
# 5. WHT
# ---------------------------------------------------------------------------
def plot_wht():
    t, s = generate_S2(N, F_PR)
    X = wht(s)
    idx = np.arange(N)
    amp = np.abs(X)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax1.stem(idx, amp, linefmt="C1-", markerfmt="C1o", basefmt="gray")
    ax1.set_title("WHT (macierzowa) -- $|X(m)|$", fontsize=11)
    ax1.set_ylabel("$|X(m)|$")
    ax1.grid(True, alpha=0.4)

    ax2.stem(idx, X, linefmt="C2-", markerfmt="C2o", basefmt="gray")
    ax2.set_title("WHT (macierzowa) -- współczynniki Walsha $X(m)$", fontsize=11)
    ax2.set_xlabel("Indeks współczynnika (sequency)")
    ax2.set_ylabel("$X(m)$")
    ax2.axhline(y=0, color="gray", linewidth=0.8)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout(pad=2.5)
    fig.savefig(os.path.join(OUT_DIR, "s2_wht.png"), dpi=150)
    plt.close(fig)
    print("[OK] s2_wht.png")


# ---------------------------------------------------------------------------
# 6. Fast WHT
# ---------------------------------------------------------------------------
def plot_fwht():
    t, s = generate_S2(N, F_PR)
    X = fast_wht(s)
    idx = np.arange(N)
    amp = np.abs(X)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True)
    ax1.stem(idx, amp, linefmt="C1-", markerfmt="C1o", basefmt="gray")
    ax1.set_title("Fast WHT -- $|X(m)|$", fontsize=11)
    ax1.set_ylabel("$|X(m)|$")
    ax1.grid(True, alpha=0.4)

    ax2.stem(idx, X, linefmt="C2-", markerfmt="C2o", basefmt="gray")
    ax2.set_title("Fast WHT -- współczynniki Walsha $X(m)$", fontsize=11)
    ax2.set_xlabel("Indeks współczynnika (sequency)")
    ax2.set_ylabel("$X(m)$")
    ax2.axhline(y=0, color="gray", linewidth=0.8)
    ax2.grid(True, alpha=0.4)

    fig.tight_layout(pad=2.5)
    fig.savefig(os.path.join(OUT_DIR, "s2_fwht.png"), dpi=150)
    plt.close(fig)
    print("[OK] s2_fwht.png")


# ---------------------------------------------------------------------------
# 7. Porównanie czasów
# ---------------------------------------------------------------------------
def benchmark_and_plot():
    ns = range(1, 11)
    sizes = [2**n for n in ns]
    REPEATS = 50

    times_dft = []
    times_fft = []
    times_wht = []
    times_fwht = []

    for N_ in sizes:
        _, s = generate_S2(N_, F_PR)

        # DFT
        elapsed = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            dft(s)
            elapsed.append(time.perf_counter() - t0)
        times_dft.append(np.median(elapsed) * 1000)

        # FFT
        elapsed = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fft(s)
            elapsed.append(time.perf_counter() - t0)
        times_fft.append(np.median(elapsed) * 1000)

        # WHT
        elapsed = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            wht(s)
            elapsed.append(time.perf_counter() - t0)
        times_wht.append(np.median(elapsed) * 1000)

        # Fast WHT
        elapsed = []
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            fast_wht(s)
            elapsed.append(time.perf_counter() - t0)
        times_fwht.append(np.median(elapsed) * 1000)

    # Wydruk tabeli do konsoli (do wklejenia w LaTeX)
    print("\n" + "=" * 65)
    print(f"{'N':>6} | {'DFT [ms]':>10} | {'FFT [ms]':>10} | {'WHT [ms]':>10} | {'FWHT [ms]':>10}")
    print("-" * 65)
    for i, N_ in enumerate(sizes):
        print(f"{N_:>6} | {times_dft[i]:>10.4f} | {times_fft[i]:>10.4f} | "
              f"{times_wht[i]:>10.4f} | {times_fwht[i]:>10.4f}")
    print("=" * 65)

    # Wykres
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(sizes, times_dft, "o-", label="DFT  $O(N^2)$", color="C0")
    ax1.plot(sizes, times_fft, "s-", label="FFT  $O(N \\log N)$", color="C1")
    ax1.set_title("Fourier: DFT vs FFT", fontsize=11)
    ax1.set_xlabel("N (liczba próbek)")
    ax1.set_ylabel("Czas [ms]")
    ax1.set_xscale("log", base=2)
    ax1.legend()
    ax1.grid(True, alpha=0.4)

    ax2.plot(sizes, times_wht, "o-", label="WHT  $O(N^2)$", color="C2")
    ax2.plot(sizes, times_fwht, "s-", label="Fast WHT  $O(N \\log N)$", color="C3")
    ax2.set_title("Walsh-Hadamard: WHT vs Fast WHT", fontsize=11)
    ax2.set_xlabel("N (liczba próbek)")
    ax2.set_ylabel("Czas [ms]")
    ax2.set_xscale("log", base=2)
    ax2.legend()
    ax2.grid(True, alpha=0.4)

    fig.tight_layout(pad=2.0)
    fig.savefig(os.path.join(OUT_DIR, "porownanie_czasow.png"), dpi=150)
    plt.close(fig)
    print("[OK] porownanie_czasow.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generowanie wykresów do sprawozdania Zadania 4...\n")
    plot_s2_signal()
    plot_dft()
    plot_fft()
    plot_reconstruction()
    plot_wht()
    plot_fwht()
    benchmark_and_plot()
    print(f"\nWszystkie wykresy zapisane w: {OUT_DIR}")
