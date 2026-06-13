"""
Transforms.py – implementacje transformacji dyskretnych dla Zadania 4.

Zawiera:
  - DFT / IDFT           (wzory F-1, F-2, F-6)
  - FFT / IFFT           (wzory F-7 – F-12, decymacja w czasie, Cooley-Tukey)
  - WHT / Fast-WHT       (wzory WH-1 – WH-5, Transformacja Walsha-Hadamarda)
"""

import numpy as np


# ---------------------------------------------------------------------------
# DFT / IDFT
# ---------------------------------------------------------------------------

def dft(x: np.ndarray) -> np.ndarray:
    """
    Dyskretna Transformacja Fouriera (wzór F-1/F-6).

        X(m) = (1/N) * sum_{n=0}^{N-1} x(n) * W_N^{-mn}

    gdzie  W_N = exp(j*2*pi/N).

    Parametry
    ----------
    x : tablica rzeczywista lub zespolona (N próbek)

    Zwraca
    -------
    X : tablica zespolona długości N
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)
    m = np.arange(N)
    n = m.reshape((N, 1))          # kolumna
    W = np.exp(-1j * 2 * np.pi / N * m * n)   # macierz NxN (wzór F-6)
    return (W @ x) / N


def idft(X: np.ndarray) -> np.ndarray:
    """
    Odwrotna Dyskretna Transformacja Fouriera (wzór F-2).

        x(n) = sum_{m=0}^{N-1} X(m) * W_N^{mn}
    """
    X = np.asarray(X, dtype=complex)
    N = len(X)
    n = np.arange(N)
    m = n.reshape((N, 1))
    W = np.exp(1j * 2 * np.pi / N * n * m)
    return W @ X


# ---------------------------------------------------------------------------
# FFT / IFFT  (Cooley-Tukey, decymacja w czasie – DIT)
# ---------------------------------------------------------------------------

def fft(x: np.ndarray) -> np.ndarray:
    """
    Szybka Transformacja Fouriera – algorytm Cooley-Tukey DIT (wzory F-7 – F-12).

    Wymaga N = potęga 2.  Dla N ≤ 8 przełącza się na DFT (brak rekursji).
    Wynik przeskalowany tak jak DFT (dzielenie przez N).

    Parametry
    ----------
    x : tablica próbek (długość musi być potęgą 2)

    Zwraca
    -------
    X : widmo zespolone, zgodne z dft()
    """
    x = np.asarray(x, dtype=complex)
    N = len(x)

    if N <= 1:
        return x.copy()

    if N & (N - 1):
        raise ValueError(f"FFT wymaga N będącego potęgą 2, otrzymano N={N}.")

    # Dla małych N – bezpośredni DFT
    if N <= 8:
        return dft(x)

    # Podział na parzyste / nieparzyste (wzory F-7, F-8)
    even = fft(x[0::2])   # DFT_{N/2} próbek parzystych
    odd  = fft(x[1::2])   # DFT_{N/2} próbek nieparzystych

    # Twiddle factors: W_N^{-m} = exp(-j*2*pi*m/N)  dla m = 0..N/2-1
    k = np.arange(N // 2)
    twiddle = np.exp(-1j * 2 * np.pi * k / N)

    # Motylki (wzór F-12), z uwzględnieniem skalowania 1/N:
    # X(m)       = (1/2) * [even(m) + W_N^{-m} * odd(m)]
    # X(m+N/2)   = (1/2) * [even(m) - W_N^{-m} * odd(m)]
    # Czynnik 1/2 pochodzi z (1/N) = (1/(N/2)) * (1/2)
    t = twiddle * odd
    X = np.empty(N, dtype=complex)
    X[:N // 2] = (even + t) * 0.5
    X[N // 2:] = (even - t) * 0.5
    return X


def ifft(X: np.ndarray) -> np.ndarray:
    """
    Odwrotna FFT (przez sprzężenie + FFT + N²·sprzężenie).

    Właściwość:  IFFT(X) = N * conj( FFT(conj(X)) )
    """
    X = np.asarray(X, dtype=complex)
    N = len(X)
    # IFFT przez sprzężenie i FFT (bez skalowania 1/N w IDFT)
    # Korzystamy z: idft(X) = N * conj( dft(conj(X)) )
    return np.conj(fft(np.conj(X))) * N


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform (WHT) / Fast WHT
# ---------------------------------------------------------------------------

def _hadamard_matrix(m: int) -> np.ndarray:
    """
    Buduje znormalizowaną macierz Hadamarda H_m rozmiaru 2^m × 2^m (wzór WH-3).

        H_0 = 1
        H_m = (1/sqrt(2)) * [[H_{m-1},  H_{m-1}],
                              [H_{m-1}, -H_{m-1}]]
    """
    H = np.array([[1.0]])
    for _ in range(m):
        H = (1.0 / np.sqrt(2)) * np.block([[H, H], [H, -H]])
    return H


def wht(x: np.ndarray) -> np.ndarray:
    """
    Transformacja Walsha-Hadamarda (macierzowa, wzory WH-1 / WH-3).

        X = H_m · x

    Parametry
    ----------
    x : sygnał wejściowy, długość N = 2^m

    Zwraca
    -------
    X : transformata WH (tej samej długości co x)
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N & (N - 1):
        raise ValueError(f"WHT wymaga N będącego potęgą 2, otrzymano N={N}.")

    m = int(np.log2(N))
    H = _hadamard_matrix(m)
    return H @ x


def iwht(X: np.ndarray) -> np.ndarray:
    """
    Odwrotna Transformacja Walsha-Hadamarda (wzór WH-2).

        x = H_m · X

    Macierz H_m jest symetryczna i ortogonalna (H_m^{-1} = H_m),
    więc odwrotna transformacja jest identyczna z prostą.
    """
    return wht(X)


def fast_wht(x: np.ndarray) -> np.ndarray:
    """
    Szybka Transformacja Walsha-Hadamarda (wzory WH-4 / WH-5).

    Rekurencyjny algorytm motylkowy:
        X[0..N/2-1] = H_{m-1} · (x[0..N/2-1] + x[N/2..N-1])
        X[N/2..N-1] = H_{m-1} · (x[0..N/2-1] - x[N/2..N-1])

    Złożoność: O(N log₂ N) zamiast O(N²) dla WHT macierzowej.
    """
    x = np.asarray(x, dtype=float)
    N = len(x)

    if N & (N - 1):
        raise ValueError(f"Fast WHT wymaga N będącego potęgą 2, otrzymano N={N}.")

    if N == 1:
        return x.copy()

    half = N // 2
    # Wzór WH-5: sumy i różnice pierwszej i drugiej połowy
    x_sum  = x[:half] + x[half:]
    x_diff = x[:half] - x[half:]

    # Rekurencja + normalizacja 1/sqrt(2) (z macierzy Hadamarda)
    upper = fast_wht(x_sum)
    lower = fast_wht(x_diff)

    # Każdy krok rekurencji dokłada czynnik 1/sqrt(2)
    scale = 1.0 / np.sqrt(2)
    return np.concatenate([upper, lower]) * scale


def ifast_wht(X: np.ndarray) -> np.ndarray:
    """
    Odwrotna szybka WHT – identyczna z fast_wht (macierz H jest symetryczna).
    """
    return fast_wht(X)
