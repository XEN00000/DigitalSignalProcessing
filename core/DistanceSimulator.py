"""
Symulator pomiaru odległości (radar) oparty na korelacji wzajemnej.

Schemat działania:
  1. Generuj sygnał sondujący s[n]  (probe, długość M próbek)
  2. Zbuduj bufor odebrany y[n]     (echo opóźnione o delay_samples + ewentualny szum)
  3. Oblicz korelację wzajemną:
         R_ys[k] = Σ_n  y[n] · s[n - k]
     -- metoda bezpośrednia: wg wzoru (9) z instrukcji (pętla po k)
     -- metoda przez splot:   R_ys = splot(y, s_odwrócone) wg wzoru (2)
  4. Wyjściowy wektor korelacji jest przeindeksowany tak, że indeks 0
     odpowiada lagowi k = -(M-1), a indeks (M-1) odpowiada k = 0.
  5. Pozycja maksimum dla k ≥ 0  →  opóźnienie w próbkach  →  odległość
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from core.Signal import Signal
from core.Filter import Filter


# ---------------------------------------------------------------------------
# Wynik symulacji
# ---------------------------------------------------------------------------
@dataclass
class SimulationResult:
    """Kompletny wynik jednego uruchomienia symulatora radaru."""

    # ---- parametry wejściowe ----
    fs: float
    speed: float
    distance: float
    expected_delay: float
    method: str = "via_convolution"

    # ---- sygnały ----
    t_axis: np.ndarray = field(default_factory=lambda: np.array([]))
    sent_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    received_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    probe: np.ndarray = field(default_factory=lambda: np.array([]))
    probe_t: np.ndarray = field(default_factory=lambda: np.array([]))

    # ---- korelacja ----
    lags_samples: np.ndarray = field(default_factory=lambda: np.array([]))
    lags_time: np.ndarray = field(default_factory=lambda: np.array([]))
    correlation: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_lag_samples: int = 0
    peak_lag_time: float = 0.0

    # ---- wyniki pomiaru ----
    measured_delay: float = 0.0
    measured_distance: float = 0.0
    delay_error: float = 0.0
    distance_error: float = 0.0

    # ---- parametry sygnału sondującego ----
    signal_type: str = "chirp"
    signal_duration: float = 0.05
    signal_freq: float = 1000.0
    signal_freq_end: float = 4000.0


# ---------------------------------------------------------------------------
# Symulator
# ---------------------------------------------------------------------------
class DistanceSimulator:

    # ------------------------------------------------------------------
    # Generatory sygnałów sondujących
    # ------------------------------------------------------------------
    @staticmethod
    def generate_probe_signal(
        fs: float,
        duration: float,
        signal_type: str,
        freq: float = 1000.0,
        freq_end: float = 4000.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generuje próbki sygnału sondującego wybranego typu."""
        n = int(fs * duration)
        t = np.linspace(0, duration, n, endpoint=False)

        if signal_type == "sinusoid":
            return amplitude * np.sin(2 * np.pi * freq * t)

        elif signal_type == "chirp":
            k = (freq_end - freq) / duration
            phase = 2 * np.pi * (freq * t + 0.5 * k * t ** 2)
            return amplitude * np.sin(phase)

        elif signal_type == "rectangular":
            return amplitude * np.ones(n)

        elif signal_type == "gaussian_pulse":
            center = duration / 2
            sigma = duration / 8
            return amplitude * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))

        else:
            raise ValueError(f"Nieznany typ sygnału: {signal_type}")

    # ------------------------------------------------------------------
    # Główna metoda symulacji
    # ------------------------------------------------------------------
    @staticmethod
    def simulate(
        fs: float,
        speed: float,
        distance: float,
        method: str = "via_convolution",   # "direct" | "via_convolution"
        signal_type: str = "chirp",
        signal_duration: float = 0.05,
        signal_freq: float = 1000.0,
        signal_freq_end: float = 4000.0,
        noise_level: float = 0.0,
        total_duration: Optional[float] = None,
    ) -> SimulationResult:
        """
        Symulacja radarowego pomiaru odległości.

        Korelacja wzajemna R_ys[k] = Σ_n y[n] · s[n-k]:
          - sygnał_a (x) = y  – bufor odebrany  (długość N, duży)
          - sygnał_b (h) = s  – probe / wzorzec  (długość M, mały)

        Wyjście Filter zwraca lagi od -(M-1) do (N-1).
        Przeindeksowanie do tablicy 0-based: indeks i ↔ lag i - (M-1).
        Szukamy maksimum dla lagów k ≥ 0 (echo nie może przyjść przed wysłaniem).
        """
        if method not in ("direct", "via_convolution"):
            raise ValueError(f"Nieznana metoda: {method!r}. Wybierz 'direct' lub 'via_convolution'.")

        # --- Oczekiwane opóźnienie (tam + z powrotem) -----------------
        expected_delay = 2.0 * distance / speed
        delay_samples = int(round(expected_delay * fs))

        # --- Sygnał sondujący (probe) ---------------------------------
        probe = DistanceSimulator.generate_probe_signal(
            fs=fs,
            duration=signal_duration,
            signal_type=signal_type,
            freq=signal_freq,
            freq_end=signal_freq_end,
        )
        n_probe = len(probe)
        probe_t = np.linspace(0, signal_duration, n_probe, endpoint=False)

        # --- Bufor czasowy -------------------------------------------
        if total_duration is None:
            total_duration = signal_duration + expected_delay + 3 * signal_duration
        n_total = int(fs * total_duration)
        t_axis = np.linspace(0, total_duration, n_total, endpoint=False)

        # --- Sygnał wysłany (probe w buforze 0..n_total) -------------
        sent = np.zeros(n_total)
        sent[:n_probe] = probe

        # --- Sygnał odebrany (echo opóźnione o delay_samples) --------
        received = np.zeros(n_total)
        end_idx = min(delay_samples + n_probe, n_total)
        copy_len = end_idx - delay_samples
        if copy_len > 0:
            received[delay_samples:end_idx] = probe[:copy_len]

        # --- Szum addytywny ------------------------------------------
        if noise_level > 0.0:
            rng = np.random.default_rng(42)
            received = received + noise_level * rng.standard_normal(n_total)

        # =================================================================
        # KORELACJA WZAJEMNA przez Filter (obie metody)
        #
        #   signal_a = received  (y, długość N)
        #   signal_b = probe     (s, długość M)
        #
        #   Lagi wyjściowe z Filter: k ∈ {-(M-1), ..., N-1}
        #   Wyjście przeindeksowane: indeks 0 ↔ k = -(M-1)
        # =================================================================
        sig_received = Signal(start_time=0.0, sampling_freq=fs, amplitudes=received)
        sig_probe    = Signal(start_time=0.0, sampling_freq=fs, amplitudes=probe)

        if method == "direct":
            corr_signal = Filter.cross_correlation_direct(sig_received, sig_probe)
        else:
            corr_signal = Filter.cross_correlation_via_convolution(sig_received, sig_probe)

        # Lagi i wartości korelacji (tablica 0-based, przeindeksowana przez Filter)
        lags_samples = corr_signal.lags          # np.ndarray, k od -(M-1) do N-1
        corr_values  = corr_signal.amplitudes    # wartości R_ys[k], indeks od 0
        lags_time    = lags_samples / fs

        # --- Szukaj maksimum dla k >= 0 (fizyczne opóźnienie > 0) ----
        nonneg_mask    = lags_samples >= 0
        pos_lags       = lags_samples[nonneg_mask]
        pos_corr       = corr_values[nonneg_mask]

        peak_idx_pos   = int(np.argmax(pos_corr))
        peak_lag_samp  = int(pos_lags[peak_idx_pos])
        peak_lag_time  = peak_lag_samp / fs

        measured_delay    = peak_lag_time
        measured_distance = (measured_delay * speed) / 2.0

        return SimulationResult(
            fs=fs,
            speed=speed,
            distance=distance,
            expected_delay=expected_delay,
            method=method,
            t_axis=t_axis,
            sent_signal=sent,
            received_signal=received,
            probe=probe,
            probe_t=probe_t,
            lags_samples=lags_samples,
            lags_time=lags_time,
            correlation=corr_values,
            peak_lag_samples=peak_lag_samp,
            peak_lag_time=peak_lag_time,
            measured_delay=measured_delay,
            measured_distance=measured_distance,
            delay_error=abs(measured_delay - expected_delay),
            distance_error=abs(measured_distance - distance),
            signal_type=signal_type,
            signal_duration=signal_duration,
            signal_freq=signal_freq,
            signal_freq_end=signal_freq_end,
        )
