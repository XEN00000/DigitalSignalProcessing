"""
Symulator pomiaru odległości oparty na korelacji wzajemnej.
Implementacja dla sygnału ciągłego (np. CW - Continuous Wave).
Demonstruje zjawisko wieloznaczności zasięgu dla nieprzerwanej fali sinusoidalnej.
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
    """Kompletny wynik jednego uruchomienia symulatora z falą ciągłą."""

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
    signal_type: str = "sinusoid"
    signal_duration: float = 0.0
    signal_freq: float = 1000.0
    signal_freq_end: float = 4000.0


# ---------------------------------------------------------------------------
# Symulator
# ---------------------------------------------------------------------------
class DistanceSimulator:

    @staticmethod
    def generate_probe_signal(
        fs: float,
        duration: float,
        signal_type: str,
        freq: float = 1000.0,
        freq_end: float = 4000.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generuje próbki sygnału o zadanej długości."""
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

    @staticmethod
    def simulate(
        fs: float,
        speed: float,
        distance: float,
        method: str = "via_convolution",
        signal_type: str = "sinusoid",
        signal_duration: float = 0.05, 
        signal_freq: float = 10.0,   
        signal_freq_end: float = 4000.0,
        noise_level: float = 0.0,
        total_duration: Optional[float] = None,
    ) -> SimulationResult:
        """
        Symulacja ciągłego pomiaru odległości.
        Nadajnik generuje nieprzerwany sygnał, co rodzi wieloznaczność zależną od okresu fali.
        """
        if method not in ("direct", "via_convolution"):
            raise ValueError(f"Nieznana metoda: {method!r}.")

        # --- Oczekiwane opóźnienie (tam + z powrotem) -----------------
        expected_delay = 2.0 * distance / speed
        delay_samples = int(round(expected_delay * fs))

        # --- Ustawienie czasu trwania symulacji -----------------------
        # Jeśli symulujemy falę ciągłą, traktujemy cały czas total_duration jako nadawanie.
        if total_duration is None:
            # Tworzymy wystarczająco długie okno, aby zaobserwować zjawisko dla zadanej odległości
            total_duration = max(signal_duration, expected_delay + 3.0 * (1.0 / signal_freq))
            
        n_total = int(fs * total_duration)
        t_axis = np.linspace(0, total_duration, n_total, endpoint=False)

        # --- Sygnał nadawany (Ciągły) ---------------------------------
        # Zastępujemy bufor pojedynczym, nieprzerwanym sygnałem przez cały czas t_axis.
        sent = DistanceSimulator.generate_probe_signal(
            fs=fs,
            duration=total_duration,
            signal_type=signal_type,
            freq=signal_freq,
            freq_end=signal_freq_end,
        )
        
        # Wzorzec do wykresów (opcjonalnie przyjmujemy mały fragment dla GUI, żeby nie rysować całości)
        probe = sent[:int(fs * signal_duration)] if total_duration > signal_duration else sent
        probe_t = t_axis[:len(probe)]

        # --- Sygnał odebrany (ciągłe przesunięcie) --------------------
        received = np.zeros(n_total)
        if delay_samples < n_total:
            received[delay_samples:] = sent[:n_total - delay_samples]

        # --- Szum addytywny ------------------------------------------
        if noise_level > 0.0:
            rng = np.random.default_rng(42)
            received = received + noise_level * rng.standard_normal(n_total)

        # =================================================================
        # KORELACJA WZAJEMNA
        # W radarze ciągłym (CW) korelujemy odebraną falę z nadawaną.
        # Korelacja dwóch fal sinusoidalnych jest również określowa.
        # =================================================================
        sig_received = Signal(start_time=0.0, sampling_freq=fs, amplitudes=received)
        sig_sent     = Signal(start_time=0.0, sampling_freq=fs, amplitudes=sent)

        if method == "direct":
            corr_signal = Filter.cross_correlation_direct(sig_received, sig_sent)
        else:
            corr_signal = Filter.cross_correlation_via_convolution(sig_received, sig_sent)

        lags_samples = corr_signal.lags
        corr_values  = corr_signal.amplitudes
        lags_time    = lags_samples / fs

        # --- SZUKANIE MAKSIMUM TYLKO W OBRĘBIE JEDNEGO OKRESU FALI ---
        # Dla fali ciągłej system "gubi" orientację co jeden pełny okres (T = 1/f).
        if signal_type == "sinusoid":
            T_period = 1.0 / signal_freq
        else:
            T_period = signal_duration # Fallback dla innych sygnałów

        period_samples = int(T_period * fs)
        
        # System mierzy tylko opóźnienie modulo T
        window_mask = (lags_samples >= 0) & (lags_samples < period_samples)
        valid_lags  = lags_samples[window_mask]
        valid_corr  = corr_values[window_mask]

        if len(valid_corr) > 0:
            peak_idx_in_window = int(np.argmax(valid_corr))
            peak_lag_samp      = int(valid_lags[peak_idx_in_window])
        else:
            peak_lag_samp      = 0

        peak_lag_time = peak_lag_samp / fs

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
            signal_duration=total_duration,
            signal_freq=signal_freq,
            signal_freq_end=signal_freq_end,
        )