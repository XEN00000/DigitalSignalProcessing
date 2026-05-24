import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationResult:
    """Wyniki symulacji pomiaru odległości przez korelację."""

    # Parametry wejściowe
    fs: float
    speed: float
    distance: float
    expected_delay: float

    # Sygnały
    t_sent: np.ndarray = field(default_factory=lambda: np.array([]))
    sent_signal: np.ndarray = field(default_factory=lambda: np.array([]))
    t_received: np.ndarray = field(default_factory=lambda: np.array([]))
    received_signal: np.ndarray = field(default_factory=lambda: np.array([]))

    # Korelacja
    lags_time: np.ndarray = field(default_factory=lambda: np.array([]))
    correlation: np.ndarray = field(default_factory=lambda: np.array([]))
    peak_lag_time: float = 0.0
    peak_lag_samples: int = 0

    # Wyniki
    measured_delay: float = 0.0
    measured_distance: float = 0.0
    delay_error: float = 0.0
    distance_error: float = 0.0

    # Parametry sygnału sondującego
    signal_type: str = "chirp"
    signal_duration: float = 0.1
    signal_freq: float = 1000.0
    signal_freq_end: float = 4000.0


class DistanceSimulator:
    """
    Symulator pomiaru odległości na podstawie korelacji wzajemnej
    (splot z odwróconym sygnałem sondującym).
    """

    # ---------- fabryki sygnałów sondujących ----------

    @staticmethod
    def generate_probe_signal(
        fs: float,
        duration: float,
        signal_type: str,
        freq: float = 1000.0,
        freq_end: float = 4000.0,
        amplitude: float = 1.0,
    ) -> np.ndarray:
        """Generuje sygnał sondujący wybranego typu."""
        n = int(fs * duration)
        t = np.linspace(0, duration, n, endpoint=False)

        if signal_type == "sinusoid":
            return amplitude * np.sin(2 * np.pi * freq * t)

        elif signal_type == "chirp":
            # Sygnał chirp (liniowo zmienne częstotliwości)
            k = (freq_end - freq) / duration
            phase = 2 * np.pi * (freq * t + 0.5 * k * t ** 2)
            return amplitude * np.sin(phase)

        elif signal_type == "rectangular":
            signal = np.zeros(n)
            half = n // 2
            signal[:half] = amplitude
            return signal

        elif signal_type == "gaussian_pulse":
            center = duration / 2
            sigma = duration / 8
            return amplitude * np.exp(-((t - center) ** 2) / (2 * sigma ** 2))

        else:
            raise ValueError(f"Nieznany typ sygnału: {signal_type}")

    # ---------- główna metoda symulacji ----------

    @staticmethod
    def simulate(
        fs: float,
        speed: float,
        distance: float,
        signal_type: str = "chirp",
        signal_duration: float = 0.1,
        signal_freq: float = 1000.0,
        signal_freq_end: float = 4000.0,
        noise_level: float = 0.0,
        total_duration: Optional[float] = None,
    ) -> SimulationResult:
        """
        Przeprowadza pełną symulację:
        1. Generuje sygnał sondujący
        2. Symuluje opóźnienie (odbicie od obiektu w odległości d)
        3. Oblicza korelację wzajemną metodą splotu
        4. Identyfikuje opóźnienie z pozycji maksimum korelacji
        5. Przelicza opóźnienie na odległość
        """
        # Oczekiwane opóźnienie całkowite (tam i z powrotem)
        expected_delay = 2.0 * distance / speed
        delay_samples = int(round(expected_delay * fs))

        # Czas trwania całego bufora
        if total_duration is None:
            total_duration = signal_duration + expected_delay + 0.05  # margines

        n_total = int(fs * total_duration)
        t_full = np.linspace(0, total_duration, n_total, endpoint=False)

        # Sygnał sondujący
        probe = DistanceSimulator.generate_probe_signal(
            fs=fs,
            duration=signal_duration,
            signal_type=signal_type,
            freq=signal_freq,
            freq_end=signal_freq_end,
        )
        n_probe = len(probe)

        # Sygnał wysłany (pełny bufor)
        sent = np.zeros(n_total)
        sent[:n_probe] = probe

        # Sygnał odebrany (opóźniony)
        received = np.zeros(n_total)
        start_idx = delay_samples
        end_idx = min(start_idx + n_probe, n_total)
        actual_len = end_idx - start_idx
        received[start_idx:end_idx] = probe[:actual_len]

        # Dodanie szumu
        if noise_level > 0.0:
            rng = np.random.default_rng(42)
            received += noise_level * rng.standard_normal(n_total)

        # Korelacja wzajemna przez splot: corr(τ) = x(t) * y(-t)
        # Splot sent z odwróconym received
        corr_raw = np.convolve(sent, received[::-1], mode='full')

        # Lagi: od -(n_total-1) do (n_total-1)
        lags = np.arange(-(n_total - 1), n_total)
        lags_time = lags / fs

        # Szukamy maksimum po stronie nieujemnych lagów (opóźnienie > 0)
        positive_mask = lags >= 0
        pos_lags = lags[positive_mask]
        pos_corr = corr_raw[positive_mask]

        peak_idx_pos = int(np.argmax(pos_corr))
        peak_lag_samples = int(pos_lags[peak_idx_pos])
        peak_lag_time = peak_lag_samples / fs

        measured_delay = peak_lag_time
        measured_distance = (measured_delay * speed) / 2.0

        result = SimulationResult(
            fs=fs,
            speed=speed,
            distance=distance,
            expected_delay=expected_delay,
            t_sent=t_full,
            sent_signal=sent,
            t_received=t_full,
            received_signal=received,
            lags_time=lags_time,
            correlation=corr_raw,
            peak_lag_time=peak_lag_time,
            peak_lag_samples=peak_lag_samples,
            measured_delay=measured_delay,
            measured_distance=measured_distance,
            delay_error=abs(measured_delay - expected_delay),
            distance_error=abs(measured_distance - distance),
            signal_type=signal_type,
            signal_duration=signal_duration,
            signal_freq=signal_freq,
            signal_freq_end=signal_freq_end,
        )

        return result
