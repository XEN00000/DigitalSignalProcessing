import numpy as np
from core.Signal import Signal


class Converters:

    @staticmethod
    def sample_signal(signal, new_f):
        """ 
        (S1) Próbkowanie równomierne. 
        Traktuje sygnał o wysokiej rozdzielczości jak analogowy i próbkuję go 
        z niższą częstotliwością
        """
        t = signal.time_axis
        duration = signal.num_samples / signal.f
        new_num_samples = int(duration * new_f)

        new_time_axis = np.linspace(
            signal.t1, signal.t1 + duration, new_num_samples, endpoint=False)

        sampled_amplitudes = np.interp(new_time_axis, t, signal.amplitudes)

        return Signal(
            start_time=signal.t1,
            sampling_freq=new_f,
            amplitudes=sampled_amplitudes,
            is_complex=signal.is_complex
        )

    @staticmethod
    def quantize_truncation(signal, num_bits):
        """ 
        (Q1) Kwantyzacja równomierna z obcięciem na podstawie zadanej liczby bitów
        """
        levels = 2 ** num_bits
        min_val = np.min(signal.amplitudes)
        max_val = np.max(signal.amplitudes)

        if max_val == min_val:
            return Signal(signal.t1, signal.f, signal.amplitudes, signal.is_complex)

        # Krok kwantyzacji
        delta = (max_val - min_val) / levels

        # Obcięcie -floor
        quantized_amplitudes = np.floor(
            (signal.amplitudes - min_val) / delta) * delta + min_val

        # clipowanie do zakresu min i max
        quantized_amplitudes = np.clip(quantized_amplitudes, min_val, max_val)

        return Signal(
            start_time=signal.t1,
            sampling_freq=signal.f,
            amplitudes=quantized_amplitudes,
            is_complex=signal.is_complex
        )

    @staticmethod
    def reconstruct_zoh(sampled_signal, original_f):
        """ 
        (R1) Ekstrapolacja zerowego rzędu
        Utrzymuje wartość z próbki do kolejnej próbki
        """
        duration = sampled_signal.num_samples / sampled_signal.f
        new_num_samples = int(duration * original_f)
        t_reconstructed = np.linspace(
            sampled_signal.t1, sampled_signal.t1 + duration, new_num_samples, endpoint=False)

        Ts = 1.0 / sampled_signal.f

        indices = np.floor(
            (t_reconstructed - sampled_signal.t1) / Ts).astype(int)
        indices = np.clip(indices, 0, sampled_signal.num_samples - 1)

        reconstructed_amplitudes = sampled_signal.amplitudes[indices]

        return Signal(
            start_time=sampled_signal.t1,
            sampling_freq=original_f,
            amplitudes=reconstructed_amplitudes,
            is_complex=sampled_signal.is_complex
        )

    @staticmethod
    def reconstruct_sinc(sampled_signal, original_f, num_samples_window=None):
        """ 
        (R3) Rekonstrukcja w oparciu o funkcję sinc
        :param num_samples_window: ograniczenie liczby uwzględnianych sąsiednich próbek.
                                   Jeżeli None, zlicza wszystkie.
        """
        duration = sampled_signal.num_samples / sampled_signal.f
        new_num_samples = int(duration * original_f)
        t_reconstructed = np.linspace(
            sampled_signal.t1, sampled_signal.t1 + duration, new_num_samples, endpoint=False)

        Ts = 1.0 / sampled_signal.f
        reconstructed_amplitudes = np.zeros(new_num_samples)

        n_values = np.arange(sampled_signal.num_samples)

        for i, t in enumerate(t_reconstructed):
            # Normalizacja do funkcji sinc: (t/Ts - n)
            sinc_args = (t - sampled_signal.t1) / Ts - n_values

            if num_samples_window is not None:
                # Obliczenie lokalnego przedziału
                current_n = (t - sampled_signal.t1) / Ts
                mask = np.abs(n_values - current_n) <= num_samples_window
                valid_sinc_args = sinc_args[mask]
                reconstructed_amplitudes[i] = np.sum(
                    sampled_signal.amplitudes[mask] * np.sinc(valid_sinc_args))
            else:
                reconstructed_amplitudes[i] = np.sum(
                    sampled_signal.amplitudes * np.sinc(sinc_args))

        return Signal(
            start_time=sampled_signal.t1,
            sampling_freq=original_f,
            amplitudes=reconstructed_amplitudes,
            is_complex=sampled_signal.is_complex
        )
