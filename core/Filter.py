import numpy as np

from core.Signal import Signal


class Filter:
    @staticmethod
    def _validate_sampling_frequency(signal_a, signal_b):
        if signal_a.f != signal_b.f:
            raise ValueError(
                "Sygnały muszą mieć taką samą częstotliwość próbkowania.")

    @staticmethod
    def _window(window_type, num_taps):
        normalized = window_type.lower()
        if normalized == "prostokątne":
            return np.ones(num_taps)
        if normalized == "hanning":
            return np.hanning(num_taps)
        if normalized == "hamming":
            return np.hamming(num_taps)
        if normalized == "blackman":
            return np.blackman(num_taps)
        raise ValueError(f"Nieobsługiwany typ okna: {window_type}")

    @staticmethod
    def convolve_signals(signal_a, signal_b):
        Filter._validate_sampling_frequency(signal_a, signal_b)

        result_amplitudes = np.convolve(
            signal_a.amplitudes, signal_b.amplitudes, mode='full')

        result = Signal(
            start_time=signal_a.t1 + signal_b.t1,
            sampling_freq=signal_a.f,
            amplitudes=result_amplitudes,
            is_complex=signal_a.is_complex or signal_b.is_complex
        )
        result.plot_title = "Splot sygnałów"
        result.plot_xlabel = "Czas [s]"
        result.plot_ylabel = "Amplituda"
        return result

    @staticmethod
    def design_lowpass_fir(sampling_freq, cutoff_freq, num_taps, window_type="hanning"):
        if num_taps < 3:
            raise ValueError(
                "Liczba współczynników filtru musi być większa lub równa 3.")
        if num_taps % 2 == 0:
            raise ValueError(
                "Liczba współczynników filtru musi być nieparzysta.")
        if cutoff_freq <= 0 or cutoff_freq >= sampling_freq / 2:
            raise ValueError(
                "Częstotliwość odcięcia musi należeć do zakresu (0, fs/2).")

        n = np.arange(num_taps)
        mid = (num_taps - 1) / 2
        normalized_cutoff = cutoff_freq / sampling_freq

        ideal_impulse_response = 2 * normalized_cutoff * np.sinc(
            2 * normalized_cutoff * (n - mid)
        )

        window = Filter._window(window_type, num_taps)
        fir = ideal_impulse_response * window

        fir_sum = np.sum(fir)
        if not np.isclose(fir_sum, 0.0):
            fir = fir / fir_sum

        return fir

    @staticmethod
    def design_bandpass_fir(sampling_freq, low_cutoff, high_cutoff, num_taps, window_type="hanning"):
        if low_cutoff <= 0 or high_cutoff <= 0:
            raise ValueError("Częstotliwości graniczne muszą być dodatnie.")
        if low_cutoff >= high_cutoff:
            raise ValueError(
                "Dolna częstotliwość graniczna musi być mniejsza od górnej.")
        if high_cutoff >= sampling_freq / 2:
            raise ValueError(
                "Górna częstotliwość graniczna musi być mniejsza od fs/2.")

        band_width = high_cutoff - low_cutoff
        center_frequency = (high_cutoff + low_cutoff) / 2

        base_lowpass = Filter.design_lowpass_fir(
            sampling_freq=sampling_freq,
            cutoff_freq=band_width / 2,
            num_taps=num_taps,
            window_type=window_type
        )

        n = np.arange(num_taps)
        mid = (num_taps - 1) / 2
        modulation = 2 * np.cos(2 * np.pi * center_frequency *
                                (n - mid) / sampling_freq)

        return base_lowpass * modulation

    @staticmethod
    def filter_signal(signal, fir_coefficients):
        filtered_amplitudes = np.convolve(
            signal.amplitudes, fir_coefficients, mode='full')
        result = Signal(
            start_time=signal.t1,
            sampling_freq=signal.f,
            amplitudes=filtered_amplitudes,
            is_complex=signal.is_complex
        )
        result.plot_title = "Sygnał po filtracji FIR"
        result.plot_xlabel = "Czas [s]"
        result.plot_ylabel = "Amplituda"
        return result

    @staticmethod
    def _build_correlation_signal(correlation_values, lags, signal_a, signal_b):
        time_axis = lags / signal_a.f
        result = Signal(
            start_time=time_axis[0],
            sampling_freq=signal_a.f,
            amplitudes=correlation_values,
            is_complex=signal_a.is_complex or signal_b.is_complex
        )
        result.time_axis = time_axis
        result.lags = lags
        result.plot_title = "Korelacja wzajemna"
        result.plot_xlabel = "Przesunięcie [s]"
        result.plot_ylabel = "Wartość korelacji"
        return result

    @staticmethod
    def cross_correlation_direct(signal_a, signal_b):
        Filter._validate_sampling_frequency(signal_a, signal_b)

        x = signal_a.amplitudes
        y = signal_b.amplitudes
        n = len(x)
        m = len(y)
        lags = np.arange(-(m - 1), n)
        corr = np.zeros(len(lags), dtype=np.result_type(x, y, np.float64))

        for idx, lag in enumerate(lags):
            x_start = max(0, lag)
            y_start = max(0, -lag)
            overlap = min(n - x_start, m - y_start)
            if overlap > 0:
                corr[idx] = np.dot(
                    x[x_start:x_start + overlap],
                    np.conjugate(y[y_start:y_start + overlap])
                )

        return Filter._build_correlation_signal(corr, lags, signal_a, signal_b)

    @staticmethod
    def cross_correlation_via_convolution(signal_a, signal_b):
        Filter._validate_sampling_frequency(signal_a, signal_b)

        x = signal_a.amplitudes
        y = signal_b.amplitudes
        corr = np.convolve(x, np.conjugate(y[::-1]), mode='full')
        lags = np.arange(-(len(y) - 1), len(x))
        return Filter._build_correlation_signal(corr, lags, signal_a, signal_b)
