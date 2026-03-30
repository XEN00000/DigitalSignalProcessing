import numpy as np


class Calculator:
    @staticmethod
    def get_full_periods_data(signal, T=None):
        # non-periodic, return all samples
        if T is None or T <= 0:
            return signal.amplitudes

        duration = signal.num_samples / signal.f
        num_full_periods = int(duration // T)

        if num_full_periods == 0:
            return signal.amplitudes

        samples_to_keep = int(num_full_periods * T * signal.f)

        return signal.amplitudes[:samples_to_keep]

    @staticmethod
    def average(signal, T=None):
        full_periods_data = Calculator.get_full_periods_data(signal, T)
        return np.mean(full_periods_data)

    @staticmethod
    def abs_average(signal, T=None):
        full_periods_data = Calculator.get_full_periods_data(signal, T)
        return np.mean(np.abs(full_periods_data))

    # this is an implementation of the formula for average signal power
    # formula taken from PDF
    @staticmethod
    def average_power(signal, T=None):
        data = Calculator.get_full_periods_data(signal, T)
        return np.mean(np.square(np.abs(data)))

    @staticmethod
    def variance(signal, T=None):
        data = Calculator.get_full_periods_data(signal, T)
        mean_val = np.mean(data)
        return np.mean(np.square(np.abs(data - mean_val)))

    @staticmethod
    def rms(signal, T=None):
        return np.sqrt(Calculator.average_power(signal, T))

    @staticmethod
    def mse(original_signal, reconstructed_signal):
        """
        (C1) Błąd średniokwadratowy (MSE)
        """
        x = original_signal.amplitudes
        x_hat = reconstructed_signal.amplitudes

        if len(x) != len(x_hat):
            raise ValueError(
                "Sygnały muszą mieć tę samą długość")

        return np.mean((x - x_hat) ** 2)

    @staticmethod
    def snr(original_signal, reconstructed_signal):
        """
        (C2) Stosunek sygnał-szum (SNR - Signal to Noise Ratio) w dB
        """
        x = original_signal.amplitudes
        x_hat = reconstructed_signal.amplitudes

        if len(x) != len(x_hat):
            raise ValueError(
                "Sygnały muszą mieć tę samą długość")

        signal_power = np.sum(x ** 2)
        noise_power = np.sum((x - x_hat) ** 2)

        if noise_power == 0:
            return float('inf')

        return 10 * np.log10(signal_power / noise_power)

    @staticmethod
    def psnr(original_signal, reconstructed_signal):
        """
        (C3) Szczytowy stosunek sygnał-szum (PSNR - Peak Signal to Noise Ratio) w dB
        """
        x = original_signal.amplitudes

        mse_val = Calculator.mse(original_signal, reconstructed_signal)

        if mse_val == 0:
            return float('inf')

        max_val = np.max(np.abs(x))  # peak

        return 10 * np.log10((max_val ** 2) / mse_val)

    @staticmethod
    def md(original_signal, reconstructed_signal):
        """
        (C4) Maksymalna różnica (MD)
        """
        x = original_signal.amplitudes
        x_hat = reconstructed_signal.amplitudes

        if len(x) != len(x_hat):
            raise ValueError(
                "Sygnały muszą mieć tę samą długość")

        return np.max(np.abs(x - x_hat))

    @staticmethod
    def enob(snr_value):
        """
        Efektywna liczba bitów (ENOB - Effective Number of Bits)
        Wyliczana na podstawie teoretycznego wzoru z SNR.
        """
        if snr_value == float('inf'):
            return float('inf')

        return (snr_value - 1.76) / 6.02  # wzor z pdf
