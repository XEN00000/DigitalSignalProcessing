import numpy as np

class Calculator:
    @staticmethod
    def get_full_periods_data(signal, T=None):
        if T is None or T <= 0:
            raise ValueError("Period T must be a positive number")
        duration = signal.num_samples / signal.f
        if duration <= T:
            raise ValueError("Duration of the signal must be greater than the period T")

        num_full_periods = int(duration // T)
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
