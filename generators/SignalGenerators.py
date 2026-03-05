import numpy as np

from generators import BaseGenerator


class SinusoidalGenerator(BaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq, signal_freq, phase=0.0):
        """
        Inicjalizuje generator sygnału sinusoidalnego.

        Dodatkowe parametry względem BaseGenerator:
        :param signal_freq: Częstotliwość generowanego sygnału w hercach (Hz).
        :param phase: Przesunięcie fazowe w radianach (domyślnie 0.0).
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)

        self.signal_freq = signal_freq
        self.phase = phase

    def generate(self):
        """
        Generuje sygnał sinusoidalny.
        """
        t = self.get_time_axis()

        signal = self.A * np.sin(2 * np.pi * self.signal_freq * t + self.phase)

        return signal


class HalfWaveSineGenerator(BaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq, signal_freq, phase=0.0):
        """
        Inicjalizuje generator sygnału sinusoidalnego wyprostowanego jednopołówkowo.
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)
        self.signal_freq = signal_freq
        self.phase = phase

    def generate(self):
        """
        Generuje sygnał wyprostowany jednopołówkowo.
        """
        t = self.get_time_axis()

        # zapytac czy mozna takim sposobem to zrobic, czy wzorem
        sine_wave = self.A * \
            np.sin(2 * np.pi * self.signal_freq * t + self.phase)

        rectified_signal = np.maximum(0, sine_wave)

        return rectified_signal


class FullWaveSineGenerator(BaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq, signal_freq, phase=0.0):
        """
        Inicjalizuje generator sygnału sinusoidalnego wyprostowanego dwupołówkowo.
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)
        self.signal_freq = signal_freq
        self.phase = phase

    def generate(self):
        """
        Generuje sygnał wyprostowany dwupołówkowo.
        """
        t = self.get_time_axis()

        # Generujemy bazową sinusoidę (bez mnożenia od razu przez amplitudę,
        # choć nie ma to znaczenia matematycznego, jeśli A jest dodatnie)
        base_sine = np.sin(2 * np.pi * self.signal_freq * t + self.phase)

        # Wykonujemy prostowanie dwupołówkowe (wartość bezwzględna) i skalujemy
        rectified_signal = self.A * np.abs(base_sine)

        return rectified_signal


class RectangularGenerator(BaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq, T, kw):
        """
        Inicjalizuje generator sygnału prostokątnego.

        Dodatkowe parametry względem BaseGenerator:
        :param T: Okres sygnału (w sekundach).
        :param kw: Współczynnik wypełnienia sygnału (wartość z zakresu od 0.0 do 1.0).
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)
        self.T = T
        self.kw = kw

    def generate(self):
        """
        Generuje sygnał prostokątny na podstawie podanego wzoru.
        """
        t = self.get_time_axis()

        # len of high state
        impulse_duration = self.kw * self.T

        relative_time = (t - self.t1) % self.T

        signal = np.where(relative_time < impulse_duration, self.A, 0.0)

        return signal


class SymmetricRectangularGenerator(BaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq, T, kw=0.5):
        """
        Inicjalizuje generator sygnału prostokątnego symetrycznego.

        Dodatkowe parametry względem BaseGenerator:
        :param T: Okres sygnału (w sekundach).
        :param kw: Współczynnik wypełnienia (wartość z zakresu od 0.0 do 1.0, domyślnie 0.5).
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)
        self.T = T
        self.kw = kw

    def generate(self):
        """
        Generuje sygnał prostokątny symetryczny.
        """
        t = self.get_time_axis()

        impulse_duration = self.kw * self.T

        relative_time = (t - self.t1) % self.T

        signal = np.where(relative_time < impulse_duration, self.A, -self.A)

        return signal


class TriangularGenerator(BaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq, T, kw=0.5):
        """
        Inicjalizuje generator sygnału trójkątnego (w tym piłokształtnego).

        Dodatkowe parametry względem BaseGenerator:
        :param T: Okres sygnału (w sekundach).
        :param kw: Współczynnik wypełnienia (wartość z zakresu od 0.0 do 1.0).
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)
        self.T = T
        self.kw = kw

    def generate(self):
        """
        Generuje sygnał trójkątny/piłokształtny na podstawie podanego wzoru.
        """
        t = self.get_time_axis()

        t_rel = (t - self.t1) % self.T

        signal = np.zeros_like(t)

        mask_1 = t_rel < (self.kw * self.T)
        mask_2 = ~mask_1

        if self.kw > 0:
            signal[mask_1] = (self.A / (self.kw * self.T)) * t_rel[mask_1]

        if self.kw < 1:
            signal[mask_2] = (-self.A / (self.T * (1 - self.kw))) * \
                t_rel[mask_2] + (self.A / (1 - self.kw))

        return signal


class UnitStepGenerator(BaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq, ts):
        """
        Inicjalizuje generator skoku jednostkowego.

        Dodatkowe parametry względem BaseGenerator:
        :param ts: Czas skoku (w sekundach).
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)
        self.ts = ts

    def generate(self):
        """
        Generuje sygnał skoku jednostkowego.
        """
        t = self.get_time_axis()

        #  t < ts
        signal = np.zeros_like(t)

        # t > ts -> A
        signal[t > self.ts] = self.A

        # isclose due to precision issues
        signal[np.isclose(t, self.ts)] = 0.5 * self.A

        return signal


class DiscreteBaseGenerator(BaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq):
        """
        Rozszerza BaseGenerator o obsługę osi dyskretnej (numery próbek).
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)

        # Obliczamy całkowitą liczbę próbek (podobnie jak w klasie bazowej)
        num_samples = int(self.d * self.f)

        # Obliczamy początkowy indeks próbki n1 na podstawie czasu startowego t1
        # Zakładamy, że t=0 odpowiada n=0
        self.n1 = int(round(self.t1 * self.f))

        # Tworzymy oś próbek (zbiór liczb całkowitych: n1, n1+1, n1+2, ...)
        self.sample_axis = np.arange(self.n1, self.n1 + num_samples)

    def get_sample_axis(self):
        """
        Zwraca wygenerowaną oś numerów próbek (n).
        """
        return self.sample_axis


class UnitImpulseGenerator(DiscreteBaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq, ns):
        """
        Inicjalizuje generator dyskretnego impulsu jednostkowego (delty Kroneckera).

        Dodatkowe parametry względem DiscreteBaseGenerator:
        :param ns: Numer próbki, dla której następuje skok amplitudy.
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)
        self.ns = ns

    def generate(self):
        """
        Generuje impuls jednostkowy w dziedzinie dyskretnej.
        """
        n = self.get_sample_axis()

        signal = np.zeros_like(n, dtype=float)

        signal[n == self.ns] = self.A

        return signal


class ImpulseNoiseGenerator(DiscreteBaseGenerator):
    def __init__(self, amplitude, start_time, duration, sampling_freq, p):
        """
        Inicjalizuje generator szumu impulsowego.

        Dodatkowe parametry względem DiscreteBaseGenerator:
        :param p: Prawdopodobieństwo wystąpienia wartości A (z zakresu od 0.0 do 1.0).
        """
        super().__init__(amplitude, start_time, duration, sampling_freq)

        if not (0.0 <= p <= 1.0):
            raise ValueError(
                "Prawdopodobieństwo p musi mieścić się w przedziale [0.0, 1.0]")
        self.p = p

    def generate(self):
        """
        Generuje szum impulsowy.
        """
        n = self.get_sample_axis()
        num_samples = len(n)

        random_values = np.random.rand(num_samples)

        signal = np.where(random_values < self.p, self.A, 0.0)

        return signal
