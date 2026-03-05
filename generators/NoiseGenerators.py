import numpy as np
from generators.BaseGenerator import BaseGenerator


class UniformNoiseGenerator(BaseGenerator):
    def generate(self):
        # 1 / (high - low)
        return np.random.uniform(-self.A, self.A, size=len(self.time_axis))


class GaussianNoiseGenerator(BaseGenerator):
    def generate(self):
        mu = 0.0
        sigma = 1.0

        num_samples = len(self.time_axis)

        noise = np.random.normal(loc=mu, scale=sigma, size=num_samples)

        # scaling by amplitude
        signal = self.A * noise

        return signal
