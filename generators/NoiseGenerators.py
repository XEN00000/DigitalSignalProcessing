import numpy as np
from generators.BaseGenerator import BaseGenerator

class UniformNoiseGenerator(BaseGenerator):
    def generate(self):
        # 1 / (high - low)
        return np.random.uniform(-self.A, self.A, size=len(self.time_axis))