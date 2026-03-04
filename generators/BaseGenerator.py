from abc import ABC, abstractmethod
import numpy as np

class BaseGenerator(ABC):
    def __init__(self, amplitude, start_time, duration, sampling_freq):
        self.A = amplitude
        self.t1 = start_time
        self.d = duration
        self.f = sampling_freq

        num_samples = int(self.d * self.f)
        self.time_axis = np.linspace(self.t1, self.t1 + self.d, num_samples, endpoint=False)

    @abstractmethod
    def generate(self):
        raise NotImplementedError("Subclasses must implement the generate method.")
    
    def get_time_axis(self):
        return self.time_axis