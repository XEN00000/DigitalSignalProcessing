import numpy as np

class Signal:
    def __init__(self, start_time, sampling_freq, amplitudes, is_complex=False):
        self.t1 = start_time
        self.f = sampling_freq
        self.amplitudes = np.array(amplitudes)
        self.is_complex = is_complex
        self.num_samples = len(self.amplitudes)
        self.time_axis = np.linspace(
            self.t1, 
            self.t1 + self.num_samples / self.f, 
            self.num_samples, 
            endpoint=False
        )

    def _check_compability(self, other):
        if self.f != other.f:
            raise ValueError("Sampling frequencies do not match")
        if self.num_samples != other.num_samples:
            raise ValueError("Number of samples do not match")
        
    def __add__(self, other):
        self._check_compability(other)
        return Signal(
            start_time=self.t1,
            sampling_freq=self.f,
            amplitudes=self.amplitudes + other.amplitudes,
            is_complex=self.is_complex or other.is_complex
        )

    def __sub__(self, other):
        self._check_compability(other)
        return Signal(
            start_time=self.t1,
            sampling_freq=self.f,
            amplitudes=self.amplitudes - other.amplitudes,
            is_complex=self.is_complex or other.is_complex
        )
    
    def __mul__(self, other):
        self._check_compability(other)
        return Signal(
            start_time=self.t1,
            sampling_freq=self.f,
            amplitudes=self.amplitudes * other.amplitudes,
            is_complex=self.is_complex or other.is_complex
        )
    
    def __trudiv__(self, other):
        self._check_compability(other)
        # case when other.amplitudes is zero will result in inf or nan which is acceptable in this context
        with np.errstate(divide='ignore', invalid='ignore'):
            # numpy will change inf or nan to 0
            result_amplitudes = np.true_divide(self.amplitudes, other.amplitudes)
        return Signal(
            start_time=self.t1,
            sampling_freq=self.f,
            amplitudes=result_amplitudes,
            is_complex=self.is_complex or other.is_complex
        )