import struct
import numpy as np
from core.Signal import Signal

class FileHandler:
    @staticmethod
    def save_to_binary(filepath, signal):
        with open(filepath, 'wb') as f:
            '''
            'dd?i' = 8+8+1+4 bytes = 21 bytes

            Nagłówek: 
                t1 (double), 
                f (double), 
                is_complex (bool), 
                num_samples (int)
            '''
            f.write(struct.pack(
                'dd?i', 
                signal.t1, 
                signal.f, 
                signal.is_complex, 
                signal.num_samples)
            )

            # complex numeber will contain more data
            if signal.is_complex:
                signal.amplitudes.astype(np.complex128).tofile(f)
            else:
                signal.amplitudes.astype(np.float64).tofile(f)

    @staticmethod
    def load_from_binary(filepath):
        with open(filepath, 'rb') as f:
            pass

    @staticmethod
    def save_to_text(filepath, signal):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"t1: {signal.t1} \n")
            f.write(f"Częstotliwość próbkowania (f): {signal.f} Hz\n")
            f.write(f"Zespolony: {signal.is_complex} \n")
            f.write(f"Liczba próbek: {signal.num_samples} \n")
            f.write("-" * 30 + "\n")
            f.write("Czas [s] \t Amplituda \n")
            for t, a in zip(signal.time_axis, signal.amplitudes):
                f.write(f"{t:.6f} \t {a} \n")