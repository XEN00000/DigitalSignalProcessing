import struct
import numpy as np
from core.Signal import Signal


class FileHandler:
    # < = little-endian,   d = double (8B), ? = bool (1B), i = int (4B)
    HEADER_FORMAT = '<dd?i'
    HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

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
            # 'dd?i'
            header_bytes = f.read(21)
            if len(header_bytes) < 21:
                raise ValueError("Plik jest zbyt krótki lub uszkodzony.")

            t1, f_freq, is_complex, num_samples = struct.unpack(
                'dd?i', header_bytes)

            dtype = np.complex128 if is_complex else np.float64

            amplitudes = np.fromfile(f, dtype=dtype)

            signal = Signal(
                start_time=t1,
                sampling_freq=f_freq,
                amplitudes=amplitudes,
                is_complex=is_complex
            )

            return signal

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
