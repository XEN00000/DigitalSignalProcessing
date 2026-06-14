"""
TransformWindow.py – zakładka GUI dla Zadania 4 (Transformacje sygnałów).

Zawiera:
  - Generowanie sygnału testowego S2 (f_pr=16 Hz)
  - DFT / FFT (z pomiarem czasu)
  - WHT (Walsha-Hadamarda) / Fast WHT (z pomiarem czasu)
  - Wizualizacja: sygnał wejściowy, widmo amplitudy, widmo fazy, sygnał odtworzony
"""

import time
import numpy as np
from PySide6.QtWidgets import (QWidget, QHBoxLayout, QVBoxLayout, QGroupBox,
                               QLabel, QComboBox, QLineEdit, QPushButton,
                               QTextEdit, QSizePolicy)
from PySide6.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from core.Transforms import dft, idft, fft, ifft, wht, iwht, fast_wht, ifast_wht


# ---------------------------------------------------------------------------
# Sygnały testowe (f_pr = 16 Hz)
# ---------------------------------------------------------------------------

def _generate_S2(N: int = 16, f_pr: float = 16.0) -> tuple[np.ndarray, np.ndarray]:
    """
    S2(t) = 2·sin(2π/2·t) + sin(2π/1·t) + 5·sin(2π/0.5·t)
          = 2·sin(π·t) + sin(2π·t) + 5·sin(4π·t)

    Składowe: f=0.5 Hz (amp=2), f=1 Hz (amp=1), f=2 Hz (amp=5).
    """
    t = np.arange(N) / f_pr
    s = (2.0 * np.sin(2 * np.pi / 2.0 * t)
         + np.sin(2 * np.pi / 1.0 * t)
         + 5.0 * np.sin(2 * np.pi / 0.5 * t))
    return t, s


# ---------------------------------------------------------------------------
# Okno transformacji
# ---------------------------------------------------------------------------

class TransformWindow(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._current_signal: np.ndarray | None = None
        self._current_time:   np.ndarray | None = None
        self._current_spectrum: np.ndarray | None = None  # wynik transformacji
        self._f_pr: float = 16.0
        self._transform_name: str = ""

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ── lewy panel ──────────────────────────────────────────────────────
        left = QWidget()
        left.setFixedWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setAlignment(Qt.AlignTop)

        # --- Sygnał wejściowy ---
        gb_sig = QGroupBox("Sygnał wejściowy")
        vb_sig = QVBoxLayout(gb_sig)

        vb_sig.addWidget(QLabel("Wybierz sygnał:"))
        self.cb_signal = QComboBox()
        self.cb_signal.addItems(["S2 (testowy, f_pr=16 Hz)"])
        vb_sig.addWidget(self.cb_signal)

        vb_sig.addWidget(QLabel("Liczba próbek N (potęga 2):"))
        self.entry_N = QLineEdit("16")
        vb_sig.addWidget(self.entry_N)

        vb_sig.addWidget(QLabel("Częstotliwość próbkowania f_pr [Hz]:"))
        self.entry_fpr = QLineEdit("16.0")
        vb_sig.addWidget(self.entry_fpr)

        btn_load = QPushButton("Wczytaj / generuj sygnał")
        btn_load.clicked.connect(self._load_signal)
        vb_sig.addWidget(btn_load)

        left_layout.addWidget(gb_sig)

        # --- Transformacja ---
        gb_tr = QGroupBox("Transformacja")
        vb_tr = QVBoxLayout(gb_tr)

        vb_tr.addWidget(QLabel("Wybierz transformację:"))
        self.cb_transform = QComboBox()
        self.cb_transform.addItems([
            "DFT  (Dyskretna Fouriera)",
            "FFT  (Szybka Fouriera)",
            "WHT  (Walsha-Hadamarda)",
            "Fast WHT  (Szybka Walsh-Hadamarda)",
        ])
        vb_tr.addWidget(self.cb_transform)

        btn_forward = QPushButton("Oblicz transformację →")
        btn_forward.clicked.connect(self._compute_forward)
        vb_tr.addWidget(btn_forward)

        btn_inverse = QPushButton("Oblicz odwrotną transformację ←")
        btn_inverse.clicked.connect(self._compute_inverse)
        vb_tr.addWidget(btn_inverse)

        left_layout.addWidget(gb_tr)

        # --- Wyniki / log ---
        gb_log = QGroupBox("Wyniki")
        vb_log = QVBoxLayout(gb_log)
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFixedHeight(220)
        vb_log.addWidget(self.txt_log)
        left_layout.addWidget(gb_log)

        left_layout.addStretch()
        main_layout.addWidget(left)

        # ── prawy panel – wykresy ────────────────────────────────────────────
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right_layout.addWidget(self.canvas)

        main_layout.addWidget(right, stretch=1)

        # Inicjalizacja pustych wykresów
        self._init_plots()

    # -----------------------------------------------------------------------
    # Inicjalizacja / reset wykresów
    # -----------------------------------------------------------------------

    def _init_plots(self):
        self.figure.clear()
        self.ax_sig  = self.figure.add_subplot(411)  # sygnał wejściowy
        self.ax_amp  = self.figure.add_subplot(412)  # widmo amplitudy
        self.ax_phase= self.figure.add_subplot(413)  # widmo fazy
        self.ax_rec  = self.figure.add_subplot(414)  # sygnał odtworzony
        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

    # -----------------------------------------------------------------------
    # Wczytanie / generacja sygnału
    # -----------------------------------------------------------------------

    def _load_signal(self):
        try:
            N    = int(self.entry_N.text())
            f_pr = float(self.entry_fpr.text())
        except ValueError:
            self._log("❌ Niepoprawna wartość N lub f_pr.")
            return

        if N <= 0:
            self._log(f"❌ N musi być liczbą całkowitą dodatnią.")
            return

        is_pow2 = (N & (N - 1)) == 0
        if not is_pow2:
            self._log(
                f"⚠️  N={N} nie jest potęgą 2.\n"
                f"   DFT i WHT macierzowe zadziałają dla dowolnego N.\n"
                f"   FFT i Fast WHT wymagają potęgi 2 – zmień N lub wybierz DFT/WHT."
            )

        choice = self.cb_signal.currentText()
        if "S2" in choice:
            t, s = _generate_S2(N, f_pr)
            self._current_signal = s
            self._current_time   = t
            self._f_pr           = f_pr
            self._current_spectrum = None
            self._log(
                f"✅ Sygnał S2 wygenerowany  (N={N}, f_pr={f_pr} Hz).\n"
                f"   S(t) = 2·sin(2π·0.5·t) + sin(2π·1·t) + 5·sin(2π·2·t)\n"
                f"   Próbki: {np.round(s, 4)}"
            )
            self._plot_signal_only(t, s)

    # -----------------------------------------------------------------------
    # Obliczenie transformaty prostej
    # -----------------------------------------------------------------------

    def _compute_forward(self):
        if self._current_signal is None:
            self._log("❌ Brak sygnału – najpierw wczytaj / wygeneruj sygnał.")
            return

        x    = self._current_signal
        name = self.cb_transform.currentText()

        t0 = time.perf_counter()
        try:
            if "DFT" in name:
                X = dft(x)
                self._transform_name = "DFT"
            elif name.startswith("FFT"):
                N = len(x)
                if N & (N - 1):
                    self._log(f"❌ FFT wymaga N będącego potęgą 2, obecne N={N}.\n"
                              f"   Zmień N na potęgę 2 (np. 8, 16, 32) i wczytaj sygnał ponownie,\n"
                              f"   lub wybierz DFT (działa dla dowolnego N).")
                    return
                X = fft(x)
                self._transform_name = "FFT"
            elif "Fast WHT" in name:
                N = len(x)
                if N & (N - 1):
                    self._log(f"❌ Fast WHT wymaga N będącego potęgą 2, obecne N={N}.\n"
                              f"   Zmień N na potęgę 2 lub wybierz WHT macierzowe.")
                    return
                X = fast_wht(x)
                self._transform_name = "Fast WHT"
            else:  # WHT macierzowe
                X = wht(x)
                self._transform_name = "WHT"
        except Exception as e:
            self._log(f"❌ Błąd transformacji: {e}")
            return
        elapsed = (time.perf_counter() - t0) * 1000

        self._current_spectrum = X

        # Widmo amplitudy i fazy
        amp   = np.abs(X)
        phase = np.angle(X)

        N     = len(x)
        f_pr  = self._f_pr
        is_wht = self._transform_name in ("WHT", "Fast WHT")

        if is_wht:
            # WHT: oś to indeks współczynnika (sequency)
            x_axis = np.arange(N)
            self._log(
                f"✅ {self._transform_name}  (N={N}, f_pr={f_pr} Hz)\n"
                f"   Czas obliczeń: {elapsed:.3f} ms\n"
                f"   Współczynniki Walsha |X(m)|: {np.round(amp, 4)}\n"
                f"   Max amplituda przy indeksie m={np.argmax(amp)}"
            )
        else:
            # DFT/FFT: oś częstotliwości w Hz (wzór F-3)
            x_axis = np.arange(N) * f_pr / N
            self._log(
                f"✅ {self._transform_name}  (N={N}, f_pr={f_pr} Hz)\n"
                f"   Czas obliczeń: {elapsed:.3f} ms\n"
                f"   Amplitudy |X(m)|: {np.round(amp, 4)}\n"
                f"   Max amplituda przy indeksie m={np.argmax(amp)}  "
                f"(f≈{x_axis[np.argmax(amp)]:.2f} Hz)"
            )

        self._plot_full(self._current_time, x, x_axis, amp, phase, None)

    # -----------------------------------------------------------------------
    # Obliczenie transformaty odwrotnej
    # -----------------------------------------------------------------------

    def _compute_inverse(self):
        if self._current_spectrum is None:
            self._log("❌ Brak widma – najpierw oblicz transformację prostą.")
            return

        X    = self._current_spectrum
        name = self._transform_name

        t0 = time.perf_counter()
        try:
            if name == "DFT":
                x_rec = idft(X).real
            elif name == "FFT":
                x_rec = ifft(X).real
            elif name == "Fast WHT":
                x_rec = ifast_wht(X)
            else:  # WHT
                x_rec = iwht(X)
        except Exception as e:
            self._log(f"❌ Błąd odwrotnej transformacji: {e}")
            return
        elapsed = (time.perf_counter() - t0) * 1000

        # Błąd rekonstrukcji
        err = np.max(np.abs(x_rec - self._current_signal))

        self._log(
            f"✅ Odwrotna {name}  (czas: {elapsed:.3f} ms)\n"
            f"   Maksymalny błąd rekonstrukcji: {err:.2e}\n"
            f"   Odtworzone próbki: {np.round(x_rec, 4)}"
        )

        # Zaktualizuj wykres z sygnałem odtworzonym
        is_wht = self._transform_name in ("WHT", "Fast WHT")
        N = len(X)
        if is_wht:
            x_axis = np.arange(N)
        else:
            x_axis = self._freqs_for_plot()
        amp   = np.abs(X)
        phase = np.angle(X)
        self._plot_full(self._current_time, self._current_signal,
                        x_axis, amp, phase, x_rec)

    # -----------------------------------------------------------------------
    # Rysowanie
    # -----------------------------------------------------------------------

    def _freqs_for_plot(self) -> np.ndarray:
        """Oś częstotliwości dla aktualnego sygnału."""
        if self._current_signal is None:
            return np.array([])
        N = len(self._current_signal)
        return np.arange(N) * self._f_pr / N

    def _plot_signal_only(self, t: np.ndarray, s: np.ndarray):
        """Rysuje tylko sygnał wejściowy, pozostałe panele puste."""
        self.figure.clear()
        self.ax_sig   = self.figure.add_subplot(411)
        self.ax_amp   = self.figure.add_subplot(412)
        self.ax_phase = self.figure.add_subplot(413)
        self.ax_rec   = self.figure.add_subplot(414)

        self.ax_sig.stem(t, s, linefmt='C0-', markerfmt='C0o', basefmt='gray')
        self.ax_sig.set_title("Sygnał wejściowy x(n)", fontsize=10)
        self.ax_sig.set_xlabel("Czas [s]")
        self.ax_sig.set_ylabel("Amplituda")
        self.ax_sig.grid(True, alpha=0.4)

        for ax in [self.ax_amp, self.ax_phase, self.ax_rec]:
            ax.set_visible(True)
            ax.text(0.5, 0.5, "Brak danych – oblicz transformację",
                    ha='center', va='center', transform=ax.transAxes,
                    color='gray', fontsize=9)

        self.figure.tight_layout(pad=2.5)
        self.canvas.draw()

    def _plot_full(self, t, x, x_axis, amp, phase, x_rec):
        """Pełny wykres: sygnał + widmo amp + widmo fazy (lub wartości) + rekonstrukcja."""
        is_wht = self._transform_name in ("WHT", "Fast WHT")

        self.figure.clear()
        self.ax_sig   = self.figure.add_subplot(411)
        self.ax_amp   = self.figure.add_subplot(412)
        self.ax_phase = self.figure.add_subplot(413)
        self.ax_rec   = self.figure.add_subplot(414)

        # 1. Sygnał wejściowy
        self.ax_sig.stem(t, x, linefmt='C0-', markerfmt='C0o', basefmt='gray')
        self.ax_sig.set_title("Sygnał wejściowy x(n)", fontsize=10)
        self.ax_sig.set_xlabel("Czas [s]")
        self.ax_sig.set_ylabel("Amplituda")
        self.ax_sig.grid(True, alpha=0.4)

        if is_wht:
            # --- WHT: widmo amplitudy z osią sequency ---
            self.ax_amp.stem(
                x_axis, amp, linefmt='C1-', markerfmt='C1o', basefmt='gray')
            self.ax_amp.set_title(
                f"Widmo amplitudy Walsha |X(m)|  ({self._transform_name})", fontsize=10)
            self.ax_amp.set_xlabel("Indeks współczynnika (sequency)")
            self.ax_amp.set_ylabel("|X(m)|")
            self.ax_amp.grid(True, alpha=0.4)

            # --- WHT: zamiast fazy – wartości współczynników (ze znakiem) ---
            spectrum_real = self._current_spectrum
            if spectrum_real is not None:
                colors = ['C2' if v >= 0 else 'C3' for v in spectrum_real]
                markerline, stemlines, baseline = self.ax_phase.stem(
                    x_axis, spectrum_real, linefmt='C2-', markerfmt='C2o', basefmt='gray')
                self.ax_phase.set_title(
                    f"Współczynniki Walsha X(m)  ({self._transform_name})", fontsize=10)
                self.ax_phase.set_xlabel("Indeks współczynnika (sequency)")
                self.ax_phase.set_ylabel("X(m)")
                self.ax_phase.axhline(y=0, color='gray', linewidth=0.8)
                self.ax_phase.grid(True, alpha=0.4)
            else:
                self.ax_phase.text(0.5, 0.5, "Brak danych",
                                   ha='center', va='center',
                                   transform=self.ax_phase.transAxes,
                                   color='gray', fontsize=9)
        else:
            # --- Fourier: klasyczne widmo amplitudy i fazy ---
            self.ax_amp.stem(
                x_axis, amp, linefmt='C1-', markerfmt='C1o', basefmt='gray')
            self.ax_amp.set_title(
                f"Widmo amplitudy |X(m)|  ({self._transform_name})", fontsize=10)
            self.ax_amp.set_xlabel("Częstotliwość [Hz]")
            self.ax_amp.set_ylabel("|X(m)|")
            self.ax_amp.grid(True, alpha=0.4)

            self.ax_phase.stem(
                x_axis, phase, linefmt='C2-', markerfmt='C2o', basefmt='gray')
            self.ax_phase.set_title(
                f"Widmo fazy ∠X(m)  ({self._transform_name})", fontsize=10)
            self.ax_phase.set_xlabel("Częstotliwość [Hz]")
            self.ax_phase.set_ylabel("Faza [rad]")
            self.ax_phase.grid(True, alpha=0.4)

        # 4. Sygnał odtworzony (lub info)
        if x_rec is not None:
            self.ax_rec.stem(t, x_rec, linefmt='C3-', markerfmt='C3o', basefmt='gray')
            self.ax_rec.plot(t, x, 'C0--', alpha=0.5, label='oryginał')
            self.ax_rec.set_title("Sygnał odtworzony (odwrotna transformacja)", fontsize=10)
            self.ax_rec.legend(fontsize=8)
        else:
            self.ax_rec.text(0.5, 0.5, "Oblicz odwrotną transformację aby zobaczyć rekonstrukcję",
                             ha='center', va='center', transform=self.ax_rec.transAxes,
                             color='gray', fontsize=9)
            self.ax_rec.set_title("Sygnał odtworzony", fontsize=10)
        self.ax_rec.set_xlabel("Czas [s]")
        self.ax_rec.set_ylabel("Amplituda")
        self.ax_rec.grid(True, alpha=0.4)

        self.figure.tight_layout(pad=2.5)
        self.canvas.draw()

    # -----------------------------------------------------------------------
    # Pomocnicza metoda logowania
    # -----------------------------------------------------------------------

    def _log(self, msg: str):
        self.txt_log.append(msg + "\n")

    # -----------------------------------------------------------------------
    # Metoda publiczna: załaduj sygnał z zewnątrz (np. z głównego okna)
    # -----------------------------------------------------------------------

    def load_external_signal(self, time_axis: np.ndarray, amplitudes: np.ndarray,
                              f_pr: float, label: str = "Zewnętrzny"):
        """Pozwala głównemu oknu przekazać bieżący sygnał do zakładki."""
        N = len(amplitudes)
        # Przytnij do najbliższej potęgi 2 (dla FFT/WHT)
        N_pow2 = 1 << (N - 1).bit_length()
        if N_pow2 > N:
            N_pow2 >>= 1
        amplitudes = amplitudes[:N_pow2]
        time_axis  = time_axis[:N_pow2]

        self._current_signal = amplitudes
        self._current_time   = time_axis
        self._f_pr           = f_pr
        self._current_spectrum = None

        # Zaktualizuj pola
        self.entry_N.setText(str(N_pow2))
        self.entry_fpr.setText(str(f_pr))

        self._log(
            f"✅ Załadowano sygnał zewnętrzny: '{label}'\n"
            f"   N={N_pow2} (przycięto do potęgi 2 z {N}), f_pr={f_pr} Hz"
        )
        self._plot_signal_only(time_axis, amplitudes)
