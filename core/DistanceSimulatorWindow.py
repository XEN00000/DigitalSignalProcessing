"""
Okno eksperymentu: Wyznaczanie opóźnienia na podstawie korelacji / splotu.
"""

import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QFormLayout, QScrollArea,
    QSizePolicy, QFrame, QTextEdit,
)
from PySide6.QtCore import Qt, QThread, Signal as QtSignal
from PySide6.QtGui import QFont

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec

from core.DistanceSimulator import DistanceSimulator, SimulationResult


# ---------------------------------------------------------------------------
# Worker thread – by nie blokować GUI podczas obliczeń
# ---------------------------------------------------------------------------
class SimulationWorker(QThread):
    finished = QtSignal(object)   # SimulationResult
    error = QtSignal(str)

    def __init__(self, params: dict):
        super().__init__()
        self.params = params

    def run(self):
        try:
            result = DistanceSimulator.simulate(**self.params)
            self.finished.emit(result)
        except Exception as exc:
            self.error.emit(str(exc))


# ---------------------------------------------------------------------------
# Główny widget eksperymentu
# ---------------------------------------------------------------------------
class DistanceSimulatorWindow(QWidget):
    """
    Zakładka do eksperymentu z wyznaczaniem opóźnienia
    na podstawie korelacji wzajemnej obliczanej przez splot.
    """

    _DEFAULTS = {
        "fs":             "10000",
        "speed":          "343",
        "distance":       "17.15",
        "probe_duration": "0.05",
        "probe_freq":     "1000",
        "probe_freq_end": "4000",
        "noise_level":    "0.0",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: SimulationResult | None = None
        self._worker: SimulationWorker | None = None
        self._setup_ui()

    # ------------------------------------------------------------------
    # Budowanie interfejsu
    # ------------------------------------------------------------------
    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(6, 6, 6, 6)
        main_layout.setSpacing(6)

        # ---- Panel lewy (parametry + wyniki) -------------------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(320)
        left_scroll.setFrameShape(QFrame.NoFrame)

        left_container = QWidget()
        left_vbox = QVBoxLayout(left_container)
        left_vbox.setAlignment(Qt.AlignTop)
        left_vbox.setSpacing(6)

        left_vbox.addWidget(self._build_params_group())
        left_vbox.addWidget(self._build_probe_signal_group())
        left_vbox.addWidget(self._build_run_button())
        left_vbox.addWidget(self._build_results_group())
        left_vbox.addWidget(self._build_log_group())

        left_scroll.setWidget(left_container)
        main_layout.addWidget(left_scroll)

        # ---- Panel prawy (wykresy) -----------------------------------
        self._figure = Figure(figsize=(11, 9), dpi=95)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        main_layout.addWidget(self._canvas, stretch=1)

        self._init_plots()

    # ------------------------------------------------------------------
    def _build_params_group(self) -> QGroupBox:
        gb = QGroupBox("Parametry eksperymentu")
        form = QFormLayout(gb)

        self._e = {}

        fields = [
            ("fs",       "Częstotliwość próbkowania f_s [Hz]:"),
            ("speed",    "Prędkość fali V [m/s]:"),
            ("distance", "Zadana odległość obiektu d [m]:"),
        ]

        for key, label in fields:
            le = QLineEdit(self._DEFAULTS[key])
            form.addRow(label, le)
            self._e[key] = le

        self._lbl_expected_delay = QLabel("–")
        form.addRow("Oczekiwane opóźnienie Δt [s]:", self._lbl_expected_delay)

        for key in ("speed", "distance"):
            self._e[key].textChanged.connect(self._update_expected_delay)
        self._update_expected_delay()

        return gb

    def _build_probe_signal_group(self) -> QGroupBox:
        gb = QGroupBox("Sygnał sondujący")
        form = QFormLayout(gb)

        self._cb_probe_type = QComboBox()
        self._cb_probe_type.addItems([
            "chirp (liniowo-częstotliwościowy)",
            "sinusoid (jednoczęstotliwościowy)",
            "gaussian_pulse (impuls Gaussa)",
            "rectangular (prostokątny)",
        ])
        form.addRow("Typ sygnału:", self._cb_probe_type)
        self._cb_probe_type.currentIndexChanged.connect(self._on_probe_type_changed)

        le_dur = QLineEdit(self._DEFAULTS["probe_duration"])
        form.addRow("Czas trwania [s]:", le_dur)
        self._e["probe_duration"] = le_dur

        self._lbl_freq = QLabel("Częstotliwość startowa [Hz]:")
        self._le_freq = QLineEdit(self._DEFAULTS["probe_freq"])
        form.addRow(self._lbl_freq, self._le_freq)
        self._e["probe_freq"] = self._le_freq

        self._lbl_freq_end = QLabel("Częstotliwość końcowa [Hz]:")
        self._le_freq_end = QLineEdit(self._DEFAULTS["probe_freq_end"])
        form.addRow(self._lbl_freq_end, self._le_freq_end)
        self._e["probe_freq_end"] = self._le_freq_end

        le_noise = QLineEdit(self._DEFAULTS["noise_level"])
        form.addRow("Poziom szumu (0 = brak):", le_noise)
        self._e["noise_level"] = le_noise

        lbl_method = QLabel("Korelacja przez splot")
        lbl_method.setWordWrap(True)
        form.addRow("Metoda obliczeniowa:", lbl_method)

        return gb

    def _build_run_button(self) -> QPushButton:
        btn = QPushButton("Uruchom symulację")
        btn.setMinimumHeight(32)
        btn.clicked.connect(self._run_simulation)
        self._btn_run = btn
        return btn

    def _build_results_group(self) -> QGroupBox:
        gb = QGroupBox("Wyniki pomiaru")
        vbox = QVBoxLayout(gb)

        self._lbl_results = QLabel(
            "Zmierzone opóźnienie:     –\n"
            "Oczekiwane opóźnienie:    –\n"
            "Błąd opóźnienia:          –\n\n"
            "Zmierzona odległość:      –\n"
            "Zadana odległość:         –\n"
            "Błąd odległości:          –\n\n"
            "Próbka maksimum (lag):    –"
        )
        self._lbl_results.setFont(QFont("Courier New", 9))
        self._lbl_results.setWordWrap(True)
        vbox.addWidget(self._lbl_results)
        return gb

    def _build_log_group(self) -> QGroupBox:
        gb = QGroupBox("Dziennik")
        vbox = QVBoxLayout(gb)
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(110)
        self._log.setFont(QFont("Courier New", 8))
        vbox.addWidget(self._log)
        return gb

    # ------------------------------------------------------------------
    # Wykresy – inicjalizacja
    # ------------------------------------------------------------------
    def _init_plots(self):
        self._figure.clear()
        gs = GridSpec(3, 2, figure=self._figure,
                      hspace=0.55, wspace=0.35,
                      left=0.08, right=0.97, top=0.95, bottom=0.07)

        self._ax_sent      = self._figure.add_subplot(gs[0, 0])
        self._ax_received  = self._figure.add_subplot(gs[0, 1])
        self._ax_probe     = self._figure.add_subplot(gs[1, 0])
        self._ax_corr      = self._figure.add_subplot(gs[1, 1])
        self._ax_corr_zoom = self._figure.add_subplot(gs[2, :])

        self._canvas.draw()

    # ------------------------------------------------------------------
    # Logika pomocnicza
    # ------------------------------------------------------------------
    def _update_expected_delay(self):
        try:
            V = float(self._e["speed"].text())
            d = float(self._e["distance"].text())
            self._lbl_expected_delay.setText(f"{2.0 * d / V:.6f} s")
        except ValueError:
            self._lbl_expected_delay.setText("–")

    def _on_probe_type_changed(self):
        sig_key = self._cb_probe_type.currentText().split()[0]
        has_freq_end = (sig_key == "chirp")
        self._lbl_freq_end.setVisible(has_freq_end)
        self._le_freq_end.setVisible(has_freq_end)
        has_freq = sig_key not in ("gaussian_pulse", "rectangular")
        self._lbl_freq.setVisible(has_freq)
        self._le_freq.setVisible(has_freq)

    def _log_msg(self, msg: str):
        self._log.append(msg)

    # ------------------------------------------------------------------
    # Uruchomienie symulacji
    # ------------------------------------------------------------------
    def _run_simulation(self):
        try:
            fs       = float(self._e["fs"].text())
            speed    = float(self._e["speed"].text())
            distance = float(self._e["distance"].text())
            duration = float(self._e["probe_duration"].text())
            freq     = float(self._le_freq.text()) if self._le_freq.isVisible() else 1000.0
            freq_end = float(self._le_freq_end.text()) if self._le_freq_end.isVisible() else freq
            noise    = float(self._e["noise_level"].text())
        except ValueError:
            self._log_msg("[BŁĄD] Nieprawidłowe wartości parametrów.")
            return

        sig_type = self._cb_probe_type.currentText().split()[0]

        params = dict(
            fs=fs, speed=speed, distance=distance,
            signal_type=sig_type,
            signal_duration=duration,
            signal_freq=freq,
            signal_freq_end=freq_end,
            noise_level=noise,
        )

        self._btn_run.setEnabled(False)
        self._btn_run.setText("Obliczanie…")
        self._log_msg(f"[START] fs={fs} Hz | V={speed} m/s | d={distance} m | "
                      f"typ={sig_type} | szum={noise}")

        self._worker = SimulationWorker(params)
        self._worker.finished.connect(self._on_simulation_done)
        self._worker.error.connect(self._on_simulation_error)
        self._worker.start()

    def _on_simulation_error(self, msg: str):
        self._log_msg(f"[BŁĄD] {msg}")
        self._btn_run.setEnabled(True)
        self._btn_run.setText("Uruchom symulację")

    def _on_simulation_done(self, result: SimulationResult):
        self._result = result
        self._btn_run.setEnabled(True)
        self._btn_run.setText("Uruchom symulację")

        self._lbl_results.setText(
            f"Zmierzone opóźnienie:     {result.measured_delay:.6f} s\n"
            f"Oczekiwane opóźnienie:    {result.expected_delay:.6f} s\n"
            f"Błąd opóźnienia:          {result.delay_error:.6f} s\n\n"
            f"Zmierzona odległość:      {result.measured_distance:.4f} m\n"
            f"Zadana odległość:         {result.distance:.4f} m\n"
            f"Błąd odległości:          {result.distance_error:.4f} m\n\n"
            f"Próbka maksimum (lag):    {result.peak_lag_samples}"
        )

        self._log_msg(
            f"[OK] Δt_meas={result.measured_delay:.6f}s | "
            f"d_meas={result.measured_distance:.4f}m | "
            f"err_d={result.distance_error:.4f}m"
        )

        self._draw_plots(result)

    # ------------------------------------------------------------------
    # Rysowanie wykresów
    # ------------------------------------------------------------------
    def _draw_plots(self, r: SimulationResult):
        display_len = min(len(r.t_sent), int(r.fs * 0.6))

        # --- Sygnał wysłany ---
        ax = self._ax_sent
        ax.clear()
        ax.plot(r.t_sent[:display_len], r.sent_signal[:display_len], color='blue', lw=0.9)
        ax.set_title("Sygnał wysłany x(t)")
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid(True)

        # --- Sygnał odebrany ---
        ax = self._ax_received
        ax.clear()
        ax.plot(r.t_received[:display_len], r.received_signal[:display_len],
                color='red', lw=0.9)
        ax.axvline(r.expected_delay, color='green', lw=1.0, ls='--',
                   label=f"Δt_oczek = {r.expected_delay:.4f}s")
        ax.legend(fontsize=7)
        ax.set_title("Sygnał odebrany y(t) (opóźniony)")
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid(True)

        # --- Sygnał sondujący ---
        ax = self._ax_probe
        ax.clear()
        n_probe = int(r.fs * r.signal_duration)
        ax.plot(r.t_sent[:n_probe], r.sent_signal[:n_probe], color='green', lw=0.9)
        ax.set_title(f"Sygnał sondujący (typ: {r.signal_type})")
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid(True)

        # --- Korelacja wzajemna (lagi >= 0) ---
        ax = self._ax_corr
        ax.clear()
        pos_mask = r.lags_time >= 0
        t_pos = r.lags_time[pos_mask]
        c_pos = r.correlation[pos_mask]
        display_corr_len = min(len(t_pos), int(r.fs * 0.6))
        ax.plot(t_pos[:display_corr_len], c_pos[:display_corr_len],
                color='orange', lw=0.9)
        ax.axvline(r.peak_lag_time, color='purple', lw=1.2, ls='--',
                   label=f"lag_max = {r.peak_lag_time:.4f}s")
        ax.legend(fontsize=7)
        ax.set_title("Korelacja wzajemna R(τ) – lagi ≥ 0")
        ax.set_xlabel("Przesunięcie τ [s]")
        ax.set_ylabel("R(τ)")
        ax.grid(True)

        # --- Korelacja – zoom wokół maksimum ---
        ax = self._ax_corr_zoom
        ax.clear()
        peak_t = r.peak_lag_time
        half_win = max(r.expected_delay * 0.15, 5 / r.fs)
        lo = max(peak_t - half_win, 0.0)
        hi = peak_t + half_win
        zoom_mask = (r.lags_time >= lo) & (r.lags_time <= hi)
        t_zoom = r.lags_time[zoom_mask]
        c_zoom = r.correlation[zoom_mask]

        if len(t_zoom) > 0:
            ax.plot(t_zoom, c_zoom, color='orange', lw=1.2)
            ax.axvline(r.peak_lag_time, color='purple', lw=1.5, ls='--',
                       label=f"Zmierzone Δt = {r.peak_lag_time:.6f} s")
            ax.axvline(r.expected_delay, color='green', lw=1.0, ls=':',
                       label=f"Oczekiwane Δt = {r.expected_delay:.6f} s")

            peak_val = float(np.max(c_zoom))
            ax.annotate(
                f"lag_max = {r.peak_lag_samples} próbek\n"
                f"Δt_meas = {r.measured_delay:.6f} s\n"
                f"d_meas  = {r.measured_distance:.4f} m\n"
                f"błąd_d  = {r.distance_error:.4f} m",
                xy=(r.peak_lag_time, peak_val),
                xytext=(r.peak_lag_time + half_win * 0.3, peak_val * 0.75),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", alpha=0.9, ec="gray"),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
            )

        ax.legend(fontsize=8)
        ax.set_title("Korelacja wzajemna – powiększenie wokół maksimum (wyznaczone opóźnienie)")
        ax.set_xlabel("Przesunięcie τ [s]")
        ax.set_ylabel("R(τ)")
        ax.grid(True)

        self._figure.tight_layout(pad=2.0)
        self._canvas.draw()
