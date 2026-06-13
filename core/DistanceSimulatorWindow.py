"""
Okno eksperymentu: Wyznaczanie opóźnienia na podstawie korelacji.

Metody:
  - bezpośrednia (wg wzoru na R_xy z pętlą)
  - przez splot   (R_xy = splot(y, s_odwrócone))
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
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.gridspec import GridSpec

from core.DistanceSimulator import DistanceSimulator, SimulationResult


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------
class SimulationWorker(QThread):
    finished = QtSignal(object)
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

        # ---- panel lewy ----
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(420)
        left_scroll.setFrameShape(QFrame.NoFrame)

        left_container = QWidget()
        lv = QVBoxLayout(left_container)
        lv.setAlignment(Qt.AlignTop)
        lv.setSpacing(6)

        lv.addWidget(self._build_params_group())
        lv.addWidget(self._build_probe_group())
        lv.addWidget(self._build_method_group())
        lv.addWidget(self._build_run_button())
        lv.addWidget(self._build_results_group())
        lv.addWidget(self._build_log_group())

        left_scroll.setWidget(left_container)
        main_layout.addWidget(left_scroll)

        # ---- panel prawy (wykresy) ----
        self._figure = Figure(figsize=(12, 9), dpi=95)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding)

        # 1. Tworzymy pasek narzędzi Matplotlib
        self._toolbar = NavigationToolbar(self._canvas, self)

        # 2. Tworzymy kontener na prawy panel, aby ułożyć pasek nad wykresem
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        
        # 3. Dodajemy pasek i płótno do prawego kontenera
        right_layout.addWidget(self._toolbar)
        right_layout.addWidget(self._canvas, stretch=1)

        # 4. Dodajemy cały prawy kontener do głównego układu okna
        main_layout.addWidget(right_container, stretch=1)

        self._init_plots()

    # ------------------------------------------------------------------
    def _build_params_group(self) -> QGroupBox:
        gb = QGroupBox("Parametry eksperymentu")
        form = QFormLayout(gb)
        self._e = {}

        for key, lbl in [
            ("fs",       "Częstotliwość próbkowania f_s [Hz]:"),
            ("speed",    "Prędkość fali V [m/s]:"),
            ("distance", "Zadana odległość obiektu d [m]:"),
        ]:
            le = QLineEdit(self._DEFAULTS[key])
            form.addRow(lbl, le)
            self._e[key] = le

        self._lbl_dt = QLabel("–")
        form.addRow("Oczekiwane opóźnienie Δt [s]:", self._lbl_dt)

        for key in ("speed", "distance"):
            self._e[key].textChanged.connect(self._update_expected_delay)
        self._update_expected_delay()
        return gb

    def _build_probe_group(self) -> QGroupBox:
        gb = QGroupBox("Sygnał sondujący s[n]")
        form = QFormLayout(gb)

        self._cb_probe_type = QComboBox()
        self._cb_probe_type.addItems([
            "chirp (liniowo-częstotliwościowy)",
            "sinusoid (jednoczęstotliwościowy)",
            "gaussian_pulse (impuls Gaussa)",
            "rectangular (prostokątny)",
        ])
        form.addRow("Typ:", self._cb_probe_type)
        self._cb_probe_type.currentIndexChanged.connect(
            self._on_probe_type_changed)

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

        return gb

    def _build_method_group(self) -> QGroupBox:
        gb = QGroupBox("Metoda obliczenia korelacji wzajemnej")
        form = QFormLayout(gb)

        self._cb_method = QComboBox()
        self._cb_method.addItems([
            "Korelacja przez splot",
            "Korelacja bezpośrednia",
        ])
        form.addRow("Metoda:", self._cb_method)

        # etykieta objaśniająca
        self._lbl_method_info = QLabel("")
        self._lbl_method_info.setWordWrap(True)
        form.addRow(self._lbl_method_info)
        self._cb_method.currentIndexChanged.connect(self._on_method_changed)
        self._on_method_changed()

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
                      height_ratios=[1, 1, 1.4])

        self._ax_sent = self._figure.add_subplot(gs[0, 0])
        self._ax_received = self._figure.add_subplot(gs[0, 1])
        self._ax_probe = self._figure.add_subplot(gs[1, 0])
        self._ax_corr = self._figure.add_subplot(gs[1, 1])
        self._ax_corr_zoom = self._figure.add_subplot(gs[2, :])

        self._figure.subplots_adjust(
            left=0.07, right=0.98, top=0.95, bottom=0.07,
            hspace=0.60, wspace=0.30
        )
        self._canvas.draw()

    # ------------------------------------------------------------------
    # Logika pomocnicza
    # ------------------------------------------------------------------
    def _update_expected_delay(self):
        try:
            V = float(self._e["speed"].text())
            d = float(self._e["distance"].text())
            self._lbl_dt.setText(f"{2.0 * d / V:.6f} s")
        except ValueError:
            self._lbl_dt.setText("–")

    def _on_probe_type_changed(self):
        sig_key = self._cb_probe_type.currentText().split()[0]
        has_chirp = (sig_key == "chirp")
        has_freq = sig_key not in ("gaussian_pulse", "rectangular")
        self._lbl_freq_end.setVisible(has_chirp)
        self._le_freq_end.setVisible(has_chirp)
        self._lbl_freq.setVisible(has_freq)
        self._le_freq.setVisible(has_freq)

    def _on_method_changed(self):
        idx = self._cb_method.currentIndex()
        if idx == 0:
            self._lbl_method_info.setText(
                ""
            )
        else:
            self._lbl_method_info.setText(
                ""
            )

    def _selected_method(self) -> str:
        return "via_convolution" if self._cb_method.currentIndex() == 0 else "direct"

    def _log_msg(self, msg: str):
        self._log.append(msg)

    # ------------------------------------------------------------------
    # Uruchomienie symulacji
    # ------------------------------------------------------------------
    def _run_simulation(self):
        try:
            fs = float(self._e["fs"].text())
            speed = float(self._e["speed"].text())
            distance = float(self._e["distance"].text())
            duration = float(self._e["probe_duration"].text())
            freq = float(self._le_freq.text()
                         ) if self._le_freq.isVisible() else 1000.0
            freq_end = float(self._le_freq_end.text()
                             ) if self._le_freq_end.isVisible() else freq
            noise = float(self._e["noise_level"].text())
        except ValueError:
            self._log_msg("[BŁĄD] Nieprawidłowe wartości parametrów.")
            return

        sig_type = self._cb_probe_type.currentText().split()[0]
        method = self._selected_method()

        params = dict(
            fs=fs, speed=speed, distance=distance,
            method=method,
            signal_type=sig_type,
            signal_duration=duration,
            signal_freq=freq,
            signal_freq_end=freq_end,
            noise_level=noise,
        )

        self._btn_run.setEnabled(False)
        self._btn_run.setText("Obliczanie…")
        self._log_msg(
            f"[START] fs={fs} Hz | V={speed} m/s | d={distance} m | "
            f"typ={sig_type} | metoda={method} | szum={noise}"
        )

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
            f"Metoda:                   {result.method}\n\n"
            f"Zmierzone opóźnienie:     {result.measured_delay:.6f} s\n"
            f"Oczekiwane opóźnienie:    {result.expected_delay:.6f} s\n"
            f"Błąd opóźnienia:          {result.delay_error:.6f} s\n\n"
            f"Zmierzona odległość:      {result.measured_distance:.4f} m\n"
            f"Zadana odległość:         {result.distance:.4f} m\n"
            f"Błąd odległości:          {result.distance_error:.4f} m\n\n"
            f"Próbka maksimum (lag k):  {result.peak_lag_samples}"
        )
        self._log_msg(
            f"[OK] Δt={result.measured_delay:.6f}s | "
            f"d={result.measured_distance:.4f}m | "
            f"err={result.distance_error:.4f}m"
        )
        self._draw_plots(result)

    # ------------------------------------------------------------------
    # Rysowanie wykresów
    # ------------------------------------------------------------------
    def _draw_plots(self, r: SimulationResult):
        display_s = min(len(r.t_axis), int(
            r.fs * (r.signal_duration + r.expected_delay + 0.05)))

        # ---- Sygnał wysłany ----
        ax = self._ax_sent
        ax.clear()
        ax.plot(r.t_axis[:display_s],
                r.sent_signal[:display_s], color='blue', lw=0.9)
        ax.set_title("Sygnał wysłany x(t)")
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid(True)

        # ---- Sygnał odebrany ----
        ax = self._ax_received
        ax.clear()
        ax.plot(r.t_axis[:display_s],
                r.received_signal[:display_s], color='red', lw=0.9)
        ax.axvline(r.expected_delay, color='green', lw=1.0, ls='--',
                   label=f"Δt_oczek = {r.expected_delay:.4f} s")
        ax.legend(fontsize=7)
        ax.set_title("Sygnał odebrany y(t)")
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid(True)

        # ---- Sygnał sondujący (probe) ----
        ax = self._ax_probe
        ax.clear()
        ax.plot(r.probe_t, r.probe, color='green', lw=0.9)
        ax.set_title(
            f"Sygnał sondujący s[n]  (typ: {r.signal_type},  M={len(r.probe)} próbek)")
        ax.set_xlabel("Czas [s]")
        ax.set_ylabel("Amplituda")
        ax.grid(True)

        # ---- Korelacja wzajemna R_ys (lagi ≥ 0) ----
        ax = self._ax_corr
        ax.clear()
        nonneg = r.lags_samples >= 0
        t_pos = r.lags_time[nonneg]
        c_pos = r.correlation[nonneg]
        disp_c = min(len(t_pos), int(r.fs * (r.expected_delay * 1.5 + 0.05)))
        ax.plot(t_pos[:disp_c], c_pos[:disp_c], color='orange', lw=0.9)
        ax.axvline(r.peak_lag_time, color='purple', lw=1.2, ls='--',
                   label=f"max @ k={r.peak_lag_samples}")
        ax.legend(fontsize=7)
        method_lbl = "przez splot" if r.method == "via_convolution" else "bezpośrednia"
        ax.set_title(
            f"Korelacja wzajemna R_ys[k]  (metoda: {method_lbl},  lagi ≥ 0)")
        ax.set_xlabel("Lag τ [s]")
        ax.set_ylabel("R_ys(τ)")
        ax.grid(True)

        # ---- Korelacja – zoom wokół maksimum ----
        ax = self._ax_corr_zoom
        ax.clear()
        peak_t = r.peak_lag_time
        half_win = max(r.expected_delay * 0.12, 8 / r.fs)
        lo = max(peak_t - half_win, 0.0)
        hi = peak_t + half_win
        zm = (r.lags_time >= lo) & (r.lags_time <= hi)
        t_z = r.lags_time[zm]
        c_z = r.correlation[zm]

        if len(t_z) > 0:
            ax.plot(t_z, c_z, color='orange', lw=1.2)
            ax.axvline(r.peak_lag_time, color='purple', lw=1.5, ls='--',
                       label=f"Zmierzone Δt = {r.peak_lag_time:.6f} s  (k={r.peak_lag_samples})")
            ax.axvline(r.expected_delay, color='green', lw=1.0, ls=':',
                       label=f"Oczekiwane Δt = {r.expected_delay:.6f} s")

            peak_val = float(np.max(c_z))
            ax.annotate(
                f"k_max  = {r.peak_lag_samples} próbek\n"
                f"Δt_meas = {r.measured_delay:.6f} s\n"
                f"d_meas  = {r.measured_distance:.4f} m\n"
                f"błąd_d  = {r.distance_error:.4f} m",
                xy=(r.peak_lag_time, peak_val),
                xytext=(r.peak_lag_time + half_win * 0.35, peak_val * 0.72),
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.4",
                          fc="lightyellow", alpha=0.9, ec="gray"),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
            )

        ax.legend(fontsize=8)
        ax.set_title(
            "Korelacja wzajemna R_ys[k] – powiększenie wokół maksimum"
            f"  →  d_meas = {r.measured_distance:.4f} m"
        )
        ax.set_xlabel("Lag τ [s]")
        ax.set_ylabel("R_ys(τ)")
        ax.grid(True)

        self._figure.subplots_adjust(
            left=0.07, right=0.98, top=0.95, bottom=0.07,
            hspace=0.60, wspace=0.30
        )
        self._canvas.draw()
