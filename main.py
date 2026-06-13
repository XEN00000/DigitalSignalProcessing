import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGroupBox, QLabel, QComboBox,
                               QLineEdit, QPushButton, QSlider, QMessageBox,
                               QFileDialog, QFormLayout, QTextEdit, QDialog,
                               QTabWidget)
from PySide6.QtCore import Qt

from core.DistanceSimulatorWindow import DistanceSimulatorWindow
from core.TransformWindow import TransformWindow

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from core.Converters import Converters
from core.FileHandler import FileHandler
from core.Calculator import Calculator
from core.Signal import Signal
from core.Filter import Filter
from generators.NoiseGenerators import GaussianNoiseGenerator, UniformNoiseGenerator
from generators.SignalGenerators import (FullWaveSineGenerator, HalfWaveSineGenerator,
                                         ImpulseNoiseGenerator, RectangularGenerator,
                                         SinusoidalGenerator, SymmetricRectangularGenerator,
                                         TriangularGenerator, UnitImpulseGenerator,
                                         UnitStepGenerator)


class SignalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Cyfrowe Przetwarzanie Sygnałów")
        self.resize(1600, 900)

        self.current_signal_time = np.array([])
        self.current_signal_values = np.array([])
        self.reconstructed_signal = None
        self.plot_mode = "standard"

        self.signal_history = []  # historia sygnałów

        # QTabWidget jako główny widget
        self._tabs = QTabWidget()
        self.setCentralWidget(self._tabs)

        # Zakładka 1 – Generator i Analizator
        signal_tab = QWidget()
        self.main_layout = QHBoxLayout(signal_tab)
        self._tabs.addTab(signal_tab, "Generator i Analizator Sygnałów")

        # Zakładka 2 – Symulator odległości (korelacja)
        dist_sim = DistanceSimulatorWindow()
        self._tabs.addTab(dist_sim, "Wyznaczanie Opóźnienia (Korelacja)")

        # Zakładka 3 – Transformacje sygnałów (Zadanie 4)
        self._transform_tab = TransformWindow()
        self._tabs.addTab(self._transform_tab, "Transformacje (Zadanie 4)")

        self.create_widgets()

    def create_widgets(self):
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(350)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setAlignment(Qt.AlignTop)

        # operacje konwersji
        self.middle_panel = QWidget()
        self.middle_panel.setFixedWidth(350)
        self.middle_layout = QVBoxLayout(self.middle_panel)
        self.middle_layout.setAlignment(Qt.AlignTop)

        # wykresy
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        self.main_layout.addWidget(self.left_panel)
        self.main_layout.addWidget(self.middle_panel)
        self.main_layout.addWidget(self.right_panel, stretch=1)

        self._build_controls()
        self._build_plots()

    def _build_controls(self):
        # Konfiguracja Sygnału
        gb_signal = QGroupBox("Konfiguracja Sygnału")
        layout_signal = QVBoxLayout(gb_signal)

        layout_signal.addWidget(QLabel("Rodzaj sygnału:"))

        lista_sygnalow = [
            "Szum Gaussowski",
            "Szum jednostajny",
            "Szum Impulsowy",
            "Sygnał Sinusoidalny",
            "Sygnał Sinusoidalny wyprostowany jednopołówkowo",
            "Sygnał Sinusoidalny wyprostowany dwupołówkowo",
            "Sygnał Prostokątny",
            "Sygnał Prostokątny symetryczny",
            "Sygnał Trójkątny",
            "Skok jednostkowy",
            "Impuls jednostkowy"
        ]

        self.cb_signal_type = QComboBox()
        self.cb_signal_type.addItems(lista_sygnalow)
        self.cb_signal_type.setCurrentText("Sygnał Sinusoidalny")
        layout_signal.addWidget(self.cb_signal_type)

        # Powiązanie zmiany typu sygnału z odświeżaniem formularza
        self.cb_signal_type.currentIndexChanged.connect(
            self.update_param_fields)

        # Miejsce na dynamiczne parametry
        self.params_widget = QWidget()
        self.params_layout = QFormLayout(self.params_widget)
        layout_signal.addWidget(self.params_widget)

        self.entries = {}
        self.update_param_fields()

        btn_generate = QPushButton("Generuj")
        btn_generate.clicked.connect(self.generate_and_plot)
        layout_signal.addWidget(btn_generate)

        self.left_layout.addWidget(gb_signal)

        # Parametry Sygnału
        gb_stats = QGroupBox("Parametry Sygnału")
        layout_stats = QVBoxLayout(gb_stats)
        self.lbl_stats = QLabel(
            "Średnia: -\nŚrednia bezwzględna: -\nRMS: -\nWariancja: -\nMoc: -")
        layout_stats.addWidget(self.lbl_stats)
        self.left_layout.addWidget(gb_stats)

        # Ustawienia Histogramu
        gb_hist = QGroupBox("Ustawienia Histogramu")
        layout_hist = QVBoxLayout(gb_hist)
        layout_hist.addWidget(QLabel("Liczba przedziałów:"))

        self.scale_hist = QSlider(Qt.Horizontal)
        self.scale_hist.setRange(5, 20)
        self.scale_hist.setValue(10)
        self.scale_hist.setTickPosition(QSlider.TicksBelow)
        self.scale_hist.setTickInterval(1)
        self.scale_hist.valueChanged.connect(self.update_histogram)

        layout_hist.addWidget(self.scale_hist)
        self.left_layout.addWidget(gb_hist)

        # Operacje Plikowe
        gb_files = QGroupBox("Operacje Plikowe")
        layout_files = QVBoxLayout(gb_files)

        btn_save_bin = QPushButton("Zapisz do pliku (BIN)")
        btn_save_bin.clicked.connect(self.save_to_bin)
        layout_files.addWidget(btn_save_bin)

        btn_load_bin = QPushButton("Wczytaj z pliku (BIN)")
        btn_load_bin.clicked.connect(self.load_from_bin)
        layout_files.addWidget(btn_load_bin)

        btn_show_text = QPushButton("Pokaż dane tekstowo")
        btn_show_text.clicked.connect(self.show_text_data)
        layout_files.addWidget(btn_show_text)

        self.left_layout.addWidget(gb_files)

        # Wyślij bieżący sygnał do zakładki Transformacje
        gb_transform_send = QGroupBox("Transformacje (Zadanie 4)")
        vb_ts = QVBoxLayout(gb_transform_send)
        btn_send_to_transform = QPushButton("Wyślij bieżący sygnał do Transformacji →")
        btn_send_to_transform.clicked.connect(self._send_to_transform_tab)
        vb_ts.addWidget(btn_send_to_transform)
        self.left_layout.addWidget(gb_transform_send)

        gb_operations = QGroupBox("Operacje na sygnałach")
        layout_operations = QVBoxLayout(gb_operations)

        self.cb_operation = QComboBox()
        self.cb_operation.addItems(
            ["Dodawanie", "Odejmowanie", "Mnożenie", "Dzielenie",
             "Splot", "Korelacja (bezpośrednia)", "Korelacja (przez splot)"])
        layout_operations.addWidget(self.cb_operation)

        self.cb_sig1 = QComboBox()
        self.cb_sig1.addItem("Wczytaj z pliku (BIN)...")
        layout_operations.addWidget(QLabel("Sygnał 1:"))
        layout_operations.addWidget(self.cb_sig1)

        self.cb_sig2 = QComboBox()
        self.cb_sig2.addItem("Wczytaj z pliku (BIN)...")
        layout_operations.addWidget(QLabel("Sygnał 2:"))
        layout_operations.addWidget(self.cb_sig2)

        btn_perform_op = QPushButton("Wykonaj operację")
        btn_perform_op.clicked.connect(self.perform_operation)
        layout_operations.addWidget(btn_perform_op)

        self.left_layout.addWidget(gb_operations)

        gb_conversion = QGroupBox("Konwersja A/C i C/A")
        layout_conversion = QVBoxLayout(gb_conversion)

        layout_conversion.addWidget(
            QLabel("Nowa częstotliwość próbkowania (Hz):"))
        self.entry_new_f = QLineEdit("50.0")
        layout_conversion.addWidget(self.entry_new_f)

        layout_conversion.addWidget(QLabel("Liczba bitów kwantyzacji (b):"))
        self.entry_bits = QLineEdit("8")
        layout_conversion.addWidget(self.entry_bits)

        layout_conversion.addWidget(QLabel("Metoda rekonstrukcji:"))
        self.cb_reconstruction = QComboBox()
        self.cb_reconstruction.addItems(
            ["Ekstrapolacja zerowego rzędu (ZOH)", "Rekonstrukcja Sinc"])
        layout_conversion.addWidget(self.cb_reconstruction)

        layout_conversion.addWidget(
            QLabel("Okno dla Sinc (liczba próbek, puste=wszystkie):"))
        self.entry_sinc_window = QLineEdit("")
        layout_conversion.addWidget(self.entry_sinc_window)

        btn_convert = QPushButton("Wykonaj konwersję")
        btn_convert.clicked.connect(self.perform_conversion)
        layout_conversion.addWidget(btn_convert)

        self.middle_layout.addWidget(gb_conversion)

        gb_errors = QGroupBox("Miary błędów (po konwersji)")
        layout_errors = QVBoxLayout(gb_errors)
        self.lbl_errors = QLabel("MSE: -\nSNR: -\nPSNR: -\nMD: -\nENOB: -")
        layout_errors.addWidget(self.lbl_errors)
        self.middle_layout.addWidget(gb_errors)

        gb_task3 = QGroupBox("Filtracja FIR")
        layout_task3 = QVBoxLayout(gb_task3)

        layout_task3.addWidget(QLabel("Typ filtru:"))
        self.cb_fir_type = QComboBox()
        self.cb_fir_type.addItems(
            ["Dolnoprzepustowy", "Pasmowoprzepustowy (F1)"])
        layout_task3.addWidget(self.cb_fir_type)

        layout_task3.addWidget(QLabel("Typ okna:"))
        self.cb_fir_window = QComboBox()
        self.cb_fir_window.addItems(
            ["Hanning", "Prostokątne"])
        self.cb_fir_window.setCurrentText("Hanning")
        layout_task3.addWidget(self.cb_fir_window)

        layout_task3.addWidget(
            QLabel("Liczba współczynników filtru (nieparzysta):"))
        self.entry_fir_taps = QLineEdit("51")
        layout_task3.addWidget(self.entry_fir_taps)

        self.lbl_fir_cutoff = QLabel("Częstotliwość odcięcia LP [Hz]:")
        layout_task3.addWidget(self.lbl_fir_cutoff)
        self.entry_fir_cutoff = QLineEdit("15.0")
        layout_task3.addWidget(self.entry_fir_cutoff)

        self.lbl_fir_low = QLabel("Pasmo F1: dolna częstotliwość [Hz]:")
        layout_task3.addWidget(self.lbl_fir_low)
        self.entry_fir_low = QLineEdit("8.0")
        layout_task3.addWidget(self.entry_fir_low)

        self.lbl_fir_high = QLabel("Pasmo F1: górna częstotliwość [Hz]:")
        layout_task3.addWidget(self.lbl_fir_high)
        self.entry_fir_high = QLineEdit("18.0")
        layout_task3.addWidget(self.entry_fir_high)

        # Reagowanie na zmianę wybranego filtru
        self.cb_fir_type.currentIndexChanged.connect(self.toggle_fir_inputs)

        btn_apply_filter = QPushButton("Zastosuj filtr FIR")
        btn_apply_filter.clicked.connect(self.apply_fir_filter)
        layout_task3.addWidget(btn_apply_filter)

        self.middle_layout.addWidget(gb_task3)

        self.toggle_fir_inputs()

    def toggle_fir_inputs(self):
        """Pokazuje/ukrywa parametry filtru w zależności od jego typu."""
        filter_type = self.cb_fir_type.currentText()

        if filter_type == "Dolnoprzepustowy":
            # Pokaż parametry dolnoprzepustowego
            self.lbl_fir_cutoff.show()
            self.entry_fir_cutoff.show()
            # Ukryj parametry pasmowoprzepustowego
            self.lbl_fir_low.hide()
            self.entry_fir_low.hide()
            self.lbl_fir_high.hide()
            self.entry_fir_high.hide()

        elif filter_type == "Pasmowoprzepustowy (F1)":
            # Ukryj parametry dolnoprzepustowego
            self.lbl_fir_cutoff.hide()
            self.entry_fir_cutoff.hide()
            # Pokaż parametry pasmowoprzepustowego
            self.lbl_fir_low.show()
            self.entry_fir_low.show()
            self.lbl_fir_high.show()
            self.entry_fir_high.show()

    def update_param_fields(self):
        """Dynamicznie buduje pola wprowadzania danych w zależności od wybranego sygnału."""
        # Czyszczenie poprzedniego formularza
        while self.params_layout.count():
            item = self.params_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.entries.clear()
        signal_type = self.cb_signal_type.currentText()

        # Wspólne parametry
        params_to_create = [
            ("Amplituda (A)", "amplitude", "1.0"),
            ("Czas początkowy (t1) [s]", "start_time", "0.0"),
            ("Czas trwania (d) [s]", "duration", "1.0"),
            ("Częstotliwość próbkowania (f) [Hz]", "sampling_freq", "100.0")
        ]

        sinusoidalne = ["Sygnał Sinusoidalny", "Sygnał Sinusoidalny wyprostowany jednopołówkowo",
                        "Sygnał Sinusoidalny wyprostowany dwupołówkowo"]
        okresowe = ["Sygnał Prostokątny",
                    "Sygnał Prostokątny symetryczny", "Sygnał Trójkątny"]

        if signal_type in sinusoidalne:
            params_to_create.extend([
                ("Częstotliwość sygnału [Hz]", "signal_freq", "2.0"),
                ("Przesunięcie fazowe [rad]", "phase", "0.0")
            ])
        elif signal_type in okresowe:
            params_to_create.extend([
                ("Okres (T) [s]", "T", "0.5"),
                ("Współczynnik wypełnienia (kw)", "kw", "0.5")
            ])
        elif signal_type == "Skok jednostkowy":
            params_to_create.append(("Czas skoku (ts) [s]", "ts", "0.5"))
        elif signal_type == "Impuls jednostkowy":
            params_to_create.append(("Numer próbki skoku (ns)", "ns", "50"))
        elif signal_type == "Szum Impulsowy":
            params_to_create.append(("Prawdopodobieństwo (p)", "p", "0.1"))

        # Tworzenie widgetów
        for label_text, key, default_val in params_to_create:
            entry = QLineEdit(default_val)
            self.params_layout.addRow(label_text, entry)
            self.entries[key] = entry

    def _build_plots(self):
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.ax_signal = self.figure.add_subplot(211)
        self.ax_hist = self.figure.add_subplot(212)
        self.figure.tight_layout(pad=3.0)

        self.canvas = FigureCanvas(self.figure)
        self.right_layout.addWidget(self.canvas)

    def _reset_analysis_state(self):
        self.reconstructed_signal = None
        self.plot_mode = "standard"
        self.lbl_errors.setText("MSE: -\nSNR: -\nPSNR: -\nMD: -\nENOB: -")

    def generate_and_plot(self):
        """Metoda wywoływana po kliknięciu 'Generuj'. Łączy GUI z logiką generatorów."""
        try:
            # Pobieranie parametrów (.text() w Qt zamiast .get() w Tk)
            A = float(self.entries["amplitude"].text())
            t1 = float(self.entries["start_time"].text())
            d = float(self.entries["duration"].text())
            f = float(self.entries["sampling_freq"].text())

            signal_type = self.cb_signal_type.currentText()
            generator = None

            if signal_type == "Szum Gaussowski":
                generator = GaussianNoiseGenerator(A, t1, d, f)

            elif signal_type == "Szum jednostajny":
                generator = UniformNoiseGenerator(A, t1, d, f)

            elif signal_type == "Szum Impulsowy":
                p = float(self.entries["p"].text())
                generator = ImpulseNoiseGenerator(A, t1, d, f, p)

            elif signal_type == "Sygnał Sinusoidalny":
                sig_f = float(self.entries["signal_freq"].text())
                phase = float(self.entries["phase"].text())
                generator = SinusoidalGenerator(A, t1, d, f, sig_f, phase)

            elif signal_type == "Sygnał Sinusoidalny wyprostowany jednopołówkowo":
                sig_f = float(self.entries["signal_freq"].text())
                phase = float(self.entries["phase"].text())
                generator = HalfWaveSineGenerator(A, t1, d, f, sig_f, phase)

            elif signal_type == "Sygnał Sinusoidalny wyprostowany dwupołówkowo":
                sig_f = float(self.entries["signal_freq"].text())
                phase = float(self.entries["phase"].text())
                generator = FullWaveSineGenerator(A, t1, d, f, sig_f, phase)

            elif signal_type == "Sygnał Prostokątny":
                T = float(self.entries["T"].text())
                kw = float(self.entries["kw"].text())
                generator = RectangularGenerator(A, t1, d, f, T, kw)

            elif signal_type == "Sygnał Prostokątny symetryczny":
                T = float(self.entries["T"].text())
                kw = float(self.entries["kw"].text())
                generator = SymmetricRectangularGenerator(A, t1, d, f, T, kw)

            elif signal_type == "Sygnał Trójkątny":
                T = float(self.entries["T"].text())
                kw = float(self.entries["kw"].text())
                generator = TriangularGenerator(A, t1, d, f, T, kw)

            elif signal_type == "Skok jednostkowy":
                ts = float(self.entries["ts"].text())
                generator = UnitStepGenerator(A, t1, d, f, ts)

            elif signal_type == "Impuls jednostkowy":
                ns = int(self.entries["ns"].text())
                generator = UnitImpulseGenerator(A, t1, d, f, ns)

            else:
                QMessageBox.critical(
                    self, "Błąd", "Wybrano nieznany typ sygnału.")
                return

            # Generowanie sygnału
            y = generator.generate()

            if hasattr(generator, 'get_sample_axis'):
                t = generator.get_sample_axis()
            else:
                t = generator.get_time_axis()

            self.current_signal = Signal(
                start_time=t1, sampling_freq=f, amplitudes=y, is_complex=False)
            self.current_signal.time_axis = t
            self.current_signal.is_discrete = signal_type in [
                "Szum Impulsowy", "Impuls jednostkowy"]

            T_val = None
            if signal_type in ["Sygnał Sinusoidalny", "Sygnał Sinusoidalny wyprostowany jednopołówkowo", "Sygnał Sinusoidalny wyprostowany dwupołówkowo"]:
                sig_f = float(self.entries["signal_freq"].text())
                if sig_f > 0:
                    T_val = 1.0 / sig_f
            elif signal_type in ["Sygnał Prostokątny", "Sygnał Prostokątny symetryczny", "Sygnał Trójkątny"]:
                T_val = float(self.entries["T"].text())

            self.current_signal.T = T_val
            self.current_signal_time = t
            self.current_signal_values = y

            self._reset_analysis_state()

            self.calculate_parameters()
            self.update_plots()

            # dodanie do historii
            hist_name = f"[{len(self.signal_history)+1}] Gen: {signal_type}"
            self.add_to_history(self.current_signal, hist_name)

        except ValueError:
            QMessageBox.critical(
                self, "Błąd danych", "Upewnij się, że wszystkie wpisane parametry są poprawnymi liczbami (używaj kropek, nie przecinków).")
        except KeyError as e:
            QMessageBox.critical(
                self, "Błąd GUI", f"Brak pola wprowadzania dla parametru: {e}")

    def calculate_parameters(self):
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            return

        T = getattr(self.current_signal, 'T', None)
        mean_val = Calculator.average(self.current_signal, T)
        abs_mean = Calculator.abs_average(self.current_signal, T)
        rms = Calculator.rms(self.current_signal, T)
        variance = Calculator.variance(self.current_signal, T)
        power = Calculator.average_power(self.current_signal, T)

        stats_text = (f"Średnia: {mean_val:.4f}\n"
                      f"Średnia bezwzględna: {abs_mean:.4f}\n"
                      f"RMS: {rms:.4f}\n"
                      f"Wariancja: {variance:.4f}\n"
                      f"Moc średnia: {power:.4f}")
        self.lbl_stats.setText(stats_text)

    def update_plots(self):
        if len(self.current_signal_values) == 0:
            return

        self.ax_signal.clear()
        is_discrete = getattr(self.current_signal, 'is_discrete', False)
        signal_type = self.cb_signal_type.currentText()
        plot_title = getattr(
            self.current_signal, 'plot_title', "Wykres amplitudy od czasu")
        plot_xlabel = getattr(self.current_signal, 'plot_xlabel', "Czas [s]")
        plot_ylabel = getattr(self.current_signal, 'plot_ylabel', "Amplituda")

        if is_discrete:
            if signal_type == "Impuls jednostkowy" or signal_type == "Szum Impulsowy":
                self.ax_signal.plot(
                    self.current_signal_time,
                    self.current_signal_values,
                    'bo'
                )
            else:
                self.ax_signal.stem(
                    self.current_signal_time,
                    self.current_signal_values,
                    basefmt="black",
                    linefmt="blue",
                    markerfmt="bo"
                )
        else:
            self.ax_signal.plot(
                self.current_signal_time,
                self.current_signal_values,
                color='blue',
                label='Sygnał oryginalny'
            )

        # zrekonstruowany
        if hasattr(self, 'reconstructed_signal') and self.reconstructed_signal is not None:
            self.ax_signal.plot(
                self.reconstructed_signal.time_axis,
                self.reconstructed_signal.amplitudes,
                color='red',
                label='Sygnał zrekonstruowany',
                alpha=0.8
            )
            self.ax_signal.legend()

        self.ax_signal.set_title(plot_title)
        self.ax_signal.set_xlabel(plot_xlabel)
        self.ax_signal.set_ylabel(plot_ylabel)
        self.ax_signal.grid(True)

        # co rysować na dolnym wykresie
        if hasattr(self, 'reconstructed_signal') and self.reconstructed_signal is not None:
            self.update_error_plot()
        elif self.plot_mode == "correlation":
            self.update_correlation_plot()
        else:
            self.update_histogram()

    def update_histogram(self):
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            return

        bins = self.scale_hist.value()
        self.ax_hist.clear()

        T = getattr(self.current_signal, 'T', None)
        hist_data = Calculator.get_full_periods_data(self.current_signal, T)

        counts, bin_edges, patches = self.ax_hist.hist(
            hist_data, bins=bins, color='green', edgecolor='black'
        )

        # self.ax_hist.hist(hist_data, bins=bins, color='green', edgecolor='black', alpha=0.7)
        self.ax_hist.set_title(f"Histogram ({bins} przedziałów)")
        self.ax_hist.set_xlabel("Wartość")
        self.ax_hist.set_ylabel("Liczba wystąpień")
        self.ax_hist.grid(axis='y')

        self.ax_hist.set_xticks(bin_edges)
        self.ax_hist.set_xticklabels([f'{val:.2f}' for val in bin_edges])

        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

    def update_error_plot(self):
        """Rysuje sygnał błędu różnicowego (oryginał - rekonstrukcja) zamiast histogramu."""
        if not hasattr(self, 'current_signal') or not hasattr(self, 'reconstructed_signal'):
            return

        self.ax_hist.clear()  # dolny wykres

        error_values = self.current_signal.amplitudes - \
            self.reconstructed_signal.amplitudes

        self.ax_hist.plot(self.current_signal.time_axis,
                          error_values, color='purple')
        self.ax_hist.set_title(
            "Sygnał błędu (różnica między oryginałem a rekonstrukcją)")
        self.ax_hist.set_xlabel("Czas [s]")
        self.ax_hist.set_ylabel("Błąd amplitudy")
        self.ax_hist.grid(True)

        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

    def update_correlation_plot(self):
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            return

        self.ax_hist.clear()
        self.ax_hist.plot(self.current_signal.time_axis,
                          self.current_signal.amplitudes, color='orange')
        self.ax_hist.set_title("Wykres korelacji")
        self.ax_hist.set_xlabel("Przesunięcie [s]")
        self.ax_hist.set_ylabel("Wartość korelacji")
        self.ax_hist.grid(True)

        self.figure.tight_layout(pad=3.0)
        self.canvas.draw()

    def add_to_history(self, signal, name):
        # zapisujemy nazwe sygnału
        signal.name = name
        self.signal_history.append(signal)

        # dodanie do comboboxów
        self.cb_sig1.addItem(name)
        self.cb_sig2.addItem(name)

        self.cb_sig1.setCurrentIndex(self.cb_sig1.count() - 1)

    def save_to_bin(self):
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            QMessageBox.warning(self, "Brak danych",
                                "Brak wygenerowanego sygnału do zapisu.")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Zapisz do pliku", "", "Pliki binarne (*.bin)")
        if filepath:
            try:
                FileHandler.save_to_binary(filepath, self.current_signal)
                QMessageBox.information(
                    self, "Sukces", "Sygnał zapisany poprawnie.")
            except Exception as e:
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się zapisać pliku:\n{e}")

    def load_from_bin(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Wczytaj z pliku", "", "Pliki binarne (*.bin)")
        if filepath:
            try:
                self.current_signal = FileHandler.load_from_binary(filepath)
                self.current_signal_time = self.current_signal.time_axis
                self.current_signal_values = self.current_signal.amplitudes
                self._reset_analysis_state()

                self.calculate_parameters()
                self.update_plots()

                # bierzemy nazwę pliku ze scieżki
                filename = filepath.split('/')[-1]
                self.add_to_history(
                    self.current_signal, f"[{len(self.signal_history)+1}] Z pliku: {filename}")

            except Exception as e:
                QMessageBox.critical(
                    self, "Błąd", f"Nie udało się wczytać pliku:\n{e}")

    def show_text_data(self):
        if len(self.current_signal_values) == 0:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Dane tekstowe")
        dialog.resize(400, 500)

        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit()
        text_edit.setReadOnly(True)
        text_edit.setLineWrapMode(QTextEdit.NoWrap)
        layout.addWidget(text_edit)

        display_limit = min(100, len(self.current_signal_values))
        lines = [f"t={t:.4f}s \t val={v:.4f}" for t, v in zip(
            self.current_signal_time[:display_limit], self.current_signal_values[:display_limit])]

        content = "\n".join(lines)
        if len(self.current_signal_values) > 100:
            content += "\n... (wyświetlono tylko pierwsze 100 próbek)"

        text_edit.setPlainText(content)
        dialog.exec()

    def _get_signal_for_operation(self, index, cb_widget, title):
        if index == 0:  # index 0, oznacza "wczytaj z pliku"
            filepath, _ = QFileDialog.getOpenFileName(
                self, title, "", "Pliki binarne (*.bin)")
            if filepath:
                signal = FileHandler.load_from_binary(filepath)
                # nazwa sygnalu jako nazwa pliku
                name = filepath.split('/')[-1]
                return signal, name
            return None, None
        else:
            # element na liście o indexie > 0 to index-1 w self.signal_history
            print(title)
            signal = self.signal_history[index - 1]
            name = cb_widget.currentText()
            print('udało sięwczytać sygnał:', title)
            return signal, name

    def perform_operation(self):
        try:
            # co użytkownik wybrał z listy
            idx1 = self.cb_sig1.currentIndex()
            s1, name1 = self._get_signal_for_operation(
                idx1, self.cb_sig1, "Wybierz pierwszy sygnał (S1)")
            if s1 is None:
                return

            idx2 = self.cb_sig2.currentIndex()
            s2, name2 = self._get_signal_for_operation(
                idx2, self.cb_sig2, "Wybierz drugi sygnał (S2)")
            if s2 is None:
                return

            operation = self.cb_operation.currentText()

            if operation == "Dodawanie":
                result_signal = s1 + s2
            elif operation == "Odejmowanie":
                result_signal = s1 - s2
            elif operation == "Mnożenie":
                result_signal = s1 * s2
            elif operation == "Dzielenie":
                result_signal = s1 / s2
            elif operation == "Splot":
                result_signal = Filter.convolve_signals(s1, s2)
            elif operation == "Korelacja (bezpośrednia)":
                result_signal = Filter.cross_correlation_direct(s1, s2)
            elif operation == "Korelacja (przez splot)":
                result_signal = Filter.cross_correlation_via_convolution(
                    s1, s2)
            else:
                return

            self._reset_analysis_state()
            self.current_signal = result_signal
            self.current_signal_time = self.current_signal.time_axis
            self.current_signal_values = self.current_signal.amplitudes
            if operation in ["Korelacja (bezpośrednia)", "Korelacja (przez splot)"]:
                self.plot_mode = "correlation"

            self.calculate_parameters()
            self.update_plots()

            op_symbol_map = {"Dodawanie": "+", "Odejmowanie": "-",
                             "Mnożenie": "*", "Dzielenie": "/"}
            if operation in op_symbol_map:
                res_name = f"({name1}) {op_symbol_map[operation]} ({name2})"
            elif operation == "Splot":
                res_name = f"Splot({name1}, {name2})"
            else:
                res_name = f"Korelacja({name1}, {name2})"
            self.add_to_history(result_signal, res_name)

            QMessageBox.information(
                self, "Sukces", "Operacja wykonana pomyślnie. Sygnał dodany do historii. Wskaż miejsce zapisu sygnału wynikowego.")
            self.save_to_bin()

        except ValueError as e:
            QMessageBox.critical(
                self, "Błąd kompatybilności sygnałów", f"Nie można wykonać operacji:\n{e}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd niespodziewany",
                                 f"Wystąpił problem:\n{e}")

    def apply_fir_filter(self):
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            QMessageBox.warning(
                self, "Błąd", "Brak sygnału do filtracji. Wygeneruj lub wczytaj sygnał.")
            return

        try:
            filter_type = self.cb_fir_type.currentText()
            window_type = self.cb_fir_window.currentText()
            taps = int(self.entry_fir_taps.text())

            if filter_type == "Dolnoprzepustowy":
                cutoff = float(self.entry_fir_cutoff.text())
                fir = Filter.design_lowpass_fir(
                    sampling_freq=self.current_signal.f,
                    cutoff_freq=cutoff,
                    num_taps=taps,
                    window_type=window_type
                )
            else:
                low = float(self.entry_fir_low.text())
                high = float(self.entry_fir_high.text())
                fir = Filter.design_bandpass_fir(
                    sampling_freq=self.current_signal.f,
                    low_cutoff=low,
                    high_cutoff=high,
                    num_taps=taps,
                    window_type=window_type
                )

            filtered_signal = Filter.filter_signal(self.current_signal, fir)
            self._reset_analysis_state()
            self.current_signal = filtered_signal
            self.current_signal_time = filtered_signal.time_axis
            self.current_signal_values = filtered_signal.amplitudes

            self.calculate_parameters()
            self.update_plots()

            filter_label = "LP" if filter_type == "Dolnoprzepustowy" else "BP F1"
            hist_name = f"[{len(self.signal_history)+1}] Filtr FIR {filter_label} ({window_type})"
            self.add_to_history(filtered_signal, hist_name)
        except ValueError as e:
            QMessageBox.critical(
                self, "Błąd parametrów filtru", f"Niepoprawne parametry filtracji:\n{e}")
        except Exception as e:
            QMessageBox.critical(
                self, "Błąd filtracji", f"Wystąpił błąd podczas filtracji:\n{e}")

    def perform_conversion(self):
        """Przeprowadza proces S1 -> Q1 -> R1/R3 i liczy miary błędów."""
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            QMessageBox.warning(
                self, "Błąd", "Brak oryginalnego sygnału do konwersji! Wygeneruj najpierw sygnał.")
            return

        try:
            new_f = float(self.entry_new_f.text())
            bits = int(self.entry_bits.text())
            recon_method = self.cb_reconstruction.currentText()

            sinc_window_str = self.entry_sinc_window.text().strip()
            sinc_window = int(sinc_window_str) if sinc_window_str else None

            original_signal = self.current_signal

            # Próbkowanie S1
            sampled_signal = Converters.sample_signal(original_signal, new_f)

            # Kwantyzacja Q1
            quantized_signal = Converters.quantize_truncation(
                sampled_signal, bits)

            # Rekonstrukcja R1 lub R3
            if recon_method == "Ekstrapolacja zerowego rzędu (ZOH)":
                reconstructed_signal = Converters.reconstruct_zoh(
                    quantized_signal, original_signal.f)
            else:
                reconstructed_signal = Converters.reconstruct_sinc(
                    quantized_signal, original_signal.f, sinc_window)

            self.reconstructed_signal = reconstructed_signal
            self.plot_mode = "standard"

            # miary błędów (C1-C4)
            mse_val = Calculator.mse(original_signal, reconstructed_signal)
            snr_val = Calculator.snr(original_signal, reconstructed_signal)
            psnr_val = Calculator.psnr(original_signal, reconstructed_signal)
            md_val = Calculator.md(original_signal, reconstructed_signal)
            enob_val = Calculator.enob(snr_val)

            errors_text = (f"MSE: {mse_val:.4f}\n"
                           f"SNR: {snr_val:.4f} dB\n"
                           f"PSNR: {psnr_val:.4f} dB\n"
                           f"MD: {md_val:.4f}\n"
                           f"ENOB: {enob_val:.4f} bitów")
            self.lbl_errors.setText(errors_text)

            self.update_plots()

            hist_name = f"[{len(self.signal_history)+1}] Zrekonstr. ({bits}bit, {new_f}Hz)"
            self.add_to_history(reconstructed_signal, hist_name)

        except ValueError as e:
            QMessageBox.critical(
                self, "Błąd wprowadzania", f"Sprawdź poprawność liczb w formularzu.\n\nSzczegóły: {e}")
        except Exception as e:
            QMessageBox.critical(self, "Błąd przetwarzania",
                                 f"Wystąpił błąd:\n{e}")


    def _send_to_transform_tab(self):
        """Wysyła bieżący sygnał do zakładki Transformacje (Zadanie 4)."""
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            QMessageBox.warning(
                self, "Brak sygnału",
                "Brak wygenerowanego sygnału. Wygeneruj lub wczytaj sygnał.")
            return

        label = getattr(self.current_signal, 'name', 'Bieżący sygnał')
        self._transform_tab.load_external_signal(
            time_axis=self.current_signal.time_axis,
            amplitudes=self.current_signal.amplitudes,
            f_pr=self.current_signal.f,
            label=label
        )
        # Przełącz na zakładkę Transformacje
        self._tabs.setCurrentWidget(self._transform_tab)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalApp()
    window.show()
    sys.exit(app.exec())
