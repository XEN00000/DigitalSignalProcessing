import sys
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QGroupBox, QLabel, QComboBox,
                               QLineEdit, QPushButton, QSlider, QMessageBox,
                               QFileDialog, QFormLayout, QTextEdit, QDialog)
from PySide6.QtCore import Qt

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from core.FileHandler import FileHandler
from core.Calculator import Calculator
from core.Signal import Signal
from generators.NoiseGenerators import GaussianNoiseGenerator
from generators.SignalGenerators import (FullWaveSineGenerator, HalfWaveSineGenerator,
                                         ImpulseNoiseGenerator, RectangularGenerator,
                                         SinusoidalGenerator, SymmetricRectangularGenerator,
                                         TriangularGenerator, UnitImpulseGenerator,
                                         UnitStepGenerator)


class SignalApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Generator i Analizator Sygnałów")
        self.resize(1200, 800)

        self.current_signal_time = np.array([])
        self.current_signal_values = np.array([])

        self.signal_history = []  # historia sygnałów

        # Centralny widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.main_layout = QHBoxLayout(central_widget)

        self.create_widgets()

    def create_widgets(self):
        # Panel lewy (sterowanie)
        self.left_panel = QWidget()
        self.left_panel.setFixedWidth(450)
        self.left_layout = QVBoxLayout(self.left_panel)
        self.left_layout.setAlignment(Qt.AlignTop)

        # Panel prawy (wykresy)
        self.right_panel = QWidget()
        self.right_layout = QVBoxLayout(self.right_panel)

        self.main_layout.addWidget(self.left_panel)
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

        # --- Ustawienia Histogramu ---
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

        # --- Operacje Plikowe ---
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

        # --- Operacje na sygnałach ---
        gb_operations = QGroupBox("Operacje na sygnałach")
        layout_operations = QVBoxLayout(gb_operations)

        self.cb_operation = QComboBox()
        self.cb_operation.addItems(
            ["Dodawanie", "Odejmowanie", "Mnożenie", "Dzielenie"])
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
                color='blue'
            )

        self.ax_signal.set_title("Wykres amplitudy od czasu")
        self.ax_signal.set_xlabel("Czas [s]")
        self.ax_signal.set_ylabel("Amplituda")
        self.ax_signal.grid(True)

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

    def _get_signal_for_operation(self, index, title):
        if index == 0:  # index 0, oznacza "wczytaj z pliku"
            filepath, _ = QFileDialog.getOpenFileName(
                self, title, "", "Pliki binarne (*.bin)")
            if filepath:
                return FileHandler.load_from_binary(filepath)
            return None
        else:
            # element na liście o indexie > 0 to index-1 w self.signal_history
            return self.signal_history[index - 1]

    def perform_operation(self):
        try:
            # co użytkownik wybrał z listy
            idx1 = self.cb_sig1.currentIndex()
            s1 = self._get_signal_for_operation(
                idx1, "Wybierz pierwszy sygnał (S1)")
            if s1 is None:
                return

            idx2 = self.cb_sig2.currentIndex()
            s2 = self._get_signal_for_operation(
                idx2, "Wybierz drugi sygnał (S2)")
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
            else:
                return

            self.current_signal = result_signal
            self.current_signal_time = self.current_signal.time_axis
            self.current_signal_values = self.current_signal.amplitudes

            self.calculate_parameters()
            self.update_plots()

            op_symbol = {"Dodawanie": "+", "Odejmowanie": "-",
                         "Mnożenie": "*", "Dzielenie": "/"}[operation]
            res_name = f"[{len(self.signal_history)+1}] Wynik: S{idx1} {op_symbol} S{idx2}"
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


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SignalApp()
    window.show()
    sys.exit(app.exec())
