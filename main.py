import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.FileHandler import FileHandler
from core.Calculator import Calculator
from core.Signal import Signal
from generators.NoiseGenerators import GaussianNoiseGenerator
from generators.SignalGenerators import FullWaveSineGenerator, HalfWaveSineGenerator, ImpulseNoiseGenerator, RectangularGenerator, SinusoidalGenerator, SymmetricRectangularGenerator, TriangularGenerator, UnitImpulseGenerator, UnitStepGenerator


class SignalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generator i Analizator Sygnałów")
        self.geometry("1200x800")

        self.current_signal_time = np.array([])
        self.current_signal_values = np.array([])
        self.hist_bins = tk.IntVar(value=10)

        self.create_widgets()

    def create_widgets(self):
        # constrols section
        self.left_panel = ttk.Frame(self, width=400, padding=10)
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, expand=False)

        # plots section
        self.right_panel = ttk.Frame(self, padding=10)
        self.right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self._build_controls()
        self._build_plots()

    def _build_controls(self):
        lf_signal = ttk.LabelFrame(
            self.left_panel, text="Konfiguracja Sygnału", padding=10)
        lf_signal.pack(fill=tk.X, pady=5)

        ttk.Label(lf_signal, text="Rodzaj sygnału:").pack(anchor=tk.W)

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

        self.cb_signal_type = ttk.Combobox(
            lf_signal, values=lista_sygnalow, state="readonly")
        self.cb_signal_type.pack(fill=tk.X, pady=5)
        self.cb_signal_type.set("Sygnał Sinusoidalny")

        # selection type binding, refreshing
        self.cb_signal_type.bind(
            "<<ComboboxSelected>>", self.update_param_fields)

        # frame for entries
        self.params_frame = ttk.Frame(lf_signal)
        self.params_frame.pack(fill=tk.X, pady=5)

        # dict for storing references to text fields
        self.entries = {}

        # generate fields for the default signal
        self.update_param_fields()

        ttk.Button(lf_signal, text="Generuj",
                   command=self.generate_and_plot).pack(fill=tk.X, pady=10)

        # params
        lf_stats = ttk.LabelFrame(
            self.left_panel, text="Parametry Sygnału", padding=10)
        lf_stats.pack(fill=tk.X, pady=5)
        self.lbl_stats = ttk.Label(
            lf_stats, text="Średnia: -\nŚrednia bezwzględna: -\nRMS: -\nWariancja: -\nMoc: -", justify=tk.LEFT)
        self.lbl_stats.pack(anchor=tk.W)

        lf_hist = ttk.LabelFrame(
            self.left_panel, text="Ustawienia Histogramu", padding=10)
        lf_hist.pack(fill=tk.X, pady=5)
        ttk.Label(lf_hist, text="Liczba przedziałów:").pack(anchor=tk.W)
        scale_hist = tk.Scale(lf_hist, variable=self.hist_bins, from_=5,
                              to=20, orient=tk.HORIZONTAL, command=self.update_histogram)
        scale_hist.pack(fill=tk.X)

        # files operations
        lf_files = ttk.LabelFrame(
            self.left_panel, text="Operacje Plikowe", padding=10)
        lf_files.pack(fill=tk.X, pady=5)
        ttk.Button(lf_files, text="Zapisz do pliku (BIN)",
                   command=self.save_to_bin).pack(fill=tk.X, pady=2)
        ttk.Button(lf_files, text="Wczytaj z pliku (BIN)",
                   command=self.load_from_bin).pack(fill=tk.X, pady=2)
        ttk.Button(lf_files, text="Pokaż dane tekstowo",
                   command=self.show_text_data).pack(fill=tk.X, pady=2)

        # Operations on signal
        lf_operations = ttk.LabelFrame(
            self.left_panel, text="Operacje na sygnałach", padding=10)
        lf_operations.pack(fill=tk.X, pady=5)

        self.cb_operation = ttk.Combobox(
            lf_operations,
            values=["Dodawanie", "Odejmowanie", "Mnożenie", "Dzielenie"],
            state="readonly"
        )
        self.cb_operation.pack(fill=tk.X, pady=5)
        self.cb_operation.set("Dodawanie")

        ttk.Button(lf_operations, text="Wczytaj 2 pliki i wykonaj",
                   command=self.perform_operation).pack(fill=tk.X, pady=2)

    def update_param_fields(self, event=None):
        """Dynamicznie buduje pola wprowadzania danych w zależności od wybranego sygnału."""
        # clear params frame
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        self.entries.clear()
        signal_type = self.cb_signal_type.get()

        # params common to every signal
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

        # label + entry
        for i, (label_text, key, default_val) in enumerate(params_to_create):
            ttk.Label(self.params_frame, text=label_text).grid(
                row=i, column=0, sticky=tk.W, pady=2, padx=2)

            entry = ttk.Entry(self.params_frame, width=10)
            entry.insert(0, default_val)
            entry.grid(row=i, column=1, sticky=tk.E, pady=2, padx=2)

            # safe reference
            self.entries[key] = entry

    def _build_plots(self):
        self.figure = Figure(figsize=(8, 6), dpi=100)
        # signal plot
        self.ax_signal = self.figure.add_subplot(211)

        # hist
        self.ax_hist = self.figure.add_subplot(212)

        self.figure.tight_layout(pad=3.0)

        # setting on canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def generate_and_plot(self):
        """Metoda wywoływana po kliknięciu 'Generuj'. Łączy GUI z logiką generatorów."""
        try:

            # getting params from entries - names are defined in update_param_fields
            A = float(self.entries["amplitude"].get())
            t1 = float(self.entries["start_time"].get())
            d = float(self.entries["duration"].get())
            f = float(self.entries["sampling_freq"].get())

            signal_type = self.cb_signal_type.get()
            generator = None

            if signal_type == "Szum Gaussowski":
                generator = GaussianNoiseGenerator(A, t1, d, f)

            elif signal_type == "Szum Impulsowy":
                p = float(self.entries["p"].get())
                generator = ImpulseNoiseGenerator(A, t1, d, f, p)

            elif signal_type == "Sygnał Sinusoidalny":
                sig_f = float(self.entries["signal_freq"].get())
                phase = float(self.entries["phase"].get())
                generator = SinusoidalGenerator(A, t1, d, f, sig_f, phase)

            elif signal_type == "Sygnał Sinusoidalny wyprostowany jednopołówkowo":
                sig_f = float(self.entries["signal_freq"].get())
                phase = float(self.entries["phase"].get())
                generator = HalfWaveSineGenerator(A, t1, d, f, sig_f, phase)

            elif signal_type == "Sygnał Sinusoidalny wyprostowany dwupołówkowo":
                sig_f = float(self.entries["signal_freq"].get())
                phase = float(self.entries["phase"].get())
                generator = FullWaveSineGenerator(A, t1, d, f, sig_f, phase)

            elif signal_type == "Sygnał Prostokątny":
                T = float(self.entries["T"].get())
                kw = float(self.entries["kw"].get())
                generator = RectangularGenerator(A, t1, d, f, T, kw)

            elif signal_type == "Sygnał Prostokątny symetryczny":
                T = float(self.entries["T"].get())
                kw = float(self.entries["kw"].get())
                generator = SymmetricRectangularGenerator(A, t1, d, f, T, kw)

            elif signal_type == "Sygnał Trójkątny":
                T = float(self.entries["T"].get())
                kw = float(self.entries["kw"].get())
                generator = TriangularGenerator(A, t1, d, f, T, kw)

            elif signal_type == "Skok jednostkowy":
                ts = float(self.entries["ts"].get())
                generator = UnitStepGenerator(A, t1, d, f, ts)

            elif signal_type == "Impuls jednostkowy":
                ns = int(self.entries["ns"].get())
                generator = UnitImpulseGenerator(A, t1, d, f, ns)

            else:
                messagebox.showerror("Błąd", "Wybrano nieznany typ sygnału.")
                return

            # generating signal
            y = generator.generate()

            # axis depends on signal type/generator type
            if hasattr(generator, 'get_sample_axis'):
                t = generator.get_sample_axis()
            else:
                t = generator.get_time_axis()

            # self.current_signal is needed to save the file
            self.current_signal = Signal(
                start_time=t1, sampling_freq=f, amplitudes=y, is_complex=False)
            self.current_signal.time_axis = t

            # flag needed to draw the plot
            self.current_signal.is_discrete = signal_type in [
                "Szum Impulsowy", "Impuls jednostkowy"]

            # T = 1/f - period
            T_val = None
            if signal_type in ["Sygnał Sinusoidalny", "Sygnał Sinusoidalny wyprostowany jednopołówkowo", "Sygnał Sinusoidalny wyprostowany dwupołówkowo"]:
                sig_f = float(self.entries["signal_freq"].get())
                if sig_f > 0:
                    T_val = 1.0 / sig_f
            elif signal_type in ["Sygnał Prostokątny", "Sygnał Prostokątny symetryczny", "Sygnał Trójkątny"]:
                T_val = float(self.entries["T"].get())

            self.current_signal.T = T_val

            # GUI update
            self.current_signal_time = t
            self.current_signal_values = y

            self.calculate_parameters()
            self.update_plots()

        except ValueError:
            messagebox.showerror(
                "Błąd danych", "Upewnij się, że wszystkie wpisane parametry są poprawnymi liczbami (używaj kropek, nie przecinków).")
        except KeyError as e:
            messagebox.showerror(
                "Błąd GUI", f"Brak pola wprowadzania dla parametru: {e}")

    def calculate_parameters(self):
        """Oblicza średnią, RMS, wariancję itd. korzystając z pełnych okresów."""
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
        self.lbl_stats.config(text=stats_text)

    def update_plots(self):
        """Odświeża wykres sygnału i wywołuje odświeżenie histogramu."""
        if len(self.current_signal_values) == 0:
            return

        self.ax_signal.clear()

        is_discrete = False
        if hasattr(self, 'current_signal') and hasattr(self.current_signal, 'is_discrete'):
            is_discrete = self.current_signal.is_discrete

        if is_discrete:
            # todo: change stem to dots
            self.ax_signal.stem(
                self.current_signal_time,
                self.current_signal_values,
                basefmt="black",
                linefmt="blue",
                markerfmt="bo"
            )
        else:
            # for continuous signals use line
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

    def update_histogram(self, event=None):
        """Aktualizuje tylko histogram w oparciu o suwak (na pełnych okresach)."""
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            return

        bins = self.hist_bins.get()
        self.ax_hist.clear()

        T = getattr(self.current_signal, 'T', None)
        hist_data = Calculator.get_full_periods_data(self.current_signal, T)

        self.ax_hist.hist(hist_data, bins=bins, color='green',
                          edgecolor='black', alpha=0.7)
        self.ax_hist.set_title(f"Histogram ({bins} przedziałów)")
        self.ax_hist.set_xlabel("Wartość")
        self.ax_hist.set_ylabel("Liczba wystąpień")
        self.ax_hist.grid(axis='y')

        self.canvas.draw()

    def save_to_bin(self):
        """Zapisuje sygnał do pliku binarnego za pomocą FileHandler."""
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            messagebox.showwarning(
                "Brak danych", "Brak wygenerowanego sygnału do zapisu.")
            return

        filepath = filedialog.asksaveasfilename(
            defaultextension=".bin", filetypes=[("Pliki binarne", "*.bin")])
        if filepath:
            try:
                FileHandler.save_to_binary(filepath, self.current_signal)
                messagebox.showinfo("Sukces", "Sygnał zapisany poprawnie.")
            except Exception as e:
                messagebox.showerror(
                    "Błąd", f"Nie udało się zapisać pliku: {e}")

    def load_from_bin(self):
        """Wczytuje sygnał z pliku binarnego za pomocą FileHandler."""
        filepath = filedialog.askopenfilename(
            filetypes=[("Pliki binarne", "*.bin")])
        if filepath:
            try:
                self.current_signal = FileHandler.load_from_binary(filepath)

                self.current_signal_time = self.current_signal.time_axis
                self.current_signal_values = self.current_signal.amplitudes

                self.calculate_parameters()
                self.update_plots()
            except Exception as e:
                messagebox.showerror(
                    "Błąd", f"Nie udało się wczytać pliku: {e}")

    def save_to_txt(self):
        """Zapisuje sygnał do pliku tekstowego za pomocą FileHandler."""
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            return

        filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[
                                                ("Pliki tekstowe", "*.txt")])
        if filepath:
            try:
                FileHandler.save_to_text(filepath, self.current_signal)
                messagebox.showinfo(
                    "Sukces", "Eksport do TXT zakończony sukcesem.")
            except Exception as e:
                messagebox.showerror(
                    "Błąd", f"Nie udało się zapisać pliku: {e}")

    def show_text_data(self):
        """Wyświetla próbki w nowym oknie tekstowym."""
        if len(self.current_signal_values) == 0:
            return
        top = tk.Toplevel(self)
        top.title("Dane tekstowe")
        text = tk.Text(top, wrap="none")
        text.pack(expand=True, fill=tk.BOTH)

        display_limit = min(100, len(self.current_signal_values))
        lines = [f"t={t:.4f}s \t val={v:.4f}" for t, v in zip(
            self.current_signal_time[:display_limit], self.current_signal_values[:display_limit])]
        text.insert(tk.END, "\n".join(lines))
        if len(self.current_signal_values) > 100:
            text.insert(
                tk.END, "\n... (wyświetlono tylko pierwsze 100 próbek)")

    def perform_operation(self):
        """Wczytuje dwa sygnały z plików binarnych, wykonuje operację i aktualizuje GUI."""

        # load first file
        filepath1 = filedialog.askopenfilename(
            title="Wybierz pierwszy sygnał (S1)",
            filetypes=[("Pliki binarne", "*.bin")]
        )
        if not filepath1:
            return

        # load second file
        filepath2 = filedialog.askopenfilename(
            title="Wybierz drugi sygnał (S2)",
            filetypes=[("Pliki binarne", "*.bin")]
        )
        if not filepath2:
            return

        try:
            s1 = FileHandler.load_from_binary(filepath1)
            s2 = FileHandler.load_from_binary(filepath2)

            operation = self.cb_operation.get()

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

            # set result to current signal
            self.current_signal = result_signal
            self.current_signal_time = self.current_signal.time_axis
            self.current_signal_values = self.current_signal.amplitudes

            self.calculate_parameters()
            self.update_plots()

            messagebox.showinfo(
                "Sukces",
                "Operacja wykonana pomyślnie. Wskaż miejsce zapisu sygnału wynikowego."
            )
            self.save_to_bin()

        except ValueError as e:
            messagebox.showerror(
                "Błąd kompatybilności sygnałów", f"Nie można wykonać operacji:\n{e}")
        except Exception as e:
            messagebox.showerror("Błąd niespodziewany",
                                 f"Wystąpił problem:\n{e}")


if __name__ == "__main__":
    app = SignalApp()
    app.mainloop()
