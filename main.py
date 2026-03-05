from logging import FileHandler
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.Calculator import Calculator
from core.Signal import Signal
from generators.NoiseGenerators import GaussianNoiseGenerator
from generators.SignalGenerators import FullWaveSineGenerator, HalfWaveSineGenerator, ImpulseNoiseGenerator, RectangularGenerator, SinusoidalGenerator, SymmetricRectangularGenerator, TriangularGenerator, UnitImpulseGenerator, UnitStepGenerator


class SignalApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Generator i Analizator Sygnałów")
        self.geometry("1200x800")

        # Inicjalizacja zmiennych
        self.current_signal_time = np.array([])
        self.current_signal_values = np.array([])
        self.hist_bins = tk.IntVar(value=10)  # Domyślna liczba przedziałów

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
        # 1. Wybór sygnału i parametry
        lf_signal = ttk.LabelFrame(
            self.left_panel, text="Konfiguracja Sygnału", padding=10)
        lf_signal.pack(fill=tk.X, pady=5)

        ttk.Label(lf_signal, text="Rodzaj sygnału:").pack(anchor=tk.W)

        # Pełna lista zaimplementowanych przez nas sygnałów
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

        # Podpinamy zdarzenie zmiany wyboru na liście do funkcji odświeżającej pola
        self.cb_signal_type.bind(
            "<<ComboboxSelected>>", self.update_param_fields)

        # Ramka na dynamiczne pola parametrów (Entry)
        self.params_frame = ttk.Frame(lf_signal)
        self.params_frame.pack(fill=tk.X, pady=5)

        # Słownik do przechowywania referencji do pól tekstowych
        self.entries = {}

        # Inicjalne wygenerowanie pól dla domyślnego sygnału
        self.update_param_fields()

        ttk.Button(lf_signal, text="Generuj",
                   command=self.generate_and_plot).pack(fill=tk.X, pady=10)

        # 2. Parametry obliczone (Wartość średnia, RMS, itd.)
        lf_stats = ttk.LabelFrame(
            self.left_panel, text="Parametry Sygnału", padding=10)
        lf_stats.pack(fill=tk.X, pady=5)
        self.lbl_stats = ttk.Label(
            lf_stats, text="Średnia: -\nŚrednia bezwzględna: -\nRMS: -\nWariancja: -\nMoc: -", justify=tk.LEFT)
        self.lbl_stats.pack(anchor=tk.W)

        # 3. Ustawienia Histogramu
        lf_hist = ttk.LabelFrame(
            self.left_panel, text="Ustawienia Histogramu", padding=10)
        lf_hist.pack(fill=tk.X, pady=5)
        ttk.Label(lf_hist, text="Liczba przedziałów:").pack(anchor=tk.W)
        scale_hist = tk.Scale(lf_hist, variable=self.hist_bins, from_=5,
                              to=20, orient=tk.HORIZONTAL, command=self.update_histogram)
        scale_hist.pack(fill=tk.X)

        # 4. Operacje Plikowe
        lf_files = ttk.LabelFrame(
            self.left_panel, text="Operacje Plikowe", padding=10)
        lf_files.pack(fill=tk.X, pady=5)
        ttk.Button(lf_files, text="Zapisz do pliku (BIN)",
                   command=self.save_to_bin).pack(fill=tk.X, pady=2)
        ttk.Button(lf_files, text="Wczytaj z pliku (BIN)",
                   command=self.load_from_bin).pack(fill=tk.X, pady=2)
        ttk.Button(lf_files, text="Pokaż dane tekstowo",
                   command=self.show_text_data).pack(fill=tk.X, pady=2)

    def update_param_fields(self, event=None):
        """Dynamicznie buduje pola wprowadzania danych w zależności od wybranego sygnału."""
        # Czyścimy starą ramkę z polami
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        self.entries.clear()
        signal_type = self.cb_signal_type.get()

        # Parametry wspólne dla absolutnie każdego sygnału
        params_to_create = [
            ("Amplituda (A)", "amplitude", "1.0"),
            ("Czas początkowy (t1) [s]", "start_time", "0.0"),
            ("Czas trwania (d) [s]", "duration", "1.0"),
            ("Częstotliwość próbkowania (f) [Hz]", "sampling_freq", "100.0")
        ]

        # Parametry specyficzne dodawane warunkowo
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

        # Renderowanie układu (Label + Entry)
        for i, (label_text, key, default_val) in enumerate(params_to_create):
            ttk.Label(self.params_frame, text=label_text).grid(
                row=i, column=0, sticky=tk.W, pady=2, padx=2)

            entry = ttk.Entry(self.params_frame, width=10)
            entry.insert(0, default_val)
            entry.grid(row=i, column=1, sticky=tk.E, pady=2, padx=2)

            # Zapisujemy referencję do słownika, by potem łatwo pobrać wpisaną wartość
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
            # 1. Pobranie parametrów bazowych, wspólnych dla każdego sygnału
            # self.entries to słownik, w którym kluczami są nazwy parametrów zdefiniowane w update_param_fields
            A = float(self.entries["amplitude"].get())
            t1 = float(self.entries["start_time"].get())
            d = float(self.entries["duration"].get())
            f = float(self.entries["sampling_freq"].get())

            signal_type = self.cb_signal_type.get()
            generator = None

            # 2. Rozpoznanie wybranego sygnału, pobranie specyficznych parametrów i utworzenie generatora
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
                ns = int(self.entries["ns"].get())  # ns to liczba całkowita!
                generator = UnitImpulseGenerator(A, t1, d, f, ns)

            else:
                messagebox.showerror("Błąd", "Wybrano nieznany typ sygnału.")
                return

            # 3. Wygenerowanie sygnału z instancji odpowiedniej klasy
            y = generator.generate()

            # Pobranie odpowiedniej osi (czasowej lub próbek, zależnie od typu generatora)
            if hasattr(generator, 'get_sample_axis'):
                t = generator.get_sample_axis()
            else:
                t = generator.get_time_axis()

            # 4. Opakowanie danych w obiekt Signal
            # Dzięki temu FileHandler będzie mógł zapisać ten sygnał, jeśli użytkownik kliknie "Zapisz do pliku"
            self.current_signal = Signal(
                start_time=t1, sampling_freq=f, amplitudes=y, is_complex=False)
            self.current_signal.time_axis = t

            # Flaga pomocnicza do rysowania odpowiednich wykresów (ciągły vs prążkowy) w update_plots
            self.current_signal.is_discrete = signal_type in [
                "Szum Impulsowy", "Impuls jednostkowy"]

            # Przypisanie okresu T do sygnału, by użyć go w obliczeniach
            T_val = None
            if signal_type in ["Sygnał Sinusoidalny", "Sygnał Sinusoidalny wyprostowany jednopołówkowo", "Sygnał Sinusoidalny wyprostowany dwupołówkowo"]:
                sig_f = float(self.entries["signal_freq"].get())
                if sig_f > 0:
                    T_val = 1.0 / sig_f
            elif signal_type in ["Sygnał Prostokątny", "Sygnał Prostokątny symetryczny", "Sygnał Trójkątny"]:
                T_val = float(self.entries["T"].get())

            self.current_signal.T = T_val
            # 5. Aktualizacja starych zmiennych dla reszty GUI
            self.current_signal_time = t
            self.current_signal_values = y

            # 6. Przeliczenie statystyk i odrysowanie wykresów
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

        # Pobieramy T, jeśli nie istnieje - zwracamy None (Calculator sobie z tym poradzi)
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
        # Zabezpieczenie: nie rysujemy, jeśli tablica wartości jest pusta
        if len(self.current_signal_values) == 0:
            return

        # Czyszczenie górnego obszaru roboczego (osi sygnału)
        self.ax_signal.clear()

        # Sprawdzamy, czy sygnał ma ustawioną flagę dyskretności
        # getattr bezpiecznie sprawdza atrybut, domyślnie zwracając False, jeśli go nie ma
        is_discrete = False
        if hasattr(self, 'current_signal') and hasattr(self.current_signal, 'is_discrete'):
            is_discrete = self.current_signal.is_discrete

        # Rysowanie właściwego typu wykresu
        if is_discrete:
            # Dla sygnałów dyskretnych (np. impuls jednostkowy) używamy wykresu prążkowego (stem)
            self.ax_signal.stem(
                self.current_signal_time,
                self.current_signal_values,
                basefmt="black",
                linefmt="blue",
                markerfmt="bo"
            )
        else:
            # Dla sygnałów ciągłych (np. sinusoida, trójkąt) używamy klasycznej linii
            self.ax_signal.plot(
                self.current_signal_time,
                self.current_signal_values,
                color='blue'
            )

        # Konfiguracja wyglądu górnego wykresu
        self.ax_signal.set_title("Wykres amplitudy od czasu")
        self.ax_signal.set_xlabel("Czas [s]")
        self.ax_signal.set_ylabel("Amplituda")
        self.ax_signal.grid(True)

        # Na koniec wywołujemy odświeżenie histogramu,
        # które przy okazji przerysuje całe płótno (self.canvas.draw())
        self.update_histogram()

    def update_histogram(self, event=None):
        """Aktualizuje tylko histogram w oparciu o suwak (na pełnych okresach)."""
        if not hasattr(self, 'current_signal') or self.current_signal is None:
            return

        bins = self.hist_bins.get()
        self.ax_hist.clear()

        # Pobieramy dane obcięte do pełnych okresów dla histogramu
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


if __name__ == "__main__":
    app = SignalApp()
    app.mainloop()
