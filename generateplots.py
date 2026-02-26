import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import numpy as np
import nidaqmx
import nidaqmx.constants
from nidaqmx.constants import AcquisitionType
from scipy.signal import stft
from scipy.signal.windows import hann
import warnings
import os
import math
import platform
import ctypes as ct
import matplotlib
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

# ─────────────────────────────────────────────
#  DAQ / Signal constants
# ─────────────────────────────────────────────
CHANNEL     = "cDAQ1Mod3/ai0"
FS          = 250_000          # Sampling frequency (Hz)
CHUNK_DUR   = 0.05             # Seconds of data per acquisition chunk
SAMPLES     = int(FS * CHUNK_DUR) #Number of voltage samples
NPERSEG     = 5_000 #Number of samples per segment
HOP_SIZE    = 100 #How many points to "jump" across to the next segment
NFFT        = 2 ** 14 #Length of FFT used. Defaults to Nperseg
NOVERLAP    = NPERSEG - HOP_SIZE #Overlapping data points
WINDOW      = hann(NPERSEG) #Hanning window to minimize spectral leakage for FFT

# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────
def dark_title_bar(window):
    if "Windows" in platform.platform():
        window.update()
        dwm   = ct.windll.dwmapi.DwmSetWindowAttribute
        hwnd  = ct.windll.user32.GetParent(window.winfo_id())
        value = ct.c_int(2)
        dwm(hwnd, 20, ct.byref(value), 4)

def move_figure(fig, x, y):
    fig.canvas.manager.window.wm_geometry(f"+{x}+{y}")

def compute_spectrum(samples_array):
    """Return (frequencies, mean_magnitudes) for a voltage array."""
    f, _, Zxx = stft(samples_array, fs=FS, window=WINDOW,
                     nperseg=NPERSEG, noverlap=NOVERLAP, nfft=NFFT)
    mag      = np.abs(Zxx)
    mean_mag = np.mean(mag, axis=1)
    peak_idx = np.argmax(mean_mag)
    freq     = f[peak_idx]
    return f, mean_mag

# ─────────────────────────────────────────────
#  Mode 1 – RECORD
# ─────────────────────────────────────────────
def run_record_mode(duration_sec: float, save_dir: str):
    """
    Acquire data for `duration_sec` seconds, then save:
      1. voltage_<timestamp>.npy: raw voltage time series
      2. freq_amplitudes_<ts>.npy: mean FFT magnitudes (last chunk)
      3. voltage_plot_<ts>.png
      4. spectrum_plot_<ts>.png
    """
    print(f"[RECORD] Recording {duration_sec}s of data → {save_dir}")
    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    n_chunks      = max(1, int(duration_sec / CHUNK_DUR))
    all_voltages  = []
    last_freqs    = None
    last_mean_mag = None

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(CHANNEL, min_val=0, max_val=10)
        task.timing.cfg_samp_clk_timing(FS, sample_mode=AcquisitionType.FINITE,
                                        samps_per_chan=SAMPLES)
        for i in range(n_chunks):
            task.start()
            raw = task.read(number_of_samples_per_channel=SAMPLES)
            task.stop()
            arr = np.array(raw)
            all_voltages.append(arr)
            last_freqs, last_mean_mag = compute_spectrum(arr)
            print(f"  chunk {i+1}/{n_chunks} acquired", end="\r")

    voltage_ts = np.concatenate(all_voltages)
    time_axis  = np.linspace(0, duration_sec, len(voltage_ts))

    # ── Save arrays ──────────────────────────────────────────────────────────
    v_path = os.path.join(save_dir, f"voltage_{ts}.npy")
    f_path = os.path.join(save_dir, f"freq_amplitudes_{ts}.npy")
    np.save(v_path, np.column_stack([time_axis, voltage_ts]))
    np.save(f_path, np.column_stack([last_freqs, last_mean_mag]))
    print(f"\n[RECORD] Arrays saved:\n  {v_path}\n  {f_path}")

    # ── Voltage plot ─────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig_v, ax_v = plt.subplots(figsize=(10, 4))
    ax_v.plot(time_axis, voltage_ts, color="#00e5ff", linewidth=0.5)
    ax_v.set_facecolor("black")
    ax_v.set_title(f"Recorded Voltage  ({duration_sec}s)", color="white")
    ax_v.set_xlabel("Time (s)")
    ax_v.set_ylabel("Voltage (V)")
    ax_v.grid(True, alpha=0.3)
    plt_v_path = os.path.join(save_dir, f"voltage_plot_{ts}.png")
    fig_v.tight_layout()
    fig_v.savefig(plt_v_path, dpi=150)
    plt.close(fig_v)

    # ── Spectrum plot ─────────────────────────────────────────────────────────
    fig_s, ax_s = plt.subplots(figsize=(10, 4))
    ax_s.plot(last_freqs, last_mean_mag, color="#ff6f00", linewidth=0.8)
    ax_s.set_facecolor("black")
    ax_s.set_title("Frequency Amplitudes (final chunk)", color="white")
    ax_s.set_xlabel("Frequency (Hz)")
    ax_s.set_ylabel("Amplitude")
    ax_s.set_xlim(0, FS / 2)
    ax_s.grid(True, alpha=0.3)
    plt_s_path = os.path.join(save_dir, f"spectrum_plot_{ts}.png")
    fig_s.tight_layout()
    fig_s.savefig(plt_s_path, dpi=150)
    plt.close(fig_s)

    print(f"[RECORD] Plots saved:\n  {plt_v_path}\n  {plt_s_path}")

    # ── Show saved plots ──────────────────────────────────────────────────────
    fig_show, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(11, 7),
                                               num="Recorded Data")
    fig_show.patch.set_facecolor("#0d0d0d")
    dark_title_bar(fig_show.canvas.manager.window)
    fig_show.canvas.toolbar.pack_forget()

    ax_top.plot(time_axis, voltage_ts, color="#00e5ff", linewidth=0.4)
    ax_top.set_facecolor("#111")
    ax_top.set_title(f"Voltage over {duration_sec}s", color="white")
    ax_top.set_xlabel("Time (s)", color="gray")
    ax_top.set_ylabel("Voltage (V)", color="gray")
    ax_top.tick_params(colors="gray")
    ax_top.grid(True, color="#333")

    ax_bot.plot(last_freqs, last_mean_mag, color="#ff6f00", linewidth=0.8)
    ax_bot.set_facecolor("#111")
    ax_bot.set_title("Frequency Amplitudes (final chunk)", color="white")
    ax_bot.set_xlabel("Frequency (Hz)", color="gray")
    ax_bot.set_ylabel("Amplitude", color="gray")
    ax_bot.set_xlim(0, FS / 2)
    ax_bot.tick_params(colors="gray")
    ax_bot.grid(True, color="#333")

    fig_show.tight_layout()
    move_figure(fig_show, 100, 80)
    plt.show()

# ─────────────────────────────────────────────
#  Mode 2 – LIVE
# ─────────────────────────────────────────────
def run_live_mode():
    """
    Two subplots updated in real time:
      top: rolling voltage waveform
      bottom: live FFT magnitude spectrum
    """
    print("[LIVE] Starting live display…")
    plt.style.use("dark_background")
    fig = plt.figure(num="Live Voltage & Spectrum", figsize=(11, 7))
    fig.patch.set_facecolor("#0d0d0d")
    gs  = gridspec.GridSpec(2, 1, hspace=0.45)
    ax_volt = fig.add_subplot(gs[0])
    ax_spec = fig.add_subplot(gs[1])
    move_figure(fig, 100, 80)
    dark_title_bar(fig.canvas.manager.window)
    fig.canvas.toolbar.pack_forget()

    HISTORY = 50          # rolling voltage chunks to show
    volt_buf: list[float] = []
    anim_ref = []

    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(CHANNEL, min_val=0, max_val=10)
        task.timing.cfg_samp_clk_timing(FS, sample_mode=AcquisitionType.FINITE,
                                        samps_per_chan=SAMPLES)

        def update(_frame):
            task.start()
            raw = task.read(number_of_samples_per_channel=SAMPLES)
            task.stop()

            arr = np.array(raw)
            volt_buf.extend(arr.tolist())
            # keep last HISTORY chunks worth of samples
            max_pts = HISTORY * SAMPLES
            if len(volt_buf) > max_pts:
                del volt_buf[: len(volt_buf) - max_pts]

            freqs, mean_mag = compute_spectrum(arr)

            # ── Voltage plot ────────────────────────────────────────────────
            ax_volt.clear()
            t = np.linspace(0, len(volt_buf) / FS, len(volt_buf))
            ax_volt.plot(t, volt_buf, color="#00e5ff", linewidth=0.5)
            ax_volt.set_facecolor("#111")
            ax_volt.set_title("Live Voltage", color="white", pad=6)
            ax_volt.set_xlabel("Time (s)", color="gray", fontsize=9)
            ax_volt.set_ylabel("Voltage (V)", color="gray", fontsize=9)
            ax_volt.tick_params(colors="gray", labelsize=8)
            ax_volt.grid(True, color="#333", linewidth=0.5)

            # ── Spectrum plot ────────────────────────────────────────────────
            ax_spec.clear()
            ax_spec.plot(freqs, mean_mag, color="#ff6f00", linewidth=0.8)
            ax_spec.set_facecolor("#111")
            ax_spec.set_title("Live Frequency Amplitudes", color="white", pad=6)
            ax_spec.set_xlabel("Frequency (Hz)", color="gray", fontsize=9)
            ax_spec.set_ylabel("Amplitude", color="gray", fontsize=9)
            ax_spec.set_xlim(0, FS / 2)
            ax_spec.tick_params(colors="gray", labelsize=8)
            ax_spec.grid(True, color="#333", linewidth=0.5)

        anim_obj = animation.FuncAnimation(fig, update, interval=50, cache_frame_data=False)
        anim_ref.append(anim_obj)   # keep reference alive
        plt.show()

# ─────────────────────────────────────────────
#  Mode Selector GUI
# ─────────────────────────────────────────────
def launch_selector():
    root = tk.Tk()
    root.title("Voltage Analyzer: Mode Select")
    root.configure(bg="#1a1a1a")
    root.resizable(False, False)
    root.geometry("380x280")

    # ── Title ─────────────────────────────────────────────────────────────────
    tk.Label(root, text="Voltage Analyzer", font=("Segoe UI", 16, "bold"),
             bg="#1a1a1a", fg="#00e5ff").pack(pady=(22, 4))
    tk.Label(root, text="Select an operating mode", font=("Segoe UI", 9),
             bg="#1a1a1a", fg="#888").pack(pady=(0, 16))

    # ── Mode variable ─────────────────────────────────────────────────────────
    mode_var = tk.StringVar(value="live")

    frame_modes = tk.Frame(root, bg="#1a1a1a")
    frame_modes.pack()
    for text, val in [("Live Display", "live"), ("💾  Record & Save", "record")]:
        tk.Radiobutton(frame_modes, text=text, variable=mode_var, value=val,
                       font=("Segoe UI", 10), bg="#1a1a1a", fg="white",
                       selectcolor="#003344", activebackground="#1a1a1a",
                       activeforeground="#00e5ff").pack(anchor="w", pady=3)

    # ── Record options ─────────────────────────────────────────────────────────
    rec_frame = tk.LabelFrame(root, text=" Record Options ", bg="#1a1a1a",
                              fg="#888", font=("Segoe UI", 8), bd=1,
                              relief="groove")
    rec_frame.pack(fill="x", padx=30, pady=(12, 0))

    tk.Label(rec_frame, text="Duration (s):", bg="#1a1a1a", fg="white",
             font=("Segoe UI", 9)).grid(row=0, column=0, padx=8, pady=6, sticky="w")
    dur_var = tk.StringVar(value="5")
    tk.Entry(rec_frame, textvariable=dur_var, width=7, bg="#111", fg="#00e5ff",
             insertbackground="white", relief="flat").grid(row=0, column=1, padx=4)

    tk.Label(rec_frame, text="Save folder:", bg="#1a1a1a", fg="white",
             font=("Segoe UI", 9)).grid(row=1, column=0, padx=8, pady=6, sticky="w")
    dir_var = tk.StringVar(value="./recordings")
    tk.Entry(rec_frame, textvariable=dir_var, width=18, bg="#111", fg="#00e5ff",
             insertbackground="white", relief="flat").grid(row=1, column=1, padx=4)

    # ── Start button ─────────────────────────────────────────────────────────
    chosen = {}

    def on_start():
        m = mode_var.get()
        if m == "record":
            try:
                dur = float(dur_var.get())
                assert dur > 0
            except Exception:
                messagebox.showerror("Invalid Input", "Duration must be a positive number.")
                return
            chosen["mode"]     = "record"
            chosen["duration"] = dur
            chosen["save_dir"] = dir_var.get().strip() or "./recordings"
        else:
            chosen["mode"] = "live"
        root.destroy()

    tk.Button(root, text="Start", command=on_start, bg="#00e5ff", fg="black",
              font=("Segoe UI", 10, "bold"), relief="flat", padx=20, pady=6,
              cursor="hand2").pack(pady=16)

    root.mainloop()
    return chosen

# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    matplotlib.use("TkAgg")
    warnings.filterwarnings("ignore")

    cfg = launch_selector()
    if not cfg:
        print("No mode selected. exiting.")
        os._exit(0)

    if cfg["mode"] == "live":
        run_live_mode()
    elif cfg["mode"] == "record":
        run_record_mode(cfg["duration"], cfg["save_dir"])

    os._exit(0)