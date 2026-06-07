# ============================================================
#  ECG & HRV ANALYSIS DASHBOARD
#  Student: esha 
#  Subject: Biomedical Signal Processing — Experiment #5
#  Riphah International University, Lahore
#  Department of Biomedical Engineering
# ============================================================
#
#  HOW TO CHANGE NAME:
#  Search for "esha " and replace with your name
#  Search for "Esha Fatima"   and replace with your friend's name
#
# ============================================================

# ---------- IMPORTS ----------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')           # use this backend for pop-up window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider, RadioButtons
from scipy import signal
from scipy.signal import butter, filtfilt, find_peaks
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# ============================================================
#  STEP 1 – GENERATE REALISTIC SYNTHETIC ECG SIGNAL
#  (No hardware needed — works on any PC)
# ============================================================

def generate_ecg(duration=60, fs=500, heart_rate=72, noise_level=0.05, seed=42):
    """
    Generate a realistic ECG signal using the sum-of-Gaussians model.
    duration    : seconds
    fs          : sampling frequency (Hz)
    heart_rate  : beats per minute
    noise_level : amount of random noise to add
    """
    np.random.seed(seed)
    t = np.arange(0, duration, 1/fs)
    ecg = np.zeros(len(t))

    # --- PQRST Gaussian parameters (amplitude, mean, width) ---
    pqrst = {
        'P': ( 0.25, -0.20, 0.09),
        'Q': (-0.10, -0.05, 0.03),
        'R': ( 1.60,  0.00, 0.025),
        'S': (-0.25,  0.06, 0.03),
        'T': ( 0.45,  0.20, 0.10),
    }

    rr_interval = 60.0 / heart_rate          # seconds between beats
    beat_times  = np.arange(0, duration, rr_interval)

    # Add slight heart-rate variability
    beat_times += np.random.normal(0, 0.02, len(beat_times))
    beat_times  = np.clip(beat_times, 0, duration - 0.5)

    for bt in beat_times:
        for wave, (amp, offset, width) in pqrst.items():
            center = bt + offset
            ecg   += amp * np.exp(-((t - center)**2) / (2 * width**2))

    # Add baseline wander + noise
    bw_freq = 0.15
    ecg    += 0.05 * np.sin(2 * np.pi * bw_freq * t)
    ecg    += noise_level * np.random.randn(len(t))
    return t, ecg, fs


# ============================================================
#  STEP 2 – SIGNAL PREPROCESSING (FILTERS)
# ============================================================

def bandpass_filter(ecg, fs, low=0.5, high=40.0, order=4):
    """Butterworth band-pass filter to remove baseline wander and high-freq noise."""
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype='band')
    return filtfilt(b, a, ecg)


def notch_filter(ecg, fs, freq=50.0, Q=30):
    """Notch filter to remove 50 Hz power-line interference."""
    nyq  = fs / 2
    b, a = signal.iirnotch(freq/nyq, Q)
    return filtfilt(b, a, ecg)


# ============================================================
#  STEP 3 – R-PEAK DETECTION & RR-INTERVALS
# ============================================================

def detect_r_peaks(ecg_filtered, fs, min_distance_ms=400):
    """
    Detect R-peaks using scipy's find_peaks with a minimum distance
    constraint (no two peaks closer than min_distance_ms milliseconds).
    """
    min_dist_samples = int((min_distance_ms / 1000) * fs)
    threshold = 0.5 * np.max(ecg_filtered)
    peaks, props = find_peaks(ecg_filtered,
                              height=threshold,
                              distance=min_dist_samples)
    return peaks


def compute_rr_intervals(r_peaks, fs):
    """Convert R-peak sample indices to RR intervals in milliseconds."""
    rr_ms = np.diff(r_peaks) / fs * 1000   # milliseconds
    return rr_ms


# ============================================================
#  STEP 4 – TIME-DOMAIN HRV FEATURES
# ============================================================

def time_domain_hrv(rr_ms):
    """
    Compute standard time-domain HRV metrics.

    SDNN   : Standard deviation of all NN intervals
    RMSSD  : Root mean square of successive differences
    pNN50  : Percentage of successive differences > 50 ms
    Mean HR: Mean heart rate
    """
    if len(rr_ms) < 2:
        return {}

    sdnn    = np.std(rr_ms, ddof=1)
    diff_rr = np.diff(rr_ms)
    rmssd   = np.sqrt(np.mean(diff_rr**2))
    pnn50   = (np.sum(np.abs(diff_rr) > 50) / len(diff_rr)) * 100
    mean_rr = np.mean(rr_ms)
    mean_hr = 60000 / mean_rr

    return {
        'Mean RR (ms)'  : round(mean_rr, 2),
        'SDNN (ms)'     : round(sdnn, 2),
        'RMSSD (ms)'    : round(rmssd, 2),
        'pNN50 (%)'     : round(pnn50, 2),
        'Mean HR (bpm)' : round(mean_hr, 2),
        'Min HR (bpm)'  : round(60000 / np.max(rr_ms), 2),
        'Max HR (bpm)'  : round(60000 / np.min(rr_ms), 2),
    }


# ============================================================
#  STEP 5 – FREQUENCY-DOMAIN HRV (PSD — LF / HF BANDS)
# ============================================================

def frequency_domain_hrv(rr_ms, method='welch'):
    """
    Compute LF/HF power from the RR tachogram using Welch's PSD method.

    VLF : 0.003 – 0.04  Hz  (Very Low Frequency)
    LF  : 0.04  – 0.15  Hz  (Low Frequency  — sympathetic + parasympathetic)
    HF  : 0.15  – 0.40  Hz  (High Frequency — parasympathetic / vagal)
    """
    if len(rr_ms) < 10:
        return {}, None, None

    # Interpolate RR intervals onto uniform time grid (4 Hz)
    rr_times  = np.cumsum(rr_ms) / 1000          # seconds
    rr_times  = np.insert(rr_times, 0, 0)
    rr_values = np.insert(rr_ms, 0, rr_ms[0])

    fs_interp = 4.0
    t_uniform = np.arange(rr_times[0], rr_times[-1], 1/fs_interp)
    interp_fn = interp1d(rr_times, rr_values, kind='cubic',
                         fill_value='extrapolate')
    rr_uniform = interp_fn(t_uniform)

    # Welch PSD
    freqs, psd = signal.welch(rr_uniform,
                               fs=fs_interp,
                               nperseg=min(256, len(rr_uniform)//2),
                               window='hann')

    # Band powers
    def band_power(f, p, flo, fhi):
        idx = (f >= flo) & (f <= fhi)
        return np.trapezoid(p[idx], f[idx])

    vlf = band_power(freqs, psd, 0.003, 0.04)
    lf  = band_power(freqs, psd, 0.04,  0.15)
    hf  = band_power(freqs, psd, 0.15,  0.40)
    tp  = vlf + lf + hf

    lf_hf = lf / hf if hf > 0 else np.nan
    lf_nu = (lf / (lf + hf)) * 100 if (lf + hf) > 0 else np.nan
    hf_nu = (hf / (lf + hf)) * 100 if (lf + hf) > 0 else np.nan

    results = {
        'VLF Power (ms²)' : round(vlf, 2),
        'LF Power (ms²)'  : round(lf, 2),
        'HF Power (ms²)'  : round(hf, 2),
        'Total Power'     : round(tp, 2),
        'LF/HF Ratio'     : round(lf_hf, 3),
        'LF (n.u.)'       : round(lf_nu, 2),
        'HF (n.u.)'       : round(hf_nu, 2),
    }
    return results, freqs, psd


# ============================================================
#  STEP 6 – NON-LINEAR HRV (POINCARÉ + ENTROPY)
# ============================================================

def nonlinear_hrv(rr_ms):
    """
    Poincaré plot descriptors (SD1, SD2) and Sample Entropy.

    SD1 : Short-term variability (beat-to-beat, parasympathetic)
    SD2 : Long-term variability (overall)
    """
    if len(rr_ms) < 4:
        return {}

    diff_rr = np.diff(rr_ms)
    sd1 = np.std(diff_rr, ddof=1) / np.sqrt(2)
    sd2 = np.sqrt(2 * np.std(rr_ms, ddof=1)**2 - 0.5 * np.std(diff_rr, ddof=1)**2)

    # Approximate Entropy (simple version)
    def approx_entropy(ts, m=2, r_factor=0.2):
        r = r_factor * np.std(ts)
        N = len(ts)
        def phi(m):
            count = 0
            templates = np.array([ts[i:i+m] for i in range(N-m+1)])
            for tmpl in templates:
                diffs = np.max(np.abs(templates - tmpl), axis=1)
                count += np.sum(diffs <= r) - 1
            return count / ((N - m + 1) * (N - m))
        return np.log(phi(m) / phi(m+1)) if phi(m+1) > 0 else 0

    apen = approx_entropy(rr_ms[:50])   # use subset for speed

    return {
        'SD1 (ms)'  : round(sd1, 2),
        'SD2 (ms)'  : round(sd2, 2),
        'SD2/SD1'   : round(sd2/sd1, 3) if sd1 > 0 else 'N/A',
        'Approx. Entropy': round(float(apen), 4),
    }


# ============================================================
#  STEP 7 – INTERACTIVE DASHBOARD (MATPLOTLIB)
# ============================================================

class ECGHRVDashboard:
    """
    Full interactive ECG & HRV dashboard.
    Controls:
      • Slider  : zoom into ECG window
      • Buttons : regenerate with Normal / Stress / Sleep presets
    """

    def __init__(self):
        # --- Generate data ---
        self.fs = 500
        self._generate_data(heart_rate=72, noise=0.05, seed=42)

        # --- Build figure ---
        self.fig = plt.figure(figsize=(20, 13), facecolor='#0d1117')
        self.fig.canvas.manager.set_window_title(
            'ECG & HRV Dashboard — esha  | Riphah International University')
        self._build_layout()
        self._plot_all()
        plt.tight_layout(rect=[0, 0.05, 1, 0.96])
        plt.show()

    # ----------------------------------------------------------
    def _generate_data(self, heart_rate=72, noise=0.05, seed=42):
        self.t, self.ecg_raw, self.fs = generate_ecg(
            duration=60, fs=self.fs,
            heart_rate=heart_rate, noise_level=noise, seed=seed)
        self.ecg_filt   = notch_filter(
                            bandpass_filter(self.ecg_raw, self.fs), self.fs)
        self.r_peaks    = detect_r_peaks(self.ecg_filt, self.fs)
        self.rr_ms      = compute_rr_intervals(self.r_peaks, self.fs)
        self.td_features = time_domain_hrv(self.rr_ms)
        self.fd_results, self.freqs, self.psd = frequency_domain_hrv(self.rr_ms)
        self.nl_features = nonlinear_hrv(self.rr_ms)

    # ----------------------------------------------------------
    def _build_layout(self):
        C = {
            'bg'     : '#0d1117',
            'panel'  : '#161b22',
            'accent' : '#58a6ff',
            'green'  : '#3fb950',
            'red'    : '#f85149',
            'yellow' : '#d29922',
            'purple' : '#bc8cff',
            'text'   : '#e6edf3',
            'muted'  : '#8b949e',
        }
        self.C = C

        gs_main = gridspec.GridSpec(4, 3, figure=self.fig,
                                    hspace=0.55, wspace=0.38,
                                    left=0.06, right=0.97,
                                    top=0.92, bottom=0.18)

        # Row 0 – ECG (full width)
        self.ax_ecg  = self.fig.add_subplot(gs_main[0, :])
        # Row 1 – filtered ECG + R-peaks | RR tachogram
        self.ax_filt = self.fig.add_subplot(gs_main[1, :2])
        self.ax_rr   = self.fig.add_subplot(gs_main[1, 2])
        # Row 2 – PSD | Poincaré | Time-domain metrics
        self.ax_psd  = self.fig.add_subplot(gs_main[2, 0])
        self.ax_pc   = self.fig.add_subplot(gs_main[2, 1])
        self.ax_td   = self.fig.add_subplot(gs_main[2, 2])
        # Row 3 – Frequency metrics | Non-linear metrics | Autonomic balance
        self.ax_fd   = self.fig.add_subplot(gs_main[3, 0])
        self.ax_nl   = self.fig.add_subplot(gs_main[3, 1])
        self.ax_auto = self.fig.add_subplot(gs_main[3, 2])

        for ax in self.fig.axes:
            ax.set_facecolor(C['panel'])
            ax.tick_params(colors=C['muted'], labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')

        # Title
        self.fig.text(0.5, 0.965,
                      'ECG & HRV ANALYSIS DASHBOARD',
                      ha='center', va='top', fontsize=17,
                      color=C['accent'], fontweight='bold',
                      fontfamily='monospace')
        self.fig.text(0.5, 0.948,
                      'esha   |  Riphah International University  |  Biomedical Signal Processing',
                      ha='center', va='top', fontsize=9,
                      color=C['muted'], fontfamily='monospace')

        # --- Slider ---
        ax_slider = self.fig.add_axes([0.15, 0.09, 0.55, 0.025],
                                       facecolor='#21262d')
        self.slider = Slider(ax_slider, 'ECG Window Start (s)',
                             0, 50, valinit=0, color=C['accent'])
        self.slider.label.set_color(C['text'])
        self.slider.valtext.set_color(C['accent'])
        self.slider.on_changed(self._on_slider)

        # --- Preset buttons ---
        btn_positions = [(0.15, 0.04), (0.30, 0.04), (0.45, 0.04)]
        labels  = ['Normal (72 bpm)', 'Stress (100 bpm)', 'Sleep (55 bpm)']
        colors  = [C['green'], C['red'], C['purple']]
        configs = [(72, 0.05, 42), (100, 0.08, 7), (55, 0.03, 99)]
        self._btn_axes = []
        self._buttons  = []
        for (x, y), lbl, col, cfg in zip(btn_positions, labels, colors, configs):
            bax = self.fig.add_axes([x, y, 0.12, 0.03], facecolor='#21262d')
            btn = Button(bax, lbl, color='#21262d', hovercolor=col)
            btn.label.set_color(col)
            btn.label.set_fontsize(8)
            btn._cfg = cfg
            btn.on_clicked(self._on_preset)
            self._btn_axes.append(bax)
            self._buttons.append(btn)

    # ----------------------------------------------------------
    def _clear_axes(self):
        for ax in [self.ax_ecg, self.ax_filt, self.ax_rr,
                   self.ax_psd, self.ax_pc, self.ax_td,
                   self.ax_fd, self.ax_nl, self.ax_auto]:
            ax.cla()
            ax.set_facecolor(self.C['panel'])
            ax.tick_params(colors=self.C['muted'], labelsize=7)
            for spine in ax.spines.values():
                spine.set_edgecolor('#30363d')

    # ----------------------------------------------------------
    def _plot_all(self, window_start=0):
        C = self.C
        win = 10          # seconds shown in ECG window
        i0  = int(window_start * self.fs)
        i1  = int((window_start + win) * self.fs)
        i1  = min(i1, len(self.t))

        # ---- 1. Raw ECG ----------------------------------------
        ax = self.ax_ecg
        ax.plot(self.t[i0:i1], self.ecg_raw[i0:i1],
                color=C['muted'], lw=0.6, alpha=0.6, label='Raw ECG')
        ax.plot(self.t[i0:i1], self.ecg_filt[i0:i1],
                color=C['accent'], lw=1.1, label='Filtered ECG')
        # R-peaks in window
        mask = (self.r_peaks >= i0) & (self.r_peaks < i1)
        rp_win = self.r_peaks[mask]
        ax.plot(self.t[rp_win], self.ecg_filt[rp_win],
                'v', color=C['red'], ms=7, label='R-peaks')
        ax.set_title('ECG Signal — Raw vs Filtered', color=C['text'],
                     fontsize=9, pad=4)
        ax.set_xlabel('Time (s)', color=C['muted'], fontsize=7)
        ax.set_ylabel('Amplitude (mV)', color=C['muted'], fontsize=7)
        ax.legend(loc='upper right', fontsize=7,
                  facecolor='#21262d', edgecolor='#30363d',
                  labelcolor=C['text'])
        ax.grid(True, color='#21262d', lw=0.5)

        # ---- 2. Filtered ECG (full signal) ---------------------
        ax = self.ax_filt
        ax.plot(self.t, self.ecg_filt,
                color=C['green'], lw=0.7, alpha=0.8)
        ax.plot(self.t[self.r_peaks], self.ecg_filt[self.r_peaks],
                'v', color=C['red'], ms=4, zorder=5)
        ax.axvspan(window_start, window_start + win,
                   alpha=0.12, color=C['accent'])
        ax.set_title(f'Filtered ECG — Full 60s  |  {len(self.r_peaks)} beats detected',
                     color=C['text'], fontsize=9, pad=4)
        ax.set_xlabel('Time (s)', color=C['muted'], fontsize=7)
        ax.set_ylabel('Amplitude (mV)', color=C['muted'], fontsize=7)
        ax.grid(True, color='#21262d', lw=0.5)

        # ---- 3. RR Tachogram -----------------------------------
        ax = self.ax_rr
        rr_t = np.cumsum(self.rr_ms) / 1000
        ax.plot(rr_t, self.rr_ms, color=C['yellow'], lw=1.2, marker='o',
                ms=2.5, markerfacecolor=C['red'])
        ax.fill_between(rr_t, self.rr_ms,
                        alpha=0.15, color=C['yellow'])
        ax.set_title('RR Tachogram', color=C['text'], fontsize=9, pad=4)
        ax.set_xlabel('Time (s)', color=C['muted'], fontsize=7)
        ax.set_ylabel('RR Interval (ms)', color=C['muted'], fontsize=7)
        ax.grid(True, color='#21262d', lw=0.5)

        # ---- 4. Power Spectral Density -------------------------
        ax = self.ax_psd
        if self.freqs is not None:
            psd_db = 10 * np.log10(self.psd + 1e-12)
            ax.semilogy(self.freqs, self.psd,
                        color=C['accent'], lw=1.2)
            ax.axvspan(0.003, 0.04,  alpha=0.18, color=C['purple'],  label='VLF')
            ax.axvspan(0.04,  0.15,  alpha=0.22, color=C['yellow'],  label='LF')
            ax.axvspan(0.15,  0.40,  alpha=0.22, color=C['green'],   label='HF')
            ax.set_xlim(0, 0.45)
            ax.set_title('HRV Power Spectral Density', color=C['text'],
                         fontsize=9, pad=4)
            ax.set_xlabel('Frequency (Hz)', color=C['muted'], fontsize=7)
            ax.set_ylabel('PSD (ms²/Hz)', color=C['muted'], fontsize=7)
            ax.legend(loc='upper right', fontsize=6,
                      facecolor='#21262d', edgecolor='#30363d',
                      labelcolor=C['text'])
            ax.grid(True, color='#21262d', lw=0.4, which='both')

        # ---- 5. Poincaré Plot ----------------------------------
        ax = self.ax_pc
        if len(self.rr_ms) > 2:
            rr1 = self.rr_ms[:-1]
            rr2 = self.rr_ms[1:]
            ax.scatter(rr1, rr2, c=C['accent'], s=10, alpha=0.6)
            nl = self.nl_features
            sd1 = nl.get('SD1 (ms)', 0)
            sd2 = nl.get('SD2 (ms)', 0)
            cx  = np.mean(rr1)
            cy  = np.mean(rr2)
            theta = np.pi / 4
            for sd, col, lbl in [(sd1, C['red'], f'SD1={sd1}ms'),
                                  (sd2, C['yellow'], f'SD2={sd2}ms')]:
                ell_x = sd * np.cos(np.linspace(0, 2*np.pi, 100))
                ell_y = sd * np.sin(np.linspace(0, 2*np.pi, 100))
                x_rot = ell_x * np.cos(theta) - ell_y * np.sin(theta) + cx
                y_rot = ell_x * np.sin(theta) + ell_y * np.cos(theta) + cy
                ax.plot(x_rot, y_rot, color=col, lw=1.4, label=lbl)
            ax.set_title('Poincaré Plot', color=C['text'], fontsize=9, pad=4)
            ax.set_xlabel('RRn (ms)',   color=C['muted'], fontsize=7)
            ax.set_ylabel('RRn+1 (ms)', color=C['muted'], fontsize=7)
            ax.legend(loc='upper left', fontsize=6,
                      facecolor='#21262d', edgecolor='#30363d',
                      labelcolor=C['text'])
            ax.grid(True, color='#21262d', lw=0.4)

        # ---- 6. Time-Domain Metrics (bar chart) ----------------
        ax = self.ax_td
        td = self.td_features
        keys   = ['SDNN (ms)', 'RMSSD (ms)', 'pNN50 (%)']
        values = [td.get(k, 0) for k in keys]
        bars   = ax.barh(keys, values,
                         color=[C['accent'], C['green'], C['yellow']],
                         height=0.5)
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}', va='center', color=C['text'], fontsize=8)
        ax.set_title('Time-Domain HRV', color=C['text'], fontsize=9, pad=4)
        ax.set_xlabel('Value', color=C['muted'], fontsize=7)
        ax.grid(True, color='#21262d', lw=0.4, axis='x')
        ax.tick_params(colors=C['text'], labelsize=8)

        # ---- 7. Frequency Metrics table ------------------------
        ax = self.ax_fd
        ax.axis('off')
        fd = self.fd_results
        rows = [['Metric', 'Value']]
        for k, v in fd.items():
            rows.append([k, str(v)])
        tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
                       loc='center', cellLoc='left')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(7.5)
        tbl.scale(1, 1.35)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_facecolor('#21262d' if r % 2 == 0 else '#161b22')
            cell.set_edgecolor('#30363d')
            cell.set_text_props(color=C['text'] if r > 0 else C['accent'])
        ax.set_title('Frequency-Domain HRV', color=C['text'], fontsize=9, pad=4)

        # ---- 8. Non-linear Metrics table -----------------------
        ax = self.ax_nl
        ax.axis('off')
        nl = self.nl_features
        rows_nl = [['Metric', 'Value']]
        for k, v in nl.items():
            rows_nl.append([k, str(v)])
        tbl2 = ax.table(cellText=rows_nl[1:], colLabels=rows_nl[0],
                        loc='center', cellLoc='left')
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(7.5)
        tbl2.scale(1, 1.35)
        for (r, c), cell in tbl2.get_celld().items():
            cell.set_facecolor('#21262d' if r % 2 == 0 else '#161b22')
            cell.set_edgecolor('#30363d')
            cell.set_text_props(color=C['text'] if r > 0 else C['purple'])
        ax.set_title('Non-Linear HRV (Poincaré & Entropy)',
                     color=C['text'], fontsize=9, pad=4)

        # ---- 9. Autonomic Balance (donut chart) ----------------
        ax = self.ax_auto
        lf  = self.fd_results.get('LF Power (ms²)', 1)
        hf  = self.fd_results.get('HF Power (ms²)', 1)
        vlf = self.fd_results.get('VLF Power (ms²)', 1)
        sizes  = [vlf, lf, hf]
        labels = ['VLF\n(Very Low)', 'LF\n(Sympathetic)', 'HF\n(Parasympathetic)']
        colors = [C['purple'], C['yellow'], C['green']]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=90,
            wedgeprops=dict(width=0.55, edgecolor='#0d1117', linewidth=2),
            textprops=dict(color=C['text'], fontsize=7))
        for at in autotexts:
            at.set_fontsize(7)
            at.set_color('#0d1117')
        ax.set_title('Autonomic Nervous System Balance',
                     color=C['text'], fontsize=9, pad=4)
        lf_hf = self.fd_results.get('LF/HF Ratio', 0)
        ax.text(0, -1.35, f'LF/HF Ratio = {lf_hf}',
                ha='center', color=C['yellow'], fontsize=8,
                fontweight='bold')

        self.fig.canvas.draw_idle()

    # ----------------------------------------------------------
    def _on_slider(self, val):
        self._clear_axes()
        self._plot_all(window_start=val)

    def _on_preset(self, event):
        for btn in self._buttons:
            if event.inaxes == btn.ax:
                hr, noise, seed = btn._cfg
                self._generate_data(heart_rate=hr, noise=noise, seed=seed)
                self._clear_axes()
                self._plot_all(window_start=self.slider.val)
                break


# ============================================================
#  MAIN — Run the dashboard
# ============================================================

if __name__ == '__main__':
    print("=" * 60)
    print("  ECG & HRV Dashboard")
    print("  Student : esha ")
    print("  Subject : Biomedical Signal Processing")
    print("  Riphah International University, Lahore")
    print("=" * 60)
    print("\nGenerating ECG signal and computing HRV features...")
    dashboard = ECGHRVDashboard()
