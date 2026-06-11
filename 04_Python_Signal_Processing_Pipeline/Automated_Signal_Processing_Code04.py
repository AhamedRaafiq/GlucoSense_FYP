import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis
from scipy.signal import find_peaks, welch, savgol_filter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import json
import glob
import shutil
import tkinter as tk
from tkinter import filedialog
import sys
from datetime import datetime

# ==========================================
# 🔧 STEP 1: MASTER HYPERPARAMETERS
# ==========================================

# --- File I/O ---
INPUT_ROOT_PATH = r'C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\03_Windowed'  # Default search dir
OUTPUT_ROOT_PATH = r'C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\04_Filtered' # Save location
SAVE_ROOT_FIXED = OUTPUT_ROOT_PATH

# --- Signal Properties ---
FS = 400.0                            # Sampling Frequency (Hz). Typical: 100-1000
USE_FULL_FILE = True                  # Process whole CSV if no manual cut needed

# --- Data Selection ---
SAMPLE_START = 0                      # Start index (0 for full)
SAMPLE_END = 0                        # End index (0 for full)
THRESHOLD_RED = 0                     # Validity threshold for RED channel
THRESHOLD_IR = 0                      # Validity threshold for IR channel

# --- Pre-processing ---
SPIKE_ENABLE = True                   # Enable median filter?
KERNEL_SIZE = 3                       # Window size for median filter (Odd integer: 3-9)
INVERT_ENABLE = True                  # Flip signal so systole peaks up? (True for MAX30102)

# --- Filters ---
LP_ENABLE = True                      # Low Pass Filter (Noise removal)
LP_CUTOFF = 16.0                      # Cutoff frequency (Hz). Typical: 10-20
LP_ORDER = 4                          # Butterworth order. Higher = steeper but more phase delay

HP_ENABLE = True                      # High Pass Filter (Drift removal)
HP_CUTOFF = 0.5                       # Cutoff frequency (Hz). Typical: 0.3-1.0
HP_ORDER = 4                          # Butterworth order

SG_ENABLE = False                     # Savitzky-Golay Smoothing?
SG_WINDOW = 31                        # Must be odd integer (> Poly Order). Smoothness vs Noise
SG_POLY = 3                           # Polynomial order. Typical: 2-4

# --- Normalization ---
NORM_SELECTION = 1                    # 1=MinMax (0-1), 2=ZScore (Mean=0, Std=1)

# --- Ensemble Settings ---
ENSEMBLE_TARGET_LEN = 220             # Target length for resampled average beat

# 🔑 WINDOW REJECTION LIMITS (Tune per channel)
MIN_VALID_BEATS_IR = 8                # Min valid beats required for IR ensemble (reject window if less)
MIN_VALID_BEATS_RED = 8               # Min valid beats required for RED ensemble (reject window if less)

PEAK_MIN_DISTANCE_SEC = 0.40          # Min time between peaks (sec). Heart rate dependent
PEAK_PROM_FACTOR = 0.20               # Peak prominence factor relative to STD
VALLEY_MIN_DISTANCE_SEC = 0.35        # Min time between valleys (sec)
VALLEY_PROM_FACTOR = 0.10             # Valley prominence factor
MIN_FOOT_TO_PEAK_SEC = 0.12           # Min valid foot-to-peak duration
MAX_FOOT_TO_PEAK_SEC = 0.4            # Max valid foot-to-peak duration
MAX_VALLEY_TO_FOOT_SEC = 0.20         # Max distance valley to refined foot
MAX_FOOT_REL_HEIGHT = 0.20            # Max relative height of foot within pulse
MAX_ABS_VPG_AT_FOOT = 0.5             # Max VPG value allowed at foot (near zero-cross)
EDGE_EXCLUSION_SEC = 0.1              # Ignore candidates near signal edges
MIN_BEAT_DURATION_SEC = 0.35          # Min valid beat duration
MAX_BEAT_DURATION_SEC = 1.50          # Max valid beat duration
MAIN_PEAK_SEARCH_WINDOW_SEC = 0.2     # Search window after foot for main peak
MAIN_PEAK_MIN_DELAY_SEC = 0.02        # Min delay before searching for main peak
START_INCOMPLETE_MARGIN_SEC = 0.10    # Margin for start edge incompleteness
END_INCOMPLETE_MARGIN_SEC = 0.01      # Margin for end edge incompleteness
VERBOSE_REJECTION = False             # Print rejected pulses? (Set True for debugging only)

# --- Plotting ---
SUBPLOT_HEIGHT = 4.0                  # Height of individual subplots (inches)
TIME_VERTICAL_LINE_INTERVAL = 1.0     # Grid interval for time axis (seconds)
FREQ_VERTICAL_LINE_INTERVAL = 1.0     # Grid interval for freq axis (Hz)

# --- Quality Control Limits ---
SQI_LIMITS = {
    'SKEWNESS_MIN': 0.0, 'SKEWNESS_MAX': 2.5,   # Shape symmetry
    'KURTOSIS_MIN': 1.5, 'KURTOSIS_MAX': 7.0,   # Sharpness
    'PI_MIN': 0.1,       'PI_MAX': 10.0,        # Perfusion Index (%)
    'SNR_MIN_DB': 5.0,   'SNR_MAX_DB': 25.0,    # Signal-to-Noise Ratio
    'ZCR_MIN': 1.0,      'ZCR_MAX': 4.0         # Zero Crossing Rate (Hz)
}

print("✅ Master Configuration Loaded.")

# ==========================================
# 🛠️ HELPER FUNCTIONS
# ==========================================

def apply_normalization(data, selection=1):
    """Normalizes signal: 1=MinMax, 2=ZScore."""
    data = np.asarray(data)
    if data.size == 0: return data
    if selection == 1: # MinMax
        den = np.max(data) - np.min(data)
        if den == 0: return np.zeros_like(data)
        return (data - np.min(data)) / den
    elif selection == 2: # Z-Score
        sd = np.std(data)
        if sd == 0: return np.zeros_like(data)
        return (data - np.mean(data)) / sd
    return data

def compute_fft(signal_data, fs):
    """Computes FFT magnitude spectrum."""
    signal_data = np.asarray(signal_data)
    N = len(signal_data)
    yf = fft(signal_data - np.mean(signal_data))
    xf = fftfreq(N, 1 / fs)
    return xf[:N // 2], 2.0 / N * np.abs(yf[:N // 2])

def calculate_snr(signal_data, fs):
    """Calculates SNR (0.5-5Hz band vs noise)."""
    freqs, psd = signal.periodogram(signal_data, fs)
    sig_mask = (freqs >= 0.5) & (freqs <= 5.0)
    noise_mask = ~sig_mask
    sig_power = np.sum(psd[sig_mask])
    noise_power = np.sum(psd[noise_mask])
    return 10 * np.log10(sig_power / noise_power) if noise_power > 0 else 100.0

def calculate_zcr(signal_data, fs):
    """Calculates Zero Crossing Rate (Hz)."""
    centered = signal_data - np.mean(signal_data)
    crossings = np.diff(np.signbit(centered)).sum()
    return crossings / (len(signal_data) / fs)

def to_json_safe(obj):
    """Recursively converts numpy/pandas objects to JSON serializable types."""
    if isinstance(obj, dict):
        return {str(k): to_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [to_json_safe(v) for v in obj]
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating): return float(obj)
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif pd.isna(obj): return None
    return str(obj) if not isinstance(obj, (str, int, float, bool)) else obj

def safe_savgol(x, win, poly=3):
    """Safe Savitzky-Golay with boundary checks."""
    x = np.asarray(x)
    n = len(x)
    if n < 5: return x.copy()
    win = int(win)
    if win % 2 == 0: win += 1
    if win >= n: win = n - 1 if (n - 1) % 2 == 1 else n - 2
    if win < 5: win = 5
    if win >= n or win <= poly: return x.copy()
    poly = min(poly, win - 2)
    return savgol_filter(x, window_length=win, polyorder=poly)

def resample_1d(sig, target_len):
    """Resamples signal to target length using interpolation."""
    x_old = np.linspace(0, 1, len(sig))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, sig)

def ensure_clean_output(root_path):
    """Ensures output directory exists and creates fresh structure if needed."""
    if os.path.exists(root_path):
        print(f"⚠️ Output path exists: {root_path}. New runs will overwrite specific window folders.")
    else:
        os.makedirs(root_path, exist_ok=True)
        print(f"📂 Created new output root: {root_path}")

# ==========================================
# 🖥️ STEP 1: SELECT INPUT (BATCH OR SINGLE)
# ==========================================

def prompt_processing_mode():
    """Prompts user to choose between Batch (all subfolders) or Single (one folder)."""
    print("\n" + "=" * 70)
    print("  SELECT PROCESSING MODE")
    print("=" * 70)
    print(f"  1) BATCH  — Process ALL subfolders inside:")
    print(f"              {INPUT_ROOT_PATH}")
    print(f"  2) SINGLE — Pop up dialog to choose ONE folder")
    print("=" * 70)
    while True:
        choice = input("  Enter choice [1 or 2]: ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("  ❌ Invalid choice. Please enter 1 or 2.")

def collect_folders_to_process(mode):
    """Returns a list of folders to process based on mode."""
    folders = []
    if mode == 1:
        if not os.path.isdir(INPUT_ROOT_PATH):
            raise FileNotFoundError(f"❌ Batch root not found: {INPUT_ROOT_PATH}")
        subfolders = sorted([
            os.path.join(INPUT_ROOT_PATH, d)
            for d in os.listdir(INPUT_ROOT_PATH)
            if os.path.isdir(os.path.join(INPUT_ROOT_PATH, d))
        ])
        if not subfolders:
            raise FileNotFoundError(f"❌ No subfolders inside: {INPUT_ROOT_PATH}")
        folders = subfolders
        print(f"\n📦 BATCH MODE — found {len(folders)} subfolder(s) to process:")
        for f in folders:
            print(f"   • {os.path.basename(f)}")
    else:
        selected = None
        # Try Tk dialog with safe init/teardown
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes('-topmost', True)
            root.update_idletasks()
            root.update()
            selected = filedialog.askdirectory(
                parent=root,
                initialdir=INPUT_ROOT_PATH,
                title="Select Input Folder (containing .csv files)"
            )
            root.update()
            root.destroy()
        except Exception as tk_err:
            print(f"\n⚠️  GUI dialog failed: {tk_err}")
            print("    Falling back to manual path entry.")
            selected = None
        
        # Fallback: manual entry
        if not selected:
            print(f"\n📁 Default base path: {INPUT_ROOT_PATH}")
            typed = input("    Type folder path (or press Enter to use default): ").strip().strip('"').strip("'")
            if typed:
                selected = typed
            else:
                selected = INPUT_ROOT_PATH
        
        if not selected or not os.path.isdir(selected):
            raise ValueError(f"❌ Invalid or missing folder: {selected}")
        
        folders = [selected]
        print(f"\n📁 SINGLE MODE — selected: {selected}")
    return folders

MODE = prompt_processing_mode()
folders_to_process = collect_folders_to_process(MODE)

ensure_clean_output(SAVE_ROOT_FIXED)


#------------------------------------------------------------------------------------------------------------------


# ==========================================
# ⚙️ STEP 2-8: SIGNAL PROCESSING CORE FUNCTIONS
# ==========================================

def save_and_close(fig, save_path, filename):
    """Saves a figure to disk and closes it (no display)."""
    os.makedirs(save_path, exist_ok=True)
    full_path = os.path.join(save_path, filename)
    fig.savefig(full_path, dpi=100, bbox_inches='tight')
    plt.close(fig)

# ------------------------------------------
# STEP 2: LOAD CSV & VALIDATE
# ------------------------------------------
def load_and_validate_csv(filepath):
    """Loads CSV, normalizes column names, applies validity thresholds."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.upper()
    
    rename_map = {
        'RED_VALUE': 'RED', 'IR_VALUE': 'IR',
        'RED VALUE': 'RED', 'IR VALUE': 'IR',
        'RED_RAW': 'RED',   'IR_RAW': 'IR'
    }
    df = df.rename(columns=rename_map)
    
    required_cols = {'RED', 'IR'}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    mask = (df['RED'] > THRESHOLD_RED) & (df['IR'] > THRESHOLD_IR)
    full_clean_df = df.loc[mask, ['RED', 'IR']].copy().reset_index(drop=True)
    
    if USE_FULL_FILE:
        selected_df = full_clean_df.copy()
        s_start, s_end = 0, len(selected_df)
    else:
        s_start = SAMPLE_START
        s_end = SAMPLE_END if SAMPLE_END > 0 else len(full_clean_df)
        s_end = min(s_end, len(full_clean_df))
        selected_df = full_clean_df.iloc[s_start:s_end].copy().reset_index(drop=True)
    
    if len(selected_df) < 50:
        raise ValueError(f"Selected range too short ({len(selected_df)} samples)")
    
    return full_clean_df, selected_df, s_start, s_end

# ------------------------------------------
# STEP 2 PLOT: RAW + FFT
# ------------------------------------------
def plot_step2_raw(full_clean_df, selected_df, s_start, s_end, save_path, base_name):
    """Plots raw full signal, selection area, time domain & FFT."""
    time_axis = np.arange(len(selected_df)) / FS
    fig, axes = plt.subplots(6, 1, figsize=(12, SUBPLOT_HEIGHT * 6))
    plt.subplots_adjust(hspace=0.6)
    
    # Full IR
    axes[0].plot(full_clean_df['IR'].values, color='black', alpha=0.6, label='Full IR')
    axes[0].axvspan(s_start, s_end, color='yellow', alpha=0.3, label='Selected')
    axes[0].set_title(f'FULL RAW (IR) - {base_name}')
    axes[0].legend(loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # Full RED
    axes[1].plot(full_clean_df['RED'].values, color='darkred', alpha=0.6, label='Full RED')
    axes[1].axvspan(s_start, s_end, color='yellow', alpha=0.3)
    axes[1].set_title('FULL RAW (RED)')
    axes[1].grid(True, alpha=0.3)
    
    # Selected IR
    axes[2].plot(time_axis, selected_df['IR'].values, color='black', linewidth=1.5)
    axes[2].set_title(f'SELECTED RAW (IR) - {len(selected_df)/FS:.1f}s')
    axes[2].set_xlabel('Time (s)')
    axes[2].grid(True, alpha=0.3)
    
    # Selected RED
    axes[3].plot(time_axis, selected_df['RED'].values, color='red', linewidth=1.5)
    axes[3].set_title('SELECTED RAW (RED)')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)
    
    # FFT IR
    xf_ir, yf_ir = compute_fft(selected_df['IR'].values, FS)
    axes[4].plot(xf_ir, yf_ir, color='purple')
    axes[4].set_title('FFT (IR)')
    axes[4].set_xlim(0, 20)
    axes[4].grid(True, alpha=0.3)
    
    # FFT RED
    xf_red, yf_red = compute_fft(selected_df['RED'].values, FS)
    axes[5].plot(xf_red, yf_red, color='orange')
    axes[5].set_title('FFT (RED)')
    axes[5].set_xlim(0, 20)
    axes[5].set_xlabel('Frequency (Hz)')
    axes[5].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_close(fig, save_path, f"{base_name}_02_Raw.png")

# ------------------------------------------
# STEP 3: SPIKE REMOVAL
# ------------------------------------------
def step3_spike_removal(selected_df, save_path, base_name):
    if SPIKE_ENABLE:
        ir_despiked = signal.medfilt(selected_df['IR'].values, kernel_size=KERNEL_SIZE)
        red_despiked = signal.medfilt(selected_df['RED'].values, kernel_size=KERNEL_SIZE)
    else:
        ir_despiked = selected_df['IR'].values.copy()
        red_despiked = selected_df['RED'].values.copy()
    
    time_axis = np.arange(len(selected_df)) / FS
    fig, axes = plt.subplots(4, 1, figsize=(12, SUBPLOT_HEIGHT * 4))
    plt.subplots_adjust(hspace=0.6)
    
    axes[0].plot(time_axis, selected_df['IR'].values, color='lightgray', label='Raw', linewidth=2)
    axes[0].plot(time_axis, ir_despiked, color='black', linewidth=1.5, label='Despiked')
    axes[0].set_title(f'STEP 3: IR Despike - {base_name}')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_axis, selected_df['RED'].values, color='mistyrose', label='Raw', linewidth=2)
    axes[1].plot(time_axis, red_despiked, color='red', linewidth=1.5, label='Despiked')
    axes[1].set_title('STEP 3: RED Despike')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    xf_ir_raw, yf_ir_raw = compute_fft(selected_df['IR'].values, FS)
    xf_ir_des, yf_ir_des = compute_fft(ir_despiked, FS)
    axes[2].plot(xf_ir_raw, yf_ir_raw, color='lightgray', linewidth=2, label='Raw')
    axes[2].plot(xf_ir_des, yf_ir_des, color='black', linewidth=1.5, label='Despiked')
    axes[2].set_xlim(0, 20); axes[2].set_title('IR FFT'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    xf_red_raw, yf_red_raw = compute_fft(selected_df['RED'].values, FS)
    xf_red_des, yf_red_des = compute_fft(red_despiked, FS)
    axes[3].plot(xf_red_raw, yf_red_raw, color='mistyrose', linewidth=2, label='Raw')
    axes[3].plot(xf_red_des, yf_red_des, color='red', linewidth=1.5, label='Despiked')
    axes[3].set_xlim(0, 20); axes[3].set_title('RED FFT'); axes[3].set_xlabel('Freq (Hz)')
    axes[3].legend(); axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_close(fig, save_path, f"{base_name}_03_Despiked.png")
    
    return ir_despiked, red_despiked

# ------------------------------------------
# STEP 4: SIGNAL INVERSION
# ------------------------------------------
def step4_inversion(ir_despiked, red_despiked, save_path, base_name):
    if INVERT_ENABLE:
        ir_inverted = -1 * ir_despiked
        red_inverted = -1 * red_despiked
    else:
        ir_inverted = ir_despiked.copy()
        red_inverted = red_despiked.copy()
    
    time_axis = np.arange(len(ir_despiked)) / FS
    fig, axes = plt.subplots(4, 1, figsize=(12, SUBPLOT_HEIGHT * 4), sharex=True)
    plt.subplots_adjust(hspace=0.6)
    
    axes[0].plot(time_axis, ir_despiked, color='lightgray', linewidth=1.5)
    axes[0].set_title(f'STEP 4: IR BEFORE Inversion - {base_name}')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_axis, ir_inverted, color='black', linewidth=1.5)
    axes[1].set_title(f'STEP 4: IR AFTER Inversion (Enabled={INVERT_ENABLE})')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(time_axis, red_despiked, color='lightcoral', linewidth=1.5)
    axes[2].set_title('STEP 4: RED BEFORE Inversion')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(time_axis, red_inverted, color='red', linewidth=1.5)
    axes[3].set_title('STEP 4: RED AFTER Inversion')
    axes[3].set_xlabel('Time (s)')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_close(fig, save_path, f"{base_name}_04_Inverted.png")
    
    return ir_inverted, red_inverted

# ------------------------------------------
# STEP 5: LOW-PASS FILTER
# ------------------------------------------
def step5_lowpass(ir_inverted, red_inverted, save_path, base_name):
    if LP_ENABLE:
        nyquist = 0.5 * FS
        sos = signal.butter(LP_ORDER, LP_CUTOFF / nyquist, btype='low', output='sos')
        ir_filtered = signal.sosfiltfilt(sos, ir_inverted)
        red_filtered = signal.sosfiltfilt(sos, red_inverted)
    else:
        ir_filtered = ir_inverted.copy()
        red_filtered = red_inverted.copy()
    
    time_axis = np.arange(len(ir_inverted)) / FS
    fig, axes = plt.subplots(4, 1, figsize=(12, SUBPLOT_HEIGHT * 4))
    plt.subplots_adjust(hspace=0.6)
    
    axes[0].plot(time_axis, ir_inverted, color='lightgray', label='Input', linewidth=1.5)
    axes[0].plot(time_axis, ir_filtered, color='black', label='LP Out', linewidth=1.5)
    axes[0].set_title(f'STEP 5: IR Low-Pass ({LP_CUTOFF}Hz) - {base_name}')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_axis, red_inverted, color='mistyrose', label='Input', linewidth=1.5)
    axes[1].plot(time_axis, red_filtered, color='red', label='LP Out', linewidth=1.5)
    axes[1].set_title('STEP 5: RED Low-Pass')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    xf_ir_in, yf_ir_in = compute_fft(ir_inverted, FS)
    xf_ir_out, yf_ir_out = compute_fft(ir_filtered, FS)
    axes[2].plot(xf_ir_in, yf_ir_in, color='lightgray', label='In', linewidth=1.5)
    axes[2].plot(xf_ir_out, yf_ir_out, color='black', label='Out', linewidth=1.5)
    axes[2].set_xlim(0, 20); axes[2].set_title('IR FFT'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    xf_red_in, yf_red_in = compute_fft(red_inverted, FS)
    xf_red_out, yf_red_out = compute_fft(red_filtered, FS)
    axes[3].plot(xf_red_in, yf_red_in, color='mistyrose', label='In', linewidth=1.5)
    axes[3].plot(xf_red_out, yf_red_out, color='red', label='Out', linewidth=1.5)
    axes[3].set_xlim(0, 20); axes[3].set_title('RED FFT'); axes[3].set_xlabel('Freq (Hz)')
    axes[3].legend(); axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_close(fig, save_path, f"{base_name}_05_LowPass.png")
    
    return ir_filtered, red_filtered

# ------------------------------------------
# STEP 6: SAVITZKY-GOLAY SMOOTHING
# ------------------------------------------
def step6_savgol(ir_filtered, red_filtered, save_path, base_name):
    safe_window = int(SG_WINDOW)
    safe_poly = int(SG_POLY)
    if safe_window % 2 == 0: safe_window += 1
    if safe_window <= safe_poly:
        safe_window = safe_poly + 2
        if safe_window % 2 == 0: safe_window += 1
    
    if SG_ENABLE:
        ir_smoothed = signal.savgol_filter(ir_filtered, window_length=safe_window, polyorder=safe_poly)
        red_smoothed = signal.savgol_filter(red_filtered, window_length=safe_window, polyorder=safe_poly)
    else:
        ir_smoothed = ir_filtered.copy()
        red_smoothed = red_filtered.copy()
    
    time_axis = np.arange(len(ir_filtered)) / FS
    fig, axes = plt.subplots(4, 1, figsize=(12, SUBPLOT_HEIGHT * 4))
    plt.subplots_adjust(hspace=0.6)
    
    axes[0].plot(time_axis, ir_filtered, color='lightgray', label='Input', linewidth=1.5)
    axes[0].plot(time_axis, ir_smoothed, color='black', label='SG Out', linewidth=1.5)
    axes[0].set_title(f'STEP 6: IR SG (Enabled={SG_ENABLE}) - {base_name}')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_axis, red_filtered, color='mistyrose', label='Input', linewidth=1.5)
    axes[1].plot(time_axis, red_smoothed, color='red', label='SG Out', linewidth=1.5)
    axes[1].set_title('STEP 6: RED SG')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    xf_ir_in, yf_ir_in = compute_fft(ir_filtered, FS)
    xf_ir_out, yf_ir_out = compute_fft(ir_smoothed, FS)
    axes[2].plot(xf_ir_in, yf_ir_in, color='lightgray', label='In', linewidth=1.5)
    axes[2].plot(xf_ir_out, yf_ir_out, color='black', label='Out', linewidth=1.5)
    axes[2].set_xlim(0, 20); axes[2].set_title('IR FFT'); axes[2].legend(); axes[2].grid(True, alpha=0.3)
    
    xf_red_in, yf_red_in = compute_fft(red_filtered, FS)
    xf_red_out, yf_red_out = compute_fft(red_smoothed, FS)
    axes[3].plot(xf_red_in, yf_red_in, color='mistyrose', label='In', linewidth=1.5)
    axes[3].plot(xf_red_out, yf_red_out, color='red', label='Out', linewidth=1.5)
    axes[3].set_xlim(0, 20); axes[3].set_title('RED FFT'); axes[3].set_xlabel('Freq (Hz)')
    axes[3].legend(); axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_close(fig, save_path, f"{base_name}_06_SG_Smoothed.png")
    
    # Bonus: Post-SG Spectrum plot
    fig2 = plt.figure(figsize=(12, 6))
    sig_red = red_smoothed - np.mean(red_smoothed)
    sig_ir = ir_smoothed - np.mean(ir_smoothed)
    N = len(sig_red)
    window = np.hanning(N)
    freqs = np.fft.rfftfreq(N, d=1/FS)
    mag_red = np.maximum(np.abs(np.fft.rfft(sig_red * window)), 1e-12)
    mag_ir = np.maximum(np.abs(np.fft.rfft(sig_ir * window)), 1e-12)
    
    plt.semilogy(freqs, mag_ir, label='IR (SG)')
    plt.semilogy(freqs, mag_red, label='RED (SG)')
    plt.axvspan(0.4, 4.0, alpha=0.15, label='HR Band')
    plt.title(f"Frequency Spectrum After SG - {base_name}")
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Magnitude")
    plt.xlim(0, 20); plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    save_and_close(fig2, save_path, f"{base_name}_06b_SG_Spectrum.png")
    
    return ir_smoothed, red_smoothed

# ------------------------------------------
# STEP 7: HIGH-PASS FILTER
# ------------------------------------------
def step7_highpass(ir_smoothed, red_smoothed, save_path, base_name):
    if HP_ENABLE:
        nyquist = 0.5 * FS
        b, a = signal.butter(HP_ORDER, HP_CUTOFF / nyquist, btype='high', analog=False)
        ir_hpf = signal.filtfilt(b, a, ir_smoothed)
        red_hpf = signal.filtfilt(b, a, red_smoothed)
    else:
        ir_hpf = ir_smoothed.copy()
        red_hpf = red_smoothed.copy()
    
    t = np.arange(len(ir_smoothed)) / FS
    LOW_FREQ_ZOOM_MAX = 5.0
    
    fig, axes = plt.subplots(4, 1, figsize=(12, 16))
    
    # IR Time
    ax1 = axes[0]
    ax1.plot(t, ir_smoothed, color='gray', linestyle='--', alpha=0.45, linewidth=1.6)
    ax1.set_ylabel("Input (DC)", color='gray')
    ax1.set_title(f"STEP 7: IR HP Filter - {base_name}")
    ax1.grid(True, alpha=0.25)
    ax1b = ax1.twinx()
    ax1b.plot(t, ir_hpf, color='black', linewidth=1.5)
    ax1b.set_ylabel("Output (AC)", color='black')
    
    # RED Time
    ax2 = axes[1]
    ax2.plot(t, red_smoothed, color='#ff8a80', linestyle='--', alpha=0.55, linewidth=1.6)
    ax2.set_ylabel("Input (DC)", color='#ff6f61')
    ax2.set_title("STEP 7: RED HP Filter")
    ax2.grid(True, alpha=0.25)
    ax2b = ax2.twinx()
    ax2b.plot(t, red_hpf, color='red', linewidth=1.5)
    ax2b.set_ylabel("Output (AC)", color='red')
    ax2.set_xlabel("Time (s)")
    
    # FFT IR
    def fft_mag(x, fs):
        x = np.asarray(x) - np.mean(x)
        N = len(x); w = np.hanning(N)
        f = np.fft.rfftfreq(N, 1/fs)
        return f, np.abs(np.fft.rfft(x * w))
    
    f_ir_in, m_ir_in = fft_mag(ir_smoothed, FS)
    f_ir_out, m_ir_out = fft_mag(ir_hpf, FS)
    mask_ir = f_ir_in <= LOW_FREQ_ZOOM_MAX
    axes[2].plot(f_ir_in[mask_ir], m_ir_in[mask_ir], color='lightgray', linewidth=2, label='In')
    axes[2].plot(f_ir_out[mask_ir], m_ir_out[mask_ir], color='purple', linewidth=1.5, label='Out')
    axes[2].axvline(HP_CUTOFF, color='green', linestyle='--', label=f'Cutoff {HP_CUTOFF}Hz')
    axes[2].set_title("IR Freq Response (Zoom)")
    axes[2].set_xlim(0, LOW_FREQ_ZOOM_MAX); axes[2].grid(True, alpha=0.25); axes[2].legend()
    
    f_red_in, m_red_in = fft_mag(red_smoothed, FS)
    f_red_out, m_red_out = fft_mag(red_hpf, FS)
    mask_red = f_red_in <= LOW_FREQ_ZOOM_MAX
    axes[3].plot(f_red_in[mask_red], m_red_in[mask_red], color='#f28b82', linewidth=2, label='In')
    axes[3].plot(f_red_out[mask_red], m_red_out[mask_red], color='orange', linewidth=1.5, label='Out')
    axes[3].axvline(HP_CUTOFF, color='green', linestyle='--', label=f'Cutoff {HP_CUTOFF}Hz')
    axes[3].set_title("RED Freq Response (Zoom)")
    axes[3].set_xlim(0, LOW_FREQ_ZOOM_MAX); axes[3].set_xlabel("Freq (Hz)")
    axes[3].grid(True, alpha=0.25); axes[3].legend()
    
    plt.tight_layout()
    save_and_close(fig, save_path, f"{base_name}_07_HighPass.png")
    
    return ir_hpf, red_hpf

# ------------------------------------------
# STEP 8: NORMALIZATION
# ------------------------------------------
def step8_normalize(ir_hpf, red_hpf, save_path, base_name):
    ir_norm = apply_normalization(ir_hpf, selection=NORM_SELECTION)
    red_norm = apply_normalization(red_hpf, selection=NORM_SELECTION)
    norm_type = "MinMax (0-1)" if NORM_SELECTION == 1 else "Z-Score"
    
    time_axis = np.arange(len(ir_hpf)) / FS
    fig, axes = plt.subplots(2, 1, figsize=(12, SUBPLOT_HEIGHT * 2), sharex=True)
    plt.subplots_adjust(hspace=0.4)
    
    axes[0].plot(time_axis, ir_norm, color='black', linewidth=1.5, label='IR')
    axes[0].set_title(f'STEP 8: IR Normalized ({norm_type}) - {base_name}')
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_axis, red_norm, color='red', linewidth=1.5, label='RED')
    axes[1].set_title(f'STEP 8: RED Normalized ({norm_type})')
    axes[1].set_xlabel('Time (s)'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_and_close(fig, save_path, f"{base_name}_08_Normalized.png")
    
    return ir_norm, red_norm

print("✅ Core processing functions (Steps 2-8) loaded.")



#------------------------------------------------------------------------------------------------------------------



# ==========================================
# ⚙️ STEP 9, 10, 12: QUALITY & FEATURE FUNCTIONS
# ==========================================

# ------------------------------------------
# STEP 9: ADVANCED SIGNAL QUALITY CHECK
# ------------------------------------------
def step9_quality_check(clean_sig, raw_sig_for_dc, fs):
    """Returns SQI metrics for a single channel."""
    sk = float(skew(clean_sig))
    ku = float(kurtosis(clean_sig))
    
    dc_val = float(np.mean(raw_sig_for_dc))
    ac_amp = float(np.max(clean_sig) - np.min(clean_sig))
    pi = (ac_amp / dc_val) * 100 if dc_val != 0 else 0.0
    
    zcr = float(calculate_zcr(clean_sig, fs))
    snr = float(calculate_snr(clean_sig, fs))
    
    def status(val, lo, hi):
        return "PASS" if lo <= val <= hi else "FAIL"
    
    return {
        "skewness": sk,
        "skewness_status": status(sk, SQI_LIMITS['SKEWNESS_MIN'], SQI_LIMITS['SKEWNESS_MAX']),
        "kurtosis": ku,
        "kurtosis_status": status(ku, SQI_LIMITS['KURTOSIS_MIN'], SQI_LIMITS['KURTOSIS_MAX']),
        "perfusion_index_pct": pi,
        "perfusion_index_status": status(pi, SQI_LIMITS['PI_MIN'], SQI_LIMITS['PI_MAX']),
        "zcr_hz": zcr,
        "zcr_status": status(zcr, SQI_LIMITS['ZCR_MIN'], SQI_LIMITS['ZCR_MAX']),
        "snr_db": snr,
        "snr_status": status(snr, SQI_LIMITS['SNR_MIN_DB'], SQI_LIMITS['SNR_MAX_DB'])
    }

# ------------------------------------------
# STEP 10: FULL PIPELINE DIAGNOSTIC
# ------------------------------------------
def calculate_sqi_metrics(sig_ac, sig_raw_for_dc, fs):
    """Calculates raw SQI for a stage (used by pipeline diagnostic)."""
    sk = float(skew(sig_ac))
    ku = float(kurtosis(sig_ac))
    
    centered = sig_ac - np.mean(sig_ac)
    zcr_count = ((centered[:-1] * centered[1:]) < 0).sum()
    zcr = float(zcr_count / (len(sig_ac) / fs))
    
    f, Pxx = welch(sig_ac, fs=fs, nperseg=min(len(sig_ac), 1024))
    band_mask = (f >= 0.5) & (f <= 4.0)
    signal_power = np.sum(Pxx[band_mask])
    noise_power = np.sum(Pxx) - signal_power
    snr = float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else 0.0
    
    ac_amp = float(np.max(sig_ac) - np.min(sig_ac))
    dc_val = float(np.mean(sig_raw_for_dc))
    pi = (ac_amp / dc_val) * 100 if dc_val != 0 else 0.0
    
    return {"skew": sk, "kurt": ku, "snr_db": snr, "zcr_hz": zcr, "pi_pct": pi}

def step10_pipeline_diagnostic(stages_dict, raw_ir, raw_red, fs):
    """
    Generates full pipeline diagnostic across all stages.
    stages_dict: {stage_name: (ir_signal, red_signal)}
    """
    diagnostic = {}
    for stage_name, (ir_sig, red_sig) in stages_dict.items():
        ir_metrics = calculate_sqi_metrics(ir_sig, raw_ir, fs)
        red_metrics = calculate_sqi_metrics(red_sig, raw_red, fs)
        
        ir_pi = ir_metrics["pi_pct"]
        red_pi = red_metrics["pi_pct"]
        r_val = (red_pi / ir_pi) if ir_pi != 0 else None
        
        diagnostic[stage_name] = {
            "IR": ir_metrics,
            "RED": red_metrics,
            "R_ratio": r_val
        }
    return diagnostic

# ------------------------------------------
# STEP 11 (FEATURES PART): PPG FEATURE EXTRACTION
# ------------------------------------------
def extract_glucose_features(clean_ppg, raw_ppg, fs):
    """Extracts 9 key features (7 glucose-shape + 2 SQI-style)."""
    peaks, _ = find_peaks(clean_ppg, distance=int(0.4 * fs), prominence=0.1)
    
    if len(peaks) > 1:
        avg_peak_interval = np.mean(np.diff(peaks)) / fs
        hr = 60.0 / avg_peak_interval if avg_peak_interval > 0 else 0
    else:
        hr = 0
    
    if len(peaks) > 0:
        widths = []
        for pk in peaks:
            half_max = clean_ppg[pk] * 0.5
            left, right = pk, pk
            while left > 0 and clean_ppg[left] > half_max:
                left -= 1
            while right < len(clean_ppg) - 1 and clean_ppg[right] > half_max:
                right += 1
            widths.append((right - left) / fs)
        avg_width = float(np.mean(widths)) if widths else 0
        sys_amp = float(np.mean(clean_ppg[peaks]))
    else:
        avg_width = 0
        sys_amp = 0
    
    if hasattr(np, "trapezoid"):
        area = float(np.trapezoid(clean_ppg))
    else:
        area = float(np.trapz(clean_ppg))
    
    sk = float(skew(clean_ppg))
    ku = float(kurtosis(clean_ppg))
    
    ac_global = float(np.max(raw_ppg) - np.min(raw_ppg))
    dc_global = float(np.mean(raw_ppg))
    pi = (ac_global / dc_global) * 100 if dc_global > 0 else 0
    
    centered = clean_ppg - np.mean(clean_ppg)
    zcr = float(((centered[:-1] * centered[1:]) < 0).sum() / (len(clean_ppg) / fs))
    
    f, Pxx = welch(clean_ppg, fs=fs, nperseg=min(len(clean_ppg), 1024))
    band_mask = (f >= 0.5) & (f <= 4.0)
    signal_power = np.sum(Pxx[band_mask])
    total_power = np.sum(Pxx)
    noise_power = total_power - signal_power
    snr = float(10 * np.log10(signal_power / noise_power)) if noise_power > 0 else 100.0
    
    return {
        'hr': hr, 'width': avg_width, 'sys_amp': sys_amp, 'area': area,
        'skew': sk, 'kurt': ku, 'pi': pi, 'zcr': zcr, 'snr': snr
    }

# ------------------------------------------
# STEP 12: GOLDEN STANDARD (Ensemble-based features)
# ------------------------------------------
def extract_features_from_ensemble_wave(beat_wave, global_hr, raw_signal_for_dc, fs_eff):
    """Extracts morphology features from the ensemble waveform."""
    if beat_wave is None:
        return {}
    
    peak_idx = int(np.argmax(beat_wave))
    half_max = beat_wave[peak_idx] * 0.5
    
    left, right = peak_idx, peak_idx
    while left > 0 and beat_wave[left] > half_max:
        left -= 1
    while right < len(beat_wave) - 1 and beat_wave[right] > half_max:
        right += 1
    
    pulse_width = (right - left) / fs_eff
    
    if hasattr(np, 'trapezoid'):
        area = np.trapezoid(beat_wave)
    else:
        area = np.trapz(beat_wave)
    
    sys_amp = np.max(beat_wave) - np.min(beat_wave)
    sk = skew(beat_wave)
    ku = kurtosis(beat_wave)
    
    ac_global = np.max(raw_signal_for_dc) - np.min(raw_signal_for_dc)
    dc_global = np.mean(raw_signal_for_dc)
    pi = (ac_global / dc_global) * 100 if dc_global > 0 else 0
    
    return {
        'HR_Global': float(global_hr),
        'Width_Avg': float(pulse_width),
        'Area_Avg': float(area),
        'Amp_Avg': float(sys_amp),
        'Skew_Avg': float(sk),
        'Kurt_Avg': float(ku),
        'PI_Global': float(pi)
    }

print("✅ Quality & feature extraction functions (Steps 9, 10, 12) loaded.")


#------------------------------------------------------------------------------------------------------------------


# ==========================================
# ⚙️ STEP 11: ENSEMBLE DETECTION + SDPPG CORE
# ==========================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter

# -------------------------------------------------
# SMALL HELPERS
# -------------------------------------------------
def save_if_requested(fig, save_file=None):
    """Save figure if path is given, otherwise show it."""
    if save_file:
        folder, name = os.path.split(save_file)
        save_and_close(fig, folder, name)
        return save_file
    else:
        plt.show()
        plt.close(fig)
        return None

def summarize_reason_counts(items, key_name="reason_key"):
    """Count rejection reasons."""
    out = {}
    for item in items:
        k = item.get(key_name, "unknown")
        out[k] = out.get(k, 0) + 1
    return out

# -------------------------------------------------
# TEXT HELPERS
# -------------------------------------------------
def explain_candidate_reason(reason_key):
    reason_map = {
        "edge_candidate_peak_start": "Candidate peak too close to start.",
        "edge_candidate_peak_end": "Candidate peak too close to end.",
        "no_left_valley_found": "No valid valley before candidate peak.",
        "valley_too_close_to_peak": "Valley too close to candidate peak.",
        "foot_missing": "Foot could not be found.",
        "invalid_ordering": "Invalid candidate ordering.",
        "pulse_amplitude_non_positive": "Pulse amplitude is non-positive.",
        "foot_to_peak_too_short": "Foot-to-peak too short.",
        "foot_to_peak_too_long": "Foot-to-peak too long.",
        "foot_too_far_from_valley": "Foot too far from valley.",
        "foot_too_high_on_pulse": "Foot too high on pulse.",
        "vpg_not_near_zero_at_foot": "VPG not near zero at foot.",
        "peak_foot_amplitude_too_small": "Foot-to-peak amplitude too small.",
    }
    return reason_map.get(reason_key, reason_key)

def explain_pulse_reason(reason_key):
    reason_map = {
        "no_peak_between_two_feet": "No peak found between two feet.",
        "no_main_peak_after_foot": "No valid main peak after foot.",
        "invalid_ordering": "Pulse order invalid.",
        "beat_duration_out_of_range": "Beat duration out of range.",
        "foot_to_peak_distance_out_of_range": "Foot-to-peak distance out of range.",
        "peak_foot_amplitude_too_small": "Peak-foot amplitude too small.",
        "upstroke_too_weak": "Upstroke too weak.",
        "beat_too_short_after_segmentation": "Segment too short.",
        "incomplete_at_signal_start": "Pulse incomplete at start.",
        "incomplete_at_signal_end": "Pulse incomplete at end.",
        "next_foot_too_close_to_signal_end": "Next foot too close to end.",
    }
    return reason_map.get(reason_key, reason_key)

# -------------------------------------------------
# REJECTION LOGS
# -------------------------------------------------
def log_candidate_rejection(rejected_candidates, reason_key, candidate_num=None, **kwargs):
    item = {
        "candidate_num": candidate_num,
        "reason_key": reason_key,
        "reason_text": explain_candidate_reason(reason_key)
    }
    item.update(kwargs)
    rejected_candidates.append(item)

    if VERBOSE_REJECTION:
        msg = f"❌ Candidate {candidate_num} -> {item['reason_text']}" if candidate_num is not None else f"❌ Candidate -> {item['reason_text']}"
        print(msg)

def log_pulse_rejection(rejected_pulses, reason_key, pulse_num=None, **kwargs):
    item = {
        "pulse_num": pulse_num,
        "reason_key": reason_key,
        "reason_text": explain_pulse_reason(reason_key)
    }
    item.update(kwargs)
    rejected_pulses.append(item)

    if VERBOSE_REJECTION:
        msg = f"❌ Pulse {pulse_num} -> {item['reason_text']}" if pulse_num is not None else f"❌ Pulse -> {item['reason_text']}"
        print(msg)

# -------------------------------------------------
# SMOOTHED DERIVATIVES
# -------------------------------------------------
def compute_smoothed_derivatives(signal_data, fs):
    signal_data = np.asarray(signal_data)

    win_ppg = max(7, int(0.05 * fs))
    if win_ppg % 2 == 0:
        win_ppg += 1

    smooth_sig = safe_savgol(signal_data, win_ppg, poly=3)

    vpg = np.gradient(smooth_sig) * fs
    win_vpg = max(5, int(0.03 * fs))
    if win_vpg % 2 == 0:
        win_vpg += 1
    vpg = safe_savgol(vpg, win_vpg, poly=2)

    sdppg = np.gradient(vpg) * fs
    win_sd = max(5, int(0.03 * fs))
    if win_sd % 2 == 0:
        win_sd += 1
    sdppg = safe_savgol(sdppg, win_sd, poly=2)

    return smooth_sig, vpg, sdppg

# -------------------------------------------------
# FOOT / PEAK HELPERS
# -------------------------------------------------
def find_last_zero_crossing_before_max(vpg_seg, idx_seg):
    candidates = []
    for i in range(len(vpg_seg) - 1):
        y1 = vpg_seg[i]
        y2 = vpg_seg[i + 1]
        if y1 < 0 and y2 >= 0:
            if abs(y1) <= abs(y2):
                candidates.append(int(idx_seg[i]))
            else:
                candidates.append(int(idx_seg[i + 1]))

    if len(candidates) == 0:
        return None

    return int(candidates[-1])

def refine_foot_zero_crossing(valley_idx, peak_idx, smooth_sig, vpg, sdppg, fs):
    if peak_idx <= valley_idx + 3:
        return int(valley_idx)

    seg_idx = np.arange(valley_idx, peak_idx + 1)
    if len(seg_idx) < 6:
        return int(valley_idx)

    seg_vpg = vpg[seg_idx]
    i_vpgmax_local = int(np.argmax(seg_vpg))
    i_vpgmax = int(seg_idx[i_vpgmax_local])

    if i_vpgmax <= valley_idx + 1:
        return int(valley_idx)

    search_idx = np.arange(valley_idx, i_vpgmax + 1)
    search_vpg = vpg[search_idx]
    search_sdppg = sdppg[search_idx]

    if len(search_idx) < 3:
        return int(valley_idx)

    foot_zero = find_last_zero_crossing_before_max(search_vpg, search_idx)

    if foot_zero is None:
        early_end = max(3, int(0.6 * len(search_idx)))
        early_idx = search_idx[:early_end]
        early_vpg = search_vpg[:early_end]
        foot_zero = int(early_idx[np.argmin(np.abs(early_vpg))])

    foot = int(foot_zero)

    sdppg_peak = int(search_idx[np.argmax(search_sdppg)])
    if foot > sdppg_peak:
        left_part = search_idx[search_idx <= sdppg_peak]
        if len(left_part) > 0:
            foot = int(left_part[np.argmin(np.abs(vpg[left_part]))])

    foot = max(valley_idx, min(foot, i_vpgmax - 1))
    return int(foot)

def select_main_peak_after_foot(candidate_peaks, foot_idx, smooth_sig, fs):
    if len(candidate_peaks) == 0:
        return None

    min_idx = foot_idx + int(MAIN_PEAK_MIN_DELAY_SEC * fs)
    max_idx = foot_idx + int(MAIN_PEAK_SEARCH_WINDOW_SEC * fs)

    valid_peaks = candidate_peaks[(candidate_peaks >= min_idx) & (candidate_peaks <= max_idx)]
    if len(valid_peaks) == 0:
        return None

    return int(valid_peaks[0])

def is_valid_peak_foot_pair(foot_idx, peak_idx, valley_idx, smooth_sig, fs):
    if foot_idx is None:
        return False, "foot_missing"

    if not (0 <= valley_idx <= foot_idx < peak_idx < len(smooth_sig)):
        return False, "invalid_ordering"

    foot_to_peak_sec = (peak_idx - foot_idx) / fs
    if foot_to_peak_sec < MIN_FOOT_TO_PEAK_SEC:
        return False, "foot_to_peak_too_short"
    if foot_to_peak_sec > MAX_FOOT_TO_PEAK_SEC:
        return False, "foot_to_peak_too_long"

    amp = smooth_sig[peak_idx] - smooth_sig[foot_idx]
    if amp <= max(0.01, 0.02 * np.std(smooth_sig)):
        return False, "peak_foot_amplitude_too_small"

    return True, "ok"

def is_ambiguous_foot_peak_pair(foot_idx, peak_idx, valley_idx, smooth_sig, vpg, fs):
    if foot_idx is None:
        return True, "foot_missing"

    if not (0 <= valley_idx <= foot_idx < peak_idx < len(smooth_sig)):
        return True, "invalid_ordering"

    foot_to_peak_sec = (peak_idx - foot_idx) / fs
    valley_to_foot_sec = (foot_idx - valley_idx) / fs

    peak_amp = smooth_sig[peak_idx]
    foot_amp = smooth_sig[foot_idx]
    valley_amp = smooth_sig[valley_idx]

    pulse_amp = peak_amp - valley_amp
    if pulse_amp <= 0:
        return True, "pulse_amplitude_non_positive"

    if foot_to_peak_sec < MIN_FOOT_TO_PEAK_SEC:
        return True, "foot_to_peak_too_short"
    if foot_to_peak_sec > MAX_FOOT_TO_PEAK_SEC:
        return True, "foot_to_peak_too_long"
    if valley_to_foot_sec > MAX_VALLEY_TO_FOOT_SEC:
        return True, "foot_too_far_from_valley"

    rel_height = (foot_amp - valley_amp) / pulse_amp
    if rel_height > MAX_FOOT_REL_HEIGHT:
        return True, "foot_too_high_on_pulse"

    if abs(vpg[foot_idx]) > MAX_ABS_VPG_AT_FOOT:
        return True, "vpg_not_near_zero_at_foot"

    return False, "ok"

def is_valid_beat(foot_idx, peak_idx, next_foot_idx, smooth_sig, vpg, fs):
    if not (0 <= foot_idx < peak_idx < next_foot_idx < len(smooth_sig)):
        return False, "invalid_ordering"

    beat_duration_sec = (next_foot_idx - foot_idx) / fs
    foot_to_peak_sec = (peak_idx - foot_idx) / fs
    amp = smooth_sig[peak_idx] - smooth_sig[foot_idx]
    upstroke_max = np.max(vpg[foot_idx:peak_idx + 1]) if peak_idx > foot_idx else 0.0

    if not (MIN_BEAT_DURATION_SEC <= beat_duration_sec <= MAX_BEAT_DURATION_SEC):
        return False, "beat_duration_out_of_range"
    if not (MIN_FOOT_TO_PEAK_SEC <= foot_to_peak_sec <= MAX_FOOT_TO_PEAK_SEC):
        return False, "foot_to_peak_distance_out_of_range"
    if amp <= max(0.01, 0.02 * np.std(smooth_sig)):
        return False, "peak_foot_amplitude_too_small"
    if upstroke_max <= max(0.05, 0.05 * np.std(vpg)):
        return False, "upstroke_too_weak"

    return True, "ok"

def is_incomplete_edge_pulse(foot_idx, peak_idx, next_foot_idx, signal_len, fs):
    start_margin = int(START_INCOMPLETE_MARGIN_SEC * fs)
    end_margin = int(END_INCOMPLETE_MARGIN_SEC * fs)

    if foot_idx < start_margin:
        return True, "incomplete_at_signal_start"
    if peak_idx > signal_len - end_margin:
        return True, "incomplete_at_signal_end"
    if next_foot_idx > signal_len - end_margin:
        return True, "next_foot_too_close_to_signal_end"

    return False, "ok"

# -------------------------------------------------
# CANDIDATE BUILDING
# -------------------------------------------------
def build_candidate_foot_peak_pairs(smooth_sig, vpg, sdppg, peaks, valleys, fs):
    candidate_pairs = []
    rejected_candidates = []
    n = len(smooth_sig)
    candidate_num = 0

    edge_margin_samples = int(EDGE_EXCLUSION_SEC * fs)
    min_ft_pk_samples = int(MIN_FOOT_TO_PEAK_SEC * fs)

    for p in peaks:
        p = int(p)
        candidate_num += 1

        if p < edge_margin_samples:
            log_candidate_rejection(
                rejected_candidates, "edge_candidate_peak_start",
                candidate_num=candidate_num, peak=p,
                actual_value=f"peak_idx={p}",
                accepted_range=f">= {edge_margin_samples} samples"
            )
            continue
        if p > n - edge_margin_samples:
            log_candidate_rejection(
                rejected_candidates, "edge_candidate_peak_end",
                candidate_num=candidate_num, peak=p,
                actual_value=f"peak_idx={p}",
                accepted_range=f"<= {n - edge_margin_samples} samples"
            )
            continue

        left_valleys = valleys[valleys < p]
        if len(left_valleys) == 0:
            log_candidate_rejection(
                rejected_candidates, "no_left_valley_found",
                candidate_num=candidate_num, peak=p,
                actual_value="0 valleys before peak",
                accepted_range=">= 1 valley"
            )
            continue

        valley_idx = int(left_valleys[-1])

        if (p - valley_idx) < min_ft_pk_samples:
            log_candidate_rejection(
                rejected_candidates, "valley_too_close_to_peak",
                candidate_num=candidate_num, peak=p, valley=valley_idx,
                actual_value=f"{(p - valley_idx)/fs:.3f}s",
                accepted_range=f">= {MIN_FOOT_TO_PEAK_SEC:.3f}s"
            )
            continue

        refined_foot = refine_foot_zero_crossing(
            valley_idx=valley_idx, peak_idx=p,
            smooth_sig=smooth_sig, vpg=vpg, sdppg=sdppg, fs=fs
        )

        ambiguous, ambiguity_reason = is_ambiguous_foot_peak_pair(
            refined_foot, p, valley_idx, smooth_sig, vpg, fs
        )
        if ambiguous:
            # Compute actual + accepted-range per specific reason
            av, ar = "n/a", "n/a"
            if refined_foot is not None:
                ftp = (p - refined_foot) / fs
                vtf = (refined_foot - valley_idx) / fs
                pulse_amp = float(smooth_sig[p] - smooth_sig[valley_idx])
                if ambiguity_reason == "foot_to_peak_too_short":
                    av, ar = f"{ftp:.3f}s", f">= {MIN_FOOT_TO_PEAK_SEC:.3f}s"
                elif ambiguity_reason == "foot_to_peak_too_long":
                    av, ar = f"{ftp:.3f}s", f"<= {MAX_FOOT_TO_PEAK_SEC:.3f}s"
                elif ambiguity_reason == "foot_too_far_from_valley":
                    av, ar = f"{vtf:.3f}s", f"<= {MAX_VALLEY_TO_FOOT_SEC:.3f}s"
                elif ambiguity_reason == "foot_too_high_on_pulse":
                    rel = (smooth_sig[refined_foot] - smooth_sig[valley_idx]) / pulse_amp if pulse_amp > 0 else 0
                    av, ar = f"{rel:.3f}", f"<= {MAX_FOOT_REL_HEIGHT:.3f}"
                elif ambiguity_reason == "vpg_not_near_zero_at_foot":
                    av, ar = f"{float(vpg[refined_foot]):.3f}", f"|x| <= {MAX_ABS_VPG_AT_FOOT:.3f}"
                elif ambiguity_reason == "pulse_amplitude_non_positive":
                    av, ar = f"{pulse_amp:.4f}", "> 0"
            log_candidate_rejection(
                rejected_candidates, ambiguity_reason,
                candidate_num=candidate_num, peak=p, foot=refined_foot,
                valley=valley_idx,
                vpg_at_foot=float(vpg[refined_foot]) if refined_foot is not None else None,
                actual_value=av, accepted_range=ar
            )
            continue

        valid_pair, invalid_reason = is_valid_peak_foot_pair(
            refined_foot, p, valley_idx, smooth_sig, fs
        )
        if not valid_pair:
            av, ar = "n/a", "n/a"
            if refined_foot is not None:
                ftp = (p - refined_foot) / fs
                if invalid_reason == "foot_to_peak_too_short":
                    av, ar = f"{ftp:.3f}s", f">= {MIN_FOOT_TO_PEAK_SEC:.3f}s"
                elif invalid_reason == "foot_to_peak_too_long":
                    av, ar = f"{ftp:.3f}s", f"<= {MAX_FOOT_TO_PEAK_SEC:.3f}s"
                elif invalid_reason == "peak_foot_amplitude_too_small":
                    amp = float(smooth_sig[p] - smooth_sig[refined_foot])
                    thr = max(0.01, 0.02 * float(np.std(smooth_sig)))
                    av, ar = f"{amp:.4f}", f"> {thr:.4f}"
            log_candidate_rejection(
                rejected_candidates, invalid_reason,
                candidate_num=candidate_num, peak=p, foot=refined_foot, valley=valley_idx,
                actual_value=av, accepted_range=ar
            )
            continue

        candidate_pairs.append({
            "candidate_num": candidate_num,
            "peak": p,
            "foot": refined_foot,
            "valley": valley_idx
        })

    return candidate_pairs, rejected_candidates

# -------------------------------------------------
# MAIN BEAT DETECTOR
# -------------------------------------------------
def detect_beats_foot_to_foot(signal_data, fs):
    smooth_sig, vpg, sdppg = compute_smoothed_derivatives(signal_data, fs)

    peaks, _ = find_peaks(
        smooth_sig,
        distance=int(PEAK_MIN_DISTANCE_SEC * fs),
        prominence=max(0.015, np.std(smooth_sig) * PEAK_PROM_FACTOR)
    )

    valleys, _ = find_peaks(
        -smooth_sig,
        distance=int(VALLEY_MIN_DISTANCE_SEC * fs),
        prominence=max(0.01, np.std(smooth_sig) * VALLEY_PROM_FACTOR)
    )

    candidate_pairs, rejected_candidates = build_candidate_foot_peak_pairs(
        smooth_sig, vpg, sdppg, peaks, valleys, fs
    )

    if len(candidate_pairs) == 0:
        return (
            peaks, np.array([]), [], [], [],
            smooth_sig, vpg, sdppg, valleys,
            rejected_candidates, [], []
        )

    feet = []
    feet_seen = set()

    for item in candidate_pairs:
        f = int(item["foot"])
        if f not in feet_seen:
            feet.append(f)
            feet_seen.add(f)

    feet = np.array(sorted(feet), dtype=int)

    segmented_pulses_all = []
    for i in range(len(feet) - 1):
        f1 = int(feet[i])
        f2 = int(feet[i + 1])
        segmented_pulses_all.append({
            "pulse_num": i + 1,
            "foot": f1,
            "next_foot": f2,
            "mid_idx": int((f1 + f2) // 2),
            "status": "unknown",
            "reason_text": ""
        })

    beats = []
    valid_feet = []
    beat_info = []
    rejected_pulses = []

    sig_len = len(smooth_sig)
    end_margin_samples = int(END_INCOMPLETE_MARGIN_SEC * fs)
    start_margin_samples = int(START_INCOMPLETE_MARGIN_SEC * fs)
    search_lo_s = MAIN_PEAK_MIN_DELAY_SEC
    search_hi_s = MAIN_PEAK_SEARCH_WINDOW_SEC

    for seg in segmented_pulses_all:
        pulse_num = seg["pulse_num"]
        f1 = seg["foot"]
        f2 = seg["next_foot"]

        candidate_peaks = peaks[(peaks > f1) & (peaks < f2)]
        if len(candidate_peaks) == 0:
            seg["status"] = "rejected"
            seg["reason_text"] = explain_pulse_reason("no_peak_between_two_feet")
            log_pulse_rejection(
                rejected_pulses, "no_peak_between_two_feet",
                pulse_num=pulse_num, foot=f1, next_foot=f2,
                actual_value="0 peaks",
                accepted_range=">= 1 peak between feet"
            )
            continue

        p = select_main_peak_after_foot(candidate_peaks, f1, smooth_sig, fs)
        if p is None:
            seg["status"] = "rejected"
            seg["reason_text"] = explain_pulse_reason("no_main_peak_after_foot")
            log_pulse_rejection(
                rejected_pulses, "no_main_peak_after_foot",
                pulse_num=pulse_num, foot=f1, next_foot=f2,
                actual_value=f"no peak in foot+[{search_lo_s:.3f}s,{search_hi_s:.3f}s]",
                accepted_range=f"peak within foot+[{search_lo_s:.3f}s,{search_hi_s:.3f}s]"
            )
            continue

        edge_incomplete, edge_reason = is_incomplete_edge_pulse(
            foot_idx=f1, peak_idx=p, next_foot_idx=f2,
            signal_len=sig_len, fs=fs
        )
        if edge_incomplete:
            seg["status"] = "rejected"
            seg["reason_text"] = explain_pulse_reason(edge_reason)
            av, ar = "n/a", "n/a"
            if edge_reason == "incomplete_at_signal_start":
                av, ar = f"foot_idx={f1}", f">= {start_margin_samples} samples"
            elif edge_reason == "incomplete_at_signal_end":
                av, ar = f"peak_idx={p}", f"<= {sig_len - end_margin_samples} samples"
            elif edge_reason == "next_foot_too_close_to_signal_end":
                av, ar = f"next_foot_idx={f2}", f"<= {sig_len - end_margin_samples} samples"
            log_pulse_rejection(
                rejected_pulses, edge_reason,
                pulse_num=pulse_num, foot=f1, peak=p, next_foot=f2,
                actual_value=av, accepted_range=ar
            )
            continue

        valid_beat, beat_reason = is_valid_beat(f1, p, f2, smooth_sig, vpg, fs)
        if not valid_beat:
            seg["status"] = "rejected"
            seg["reason_text"] = explain_pulse_reason(beat_reason)
            av, ar = "n/a", "n/a"
            beat_dur = (f2 - f1) / fs
            ftp_dur = (p - f1) / fs
            if beat_reason == "beat_duration_out_of_range":
                av, ar = f"{beat_dur:.3f}s", f"[{MIN_BEAT_DURATION_SEC:.2f}, {MAX_BEAT_DURATION_SEC:.2f}]s"
            elif beat_reason == "foot_to_peak_distance_out_of_range":
                av, ar = f"{ftp_dur:.3f}s", f"[{MIN_FOOT_TO_PEAK_SEC:.2f}, {MAX_FOOT_TO_PEAK_SEC:.2f}]s"
            elif beat_reason == "peak_foot_amplitude_too_small":
                amp = float(smooth_sig[p] - smooth_sig[f1])
                thr = max(0.01, 0.02 * float(np.std(smooth_sig)))
                av, ar = f"{amp:.4f}", f"> {thr:.4f}"
            elif beat_reason == "upstroke_too_weak":
                upmax = float(np.max(vpg[f1:p + 1])) if p > f1 else 0.0
                thr = max(0.05, 0.05 * float(np.std(vpg)))
                av, ar = f"{upmax:.4f}", f"> {thr:.4f}"
            log_pulse_rejection(
                rejected_pulses, beat_reason,
                pulse_num=pulse_num, peak=p, foot=f1, next_foot=f2,
                actual_value=av, accepted_range=ar
            )
            continue

        beat = signal_data[f1:f2 + 1]
        if len(beat) <= 10:
            seg["status"] = "rejected"
            seg["reason_text"] = explain_pulse_reason("beat_too_short_after_segmentation")
            log_pulse_rejection(
                rejected_pulses, "beat_too_short_after_segmentation",
                pulse_num=pulse_num, peak=p, foot=f1, next_foot=f2, beat_len=len(beat),
                actual_value=f"{len(beat)} samples",
                accepted_range="> 10 samples"
            )
            continue

        seg["status"] = "accepted"
        beats.append(beat)
        valid_feet.append((f1, f2))

        beat_info.append({
            "pulse_num": pulse_num,
            "foot": f1,
            "peak": p,
            "next_foot": f2,
            "mid_idx": int((f1 + f2) // 2),
            "beat_duration": float((f2 - f1) / fs),
            "foot_to_peak_sec": float((p - f1) / fs),
            "amplitude": float(smooth_sig[p] - smooth_sig[f1]),
            "vpg_at_foot": float(vpg[f1]),
            "sdppg_at_foot": float(sdppg[f1]),
            "status": "accepted"
        })

    pulse_map = []
    for seg in segmented_pulses_all:
        pulse_map.append({
            "pulse_num": seg["pulse_num"],
            "foot": seg["foot"],
            "next_foot": seg["next_foot"],
            "mid_idx": seg["mid_idx"],
            "status": seg["status"],
            "reason_text": seg["reason_text"]
        })

    return (
        peaks, feet, beats, valid_feet, beat_info,
        smooth_sig, vpg, sdppg, valleys,
        rejected_candidates, rejected_pulses, pulse_map
    )

# -------------------------------------------------
# ENSEMBLE BUILDING
# -------------------------------------------------
def build_ensemble_from_beats(beats, target_len=200):
    if len(beats) == 0:
        return None, None, None

    beats_rs = np.array([resample_1d(b, target_len) for b in beats])
    avg_wave = np.mean(beats_rs, axis=0)
    std_wave = np.std(beats_rs, axis=0)
    return beats_rs, avg_wave, std_wave

def align_beats_by_vpg(beats_rs, fs_eff):
    if beats_rs is None or len(beats_rs) == 0:
        return beats_rs

    vpg_all = np.gradient(beats_rs, axis=1) * fs_eff
    vpg_all = np.array([
        safe_savgol(v, max(5, int(0.03 * fs_eff)), poly=2)
        for v in vpg_all
    ])

    ref_idx = int(np.argmax(np.mean(vpg_all, axis=0)))

    aligned = []
    for i in range(len(beats_rs)):
        vpg_i = vpg_all[i]
        idx_i = int(np.argmax(vpg_i))
        shift = ref_idx - idx_i

        shifted = np.roll(beats_rs[i], shift)

        if shift > 0:
            shifted[:shift] = shifted[shift]
        elif shift < 0:
            shifted[shift:] = shifted[shift - 1]

        aligned.append(shifted)

    return np.array(aligned)

def compute_derivatives(signal_data, fs_eff):
    smooth = safe_savgol(signal_data, max(7, int(0.05 * fs_eff)), poly=3)

    vpg = np.gradient(smooth) * fs_eff
    vpg = safe_savgol(vpg, max(5, int(0.03 * fs_eff)), poly=2)

    sdppg = np.gradient(vpg) * fs_eff
    sdppg = safe_savgol(sdppg, max(5, int(0.03 * fs_eff)), poly=2)

    return vpg, sdppg

def detect_sdppg_abcde(sdppg, fs_eff):
    n = len(sdppg)

    win = max(7, int(0.05 * fs_eff))
    if win % 2 == 0:
        win += 1
    if win >= n:
        win = n - 1 if (n - 1) % 2 == 1 else n - 2
    if win < 5:
        win = 5

    sd_s = safe_savgol(sdppg, win, poly=3)

    pos_peaks, _ = find_peaks(sd_s, distance=max(3, int(0.05 * fs_eff)))
    neg_peaks, _ = find_peaks(-sd_s, distance=max(3, int(0.05 * fs_eff)))

    fid = {"a": None, "b": None, "c": None, "d": None, "e": None}

    if len(pos_peaks) == 0:
        return fid, sd_s

    early_pos = pos_peaks[pos_peaks < int(0.35 * n)]
    if len(early_pos) == 0:
        return fid, sd_s

    a = int(early_pos[np.argmax(sd_s[early_pos])])
    fid["a"] = a

    b_candidates = neg_peaks[neg_peaks > a]
    if len(b_candidates) == 0:
        return fid, sd_s
    b = int(b_candidates[0])
    fid["b"] = b

    c_candidates = pos_peaks[pos_peaks > b]
    if len(c_candidates) == 0:
        return fid, sd_s
    c = int(c_candidates[0])
    fid["c"] = c

    d_candidates = neg_peaks[neg_peaks > c]
    if len(d_candidates) == 0:
        return fid, sd_s
    d = int(d_candidates[0])
    fid["d"] = d

    e_candidates = pos_peaks[pos_peaks > d]
    if len(e_candidates) > 0:
        fid["e"] = int(e_candidates[0])

    return fid, sd_s

# -------------------------------------------------
# PLOTTERS
# -------------------------------------------------
def plot_ensemble_sdppg(signal_data, fs, title, color_avg, min_valid_beats=3,
                        target_len=220, save_file=None, verbose=False):
    """
    Returns:
      - On success: tuple of (avg_wave, vpg, sdppg_s, beats_rs_aligned, n_beats, fid, meta)
      - On rejection: dict with rejection info (status="REJECTED", reason, n_beats, etc.)
    """
    (
        peaks, feet, beats, valid_feet, beat_info,
        smooth_sig, vpg_raw, sdppg_raw, valleys,
        rejected_candidates, rejected_pulses, pulse_map
    ) = detect_beats_foot_to_foot(signal_data, fs)

    if len(beats) < min_valid_beats:
        reason = (f"Not enough valid beats for ensemble: found {len(beats)}, "
                  f"required >= {min_valid_beats}.")
        if verbose:
            print(f"⚠️ {title}: {reason}")
        # Return rejection diagnostic instead of None
        return {
            "status": "REJECTED",
            "reason": reason,
            "beats_found": int(len(beats)),
            "min_required": int(min_valid_beats),
            "rejected_candidates": int(len(rejected_candidates)),
            "rejected_pulses": int(len(rejected_pulses)),
            "segmented_pulses_total": int(len(pulse_map)),
            "rejected_candidates_info": rejected_candidates,
            "rejected_pulses_info": rejected_pulses,
            "pulse_map": pulse_map,
            "smooth_sig": smooth_sig,
            "peaks": peaks,
            "valleys": valleys,
            "feet": feet,
            "rejection_reason_summary": summarize_reason_counts(rejected_pulses, "reason_key")
        }

    beats_rs, _, _ = build_ensemble_from_beats(beats, target_len=target_len)

    beat_durations = [len(b) / fs for b in beats]
    avg_duration = float(np.median(beat_durations))
    fs_eff = target_len / avg_duration

    beats_rs_aligned = align_beats_by_vpg(beats_rs, fs_eff)

    avg_wave = np.mean(beats_rs_aligned, axis=0)
    std_wave = np.std(beats_rs_aligned, axis=0)

    vpg, sdppg = compute_derivatives(avg_wave, fs_eff)
    fid, sdppg_s = detect_sdppg_abcde(sdppg, fs_eff)

    time_axis = np.linspace(0, avg_duration, target_len)

    meta = {
        "title": title,
        "target_len": int(target_len),
        "beats_used": int(len(beats)),
        "rejected_candidates": int(len(rejected_candidates)),
        "rejected_pulses": int(len(rejected_pulses)),
        "segmented_pulses_total": int(len(pulse_map)),
        "avg_duration": float(avg_duration),
        "fs_eff": float(fs_eff),
        "time_axis": time_axis.copy(),
        "feet": [tuple(map(int, x)) for x in valid_feet],
        "beat_info": beat_info,
        "pulse_map": pulse_map,
        "rejected_candidates_info": rejected_candidates,
        "rejected_pulses_info": rejected_pulses,
        "fiducials": fid.copy()
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax1 = axes[0]
    for b in beats_rs_aligned:
        ax1.plot(time_axis, b, color='black', alpha=0.12, linewidth=1)
    ax1.plot(time_axis, avg_wave, color=color_avg, linewidth=3, label='Ensemble Avg')
    ax1.fill_between(time_axis, avg_wave - std_wave, avg_wave + std_wave, color=color_avg, alpha=0.18, label='Std Dev')
    ax1.set_title(f"{title} Ensemble Average")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper right')

    ax2 = axes[1]
    ax2.plot(time_axis, vpg, color=color_avg, linewidth=2)
    ax2.axhline(0, color='black', alpha=0.3)
    ax2.set_title("1st Derivative (VPG)")
    ax2.set_ylabel("dPPG/dt")
    ax2.grid(True, alpha=0.3)

    ax3 = axes[2]
    ax3.plot(time_axis, sdppg_s, color=color_avg, linewidth=2)
    ax3.axhline(0, color='black', alpha=0.3)
    ax3.set_title("2nd Derivative (SDPPG)")
    ax3.set_ylabel("d²PPG/dt²")
    ax3.set_xlabel("Time (s)")
    ax3.grid(True, alpha=0.3)

    for name in ["a", "b", "c", "d", "e"]:
        idx = fid[name]
        if idx is not None:
            ax3.plot(time_axis[idx], sdppg_s[idx], 'o', markersize=7)
            ax3.text(time_axis[idx], sdppg_s[idx], f" {name}", fontsize=11, weight='bold')

    plt.tight_layout()
    save_if_requested(fig, save_file)

    if verbose:
        print(f"✅ {title}: beats={len(beats)}, fs_eff={fs_eff:.2f}")

    return avg_wave, vpg, sdppg_s, beats_rs_aligned, len(beats), fid, meta

def plot_pulse_numbering(signal_data, fs, title="Pulse Numbering", save_file=None, verbose=False):
    (
        peaks, feet, beats, valid_feet, beat_info,
        smooth_sig, vpg, sdppg, valleys,
        rejected_candidates, rejected_pulses, pulse_map
    ) = detect_beats_foot_to_foot(signal_data, fs)

    t = np.arange(len(signal_data)) / fs
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(t, smooth_sig, label="Smoothed PPG", linewidth=1.8)

    if len(valleys):
        ax.plot(t[valleys], smooth_sig[valleys], "gv", alpha=0.7, label="Valleys")

    for item in pulse_map:
        pulse_num = item["pulse_num"]
        mid_idx = item["mid_idx"]
        y = smooth_sig[mid_idx]

        if item["status"] == "accepted":
            ax.text(t[mid_idx], y + 0.06, str(pulse_num),
                    color="blue", fontsize=11, fontweight="bold",
                    ha="center", va="bottom")
            ax.axvspan(t[item["foot"]], t[item["next_foot"]], alpha=0.05, color="green")
        else:
            ax.text(t[mid_idx], y + 0.06, str(pulse_num),
                    color="red", fontsize=11, fontweight="bold",
                    ha="center", va="bottom")
            ax.axvspan(t[item["foot"]], t[item["next_foot"]], alpha=0.10, color="red")

    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()

    save_if_requested(fig, save_file)

    if verbose:
        print(f"✅ {title}: pulses={len(pulse_map)}")

def debug_plot_feet(signal_data, fs, title="Foot Check", save_file=None, verbose=False):
    (
        peaks, feet, beats, valid_feet, beat_info,
        smooth_sig, vpg, sdppg, valleys,
        rejected_candidates, rejected_pulses, pulse_map
    ) = detect_beats_foot_to_foot(signal_data, fs)

    t = np.arange(len(signal_data)) / fs

    accepted_peaks = np.array([b["peak"] for b in beat_info], dtype=int) if len(beat_info) else np.array([], dtype=int)
    accepted_feet = np.array([b["foot"] for b in beat_info], dtype=int) if len(beat_info) else np.array([], dtype=int)

    rejected_peaks = np.array(
        [x["peak"] for x in rejected_pulses if "peak" in x and x["peak"] is not None],
        dtype=int
    ) if len(rejected_pulses) else np.array([], dtype=int)

    rejected_feet = np.array(
        [x["foot"] for x in rejected_pulses if "foot" in x and x["foot"] is not None],
        dtype=int
    ) if len(rejected_pulses) else np.array([], dtype=int)

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    axes[0].plot(t, smooth_sig, label="Smoothed PPG")
    if len(valleys):
        axes[0].plot(t[valleys], smooth_sig[valleys], "gv", label="Valleys", alpha=0.7)
    if len(accepted_peaks):
        axes[0].plot(t[accepted_peaks], smooth_sig[accepted_peaks], "ro", label="Accepted Peaks")
    if len(accepted_feet):
        axes[0].plot(t[accepted_feet], smooth_sig[accepted_feet], "ko", label="Accepted Feet")
    if len(rejected_peaks):
        axes[0].plot(t[rejected_peaks], smooth_sig[rejected_peaks], "x", color="orange", markersize=8, label="Rejected Peaks")
    axes[0].set_title(title + " - PPG")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(t, vpg, label="VPG")
    axes[1].axhline(0, color="black", alpha=0.4)
    if len(accepted_feet):
        axes[1].plot(t[accepted_feet], vpg[accepted_feet], "ko", label="Accepted Feet on VPG")
    if len(accepted_peaks):
        axes[1].plot(t[accepted_peaks], vpg[accepted_peaks], "ro", label="Accepted Peaks on VPG", alpha=0.7)
    if len(rejected_feet):
        axes[1].plot(t[rejected_feet], vpg[rejected_feet], "x", color="orange", markersize=8, label="Rejected Feet on VPG")
    axes[1].set_title("1st Derivative (VPG)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, sdppg, label="SDPPG")
    axes[2].axhline(0, color="black", alpha=0.4)
    if len(accepted_feet):
        axes[2].plot(t[accepted_feet], sdppg[accepted_feet], "ko", label="Accepted Feet on SDPPG")
    if len(rejected_feet):
        axes[2].plot(t[rejected_feet], sdppg[rejected_feet], "x", color="orange", markersize=8, label="Rejected Feet on SDPPG")
    axes[2].set_title("2nd Derivative (SDPPG)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    save_if_requested(fig, save_file)

    if verbose:
        print(f"✅ {title}: peaks={len(peaks)}, feet={len(feet)}, accepted={len(beats)}")

# -------------------------------------------------
# REJECTED WINDOW PLOT
# -------------------------------------------------
def plot_rejected_window(signal_data, fs, title, color_avg, reason_text,
                         beats_found, min_required, save_file=None):
    """
    Generates a rejection summary plot showing the normalized signal,
    detected peaks/feet/valleys, and a clear text annotation of the reason.
    """
    (
        peaks, feet, beats, valid_feet, beat_info,
        smooth_sig, vpg, sdppg, valleys,
        rejected_candidates, rejected_pulses, pulse_map
    ) = detect_beats_foot_to_foot(signal_data, fs)

    t = np.arange(len(signal_data)) / fs
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(t, smooth_sig, color=color_avg, linewidth=1.6, label="Smoothed PPG")

    if len(peaks):
        ax.plot(t[peaks], smooth_sig[peaks], "ro", markersize=6, label=f"Detected Peaks ({len(peaks)})")
    if len(valleys):
        ax.plot(t[valleys], smooth_sig[valleys], "gv", markersize=6, alpha=0.7, label=f"Valleys ({len(valleys)})")

    accepted_feet = np.array([b["foot"] for b in beat_info], dtype=int) if len(beat_info) else np.array([], dtype=int)
    if len(accepted_feet):
        ax.plot(t[accepted_feet], smooth_sig[accepted_feet], "ko", markersize=7, label=f"Accepted Feet ({len(accepted_feet)})")

    # Highlight rejected pulse regions
    for item in pulse_map:
        if item["status"] != "accepted":
            ax.axvspan(t[item["foot"]], t[item["next_foot"]], alpha=0.10, color="red")

    # Build reason text
    rej_summary = summarize_reason_counts(rejected_pulses, "reason_key")
    rej_lines = [f"  • {k}: {v}" for k, v in rej_summary.items()]
    rej_block = "\n".join(rej_lines) if rej_lines else "  (no per-pulse rejections logged)"

    info_text = (
        f"REJECTION REASON:\n  {reason_text}\n\n"
        f"Beats Found: {beats_found}    |    Required: {min_required}\n"
        f"Total Pulses Segmented: {len(pulse_map)}\n"
        f"Pulse-level Rejections:\n{rej_block}"
    )

    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                      edgecolor='red', alpha=0.9))

    ax.set_title(f"[REJECTED WINDOW] — {title}", fontsize=13, fontweight='bold', color='darkred')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")
    plt.tight_layout()

    save_if_requested(fig, save_file)

# -------------------------------------------------
# FULL ENSEMBLE WRAPPER
# -------------------------------------------------
def run_step11_ensemble(signal_data, fs, channel_name, plot_root, base_name,
                        target_len=ENSEMBLE_TARGET_LEN, min_valid_beats=3):
    """
    Runs Step 11 and saves the 3 ensemble plots:
      1) Ensemble average    → 10_Ensemble/
      2) Foot verification   → 09_Debug/
      3) Pulse numbering     → 11_PulseNumbering/
    
    Returns:
      - On success: dict with avg_wave, vpg, sdppg, meta, plot_paths
      - On rejection: dict with status="REJECTED", reason, rejection_plot path
    """
    channel_tag = str(channel_name).upper()

    # 🆕 RENAMED FOLDERS (per user request)
    ensemble_dir = os.path.join(plot_root, "10_Ensemble")
    debug_dir = os.path.join(plot_root, "09_Debug")
    numbering_dir = os.path.join(plot_root, "11_PulseNumbering")

    os.makedirs(ensemble_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    os.makedirs(numbering_dir, exist_ok=True)

    # 🆕 RENAMED FILES (per user request)
    ensemble_plot = os.path.join(ensemble_dir, f"{base_name}_{channel_tag}_10_Ensemble.png")
    debug_plot = os.path.join(debug_dir, f"{base_name}_{channel_tag}_09_FootCheck.png")
    numbering_plot = os.path.join(numbering_dir, f"{base_name}_{channel_tag}_11_Numbering.png")

    ensemble_result = plot_ensemble_sdppg(
        signal_data=signal_data,
        fs=fs,
        title=f"{channel_tag} Channel",
        color_avg=("red" if channel_tag == "RED" else "blue"),
        min_valid_beats=min_valid_beats,
        target_len=target_len,
        save_file=ensemble_plot,
        verbose=False
    )

    # Always save the supporting plots (foot check + numbering) even on rejection
    # so user can visually inspect why
    debug_plot_feet(signal_data, fs, title=f"{channel_tag} Foot Verification", save_file=debug_plot, verbose=False)
    plot_pulse_numbering(signal_data, fs, title=f"{channel_tag} Pulse Numbering", save_file=numbering_plot, verbose=False)

    # If rejected → return rejection package + generate dedicated rejection plot
    if isinstance(ensemble_result, dict) and ensemble_result.get("status") == "REJECTED":
        rejection_plot = os.path.join(ensemble_dir, f"{base_name}_{channel_tag}_REJECTED.png")
        plot_rejected_window(
            signal_data=signal_data,
            fs=fs,
            title=f"{channel_tag} Channel — {base_name}",
            color_avg=("red" if channel_tag == "RED" else "blue"),
            reason_text=ensemble_result["reason"],
            beats_found=ensemble_result["beats_found"],
            min_required=ensemble_result["min_required"],
            save_file=rejection_plot
        )
        ensemble_result["plot_paths"] = {
            "rejection_plot": rejection_plot,
            "debug_plot": debug_plot,
            "numbering_plot": numbering_plot
        }
        return ensemble_result

    avg_wave, vpg, sdppg_s, beats_rs_aligned, n_beats, fid, meta = ensemble_result

    # Add plot paths into metadata
    meta["plot_paths"] = {
        "ensemble_plot": ensemble_plot,
        "debug_plot": debug_plot,
        "numbering_plot": numbering_plot
    }

    return {
        "status": "SUCCESS",
        "avg_wave": avg_wave,
        "vpg": vpg,
        "sdppg": sdppg_s,
        "beats_rs_aligned": beats_rs_aligned,
        "beats_used": n_beats,
        "fiducials": fid,
        "meta": meta,
        "plot_paths": meta["plot_paths"]
    }

print("✅ Ensemble + SDPPG core functions loaded.")



#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ==========================================
# ⚙️ STEP 13: PER-WINDOW PROCESSING + SAVING + VALIDATION
# ==========================================

def make_padded_column(arr, target_len):
    """Pads array with NaN to target length (for ensemble CSV)."""
    arr = np.asarray(arr, dtype=float)
    out_arr = np.full(target_len, np.nan, dtype=float)
    out_arr[:len(arr)] = arr
    return out_arr

def build_window_paths(filepath, output_root):
    """Builds standardized output paths for a single window file."""
    filename_no_ext = os.path.splitext(os.path.basename(filepath))[0]
    
    # Extract subject prefix (strip _WinX suffix if present)
    if "_Win" in filename_no_ext:
        subject_prefix = filename_no_ext.split("_Win")[0]
    else:
        subject_prefix = filename_no_ext
    
    main_folder = f"{subject_prefix}_Filtered"
    window_folder = f"{filename_no_ext}_Filtered"
    additional_folder = f"{subject_prefix}_Additional"
    
    main_path = os.path.join(output_root, main_folder)
    window_path = os.path.join(main_path, window_folder)
    additional_path = os.path.join(main_path, additional_folder)
    plots_root = os.path.join(additional_path, "Plots")
    
    return {
        "filename_no_ext": filename_no_ext,
        "subject_prefix": subject_prefix,
        "main_folder": main_folder,
        "window_folder": window_folder,
        "additional_folder": additional_folder,
        "main_path": main_path,
        "window_path": window_path,
        "additional_path": additional_path,
        "plots_root": plots_root,
        "file_full": f"{filename_no_ext}_Filtered_Full.csv",
        "file_ensemble": f"{filename_no_ext}_Filtered_Ensemble.csv",
        "file_config": f"{filename_no_ext}_Filtered_Configuration.json"
    }

def validate_saved_data(window_path, paths, expected_lengths, in_memory_arrays=None):
    """Validates saved CSVs (file existence, row counts, numerical integrity)."""
    report = {"status": "PASS", "checks": [], "issues": []}
    
    RTOL = 1e-9
    ATOL = 1e-9
    
    def compare_arrays(name, mem_arr, disk_arr):
        mem = np.asarray(mem_arr, dtype=np.float64)
        disk = np.asarray(disk_arr, dtype=np.float64)
        
        if len(mem) != len(disk):
            return {"match": False, "length_mem": len(mem), "length_disk": len(disk),
                    "max_abs_diff": None, "reason": "length_mismatch"}
        
        nan_mem = np.isnan(mem)
        nan_disk = np.isnan(disk)
        
        if not np.array_equal(nan_mem, nan_disk):
            return {"match": False, "length_mem": len(mem), "length_disk": len(disk),
                    "max_abs_diff": None, "reason": "nan_position_mismatch"}
        
        valid_mask = ~nan_mem
        if not np.any(valid_mask):
            return {"match": True, "length_mem": len(mem), "length_disk": len(disk),
                    "max_abs_diff": 0.0, "reason": "all_nan_match"}
        
        diff = np.abs(mem[valid_mask] - disk[valid_mask])
        max_abs_diff = float(np.max(diff))
        
        close = np.allclose(mem[valid_mask], disk[valid_mask],
                            rtol=RTOL, atol=ATOL, equal_nan=False)
        
        return {"match": bool(close), "length_mem": len(mem), "length_disk": len(disk),
                "max_abs_diff": max_abs_diff,
                "reason": "within_tolerance" if close else "exceeds_tolerance"}
    
    # Full CSV
    full_csv_path = os.path.join(window_path, paths["file_full"])
    if os.path.exists(full_csv_path):
        df_full = pd.read_csv(full_csv_path)
        check = {"file": paths["file_full"], "exists": True, "rows": len(df_full),
                 "cols": list(df_full.columns), "expected_rows": expected_lengths["full_len"],
                 "row_match": len(df_full) == expected_lengths["full_len"]}
        if not check["row_match"]:
            report["issues"].append(f"Full CSV row mismatch: got {len(df_full)}, expected {expected_lengths['full_len']}")
            report["status"] = "FAIL"
        
        if in_memory_arrays and "full" in in_memory_arrays:
            data_check = {}
            for col, in_mem_arr in in_memory_arrays["full"].items():
                if col in df_full.columns:
                    cmp = compare_arrays(col, in_mem_arr, df_full[col].to_numpy())
                    data_check[col] = cmp
                    if not cmp["match"]:
                        report["issues"].append(
                            f"Full CSV column '{col}' mismatch ({cmp['reason']}, max_diff={cmp['max_abs_diff']})"
                        )
                        report["status"] = "FAIL"
            check["data_check"] = data_check
        report["checks"].append(check)
    else:
        report["issues"].append(f"Missing file: {paths['file_full']}")
        report["status"] = "FAIL"
    
    # Ensemble CSV
    ens_csv_path = os.path.join(window_path, paths["file_ensemble"])
    if os.path.exists(ens_csv_path):
        df_ens = pd.read_csv(ens_csv_path)
        check = {"file": paths["file_ensemble"], "exists": True, "rows": len(df_ens),
                 "cols": list(df_ens.columns), "expected_rows": expected_lengths["ensemble_len"],
                 "row_match": len(df_ens) == expected_lengths["ensemble_len"]}
        if not check["row_match"]:
            report["issues"].append(f"Ensemble CSV row mismatch: got {len(df_ens)}, expected {expected_lengths['ensemble_len']}")
            report["status"] = "FAIL"
        
        if in_memory_arrays and "ensemble" in in_memory_arrays:
            data_check = {}
            for col, in_mem_arr in in_memory_arrays["ensemble"].items():
                if col in df_ens.columns:
                    cmp = compare_arrays(col, in_mem_arr, df_ens[col].to_numpy())
                    data_check[col] = cmp
                    if not cmp["match"]:
                        report["issues"].append(
                            f"Ensemble CSV column '{col}' mismatch ({cmp['reason']}, max_diff={cmp['max_abs_diff']})"
                        )
                        report["status"] = "FAIL"
            check["data_check"] = data_check
        report["checks"].append(check)
    else:
        report["issues"].append(f"Missing file: {paths['file_ensemble']}")
        report["status"] = "FAIL"
    
    # Config JSON
    cfg_path = os.path.join(window_path, paths["file_config"])
    if os.path.exists(cfg_path):
        report["checks"].append({"file": paths["file_config"], "exists": True,
                                 "size_bytes": os.path.getsize(cfg_path)})
    else:
        report["issues"].append(f"Missing file: {paths['file_config']}")
        report["status"] = "FAIL"
    
    return report

def process_single_window(filepath, output_root, file_idx, total_files):
    """Runs full pipeline for one CSV. Returns dict with status SUCCESS / REJECTED / FAILED."""
    filename = os.path.basename(filepath)
    paths = build_window_paths(filepath, output_root)
    
    result = {
        "file_index": file_idx,
        "filename": filename,
        "filename_no_ext": paths["filename_no_ext"],
        "status": "SUCCESS",
        "error": None,
        "rejection_reason": None,
        "output_window_folder": paths["window_path"]
    }
    
    try:
        # Prepare folders
        if os.path.exists(paths["window_path"]):
            shutil.rmtree(paths["window_path"])
            result["folder_replaced"] = True
        else:
            result["folder_replaced"] = False
        
        os.makedirs(paths["window_path"], exist_ok=True)
        os.makedirs(paths["additional_path"], exist_ok=True)
        os.makedirs(paths["plots_root"], exist_ok=True)
        
        plot_dirs = {
            "02": os.path.join(paths["plots_root"], "02_Raw_Selection"),
            "03": os.path.join(paths["plots_root"], "03_Despiked"),
            "04": os.path.join(paths["plots_root"], "04_Inverted"),
            "05": os.path.join(paths["plots_root"], "05_LowPass_Filtered"),
            "06": os.path.join(paths["plots_root"], "06_SG_Smoothed"),
            "07": os.path.join(paths["plots_root"], "07_HighPass_Filtered"),
            "08": os.path.join(paths["plots_root"], "08_Normalized"),
        }
        for d in plot_dirs.values():
            os.makedirs(d, exist_ok=True)
        
        base_name = paths["filename_no_ext"]
        
        # STEP 2-8
        full_clean_df, selected_df, s_start, s_end = load_and_validate_csv(filepath)
        plot_step2_raw(full_clean_df, selected_df, s_start, s_end, plot_dirs["02"], base_name)
        ir_despiked, red_despiked = step3_spike_removal(selected_df, plot_dirs["03"], base_name)
        ir_inverted, red_inverted = step4_inversion(ir_despiked, red_despiked, plot_dirs["04"], base_name)
        ir_filtered, red_filtered = step5_lowpass(ir_inverted, red_inverted, plot_dirs["05"], base_name)
        ir_smoothed, red_smoothed = step6_savgol(ir_filtered, red_filtered, plot_dirs["06"], base_name)
        ir_hpf, red_hpf = step7_highpass(ir_smoothed, red_smoothed, plot_dirs["07"], base_name)
        ir_norm, red_norm = step8_normalize(ir_hpf, red_hpf, plot_dirs["08"], base_name)
        
        # STEP 9
        sqi_ir = step9_quality_check(ir_hpf, selected_df['IR'].values, FS)
        sqi_red = step9_quality_check(red_hpf, selected_df['RED'].values, FS)
        
        # STEP 10
        stages = {
            "1_Raw":       (selected_df['IR'].values, selected_df['RED'].values),
            "2_Despiked":  (ir_despiked, red_despiked),
            "3_Inverted":  (ir_inverted, red_inverted),
            "4_LowPass":   (ir_filtered, red_filtered),
            "5_SG":        (ir_smoothed, red_smoothed),
            "6_HighPass":  (ir_hpf, red_hpf)
        }
        pipeline_diag = step10_pipeline_diagnostic(
            stages, selected_df['IR'].values, selected_df['RED'].values, FS
        )
        
        # STEP 11a — per-channel features
        feat_red = extract_glucose_features(red_norm, selected_df['RED'].values, FS)
        feat_ir = extract_glucose_features(ir_norm, selected_df['IR'].values, FS)
        
        # STEP 11b — Ensemble (with per-channel min beat thresholds)
        ens_red = run_step11_ensemble(
            red_norm, FS, "RED", paths["plots_root"], base_name,
            target_len=ENSEMBLE_TARGET_LEN, min_valid_beats=MIN_VALID_BEATS_RED
        )
        ens_ir = run_step11_ensemble(
            ir_norm, FS, "IR", paths["plots_root"], base_name,
            target_len=ENSEMBLE_TARGET_LEN, min_valid_beats=MIN_VALID_BEATS_IR
        )
        
        # 🆕 SOFT REJECTION HANDLING (instead of raising exception)
        red_rejected = (ens_red.get("status") == "REJECTED")
        ir_rejected = (ens_ir.get("status") == "REJECTED")
        
        if red_rejected or ir_rejected:
            reason_parts = []
            if ir_rejected:
                reason_parts.append(f"IR: {ens_ir['reason']}")
            if red_rejected:
                reason_parts.append(f"RED: {ens_red['reason']}")
            full_reason = " | ".join(reason_parts)
            
            # Save a rejection-config JSON so it's preserved on disk
            rejection_config = {
                "metadata": {
                    "source_file": filepath,
                    "filename": filename,
                    "sampling_rate_fs": float(FS),
                    "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "status": "REJECTED"
                },
                "rejection": {
                    "overall_reason": full_reason,
                    "min_valid_beats_ir": MIN_VALID_BEATS_IR,
                    "min_valid_beats_red": MIN_VALID_BEATS_RED,
                    "IR_channel": {
                        "rejected": ir_rejected,
                        "reason": ens_ir.get("reason") if ir_rejected else None,
                        "beats_found": ens_ir.get("beats_found") if ir_rejected else ens_ir.get("beats_used"),
                        "rejected_pulses": ens_ir.get("rejected_pulses"),
                        "pulse_rejection_summary": ens_ir.get("rejection_reason_summary") if ir_rejected else None
                    },
                    "RED_channel": {
                        "rejected": red_rejected,
                        "reason": ens_red.get("reason") if red_rejected else None,
                        "beats_found": ens_red.get("beats_found") if red_rejected else ens_red.get("beats_used"),
                        "rejected_pulses": ens_red.get("rejected_pulses"),
                        "pulse_rejection_summary": ens_red.get("rejection_reason_summary") if red_rejected else None
                    },
                    "rejection_plots": {
                        "IR": ens_ir.get("plot_paths", {}).get("rejection_plot") if ir_rejected else None,
                        "RED": ens_red.get("plot_paths", {}).get("rejection_plot") if red_rejected else None
                    }
                },
                "signal_quality": {"IR": sqi_ir, "RED": sqi_red},
                "pipeline_diagnostic": pipeline_diag
            }
            
            path_config = os.path.join(paths["window_path"], paths["file_config"])
            with open(path_config, "w") as f:
                json.dump(to_json_safe(rejection_config), f, indent=4)
            
            result["status"] = "REJECTED"
            result["rejection_reason"] = full_reason
            result["rejection_details"] = {
                "IR_rejected": ir_rejected,
                "RED_rejected": red_rejected,
                "IR_reason": ens_ir.get("reason") if ir_rejected else None,
                "RED_reason": ens_red.get("reason") if red_rejected else None,
                "IR_beats_found": ens_ir.get("beats_found") if ir_rejected else ens_ir.get("beats_used"),
                "RED_beats_found": ens_red.get("beats_found") if red_rejected else ens_red.get("beats_used"),
                "min_required_IR": MIN_VALID_BEATS_IR,
                "min_required_RED": MIN_VALID_BEATS_RED
            }
            # 🆕 Per-channel rejected-beat info (for terminal detail table)
            result["rejection_beat_details"] = {
                "IR": {
                    "rejected": ir_rejected,
                    "beats_found": ens_ir.get("beats_found", ens_ir.get("beats_used", 0)),
                    "min_required": MIN_VALID_BEATS_IR,
                    "segmented_pulses_total": ens_ir.get("segmented_pulses_total", 0),
                    "rejected_pulses_info": ens_ir.get("rejected_pulses_info", []),
                    "rejected_candidates_info": ens_ir.get("rejected_candidates_info", [])
                },
                "RED": {
                    "rejected": red_rejected,
                    "beats_found": ens_red.get("beats_found", ens_red.get("beats_used", 0)),
                    "min_required": MIN_VALID_BEATS_RED,
                    "segmented_pulses_total": ens_red.get("segmented_pulses_total", 0),
                    "rejected_pulses_info": ens_red.get("rejected_pulses_info", []),
                    "rejected_candidates_info": ens_red.get("rejected_candidates_info", [])
                }
            }
            result["sqi_ir"] = sqi_ir
            result["sqi_red"] = sqi_red
            return result
        
        # SUCCESS PATH
        avg_wave_red = ens_red["avg_wave"]
        vpg_red = ens_red["vpg"]
        sdppg_red = ens_red["sdppg"]
        meta_red = ens_red["meta"]
        
        avg_wave_ir = ens_ir["avg_wave"]
        vpg_ir = ens_ir["vpg"]
        sdppg_ir = ens_ir["sdppg"]
        meta_ir = ens_ir["meta"]
        
        # STEP 12
        gs_red = extract_features_from_ensemble_wave(
            avg_wave_red, feat_red['hr'], selected_df['RED'].values, meta_red['fs_eff']
        )
        gs_ir = extract_features_from_ensemble_wave(
            avg_wave_ir, feat_ir['hr'], selected_df['IR'].values, meta_ir['fs_eff']
        )
        
        # STEP 13a — Full CSV
        df_full = pd.DataFrame({
            "Red_AC_HighPass": red_hpf,
            "IR_AC_HighPass": ir_hpf,
            "Red_DC_LowPass": red_filtered,
            "IR_DC_LowPass": ir_filtered,
            "Red_Normalized": red_norm,
            "IR_Normalized": ir_norm
        })
        path_full = os.path.join(paths["window_path"], paths["file_full"])
        df_full.to_csv(path_full, index=False)
        
        # STEP 13b — Ensemble CSV
        max_len = max(len(avg_wave_red), len(avg_wave_ir))
        df_ensemble = pd.DataFrame({
            "Time_Red_s":       make_padded_column(meta_red["time_axis"], max_len),
            "Red_Ensemble_Avg": make_padded_column(avg_wave_red, max_len),
            "Red_VPG":          make_padded_column(vpg_red, max_len),
            "Red_SDPPG":        make_padded_column(sdppg_red, max_len),
            "Time_IR_s":        make_padded_column(meta_ir["time_axis"], max_len),
            "IR_Ensemble_Avg":  make_padded_column(avg_wave_ir, max_len),
            "IR_VPG":           make_padded_column(vpg_ir, max_len),
            "IR_SDPPG":         make_padded_column(sdppg_ir, max_len),
        })
        path_ensemble = os.path.join(paths["window_path"], paths["file_ensemble"])
        df_ensemble.to_csv(path_ensemble, index=False)
        
        # STEP 13c — Config JSON
        total_samples = len(red_hpf)
        plot_time = total_samples / float(FS)
        nyquist_rate = float(FS) / 2.0
        
        window_config = {
            "metadata": {
                "source_file": filepath,
                "filename": filename,
                "sampling_rate_fs": float(FS),
                "total_samples": int(total_samples),
                "duration_sec": float(plot_time),
                "nyquist_rate": float(nyquist_rate),
                "date_processed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "status": "SUCCESS"
            },
            "hyperparameters": {
                "use_full_file": USE_FULL_FILE,
                "sample_start": SAMPLE_START,
                "sample_end": SAMPLE_END,
                "threshold_red": THRESHOLD_RED,
                "threshold_ir": THRESHOLD_IR,
                "spike_enable": SPIKE_ENABLE,
                "kernel_size": KERNEL_SIZE,
                "invert_enable": INVERT_ENABLE,
                "lp_enable": LP_ENABLE,
                "lp_cutoff_hz": LP_CUTOFF,
                "lp_order": LP_ORDER,
                "hp_enable": HP_ENABLE,
                "hp_cutoff_hz": HP_CUTOFF,
                "hp_order": HP_ORDER,
                "sg_enable": SG_ENABLE,
                "sg_window": SG_WINDOW,
                "sg_poly": SG_POLY,
                "norm_selection": NORM_SELECTION,
                "norm_type": "MinMax" if NORM_SELECTION == 1 else "ZScore",
                "ensemble_target_len": ENSEMBLE_TARGET_LEN,
                "min_valid_beats_ir": MIN_VALID_BEATS_IR,
                "min_valid_beats_red": MIN_VALID_BEATS_RED,
                "peak_min_distance_sec": PEAK_MIN_DISTANCE_SEC,
                "peak_prom_factor": PEAK_PROM_FACTOR,
                "valley_min_distance_sec": VALLEY_MIN_DISTANCE_SEC,
                "valley_prom_factor": VALLEY_PROM_FACTOR,
                "min_foot_to_peak_sec": MIN_FOOT_TO_PEAK_SEC,
                "max_foot_to_peak_sec": MAX_FOOT_TO_PEAK_SEC,
                "max_valley_to_foot_sec": MAX_VALLEY_TO_FOOT_SEC,
                "max_foot_rel_height": MAX_FOOT_REL_HEIGHT,
                "max_abs_vpg_at_foot": MAX_ABS_VPG_AT_FOOT,
                "edge_exclusion_sec": EDGE_EXCLUSION_SEC,
                "min_beat_duration_sec": MIN_BEAT_DURATION_SEC,
                "max_beat_duration_sec": MAX_BEAT_DURATION_SEC,
                "main_peak_search_window_sec": MAIN_PEAK_SEARCH_WINDOW_SEC,
                "main_peak_min_delay_sec": MAIN_PEAK_MIN_DELAY_SEC,
                "start_incomplete_margin_sec": START_INCOMPLETE_MARGIN_SEC,
                "end_incomplete_margin_sec": END_INCOMPLETE_MARGIN_SEC,
                "sqi_limits": SQI_LIMITS
            },
            "folder_structure": {
                "output_root": output_root,
                "main_folder": paths["main_path"],
                "window_folder": paths["window_path"],
                "additional_folder": paths["additional_path"]
            },
            "signal_quality": {"IR": sqi_ir, "RED": sqi_red},
            "pipeline_diagnostic": pipeline_diag,
            "ppg_features_per_channel": {"IR": feat_ir, "RED": feat_red},
            "golden_standard_features": {"IR": gs_ir, "RED": gs_red},
            "ensemble_RED": meta_red,
            "ensemble_IR": meta_ir
        }
        
        path_config = os.path.join(paths["window_path"], paths["file_config"])
        with open(path_config, "w") as f:
            json.dump(to_json_safe(window_config), f, indent=4)
        
        # STEP 13d — Validate
        in_memory_for_validation = {
            "full": {
                "Red_AC_HighPass": red_hpf,
                "IR_AC_HighPass":  ir_hpf,
                "Red_DC_LowPass":  red_filtered,
                "IR_DC_LowPass":   ir_filtered,
                "Red_Normalized":  red_norm,
                "IR_Normalized":   ir_norm
            },
            "ensemble": {
                "Time_Red_s":       make_padded_column(meta_red["time_axis"], max_len),
                "Red_Ensemble_Avg": make_padded_column(avg_wave_red, max_len),
                "Red_VPG":          make_padded_column(vpg_red, max_len),
                "Red_SDPPG":        make_padded_column(sdppg_red, max_len),
                "Time_IR_s":        make_padded_column(meta_ir["time_axis"], max_len),
                "IR_Ensemble_Avg":  make_padded_column(avg_wave_ir, max_len),
                "IR_VPG":           make_padded_column(vpg_ir, max_len),
                "IR_SDPPG":         make_padded_column(sdppg_ir, max_len),
            }
        }
        
        validation = validate_saved_data(
            paths["window_path"], paths,
            expected_lengths={"full_len": len(df_full), "ensemble_len": len(df_ensemble)},
            in_memory_arrays=in_memory_for_validation
        )
        
        result.update({
            "samples_processed": int(total_samples),
            "duration_sec": float(plot_time),
            "sqi_ir": sqi_ir,
            "sqi_red": sqi_red,
            "features_ir": feat_ir,
            "features_red": feat_red,
            "golden_ir": gs_ir,
            "golden_red": gs_red,
            "ensemble_red_summary": {
                "beats_used": meta_red["beats_used"],
                "total_segmented": meta_red["segmented_pulses_total"],
                "rejected_candidates": meta_red["rejected_candidates"],
                "rejected_pulses": meta_red["rejected_pulses"],
                "avg_beat_duration_s": meta_red["avg_duration"],
                "fiducials": meta_red["fiducials"]
            },
            "ensemble_ir_summary": {
                "beats_used": meta_ir["beats_used"],
                "total_segmented": meta_ir["segmented_pulses_total"],
                "rejected_candidates": meta_ir["rejected_candidates"],
                "rejected_pulses": meta_ir["rejected_pulses"],
                "avg_beat_duration_s": meta_ir["avg_duration"],
                "fiducials": meta_ir["fiducials"]
            },
            "validation": validation
        })
        
    except Exception as e:
        import traceback
        result["status"] = "FAILED"
        result["error"] = str(e)
        result["traceback"] = traceback.format_exc()
    
    return result

print("✅ Per-window pipeline runner loaded.")


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


# ==========================================
# 🚀 STEP 14: MAIN AUTOMATION LOOP
# ==========================================

def print_section(title, char="=", width=110):
    print("\n" + char * width)
    print(f"  {title}")
    print(char * width)

def fmt_num(v, decimals=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, (int, np.integer)):
        return f"{v}"
    return f"{v:.{decimals}f}"

def print_sqi_table(results):
    print_section("📋 SIGNAL QUALITY INDEX (SQI) — STEP 9")
    header = f"{'Window':<35} | {'CH':<4} | {'Skew':>7} | {'Kurt':>7} | {'PI%':>7} | {'ZCR Hz':>7} | {'SNR dB':>7} | {'Pass'}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] not in ("SUCCESS", "REJECTED"):
            continue
        if "sqi_ir" not in r or "sqi_red" not in r:
            continue
        for ch, sqi in [("IR", r["sqi_ir"]), ("RED", r["sqi_red"])]:
            passes = sum(1 for k in ["skewness_status", "kurtosis_status", "perfusion_index_status", "zcr_status", "snr_status"]
                        if sqi[k] == "PASS")
            print(f"{r['filename_no_ext']:<35} | {ch:<4} | "
                  f"{fmt_num(sqi['skewness'],2):>7} | {fmt_num(sqi['kurtosis'],2):>7} | "
                  f"{fmt_num(sqi['perfusion_index_pct'],2):>7} | {fmt_num(sqi['zcr_hz'],2):>7} | "
                  f"{fmt_num(sqi['snr_db'],2):>7} | {passes}/5")

def print_feature_table(results):
    print_section("🧬 PPG FEATURE EXTRACTION (per channel) — STEP 11")
    header = f"{'Window':<35} | {'CH':<4} | {'HR':>6} | {'Width':>7} | {'Amp':>7} | {'Area':>9} | {'Skew':>6} | {'Kurt':>6}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] != "SUCCESS":
            continue
        for ch, f in [("IR", r["features_ir"]), ("RED", r["features_red"])]:
            print(f"{r['filename_no_ext']:<35} | {ch:<4} | "
                  f"{fmt_num(f['hr'],1):>6} | {fmt_num(f['width'],3):>7} | "
                  f"{fmt_num(f['sys_amp'],3):>7} | {fmt_num(f['area'],2):>9} | "
                  f"{fmt_num(f['skew'],2):>6} | {fmt_num(f['kurt'],2):>6}")

def print_golden_table(results):
    print_section("🌟 GOLDEN STANDARD FEATURES (Ensemble Avg) — STEP 12")
    header = f"{'Window':<35} | {'CH':<4} | {'HR':>6} | {'Width':>7} | {'Amp':>7} | {'Area':>9} | {'PI%':>7}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] != "SUCCESS":
            continue
        for ch, g in [("IR", r["golden_ir"]), ("RED", r["golden_red"])]:
            print(f"{r['filename_no_ext']:<35} | {ch:<4} | "
                  f"{fmt_num(g['HR_Global'],1):>6} | {fmt_num(g['Width_Avg'],3):>7} | "
                  f"{fmt_num(g['Amp_Avg'],3):>7} | {fmt_num(g['Area_Avg'],2):>9} | "
                  f"{fmt_num(g['PI_Global'],2):>7}")

def print_ensemble_table(results):
    print_section("🫀 ENSEMBLE SUMMARY — STEP 11")
    header = f"{'Window':<35} | {'CH':<4} | {'Beats':>6} | {'Segmented':>10} | {'RejPulses':>10} | {'AvgDur(s)':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] != "SUCCESS":
            continue
        for ch, e in [("IR", r["ensemble_ir_summary"]), ("RED", r["ensemble_red_summary"])]:
            print(f"{r['filename_no_ext']:<35} | {ch:<4} | "
                  f"{e['beats_used']:>6} | {e['total_segmented']:>10} | "
                  f"{e['rejected_pulses']:>10} | {fmt_num(e['avg_beat_duration_s'],3):>10}")

def print_rejection_table(results):
    """Prints a clear table of REJECTED windows with reasons."""
    rejected = [r for r in results if r["status"] == "REJECTED"]
    if not rejected:
        return
    print_section("⚠️  REJECTED WINDOWS — INSUFFICIENT BEATS")
    header = f"{'Window':<35} | {'CH':<4} | {'Beats':>6} | {'Required':>9} | {'Reason'}"
    print(header)
    print("-" * 110)
    for r in rejected:
        d = r.get("rejection_details", {})
        if d.get("IR_rejected"):
            print(f"{r['filename_no_ext']:<35} | {'IR':<4} | "
                  f"{d.get('IR_beats_found','?'):>6} | {d.get('min_required_IR','?'):>9} | "
                  f"{d.get('IR_reason','')}")
        if d.get("RED_rejected"):
            print(f"{r['filename_no_ext']:<35} | {'RED':<4} | "
                  f"{d.get('RED_beats_found','?'):>6} | {d.get('min_required_RED','?'):>9} | "
                  f"{d.get('RED_reason','')}")

def print_validation_table(results):
    print_section("✅ SAVE & VALIDATION CHECK — STEP 13")
    header = f"{'Window':<35} | {'Folder Replaced':>16} | {'Files Saved':>12} | {'Validation':>10}"
    print(header)
    print("-" * len(header))
    for r in results:
        if r["status"] != "SUCCESS":
            label = r["status"]
            print(f"{r['filename_no_ext']:<35} | {'-':>16} | {'-':>12} | {label:>10}")
            continue
        replaced = "YES" if r.get("folder_replaced") else "NO"
        n_files = len([c for c in r["validation"]["checks"] if c.get("exists")])
        status = r["validation"]["status"]
        print(f"{r['filename_no_ext']:<35} | {replaced:>16} | {n_files:>12} | {status:>10}")

def print_error_summary(results):
    failed = [r for r in results if r["status"] == "FAILED"]
    if not failed:
        print_section("✅ NO PROCESSING ERRORS")
        return
    
    print_section("❌ FILES WITH ERRORS")
    for r in failed:
        print(f"  • {r['filename']}: {r['error']}")

def print_final_summary(results, output_root):
    success = sum(1 for r in results if r["status"] == "SUCCESS")
    rejected = sum(1 for r in results if r["status"] == "REJECTED")
    failed = sum(1 for r in results if r["status"] == "FAILED")
    
    print_section("🎯 FINAL SUMMARY", char="=")
    print(f"  Total files     : {len(results)}")
    print(f"  ✅ Successful   : {success}")
    print(f"  ⚠️  Rejected    : {rejected}")
    print(f"  ❌ Failed       : {failed}")
    print(f"  📂 Output root  : {output_root}")
    print("=" * 110)

# -------------------------------------------------
# REJECTED-BEATS DETAIL TABLE (per window)
# -------------------------------------------------
def print_rejected_beats_detail(result):
    """
    Prints a per-window table of all individual beats/pulses that were
    rejected inside a REJECTED window, with reason + actual value + accepted range.
    Reads from result['rejection_beat_details'] (set in process_single_window).
    """
    details = result.get("rejection_beat_details", {})
    if not details:
        return
    
    for ch in ("IR", "RED"):
        ch_info = details.get(ch)
        if not ch_info:
            continue
        
        rej_pulses = ch_info.get("rejected_pulses_info", [])
        seg_total = ch_info.get("segmented_pulses_total", 0)
        beats_found = ch_info.get("beats_found", 0)
        min_req = ch_info.get("min_required", 0)
        
        if not rej_pulses and seg_total == 0:
            continue
        
        print(f"          └─ {ch} channel — segmented pulses: {seg_total}, "
              f"accepted: {beats_found}, required: >= {min_req}")
        
        if not rej_pulses:
            print(f"             (no per-pulse rejections logged — likely too few pulses segmented)")
            continue
        
        # Header (with two new columns: Actual + AcceptedRange)
        print(f"             {'Pulse#':>6} | {'Foot':>6} | {'Peak':>6} | {'NextFt':>6} | "
              f"{'Actual':>22} | {'AcceptedRange':>28} | Reason")
        print(f"             {'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-"
              f"{'-'*22}-+-{'-'*28}-+-{'-'*40}")
        
        for rp in rej_pulses:
            pnum   = rp.get("pulse_num", "?")
            foot   = rp.get("foot", "-")
            peak   = rp.get("peak", "-")
            nfoot  = rp.get("next_foot", "-")
            reason = rp.get("reason_text", rp.get("reason_key", "unknown"))
            actual = rp.get("actual_value", "n/a")
            acc    = rp.get("accepted_range", "n/a")
            
            foot_s   = str(foot)   if foot   is not None else "-"
            peak_s   = str(peak)   if peak   is not None else "-"
            nfoot_s  = str(nfoot)  if nfoot  is not None else "-"
            actual_s = str(actual) if actual is not None else "n/a"
            acc_s    = str(acc)    if acc    is not None else "n/a"
            
            # Truncate long fields
            if len(actual_s) > 22: actual_s = actual_s[:19] + "..."
            if len(acc_s)    > 28: acc_s    = acc_s[:25] + "..."
            
            print(f"             {pnum:>6} | {foot_s:>6} | {peak_s:>6} | {nfoot_s:>6} | "
                  f"{actual_s:>22} | {acc_s:>28} | {reason}")


# -------------------------------------------------
# MAIN LOOP (handles batch & single)
# -------------------------------------------------
print_section("🚀 STARTING AUTOMATED PIPELINE", char="=")
print(f"  Mode             : {'BATCH' if MODE == 1 else 'SINGLE'}")
print(f"  Folders to run   : {len(folders_to_process)}")
print(f"  Output folder    : {SAVE_ROOT_FIXED}")
print(f"  Sampling rate    : {FS} Hz")
print(f"  MIN_VALID_BEATS_IR  : {MIN_VALID_BEATS_IR}")
print(f"  MIN_VALID_BEATS_RED : {MIN_VALID_BEATS_RED}")
print("=" * 110)

# Process each folder in sequence
all_results_global = []        # all windows across all folders
folder_results_map = {}        # {folder_path: [results]}

for folder_idx, folder_path in enumerate(folders_to_process, start=1):
    print_section(f"📁 [{folder_idx}/{len(folders_to_process)}] FOLDER: {os.path.basename(folder_path)}", char="-")
    
    csv_files = sorted(glob.glob(os.path.join(folder_path, "*.csv")))
    if not csv_files:
        print(f"  ⚠️  No CSV files in this folder — skipping.")
        continue
    
    print(f"  🔍 Found {len(csv_files)} CSV file(s)")
    
    # 🆕 WIPE EXISTING SUBJECT OUTPUT FOLDERS (force full reprocessing)
    subject_prefixes_in_folder = set()
    for fp in csv_files:
        fn_noext = os.path.splitext(os.path.basename(fp))[0]
        subj = fn_noext.split("_Win")[0] if "_Win" in fn_noext else fn_noext
        subject_prefixes_in_folder.add(subj)
    
    for subj in sorted(subject_prefixes_in_folder):
        subj_out_folder = os.path.join(SAVE_ROOT_FIXED, f"{subj}_Filtered")
        if os.path.exists(subj_out_folder):
            try:
                shutil.rmtree(subj_out_folder)
                print(f"  🗑️  Wiped existing output folder: {subj}_Filtered")
            except Exception as wipe_err:
                print(f"  ⚠️  Could not wipe {subj}_Filtered: {wipe_err}")
    
    folder_results = []
    for idx, filepath in enumerate(csv_files, start=1):
        filename = os.path.basename(filepath)
        print(f"\n  [{idx:>3}/{len(csv_files)}] 🔄 {filename}")
        
        result = process_single_window(filepath, SAVE_ROOT_FIXED, idx, len(csv_files))
        folder_results.append(result)
        all_results_global.append(result)
        
        if result["status"] == "SUCCESS":
            beats_r = result["ensemble_red_summary"]["beats_used"]
            beats_i = result["ensemble_ir_summary"]["beats_used"]
            replaced_str = " (overwrote existing)" if result.get("folder_replaced") else ""
            val_str = result["validation"]["status"]
            print(f"        ✅ DONE — RED beats: {beats_r}, IR beats: {beats_i} | Validation: {val_str}{replaced_str}")
        elif result["status"] == "REJECTED":
            print(f"        ⚠️  REJECTED — {result['rejection_reason']}")
            print_rejected_beats_detail(result)
        else:
            print(f"        ❌ FAILED — {result['error']}")
    
    folder_results_map[folder_path] = folder_results

# -------------------------------------------------
# PRINT SUMMARY TABLES (across all folders)
# -------------------------------------------------
print_sqi_table(all_results_global)
print_feature_table(all_results_global)
print_golden_table(all_results_global)
print_ensemble_table(all_results_global)
print_rejection_table(all_results_global)
print_validation_table(all_results_global)
print_error_summary(all_results_global)

# -------------------------------------------------
# WRITE PER-SUBJECT COMBINED JSON REPORTS
# -------------------------------------------------
combined_report_root = {
    "processing_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "mode": "BATCH" if MODE == 1 else "SINGLE",
    "folders_processed": [str(f) for f in folders_to_process],
    "output_root": SAVE_ROOT_FIXED,
    "min_valid_beats_ir": MIN_VALID_BEATS_IR,
    "min_valid_beats_red": MIN_VALID_BEATS_RED,
    "total_files": len(all_results_global),
    "total_success": sum(1 for r in all_results_global if r["status"] == "SUCCESS"),
    "total_rejected": sum(1 for r in all_results_global if r["status"] == "REJECTED"),
    "total_failed": sum(1 for r in all_results_global if r["status"] == "FAILED"),
}

# Build summary rows
summary_rows = []
for r in all_results_global:
    if r["status"] == "FAILED":
        summary_rows.append({
            "filename": r["filename"],
            "status": "FAILED",
            "error": r["error"]
        })
        continue
    
    if r["status"] == "REJECTED":
        summary_rows.append({
            "filename": r["filename"],
            "status": "REJECTED",
            "rejection_reason": r["rejection_reason"],
            "rejection_details": r.get("rejection_details", {})
        })
        continue
    
    for ch in ["IR", "RED"]:
        sqi = r[f"sqi_{ch.lower()}"]
        feat = r[f"features_{ch.lower()}"]
        gold = r[f"golden_{ch.lower()}"]
        ens = r[f"ensemble_{ch.lower()}_summary"]
        
        summary_rows.append({
            "filename": r["filename"],
            "channel": ch,
            "status": "SUCCESS",
            "samples": r["samples_processed"],
            "duration_sec": r["duration_sec"],
            "sqi": sqi,
            "ppg_features": feat,
            "golden_features": gold,
            "ensemble": ens,
            "validation_status": r["validation"]["status"]
        })

# Group by subject prefix
subject_groups = {}
for r in all_results_global:
    fname = r["filename_no_ext"]
    subj = fname.split("_Win")[0] if "_Win" in fname else fname
    subject_groups.setdefault(subj, []).append(r)

print_section("📝 WRITING COMBINED JSON REPORTS", char="=")
for subj, subj_results in subject_groups.items():
    main_folder = os.path.join(SAVE_ROOT_FIXED, f"{subj}_Filtered")
    additional_folder = os.path.join(main_folder, f"{subj}_Additional")
    os.makedirs(additional_folder, exist_ok=True)
    
    subj_summary_rows = [row for row in summary_rows
                         if row.get("filename", "").startswith(subj)]
    
    subj_report = {
        "subject": subj,
        "processing_date": combined_report_root["processing_date"],
        "mode": combined_report_root["mode"],
        "min_valid_beats_ir": MIN_VALID_BEATS_IR,
        "min_valid_beats_red": MIN_VALID_BEATS_RED,
        "output_root": SAVE_ROOT_FIXED,
        "total_windows": len(subj_results),
        "windows_success": [r["filename"] for r in subj_results if r["status"] == "SUCCESS"],
        "windows_rejected": [
            {"filename": r["filename"],
             "reason": r["rejection_reason"],
             "details": r.get("rejection_details", {})}
            for r in subj_results if r["status"] == "REJECTED"
        ],
        "windows_failed": [
            {"filename": r["filename"], "error": r["error"]}
            for r in subj_results if r["status"] == "FAILED"
        ],
        "summary_table": subj_summary_rows,
        "validation_reports": {
            r["filename"]: r["validation"]
            for r in subj_results if r["status"] == "SUCCESS"
        }
    }
    
    json_path = os.path.join(additional_folder, f"{subj}_Combined_Report.json")
    with open(json_path, "w") as f:
        json.dump(to_json_safe(subj_report), f, indent=4)
    
    n_succ = len(subj_report["windows_success"])
    n_rej = len(subj_report["windows_rejected"])
    n_fail = len(subj_report["windows_failed"])
    print(f"  ✅ {subj}: {len(subj_results)} windows ({n_succ} ok, {n_rej} rejected, {n_fail} failed) → {json_path}")

print_final_summary(all_results_global, SAVE_ROOT_FIXED)
print("\n🎉 Pipeline complete.\n")