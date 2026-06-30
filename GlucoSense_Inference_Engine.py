# ==============================================================================
#  ██████╗ ██╗     ██╗   ██╗ ██████╗ ██████╗ ███████╗███████╗███╗   ██╗███████╗███████╗
# ██╔════╝ ██║     ██║   ██║██╔════╝██╔═══██╗██╔════╝██╔════╝████╗  ██║██╔════╝██╔════╝
# ██║  ███╗██║     ██║   ██║██║     ██║   ██║███████╗█████╗  ██╔██╗ ██║███████╗█████╗
# ██║   ██║██║     ██║   ██║██║     ██║   ██║╚════██║██╔══╝  ██║╚██╗██║╚════██║██╔══╝
# ╚██████╔╝███████╗╚██████╔╝╚██████╗╚██████╔╝███████║███████╗██║ ╚████║███████║███████╗
#  ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝╚══════╝
#
#  GlucoSense Inference Engine
#  End-to-End Non-Invasive Glucose Prediction from PPG Signal
#  ─────────────────────────────────────────────────────────
#  Integrates pipeline stages: Code02 → Code03 → Code04 → Code05
#                               → Code06 → Code08 → Code10 → Code11
#
#  HOW TO RUN:
#    python GlucoSense_Inference_Engine.py
#    Then select Mode 1 (existing CSV) or Mode 2 (ESP32 live acquisition)
#
#  EDIT ONLY THE HYPERPARAMETER SECTION BELOW. Do not edit the functions.
# ==============================================================================


# ==============================================================================
# ⚙️  SECTION A — FILE & OUTPUT PATHS
# ==============================================================================

OUTPUT_ROOT = r"C:\Users\DELL\Documents\GitHub\fyp\Inference_Engine_Outputs"
# Root directory for ALL inference outputs. A timestamped subfolder is
# created automatically inside this directory on every run.

EXISTING_CSV_PATH = r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\02_Verified_Raw"
# Used ONLY in Mode 1 (existing CSV bypass).
# Can be:
#   - A full path to a specific .csv file  → loaded directly, no dialog
#   - A path to a directory               → opens a file-picker dialog

MODEL_PKL_PATH = r"C:\Users\DELL\Documents\GitHub\fyp\08_Results_and_Visualizations\XGBoost_Results_&_Conclusions\XGBoost results & Conclusions 2026-06-26 02-08-33\model\xgboost_glucose_model.pkl"
# Path to the trained XGBoost model .pkl file produced by Code 11.
# Must match the model whose MANUAL_FEATURE_SELECTION was used below.

SCALER_JSON_PATH = r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\10_Robust_Scaled_and_Train_Test_Splitted_Data_Set\Master dataset 24F split scaled 2026-06-21 01-48-33\json\Master dataset 24F split scaled 2026-06-21 01-48-33.json"
# Path to the Code 10 JSON output file that contains:
#   "feature_scaler_parameters": [{"feature":..., "center_median":..., "scale_iqr":...}, ...]
# The scaler formula applied here: X_scaled = (X - center_median) / scale_iqr


# ==============================================================================
# 🧑 SECTION B — SESSION METADATA
# ==============================================================================

SUBJECT_ID = "Subject_X"
# Unique identifier for this measurement session.
# Used in run folder names, JSON output, and the prediction log CSV.

SESSION_NOTES = ""
# Optional free-text notes about this session.
# Examples: "Fasting", "Post-meal 2h", "Exercise 30min ago"
# Saved in session JSON and prediction log.


# ==============================================================================
# 📡 SECTION C — ESP32 DATA ACQUISITION  (Mode 2 only)
# ==============================================================================

SERIAL_PORT = 'COM12'
# Serial port of the ESP32 device.
# Check Device Manager on Windows (e.g., COM3, COM12).
# On Linux/Mac use /dev/ttyUSB0 or similar.

BAUD_RATE = 115200
# Communication baud rate. Must exactly match the value set in the ESP32
# firmware. Common values: 115200 or 921600.

ACQUISITION_WINDOW = 4000
# Number of samples shown in the live rolling plot buffer.
# 4000 samples @ 400 Hz = 10 seconds of visible signal in the plot.


# ==============================================================================
# 🪟 SECTION D — WINDOW SLICER
# ==============================================================================

FS = 400
# Sampling frequency in Hz. Must match the ESP32 firmware setting.
# Used across ALL signal processing and feature extraction stages.

WINDOW_DURATION_SEC = 15
# Duration of each selected signal window in seconds.
# Each "Add Block" click saves exactly this many seconds of signal.
# 15 seconds @ 400 Hz = 6000 samples per window.

VIEW_WINDOW_SEC = 30
# Width of the visible scrolling region displayed in the slicer GUI (seconds).
# Increase for a wider scroll view; decrease to zoom into finer detail.


# ==============================================================================
# 🔧 SECTION E — SIGNAL PROCESSING  (Code 04 parameters)
# ==============================================================================

# ── Pre-processing ────────────────────────────────────────────────────────────
SPIKE_ENABLE = True
# Enable median filter for impulse/spike noise removal BEFORE main filters.
# Recommended: True for MAX30102 sensor.

KERNEL_SIZE = 3
# Window size for the median (spike) filter. Must be an odd integer.
# Typical values: 3 (light), 5 (moderate), 7 (heavy).

INVERT_ENABLE = True
# Flip the signal so systolic peaks point upward.
# Set True for MAX30102 (sensor outputs inverted signal by default).
# Set False if peaks already point up in your raw data.

# ── Low-Pass Filter ───────────────────────────────────────────────────────────
LP_ENABLE = True
# Enable Butterworth Low-Pass Filter to remove high-frequency noise.

LP_CUTOFF = 16.0
# Low-pass cutoff frequency in Hz. Typical range: 10–20 Hz.
# Frequencies above this are attenuated. 16 Hz passes all cardiac harmonics.

LP_ORDER = 4
# Butterworth filter order. Higher = steeper rolloff but more phase distortion.
# Typical: 4. Range: 2–6.

# ── Savitzky-Golay Smoothing ──────────────────────────────────────────────────
SG_ENABLE = False
# Enable Savitzky-Golay polynomial smoothing filter.
# Usually NOT needed when LP filter is enabled. Set True only to add extra smoothing.

SG_WINDOW = 31
# SG filter window length in samples. Must be an odd integer and > SG_POLY.
# Larger window = more smoothing but can distort sharp peaks.

SG_POLY = 3
# SG polynomial order. Typical: 2–4. Must be less than SG_WINDOW.

# ── High-Pass Filter ──────────────────────────────────────────────────────────
HP_ENABLE = True
# Enable Butterworth High-Pass Filter to remove slow baseline drift (DC wander).

HP_CUTOFF = 0.5
# High-pass cutoff frequency in Hz. Typical range: 0.3–1.0 Hz.
# Frequencies below this (slow drift, motion baseline) are removed.

HP_ORDER = 4
# Butterworth filter order for high-pass. Typical: 4.

# ── Normalization ─────────────────────────────────────────────────────────────
NORM_SELECTION = 1
# Signal normalization method applied after filtering:
#   1 = Min-Max normalization (scales to 0–1 range)
#   2 = Z-Score normalization (mean=0, std=1)

# ── Ensemble Construction ─────────────────────────────────────────────────────
ENSEMBLE_TARGET_LEN = 220
# Target length (samples) for resampled individual beats before averaging.
# All detected beats are interpolated to this fixed length for alignment.

MIN_VALID_BEATS_IR = 8
# Minimum number of valid IR-channel beats required per window.
# Windows with fewer valid beats are marked REJECTED and skipped.

MIN_VALID_BEATS_RED = 8
# Minimum number of valid RED-channel beats required per window.
# Windows with fewer valid beats are marked REJECTED and skipped.

# ── Beat Detection Parameters ─────────────────────────────────────────────────
PEAK_MIN_DISTANCE_SEC = 0.40
# Minimum time gap (seconds) between two consecutive systolic peaks.
# 0.40s corresponds to maximum detectable HR of ~150 BPM.
# Increase for slow heart rates; decrease for fast heart rates.

PEAK_PROM_FACTOR = 0.20
# Peak prominence threshold = PEAK_PROM_FACTOR × signal_range.
# Higher = only prominent peaks detected (stricter). Lower = more peaks found.

VALLEY_MIN_DISTANCE_SEC = 0.35
# Minimum time gap (seconds) between two consecutive diastolic valleys.

VALLEY_PROM_FACTOR = 0.10
# Valley prominence threshold relative to signal range.
# Typically lower than PEAK_PROM_FACTOR as valleys are less pronounced.

MIN_FOOT_TO_PEAK_SEC = 0.08
# Minimum allowed duration from pulse foot to systolic peak (seconds).
# Beats shorter than this are rejected as artifacts (too fast to be real).

MAX_FOOT_TO_PEAK_SEC = 0.5
# Maximum allowed duration from foot to systolic peak (seconds).
# Beats longer than this are rejected (abnormally slow systolic upstroke).

MAX_VALLEY_TO_FOOT_SEC = 0.20
# Maximum allowed gap between a detected valley and its refined foot (seconds).
# Keeps foot refinement within a physiologically valid search window.

MAX_FOOT_REL_HEIGHT = 0.20
# Maximum relative height of the foot within a pulse (0=bottom, 1=peak).
# Feet sitting too high on the waveform are rejected as likely misdetections.

MAX_ABS_VPG_AT_FOOT = 0.5
# Maximum absolute value of the 1st derivative (VPG) at the foot location.
# Ensures the foot is near the zero-crossing of the velocity signal.

EDGE_EXCLUSION_SEC = 0.1
# Time margin at signal start/end (seconds) where peak candidates are ignored.
# Prevents edge artifacts from entering beat detection.

MIN_BEAT_DURATION_SEC = 0.35
# Minimum valid beat duration (foot-to-next-foot interval) in seconds.
# Corresponds to maximum HR of ~171 BPM. Beats shorter than this are rejected.

MAX_BEAT_DURATION_SEC = 1.50
# Maximum valid beat duration in seconds. Safety net against missed peaks.
# Corresponds to minimum HR of ~40 BPM.

BEAT_DURATION_MEDIAN_TOLERANCE = 1.35
# Adaptive outlier filter: reject beats whose duration deviates from the median.
# A beat is rejected if duration > median × tolerance OR < median / tolerance.
# 1.35 = ±35% tolerance. Set 0 to disable this adaptive filter.

MAIN_PEAK_SEARCH_WINDOW_SEC = 0.3
# Search window duration (seconds) after each foot to locate the main systolic peak.

MAIN_PEAK_MIN_DELAY_SEC = 0.02
# Minimum delay (seconds) after foot before searching for the main systolic peak.
# Prevents the foot itself from being identified as the peak.

START_INCOMPLETE_MARGIN_SEC = 0.10
# Margin (seconds) at the start of the signal for detecting incomplete beats.
# Beats starting within this margin are rejected as potentially truncated.

END_INCOMPLETE_MARGIN_SEC = 0.01
# Margin (seconds) at the end of the signal for detecting incomplete beats.

VERBOSE_REJECTION = False
# Print detailed rejection info for each rejected beat to the terminal.
# Set True for debugging beat detection issues.

VERBOSE_BEAT_DETECTION_DIAG = False
# Print detailed peak/valley/candidate diagnostics.
# Set True when tuning PEAK_MIN_DISTANCE_SEC, PEAK_PROM_FACTOR, etc.

# ── Signal Quality Index (SQI) Limits ─────────────────────────────────────────
SQI_ENABLE = False
# If True, windows failing the SQI limits below are REJECTED.
# If False, SQI is still calculated and logged, but windows are NOT rejected.

SQI_LIMITS = {
    'SKEWNESS_MIN': 0.0,  'SKEWNESS_MAX': 2.5,   # Signal shape symmetry (left=+)
    'KURTOSIS_MIN': 1.5,  'KURTOSIS_MAX': 7.0,   # Peak sharpness
    'PI_MIN':       0.1,  'PI_MAX':       10.0,  # Perfusion Index (%)
    'SNR_MIN_DB':   5.0,  'SNR_MAX_DB':   25.0,  # Signal-to-Noise Ratio (dB)
    'ZCR_MIN':      1.0,  'ZCR_MAX':      4.0    # Zero Crossing Rate (Hz)
}
# Windows whose IR or RED channel SQI falls outside these limits are REJECTED.
# Tune these based on your sensor and subject population.


# ==============================================================================
# 🔬 SECTION F — FEATURE ENGINEERING  (Code 08 parameters)
# ==============================================================================

IR_BASE_FEATURES = [
    "IR_Skewness",
    "IR_Kurtosis",
    "IR_Shannon Entropy",
    "IR_Spectral Entropy",
    "IR_pulse width",
    "IR_PPI",
    "IR_systolic amplitude",
    "IR_BPM",
    "IR_HRV",
    "IR_TEO Mean",
    "IR_TEO std dev",
    "IR_1st_Derivative_Mean",
    "IR_2nd_Derivative_Mean",
    "IR_2nd_Derivative_Skewness",
    "IR_Harmonic ratio",
    "IR_Rise time",
    "IR_Decay time",
    "IR_Dicrotic notch",
]
# The 18 IR-channel features passed directly to the 24-feature dataset.
# Do NOT add or remove items — list must match the training pipeline exactly.

ENGINEERED_FEATURES = [
    # (output_name,              operation,    operand_1,                 operand_2)
    ("Ratio_systolic_amplitude", "ratio",      "Red_systolic amplitude",  "IR_systolic amplitude"),
    ("Ratio_TEO_Mean",           "ratio",      "Red_TEO Mean",            "IR_TEO Mean"),
    ("Diff_2nd_Derivative_Mean", "difference", "Red_2nd_Derivative_Mean", "IR_2nd_Derivative_Mean"),
    ("Diff_Spectral_Entropy",    "difference", "Red_Spectral Entropy",    "IR_Spectral Entropy"),
    ("Diff_Dicrotic_notch",      "difference", "Red_Dicrotic notch",      "IR_Dicrotic notch"),
]
# 5 engineered features combining RED and IR channels.
# "ratio" = operand_1 / operand_2   |   "difference" = operand_1 - operand_2
# Do NOT change unless the model was retrained with a different feature set.


# ==============================================================================
# 🎚️ SECTION G — FEATURE SELECTION  (Code 11 parameters)
# ==============================================================================

MANUAL_FEATURE_SELECTION = {
    # 1 = USE this feature for prediction   |   0 = DROP it
    # Must match EXACTLY the feature selection used when the model was trained.
    # Total active features (=1) must equal the model's expected input size.
    "IR_Skewness":                  1,
    "IR_Kurtosis":                  0,
    "IR_Shannon Entropy":           0,
    "IR_Spectral Entropy":          1,
    "IR_pulse width":               1,
    "IR_PPI":                       1,
    "IR_systolic amplitude":        0,
    "IR_BPM":                       0,
    "IR_HRV":                       1,
    "IR_TEO Mean":                  1,
    "IR_TEO std dev":               0,
    "IR_1st_Derivative_Mean":       1,
    "IR_2nd_Derivative_Mean":       1,
    "IR_2nd_Derivative_Skewness":   1,
    "IR_Harmonic ratio":            0,
    "IR_Rise time":                 0,
    "IR_Decay time":                1,
    "IR_Dicrotic notch":            1,
    "Ensemble ratio":               1,
    "Ratio_TEO_Mean":               0,
    "Ratio_systolic_amplitude":     0,
    "Diff_Spectral_Entropy":        1,
    "Diff_2nd_Derivative_Mean":     1,
    "Diff_Dicrotic_notch":          1,
}
# All 24 features must be listed. Count of 1s here = 15 (matches the trained model).

# ==============================================================================
# END OF HYPERPARAMETERS — Do not modify below this line unless you know
#                          exactly what you are doing.
# ==============================================================================


# ──────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import sys
import os
import csv
import json
import pickle
import uuid
import traceback
from datetime import datetime
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.fft import fft, fftfreq, rfft
from scipy.signal import find_peaks, welch, savgol_filter, peak_widths
from scipy.stats import skew, kurtosis

import matplotlib
try:
    import PyQt5
    matplotlib.use("Qt5Agg")
except ImportError:
    matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tkinter import filedialog, Tk


# ──────────────────────────────────────────────────────────────────────────────
#  GLOBAL RUN STATE  (populated at runtime — do not edit)
# ──────────────────────────────────────────────────────────────────────────────
_RUN_ID         = None   # timestamped unique run identifier
_RUN_DIR        = None   # Path to this run's output subfolder
_RAW_CSV_PATH   = None   # Path to the raw CSV used in this run
_INPUT_MODE     = None   # "Mode1_ExistingCSV" | "Mode2_ESP32"


# ==============================================================================
#  UTILITY HELPERS
# ==============================================================================

def _make_run_id():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"Run_{SUBJECT_ID}_{ts}"


def _to_json_safe(obj):
    """Recursively converts numpy/pandas objects to JSON-serialisable types."""
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.integer):  return int(obj)
    if isinstance(obj, np.floating): return float(obj)
    if isinstance(obj, np.bool_):    return bool(obj)
    if isinstance(obj, np.ndarray):  return obj.tolist()
    try:
        if pd.isna(obj): return None
    except Exception:
        pass
    if not isinstance(obj, (str, int, float, bool, type(None))):
        return str(obj)
    return obj


def _banner(title, width=70, char="="):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def _safe_ratio(num, den, default=np.nan):
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return default
    return num / den


# ==============================================================================
#  STAGE 0 — STARTUP MENU
# ==============================================================================

def run_startup_menu():
    """Prints the startup banner and asks user to choose input mode."""
    global _RUN_ID, _RUN_DIR, _INPUT_MODE

    print("\n" + "═" * 70)
    print("  ██████╗ ██╗     ██╗   ██╗ ██████╗ ██████╗ ███████╗███████╗")
    print("  ██╔════╝ ██║     ██║   ██║██╔════╝██╔═══██╗██╔════╝██╔════╝")
    print("  ██║  ███╗██║     ██║   ██║██║     ██║   ██║███████╗█████╗")
    print("  ██║   ██║██║     ██║   ██║██║     ██║   ██║╚════██║██╔══╝")
    print("  ╚██████╔╝███████╗╚██████╔╝╚██████╗╚██████╔╝███████║███████╗")
    print("   ╚═════╝ ╚══════╝ ╚═════╝  ╚═════╝ ╚═════╝ ╚══════╝╚══════╝")
    print("  GlucoSense Inference Engine  —  Non-Invasive Glucose Prediction")
    print("═" * 70)
    print(f"\n  Subject ID   : {SUBJECT_ID}")
    print(f"  Session Notes: {SESSION_NOTES if SESSION_NOTES else '(none)'}")
    print(f"  Output Root  : {OUTPUT_ROOT}")
    print("\n  ─────────────────────────────────────────────────────────────")
    print("  Select Input Mode:")
    print("    [1]  Load EXISTING raw CSV  (from 02_Verified_Raw folder)")
    print("    [2]  LIVE acquisition from ESP32  (Code 02 data logger)")
    print("  ─────────────────────────────────────────────────────────────")

    while True:
        choice = input("  Enter choice [1 or 2]: ").strip()
        if choice in ("1", "2"):
            break
        print("  ❌ Invalid. Please enter 1 or 2.")

    _INPUT_MODE = "Mode1_ExistingCSV" if choice == "1" else "Mode2_ESP32"
    _RUN_ID     = _make_run_id()
    _RUN_DIR    = Path(OUTPUT_ROOT) / _RUN_ID
    _RUN_DIR.mkdir(parents=True, exist_ok=True)
    (_RUN_DIR / "windowed").mkdir(exist_ok=True)

    print(f"\n  ✅ Mode selected : {_INPUT_MODE}")
    print(f"  📁 Run folder    : {_RUN_DIR}")
    return choice


# ==============================================================================
#  STAGE 1 — DATA ACQUISITION
# ==============================================================================

# ── Mode 1: Load existing CSV ─────────────────────────────────────────────────

def _load_existing_csv():
    """Returns the path to the selected raw CSV (Mode 1)."""
    global _RAW_CSV_PATH
    src = EXISTING_CSV_PATH.strip()

    if os.path.isfile(src) and src.lower().endswith(".csv"):
        print(f"  📄 Direct file detected: {os.path.basename(src)}")
        _RAW_CSV_PATH = src
        return src

    if os.path.isdir(src):
        print("  📂 Opening File Dialog …")
        root = Tk(); root.withdraw(); root.attributes("-topmost", True)
        fp = filedialog.askopenfilename(
            initialdir=src, title="Select Raw CSV",
            filetypes=[("CSV files", "*.csv")]
        )
        root.destroy()
        if not fp:
            print("  ❌ No file selected. Exiting.")
            sys.exit(0)
        _RAW_CSV_PATH = fp
        return fp

    print(f"  ❌ EXISTING_CSV_PATH not found: {src}")
    sys.exit(1)


# ── Mode 2: ESP32 Live Acquisition ───────────────────────────────────────────

def _run_esp32_acquisition():
    """Launches live PyQtGraph logger (Code 02 logic). Returns path to saved CSV."""
    global _RAW_CSV_PATH

    # Late import so the script doesn't crash when PyQt5 is not installed
    # and the user chose Mode 1.
    try:
        import serial
        import pyqtgraph as pg
        from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout,
                                      QWidget, QMessageBox)
        from PyQt5.QtCore import QTimer
    except ImportError as e:
        print(f"\n  ❌ Mode 2 requires PyQt5 and pyserial: {e}")
        print("     Install with: pip install PyQt5 pyserial pyqtgraph")
        sys.exit(1)

    # Ask for session filename
    print(f"\n  📂 Raw CSV will be saved to: {_RUN_DIR}")
    session_name = input("  👉 Enter session filename (e.g., Subject_X_raw): ").strip()
    if not session_name:
        session_name = f"{SUBJECT_ID}_raw"
    if not session_name.lower().endswith(".csv"):
        session_name += ".csv"

    raw_dir = _RUN_DIR / "raw"
    raw_dir.mkdir(exist_ok=True)
    out_path = raw_dir / session_name

    if out_path.exists():
        ow = input(f"  ⚠️  '{session_name}' already exists. Overwrite? (y/n): ").strip().lower()
        if ow != "y":
            print("  ❌ Cancelled.")
            sys.exit(0)

    class _LivePlotter(QMainWindow):
        def __init__(self, save_path):
            super().__init__()
            self._save_path = save_path
            self._ir_buf    = np.zeros(ACQUISITION_WINDOW)
            self._red_buf   = np.zeros(ACQUISITION_WINDOW)
            self._csv_file  = open(save_path, mode='w', newline='')
            self._writer    = csv.writer(self._csv_file)
            self._writer.writerow(["Timestamp", "IR", "RED"])
            self._init_ui()
            self._init_serial()

        def _init_ui(self):
            self.setWindowTitle(f"GlucoSense — Recording: {session_name} | {SERIAL_PORT}")
            self.resize(1200, 700)
            cw = QWidget(); self.setCentralWidget(cw)
            layout = QVBoxLayout(cw)
            pg.setConfigOption('background', 'w'); pg.setConfigOption('foreground', 'k')
            self._pw = pg.GraphicsLayoutWidget(); layout.addWidget(self._pw)
            p1 = self._pw.addPlot(title="IR Signal (Raw)")
            p1.setLabel('left', 'Amplitude'); p1.showGrid(x=True, y=True, alpha=0.3)
            self._c_ir  = p1.plot(pen=pg.mkPen('#0072bd', width=2))
            self._pw.nextRow()
            p2 = self._pw.addPlot(title="RED Signal (Raw)")
            p2.setLabel('left', 'Amplitude'); p2.setLabel('bottom', 'Samples')
            p2.showGrid(x=True, y=True, alpha=0.3); p2.setXLink(p1)
            self._c_red = p2.plot(pen=pg.mkPen('#d95319', width=2))
            self._timer = QTimer()
            self._timer.timeout.connect(self._update)
            self._timer.start(16)

        def _init_serial(self):
            try:
                self._ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
                self._ser.flushInput()
                print(f"  ✅ Serial connected: {SERIAL_PORT} @ {BAUD_RATE}")
            except Exception as e:
                QMessageBox.critical(self, "Serial Error",
                    f"Cannot open {SERIAL_PORT}.\nError: {e}")
                sys.exit(1)

        def _update(self):
            if not self._ser.is_open: return
            try:
                while self._ser.in_waiting:
                    line = self._ser.readline().decode('utf-8', errors='ignore').strip()
                    if not line: continue
                    parts = line.split(',')
                    if len(parts) == 2:
                        try:
                            ir_val  = float(parts[0])
                            red_val = float(parts[1])
                            t_now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            self._writer.writerow([t_now, ir_val, red_val])
                            self._ir_buf[:-1]  = self._ir_buf[1:];  self._ir_buf[-1]  = ir_val
                            self._red_buf[:-1] = self._red_buf[1:]; self._red_buf[-1] = red_val
                        except ValueError:
                            pass
                self._c_ir.setData(self._ir_buf)
                self._c_red.setData(self._red_buf)
            except Exception as e:
                print(f"  Serial loop error: {e}")

        def closeEvent(self, event):
            if self._ser.is_open: self._ser.close()
            if self._csv_file:    self._csv_file.close()
            print(f"\n  💾 Recording saved: {self._save_path}")
            event.accept()

    app = QApplication(sys.argv)
    win = _LivePlotter(str(out_path))
    win.show()
    print("\n  ℹ️  Close the plot window to stop recording and continue pipeline.")
    app.exec_()

    _RAW_CSV_PATH = str(out_path)
    return str(out_path)


def stage1_acquire_data(mode_choice):
    _banner("STAGE 1 — Data Acquisition")
    if mode_choice == "1":
        raw_path = _load_existing_csv()
    else:
        raw_path = _run_esp32_acquisition()

    print(f"  ✅ Raw CSV: {raw_path}")
    return raw_path


# ==============================================================================
#  STAGE 2 — MANUAL WINDOW SLICER  (Code 03 logic)
# ==============================================================================

def _load_csv_with_aliases(file_path):
    df = pd.read_csv(file_path)
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.strip().lower(): c.strip() for c in df.columns}
    rename  = {}
    for alias in ("ir_value", "ir", "infrared"):
        if alias in col_map:
            rename[col_map[alias]] = "IR_Value"; break
    for alias in ("red_value", "red"):
        if alias in col_map:
            rename[col_map[alias]] = "Red_Value"; break
    if rename:
        df = df.rename(columns=rename)
    for req in ("IR_Value", "Red_Value"):
        if req not in df.columns:
            raise ValueError(f"Required column '{req}' not found. Columns: {list(df.columns)}")
    return df


def stage2_window_slicer(raw_csv_path):
    _banner("STAGE 2 — Manual Window Slicer")

    df = _load_csv_with_aliases(raw_csv_path)
    total_duration = len(df) / FS
    print(f"  ✅ Loaded: {os.path.basename(raw_csv_path)}")
    print(f"     Duration: {total_duration:.2f}s  |  Samples: {len(df)}  |  FS={FS} Hz")
    print(f"\n  🧭 Controls:")
    print("     • SLIDER     — scroll through the signal")
    print("     • CLICK plot — place blue selection window start")
    print("     • Add Block  — save the current blue window")
    print("     • Undo Last  — remove the last saved window")
    print("     • Done       — finish and proceed to processing")

    win_sec   = float(WINDOW_DURATION_SEC)
    view_sec  = float(VIEW_WINDOW_SEC)
    if view_sec <= 2: view_sec = 10.0
    max_scroll = max(0.0, total_duration - view_sec)

    saved_windows = []

    def clamp(s):
        return float(np.clip(s, 0.0, max(0.0, total_duration - win_sec)))

    def s2i(t):
        return int(np.clip(t * FS, 0, len(df)))

    def get_seg(t0, t1):
        a, b = s2i(t0), s2i(t1)
        seg  = df.iloc[a:b].copy()
        tt   = np.linspace(t0, t1, max(1, len(seg)))
        return tt, seg

    def autoscale(ax, y):
        y = y[np.isfinite(y)]
        if len(y) == 0: return
        lo, hi = float(np.nanmin(y)), float(np.nanmax(y))
        if hi > lo:
            p = 0.07 * (hi - lo); ax.set_ylim(lo - p, hi + p)
        else:
            ax.set_ylim(lo - 1, hi + 1)

    fig   = plt.figure(figsize=(13, 7))
    gs    = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.08)
    ax_red = fig.add_subplot(gs[0])
    ax_ir  = fig.add_subplot(gs[1], sharex=ax_red)
    fig.suptitle(f"GlucoSense Window Slicer  —  {os.path.basename(raw_csv_path)}")
    ax_red.set_ylabel("RED Amplitude"); ax_ir.set_ylabel("IR Amplitude")
    ax_ir.set_xlabel("Time (seconds)")
    ax_red.grid(True); ax_ir.grid(True)

    t0_v, t1_v = 0.0, min(total_duration, view_sec)
    tt, seg = get_seg(t0_v, t1_v)
    (line_red,) = ax_red.plot(tt, seg["Red_Value"].values, color="red",  label="RED", alpha=0.85)
    (line_ir,)  = ax_ir.plot( tt, seg["IR_Value"].values,  color="blue", label="IR",  alpha=0.85)
    ax_red.legend(loc="upper right"); ax_ir.legend(loc="upper right")

    scroll_start = 0.0
    win_start    = clamp(0.0)
    win_end      = win_start + win_sec
    patch_r = ax_red.axvspan(win_start, win_end, alpha=0.18)
    patch_i = ax_ir.axvspan( win_start, win_end, alpha=0.18)
    info    = ax_ir.text(0.01, 0.03, "", transform=ax_ir.transAxes, fontsize=10,
                         bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))

    def redraw_span():
        nonlocal patch_r, patch_i
        we = win_start + win_sec
        try: patch_r.remove()
        except Exception: pass
        try: patch_i.remove()
        except Exception: pass
        patch_r = ax_red.axvspan(win_start, we, alpha=0.18)
        patch_i = ax_ir.axvspan( win_start, we, alpha=0.18)
        info.set_text(f"Window: {win_start:.3f}s → {we:.3f}s  |  Saved: {len(saved_windows)}")
        fig.canvas.draw_idle()

    ax_sl = fig.add_axes([0.10, 0.12, 0.80, 0.035])
    slider = Slider(ax_sl, "Scroll (sec)", 0.0,
                    max_scroll if max_scroll > 0 else 0.001,
                    valinit=0.0, valstep=max(1.0 / FS, 0.01))

    def on_slider(val):
        nonlocal scroll_start, win_start
        scroll_start = float(val)
        scroll_end   = min(total_duration, scroll_start + view_sec)
        tt2, seg2 = get_seg(scroll_start, scroll_end)
        line_red.set_data(tt2, seg2["Red_Value"].values)
        line_ir.set_data( tt2, seg2["IR_Value"].values)
        ax_ir.set_xlim(tt2[0], tt2[-1] if len(tt2) > 1 else scroll_end)
        autoscale(ax_red, seg2["Red_Value"].values)
        autoscale(ax_ir,  seg2["IR_Value"].values)
        if (win_start + win_sec) < scroll_start or win_start > scroll_end:
            win_start = clamp(scroll_start)
        redraw_span()

    slider.on_changed(on_slider)

    def on_click(event):
        nonlocal win_start
        if event.inaxes not in (ax_red, ax_ir): return
        if event.xdata is None: return
        win_start = clamp(float(event.xdata))
        redraw_span()

    fig.canvas.mpl_connect("button_press_event", on_click)

    ax_add  = fig.add_axes([0.10, 0.03, 0.18, 0.06])
    ax_undo = fig.add_axes([0.31, 0.03, 0.18, 0.06])
    ax_done = fig.add_axes([0.74, 0.03, 0.20, 0.06])
    btn_add  = Button(ax_add,  "Add Block")
    btn_undo = Button(ax_undo, "Undo Last")
    btn_done = Button(ax_done, "Done  ✓")

    def add_block(_):
        nonlocal win_start
        s = float(win_start)
        e = s + win_sec
        if e > total_duration:
            s = clamp(total_duration - win_sec); e = s + win_sec
            win_start = s; redraw_span()
        saved_windows.append((s, e))
        print(f"  ✅ Saved Window: {s:.3f}s → {e:.3f}s  [{len(saved_windows)} total]")
        redraw_span()

    def undo_last(_):
        if not saved_windows:
            print("  ❌ Nothing to undo."); return
        removed = saved_windows.pop()
        print(f"  ↩️  Removed: {removed[0]:.3f}s → {removed[1]:.3f}s")
        redraw_span()

    def done(_):
        plt.close(fig)

    btn_add.on_clicked(add_block)
    btn_undo.on_clicked(undo_last)
    btn_done.on_clicked(done)

    autoscale(ax_red, seg["Red_Value"].values)
    autoscale(ax_ir,  seg["IR_Value"].values)
    redraw_span()
    plt.show(block=True)

    if not saved_windows:
        print("  ❌ No windows selected. Exiting.")
        sys.exit(0)

    print(f"\n  📋 Selected {len(saved_windows)} window(s):")
    for k, (s, e) in enumerate(saved_windows):
        print(f"     {k+1}. {s:.3f}s → {e:.3f}s")

    # Save individual window CSVs
    win_dir  = _RUN_DIR / "windowed"
    base     = Path(raw_csv_path).stem
    win_paths = []
    print(f"\n  💾 Saving windows to: {win_dir}")
    for idx, (s, e) in enumerate(saved_windows):
        si, ei = int(s * FS), int(e * FS)
        chunk  = df.iloc[si:ei].copy()
        fname  = f"{base}_Win{idx}.csv"
        fpath  = win_dir / fname
        chunk.to_csv(fpath, index=False)
        win_paths.append(str(fpath))
        print(f"     [{idx}] {s:.3f}s → {e:.3f}s  →  {fname}")

    return win_paths, saved_windows


# ==============================================================================
#  STAGE 3 — SIGNAL PROCESSING  (Code 04 logic — per window, no plots)
# ==============================================================================

def _apply_norm(data, sel=1):
    data = np.asarray(data, dtype=float)
    if data.size == 0: return data
    if sel == 1:
        den = np.max(data) - np.min(data)
        return np.zeros_like(data) if den == 0 else (data - np.min(data)) / den
    if sel == 2:
        sd = np.std(data)
        return np.zeros_like(data) if sd == 0 else (data - np.mean(data)) / sd
    return data


def _safe_savgol(x, win, poly=3):
    x = np.asarray(x); n = len(x)
    if n < 5: return x.copy()
    win = int(win)
    if win % 2 == 0: win += 1
    if win >= n: win = n - 1 if (n - 1) % 2 == 1 else n - 2
    if win < 5:  win = 5
    if win >= n or win <= poly: return x.copy()
    poly = min(poly, win - 2)
    return savgol_filter(x, window_length=win, polyorder=poly)


def _resample_1d(sig, target_len):
    x_old = np.linspace(0, 1, len(sig))
    x_new = np.linspace(0, 1, target_len)
    return np.interp(x_new, x_old, sig)


def _calc_snr(sig, fs):
    freqs, psd = signal.periodogram(sig, fs)
    sm = (freqs >= 0.5) & (freqs <= 5.0)
    sp = np.sum(psd[sm]); np_ = np.sum(psd[~sm])
    return 10 * np.log10(sp / np_) if np_ > 0 else 100.0


def _calc_zcr(sig, fs):
    c = sig - np.mean(sig)
    return np.diff(np.signbit(c)).sum() / (len(sig) / fs)


def _sqi_check(norm_sig, raw_sig, fs):
    """Returns SQI dict and overall PASS/FAIL for one channel."""
    sk = float(skew(norm_sig))
    ku = float(kurtosis(norm_sig))
    ac = float(np.max(norm_sig) - np.min(norm_sig))
    dc = float(np.mean(raw_sig))
    pi = (ac / dc) * 100 if dc != 0 else 0.0
    zcr_ = float(_calc_zcr(norm_sig, fs))
    snr_ = float(_calc_snr(norm_sig, fs))

    def st(v, lo, hi): return "PASS" if lo <= v <= hi else "FAIL"

    metrics = {
        "skewness": sk,  "skewness_status": st(sk,  SQI_LIMITS['SKEWNESS_MIN'], SQI_LIMITS['SKEWNESS_MAX']),
        "kurtosis": ku,  "kurtosis_status": st(ku,  SQI_LIMITS['KURTOSIS_MIN'], SQI_LIMITS['KURTOSIS_MAX']),
        "pi_pct":   pi,  "pi_status":       st(pi,  SQI_LIMITS['PI_MIN'],       SQI_LIMITS['PI_MAX']),
        "zcr_hz":   zcr_,"zcr_status":      st(zcr_,SQI_LIMITS['ZCR_MIN'],      SQI_LIMITS['ZCR_MAX']),
        "snr_db":   snr_,"snr_status":      st(snr_,SQI_LIMITS['SNR_MIN_DB'],   SQI_LIMITS['SNR_MAX_DB']),
    }
    overall = all(v == "PASS" for k, v in metrics.items() if k.endswith("_status"))
    return metrics, overall


# ── Beat detection helpers (ported from Code 04) ──────────────────────────────

def _compute_deriv(sig, fs):
    s = _safe_savgol(sig, max(7, int(0.05 * fs)), poly=3)
    vpg = np.gradient(s) * fs
    vpg = _safe_savgol(vpg, max(5, int(0.03 * fs)), poly=2)
    sd  = np.gradient(vpg) * fs
    sd  = _safe_savgol(sd,  max(5, int(0.03 * fs)), poly=2)
    return s, vpg, sd


def _find_last_zc_before_max(vpg_seg, idx_seg):
    candidates = []
    for i in range(len(vpg_seg) - 1):
        if vpg_seg[i] < 0 and vpg_seg[i+1] >= 0:
            candidates.append(int(idx_seg[i] if abs(vpg_seg[i]) <= abs(vpg_seg[i+1]) else idx_seg[i+1]))
    return int(candidates[-1]) if candidates else None


def _refine_foot(valley_idx, peak_idx, smooth_sig, vpg, sdppg, fs):
    if peak_idx <= valley_idx + 3: return int(valley_idx)
    seg_idx = np.arange(valley_idx, peak_idx + 1)
    if len(seg_idx) < 6: return int(valley_idx)
    seg_vpg = vpg[seg_idx]
    i_vpgmax = int(seg_idx[int(np.argmax(seg_vpg))])
    if i_vpgmax <= valley_idx + 1: return int(valley_idx)
    s_idx = np.arange(valley_idx, i_vpgmax + 1)
    s_vpg = vpg[s_idx]; s_sd = sdppg[s_idx]
    if len(s_idx) < 3: return int(valley_idx)
    fz = _find_last_zc_before_max(s_vpg, s_idx)
    if fz is None:
        e2 = max(3, int(0.6 * len(s_idx)))
        fz = int(s_idx[:e2][np.argmin(np.abs(s_vpg[:e2]))])
    foot = int(fz)
    sdp  = int(s_idx[np.argmax(s_sd)])
    if foot > sdp:
        lp = s_idx[s_idx <= sdp]
        if len(lp) > 0:
            foot = int(lp[np.argmin(np.abs(vpg[lp]))])
    foot = max(valley_idx, min(foot, i_vpgmax - 1))
    return int(foot)


def _select_main_peak(cand_peaks, foot_idx, fs):
    mn = foot_idx + int(MAIN_PEAK_MIN_DELAY_SEC * fs)
    mx = foot_idx + int(MAIN_PEAK_SEARCH_WINDOW_SEC * fs)
    valid = cand_peaks[(cand_peaks >= mn) & (cand_peaks <= mx)]
    return int(valid[0]) if len(valid) > 0 else None


def _detect_beats(sig_data, fs):
    smooth, vpg, sdppg = _compute_deriv(sig_data, fs)
    sig_range = float(np.max(smooth) - np.min(smooth)) or 1.0
    pk_thr = max(0.015, sig_range * PEAK_PROM_FACTOR)
    vl_thr = max(0.010, sig_range * VALLEY_PROM_FACTOR)
    peaks,   _ = find_peaks(smooth, distance=int(PEAK_MIN_DISTANCE_SEC   * fs), prominence=pk_thr)
    valleys, _ = find_peaks(-smooth, distance=int(VALLEY_MIN_DISTANCE_SEC * fs), prominence=vl_thr)

    candidate_pairs, rej_cands = [], []
    edge_m = int(EDGE_EXCLUSION_SEC * fs)
    mf_s   = int(MIN_FOOT_TO_PEAK_SEC * fs)
    n      = len(smooth)

    for p in peaks:
        p = int(p)
        if p < edge_m or p > n - edge_m: continue
        lv = valleys[valleys < p]
        if len(lv) == 0: continue
        vi = int(lv[-1])
        if (p - vi) < mf_s: continue
        rf = _refine_foot(vi, p, smooth, vpg, sdppg, fs)
        # Ambiguity / validity checks
        ftp = (p - rf) / fs
        if not (MIN_FOOT_TO_PEAK_SEC <= ftp <= MAX_FOOT_TO_PEAK_SEC): continue
        vtf = (rf - vi) / fs
        if vtf > MAX_VALLEY_TO_FOOT_SEC: continue
        pulse_amp = smooth[p] - smooth[vi]
        if pulse_amp <= 0: continue
        rel_h = (smooth[rf] - smooth[vi]) / pulse_amp
        if rel_h > MAX_FOOT_REL_HEIGHT: continue
        if abs(vpg[rf]) > MAX_ABS_VPG_AT_FOOT: continue
        amp = smooth[p] - smooth[rf]
        if amp <= max(0.01, 0.02 * np.std(smooth)): continue
        candidate_pairs.append({"peak": p, "foot": rf, "valley": vi})

    if not candidate_pairs:
        return [], {}, smooth, vpg, sdppg

    feet = []
    seen = set()
    for cp in candidate_pairs:
        f = int(cp["foot"])
        if f not in seen:
            feet.append(f); seen.add(f)
    feet = np.array(sorted(feet), dtype=int)

    beats, beat_info = [], []
    sig_len = len(smooth)
    end_m   = int(END_INCOMPLETE_MARGIN_SEC   * fs)
    sta_m   = int(START_INCOMPLETE_MARGIN_SEC * fs)

    for i in range(len(feet) - 1):
        f1, f2 = int(feet[i]), int(feet[i+1])
        cps = peaks[(peaks > f1) & (peaks < f2)]
        if len(cps) == 0: continue
        p = _select_main_peak(cps, f1, fs)
        if p is None: continue
        if f1 < sta_m or p > sig_len - end_m or f2 > sig_len - end_m: continue
        bd  = (f2 - f1) / fs
        ftp = (p  - f1) / fs
        amp = smooth[p] - smooth[f1]
        ups = np.max(vpg[f1:p+1]) if p > f1 else 0.0
        if not (MIN_BEAT_DURATION_SEC <= bd <= MAX_BEAT_DURATION_SEC): continue
        if not (MIN_FOOT_TO_PEAK_SEC <= ftp <= MAX_FOOT_TO_PEAK_SEC): continue
        if amp <= max(0.01, 0.02 * np.std(smooth)): continue
        if ups <= max(0.05, 0.05 * np.std(vpg)): continue
        beat = sig_data[f1:f2+1]
        if len(beat) <= 10: continue
        beats.append(beat)
        beat_info.append({"foot": f1, "peak": p, "next_foot": f2,
                           "beat_duration": float(bd), "amplitude": float(amp)})

    # Adaptive duration filter
    if BEAT_DURATION_MEDIAN_TOLERANCE > 0 and len(beat_info) >= 3:
        durs = np.array([b["beat_duration"] for b in beat_info])
        md   = float(np.median(durs))
        up   = md * BEAT_DURATION_MEDIAN_TOLERANCE
        lo   = md / BEAT_DURATION_MEDIAN_TOLERANCE
        beats     = [b for b, bi in zip(beats, beat_info) if lo <= bi["beat_duration"] <= up]
        beat_info = [bi for bi in beat_info if lo <= bi["beat_duration"] <= up]

    return beats, beat_info, smooth, vpg, sdppg


def _build_ensemble(beats, target_len):
    if len(beats) == 0: return None, None, None, None, None
    beats_rs = np.array([_resample_1d(b, target_len) for b in beats])
    # Align by VPG peak
    avg_dur  = float(np.median([len(b) / FS for b in beats]))
    fs_eff   = target_len / avg_dur if avg_dur > 0 else target_len
    vpg_all  = np.gradient(beats_rs, axis=1) * fs_eff
    vpg_all  = np.array([_safe_savgol(v, max(5, int(0.03 * fs_eff)), poly=2) for v in vpg_all])
    ref_idx  = int(np.argmax(np.mean(vpg_all, axis=0)))
    aligned  = []
    for i, b in enumerate(beats_rs):
        vi = int(np.argmax(vpg_all[i]))
        sh = ref_idx - vi
        s  = np.roll(b, sh)
        if sh > 0:  s[:sh]  = s[sh]
        elif sh < 0: s[sh:] = s[sh-1]
        aligned.append(s)
    aligned = np.array(aligned)
    avg_wave = np.mean(aligned, axis=0)
    # Derivatives of ensemble average
    sm2 = _safe_savgol(avg_wave, max(7, int(0.05 * fs_eff)), poly=3)
    vpg = np.gradient(sm2) * fs_eff
    vpg = _safe_savgol(vpg, max(5, int(0.03 * fs_eff)), poly=2)
    sdp = np.gradient(vpg) * fs_eff
    sdp = _safe_savgol(sdp, max(5, int(0.03 * fs_eff)), poly=2)
    t_ens = np.linspace(0, avg_dur, target_len)
    return avg_wave, vpg, sdp, t_ens, fs_eff


def _process_one_window(win_csv_path, win_idx):
    """
    Runs full Code-04-equivalent processing on one windowed CSV.
    Returns a dict with status = 'ACCEPTED' | 'REJECTED' and all arrays needed
    for feature extraction.
    """
    df = pd.read_csv(win_csv_path)
    df.columns = [c.strip().upper() for c in df.columns]
    rename_map = {'RED_VALUE':'RED','IR_VALUE':'IR','RED VALUE':'RED','IR VALUE':'IR'}
    df = df.rename(columns=rename_map)
    for col in ('RED', 'IR'):
        if col not in df.columns:
            return {"status": "REJECTED", "reason": f"Missing column '{col}'",
                    "window_idx": win_idx}

    raw_ir  = df['IR'].values.astype(float)
    raw_red = df['RED'].values.astype(float)
    if len(raw_ir) < 50:
        return {"status": "REJECTED", "reason": "Window too short (<50 samples)",
                "window_idx": win_idx}

    # Spike removal
    ir  = signal.medfilt(raw_ir,  KERNEL_SIZE) if SPIKE_ENABLE else raw_ir.copy()
    red = signal.medfilt(raw_red, KERNEL_SIZE) if SPIKE_ENABLE else raw_red.copy()

    # Inversion
    if INVERT_ENABLE:
        ir  = -ir
        red = -red

    # LP filter
    if LP_ENABLE:
        nyq = 0.5 * FS
        sos = signal.butter(LP_ORDER, LP_CUTOFF / nyq, btype='low', output='sos')
        ir  = signal.sosfiltfilt(sos, ir)
        red = signal.sosfiltfilt(sos, red)

    # SG smoothing
    if SG_ENABLE:
        sw = int(SG_WINDOW) | 1  # ensure odd
        if sw <= SG_POLY: sw = SG_POLY + 2 | 1
        ir  = signal.savgol_filter(ir,  sw, SG_POLY)
        red = signal.savgol_filter(red, sw, SG_POLY)

    # HP filter
    if HP_ENABLE:
        nyq = 0.5 * FS
        b, a = signal.butter(HP_ORDER, HP_CUTOFF / nyq, btype='high')
        ir   = signal.filtfilt(b, a, ir)
        red  = signal.filtfilt(b, a, red)

    # Normalization
    ir_norm  = _apply_norm(ir,  NORM_SELECTION)
    red_norm = _apply_norm(red, NORM_SELECTION)

    # DC component (low-pass only) for perfusion index
    if LP_ENABLE:
        nyq = 0.5 * FS
        sos_lp = signal.butter(LP_ORDER, LP_CUTOFF / nyq, btype='low', output='sos')
        ir_dc  = signal.sosfiltfilt(sos_lp, raw_ir)
        red_dc = signal.sosfiltfilt(sos_lp, raw_red)
    else:
        ir_dc  = raw_ir.copy()
        red_dc = raw_red.copy()

    # SQI check
    ir_sqi,  ir_ok  = _sqi_check(ir_norm,  ir_dc,  FS)
    red_sqi, red_ok = _sqi_check(red_norm, red_dc, FS)
    if SQI_ENABLE and not (ir_ok and red_ok):
        fails = []
        if not ir_ok:  fails.append("IR SQI FAIL")
        if not red_ok: fails.append("RED SQI FAIL")
        reason = "; ".join(fails)
        if VERBOSE_REJECTION:
            print(f"  ⚠️  Win{win_idx}: REJECTED — {reason}")
        return {"status": "REJECTED", "reason": reason, "window_idx": win_idx,
                "ir_sqi": ir_sqi, "red_sqi": red_sqi}

    # Beat detection + ensemble
    ir_beats,  ir_info,  *_ = _detect_beats(ir_norm,  FS)
    red_beats, red_info, *_ = _detect_beats(red_norm, FS)

    if len(ir_beats) < MIN_VALID_BEATS_IR:
        reason = f"IR beats={len(ir_beats)} < {MIN_VALID_BEATS_IR}"
        if VERBOSE_REJECTION: print(f"  ⚠️  Win{win_idx}: REJECTED — {reason}")
        return {"status": "REJECTED", "reason": reason, "window_idx": win_idx,
                "ir_sqi": ir_sqi, "red_sqi": red_sqi}

    if len(red_beats) < MIN_VALID_BEATS_RED:
        reason = f"RED beats={len(red_beats)} < {MIN_VALID_BEATS_RED}"
        if VERBOSE_REJECTION: print(f"  ⚠️  Win{win_idx}: REJECTED — {reason}")
        return {"status": "REJECTED", "reason": reason, "window_idx": win_idx,
                "ir_sqi": ir_sqi, "red_sqi": red_sqi}

    ir_avg,  ir_vpg,  ir_sdp,  ir_t,  ir_fseff  = _build_ensemble(ir_beats,  ENSEMBLE_TARGET_LEN)
    red_avg, red_vpg, red_sdp, red_t, red_fseff = _build_ensemble(red_beats, ENSEMBLE_TARGET_LEN)

    if ir_avg is None or red_avg is None:
        return {"status": "REJECTED", "reason": "Ensemble build failed",
                "window_idx": win_idx}

    return {
        "status":    "ACCEPTED",
        "window_idx": win_idx,
        "ir_norm":   ir_norm,  "red_norm":  red_norm,
        "ir_dc":     ir_dc,    "red_dc":    red_dc,
        "ir_avg":    ir_avg,   "red_avg":   red_avg,
        "ir_vpg":    ir_vpg,   "red_vpg":   red_vpg,
        "ir_sdp":    ir_sdp,   "red_sdp":   red_sdp,
        "ir_t":      ir_t,     "red_t":     red_t,
        "ir_fseff":  ir_fseff, "red_fseff": red_fseff,
        "ir_sqi":    ir_sqi,   "red_sqi":   red_sqi,
        "ir_beats_count": len(ir_beats),
        "red_beats_count": len(red_beats),
    }


def stage3_signal_processing(win_paths):
    _banner("STAGE 3 — Signal Processing")
    results = []
    for idx, wp in enumerate(win_paths):
        r = _process_one_window(wp, idx)
        status = r["status"]
        print(f"  Win{idx:02d}: {status}"
              + (f"  (IR beats={r.get('ir_beats_count','?')}, RED beats={r.get('red_beats_count','?')})" if status == "ACCEPTED" else f"  — {r.get('reason','')}"))
        results.append(r)

    accepted = [r for r in results if r["status"] == "ACCEPTED"]
    rejected = [r for r in results if r["status"] == "REJECTED"]
    print(f"\n  ✅ Accepted: {len(accepted)}   ❌ Rejected: {len(rejected)}")
    if not accepted:
        print("  ❌ All windows were rejected. Cannot continue. Check SQI limits / signal quality.")
        sys.exit(1)
    return results


# ==============================================================================
#  STAGE 4 — PER-WINDOW FEATURE EXTRACTION  (Code 05 logic)
# ==============================================================================

def _shannon_entropy(x, bins=64):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 2: return np.nan
    hist, _ = np.histogram(x, bins=bins, density=False)
    hist = hist[hist > 0]
    if len(hist) == 0: return np.nan
    p = hist / hist.sum()
    return float(-np.sum(p * np.log2(p)))


def _spectral_entropy(x, fs, nperseg=None):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 4: return np.nan
    np_ = min(256, len(x)) if nperseg is None else nperseg
    _, pxx = welch(x, fs=fs, nperseg=np_)
    pxx = pxx[(np.isfinite(pxx)) & (pxx > 0)]
    if len(pxx) == 0: return np.nan
    p = pxx / pxx.sum()
    return float(-np.sum(p * np.log2(p)))


def _peak_interval(x, fs):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 3: return {"PPI": np.nan, "BPM": np.nan, "HRV": np.nan}
    x0 = x - np.mean(x)
    md = max(1, int(0.33 * fs)); pr = max(0.01, 0.10 * np.std(x0))
    pks, _ = find_peaks(x0, distance=md, prominence=pr)
    if len(pks) < 2: return {"PPI": np.nan, "BPM": np.nan, "HRV": np.nan}
    ppi = np.diff(pks) / fs
    return {"PPI": float(np.mean(ppi)),
            "BPM": float(60.0 / np.mean(ppi)),
            "HRV": float(np.std(ppi, ddof=1) * 1000 if len(ppi) > 1 else 0.0)}


def _teo(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 3: return {"TEO Mean": np.nan, "TEO std dev": np.nan}
    psi = x[1:-1]**2 - x[:-2] * x[2:]
    return {"TEO Mean": float(np.mean(psi)),
            "TEO std dev": float(np.std(psi, ddof=1) if len(psi) > 1 else 0.0)}


def _pulse_width(ens, t_ens):
    x = np.asarray(ens, dtype=float)
    if len(x) < 3: return np.nan
    pk = np.argmax(x)
    try:
        ws, _, _, _ = peak_widths(x, [pk], rel_height=0.5)
        dt = np.mean(np.diff(t_ens)) if (t_ens is not None and len(t_ens) == len(x)) else 1.0
        return float(ws[0] * dt)
    except Exception:
        return np.nan


def _harmonic_ratio(ens):
    x = np.asarray(ens, dtype=float); x = x[np.isfinite(x)]
    if len(x) < 8: return np.nan
    x = x - np.mean(x)
    spec = np.abs(rfft(x))**2
    if len(spec) < 3: return np.nan
    spec[0] = 0
    fi = np.argmax(spec[1:]) + 1
    if fi <= 0 or fi >= len(spec): return np.nan
    fp = spec[fi]; rp = np.sum(spec) - fp
    return float(fp / rp) if rp > 0 else np.nan


def _rise_time(ens, t_ens):
    x = np.asarray(ens, dtype=float)
    if len(x) < 5: return np.nan
    pk = np.argmax(x)
    if pk <= 0: return np.nan
    ft = np.argmin(x[:pk+1])
    if t_ens is not None and len(t_ens) == len(x):
        return float(t_ens[pk] - t_ens[ft])
    return float(pk - ft)


def _decay_time(ens, t_ens):
    x = np.asarray(ens, dtype=float)
    if len(x) < 5: return np.nan
    pk = np.argmax(x)
    if pk >= len(x) - 1: return np.nan
    fa = pk + np.argmin(x[pk:])
    if t_ens is not None and len(t_ens) == len(x):
        return float(t_ens[fa] - t_ens[pk])
    return float(fa - pk)


def _dicrotic_notch(ens, t_ens):
    x = np.asarray(ens, dtype=float)
    if len(x) < 5: return np.nan
    pk = np.argmax(x)
    inv = -x; minima, _ = find_peaks(inv)
    ma  = minima[minima > pk]
    ni  = ma[0] if len(ma) > 0 else (pk + 1 + np.argmin(x[pk+1:]) if pk+1 < len(x) else pk)
    if t_ens is not None and len(t_ens) == len(x):
        return float(t_ens[int(ni)])
    return float(ni)


def _ensemble_ratio(red_avg, red_dc, ir_avg, ir_dc):
    red_ac  = float(np.max(red_avg) - np.min(red_avg))
    ir_ac   = float(np.max(ir_avg)  - np.min(ir_avg))
    red_dcm = float(np.mean(red_dc))
    ir_dcm  = float(np.mean(ir_dc))
    r_red   = _safe_ratio(red_ac, red_dcm)
    r_ir    = _safe_ratio(ir_ac,  ir_dcm)
    return _safe_ratio(r_red, r_ir)


def _extract_features_from_window(proc):
    """Extracts all IR + RED features from one accepted window result dict."""
    ir_n   = proc["ir_norm"];  ir_avg  = proc["ir_avg"];  ir_t   = proc["ir_t"]
    red_n  = proc["red_norm"]; red_avg = proc["red_avg"]; red_t  = proc["red_t"]
    ir_dc  = proc["ir_dc"];    red_dc  = proc["red_dc"]
    ir_vpg = proc["ir_vpg"];   red_vpg = proc["red_vpg"]
    ir_sdp = proc["ir_sdp"];   red_sdp = proc["red_sdp"]

    def ch_feats(norm, ens, t_ens, vpg_, sdp_):
        rr  = _peak_interval(norm, FS)
        teo = _teo(norm)
        return {
            "Skewness":             float(skew(ens, bias=False))   if len(ens) > 2 else np.nan,
            "Kurtosis":             float(kurtosis(ens, fisher=True, bias=False)) if len(ens) > 3 else np.nan,
            "Shannon Entropy":      _shannon_entropy(norm),
            "Spectral Entropy":     _spectral_entropy(norm, FS),
            "pulse width":          _pulse_width(ens, t_ens),
            "PPI":                  rr["PPI"],
            "systolic amplitude":   float(np.max(ens) - np.min(ens)),
            "BPM":                  rr["BPM"],
            "HRV":                  rr["HRV"],
            "TEO Mean":             teo["TEO Mean"],
            "TEO std dev":          teo["TEO std dev"],
            "1st_Derivative_Mean":  float(np.mean(vpg_)) if len(vpg_) > 0 else np.nan,
            "2nd_Derivative_Mean":  float(np.mean(sdp_)) if len(sdp_) > 0 else np.nan,
            "2nd_Derivative_Skewness": float(skew(sdp_, bias=False)) if len(sdp_) > 2 else np.nan,
            "Harmonic ratio":       _harmonic_ratio(ens),
            "Rise time":            _rise_time(ens, t_ens),
            "Decay time":           _decay_time(ens, t_ens),
            "Dicrotic notch":       _dicrotic_notch(ens, t_ens),
        }

    ir_f  = ch_feats(ir_n,  ir_avg,  ir_t,  ir_vpg, ir_sdp)
    red_f = ch_feats(red_n, red_avg, red_t, red_vpg, red_sdp)
    ens_r = _ensemble_ratio(red_avg, red_dc, ir_avg, ir_dc)

    flat = {}
    for feat, val in ir_f.items():
        flat[f"IR_{feat}"]  = val
    for feat, val in red_f.items():
        flat[f"Red_{feat}"] = val
    flat["Ensemble ratio"] = ens_r
    return flat


def stage4_extract_features(proc_results):
    _banner("STAGE 4 — Per-Window Feature Extraction")
    accepted  = [r for r in proc_results if r["status"] == "ACCEPTED"]
    all_feats = []
    for r in accepted:
        feats = _extract_features_from_window(r)
        all_feats.append(feats)
        print(f"  Win{r['window_idx']:02d}: Extracted {len(feats)} raw features")
    print(f"\n  ✅ Features extracted from {len(all_feats)} accepted window(s).")
    return all_feats


# ==============================================================================
#  STAGE 5 — FEATURE AVERAGING  (Code 06 logic)
# ==============================================================================

def stage5_average_features(all_win_features):
    _banner("STAGE 5 — Feature Averaging Across Windows")
    if not all_win_features:
        print("  ❌ No features to average.")
        sys.exit(1)

    keys = list(all_win_features[0].keys())
    avg  = {}
    for k in keys:
        vals = [f[k] for f in all_win_features if np.isfinite(float(f[k])) if f.get(k) is not None]
        avg[k] = float(np.mean(vals)) if vals else np.nan

    print(f"  ✅ Averaged {len(all_win_features)} window(s) → {len(avg)} features")
    nan_keys = [k for k, v in avg.items() if not np.isfinite(v)]
    if nan_keys:
        print(f"  ⚠️  {len(nan_keys)} features are NaN after averaging: {nan_keys[:5]}{'...' if len(nan_keys)>5 else ''}")
        print("      Imputing NaN values → 0 (cannot drop at inference time)")
        for k in nan_keys:
            avg[k] = 0.0

    return avg


# ==============================================================================
#  STAGE 6 — FEATURE ENGINEERING → 24 FEATURES  (Code 08 logic)
# ==============================================================================

def stage6_engineer_features(avg_feats):
    _banner("STAGE 6 — Feature Engineering (24 Features)")
    feat24 = {}

    # 18 IR base features
    for col in IR_BASE_FEATURES:
        raw_key = col.replace("IR_", "IR_", 1)  # already prefixed
        feat24[col] = avg_feats.get(raw_key, np.nan)

    # 5 engineered features
    for (out_name, op, op1_key, op2_key) in ENGINEERED_FEATURES:
        v1 = avg_feats.get(op1_key, np.nan)
        v2 = avg_feats.get(op2_key, np.nan)
        if op == "ratio":
            feat24[out_name] = _safe_ratio(v1, v2)
        elif op == "difference":
            feat24[out_name] = (v1 - v2) if (np.isfinite(v1) and np.isfinite(v2)) else np.nan
        else:
            feat24[out_name] = v1

    # Ensemble ratio (keep as-is)
    feat24["Ensemble ratio"] = avg_feats.get("Ensemble ratio", np.nan)

    # Final NaN imputation
    for k in list(feat24.keys()):
        if feat24[k] is None or not np.isfinite(feat24[k]):
            feat24[k] = 0.0

    print(f"  ✅ 24-feature vector assembled:")
    for k, v in feat24.items():
        print(f"     {k:<35s} = {v:.6f}")

    return feat24


# ==============================================================================
#  STAGE 7 — ROBUST SCALING + PREDICTION  (Code 10 + Code 11 logic)
# ==============================================================================

def _load_scaler_params(json_path):
    """Reads center_median and scale_iqr per feature from the Code 10 JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Navigate to the feature_scaler_parameters list
    params_list = None
    for key in ("sub_task_4_robust_scaling", "sub_task_3_4_robust_scaling", "scaling"):
        if key in data:
            params_list = data[key].get("feature_scaler_parameters")
            if params_list: break
    if params_list is None:
        # Try nested search
        def _find_list(d):
            if isinstance(d, dict):
                if "feature_scaler_parameters" in d:
                    return d["feature_scaler_parameters"]
                for v in d.values():
                    r = _find_list(v)
                    if r is not None: return r
            return None
        params_list = _find_list(data)
    if params_list is None:
        raise ValueError("Cannot locate 'feature_scaler_parameters' in the scaler JSON.")
    scaler = {p["feature"]: {"center": p["center_median"], "scale": p["scale_iqr"]}
              for p in params_list if p.get("scale_iqr", 0) != 0}
    return scaler


def stage7_scale_and_predict(feat24):
    _banner("STAGE 7 — Scaling + XGBoost Prediction")

    # Load scaler parameters
    print(f"  📂 Loading scaler from:\n     {SCALER_JSON_PATH}")
    scaler_params = _load_scaler_params(SCALER_JSON_PATH)
    print(f"  ✅ Scaler loaded: {len(scaler_params)} feature entries")

    # Scale all 24 features
    scaled24 = {}
    for feat, val in feat24.items():
        if feat in scaler_params:
            c = scaler_params[feat]["center"]
            s = scaler_params[feat]["scale"]
            scaled24[feat] = (val - c) / s if s != 0 else 0.0
        else:
            print(f"  ⚠️  Feature '{feat}' not in scaler params — using raw value")
            scaled24[feat] = val

    # Apply MANUAL_FEATURE_SELECTION mask
    selected_feats = [f for f, flag in MANUAL_FEATURE_SELECTION.items() if flag == 1]
    feature_vector = np.array([scaled24.get(f, 0.0) for f in selected_feats], dtype=float).reshape(1, -1)

    print(f"\n  🎚️  Features used for prediction ({len(selected_feats)}):")
    for i, (fn, fv) in enumerate(zip(selected_feats, feature_vector[0])):
        print(f"     {i+1:2d}. {fn:<35s} = {fv:.6f}")

    # Load model and predict
    print(f"\n  📂 Loading model from:\n     {MODEL_PKL_PATH}")
    with open(MODEL_PKL_PATH, "rb") as f:
        model = pickle.load(f)
    print(f"  ✅ Model loaded: {type(model).__name__}")

    prediction = float(model.predict(feature_vector)[0])
    print(f"\n  🩸 Predicted Glucose : {prediction:.2f} mg/dL")

    return prediction, selected_feats, scaled24


# ==============================================================================
#  STAGE 8 — SAVE OUTPUTS
# ==============================================================================

def stage8_save_outputs(raw_csv_path, win_paths, saved_windows, proc_results,
                         all_win_features, avg_feats, feat24, scaled24,
                         selected_feats, prediction):
    _banner("STAGE 8 — Saving Outputs")

    ts = datetime.now().isoformat()

    # Collect SQI info
    sqi_info = []
    for r in proc_results:
        sqi_info.append({
            "window_idx": r["window_idx"],
            "status":     r["status"],
            "reason":     r.get("reason", ""),
            "ir_sqi":     _to_json_safe(r.get("ir_sqi",  {})),
            "red_sqi":    _to_json_safe(r.get("red_sqi", {})),
        })

    session = {
        "run_id":                   _RUN_ID,
        "timestamp":                ts,
        "subject_id":               SUBJECT_ID,
        "session_notes":            SESSION_NOTES,
        "input_mode":               _INPUT_MODE,
        "raw_csv_path":             raw_csv_path,
        "windows_selected":         len(win_paths),
        "windows_accepted":         sum(1 for r in proc_results if r["status"] == "ACCEPTED"),
        "windows_rejected":         sum(1 for r in proc_results if r["status"] == "REJECTED"),
        "rejection_reasons":        {f"Win{r['window_idx']}": r.get("reason","") for r in proc_results if r["status"] == "REJECTED"},
        "sqi_per_window":           sqi_info,
        "per_window_features_raw":  _to_json_safe(all_win_features),
        "averaged_features_36":     _to_json_safe(avg_feats),
        "averaged_features_24":     _to_json_safe(feat24),
        "scaled_features_24":       _to_json_safe(scaled24),
        "features_used_count":      len(selected_feats),
        "features_used_names":      selected_feats,
        "feature_vector_for_model": _to_json_safe([scaled24.get(f, 0.0) for f in selected_feats]),
        "predicted_glucose_mg_dL":  round(prediction, 4),
        "model_pkl_path":           MODEL_PKL_PATH,
        "scaler_json_path":         SCALER_JSON_PATH,
        "pipeline_hyperparameters": _to_json_safe({
            "FS": FS, "WINDOW_DURATION_SEC": WINDOW_DURATION_SEC,
            "SPIKE_ENABLE": SPIKE_ENABLE, "KERNEL_SIZE": KERNEL_SIZE,
            "INVERT_ENABLE": INVERT_ENABLE,
            "LP_ENABLE": LP_ENABLE, "LP_CUTOFF": LP_CUTOFF, "LP_ORDER": LP_ORDER,
            "HP_ENABLE": HP_ENABLE, "HP_CUTOFF": HP_CUTOFF, "HP_ORDER": HP_ORDER,
            "SG_ENABLE": SG_ENABLE, "SG_WINDOW": SG_WINDOW, "SG_POLY": SG_POLY,
            "NORM_SELECTION": NORM_SELECTION, "ENSEMBLE_TARGET_LEN": ENSEMBLE_TARGET_LEN,
            "MIN_VALID_BEATS_IR": MIN_VALID_BEATS_IR, "MIN_VALID_BEATS_RED": MIN_VALID_BEATS_RED,
            "PEAK_MIN_DISTANCE_SEC": PEAK_MIN_DISTANCE_SEC, "PEAK_PROM_FACTOR": PEAK_PROM_FACTOR,
            "VALLEY_MIN_DISTANCE_SEC": VALLEY_MIN_DISTANCE_SEC, "VALLEY_PROM_FACTOR": VALLEY_PROM_FACTOR,
            "MIN_FOOT_TO_PEAK_SEC": MIN_FOOT_TO_PEAK_SEC, "MAX_FOOT_TO_PEAK_SEC": MAX_FOOT_TO_PEAK_SEC,
            "MAX_VALLEY_TO_FOOT_SEC": MAX_VALLEY_TO_FOOT_SEC, "MAX_FOOT_REL_HEIGHT": MAX_FOOT_REL_HEIGHT,
            "MAX_ABS_VPG_AT_FOOT": MAX_ABS_VPG_AT_FOOT, "EDGE_EXCLUSION_SEC": EDGE_EXCLUSION_SEC,
            "MIN_BEAT_DURATION_SEC": MIN_BEAT_DURATION_SEC, "MAX_BEAT_DURATION_SEC": MAX_BEAT_DURATION_SEC,
            "BEAT_DURATION_MEDIAN_TOLERANCE": BEAT_DURATION_MEDIAN_TOLERANCE,
            "MAIN_PEAK_SEARCH_WINDOW_SEC": MAIN_PEAK_SEARCH_WINDOW_SEC,
            "MAIN_PEAK_MIN_DELAY_SEC": MAIN_PEAK_MIN_DELAY_SEC,
            "START_INCOMPLETE_MARGIN_SEC": START_INCOMPLETE_MARGIN_SEC,
            "END_INCOMPLETE_MARGIN_SEC": END_INCOMPLETE_MARGIN_SEC,
            "SQI_ENABLE": SQI_ENABLE,
            "SQI_LIMITS": SQI_LIMITS,
            "MANUAL_FEATURE_SELECTION": MANUAL_FEATURE_SELECTION,
        }),
    }

    # ── Write session JSON ────────────────────────────────────────────────────
    json_path = _RUN_DIR / "session_metadata.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(session, f, indent=4)
    print(f"  💾 Session JSON  : {json_path}")

    # ── Append to prediction_log.csv ─────────────────────────────────────────
    log_path = Path(OUTPUT_ROOT) / "prediction_log.csv"
    log_exists = log_path.exists()

    # Build the row dict
    row = {
        "run_id":                   _RUN_ID,
        "timestamp":                ts,
        "subject_id":               SUBJECT_ID,
        "session_notes":            SESSION_NOTES,
        "input_mode":               _INPUT_MODE,
        "windows_selected":         len(win_paths),
        "windows_accepted":         session["windows_accepted"],
        "windows_rejected":         session["windows_rejected"],
        "predicted_glucose_mg_dL":  round(prediction, 4),
        "model_pkl":                os.path.basename(MODEL_PKL_PATH),
    }
    # Add all 24 features to the row
    for feat in feat24:
        row[feat] = round(feat24[feat], 8) if np.isfinite(feat24[feat]) else ""

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not log_exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"  💾 Prediction Log: {log_path}  ({'appended' if log_exists else 'created'})")
    return json_path, log_path


# ==============================================================================
#  MAIN ENTRY POINT
# ==============================================================================

def main():
    try:
        # ── Stage 0: Startup ──────────────────────────────────────────────────
        mode_choice = run_startup_menu()

        # ── Stage 1: Data Acquisition ─────────────────────────────────────────
        raw_csv = stage1_acquire_data(mode_choice)

        # ── Stage 2: Window Slicer ────────────────────────────────────────────
        win_paths, saved_windows = stage2_window_slicer(raw_csv)

        # ── Stage 3: Signal Processing ────────────────────────────────────────
        proc_results = stage3_signal_processing(win_paths)

        # ── Stage 4: Feature Extraction ───────────────────────────────────────
        all_win_feats = stage4_extract_features(proc_results)

        # ── Stage 5: Averaging ────────────────────────────────────────────────
        avg_feats = stage5_average_features(all_win_feats)

        # ── Stage 6: Feature Engineering ─────────────────────────────────────
        feat24 = stage6_engineer_features(avg_feats)

        # ── Stage 7: Scale + Predict ──────────────────────────────────────────
        prediction, sel_feats, scaled24 = stage7_scale_and_predict(feat24)

        # ── Stage 8: Save Outputs ─────────────────────────────────────────────
        json_path, log_path = stage8_save_outputs(
            raw_csv, win_paths, saved_windows, proc_results,
            all_win_feats, avg_feats, feat24, scaled24, sel_feats, prediction
        )

        # ── Final Summary ─────────────────────────────────────────────────────
        print("\n" + "═" * 70)
        print("  ✅  GlucoSense Inference Complete")
        print("═" * 70)
        print(f"  Subject ID          : {SUBJECT_ID}")
        print(f"  Session Notes       : {SESSION_NOTES if SESSION_NOTES else '(none)'}")
        print(f"  Windows Selected    : {len(win_paths)}")
        print(f"  Windows Accepted    : {sum(1 for r in proc_results if r['status']=='ACCEPTED')}")
        print(f"  Windows Rejected    : {sum(1 for r in proc_results if r['status']=='REJECTED')}")
        print(f"\n  🩸 Predicted Glucose : {prediction:.2f} mg/dL")
        print(f"\n  📄 Session JSON      : {json_path}")
        print(f"  📊 Prediction Log    : {log_path}")
        print("═" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n  ⚠️  Interrupted by user. Exiting.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ❌ Pipeline error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
