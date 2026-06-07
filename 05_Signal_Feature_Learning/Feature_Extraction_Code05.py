# ==========================================
# STEP 3: FEATURE EXTRACTION (AUTO BATCH, POPUP SELECTOR)
# Updated for compatibility with new automated signal processing pipeline
# ==========================================

import os
import re
import json
import traceback
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, welch, peak_widths
from scipy.stats import skew, kurtosis
from scipy.fft import rfft


# --------------------------------------------------
# USER SETTINGS
# --------------------------------------------------
INPUT_ROOT = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\04_Filtered")
OUTPUT_ROOT = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\05_Features_")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def safe_array(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x


def safe_ratio(num, den, default=np.nan):
    if not np.isfinite(num) or not np.isfinite(den) or den == 0:
        return default
    return num / den


def shannon_entropy_signal(x, bins=64):
    x = safe_array(x)
    if len(x) < 2:
        return np.nan

    hist, _ = np.histogram(x, bins=bins, density=False)
    hist = hist[hist > 0]
    if len(hist) == 0:
        return np.nan

    p = hist / np.sum(hist)
    return -np.sum(p * np.log2(p))


def spectral_entropy_signal(x, fs, nperseg=None):
    x = safe_array(x)
    if len(x) < 4:
        return np.nan

    if nperseg is None:
        nperseg = min(256, len(x))

    _, pxx = welch(x, fs=fs, nperseg=nperseg)
    pxx = pxx[np.isfinite(pxx)]
    pxx = pxx[pxx > 0]

    if len(pxx) == 0:
        return np.nan

    p = pxx / np.sum(pxx)
    return -np.sum(p * np.log2(p))


def detect_ppg_peaks(x, fs):
    x = safe_array(x)
    if len(x) < 3:
        return np.array([], dtype=int)

    x0 = x - np.mean(x)
    min_distance = max(1, int(0.33 * fs))   # ~180 BPM upper bound
    prominence = max(0.01, 0.10 * np.std(x0))

    peaks, _ = find_peaks(x0, distance=min_distance, prominence=prominence)
    return peaks


def peak_interval_bpm_hrv(x, fs):
    peaks = detect_ppg_peaks(x, fs)

    if len(peaks) < 2:
        return {
            "PPI": np.nan,
            "BPM": np.nan,
            "HRV": np.nan,
            "Num_Peaks": len(peaks),
        }

    ppi = np.diff(peaks) / fs
    bpm = 60.0 / np.mean(ppi)
    hrv_sdnn = np.std(ppi, ddof=1) * 1000 if len(ppi) > 1 else 0.0

    return {
        "PPI": np.mean(ppi),
        "BPM": bpm,
        "HRV": hrv_sdnn,
        "Num_Peaks": len(peaks),
    }


def teo_signal(x):
    x = safe_array(x)
    if len(x) < 3:
        return np.array([])
    return x[1:-1] ** 2 - x[:-2] * x[2:]


def teo_features(x):
    psi = teo_signal(x)
    if len(psi) == 0:
        return {"TEO Mean": np.nan, "TEO std dev": np.nan}

    return {
        "TEO Mean": np.mean(psi),
        "TEO std dev": np.std(psi, ddof=1) if len(psi) > 1 else 0.0,
    }


def pulse_width_feature(signal, time_axis=None):
    x = safe_array(signal)
    if len(x) < 3:
        return np.nan

    peak_idx = np.argmax(x)
    peaks = np.array([peak_idx])

    try:
        widths, _, _, _ = peak_widths(x, peaks, rel_height=0.5)
        width_samples = widths[0]

        if time_axis is not None:
            t = safe_array(time_axis)
            if len(t) == len(x) and len(t) > 1:
                dt = np.mean(np.diff(t))
                return width_samples * dt

        return width_samples
    except Exception:
        return np.nan


def systolic_amplitude(signal):
    x = safe_array(signal)
    if len(x) == 0:
        return np.nan
    return np.max(x) - np.min(x)


def harmonic_ratio(signal):
    x = safe_array(signal)
    if len(x) < 8:
        return np.nan

    x = x - np.mean(x)
    spec = np.abs(rfft(x)) ** 2

    if len(spec) < 3:
        return np.nan

    spec[0] = 0
    fundamental_idx = np.argmax(spec[1:]) + 1

    if fundamental_idx <= 0 or fundamental_idx >= len(spec):
        return np.nan

    fundamental_power = spec[fundamental_idx]
    remaining_power = np.sum(spec) - fundamental_power

    if remaining_power <= 0:
        return np.nan

    return fundamental_power / remaining_power


def ensemble_ac_dc_ratio(ens_signal, dc_signal):
    ens_signal = safe_array(ens_signal)
    dc_signal = safe_array(dc_signal)

    if len(ens_signal) == 0 or len(dc_signal) == 0:
        return np.nan, np.nan, np.nan

    ac_amp = np.max(ens_signal) - np.min(ens_signal)
    dc_mean = np.mean(dc_signal)
    ratio = safe_ratio(ac_amp, dc_mean)

    return ratio, ac_amp, dc_mean


def ensemble_ratio_feature(red_ens, red_dc, ir_ens, ir_dc):
    red_ratio, red_ac_amp, red_dc_mean = ensemble_ac_dc_ratio(red_ens, red_dc)
    ir_ratio, ir_ac_amp, ir_dc_mean = ensemble_ac_dc_ratio(ir_ens, ir_dc)

    ratio_of_ratios = safe_ratio(red_ratio, ir_ratio)

    return {
        "Red_Ensemble_AC_Amp": red_ac_amp,
        "IR_Ensemble_AC_Amp": ir_ac_amp,
        "Red_DC_Mean": red_dc_mean,
        "IR_DC_Mean": ir_dc_mean,
        "Red_Ensemble_AC_DC_Ratio": red_ratio,
        "IR_Ensemble_AC_DC_Ratio": ir_ratio,
        "Ensemble ratio": ratio_of_ratios,
    }


def find_dicrotic_notch(signal, time_axis=None):
    x = safe_array(signal)
    if len(x) < 5:
        return np.nan

    peak_idx = np.argmax(x)
    inv = -x
    minima, _ = find_peaks(inv)
    minima_after_peak = minima[minima > peak_idx]

    if len(minima_after_peak) == 0:
        if peak_idx + 1 >= len(x):
            return np.nan
        notch_idx = peak_idx + 1 + np.argmin(x[peak_idx + 1 :])
    else:
        notch_idx = minima_after_peak[0]

    if time_axis is not None:
        t = safe_array(time_axis)
        if len(t) == len(x):
            return t[notch_idx]

    return float(notch_idx)


def rise_time(signal, time_axis=None):
    x = safe_array(signal)
    if len(x) < 5:
        return np.nan

    peak_idx = np.argmax(x)
    if peak_idx <= 0:
        return np.nan

    foot_idx = np.argmin(x[: peak_idx + 1])

    if time_axis is not None:
        t = safe_array(time_axis)
        if len(t) == len(x):
            return t[peak_idx] - t[foot_idx]

    return float(peak_idx - foot_idx)


def decay_time(signal, time_axis=None):
    x = safe_array(signal)
    if len(x) < 5:
        return np.nan

    peak_idx = np.argmax(x)
    if peak_idx >= len(x) - 1:
        return np.nan

    foot_after_idx = peak_idx + np.argmin(x[peak_idx:])

    if time_axis is not None:
        t = safe_array(time_axis)
        if len(t) == len(x):
            return t[foot_after_idx] - t[peak_idx]

    return float(foot_after_idx - peak_idx)


def remove_trailing_filtered(name):
    return re.sub(r"_Filtered$", "", name)


def remove_trailing_window(name):
    return re.sub(r"_Win\d+$", "", name)


def derive_window_base_name(file_ensemble, file_full):
    if file_ensemble.endswith("_Filtered_Ensemble.csv"):
        return file_ensemble.replace("_Filtered_Ensemble.csv", "")
    elif file_full.endswith("_Filtered_Full.csv"):
        return file_full.replace("_Filtered_Full.csv", "")
    else:
        return os.path.splitext(file_ensemble)[0].replace("_Filtered_Ensemble", "")


def derive_subject_base_name(folder_path, file_ensemble, file_full):
    input_folder_name = os.path.basename(os.path.normpath(folder_path))
    candidate = remove_trailing_filtered(input_folder_name)
    candidate = remove_trailing_window(candidate)

    if candidate and candidate.strip():
        return candidate

    fallback = derive_window_base_name(file_ensemble, file_full)
    fallback = remove_trailing_window(fallback)
    return fallback


def find_required_files(window_folder):
    files = os.listdir(window_folder)

    file_full = next((f for f in files if f.endswith("_Filtered_Full.csv")), None)
    file_ensemble = next((f for f in files if f.endswith("_Filtered_Ensemble.csv")), None)
    file_config = next((f for f in files if f.endswith("_Filtered_Configuration.json")), None)

    return file_full, file_ensemble, file_config


def load_window_data(window_folder):
    file_full, file_ensemble, file_config = find_required_files(window_folder)

    if file_full is None:
        raise FileNotFoundError("Missing *_Filtered_Full.csv")
    if file_ensemble is None:
        raise FileNotFoundError("Missing *_Filtered_Ensemble.csv")
    if file_config is None:
        raise FileNotFoundError("Missing *_Filtered_Configuration.json")

    full_path = os.path.join(window_folder, file_full)
    ens_path = os.path.join(window_folder, file_ensemble)
    cfg_path = os.path.join(window_folder, file_config)

    df_full = pd.read_csv(full_path)
    df_ens = pd.read_csv(ens_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # ----- FIX #1: Support both NEW (lowercase) and OLD (uppercase) JSON keys -----
    metadata = cfg.get("metadata") or cfg.get("Metadata") or {}

    fs = (
        metadata.get("sampling_rate_fs")
        or metadata.get("Sampling_Rate_FS")
        or None
    )

    if fs is None:
        raise ValueError(
            "Sampling rate not found in config. "
            "Expected key 'sampling_rate_fs' (new) or 'Sampling_Rate_FS' (old) "
            "under 'metadata' / 'Metadata'."
        )

    return df_full, df_ens, cfg, file_full, file_ensemble, file_config, float(fs)


def get_win_index_from_folder(folder_path):
    name = os.path.basename(folder_path)
    m = re.search(r"_Win(\d+)_", name)
    return int(m.group(1)) if m else 999999


def resolve_selected_folder():
    if not INPUT_ROOT.exists():
        raise SystemExit("❌ Invalid input root path")

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    selected = filedialog.askdirectory(
        initialdir=str(INPUT_ROOT),
        title="Select 1st-step Filtered folder (example: mirzan(...)_Filtered)"
    )

    root.destroy()

    if not selected:
        raise SystemExit("❌ No folder selected")

    folder = Path(selected)

    if not folder.exists():
        raise SystemExit("❌ Selected folder does not exist")

    return folder


def build_filtered_configuration_summary(cfg, flat_features):
    """
    Build a properly organized Filtered_Configuration JSON that preserves
    the valuable original data and appends the extracted features.

    Supports BOTH new (lowercase) and old (uppercase) JSON key conventions
    from the automated signal processing pipeline.
    """
    summary = {}

    # ----- Metadata (new: lowercase, old: uppercase) -----
    metadata = cfg.get("metadata") or cfg.get("Metadata")
    if metadata:
        summary["metadata"] = metadata

    # ----- Hyperparameters (only available in new pipeline) -----
    if "hyperparameters" in cfg:
        summary["hyperparameters"] = cfg["hyperparameters"]

    # ----- Folder Structure -----
    folder_struct = cfg.get("folder_structure") or cfg.get("Folder_Structure")
    if folder_struct:
        summary["folder_structure"] = folder_struct

    # ----- Signal Quality (only in new pipeline) -----
    if "signal_quality" in cfg:
        summary["signal_quality"] = cfg["signal_quality"]

    # ----- Pipeline Diagnostic (only in new pipeline) -----
    if "pipeline_diagnostic" in cfg:
        summary["pipeline_diagnostic"] = cfg["pipeline_diagnostic"]

    # ----- PPG Features per Channel (only in new pipeline) -----
    if "ppg_features_per_channel" in cfg:
        summary["ppg_features_per_channel"] = cfg["ppg_features_per_channel"]

    # ----- Golden Standard Features (only in new pipeline) -----
    if "golden_standard_features" in cfg:
        summary["golden_standard_features"] = cfg["golden_standard_features"]

    # ----- Compact ensemble summary (drop bulky time_axis & per-pulse logs) -----
    def ensemble_summary(ens):
        if not isinstance(ens, dict):
            return {}
        keep_keys = [
            "title",
            "target_len",
            "beats_used",
            "rejected_candidates",
            "rejected_pulses",
            "segmented_pulses_total",
            "avg_duration",
            "fs_eff",
            "fiducials",
        ]
        out = {k: ens[k] for k in keep_keys if k in ens}
        if "feet" in ens:
            out["num_feet_pairs"] = len(ens["feet"])
        if "rejected_pulses_info" in ens:
            out["rejected_pulses_info"] = ens["rejected_pulses_info"]
        return out

    # Ensemble keys: new pipeline uses lowercase prefix, old uses uppercase
    if "ensemble_RED" in cfg:
        summary["ensemble_RED"] = ensemble_summary(cfg["ensemble_RED"])
    elif "Ensemble_RED" in cfg:
        summary["ensemble_RED"] = ensemble_summary(cfg["Ensemble_RED"])

    if "ensemble_IR" in cfg:
        summary["ensemble_IR"] = ensemble_summary(cfg["ensemble_IR"])
    elif "Ensemble_IR" in cfg:
        summary["ensemble_IR"] = ensemble_summary(cfg["Ensemble_IR"])

    # ----- Append extracted features -----
    summary["extracted_features"] = flat_features

    return summary


def save_signal_plot(df_full, df_ens, output_folder_path, base_name):
    """
    Save a 4-panel plot:
      1. DC Component (Low Pass) - Baseline Drift
      2. AC Component (High Pass) - Pulsatile Signal
      3. Normalized Signal (0..1)
      4. Ensemble Average (Single Beat Template)
    """
    try:
        # Time axis for full signal
        if "Time_s" in df_full.columns:
            t_full = df_full["Time_s"].values
        elif "Time" in df_full.columns:
            t_full = df_full["Time"].values
        else:
            t_full = np.arange(len(df_full))

        red_dc = df_full["Red_DC_LowPass"].values
        ir_dc = df_full["IR_DC_LowPass"].values
        red_ac = df_full["Red_AC_HighPass"].values
        ir_ac = df_full["IR_AC_HighPass"].values
        red_norm = df_full["Red_Normalized"].values
        ir_norm = df_full["IR_Normalized"].values

        t_red_ens = df_ens["Time_Red_s"].values
        red_ens = df_ens["Red_Ensemble_Avg"].values
        t_ir_ens = df_ens["Time_IR_s"].values
        ir_ens = df_ens["IR_Ensemble_Avg"].values

        fig, axes = plt.subplots(4, 1, figsize=(10, 11))

        # 1. DC
        axes[0].plot(t_full, red_dc, color="darkred", label="Red DC", linewidth=1)
        axes[0].plot(t_full, ir_dc, color="blue", label="IR DC", linewidth=1)
        axes[0].set_title("1. DC Component (Low Pass Filtered) - Baseline Drift")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        # 2. AC
        axes[1].plot(t_full, red_ac, color="red", label="Red AC", linewidth=0.8)
        axes[1].plot(t_full, ir_ac, color="blue", label="IR AC", linewidth=0.8)
        axes[1].set_title("2. AC Component (High Pass Filtered) - Pulsatile Signal")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        # 3. Normalized
        axes[2].plot(t_full, red_norm, color="red", label="Red Norm", linewidth=0.8)
        axes[2].plot(t_full, ir_norm, color="blue", label="IR Norm", linewidth=0.8)
        axes[2].set_title("3. Normalized Signal (0 to 1 Scaled) - Shape Analysis")
        axes[2].set_ylabel("Normalized Amp")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

        # 4. Ensemble Average
        axes[3].plot(t_red_ens, red_ens, color="red", label="Red Avg Beat", linewidth=2)
        axes[3].plot(t_ir_ens, ir_ens, color="blue", label="IR Avg Beat", linewidth=2)
        axes[3].set_title("4. Ensemble Average (Cleaned Single Beat Template)")
        axes[3].set_xlabel("Time (seconds)")
        axes[3].set_ylabel("Amplitude")
        axes[3].legend(loc="upper right")
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = output_folder_path / f"{base_name}_Signal_Overview.png"
        fig.savefig(plot_path, dpi=120, bbox_inches="tight")
        plt.close(fig)

        return str(plot_path)
    except Exception as e:
        print(f"⚠️ Plot generation failed: {e}")
        return None


# --------------------------------------------------
# FILE REPLACEMENT TRACKING HELPERS
# --------------------------------------------------
def check_existing_file(file_path):
    """
    Check if a file already exists before saving.
    Returns a dict with existence status and file size info.
    """
    p = Path(file_path)
    if p.exists() and p.is_file():
        try:
            size_bytes = p.stat().st_size
            size_kb = size_bytes / 1024.0
            return {
                "exists": True,
                "path": str(p),
                "size_bytes": size_bytes,
                "size_kb": size_kb,
            }
        except Exception:
            return {"exists": True, "path": str(p), "size_bytes": None, "size_kb": None}
    return {"exists": False, "path": str(p), "size_bytes": None, "size_kb": None}


def report_replaced_files(replaced_list, output_folder_path):
    """
    Print a clean terminal report of all files that were replaced inside
    the given output folder.
    """
    if not replaced_list:
        print(f"🆕 No existing files found — all output files are newly created.")
        return

    print(f"\n♻️ REPLACED {len(replaced_list)} EXISTING FILE(S) in:")
    print(f"   📁 {output_folder_path}")
    print("   " + "-" * 56)

    for idx, info in enumerate(replaced_list, start=1):
        file_name = os.path.basename(info["path"])
        old_size = info.get("old_size_kb")
        new_size = info.get("new_size_kb")

        size_old_str = f"{old_size:.2f} KB" if old_size is not None else "N/A"
        size_new_str = f"{new_size:.2f} KB" if new_size is not None else "N/A"

        print(f"   {idx}. {file_name}")
        print(f"      ↳ Old size: {size_old_str}  →  New size: {size_new_str}")


def process_window(folder_path):
    df_full, df_ens, cfg, file_full, file_ensemble, file_config, fs = load_window_data(folder_path)

    required_full_cols = [
        "Red_AC_HighPass",
        "IR_AC_HighPass",
        "Red_DC_LowPass",
        "IR_DC_LowPass",
        "Red_Normalized",
        "IR_Normalized",
    ]

    required_ens_cols = [
        "Time_Red_s",
        "Red_Ensemble_Avg",
        "Red_VPG",
        "Red_SDPPG",
        "Time_IR_s",
        "IR_Ensemble_Avg",
        "IR_VPG",
        "IR_SDPPG",
    ]

    missing_full = [c for c in required_full_cols if c not in df_full.columns]
    missing_ens = [c for c in required_ens_cols if c not in df_ens.columns]

    if missing_full:
        raise ValueError(f"Missing columns in df_full: {missing_full}")
    if missing_ens:
        raise ValueError(f"Missing columns in df_ens: {missing_ens}")

    red_norm = df_full["Red_Normalized"].values
    ir_norm = df_full["IR_Normalized"].values

    red_dc = df_full["Red_DC_LowPass"].values
    ir_dc = df_full["IR_DC_LowPass"].values

    time_red_ens = df_ens["Time_Red_s"].values
    red_ens = df_ens["Red_Ensemble_Avg"].values
    red_vpg = df_ens["Red_VPG"].values
    red_sdppg = df_ens["Red_SDPPG"].values

    time_ir_ens = df_ens["Time_IR_s"].values
    ir_ens = df_ens["IR_Ensemble_Avg"].values
    ir_vpg = df_ens["IR_VPG"].values
    ir_sdppg = df_ens["IR_SDPPG"].values

    red_features = {}
    ir_features = {}
    combined_features = {}

    red_rr = peak_interval_bpm_hrv(red_norm, fs)
    red_teo = teo_features(red_norm)

    red_features["Shannon Entropy"] = shannon_entropy_signal(red_norm)
    red_features["Spectral Entropy"] = spectral_entropy_signal(red_norm, fs)
    red_features["PPI"] = red_rr["PPI"]
    red_features["BPM"] = red_rr["BPM"]
    red_features["HRV"] = red_rr["HRV"]
    red_features["TEO Mean"] = red_teo["TEO Mean"]
    red_features["TEO std dev"] = red_teo["TEO std dev"]

    ir_rr = peak_interval_bpm_hrv(ir_norm, fs)
    ir_teo = teo_features(ir_norm)

    ir_features["Shannon Entropy"] = shannon_entropy_signal(ir_norm)
    ir_features["Spectral Entropy"] = spectral_entropy_signal(ir_norm, fs)
    ir_features["PPI"] = ir_rr["PPI"]
    ir_features["BPM"] = ir_rr["BPM"]
    ir_features["HRV"] = ir_rr["HRV"]
    ir_features["TEO Mean"] = ir_teo["TEO Mean"]
    ir_features["TEO std dev"] = ir_teo["TEO std dev"]

    red_features["Skewness"] = skew(red_ens, bias=False) if len(red_ens) > 2 else np.nan
    red_features["Kurtosis"] = kurtosis(red_ens, fisher=True, bias=False) if len(red_ens) > 3 else np.nan
    red_features["pulse width"] = pulse_width_feature(red_ens, time_red_ens)
    red_features["systolic amplitude"] = systolic_amplitude(red_ens)
    red_features["1st_Derivative_Mean"] = np.mean(red_vpg) if len(red_vpg) > 0 else np.nan
    red_features["2nd_Derivative_Mean"] = np.mean(red_sdppg) if len(red_sdppg) > 0 else np.nan
    red_features["2nd_Derivative_Skewness"] = skew(red_sdppg, bias=False) if len(red_sdppg) > 2 else np.nan
    red_features["Harmonic ratio"] = harmonic_ratio(red_ens)
    red_features["Rise time"] = rise_time(red_ens, time_red_ens)
    red_features["Decay time"] = decay_time(red_ens, time_red_ens)
    red_features["Dicrotic notch"] = find_dicrotic_notch(red_ens, time_red_ens)

    ir_features["Skewness"] = skew(ir_ens, bias=False) if len(ir_ens) > 2 else np.nan
    ir_features["Kurtosis"] = kurtosis(ir_ens, fisher=True, bias=False) if len(ir_ens) > 3 else np.nan
    ir_features["pulse width"] = pulse_width_feature(ir_ens, time_ir_ens)
    ir_features["systolic amplitude"] = systolic_amplitude(ir_ens)
    ir_features["1st_Derivative_Mean"] = np.mean(ir_vpg) if len(ir_vpg) > 0 else np.nan
    ir_features["2nd_Derivative_Mean"] = np.mean(ir_sdppg) if len(ir_sdppg) > 0 else np.nan
    ir_features["2nd_Derivative_Skewness"] = skew(ir_sdppg, bias=False) if len(ir_sdppg) > 2 else np.nan
    ir_features["Harmonic ratio"] = harmonic_ratio(ir_ens)
    ir_features["Rise time"] = rise_time(ir_ens, time_ir_ens)
    ir_features["Decay time"] = decay_time(ir_ens, time_ir_ens)
    ir_features["Dicrotic notch"] = find_dicrotic_notch(ir_ens, time_ir_ens)

    ratio_info = ensemble_ratio_feature(red_ens, red_dc, ir_ens, ir_dc)
    combined_features["Ensemble ratio"] = ratio_info["Ensemble ratio"]

    feature_order = [
        "Skewness",
        "Kurtosis",
        "Shannon Entropy",
        "Spectral Entropy",
        "pulse width",
        "PPI",
        "systolic amplitude",
        "BPM",
        "HRV",
        "TEO Mean",
        "TEO std dev",
        "1st_Derivative_Mean",
        "2nd_Derivative_Mean",
        "2nd_Derivative_Skewness",
        "Harmonic ratio",
        "Ensemble ratio",
        "Rise time",
        "Decay time",
        "Dicrotic notch",
    ]

    rows = []
    for feat in feature_order:
        if feat == "Ensemble ratio":
            rows.append(
                {
                    "Feature": feat,
                    "Red_Value": combined_features.get(feat, np.nan),
                    "IR_Value": np.nan,
                }
            )
        else:
            rows.append(
                {
                    "Feature": feat,
                    "Red_Value": red_features.get(feat, np.nan),
                    "IR_Value": ir_features.get(feat, np.nan),
                }
            )

    df_feature_table = pd.DataFrame(rows)

    subject_base_name = derive_subject_base_name(folder_path, file_ensemble, file_full)
    main_output_folder_name = f"{subject_base_name}_Features"
    main_output_folder_path = OUTPUT_ROOT / main_output_folder_name

    base_name = derive_window_base_name(file_ensemble, file_full)
    output_folder_name = f"{base_name}_Feature"
    output_folder_path = main_output_folder_path / output_folder_name

    # --------------------------------------------------
    # CLEAN REPLACE: If the output folder already exists,
    # delete it entirely and recreate fresh.
    # This guarantees no stale files remain from previous runs.
    # --------------------------------------------------
    folder_was_replaced = False
    if output_folder_path.exists():
        try:
            import shutil
            shutil.rmtree(output_folder_path)
            folder_was_replaced = True
            print(f"♻️  Existing folder removed and will be recreated fresh:")
            print(f"   📁 {output_folder_path}")
        except Exception as e:
            print(f"⚠️  Could not remove existing folder: {e}")
            print(f"   Proceeding with overwrite of individual files instead.")

    # Recreate folders (fresh if replaced, or new if first time)
    main_output_folder_path.mkdir(parents=True, exist_ok=True)
    output_folder_path.mkdir(parents=True, exist_ok=True)

    table_csv_path = output_folder_path / f"{output_folder_name}_Features_Table.csv"

    flat_features = {}
    for feat in feature_order:
        if feat == "Ensemble ratio":
            flat_features["Ensemble ratio"] = combined_features.get(feat, np.nan)
        else:
            flat_features[f"Red_{feat}"] = red_features.get(feat, np.nan)
            flat_features[f"IR_{feat}"] = ir_features.get(feat, np.nan)

    df_features_flat = pd.DataFrame([flat_features])

    flat_csv_path = output_folder_path / f"{output_folder_name}_Features_Flat.csv"
    feature_json_path = output_folder_path / f"{output_folder_name}_Features.json"
    dst_config_path = output_folder_path / file_config
    plot_path_planned = output_folder_path / f"{output_folder_name}_Signal_Overview.png"

    # --------------------------------------------------
    # PRE-CHECK for existing files (BEFORE saving)
    # --------------------------------------------------
    pre_check_files = [
        ("Features Table CSV", table_csv_path),
        ("Features Flat CSV", flat_csv_path),
        ("Features JSON", feature_json_path),
        ("Filtered Configuration JSON", dst_config_path),
        ("Signal Overview Plot", plot_path_planned),
    ]

    existing_before = []
    for label, fp in pre_check_files:
        info = check_existing_file(fp)
        if info["exists"]:
            existing_before.append(
                {
                    "label": label,
                    "path": info["path"],
                    "old_size_kb": info["size_kb"],
                }
            )

    # --------------------------------------------------
    # Save outputs (will overwrite existing files)
    # --------------------------------------------------
    df_feature_table.to_csv(table_csv_path, index=False)
    df_features_flat.to_csv(flat_csv_path, index=False)

    # Pure features JSON (unchanged behavior)
    with open(feature_json_path, "w", encoding="utf-8") as f:
        json.dump(flat_features, f, indent=4)

    # Properly organized Filtered_Configuration JSON
    organized_config = build_filtered_configuration_summary(cfg, flat_features)
    with open(dst_config_path, "w", encoding="utf-8") as f:
        json.dump(organized_config, f, indent=4)

    # Save signal overview plot
    plot_path = save_signal_plot(df_full, df_ens, output_folder_path, output_folder_name)

    # --------------------------------------------------
    # POST-CHECK — get new sizes for replaced files
    # --------------------------------------------------
    replaced_files_info = []
    for entry in existing_before:
        new_info = check_existing_file(entry["path"])
        replaced_files_info.append(
            {
                "label": entry["label"],
                "path": entry["path"],
                "old_size_kb": entry["old_size_kb"],
                "new_size_kb": new_info["size_kb"],
            }
        )

    return {
        "window_folder": os.path.basename(folder_path),
        "output_folder": str(output_folder_path),
        "table_csv": str(table_csv_path),
        "flat_csv": str(flat_csv_path),
        "feature_json": str(feature_json_path),
        "filtered_config_json": str(dst_config_path),
        "signal_plot": plot_path,
        "replaced_files": replaced_files_info,
        "folder_was_replaced": folder_was_replaced,  # NEW
    }


def main():
    print("\n" + "=" * 60)
    print("🧠 EXTRACTING FEATURES (AUTO BATCH, PYTHON SCRIPT)")
    print("=" * 60)

    if not INPUT_ROOT.exists():
        raise SystemExit("❌ Invalid input root path")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    selected_main_folder = resolve_selected_folder()
    print(f"\n📂 Selected main folder:\n{selected_main_folder}")

    # ----- FIX #3: Skip *_Additional folder (Plots + Combined_Report.json) -----
    all_subfolders = [
        str(selected_main_folder / f)
        for f in os.listdir(selected_main_folder)
        if (selected_main_folder / f).is_dir()
    ]

    # Filter out the *_Additional folder (not a window folder)
    window_folders = [
        f for f in all_subfolders
        if not os.path.basename(f).endswith("_Additional")
    ]

    skipped_folders = [
        f for f in all_subfolders
        if os.path.basename(f).endswith("_Additional")
    ]

    window_folders = sorted(window_folders, key=get_win_index_from_folder)

    if len(window_folders) == 0:
        raise SystemExit("❌ No window folders found inside selected main folder")

    print(f"\n✅ Found {len(window_folders)} window folders")
    if skipped_folders:
        print(f"⏭️  Skipped {len(skipped_folders)} non-window folder(s):")
        for sf in skipped_folders:
            print(f"   • {os.path.basename(sf)}")

    processed_ok = []
    processed_failed = []

    # Track total replacement summary
    total_replaced_count = 0
    folders_with_replacements = 0

    for idx, folder_path in enumerate(window_folders, start=1):
        print("\n" + "=" * 60)
        print(f"🔄 PROCESSING WINDOW {idx}/{len(window_folders)}")
        print(f"📂 {folder_path}")
        print("=" * 60)

        try:
            result = process_window(folder_path)

            print("✅ Saved successfully")
            print(f"📁 Window output folder:\n{result['output_folder']}")
            print(f"💾 Table CSV:\n{result['table_csv']}")
            print(f"💾 Flat CSV:\n{result['flat_csv']}")
            print(f"💾 Features JSON:\n{result['feature_json']}")
            print(f"💾 Organized Filtered Config JSON:\n{result['filtered_config_json']}")
            if result.get("signal_plot"):
                print(f"🖼️ Signal Plot:\n{result['signal_plot']}")

            # Print file replacement report for this window
            replaced = result.get("replaced_files", [])
            folder_replaced = result.get("folder_was_replaced", False)

            if folder_replaced:
                print(f"\n♻️  WHOLE FOLDER REPLACED — fresh files written.")
                folders_with_replacements += 1
            else:
                report_replaced_files(replaced, result["output_folder"])
                if replaced:
                    total_replaced_count += len(replaced)
                    folders_with_replacements += 1

            processed_ok.append(result["window_folder"])

        except Exception as e:
            failed_name = os.path.basename(folder_path)
            processed_failed.append((failed_name, str(e)))
            print(f"❌ Failed: {failed_name}")
            print(f"   Reason: {e}")
            print("   Traceback:")
            print(traceback.format_exc())
            continue

    print("\n" + "=" * 60)
    print("📌 FINAL PROCESSING SUMMARY")
    print("=" * 60)

    print(f"✅ Successfully processed: {len(processed_ok)}")
    for name in processed_ok:
        print(f"   - {name}")

    print(f"\n❌ Failed: {len(processed_failed)}")
    for name, reason in processed_failed:
        print(f"   - {name}")
        print(f"     Reason: {reason}")

    # Global replacement summary
    print("\n" + "=" * 60)
    print("♻️ FILE REPLACEMENT SUMMARY")
    print("=" * 60)
    print(f"📊 Total files replaced:        {total_replaced_count}")
    print(f"📁 Folders with replacements:   {folders_with_replacements}")
    print(f"🆕 Folders fully fresh:         {len(processed_ok) - folders_with_replacements}")


if __name__ == "__main__":
    main()