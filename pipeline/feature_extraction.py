# ==========================================
# STEP 3: FEATURE EXTRACTION (BATCH-AWARE + REJECTED-SAFE)
# Updated for compatibility with new automated signal processing pipeline
# - Detects & skips REJECTED windows (no output folder created)
# - Supports BATCH mode (process all subject folders inside input root)
# - Supports SINGLE mode (popup folder selector)
# - Preserves all calculations, output naming, and downstream compatibility
# ==========================================

import os
import re
import json
import shutil
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
# HELPER FUNCTIONS (calculations preserved exactly)
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
    min_distance = max(1, int(0.33 * fs))
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
    if file_ensemble and file_ensemble.endswith("_Filtered_Ensemble.csv"):
        return file_ensemble.replace("_Filtered_Ensemble.csv", "")
    elif file_full and file_full.endswith("_Filtered_Full.csv"):
        return file_full.replace("_Filtered_Full.csv", "")
    elif file_ensemble:
        return os.path.splitext(file_ensemble)[0].replace("_Filtered_Ensemble", "")
    else:
        return "unknown"


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
    """Loads CSVs + config for a SUCCESS window only."""
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

    # ----- Support both NEW (lowercase) and OLD (uppercase) JSON keys -----
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


# ==========================================================
# 🆕 BATCH / SINGLE MODE SELECTOR (mirrors Automated Signal Processing)
# ==========================================================
def prompt_processing_mode():
    """Console prompt to choose BATCH vs SINGLE mode."""
    print("\n" + "=" * 70)
    print("  SELECT FEATURE EXTRACTION MODE")
    print("=" * 70)
    print(f"  1) BATCH  — Process ALL subject folders inside:")
    print(f"              {INPUT_ROOT}")
    print(f"  2) SINGLE — Pop up dialog to choose ONE subject folder")
    print("=" * 70)
    while True:
        choice = input("  Enter choice [1 or 2]: ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("  ❌ Invalid choice. Please enter 1 or 2.")


def collect_subject_folders(mode):
    """Returns a list of subject folders (e.g. *_Filtered) to process."""
    if not INPUT_ROOT.exists():
        raise SystemExit(f"❌ Invalid input root path: {INPUT_ROOT}")

    folders = []

    if mode == 1:
        # BATCH — find all *_Filtered subfolders inside INPUT_ROOT
        all_dirs = [
            INPUT_ROOT / d for d in os.listdir(INPUT_ROOT)
            if (INPUT_ROOT / d).is_dir()
        ]
        subject_dirs = sorted(
            [d for d in all_dirs if d.name.endswith("_Filtered")],
            key=lambda p: p.name.lower()
        )
        if not subject_dirs:
            raise SystemExit(f"❌ No *_Filtered subject folders found inside: {INPUT_ROOT}")
        folders = subject_dirs
        print(f"\n📦 BATCH MODE — found {len(folders)} subject folder(s) to process:")
        for f in folders:
            print(f"   • {f.name}")
    else:
        # SINGLE — popup dialog (with safe Tk init + fallback)
        selected = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update_idletasks()
            root.update()
            selected = filedialog.askdirectory(
                parent=root,
                initialdir=str(INPUT_ROOT),
                title="Select 1st-step Filtered folder (example: mirzan(...)_Filtered)"
            )
            root.update()
            root.destroy()
        except Exception as tk_err:
            print(f"\n⚠️  GUI dialog failed: {tk_err}")
            print("    Falling back to manual path entry.")
            selected = None

        if not selected:
            print(f"\n📁 Default base path: {INPUT_ROOT}")
            typed = input("    Type folder path (or press Enter to cancel): ").strip().strip('"').strip("'")
            if typed:
                selected = typed
            else:
                raise SystemExit("❌ No folder selected. Stopping.")

        folder = Path(selected)
        if not folder.exists() or not folder.is_dir():
            raise SystemExit(f"❌ Selected folder does not exist: {folder}")

        folders = [folder]
        print(f"\n📁 SINGLE MODE — selected: {folder}")

    return folders


# ==========================================================
# 🆕 CHANGE #1 — REJECTED WINDOW DETECTION (CRITICAL)
# ==========================================================
def detect_window_status(window_folder):
    """
    Checks the config JSON to determine if this window was REJECTED by
    the signal processing pipeline. Returns dict with:
      - status: "SUCCESS" | "REJECTED" | "UNKNOWN"
      - rejection_reason: text (if REJECTED or UNKNOWN)
      - rejection_details: per-channel info (if REJECTED)
      - file_full, file_ensemble, file_config: detected file names
    """
    file_full, file_ensemble, file_config = find_required_files(window_folder)

    info = {
        "status": "UNKNOWN",
        "rejection_reason": None,
        "rejection_details": None,
        "file_full": file_full,
        "file_ensemble": file_ensemble,
        "file_config": file_config,
    }

    if file_config is None:
        info["rejection_reason"] = "No configuration JSON found in folder"
        return info

    cfg_path = os.path.join(window_folder, file_config)
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        info["rejection_reason"] = f"Could not read config JSON: {e}"
        return info

    metadata = cfg.get("metadata") or cfg.get("Metadata") or {}
    pipeline_status = metadata.get("status", "SUCCESS")  # default SUCCESS for old configs

    if pipeline_status == "REJECTED":
        info["status"] = "REJECTED"
        rejection_block = cfg.get("rejection", {})
        info["rejection_reason"] = rejection_block.get("overall_reason", "Window rejected (no reason given)")
        info["rejection_details"] = {
            "IR_channel": rejection_block.get("IR_channel", {}),
            "RED_channel": rejection_block.get("RED_channel", {}),
            "min_valid_beats_ir": rejection_block.get("min_valid_beats_ir"),
            "min_valid_beats_red": rejection_block.get("min_valid_beats_red"),
        }
        return info

    # SUCCESS path: ensure required CSVs exist
    if file_full is None or file_ensemble is None:
        info["status"] = "UNKNOWN"
        missing = []
        if file_full is None:
            missing.append("_Filtered_Full.csv")
        if file_ensemble is None:
            missing.append("_Filtered_Ensemble.csv")
        info["rejection_reason"] = f"Config says SUCCESS but missing: {', '.join(missing)}"
        return info

    info["status"] = "SUCCESS"
    return info


def build_filtered_configuration_summary(cfg, flat_features):
    """Builds an organized Filtered_Configuration JSON preserving original data + features."""
    summary = {}

    metadata = cfg.get("metadata") or cfg.get("Metadata")
    if metadata:
        summary["metadata"] = metadata

    if "hyperparameters" in cfg:
        summary["hyperparameters"] = cfg["hyperparameters"]

    folder_struct = cfg.get("folder_structure") or cfg.get("Folder_Structure")
    if folder_struct:
        summary["folder_structure"] = folder_struct

    if "signal_quality" in cfg:
        summary["signal_quality"] = cfg["signal_quality"]

    if "pipeline_diagnostic" in cfg:
        summary["pipeline_diagnostic"] = cfg["pipeline_diagnostic"]

    if "ppg_features_per_channel" in cfg:
        summary["ppg_features_per_channel"] = cfg["ppg_features_per_channel"]

    if "golden_standard_features" in cfg:
        summary["golden_standard_features"] = cfg["golden_standard_features"]

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

    if "ensemble_RED" in cfg:
        summary["ensemble_RED"] = ensemble_summary(cfg["ensemble_RED"])
    elif "Ensemble_RED" in cfg:
        summary["ensemble_RED"] = ensemble_summary(cfg["Ensemble_RED"])

    if "ensemble_IR" in cfg:
        summary["ensemble_IR"] = ensemble_summary(cfg["ensemble_IR"])
    elif "Ensemble_IR" in cfg:
        summary["ensemble_IR"] = ensemble_summary(cfg["Ensemble_IR"])

    summary["extracted_features"] = flat_features

    return summary


def save_signal_plot(df_full, df_ens, output_folder_path, base_name):
    """4-panel plot of DC / AC / Normalized / Ensemble signals."""
    try:
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

        axes[0].plot(t_full, red_dc, color="darkred", label="Red DC", linewidth=1)
        axes[0].plot(t_full, ir_dc, color="blue", label="IR DC", linewidth=1)
        axes[0].set_title("1. DC Component (Low Pass Filtered) - Baseline Drift")
        axes[0].set_ylabel("Amplitude")
        axes[0].legend(loc="upper right")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(t_full, red_ac, color="red", label="Red AC", linewidth=0.8)
        axes[1].plot(t_full, ir_ac, color="blue", label="IR AC", linewidth=0.8)
        axes[1].set_title("2. AC Component (High Pass Filtered) - Pulsatile Signal")
        axes[1].set_ylabel("Amplitude")
        axes[1].legend(loc="upper right")
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(t_full, red_norm, color="red", label="Red Norm", linewidth=0.8)
        axes[2].plot(t_full, ir_norm, color="blue", label="IR Norm", linewidth=0.8)
        axes[2].set_title("3. Normalized Signal (0 to 1 Scaled) - Shape Analysis")
        axes[2].set_ylabel("Normalized Amp")
        axes[2].legend(loc="upper right")
        axes[2].grid(True, alpha=0.3)

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
    p = Path(file_path)
    if p.exists() and p.is_file():
        try:
            size_bytes = p.stat().st_size
            size_kb = size_bytes / 1024.0
            return {"exists": True, "path": str(p), "size_bytes": size_bytes, "size_kb": size_kb}
        except Exception:
            return {"exists": True, "path": str(p), "size_bytes": None, "size_kb": None}
    return {"exists": False, "path": str(p), "size_bytes": None, "size_kb": None}


def report_replaced_files(replaced_list, output_folder_path):
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


# ==========================================================
# 🆕 CHANGE #2 — process_window() now returns status dict
# ==========================================================
def process_window(folder_path):
    """
    Runs feature extraction for ONE window folder.
    Returns dict with status: "SUCCESS" | "REJECTED" | "FAILED"
    
    For REJECTED windows: NO output folder is created (per user decision).
    """
    folder_basename = os.path.basename(os.path.normpath(folder_path))
    
    # 🆕 CHANGE #1 — Detect status BEFORE attempting feature extraction
    status_info = detect_window_status(folder_path)
    
    # ----- REJECTED PATH (Option B: detect during processing) -----
    if status_info["status"] == "REJECTED":
        # 🆕 CLEANUP — If a stale output folder exists from a previous run
        # (when the window was SUCCESS but is now REJECTED), delete it.
        # This keeps the output directory consistent with the current pipeline state.
        stale_cleanup_info = None
        try:
            # Derive what the output folder path would have been
            file_full = status_info.get("file_full")
            file_ensemble = status_info.get("file_ensemble")
            
            stale_subject_name = derive_subject_base_name(folder_path, file_ensemble, file_full)
            stale_main_folder = OUTPUT_ROOT / f"{stale_subject_name}_Features"
            
            # For rejected windows, derive base name from folder (CSVs don't exist)
            if file_ensemble or file_full:
                stale_base = derive_window_base_name(file_ensemble, file_full)
            else:
                stale_base = re.sub(r"_Filtered$", "", folder_basename)
            
            stale_output_folder = stale_main_folder / f"{stale_base}_Feature"
            
            if stale_output_folder.exists():
                # Count files before deletion (for reporting)
                stale_files_count = sum(
                    1 for _ in stale_output_folder.rglob("*") if _.is_file()
                )
                shutil.rmtree(stale_output_folder)
                stale_cleanup_info = {
                    "removed_path": str(stale_output_folder),
                    "files_removed": stale_files_count,
                }
                print(f"🧹 CLEANUP — removed stale output folder (window now REJECTED):")
                print(f"   📁 {stale_output_folder}")
                print(f"   🗑️  Files removed: {stale_files_count}")
        except Exception as cleanup_err:
            print(f"⚠️  Stale folder cleanup failed: {cleanup_err}")
        
        return {
            "status": "REJECTED",
            "window_folder": folder_basename,
            "rejection_reason": status_info["rejection_reason"],
            "rejection_details": status_info["rejection_details"],
            "output_folder": None,  # Per user decision: NO output folder created
            "stale_cleanup": stale_cleanup_info,  # 🆕 cleanup info (None if nothing to clean)
        }
    
    # ----- UNKNOWN PATH (treat as failed) -----
    if status_info["status"] == "UNKNOWN":
        return {
            "status": "FAILED",
            "window_folder": folder_basename,
            "error": status_info["rejection_reason"] or "Unknown window status",
            "output_folder": None,
        }
    
    # ----- SUCCESS PATH — full feature extraction -----
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

    # ---- RED features (calculations preserved exactly) ----
    red_rr = peak_interval_bpm_hrv(red_norm, fs)
    red_teo = teo_features(red_norm)

    red_features["Shannon Entropy"] = shannon_entropy_signal(red_norm)
    red_features["Spectral Entropy"] = spectral_entropy_signal(red_norm, fs)
    red_features["PPI"] = red_rr["PPI"]
    red_features["BPM"] = red_rr["BPM"]
    red_features["HRV"] = red_rr["HRV"]
    red_features["TEO Mean"] = red_teo["TEO Mean"]
    red_features["TEO std dev"] = red_teo["TEO std dev"]

    # ---- IR features (calculations preserved exactly) ----
    ir_rr = peak_interval_bpm_hrv(ir_norm, fs)
    ir_teo = teo_features(ir_norm)

    ir_features["Shannon Entropy"] = shannon_entropy_signal(ir_norm)
    ir_features["Spectral Entropy"] = spectral_entropy_signal(ir_norm, fs)
    ir_features["PPI"] = ir_rr["PPI"]
    ir_features["BPM"] = ir_rr["BPM"]
    ir_features["HRV"] = ir_rr["HRV"]
    ir_features["TEO Mean"] = ir_teo["TEO Mean"]
    ir_features["TEO std dev"] = ir_teo["TEO std dev"]

    # ---- Ensemble-based features (calculations preserved) ----
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
            rows.append({
                "Feature": feat,
                "Red_Value": combined_features.get(feat, np.nan),
                "IR_Value": np.nan,
            })
        else:
            rows.append({
                "Feature": feat,
                "Red_Value": red_features.get(feat, np.nan),
                "IR_Value": ir_features.get(feat, np.nan),
            })

    df_feature_table = pd.DataFrame(rows)

    # ----- Output paths (preserved naming convention) -----
    subject_base_name = derive_subject_base_name(folder_path, file_ensemble, file_full)
    main_output_folder_name = f"{subject_base_name}_Features"
    main_output_folder_path = OUTPUT_ROOT / main_output_folder_name

    base_name = derive_window_base_name(file_ensemble, file_full)
    output_folder_name = f"{base_name}_Feature"
    output_folder_path = main_output_folder_path / output_folder_name

    # Clean replace
    folder_was_replaced = False
    if output_folder_path.exists():
        try:
            shutil.rmtree(output_folder_path)
            folder_was_replaced = True
            print(f"♻️  Existing folder removed and will be recreated fresh:")
            print(f"   📁 {output_folder_path}")
        except Exception as e:
            print(f"⚠️  Could not remove existing folder: {e}")
            print(f"   Proceeding with overwrite of individual files instead.")

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
            existing_before.append({
                "label": label,
                "path": info["path"],
                "old_size_kb": info["size_kb"],
            })

    df_feature_table.to_csv(table_csv_path, index=False)
    df_features_flat.to_csv(flat_csv_path, index=False)

    with open(feature_json_path, "w", encoding="utf-8") as f:
        json.dump(flat_features, f, indent=4)

    organized_config = build_filtered_configuration_summary(cfg, flat_features)
    with open(dst_config_path, "w", encoding="utf-8") as f:
        json.dump(organized_config, f, indent=4)

    plot_path = save_signal_plot(df_full, df_ens, output_folder_path, output_folder_name)

    replaced_files_info = []
    for entry in existing_before:
        new_info = check_existing_file(entry["path"])
        replaced_files_info.append({
            "label": entry["label"],
            "path": entry["path"],
            "old_size_kb": entry["old_size_kb"],
            "new_size_kb": new_info["size_kb"],
        })

    return {
        "status": "SUCCESS",
        "window_folder": folder_basename,
        "output_folder": str(output_folder_path),
        "table_csv": str(table_csv_path),
        "flat_csv": str(flat_csv_path),
        "feature_json": str(feature_json_path),
        "filtered_config_json": str(dst_config_path),
        "signal_plot": plot_path,
        "replaced_files": replaced_files_info,
        "folder_was_replaced": folder_was_replaced,
    }


# ==========================================================
# 🆕 PROCESS A SINGLE SUBJECT FOLDER
# ==========================================================
def process_subject_folder(subject_folder):
    """Processes all window folders inside ONE subject folder."""
    print("\n" + "=" * 70)
    print(f"📁 SUBJECT: {subject_folder.name}")
    print("=" * 70)

    all_subfolders = [
        str(subject_folder / f)
        for f in os.listdir(subject_folder)
        if (subject_folder / f).is_dir()
    ]

    # Skip *_Additional folder (it's not a window folder)
    window_folders = [
        f for f in all_subfolders
        if not os.path.basename(f).endswith("_Additional")
    ]
    skipped_non_window = [
        f for f in all_subfolders
        if os.path.basename(f).endswith("_Additional")
    ]

    window_folders = sorted(window_folders, key=get_win_index_from_folder)

    if len(window_folders) == 0:
        print(f"  ⚠️  No window folders found — skipping this subject.")
        return {
            "subject": subject_folder.name,
            "success": [],
            "rejected": [],
            "failed": [],
            "total_replaced_files": 0,
            "folders_with_replacements": 0,
        }

    print(f"✅ Found {len(window_folders)} window folders")
    if skipped_non_window:
        print(f"⏭️  Skipped {len(skipped_non_window)} non-window folder(s):")
        for sf in skipped_non_window:
            print(f"   • {os.path.basename(sf)}")

    success_list = []
    rejected_list = []
    failed_list = []

    total_replaced_count = 0
    folders_with_replacements = 0

    # 🆕 CHANGE #3 — Main loop handles SUCCESS / REJECTED / FAILED separately
    for idx, folder_path in enumerate(window_folders, start=1):
        print("\n" + "-" * 70)
        print(f"🔄 [{idx}/{len(window_folders)}] {os.path.basename(folder_path)}")
        print("-" * 70)

        try:
            result = process_window(folder_path)

            # ----- SUCCESS -----
            if result["status"] == "SUCCESS":
                print("✅ Saved successfully")
                print(f"📁 Window output:   {result['output_folder']}")
                print(f"💾 Table CSV:       {os.path.basename(result['table_csv'])}")
                print(f"💾 Flat CSV:        {os.path.basename(result['flat_csv'])}")
                print(f"💾 Features JSON:   {os.path.basename(result['feature_json'])}")
                print(f"💾 Config JSON:     {os.path.basename(result['filtered_config_json'])}")
                if result.get("signal_plot"):
                    print(f"🖼️ Signal Plot:     {os.path.basename(result['signal_plot'])}")

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

                success_list.append(result["window_folder"])

            # ----- REJECTED (skip silently — no output created) -----
            elif result["status"] == "REJECTED":
                print(f"⚠️  REJECTED by signal processing pipeline — SKIPPED (no output created)")
                print(f"   Reason: {result['rejection_reason']}")
                rejected_list.append({
                    "window_folder": result["window_folder"],
                    "reason": result["rejection_reason"],
                })

            # ----- FAILED (status returned by process_window) -----
            elif result["status"] == "FAILED":
                print(f"❌ Failed: {result['window_folder']}")
                print(f"   Reason: {result.get('error', 'Unknown')}")
                failed_list.append((result["window_folder"], result.get("error", "Unknown")))

        except Exception as e:
            # Unexpected crash during feature extraction
            failed_name = os.path.basename(folder_path)
            failed_list.append((failed_name, str(e)))
            print(f"❌ Failed: {failed_name}")
            print(f"   Reason: {e}")
            print("   Traceback:")
            print(traceback.format_exc())
            continue

    return {
        "subject": subject_folder.name,
        "success": success_list,
        "rejected": rejected_list,
        "failed": failed_list,
        "total_replaced_files": total_replaced_count,
        "folders_with_replacements": folders_with_replacements,
    }


# ==========================================================
# 🆕 MAIN — BATCH-AWARE WITH REJECTED HANDLING
# ==========================================================
def main():
    print("\n" + "=" * 70)
    print("🧠 EXTRACTING FEATURES (BATCH AWARE + REJECTED SAFE)")
    print("=" * 70)

    if not INPUT_ROOT.exists():
        raise SystemExit(f"❌ Invalid input root path: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    mode = prompt_processing_mode()
    subject_folders = collect_subject_folders(mode)

    all_subject_reports = []

    for subj_idx, subject_folder in enumerate(subject_folders, start=1):
        print("\n" + "=" * 70)
        print(f"📦 [{subj_idx}/{len(subject_folders)}] STARTING SUBJECT")
        print("=" * 70)

        try:
            report = process_subject_folder(subject_folder)
            all_subject_reports.append(report)
        except Exception as e:
            print(f"❌ Subject-level error for {subject_folder.name}: {e}")
            traceback.print_exc()
            all_subject_reports.append({
                "subject": subject_folder.name,
                "success": [],
                "rejected": [],
                "failed": [(subject_folder.name, str(e))],
                "total_replaced_files": 0,
                "folders_with_replacements": 0,
            })

    # ==================================================
    # FINAL SUMMARY (across all subjects)
    # ==================================================
    print("\n" + "=" * 70)
    print("📌 FINAL BATCH SUMMARY")
    print("=" * 70)

    grand_success = 0
    grand_rejected = 0
    grand_failed = 0
    grand_replaced_files = 0
    grand_folders_with_replacements = 0

    for report in all_subject_reports:
        n_succ = len(report["success"])
        n_rej = len(report["rejected"])
        n_fail = len(report["failed"])
        grand_success += n_succ
        grand_rejected += n_rej
        grand_failed += n_fail
        grand_replaced_files += report["total_replaced_files"]
        grand_folders_with_replacements += report["folders_with_replacements"]

        print(f"\n📁 {report['subject']}")
        print(f"   ✅ Success : {n_succ}")
        print(f"   ⚠️  Rejected: {n_rej}")
        print(f"   ❌ Failed  : {n_fail}")

        if report["rejected"]:
            print(f"   ── Rejected windows (skipped, no output) ──")
            for rej in report["rejected"]:
                print(f"      • {rej['window_folder']}")
                reason_short = rej["reason"]
                if reason_short and len(reason_short) > 100:
                    reason_short = reason_short[:97] + "..."
                print(f"        Reason: {reason_short}")

        if report["failed"]:
            print(f"   ── Failed windows ──")
            for fname, reason in report["failed"]:
                print(f"      • {fname}")
                reason_short = reason
                if reason_short and len(reason_short) > 100:
                    reason_short = reason_short[:97] + "..."
                print(f"        Reason: {reason_short}")

    print("\n" + "=" * 70)
    print("🎯 GRAND TOTALS")
    print("=" * 70)
    print(f"  Subjects processed          : {len(all_subject_reports)}")
    print(f"  ✅ Successful windows        : {grand_success}")
    print(f"  ⚠️  Rejected windows (skipped): {grand_rejected}")
    print(f"  ❌ Failed windows            : {grand_failed}")
    print(f"  📊 Total files replaced     : {grand_replaced_files}")
    print(f"  📁 Folders with replacements: {grand_folders_with_replacements}")
    print(f"  📂 Output root              : {OUTPUT_ROOT}")
    print("=" * 70)
    print("\n🎉 Feature extraction complete.\n")


if __name__ == "__main__":
    main()