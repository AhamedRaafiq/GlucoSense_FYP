# ==========================================
# STEP 3: FEATURE EXTRACTION (AUTO BATCH, POPUP SELECTOR)
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

    fs = cfg.get("Metadata", {}).get("Sampling_Rate_FS", None)
    if fs is None:
        raise ValueError("Sampling_Rate_FS not found in config Metadata")

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

    df_feature_table.to_csv(table_csv_path, index=False)
    df_features_flat.to_csv(flat_csv_path, index=False)

    with open(feature_json_path, "w", encoding="utf-8") as f:
        json.dump(flat_features, f, indent=4)

    with open(dst_config_path, "w", encoding="utf-8") as f:
        json.dump(flat_features, f, indent=4)

    return {
        "window_folder": os.path.basename(folder_path),
        "output_folder": str(output_folder_path),
        "table_csv": str(table_csv_path),
        "flat_csv": str(flat_csv_path),
        "feature_json": str(feature_json_path),
        "flat_json_config": str(dst_config_path),
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

    window_folders = [
        str(selected_main_folder / f)
        for f in os.listdir(selected_main_folder)
        if (selected_main_folder / f).is_dir()
    ]
    window_folders = sorted(window_folders, key=get_win_index_from_folder)

    if len(window_folders) == 0:
        raise SystemExit("❌ No 2nd-step window folders found inside selected 1st-step folder")

    print(f"\n✅ Found {len(window_folders)} window folders")

    processed_ok = []
    processed_failed = []

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
            print(f"💾 Flat JSON config:\n{result['flat_json_config']}")

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


if __name__ == "__main__":
    main()