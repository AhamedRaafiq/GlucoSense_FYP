# ==========================================
# AVERAGE FEATURE EXTRACTION (BATCH-AWARE)
# Updated for compatibility with new feature extraction pipeline
# - Adds BATCH mode (process all *_Features folders in INPUT_ROOT_PATH)
# - Adds SINGLE mode (popup folder selector — original behavior)
# - Preserves all calculations, plot generation, output naming, and file tracking
# ==========================================

import pandas as pd
import numpy as np
import os
import json
import shutil
import traceback
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("Agg")   # use non-interactive backend (no display)
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path


# --------------------------------------------------
# PATHS
# --------------------------------------------------
INPUT_ROOT_PATH = r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\05_Features_"
OUTPUT_ROOT = r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\06_Averaged_Features"


# --------------------------------------------------
# FILE REPLACEMENT TRACKING HELPERS
# --------------------------------------------------
def check_existing_file(file_path):
    """Check if a file already exists before saving."""
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


def report_replaced_files(replaced_list, output_folder_path_str):
    """Print a clean terminal report of replaced files."""
    if not replaced_list:
        return

    print(f"\n♻️ REPLACED {len(replaced_list)} EXISTING FILE(S) in:")
    print(f"   📁 {output_folder_path_str}")
    print("   " + "-" * 56)

    for idx, info in enumerate(replaced_list, start=1):
        file_name = os.path.basename(info["path"])
        old_size = info.get("old_size_kb")
        new_size = info.get("new_size_kb")

        size_old_str = f"{old_size:.2f} KB" if old_size is not None else "N/A"
        size_new_str = f"{new_size:.2f} KB" if new_size is not None else "N/A"

        print(f"   {idx}. [{info.get('label', 'File')}] {file_name}")
        print(f"      ↳ Old size: {size_old_str}  →  New size: {size_new_str}")


# --------------------------------------------------
# CONFIG HELPERS (preserved exactly)
# --------------------------------------------------
def load_json_safe(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ Failed to load JSON: {path}")
        print(f"   Reason: {e}")
        return None


def is_structured_config(cfg):
    """
    Detects whether config has structured metadata.
    Supports BOTH new (lowercase) and old (uppercase) JSON conventions.
    """
    if not isinstance(cfg, dict):
        return False
    return any(
        k in cfg for k in [
            # New keys (from updated feature extraction)
            "metadata", "hyperparameters", "signal_quality",
            "pipeline_diagnostic", "ensemble_RED", "ensemble_IR",
            "extracted_features",
            # Old keys (backward compatibility)
            "Metadata", "Processing_Settings", "Ensemble_Method"
        ]
    )


def is_flat_feature_json(cfg):
    """Detects flat feature JSON (Red_xxx / IR_xxx / Ensemble ratio)."""
    if not isinstance(cfg, dict):
        return False
    keys = list(cfg.keys())
    return any(
        k.startswith("Red_") or
        k.startswith("IR_") or
        k.strip().lower() == "ensemble ratio"
        for k in keys
    )


def get_metadata_from_config(cfg):
    if not isinstance(cfg, dict):
        return {}
    return cfg.get("metadata") or cfg.get("Metadata") or {}


def get_sampling_rate(metadata):
    if not isinstance(metadata, dict):
        return None
    return (
        metadata.get("sampling_rate_fs")
        or metadata.get("Sampling_Rate_FS")
        or None
    )


def get_window_duration(metadata):
    if not isinstance(metadata, dict):
        return None
    return (
        metadata.get("duration_sec")
        or metadata.get("Plot_Duration_Sec")
        or None
    )


def get_hyperparameters_from_config(cfg):
    if not isinstance(cfg, dict):
        return {}
    return cfg.get("hyperparameters") or {}


def get_win_index(name):
    if "_Win" in name:
        try:
            return int(name.split("_Win")[1].split("_")[0])
        except Exception:
            return 999
    return 999


# ==========================================================
# 🆕 BATCH / SINGLE MODE SELECTOR
# ==========================================================
def prompt_processing_mode():
    """Console prompt to choose BATCH vs SINGLE mode."""
    print("\n" + "=" * 70)
    print("  SELECT AVERAGE FEATURE EXTRACTION MODE")
    print("=" * 70)
    print(f"  1) BATCH  — Process ALL *_Features folders inside:")
    print(f"              {INPUT_ROOT_PATH}")
    print(f"  2) SINGLE — Pop up dialog to choose ONE *_Features folder")
    print("=" * 70)
    while True:
        choice = input("  Enter choice [1 or 2]: ").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("  ❌ Invalid choice. Please enter 1 or 2.")


def collect_features_folders(mode):
    """Returns a list of *_Features folders to process."""
    if not os.path.exists(INPUT_ROOT_PATH):
        raise SystemExit(f"❌ Invalid input root path: {INPUT_ROOT_PATH}")

    folders = []

    if mode == 1:
        # BATCH — find all *_Features subfolders inside INPUT_ROOT_PATH
        all_dirs = [
            os.path.join(INPUT_ROOT_PATH, d)
            for d in os.listdir(INPUT_ROOT_PATH)
            if os.path.isdir(os.path.join(INPUT_ROOT_PATH, d))
        ]
        feature_dirs = sorted(
            [d for d in all_dirs if os.path.basename(d).endswith("_Features")],
            key=lambda p: os.path.basename(p).lower()
        )
        if not feature_dirs:
            raise SystemExit(f"❌ No *_Features subject folders found inside: {INPUT_ROOT_PATH}")
        folders = feature_dirs
        print(f"\n📦 BATCH MODE — found {len(folders)} subject folder(s) to process:")
        for f in folders:
            print(f"   • {os.path.basename(f)}")
    else:
        # SINGLE — popup dialog (preserved original behavior)
        selected = None
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            root.update_idletasks()
            root.update()
            selected = filedialog.askdirectory(
                parent=root,
                initialdir=INPUT_ROOT_PATH,
                title="Select the Features folder"
            )
            root.update()
            root.destroy()
        except Exception as tk_err:
            print(f"\n⚠️  GUI dialog failed: {tk_err}")
            print("    Falling back to manual path entry.")
            selected = None

        if not selected:
            print(f"\n📁 Default base path: {INPUT_ROOT_PATH}")
            typed = input("    Type folder path (or press Enter to cancel): ").strip().strip('"').strip("'")
            if typed:
                selected = typed
            else:
                raise SystemExit("❌ No folder selected. Stopping.")

        if not os.path.exists(selected) or not os.path.isdir(selected):
            raise SystemExit(f"❌ Selected folder does not exist: {selected}")

        folders = [selected]
        print(f"\n📁 SINGLE MODE — selected: {selected}")

    return folders


# ==========================================================
# 🆕 PROCESS A SINGLE *_Features FOLDER (was main() body before)
# ==========================================================
def process_single_features_folder(input_features_folder):
    """
    Processes ONE *_Features folder and produces averaged outputs.
    Returns a summary dict with status: "SUCCESS" | "EMPTY" | "FAILED"
    """
    print(f"📂 Processing:\n{input_features_folder}")

    # --------------------------------------------------
    # OUTPUT PATH (preserved naming)
    # --------------------------------------------------
    input_folder_name = os.path.basename(input_features_folder)

    if input_folder_name.endswith("_Features"):
        base_name = input_folder_name[:-len("_Features")]
    else:
        base_name = input_folder_name

    output_folder_name = f"{base_name}_AveFeatures"
    output_folder_path = os.path.join(OUTPUT_ROOT, output_folder_name)

    output_csv_name = f"{base_name}_AveFeature.csv"
    output_csv_path = os.path.join(output_folder_path, output_csv_name)

    print(f"📁 Output folder:\n{output_folder_path}")

    # --------------------------------------------------
    # CAPTURE PRE-EXISTING FILES (BEFORE wiping folder)
    # --------------------------------------------------
    pre_existing_snapshot = []
    if os.path.exists(output_folder_path):
        for root_dir, _, files in os.walk(output_folder_path):
            for fname in files:
                fp = os.path.join(root_dir, fname)
                info = check_existing_file(fp)
                if info["exists"]:
                    pre_existing_snapshot.append({
                        "path": info["path"],
                        "old_size_kb": info["size_kb"],
                        "name": fname,
                    })

    # --------------------------------------------------
    # CLEAN REPLACE: wipe output folder if exists
    # --------------------------------------------------
    folder_was_replaced = False
    if os.path.exists(output_folder_path):
        try:
            shutil.rmtree(output_folder_path)
            folder_was_replaced = True
            print(f"\n♻️  Existing output folder removed and will be recreated fresh:")
            print(f"   📁 {output_folder_path}")
            print(f"   🗑️  Wiped {len(pre_existing_snapshot)} pre-existing file(s)")
        except Exception as e:
            print(f"⚠️  Could not remove existing folder: {e}")
            print(f"   Proceeding with overwrite of individual files instead.")

    os.makedirs(output_folder_path, exist_ok=True)

    # --------------------------------------------------
    # FIND WINDOW FOLDERS
    # --------------------------------------------------
    subfolders = [
        os.path.join(input_features_folder, f)
        for f in os.listdir(input_features_folder)
        if os.path.isdir(os.path.join(input_features_folder, f))
    ]

    subfolders = sorted(subfolders, key=lambda x: get_win_index(os.path.basename(x)))

    print(f"\n✅ Found {len(subfolders)} window folders")

    # --------------------------------------------------
    # LOAD FEATURES (calculations preserved)
    # --------------------------------------------------
    feature_rows = []
    config_files = []
    win_labels = []

    for folder in subfolders:
        files = os.listdir(folder)

        flat = next((f for f in files if f.endswith("_Features_Flat.csv")), None)
        cfg = next((f for f in files if f.endswith("_Filtered_Configuration.json")), None)

        if flat is None:
            print(f"⚠️ Skipping folder (no flat feature CSV): {folder}")
            continue

        flat_path = os.path.join(folder, flat)

        try:
            df = pd.read_csv(flat_path)
        except Exception as e:
            print(f"⚠️ Failed to read feature CSV: {flat_path}\n   Reason: {e}")
            continue

        if df.empty:
            print(f"⚠️ Skipping empty feature CSV: {flat_path}")
            continue

        feature_rows.append(df.iloc[0])

        win_id = get_win_index(os.path.basename(folder))
        win_labels.append(f"Win{win_id}")

        if cfg:
            config_files.append(os.path.join(folder, cfg))
        else:
            print(f"⚠️ No config JSON found in: {folder}")

    if len(feature_rows) == 0:
        print("❌ No valid feature rows found in this subject folder — skipping.")
        return {
            "status": "EMPTY",
            "subject": base_name,
            "input_folder": input_features_folder,
            "output_folder": output_folder_path,
            "windows_used": 0,
            "plots_generated": 0,
            "files_replaced": 0,
            "folder_was_replaced": folder_was_replaced,
            "pre_existing_count": len(pre_existing_snapshot),
        }

    df_all = pd.DataFrame(feature_rows)

    for col in df_all.columns:
        df_all[col] = pd.to_numeric(df_all[col], errors="coerce")

    # --------------------------------------------------
    # COMPUTE AVERAGE (preserved)
    # --------------------------------------------------
    df_avg_flat = pd.DataFrame([df_all.mean(numeric_only=True)])

    # --------------------------------------------------
    # CREATE RED / IR TABLE (preserved)
    # --------------------------------------------------
    features = []
    red_vals = []
    ir_vals = []

    for col in df_avg_flat.columns:

        if col.strip().lower() == "ensemble ratio":
            continue

        if col.startswith("Red_"):
            feature = col.replace("Red_", "")
            red_val = df_avg_flat[col].values[0]

            ir_col = "IR_" + feature

            if ir_col in df_avg_flat.columns:
                ir_val = df_avg_flat[ir_col].values[0]
            else:
                ir_val = np.nan

            features.append(feature)
            red_vals.append(red_val)
            ir_vals.append(ir_val)

    df_table = pd.DataFrame({
        "Feature": features,
        "RED": red_vals,
        "IR": ir_vals
    })

    # --------------------------------------------------
    # EXTRACT ENSEMBLE RATIO (preserved)
    # --------------------------------------------------
    ratio_cols = [c for c in df_avg_flat.columns if c.strip().lower() == "ensemble ratio"]

    if ratio_cols:
        ratio_col = ratio_cols[0]
        ratio_value = df_avg_flat[ratio_col].values[0]
        print(f"✅ Ensemble ratio column found: {ratio_col}")
    else:
        ratio_value = np.nan
        print("⚠️ Ensemble ratio column not found")

    df_ratio = pd.DataFrame({
        "Feature": ["Ensemble Ratio"],
        "Value": [ratio_value]
    })

    print("\n📋 AVERAGED FEATURE TABLE")
    print(df_table.to_string(index=False))

    print("\n📋 ENSEMBLE RATIO")
    print(df_ratio.to_string(index=False))

    # --------------------------------------------------
    # SAFE CONFIG VALIDATION (preserved)
    # --------------------------------------------------
    configs = []
    valid_config_paths = []

    for p in config_files:
        cfg = load_json_safe(p)
        if cfg is not None:
            configs.append(cfg)
            valid_config_paths.append(p)

    sampling_rate = None
    window_duration = None
    ref_hyperparameters = {}
    ref_metadata = {}
    config_validation_notes = []
    validation_status = "No configuration files loaded"

    if len(configs) == 0:
        print("\n⚠️ No configuration files found or loaded successfully")
    else:
        ref = configs[0]
        print(f"\n✅ Loaded {len(configs)} configuration file(s)")

        if is_structured_config(ref):
            ref_metadata = get_metadata_from_config(ref)
            ref_hyperparameters = get_hyperparameters_from_config(ref)

            sampling_rate = get_sampling_rate(ref_metadata)
            window_duration = get_window_duration(ref_metadata)

            if sampling_rate is None:
                print("⚠️ Warning: Reference config missing sampling rate")
            if window_duration is None:
                print("⚠️ Warning: Reference config missing window duration")

            for i, c in enumerate(configs[1:], start=1):
                label = f"config {i}"

                if not is_structured_config(c):
                    msg = f"⚠️ {label} is not a structured config JSON"
                    print(msg)
                    config_validation_notes.append(msg)
                    continue

                c_metadata = get_metadata_from_config(c)
                c_hyperparameters = get_hyperparameters_from_config(c)

                if ref_hyperparameters and c_hyperparameters:
                    if c_hyperparameters != ref_hyperparameters:
                        msg = f"⚠️ Hyperparameters mismatch in {label}"
                        print(msg)
                        config_validation_notes.append(msg)

                c_fs = get_sampling_rate(c_metadata)
                c_dur = get_window_duration(c_metadata)

                if sampling_rate is not None and c_fs is not None and c_fs != sampling_rate:
                    msg = f"⚠️ Sampling rate mismatch in {label}: {c_fs} != {sampling_rate}"
                    print(msg)
                    config_validation_notes.append(msg)

                if window_duration is not None and c_dur is not None and c_dur != window_duration:
                    msg = f"⚠️ Window duration mismatch in {label}: {c_dur} != {window_duration}"
                    print(msg)
                    config_validation_notes.append(msg)

            if len(config_validation_notes) == 0:
                print("✅ Pipeline configuration checked successfully")
                validation_status = "No major configuration mismatches detected"
            else:
                print("⚠️ Configuration check completed with warnings")
                validation_status = "Warnings detected during configuration comparison"

        elif is_flat_feature_json(ref):
            print("ℹ️ Detected flat feature JSON files instead of structured config JSON files")
            validation_status = "Flat feature JSON detected; structured config validation skipped"

        else:
            print("⚠️ Unknown config JSON format")
            validation_status = "Unknown config format"

    # --------------------------------------------------
    # CREATE AVERAGED CONFIG JSON (preserved)
    # --------------------------------------------------
    average_config = {
        "metadata": {
            "base_name": base_name,
            "source_features_folder": input_features_folder,
            "sampling_rate_fs": sampling_rate,
            "window_duration_sec": window_duration,
            "number_of_windows_used": len(win_labels),
            "date_averaged": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "hyperparameters_reference": ref_hyperparameters if isinstance(ref_hyperparameters, dict) else {},
        "averaging_info": {
            "averaging_method": "Column-wise arithmetic mean",
            "windows_used": win_labels,
            "configuration_validation": validation_status,
            "validation_warnings": config_validation_notes
        }
    }

    # --------------------------------------------------
    # BUILD LIST OF ALL PLANNED OUTPUT FILE PATHS
    # --------------------------------------------------
    plot_dir = os.path.join(output_folder_path, "Feature_Plots")
    config_path = os.path.join(output_folder_path, f"{base_name}_AveFeature_Config.json")

    plot_paths = {}
    for feat in df_all.columns:
        safe = "".join(c if c.isalnum() else "_" for c in feat)
        plot_file_path = os.path.join(plot_dir, f"{safe}.png")
        plot_paths[feat] = plot_file_path

    pre_check_files = [
        ("Averaged Features CSV", output_csv_path),
        ("Averaged Config JSON", config_path),
    ]
    for feat, fp in plot_paths.items():
        pre_check_files.append((f"Plot ({feat})", fp))

    # --------------------------------------------------
    # PRE-CHECK: which planned files existed BEFORE this run
    # --------------------------------------------------
    pre_existing_paths = {
        os.path.normpath(entry["path"]): entry
        for entry in pre_existing_snapshot
    }

    existing_before = []
    for label, fp in pre_check_files:
        normalized_fp = os.path.normpath(fp)
        if folder_was_replaced:
            if normalized_fp in pre_existing_paths:
                snap_info = pre_existing_paths[normalized_fp]
                existing_before.append({
                    "label": label,
                    "path": snap_info["path"],
                    "old_size_kb": snap_info["old_size_kb"],
                })
        else:
            info = check_existing_file(fp)
            if info["exists"]:
                existing_before.append({
                    "label": label,
                    "path": info["path"],
                    "old_size_kb": info["size_kb"],
                })

    # --------------------------------------------------
    # GENERATE PLOTS (preserved)
    # --------------------------------------------------
    os.makedirs(plot_dir, exist_ok=True)

    x = np.arange(len(win_labels))

    print(f"\n🖼️  Generating {len(df_all.columns)} feature plots...")

    for feat in df_all.columns:
        y = df_all[feat].values
        avg = np.nanmean(y)

        fig = plt.figure(figsize=(10, 4))

        plt.plot(x, y, marker="o", label="Window Value")
        plt.axhline(avg, linestyle="--", label=f"Average = {avg:.6f}")

        plt.xticks(x, win_labels)
        plt.xlabel("Window")
        plt.ylabel(feat)
        plt.title(feat)

        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        plot_file_path = plot_paths[feat]
        fig.savefig(plot_file_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"✅ All {len(plot_paths)} feature plots saved.")

    # --------------------------------------------------
    # SAVE CSV + CONFIG
    # --------------------------------------------------
    df_avg_flat.to_csv(output_csv_path, index=False)
    print(f"\n💾 Averaged CSV saved: {output_csv_path}")

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(average_config, f, indent=4)
    print(f"💾 Config JSON saved: {config_path}")

    print(f"💾 Plot folder: {plot_dir}")

    # --------------------------------------------------
    # POST-CHECK: compare pre-existing files with new sizes
    # --------------------------------------------------
    replaced_files_info = []
    for entry in existing_before:
        new_info = check_existing_file(entry["path"])
        replaced_files_info.append({
            "label": entry["label"],
            "path": entry["path"],
            "old_size_kb": entry["old_size_kb"],
            "new_size_kb": new_info["size_kb"],
        })

    # --------------------------------------------------
    # PER-SUBJECT SUMMARY
    # --------------------------------------------------
    total_planned = len(pre_check_files)
    total_replaced = len(replaced_files_info)
    total_fresh = total_planned - total_replaced

    print("\n" + "─" * 60)
    print("♻️ FILE REPLACEMENT SUMMARY (this subject)")
    print("─" * 60)

    if folder_was_replaced:
        print(f"♻️  WHOLE FOLDER WAS WIPED & RECREATED")
        print(f"   📁 {output_folder_path}")
        print(f"   🗑️  Pre-existing files removed: {len(pre_existing_snapshot)}")

    if replaced_files_info:
        report_replaced_files(replaced_files_info, output_folder_path)
    else:
        print(f"\n🆕 No matching pre-existing files — all output files are newly created.")

    print(f"\n📊 Total files written:     {total_planned}")
    print(f"♻️  Files replaced:          {total_replaced}")
    print(f"🆕 Files newly created:     {total_fresh}")

    print("\n" + "─" * 60)
    print(f"✅ Subject done — {len(win_labels)} windows averaged")
    print("─" * 60)

    return {
        "status": "SUCCESS",
        "subject": base_name,
        "input_folder": input_features_folder,
        "output_folder": output_folder_path,
        "windows_used": len(win_labels),
        "plots_generated": len(plot_paths),
        "files_written": total_planned,
        "files_replaced": total_replaced,
        "files_fresh": total_fresh,
        "folder_was_replaced": folder_was_replaced,
        "pre_existing_count": len(pre_existing_snapshot),
        "validation_warnings": config_validation_notes,
    }


# ==========================================================
# 🆕 MAIN — BATCH-AWARE
# ==========================================================
def main():
    print("\n" + "=" * 70)
    print("📊 AVERAGING FEATURE VALUES (BATCH-AWARE)")
    print("=" * 70)

    if not os.path.exists(INPUT_ROOT_PATH):
        raise SystemExit(f"❌ Invalid input root path: {INPUT_ROOT_PATH}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    mode = prompt_processing_mode()
    features_folders = collect_features_folders(mode)

    all_reports = []

    for idx, folder in enumerate(features_folders, start=1):
        print("\n" + "=" * 70)
        print(f"📦 [{idx}/{len(features_folders)}] SUBJECT FOLDER")
        print("=" * 70)

        try:
            report = process_single_features_folder(folder)
            all_reports.append(report)
        except SystemExit as e:
            # Local SystemExit from process function — record & continue with batch
            print(f"⚠️  Stopped processing this subject: {e}")
            all_reports.append({
                "status": "FAILED",
                "subject": os.path.basename(folder),
                "input_folder": folder,
                "output_folder": None,
                "error": str(e),
            })
        except Exception as e:
            print(f"❌ Unexpected error for {os.path.basename(folder)}: {e}")
            traceback.print_exc()
            all_reports.append({
                "status": "FAILED",
                "subject": os.path.basename(folder),
                "input_folder": folder,
                "output_folder": None,
                "error": str(e),
            })

    # ==================================================
    # 🆕 GRAND TOTAL SUMMARY (across all subjects)
    # ==================================================
    print("\n" + "=" * 70)
    print("🎯 GRAND TOTAL SUMMARY")
    print("=" * 70)

    n_success = sum(1 for r in all_reports if r.get("status") == "SUCCESS")
    n_empty = sum(1 for r in all_reports if r.get("status") == "EMPTY")
    n_failed = sum(1 for r in all_reports if r.get("status") == "FAILED")

    total_windows = sum(r.get("windows_used", 0) for r in all_reports if r.get("status") == "SUCCESS")
    total_plots = sum(r.get("plots_generated", 0) for r in all_reports if r.get("status") == "SUCCESS")
    total_files_written = sum(r.get("files_written", 0) for r in all_reports if r.get("status") == "SUCCESS")
    total_files_replaced = sum(r.get("files_replaced", 0) for r in all_reports if r.get("status") == "SUCCESS")
    total_files_fresh = sum(r.get("files_fresh", 0) for r in all_reports if r.get("status") == "SUCCESS")
    total_folders_wiped = sum(1 for r in all_reports if r.get("folder_was_replaced"))
    total_pre_existing = sum(r.get("pre_existing_count", 0) for r in all_reports)

    print(f"  Subjects processed       : {len(all_reports)}")
    print(f"  ✅ Successful            : {n_success}")
    print(f"  ⚠️  Empty (no features)  : {n_empty}")
    print(f"  ❌ Failed                : {n_failed}")
    print()
    print(f"  📊 Total windows averaged    : {total_windows}")
    print(f"  🖼️  Total plots generated    : {total_plots}")
    print(f"  💾 Total files written       : {total_files_written}")
    print(f"  ♻️  Total files replaced     : {total_files_replaced}")
    print(f"  🆕 Total files newly created : {total_files_fresh}")
    print(f"  🗑️  Folders fully wiped      : {total_folders_wiped}")
    print(f"  🗑️  Pre-existing files removed: {total_pre_existing}")
    print(f"  📂 Output root               : {OUTPUT_ROOT}")

    # Per-subject breakdown
    print("\n" + "─" * 70)
    print("📋 PER-SUBJECT BREAKDOWN")
    print("─" * 70)
    for r in all_reports:
        status = r.get("status", "?")
        subj = r.get("subject", "?")
        if status == "SUCCESS":
            print(f"  ✅ {subj}  →  windows: {r['windows_used']}, "
                  f"plots: {r['plots_generated']}, replaced: {r['files_replaced']}")
            if r.get("validation_warnings"):
                print(f"     ⚠️  {len(r['validation_warnings'])} config warning(s)")
        elif status == "EMPTY":
            print(f"  ⚠️  {subj}  →  no valid feature CSVs found (likely all rejected)")
        elif status == "FAILED":
            print(f"  ❌ {subj}  →  ERROR: {r.get('error', 'Unknown')}")

    print("\n" + "=" * 70)
    print("🎉 AVERAGING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()