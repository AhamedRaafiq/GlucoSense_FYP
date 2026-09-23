# ==========================================
# STEP 6: COMBINE FEATURES + GLUCOSE LEVEL
# Updated:
# - Compatible with new averaging pipeline output
# - Professional output folder naming with timestamp
# - Whole-folder replacement (wipe + recreate)
# - Pre-existence snapshot captured BEFORE folder wipe
# - FIXED: Batch mode now filters out non-subject folders (e.g., Feature_Plots)
# ==========================================

import os
import re
import json
import shutil
import traceback
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd


# --------------------------------------------------
# USER SETTINGS (PASTE YOUR PATHS HERE)
# --------------------------------------------------
INPUT_FEATURES_ROOT = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\06_Averaged_Features")
METADATA_FILE_PATH  = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\PPG Meta Data Collection Sheet (FYP Meta Data 2026-06-10).xlsx")
OUTPUT_ROOT         = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\07_Final_Data_Set")

# Metadata column name that contains the Subject ID (must match feature file naming)
METADATA_ID_COLUMN = "ID"

# Target variable column name in metadata
GLUCOSE_COLUMN = "Glucose level (mg/dl)"


# --------------------------------------------------
# OUTPUT FOLDER NAMING
# --------------------------------------------------
def build_batch_folder_name(source_folder_name):
    """
    Builds a professional output folder name for batch runs.
    Format: MasterDataset_<source_name>_YYYY-MM-DD_HH-MM-SS
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"MasterDataset_{source_folder_name}_{timestamp}"


# --------------------------------------------------
# FILE REPLACEMENT TRACKING HELPERS
# --------------------------------------------------
def check_existing_file(file_path):
    """Check if a file already exists. Returns dict with size info."""
    p = Path(file_path)
    if p.exists() and p.is_file():
        try:
            size_kb = p.stat().st_size / 1024.0
            return {"exists": True, "path": str(p), "size_kb": size_kb}
        except Exception:
            return {"exists": True, "path": str(p), "size_kb": None}
    return {"exists": False, "path": str(p), "size_kb": None}


def snapshot_folder_files(folder_path):
    """Recursively capture all files inside a folder BEFORE wiping it."""
    snapshot = []
    if not folder_path.exists():
        return snapshot
    for root_dir, _, files in os.walk(folder_path):
        for fname in files:
            fp = os.path.join(root_dir, fname)
            info = check_existing_file(fp)
            if info["exists"]:
                snapshot.append({
                    "path": info["path"],
                    "old_size_kb": info["size_kb"],
                    "name": fname,
                })
    return snapshot


def report_replaced_files(replaced_list, location_str):
    """Print a clean terminal report of replaced files."""
    if not replaced_list:
        return
    print(f"\n♻️ REPLACED {len(replaced_list)} EXISTING FILE(S) in:")
    print(f"   📁 {location_str}")
    print("   " + "-" * 56)
    for idx, info in enumerate(replaced_list, start=1):
        file_name = os.path.basename(info["path"])
        old = info.get("old_size_kb")
        new = info.get("new_size_kb")
        old_str = f"{old:.2f} KB" if old is not None else "N/A"
        new_str = f"{new:.2f} KB" if new is not None else "N/A"
        print(f"   {idx}. [{info.get('label', 'File')}] {file_name}")
        print(f"      ↳ Old size: {old_str}  →  New size: {new_str}")


# --------------------------------------------------
# METADATA + FEATURE HELPERS
# --------------------------------------------------
def load_metadata(metadata_path):
    """Load metadata file (Excel or CSV)."""
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    suffix = metadata_path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(metadata_path)
    elif suffix == ".csv":
        df = pd.read_csv(metadata_path)
    else:
        raise ValueError(f"Unsupported metadata format: {suffix}")

    if METADATA_ID_COLUMN not in df.columns:
        raise ValueError(
            f"Metadata file missing required ID column: '{METADATA_ID_COLUMN}'\n"
            f"Available columns: {list(df.columns)}"
        )

    if GLUCOSE_COLUMN not in df.columns:
        raise ValueError(
            f"Metadata file missing glucose column: '{GLUCOSE_COLUMN}'\n"
            f"Available columns: {list(df.columns)}"
        )

    print(f"✅ Loaded metadata: {len(df)} rows, {len(df.columns)} columns")
    return df


def derive_subject_id_from_folder(folder_name):
    """
    Extract subject ID from feature folder name.
    Example: 'Ali(22-enc-12)v1_Features'    → 'Ali(22-enc-12)v1'
             'Ali(22-enc-12)v1_AveFeatures' → 'Ali(22-enc-12)v1'
    """
    name = folder_name
    name = re.sub(r"_AveFeatures$", "", name)
    name = re.sub(r"_Features$", "", name)
    return name.strip()


def find_feature_csv_in_folder(folder_path):
    """Find the *_AveFeature.csv file in the folder."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return None

    # Primary: look for *_AveFeature.csv
    candidates = list(folder.glob("*_AveFeature.csv"))
    if candidates:
        return candidates[0]

    # Fallback: any CSV
    candidates = list(folder.glob("*.csv"))
    if candidates:
        return candidates[0]

    return None


def load_feature_row(feature_csv_path):
    """Load the single-row averaged feature CSV."""
    df = pd.read_csv(feature_csv_path)
    if df.empty:
        raise ValueError(f"Feature CSV is empty: {feature_csv_path}")
    return df.iloc[0]


def match_metadata_row(metadata_df, subject_id):
    """Find the row in metadata where ID matches subject_id (case-insensitive)."""
    sid_clean = str(subject_id).strip().lower()

    matches = metadata_df[
        metadata_df[METADATA_ID_COLUMN].astype(str).str.strip().str.lower() == sid_clean
    ]

    if len(matches) == 0:
        return None
    if len(matches) > 1:
        print(f"    ⚠️ Multiple metadata rows found for ID '{subject_id}'. Using first one.")
    return matches.iloc[0]


def build_combined_row(feature_row, glucose_value):
    """
    Build a single horizontal combined row dict.
    Subject_ID is intentionally omitted to keep the array purely numerical.
    Format: [Features... | Glucose level (mg/dl)]
    """
    combined = {}
    for col, val in feature_row.items():
        combined[col] = val
    combined[GLUCOSE_COLUMN] = glucose_value
    return combined


def verify_combined_row(combined_row, feature_row, glucose_value):
    """Verify feature preservation after combining."""
    feature_mismatches = []
    glucose_mismatch = None

    for col, val in feature_row.items():
        if col not in combined_row:
            feature_mismatches.append(f"Missing feature column: {col}")
            continue
        orig = val
        new = combined_row[col]
        if pd.isna(orig) and pd.isna(new):
            continue
        if pd.isna(orig) != pd.isna(new):
            feature_mismatches.append(f"{col}: NaN mismatch ({orig} vs {new})")
            continue
        try:
            if not np.isclose(float(orig), float(new), equal_nan=True, rtol=1e-9):
                feature_mismatches.append(f"{col}: {orig} != {new}")
        except (ValueError, TypeError):
            if str(orig) != str(new):
                feature_mismatches.append(f"{col}: {orig} != {new}")

    orig_g = glucose_value
    new_g = combined_row.get(GLUCOSE_COLUMN, None)

    if not (pd.isna(orig_g) and pd.isna(new_g)):
        try:
            if not np.isclose(float(orig_g), float(new_g), equal_nan=True, rtol=1e-9):
                glucose_mismatch = f"Glucose: {orig_g} != {new_g}"
        except (ValueError, TypeError):
            if str(orig_g) != str(new_g):
                glucose_mismatch = f"Glucose: {orig_g} != {new_g}"

    return {
        "passed": len(feature_mismatches) == 0 and glucose_mismatch is None,
        "feature_mismatches": feature_mismatches,
        "glucose_mismatch": glucose_mismatch,
    }


# --------------------------------------------------
# POPUP SELECTOR (Single Subject vs Batch Folder Mode)
# --------------------------------------------------
def popup_selector():
    """
    Prompts user to choose between single-subject mode and batch-folder mode.
    Returns: ('single', file_path) OR ('batch', folder_path)
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    choice = messagebox.askyesnocancel(
        title="Select Processing Pipeline Target",
        message=(
            "Select your final compilation workflow configuration:\n\n"
            "✅ YES    → Select ONE specific feature folder (e.g., Ali(22-enc-12)v1_AveFeatures)\n"
            "📁 NO     → Select WHOLE batch folder (contains multiple subfolders)\n"
            "❌ CANCEL → Terminate execution"
        ),
    )

    if choice is None:
        root.destroy()
        raise SystemExit("❌ User cancelled execution.")

    if choice:  # YES → single mode
        selected = filedialog.askdirectory(
            initialdir=str(INPUT_FEATURES_ROOT),
            title="Select ONE feature folder (e.g., Ali(22-enc-12)v1_AveFeatures)"
        )
        root.destroy()
        if not selected:
            raise SystemExit("❌ No folder selected.")
        return ("single", Path(selected))

    else:  # NO → batch mode
        selected = filedialog.askdirectory(
            initialdir=str(INPUT_FEATURES_ROOT),
            title="Select BATCH folder containing multiple subject subfolders"
        )
        root.destroy()
        if not selected:
            raise SystemExit("❌ No batch folder selected.")
        return ("batch", Path(selected))


# --------------------------------------------------
# PROCESS SINGLE FEATURE FOLDER
# --------------------------------------------------
def process_single_feature_folder(feature_folder, metadata_df, output_folder, snapshot_paths_set):
    """
    Process one subject's feature folder:
    1. Find _AveFeature.csv
    2. Match metadata by Subject ID
    3. Append glucose target column
    4. Save combined CSV
    5. Track whether file was pre-existing
    """
    folder_name = feature_folder.name
    subject_id = derive_subject_id_from_folder(folder_name)

    print(f"\n{'─' * 60}")
    print(f"🔄 Processing: {folder_name}")
    print(f"   Subject ID: {subject_id}")

    # Find feature CSV
    feature_csv = find_feature_csv_in_folder(feature_folder)
    if feature_csv is None:
        print(f"   ❌ No feature CSV found in folder.")
        return {"status": "failed", "reason": "no_feature_csv", "subject_id": subject_id}

    print(f"   📄 Source CSV: {feature_csv.name}")

    # Load feature row
    try:
        feature_row = load_feature_row(feature_csv)
    except Exception as e:
        print(f"   ❌ Failed to read feature CSV: {e}")
        return {"status": "failed", "reason": str(e), "subject_id": subject_id}

    # Match metadata
    metadata_row = match_metadata_row(metadata_df, subject_id)
    if metadata_row is None:
        print(f"   ❌ No metadata match for Subject ID: '{subject_id}'")
        return {"status": "failed", "reason": "no_metadata_match", "subject_id": subject_id}

    print(f"   ✅ Metadata matched.")

    glucose_value = metadata_row.get(GLUCOSE_COLUMN, np.nan)
    print(f"   🩸 Glucose: {glucose_value} mg/dL")

    # Build combined row
    combined_row = build_combined_row(feature_row, glucose_value)
    combined_df = pd.DataFrame([combined_row])

    # Integrity verification
    verification = verify_combined_row(combined_row, feature_row, glucose_value)
    if verification["passed"]:
        print(f"   ✅ Integrity check passed.")
    else:
        print(f"   ⚠️ Integrity warnings:")
        for m in verification["feature_mismatches"][:3]:
            print(f"      • {m}")
        if verification["glucose_mismatch"]:
            print(f"      • {verification['glucose_mismatch']}")

    # Save output
    output_folder.mkdir(parents=True, exist_ok=True)
    out_csv = output_folder / f"{subject_id}_Final_Data.csv"

    # Was this file pre-existing (BEFORE script run)?
    was_pre_existing = os.path.normpath(str(out_csv)) in snapshot_paths_set

    combined_df.to_csv(out_csv, index=False)

    print(f"   💾 Saved: {out_csv.name}")
    print(f"   📊 Shape: {len(combined_df.columns)} columns ({len(feature_row)} features + 1 glucose target)")

    return {
        "status": "success",
        "subject_id": subject_id,
        "combined_row": combined_row,
        "output_path": str(out_csv),
        "verification": verification,
        "was_pre_existing": was_pre_existing,
    }


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    print("\n" + "=" * 60)
    print("🩸 PPG FINAL DATASET BUILDER")
    print("   (Features + Glucose Level → Combined Dataset)")
    print("=" * 60)

    if not INPUT_FEATURES_ROOT.exists():
        raise SystemExit(f"❌ Input features root not found: {INPUT_FEATURES_ROOT}")
    if not METADATA_FILE_PATH.exists():
        raise SystemExit(f"❌ Metadata file not found: {METADATA_FILE_PATH}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Loading metadata: {METADATA_FILE_PATH}")
    metadata_df = load_metadata(METADATA_FILE_PATH)

    mode, selected_path = popup_selector()
    print(f"\n📌 Mode: {mode.upper()}")
    print(f"📁 Source: {selected_path}")

    # --------------------------------------------------
    # DETERMINE OUTPUT FOLDER
    # --------------------------------------------------
    if mode == "single":
        feature_folders = [selected_path]
        output_folder = OUTPUT_ROOT
        is_batch_mode = False
    else:
        # Batch mode: build timestamped output folder
        batch_folder_name = build_batch_folder_name(selected_path.name)
        output_folder = OUTPUT_ROOT / batch_folder_name
        feature_folders = [p for p in selected_path.iterdir() if p.is_dir()]
        feature_folders = sorted(feature_folders, key=lambda x: x.name.lower())
        is_batch_mode = True

    print(f"\n✅ Found {len(feature_folders)} folder(s) in source directory")

    # --------------------------------------------------
    # 🆕 BATCH MODE: FILTER OUT NON-SUBJECT FOLDERS
    # --------------------------------------------------
    if is_batch_mode:
        # Filter to only folders that contain a feature CSV
        # This skips folders like "Feature_Plots" that are subfolders inside subject folders
        valid_feature_folders = [
            p for p in feature_folders
            if find_feature_csv_in_folder(p) is not None
        ]
        
        skipped_count = len(feature_folders) - len(valid_feature_folders)
        
        if skipped_count > 0:
            skipped_names = [p.name for p in feature_folders if p not in valid_feature_folders]
            print(f"\n⚠️  Skipped {skipped_count} non-subject folder(s):")
            for name in skipped_names:
                print(f"      • {name}")
        
        if len(valid_feature_folders) == 0:
            raise SystemExit(
                "❌ No valid subject folders found in batch directory.\n\n"
                "   POSSIBLE CAUSES:\n"
                "   1. You selected a SUBJECT folder instead of the PARENT batch folder.\n"
                f"      You selected: {selected_path}\n\n"
                "   2. Expected folder structure for BATCH mode:\n"
                "      📁 06_Averaged_Features/              ← SELECT THIS (PARENT)\n"
                "          ├── 📁 abeeha(23-enc-19)v5_AveFeatures/\n"
                "          │       ├── *_AveFeature.csv\n"
                "          │       └── 📁 Feature_Plots/\n"
                "          ├── 📁 Ali(22-enc-12)v1_AveFeatures/\n"
                "          │       ├── *_AveFeature.csv\n"
                "          │       └── 📁 Feature_Plots/\n"
                "          └── ...\n\n"
                "   SOLUTION:\n"
                "      • For BATCH: Select the PARENT folder (06_Averaged_Features)\n"
                "      • For SINGLE: Select ONE subject folder and answer YES in popup"
            )
        
        feature_folders = valid_feature_folders
        print(f"✅ Valid subject folders after filtering: {len(feature_folders)}")

    print(f"📁 Output folder: {output_folder}")

    # --------------------------------------------------
    # SNAPSHOT EXISTING FILES (BEFORE any modification)
    # --------------------------------------------------
    pre_existing_snapshot = []
    if is_batch_mode:
        # In batch mode, snapshot the target folder if it exists
        pre_existing_snapshot = snapshot_folder_files(output_folder)
    else:
        # In single mode, only the specific output file matters
        target_subject_id = derive_subject_id_from_folder(selected_path.name)
        single_out_file = output_folder / f"{target_subject_id}_Final_Data.csv"
        if single_out_file.exists():
            info = check_existing_file(single_out_file)
            if info["exists"]:
                pre_existing_snapshot.append({
                    "path": info["path"],
                    "old_size_kb": info["size_kb"],
                    "name": single_out_file.name,
                })

    # --------------------------------------------------
    # WIPE OUTPUT FOLDER IF BATCH MODE & FOLDER EXISTS
    # --------------------------------------------------
    folder_was_replaced = False
    if is_batch_mode and output_folder.exists():
        try:
            shutil.rmtree(output_folder)
            folder_was_replaced = True
            print(f"\n♻️  Existing output folder removed and will be recreated fresh:")
            print(f"   📁 {output_folder}")
            print(f"   🗑️  Wiped {len(pre_existing_snapshot)} pre-existing file(s)")
        except Exception as e:
            print(f"⚠️  Could not remove existing folder: {e}")
            print(f"   Proceeding with overwrite of individual files instead.")

    output_folder.mkdir(parents=True, exist_ok=True)

    # Build snapshot path set for fast lookup
    snapshot_paths_set = {os.path.normpath(entry["path"]) for entry in pre_existing_snapshot}
    snapshot_size_map = {os.path.normpath(entry["path"]): entry["old_size_kb"] for entry in pre_existing_snapshot}

    # --------------------------------------------------
    # PROCESS EACH FEATURE FOLDER
    # --------------------------------------------------
    results = []
    all_combined_rows = []

    for fp in feature_folders:
        try:
            res = process_single_feature_folder(fp, metadata_df, output_folder, snapshot_paths_set)
            results.append(res)
            if res["status"] == "success":
                all_combined_rows.append(res["combined_row"])
        except Exception as e:
            print(f"   ❌ Unexpected error: {e}")
            print(traceback.format_exc())
            results.append({"status": "failed", "reason": str(e), "subject_id": fp.name})

    # --------------------------------------------------
    # BATCH MODE: BUILD MASTER DATASET + LOG
    # --------------------------------------------------
    master_csv_path = None
    log_path = None
    if is_batch_mode and len(all_combined_rows) > 0:
        master_df = pd.DataFrame(all_combined_rows)
        master_csv_path = output_folder / f"{selected_path.name}_MASTER_Dataset.csv"

        master_df.to_csv(master_csv_path, index=False)
        print(f"\n💾 MASTER dataset saved: {master_csv_path.name}")
        print(f"   📊 Rows: {len(master_df)}")
        print(f"   📊 Columns: {len(master_df.columns)}")

        # Build log
        log_path = output_folder / f"{selected_path.name}_DataPipeline_BuildLog.json"
        log_data = {
            "build_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_mode": mode,
            "source_features_folder": str(selected_path),
            "metadata_file_used": str(METADATA_FILE_PATH),
            "output_folder": str(output_folder),
            "format_description": "Features Row Vector + Target Glucose Scalar",
            "total_subjects_found": len(results),
            "successful_compilations": sum(1 for r in results if r["status"] == "success"),
            "failed_compilations": sum(1 for r in results if r["status"] != "success"),
            "master_dataset_path": str(master_csv_path),
            "successful_subjects": [r["subject_id"] for r in results if r["status"] == "success"],
            "failed_subjects": [
                {"subject_id": r["subject_id"], "reason": r.get("reason", "unknown")}
                for r in results if r["status"] != "success"
            ],
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
        print(f"   🧾 Build log saved: {log_path.name}")

    # --------------------------------------------------
    # COMPUTE REPLACED FILES INFO
    # --------------------------------------------------
    all_written_paths = [r["output_path"] for r in results if r["status"] == "success"]
    if master_csv_path is not None:
        all_written_paths.append(str(master_csv_path))
    if log_path is not None:
        all_written_paths.append(str(log_path))

    replaced_files_info = []
    for written_path in all_written_paths:
        normalized = os.path.normpath(written_path)
        if normalized in snapshot_paths_set:
            new_info = check_existing_file(written_path)
            if "MASTER" in os.path.basename(written_path):
                label = "MASTER Dataset"
            elif "BuildLog" in os.path.basename(written_path):
                label = "Build Log"
            else:
                label = "Individual Final Data"
            replaced_files_info.append({
                "label": label,
                "path": written_path,
                "old_size_kb": snapshot_size_map.get(normalized),
                "new_size_kb": new_info["size_kb"],
            })

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------
    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = len(results) - success_count
    total_written = len(all_written_paths)
    total_replaced = len(replaced_files_info)
    total_fresh = total_written - total_replaced

    print("\n" + "=" * 60)
    print("📌 PIPELINE EXECUTION SUMMARY")
    print("=" * 60)
    print(f"✅ Successful: {success_count}")
    print(f"❌ Failed:     {fail_count}")

    if fail_count > 0:
        print(f"\n⚠️ Failed subjects:")
        for r in results:
            if r["status"] != "success":
                print(f"   - {r['subject_id']} (reason: {r.get('reason', 'unknown')})")

    print("\n" + "=" * 60)
    print("♻️ FILE REPLACEMENT SUMMARY")
    print("=" * 60)

    if folder_was_replaced:
        print(f"♻️  WHOLE FOLDER WAS WIPED & RECREATED")
        print(f"   📁 {output_folder}")
        print(f"   🗑️  Pre-existing files removed: {len(pre_existing_snapshot)}")

    if replaced_files_info:
        report_replaced_files(replaced_files_info, str(output_folder))
    else:
        print(f"\n🆕 No matching pre-existing files — all output files are newly created.")

    print(f"\n📊 Total files written:     {total_written}")
    print(f"♻️  Files replaced:          {total_replaced}")
    print(f"🆕 Files newly created:     {total_fresh}")

    print("\n" + "=" * 60)
    print("🎉 PIPELINE COMPLETE")
    print("=" * 60)
    print(f"📁 Output: {output_folder}")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()