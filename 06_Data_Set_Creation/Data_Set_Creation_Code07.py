# ==========================================
# STEP 6: COMBINE FEATURES + GLUCOSE LEVEL
# ==========================================

import os
import re
import json
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
METADATA_FILE_PATH  = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\PPG Meta Data 21-05-2026 (FYP Meta Data).xlsx")
OUTPUT_ROOT         = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\07_Final_Data_Set")

# Metadata column name that contains the Subject ID (must match feature file naming)
METADATA_ID_COLUMN = "ID"

# Target variable column name in metadata
GLUCOSE_COLUMN = "Glucose level (mg/dl)"


# --------------------------------------------------
# HELPER FUNCTIONS
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
    Example: 'Ali(22-enc-12)v1_Features' → 'Ali(22-enc-12)v1'
             'Ali(22-enc-12)v1_AveFeatures' → 'Ali(22-enc-12)v1'
    """
    name = folder_name
    # Remove trailing pattern suffixes to extract raw meta ID matching column 'ID'
    name = re.sub(r"_AveFeatures$", "", name)
    name = re.sub(r"_Features$", "", name)
    return name.strip()


def find_feature_csv_in_folder(folder_path):
    """Find the *_AveFeature.csv file in the folder."""
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        return None

    # Query specifically for files ending with _AveFeature.csv inside the specific subject folder
    candidates = list(folder.glob("*_AveFeature.csv"))
    if candidates:
        return candidates[0]

    # Fallback structure option: check for any regular .csv entry inside the subject root
    candidates = list(folder.glob("*.csv"))
    if candidates:
        return candidates[0]

    return None


def load_feature_row(feature_csv_path):
    """Load the single-row averaged feature CSV."""
    df = pd.read_csv(feature_csv_path)
    if df.empty:
        raise ValueError(f"Feature CSV is empty: {feature_csv_path}")
    return df.iloc[0]  # Return first row vector values


def match_metadata_row(metadata_df, subject_id):
    """Find the row in metadata where ID matches subject_id."""
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
    Build a single horizontal combined row dict framework.
    CRITICAL UPDATE: Subject_ID column stripped from contents to keep array purely mathematical.
    Format: [Features... | Glucose level (mg/dl)]
    """
    combined = {}

    # Extract features iteratively
    for col, val in feature_row.items():
        combined[col] = val

    # Append absolute Target ground-truth glucose scalar to terminal column index
    combined[GLUCOSE_COLUMN] = glucose_value

    return combined


def verify_combined_row(combined_row, feature_row, glucose_value):
    """Recheck structural integration constraints to verify feature preservation metrics."""
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


def check_existing_file(file_path):
    p = Path(file_path)
    if p.exists() and p.is_file():
        try:
            size_kb = p.stat().st_size / 1024.0
            return {"exists": True, "path": str(p), "size_kb": size_kb}
        except Exception:
            return {"exists": True, "path": str(p), "size_kb": None}
    return {"exists": False, "path": str(p), "size_kb": None}


def report_replaced_files(replaced_list, location_str):
    if not replaced_list:
        print(f"🆕 No existing files found — all output files are newly created.")
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
# POPUP SELECTOR (Single Subject vs Batch Folder Mode)
# --------------------------------------------------
def popup_selector():
    """
    Tkinter window prompting manual processing profile selection maps.
    Returns: ('single', file_path) OR ('batch', folder_path)
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    choice = messagebox.askyesnocancel(
        title="Select Processing Pipeline Target",
        message=(
            "Select your final compilation workflow configuration:\n\n"
            "✅ YES    → Select ONE specific feature folder (e.g., Ali(22-enc-12)v1_Features)\n"
            "📁 NO     → Select WHOLE batch folder (contains multiple subfolders)\n"
            "❌ CANCEL → Terminate execution script parameters"
        ),
    )

    if choice is None:
        root.destroy()
        raise SystemExit("❌ User cancelled execution routine configuration.")

    if choice:  # YES → single target mode selection
        selected = filedialog.askdirectory(
            initialdir=str(INPUT_FEATURES_ROOT),
            title="Select ONE feature folder (e.g., Ali(22-enc-12)v1_Features)"
        )
        root.destroy()
        if not selected:
            raise SystemExit("❌ Script runtime terminated: No valid file directory structural link supplied.")
        return ("single", Path(selected))

    else:  # NO → multi-subject batch transformation suite select
        selected = filedialog.askdirectory(
            initialdir=str(INPUT_FEATURES_ROOT),
            title="Select BATCH folder containing multiple subfolders"
        )
        root.destroy()
        if not selected:
            raise SystemExit("❌ Script runtime terminated: No batch processing reference supplied.")
        return ("batch", Path(selected))


# --------------------------------------------------
# TRANSFORMATION CONTROL PIPELINE ENGINE
# --------------------------------------------------
def process_single_feature_folder(feature_folder, metadata_df, output_folder):
    """Extract baseline vector row, cross-reference dataset, save specific customized structural output matrix."""
    folder_name = feature_folder.name
    subject_id = derive_subject_id_from_folder(folder_name)

    print(f"\n{'─' * 60}")
    print(f"🔄 Current Extraction Frame Target: {folder_name}")
    print(f"   Derived Metadata Cross Link Index (Subject ID): {subject_id}")

    # Process internal contents to isolate source feature files metrics matrix
    feature_csv = find_feature_csv_in_folder(feature_folder)
    if feature_csv is None:
        print(f"   ❌ Execution block failure: No validation CSV data target discovered inside folder root.")
        return {"status": "failed", "reason": "no_feature_csv_target", "subject_id": subject_id}

    print(f"   📄 Located Source CSV Target: {feature_csv.name}")

    try:
        feature_row = load_feature_row(feature_csv)
    except Exception as e:
        print(f"   ❌ I/O Runtime execution exception read parsing limits failed: {e}")
        return {"status": "failed", "reason": str(e), "subject_id": subject_id}

    # Locate and map absolute verification ground truth index labels matrix mapping parameters
    metadata_row = match_metadata_row(metadata_df, subject_id)
    if metadata_row is None:
        print(f"   ❌ Matrix cross reference critical alert: No matching registration record link found for value ID: '{subject_id}'")
        return {"status": "failed", "reason": "no_metadata_match_index", "subject_id": subject_id}

    print(f"   ✅ Meta verification success: Found corresponding structural mapping record row reference.")

    glucose_value = metadata_row.get(GLUCOSE_COLUMN, np.nan)
    print(f"   🩸 Coupled Ground Truth Reference Calibration Target: {glucose_value} mg/dL")

    # Construct compiled dataset frame matrix layout sequence parameters 
    # NOTE: subject_id omitted internally to maintain a pure numerical array asset
    combined_row = build_combined_row(feature_row, glucose_value)
    combined_df = pd.DataFrame([combined_row])

    verification = verify_combined_row(combined_row, feature_row, glucose_value)
    if verification["passed"]:
        print(f"   ✅ Integrity test check complete: Frame structures matching bounds maintained.")
    else:
        print(f"   ⚠️ Frame integrity compilation alert indicators caught discrepancy variances:")
        for m in verification["feature_mismatches"][:3]:
            print(f"      Feature structural gap mismatch variance track line info: {m}")
        if verification["glucose_mismatch"]:
            print(f"      {verification['glucose_mismatch']}")

    # Output formatting routine sequence mapping execution rules parameters selection block
    output_folder.mkdir(parents=True, exist_ok=True)
    out_csv = output_folder / f"{subject_id}_Final_Data.csv"

    pre_info = check_existing_file(out_csv)
    was_existing = pre_info["exists"]
    old_size = pre_info["size_kb"]

    combined_df.to_csv(out_csv, index=False)

    print(f"   💾 Saved structural final asset target: {out_csv.name}")
    print(f"   📊 Combined Frame Matrix Shape constraints dimension limits: {len(combined_df.columns)} Columns ({len(feature_row)} Features + [1 Target Glucose Label])")

    replaced_info = None
    if was_existing:
        new_info = check_existing_file(out_csv)
        replaced_info = {
            "label": "Individual Final Combined Data Vector",
            "path": str(out_csv),
            "old_size_kb": old_size,
            "new_size_kb": new_info["size_kb"],
        }

    return {
        "status": "success",
        "subject_id": subject_id,
        "combined_row": combined_row,
        "output_path": str(out_csv),
        "verification": verification,
        "replaced": replaced_info,
    }


def main():
    print("\n" + "=" * 60)
    print("🩸 PPG FINAL DESIGN ENGINE ARTIFACT GENERATOR: DATA PREPARATION PIPELINE")
    print("=" * 60)

    if not INPUT_FEATURES_ROOT.exists():
        raise SystemExit(f"❌ Initialization failure: Target resource directory path structural root link does not exist: {INPUT_FEATURES_ROOT}")
    if not METADATA_FILE_PATH.exists():
        raise SystemExit(f"❌ Initialization failure: Target background parameter verification registry list not found: {METADATA_FILE_PATH}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"\n📂 Opening background control parameters file path register link: {METADATA_FILE_PATH}")
    metadata_df = load_metadata(METADATA_FILE_PATH)

    mode, selected_path = popup_selector()
    print(f"\n📌 Processing Pipeline execution context framework profile assigned: {mode.upper()}")
    print(f"📁 Source Directory path linkage target reference pointer trace: {selected_path}")

    # Establish target mapping directory scanning parameter sequences based on operation flags
    if mode == "single":
        feature_folders = [selected_path]
        output_folder = OUTPUT_ROOT
    else:  # Batch processing engine routine selection maps with version-controlled time-stamping
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        custom_batch_folder_name = f"06_Averaged_Features_Combined_Batch_{timestamp_str}"
        output_folder = OUTPUT_ROOT / custom_batch_folder_name
        
        feature_folders = [p for p in selected_path.iterdir() if p.is_dir()]
        feature_folders = sorted(feature_folders, key=lambda x: x.name.lower())

    print(f"\n✅ Target exploration sequence scanning verified: Found {len(feature_folders)} extraction elements package nodes matching parameter conditions.")
    print(f"📁 Target terminal compilation drop zone link reference assigned: {output_folder}")

    results = []
    all_combined_rows = []
    all_replaced = []

    for fp in feature_folders:
        try:
            res = process_single_feature_folder(fp, metadata_df, output_folder)
            results.append(res)
            if res["status"] == "success":
                all_combined_rows.append(res["combined_row"])
                if res.get("replaced"):
                    all_replaced.append(res["replaced"])
        except Exception as e:
            print(f"   ❌ Processing thread error step failure exception raised target tracking sequence broke trace link: {e}")
            print(traceback.format_exc())
            results.append({"status": "failed", "reason": str(e), "subject_id": fp.name})

    # If the system executes batch processing configuration routines, compile master datasets master records
    if mode == "batch" and len(all_combined_rows) > 0:
        master_df = pd.DataFrame(all_combined_rows)
        master_csv_path = output_folder / f"{selected_path.name}_MASTER_Dataset_Matrix.csv"

        pre_info = check_existing_file(master_csv_path)
        was_existing = pre_info["exists"]
        old_size = pre_info["size_kb"]

        master_df.to_csv(master_csv_path, index=False)
        print(f"\n💾 MASTER data compilation matrix summary structure saved: {master_csv_path}")
        print(f"   📊 Matrix element metrics summary validation row count limits: {len(master_df)} Rows")
        print(f"   📊 Matrix element metrics summary validation column features limits: {len(master_df.columns)} Columns")

        if was_existing:
            new_info = check_existing_file(master_csv_path)
            master_replaced_info = {
                "label": "MASTER BATCH COMPILATION DATASET ASSET MATRIX",
                "path": str(master_csv_path),
                "old_size_kb": old_size,
                "new_size_kb": new_info["size_kb"],
            }
            all_replaced.append(master_replaced_info)

        # Drop execution verification processing status tracking report sheets records targets file elements
        log_path = output_folder / f"{selected_path.name}_DataPipeline_BuildLog.json"
        log_data = {
            "build_date_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "processing_mode_profile": mode,
            "source_features_directory_root": str(selected_path),
            "reference_metadata_calibration_file_path": str(METADATA_FILE_PATH),
            "structural_format_layout_contains": "Unified Extracted Explored Engineered Matrix Array Features Row Vector + Target Glucose Scalar Field Label Only",
            "total_directories_discovered": len(results),
            "successful_compilations": sum(1 for r in results if r["status"] == "success"),
            "failed_compilations": sum(1 for r in results if r["status"] != "success"),
            "master_dataset_csv_asset_link": str(master_csv_path),
            "successfully_processed_subjects_list": [r["subject_id"] for r in results if r["status"] == "success"],
            "failed_subjects_error_tracking_reports": [
                {"directory_node_id": r["subject_id"], "pipeline_fault_exception_reason_trace": r.get("reason", "unknown")}
                for r in results if r["status"] != "success"
            ],
        }
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)
        print(f"   🧾 Execution pipeline logs ledger sheet registered: {log_path.name}")

    print("\n" + "=" * 60)
    print("📌 PIPELINE ENGINE COMPILATION PHASE RUN REPORT SUMMARY")
    print("=" * 60)
    success_count = sum(1 for r in results if r["status"] == "success")
    fail_count = len(results) - success_count
    print(f"✅ Active instances matching tracking constraints successfully updated: {success_count}")
    print(f"❌ Fault error compilation exception instance targets found broken:    {fail_count}")

    if fail_count > 0:
        print(f"\n⚠️ Isolated fault node trace elements inventory lists:")
        for r in results:
            if r["status"] != "success":
                print(f"   - Node directory name trace path element tag: {r['subject_id']} (Fault execution logic exception caught tracking trace description: {r.get('reason', 'unknown')})")

    if all_replaced:
        report_replaced_files(all_replaced, str(output_folder))
        print("\n" + "=" * 60)
        print("♻️ TERMINAL DROP FILES SYSTEM FILE REWRITES REPLACEMENT SUMMARY")
        print("=" * 60)
        print(f"♻️ Total old static target file resources tracking elements overwritten: {len(all_replaced)}")
    else:
        print(f"\n🆕 Environmental storage check complete: All processing outputs newly generated. No resources overwritten.")

    print("\n✅ Execution pipeline runtime tasks completed successfully!\n")


if __name__ == "__main__":
    main()