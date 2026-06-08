# ==========================================
# STEP 8 (Sub-task 1 & 2): HANDLE NaN + OUTLIERS
# Updated:
# - Popup message reflects new Step 7 naming (Master dataset with 24F)
# - Output folder naming: Master dataset 24F cleaned YYYY-MM-DD HH-MM-SS
# - Timestamp format: YYYY-MM-DD HH-MM-SS (consistent with Step 7)
# - Validation that selected folder is from Step 7
# - Auto-detection of latest Step 7 output folder with report
# - Step 7 cross-reference in JSON log for full traceability
# - Variable naming cleanup (step1_json_data → step7_json_data)
# - All NaN / Outlier calculations preserved exactly
# ==========================================

import os
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
INPUT_ROOT  = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\08_Data_set_with_24_features")
OUTPUT_ROOT = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\09_Cleaned_dataset_without_(NaN_&_outliers)")


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
TARGET_COLUMN = "Glucose level (mg/dl)"

# IQR multiplier for outlier detection
IQR_MULTIPLIER = 1.5

# Step 7 output identification patterns
STEP7_FOLDER_IDENTIFIER     = "Master dataset with 24F"
STEP7_JSON_PIPELINE_STEP_ID = "STEP 7"


# --------------------------------------------------
# AUTO-DETECT LATEST STEP 7 OUTPUT FOLDER
# --------------------------------------------------
def find_latest_step7_output_folder(root_path):
    """
    Scans INPUT_ROOT for folders whose name contains STEP7_FOLDER_IDENTIFIER.
    Sorts by modification time and returns info about all detected folders.
    This is purely informational — the user still manually selects the folder.

    Returns: dict with detection results.
    """
    root = Path(root_path)
    if not root.exists():
        return {"found": False, "latest_folder": None, "all_folders": [], "total_folders": 0}

    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and STEP7_FOLDER_IDENTIFIER.lower() in p.name.lower()
    ]

    if not candidates:
        return {"found": False, "latest_folder": None, "all_folders": [], "total_folders": 0}

    # Sort by modification time (most recent first)
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    folder_info = []
    for p in candidates:
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mtime = "Unknown"

        # Check contents
        csv_files  = list(p.glob("*.csv"))
        json_files = list(p.glob("*.json"))

        folder_info.append({
            "folder_name":    p.name,
            "full_path":      str(p),
            "last_modified":  mtime,
            "has_csv":        len(csv_files) > 0,
            "csv_name":       csv_files[0].name if csv_files else None,
            "has_json":       len(json_files) > 0,
            "json_name":      json_files[0].name if json_files else None,
        })

    return {
        "found":          True,
        "latest_folder":  folder_info[0],
        "all_folders":    folder_info,
        "total_folders":  len(folder_info),
    }


def print_step7_folder_detection_report(detection_result):
    """
    Prints a clean terminal report of detected Step 7 output folders.
    Purely informational — helps user know which folder to select.
    """
    print(f"\n{'─' * 60}")
    print(f"🔍 STEP 7 OUTPUT FOLDER AUTO-DETECTION REPORT")
    print(f"{'─' * 60}")

    if not detection_result["found"]:
        print(f"   ⚠️  No '{STEP7_FOLDER_IDENTIFIER}' folders found in:")
        print(f"       {INPUT_ROOT}")
        print(f"   Please ensure Step 7 has been run before Step 8.")
        return

    total  = detection_result["total_folders"]
    latest = detection_result["latest_folder"]

    print(f"   📁 Found {total} Step 7 output folder(s) in:")
    print(f"       {INPUT_ROOT}")
    print(f"")
    print(f"   ✅ LATEST (most recently modified):")
    print(f"      📁 {latest['folder_name']}")
    print(f"         Last modified : {latest['last_modified']}")
    print(f"         Has CSV       : {'✅ ' + latest['csv_name'] if latest['has_csv'] else '❌ No CSV'}")
    print(f"         Has JSON      : {'✅ ' + latest['json_name'] if latest['has_json'] else '❌ No JSON'}")
    print(f"")

    if len(detection_result["all_folders"]) > 1:
        print(f"   📋 All detected Step 7 folders (newest → oldest):")
        for idx, info in enumerate(detection_result["all_folders"], start=1):
            marker = "← LATEST" if idx == 1 else ""
            print(f"      {idx}. {info['folder_name']}  {marker}")
            print(f"         Modified : {info['last_modified']}")
            print(f"         CSV      : {info['csv_name'] if info['has_csv'] else 'NOT FOUND'}")
        print(f"")

    print(f"   ℹ️  The folder browser will open at the root folder.")
    print(f"       Please select the Step 7 folder listed above.")
    print(f"{'─' * 60}")


# --------------------------------------------------
# FOLDER SELECTOR POPUP (updated message)
# --------------------------------------------------
def popup_folder_selector(initial_dir):
    """
    Opens a folder dialog for user to select the Step 7 output folder.
    That folder should contain both the CSV and JSON files.
    Returns: Path to selected folder.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    messagebox.showinfo(
        title="Step 8 — Data Cleaning Pipeline",
        message=(
            "Select the Step 7 OUTPUT FOLDER to process.\n\n"
            "──────────────────────────────────────────\n"
            "EXPECTED FOLDER NAME PATTERN:\n"
            "   Master dataset with 24F <timestamp>\n\n"
            "EXPECTED CONTENTS:\n"
            "   Master dataset with 24F <timestamp>/\n"
            "       ├── Master dataset with 24F <timestamp>.csv\n"
            "       └── Master dataset with 24F <timestamp>.json\n\n"
            "──────────────────────────────────────────\n"
            "The script will automatically:\n"
            "  1. Find the CSV and JSON files inside\n"
            "  2. Validate this is a Step 7 output\n"
            "  3. Clean NaN values and clip outliers\n\n"
            "Check the terminal for which folder was\n"
            "detected as the latest one.\n\n"
            "Click OK to open the folder browser."
        ),
    )

    selected_folder = filedialog.askdirectory(
        initialdir=str(initial_dir),
        title="Select Step 7 Output FOLDER (Master dataset with 24F ...)",
    )

    root.destroy()

    if not selected_folder:
        raise SystemExit("❌ User cancelled: No folder selected. Execution terminated.")

    return Path(selected_folder)


# --------------------------------------------------
# FILE FINDERS (unchanged logic)
# --------------------------------------------------
def find_csv_and_json_in_folder(folder_path):
    """
    Automatically find the CSV and JSON files inside the selected folder.
    Returns: (csv_path, json_path)
    """
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

    csv_files  = list(folder.glob("*.csv"))
    json_files = list(folder.glob("*.json"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV file found inside folder: {folder}")
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON file found inside folder: {folder}")

    if len(csv_files) > 1:
        print(f"   ⚠️ Multiple CSV files found. Using first one: {csv_files[0].name}")
    if len(json_files) > 1:
        print(f"   ⚠️ Multiple JSON files found. Using first one: {json_files[0].name}")

    csv_path  = csv_files[0]
    json_path = json_files[0]

    return csv_path, json_path


# --------------------------------------------------
# VALIDATION: IS THIS A STEP 7 OUTPUT?
# --------------------------------------------------
def validate_is_step7_output(folder_path, df, json_data):
    """
    Validates that the selected folder is a genuine Step 7 output.

    Checks:
      1. Folder name contains STEP7_FOLDER_IDENTIFIER
      2. CSV has expected 25 columns (24 features + 1 target)
      3. TARGET_COLUMN exists in CSV
      4. CSV has more than 0 rows
      5. JSON contains pipeline_step = "STEP 7"

    Returns: dict with validation results, warnings, and errors.
    """
    folder_path = Path(folder_path)
    warnings    = []
    errors      = []

    # ── Check 1: Folder name ──
    if STEP7_FOLDER_IDENTIFIER.lower() not in folder_path.name.lower():
        warnings.append(
            f"⚠️  Folder name does not match expected Step 7 pattern.\n"
            f"   Selected : '{folder_path.name}'\n"
            f"   Expected : Contains '{STEP7_FOLDER_IDENTIFIER}'\n"
            f"   Proceeding — but please verify this is a Step 7 output folder."
        )

    # ── Check 2: Column count ──
    expected_columns = 25   # 24 features + 1 target
    if df.shape[1] != expected_columns:
        errors.append(
            f"Column count mismatch.\n"
            f"   Found    : {df.shape[1]} columns\n"
            f"   Expected : {expected_columns} (24 features + 1 target)\n"
            f"   This does not appear to be a valid Step 7 output file."
        )

    # ── Check 3: Target column exists ──
    if TARGET_COLUMN not in df.columns:
        errors.append(
            f"Target column '{TARGET_COLUMN}' not found in CSV.\n"
            f"   Available columns: {list(df.columns)}"
        )

    # ── Check 4: Row count ──
    if df.shape[0] == 0:
        errors.append("CSV has 0 rows. Cannot process an empty dataset.")
    elif df.shape[0] < 3:
        warnings.append(
            f"⚠️  CSV has only {df.shape[0]} row(s).\n"
            f"   Cleaning will proceed, but results may not be meaningful\n"
            f"   with very few samples."
        )

    # ── Check 5: JSON pipeline step ──
    pipeline_step = json_data.get("pipeline_info", {}).get("pipeline_step", "")
    if STEP7_JSON_PIPELINE_STEP_ID.lower() not in pipeline_step.lower():
        warnings.append(
            f"⚠️  JSON log does not identify as Step 7 output.\n"
            f"   pipeline_step found : '{pipeline_step}'\n"
            f"   Expected            : Contains '{STEP7_JSON_PIPELINE_STEP_ID}'\n"
            f"   Proceeding — but verify you selected the correct folder."
        )

    passed = len(errors) == 0

    return {
        "passed":             passed,
        "folder_name":        folder_path.name,
        "column_count":       df.shape[1],
        "row_count":          df.shape[0],
        "has_target_column":  TARGET_COLUMN in df.columns,
        "json_pipeline_step": pipeline_step,
        "warnings":           warnings,
        "errors":             errors,
    }


# --------------------------------------------------
# STEP 7 CROSS-REFERENCE BUILDER
# --------------------------------------------------
def build_step7_reference_section(step7_json_data, step7_json_path, step7_csv_path):
    """
    Builds a detailed cross-reference section for Step 7 to embed
    inside the Step 8 JSON log. Provides full pipeline traceability:
      Step 6 → Step 7 → Step 8

    Returns: dict ready to insert into JSON log.
    """
    if step7_json_data is None:
        return {
            "status":  "not_found",
            "message": (
                "Step 7 JSON log was not found alongside the input CSV. "
                "This may mean the JSON was moved or deleted. "
                "Step 8 can still proceed without it, but traceability is reduced."
            ),
            "step7_log_path": None,
        }

    # Safe extraction helper
    def safe_get(d, *keys, default="Not recorded"):
        current = d
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    pipeline_info      = safe_get(step7_json_data, "pipeline_info",                  default={})
    file_paths         = safe_get(step7_json_data, "file_paths",                     default={})
    dataset_summary    = safe_get(step7_json_data, "dataset_transformation_summary", default={})
    feature_comp       = safe_get(step7_json_data, "feature_composition",            default={})
    verification       = safe_get(step7_json_data, "verification_results",           default={})
    step6_ref          = safe_get(step7_json_data, "step6_pipeline_reference",       default={})
    batch_detection    = safe_get(step7_json_data, "batch_folder_detection",         default={})

    reference_section = {
        "status":         "found",
        "step7_log_file": str(step7_json_path),

        "pipeline_provenance": {
            "description": (
                "This section records the Step 7 pipeline run that produced the "
                "engineered dataset used as input to this Step 8 cleaning run. "
                "It provides full traceability: Raw Features → Step 6 (Combine) → "
                "Step 7 (Feature Engineering) → Step 8 (Cleaning)."
            ),
            "step7_pipeline_name":        safe_get(pipeline_info, "pipeline_name"),
            "step7_pipeline_step":        safe_get(pipeline_info, "pipeline_step"),
            "step7_execution_timestamp":  safe_get(pipeline_info, "execution_timestamp"),
            "step7_execution_date":       safe_get(pipeline_info, "execution_date_readable"),
            "step7_input_master_csv":     safe_get(file_paths,    "input_master_csv"),
            "step7_output_engineered_csv": safe_get(file_paths,   "output_engineered_csv"),
            "step7_csv_used_in_step8":    str(step7_csv_path),
        },

        "feature_engineering_summary": {
            "tier_1_ir_base_count":    safe_get(feature_comp, "tier_1_ir_base_count"),
            "tier_1_description":      safe_get(feature_comp, "tier_1_description"),
            "tier_2_engineered_count": safe_get(feature_comp, "tier_2_engineered_count"),
            "tier_2_description":      safe_get(feature_comp, "tier_2_description"),
            "tier_3_keep_as_is_count": safe_get(feature_comp, "tier_3_keep_as_is_count"),
            "tier_3_description":      safe_get(feature_comp, "tier_3_description"),
            "total_features":          safe_get(feature_comp, "total_features"),
            "total_columns":           safe_get(feature_comp, "total_columns"),
            "target_column":           TARGET_COLUMN,
        },

        "dataset_transformation_at_step7": {
            "original_total_columns":    safe_get(dataset_summary, "original_total_columns"),
            "original_total_rows":       safe_get(dataset_summary, "original_total_rows"),
            "engineered_total_columns":  safe_get(dataset_summary, "engineered_total_columns"),
            "engineered_total_rows":     safe_get(dataset_summary, "engineered_total_rows"),
            "reduction_summary":         safe_get(dataset_summary, "reduction_summary"),
        },

        "step7_verification_results": verification,

        "upstream_pipeline_chain": {
            "description": (
                "Step 7 may contain a cross-reference to Step 6, which in turn "
                "references the original averaged features and metadata. "
                "This section preserves that upstream chain for full audit trail."
            ),
            "step6_reference_from_step7":  step6_ref,
            "batch_detection_from_step7":  batch_detection,
        },

        "data_integrity_notes": {
            "description": (
                "The number of columns and rows entering Step 8 should match "
                "what Step 7 produced. Any mismatch indicates manual editing "
                "between pipeline steps."
            ),
            "expected_columns_from_step7": safe_get(feature_comp, "total_columns"),
            "expected_rows_from_step7":    safe_get(dataset_summary, "engineered_total_rows"),
        },

        "full_step7_log_raw": step7_json_data,
    }

    return reference_section


# --------------------------------------------------
# FILE LOADERS (unchanged)
# --------------------------------------------------
def load_csv(file_path):
    """Load a CSV file and return DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected CSV file, got: {file_path.suffix}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"CSV file is empty: {file_path}")

    print(f"✅ Loaded CSV: {file_path.name}")
    print(f"   📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def load_json(file_path):
    """Load a JSON file and return dict."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".json":
        raise ValueError(f"Expected JSON file, got: {file_path.suffix}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"✅ Loaded JSON: {file_path.name}")
    print(f"   📊 Top-level keys: {len(data)}")
    return data


def check_existing_file(file_path):
    """Check if file already exists and return info."""
    p = Path(file_path)
    if p.exists() and p.is_file():
        try:
            size_kb = p.stat().st_size / 1024.0
            return {"exists": True, "path": str(p), "size_kb": size_kb}
        except Exception:
            return {"exists": True, "path": str(p), "size_kb": None}
    return {"exists": False, "path": str(p), "size_kb": None}


# --------------------------------------------------
# SUB-TASK 1: HANDLE NaN VALUES
# ⚠️ ALL CALCULATIONS PRESERVED EXACTLY — NO CHANGES
# --------------------------------------------------
def analyze_nan_values(df):
    """
    Analyze NaN values across entire DataFrame.
    Returns: detailed NaN analysis dict.
    """
    print(f"\n{'─' * 60}")
    print(f"🔍 SUB-TASK 1: NaN ANALYSIS")
    print(f"{'─' * 60}")

    nan_analysis = {
        "total_nan_count": 0,
        "total_cells": int(df.shape[0] * df.shape[1]),
        "columns_with_nan": [],
        "columns_without_nan": [],
        "target_nan_rows": [],
        "feature_nan_details": [],
    }

    total_nans = 0

    for col in df.columns:
        col_nan_count = int(df[col].isna().sum())
        total_nans += col_nan_count

        if col_nan_count > 0:
            nan_row_indices = df.index[df[col].isna()].tolist()

            col_info = {
                "column": col,
                "nan_count": col_nan_count,
                "nan_percentage": round(col_nan_count / len(df) * 100, 2),
                "nan_row_indices": nan_row_indices,
                "is_target": col == TARGET_COLUMN,
            }

            nan_analysis["columns_with_nan"].append(col_info)

            if col == TARGET_COLUMN:
                nan_analysis["target_nan_rows"] = nan_row_indices
                print(f"   🩸 {col}: {col_nan_count} NaN(s) at rows {nan_row_indices}  ← TARGET")
            else:
                print(f"   ⚠️ {col}: {col_nan_count} NaN(s) at rows {nan_row_indices}")
        else:
            nan_analysis["columns_without_nan"].append(col)
            print(f"   ✅ {col}: No NaN")

    nan_analysis["total_nan_count"] = total_nans

    print(f"\n   📊 Total NaN cells: {total_nans} / {nan_analysis['total_cells']} "
          f"({round(total_nans / nan_analysis['total_cells'] * 100, 4) if nan_analysis['total_cells'] > 0 else 0}%)")
    print(f"   📊 Columns with NaN: {len(nan_analysis['columns_with_nan'])}")
    print(f"   📊 Columns clean: {len(nan_analysis['columns_without_nan'])}")

    if nan_analysis["target_nan_rows"]:
        print(f"\n   🚨 TARGET column has NaN at {len(nan_analysis['target_nan_rows'])} row(s)!")
        print(f"      These rows will be DROPPED entirely (cannot train without glucose label).")
    else:
        print(f"\n   ✅ TARGET column has no NaN values.")

    return nan_analysis


def handle_nan_values(df, nan_analysis):
    """
    Handle NaN values:
    1. Drop rows where target (glucose) is NaN
    2. Impute remaining NaN in features using MEDIAN

    Returns: (cleaned_df, nan_handling_log)
    """
    print(f"\n{'─' * 60}")
    print(f"🔧 SUB-TASK 1: NaN HANDLING")
    print(f"{'─' * 60}")

    df_clean = df.copy()
    nan_handling_log = {
        "rows_dropped_due_to_target_nan": [],
        "feature_imputations": [],
        "total_values_imputed": 0,
        "total_rows_dropped": 0,
        "imputation_method": "median",
    }

    # ── Step A: Drop rows where target glucose is NaN ──
    target_nan_rows = nan_analysis["target_nan_rows"]

    if target_nan_rows:
        print(f"\n   🗑️ DROPPING {len(target_nan_rows)} row(s) with NaN target (glucose):")
        for row_idx in target_nan_rows:
            row_data = {}
            for col in df_clean.columns:
                val = df_clean.loc[row_idx, col]
                row_data[col] = None if pd.isna(val) else float(val) if isinstance(val, (int, float, np.floating, np.integer)) else str(val)

            print(f"      Row {row_idx}: glucose = NaN → DROPPED")
            nan_handling_log["rows_dropped_due_to_target_nan"].append({
                "row_index": int(row_idx),
                "reason": "Target column (Glucose level) is NaN. Cannot train without label.",
                "row_data_before_drop": row_data,
            })

        df_clean = df_clean.drop(index=target_nan_rows).reset_index(drop=True)
        nan_handling_log["total_rows_dropped"] = len(target_nan_rows)

        print(f"      ✅ Dropped {len(target_nan_rows)} row(s). New shape: {df_clean.shape[0]} rows × {df_clean.shape[1]} columns")
    else:
        print(f"\n   ✅ No target NaN rows to drop.")

    # ── Step B: Impute remaining NaN in feature columns using MEDIAN ──
    feature_columns = [col for col in df_clean.columns if col != TARGET_COLUMN]
    total_imputed = 0

    print(f"\n   🔧 IMPUTING NaN in feature columns using MEDIAN:")

    for col in feature_columns:
        col_nan_count = int(df_clean[col].isna().sum())

        if col_nan_count > 0:
            median_val = float(df_clean[col].median())
            nan_indices = df_clean.index[df_clean[col].isna()].tolist()

            imputation_details = []
            for idx in nan_indices:
                imputation_details.append({
                    "row_index": int(idx),
                    "original_value": "NaN",
                    "imputed_value": median_val,
                    "imputation_method": "median",
                })

            df_clean[col] = df_clean[col].fillna(median_val)
            total_imputed += col_nan_count

            print(f"      ⚠️ {col}:")
            print(f"         NaN count: {col_nan_count}")
            print(f"         Median value used: {median_val}")
            print(f"         Imputed at row(s): {nan_indices}")

            nan_handling_log["feature_imputations"].append({
                "column": col,
                "nan_count": col_nan_count,
                "median_value_used": median_val,
                "nan_row_indices": [int(i) for i in nan_indices],
                "imputation_details": imputation_details,
            })
        else:
            pass

    nan_handling_log["total_values_imputed"] = total_imputed

    if total_imputed == 0:
        print(f"      ✅ No feature NaN values found. No imputation needed.")
    else:
        print(f"\n      📊 Total values imputed: {total_imputed}")

    # ── Final NaN verification ──
    remaining_nans = int(df_clean.isna().sum().sum())
    print(f"\n   🔍 Post-cleaning NaN verification: {remaining_nans} NaN(s) remaining")

    if remaining_nans == 0:
        print(f"   ✅ All NaN values successfully handled.")
    else:
        print(f"   ⚠️ WARNING: {remaining_nans} NaN(s) still present!")

    nan_handling_log["remaining_nans_after_cleaning"] = remaining_nans

    return df_clean, nan_handling_log


# --------------------------------------------------
# SUB-TASK 2: HANDLE OUTLIERS (IQR CLIPPING)
# ⚠️ ALL CALCULATIONS PRESERVED EXACTLY — NO CHANGES
# --------------------------------------------------
def analyze_outliers(df):
    """
    Analyze outliers using IQR method for all feature columns.
    Returns: outlier analysis dict.
    """
    print(f"\n{'─' * 60}")
    print(f"🔍 SUB-TASK 2: OUTLIER ANALYSIS (IQR Method, Multiplier={IQR_MULTIPLIER})")
    print(f"{'─' * 60}")

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    outlier_analysis = {
        "iqr_multiplier": IQR_MULTIPLIER,
        "total_outliers_detected": 0,
        "columns_with_outliers": [],
        "columns_without_outliers": [],
    }

    total_outliers = 0

    for col in feature_columns:
        values = df[col].dropna()

        if len(values) == 0:
            outlier_analysis["columns_without_outliers"].append(col)
            continue

        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr

        below_mask = df[col] < lower_bound
        above_mask = df[col] > upper_bound

        below_indices = df.index[below_mask].tolist()
        above_indices = df.index[above_mask].tolist()

        below_values = df.loc[below_mask, col].tolist()
        above_values = df.loc[above_mask, col].tolist()

        total_col_outliers = len(below_indices) + len(above_indices)
        total_outliers += total_col_outliers

        col_stats = {
            "column": col,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "outliers_below_count": len(below_indices),
            "outliers_above_count": len(above_indices),
            "total_outliers": total_col_outliers,
            "outliers_below": [
                {"row_index": int(idx), "original_value": float(val)}
                for idx, val in zip(below_indices, below_values)
            ],
            "outliers_above": [
                {"row_index": int(idx), "original_value": float(val)}
                for idx, val in zip(above_indices, above_values)
            ],
        }

        if total_col_outliers > 0:
            outlier_analysis["columns_with_outliers"].append(col_stats)
            print(f"   ⚠️ {col}:")
            print(f"      Q1={q1:.6f}  Q3={q3:.6f}  IQR={iqr:.6f}")
            print(f"      Bounds: [{lower_bound:.6f}, {upper_bound:.6f}]")
            if below_indices:
                print(f"      Below ({len(below_indices)}): rows {below_indices} → values {[f'{v:.6f}' for v in below_values]}")
            if above_indices:
                print(f"      Above ({len(above_indices)}): rows {above_indices} → values {[f'{v:.6f}' for v in above_values]}")
        else:
            outlier_analysis["columns_without_outliers"].append(col)
            print(f"   ✅ {col}: No outliers  [bounds: {lower_bound:.6f} to {upper_bound:.6f}]")

    outlier_analysis["total_outliers_detected"] = total_outliers

    print(f"\n   📊 Total outliers detected: {total_outliers}")
    print(f"   📊 Columns with outliers: {len(outlier_analysis['columns_with_outliers'])}")
    print(f"   📊 Columns clean: {len(outlier_analysis['columns_without_outliers'])}")

    return outlier_analysis


def clip_outliers(df, outlier_analysis):
    """
    Clip outlier values to IQR bounds for feature columns only.
    Target column is NOT touched.

    Returns: (clipped_df, clipping_log)
    """
    print(f"\n{'─' * 60}")
    print(f"✂️ SUB-TASK 2: OUTLIER CLIPPING")
    print(f"{'─' * 60}")

    df_clipped = df.copy()
    clipping_log = {
        "iqr_multiplier": IQR_MULTIPLIER,
        "target_column_touched": False,
        "total_values_clipped": 0,
        "clipped_features": [],
    }

    total_clipped = 0

    for col_stats in outlier_analysis["columns_with_outliers"]:
        col = col_stats["column"]
        lower_bound = col_stats["lower_bound"]
        upper_bound = col_stats["upper_bound"]

        clip_details = {
            "column": col,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "clipped_values": [],
        }

        for outlier_info in col_stats["outliers_below"]:
            row_idx = outlier_info["row_index"]
            original_val = outlier_info["original_value"]

            df_clipped.at[row_idx, col] = lower_bound
            total_clipped += 1

            clip_detail = {
                "row_index": row_idx,
                "original_value": original_val,
                "clipped_to": lower_bound,
                "direction": "below_lower_bound",
                "change_amount": round(lower_bound - original_val, 10),
            }
            clip_details["clipped_values"].append(clip_detail)

            print(f"   ✂️ {col} [row {row_idx}]: {original_val:.6f} → {lower_bound:.6f} (clipped UP to lower bound)")

        for outlier_info in col_stats["outliers_above"]:
            row_idx = outlier_info["row_index"]
            original_val = outlier_info["original_value"]

            df_clipped.at[row_idx, col] = upper_bound
            total_clipped += 1

            clip_detail = {
                "row_index": row_idx,
                "original_value": original_val,
                "clipped_to": upper_bound,
                "direction": "above_upper_bound",
                "change_amount": round(upper_bound - original_val, 10),
            }
            clip_details["clipped_values"].append(clip_detail)

            print(f"   ✂️ {col} [row {row_idx}]: {original_val:.6f} → {upper_bound:.6f} (clipped DOWN to upper bound)")

        clip_details["total_clipped_in_column"] = len(clip_details["clipped_values"])
        clipping_log["clipped_features"].append(clip_details)

    clipping_log["total_values_clipped"] = total_clipped

    if total_clipped == 0:
        print(f"   ✅ No outliers to clip. Dataset unchanged.")
    else:
        print(f"\n   📊 Total values clipped: {total_clipped}")

    # ── Post-clipping verification ──
    print(f"\n   🔍 Post-clipping outlier verification:")
    remaining_outliers = 0
    feature_columns = [col for col in df_clipped.columns if col != TARGET_COLUMN]

    for col in feature_columns:
        values = df_clipped[col].dropna()
        if len(values) == 0:
            continue

        q1 = float(np.percentile(values, 25))
        q3 = float(np.percentile(values, 75))
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr

        below_count = int((df_clipped[col] < lower_bound).sum())
        above_count = int((df_clipped[col] > upper_bound).sum())
        remaining_outliers += below_count + above_count

    print(f"      Remaining outliers (re-calculated): {remaining_outliers}")
    if remaining_outliers == 0:
        print(f"      ✅ All outliers successfully clipped.")
    else:
        print(f"      ⚠️ Note: {remaining_outliers} outlier(s) remain after clipping.")
        print(f"         This can happen when clipping shifts Q1/Q3/IQR boundaries.")
        print(f"         This is normal and acceptable for single-pass IQR clipping.")

    clipping_log["remaining_outliers_after_clipping"] = remaining_outliers

    return df_clipped, clipping_log


# --------------------------------------------------
# VERIFICATION (unchanged)
# --------------------------------------------------
def verify_cleaned_dataset(cleaned_df, original_df):
    """
    Verify the cleaned dataset integrity.
    Returns: verification result dict.
    """
    print(f"\n{'─' * 60}")
    print(f"🔍 FINAL VERIFICATION")
    print(f"{'─' * 60}")

    checks = []

    # Check 1: Column count preserved
    cols_match = cleaned_df.shape[1] == original_df.shape[1]
    print(f"   {'✅' if cols_match else '❌'} Column count: {cleaned_df.shape[1]} "
          f"(original: {original_df.shape[1]})")
    checks.append({"check": "column_count", "passed": cols_match,
                    "actual": cleaned_df.shape[1], "original": original_df.shape[1]})

    # Check 2: Column names preserved
    names_match = list(cleaned_df.columns) == list(original_df.columns)
    print(f"   {'✅' if names_match else '❌'} Column names preserved: {names_match}")
    checks.append({"check": "column_names", "passed": names_match})

    # Check 3: No NaN remaining
    remaining_nans = int(cleaned_df.isna().sum().sum())
    no_nans = remaining_nans == 0
    print(f"   {'✅' if no_nans else '❌'} No NaN remaining: {remaining_nans} NaN(s)")
    checks.append({"check": "no_nan", "passed": no_nans, "remaining_nans": remaining_nans})

    # Check 4: Row count (may be less if target NaN rows dropped)
    rows_original = original_df.shape[0]
    rows_cleaned = cleaned_df.shape[0]
    rows_dropped = rows_original - rows_cleaned
    print(f"   📊 Rows: {rows_cleaned} (original: {rows_original}, dropped: {rows_dropped})")
    checks.append({"check": "row_count", "original": rows_original,
                    "cleaned": rows_cleaned, "dropped": rows_dropped})

    # Check 5: Target values exist and are valid numbers
    target_valid = True
    if TARGET_COLUMN in cleaned_df.columns:
        target_nans = int(cleaned_df[TARGET_COLUMN].isna().sum())
        if target_nans > 0:
            target_valid = False
        try:
            cleaned_df[TARGET_COLUMN].astype(float)
        except (ValueError, TypeError):
            target_valid = False
    else:
        target_valid = False

    print(f"   {'✅' if target_valid else '❌'} Target column valid: {target_valid}")
    checks.append({"check": "target_valid", "passed": target_valid})

    # Check 6: All feature columns are numeric
    feature_columns = [col for col in cleaned_df.columns if col != TARGET_COLUMN]
    all_numeric = True
    non_numeric_cols = []
    for col in feature_columns:
        if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
            all_numeric = False
            non_numeric_cols.append(col)

    print(f"   {'✅' if all_numeric else '❌'} All features numeric: {all_numeric}")
    if not all_numeric:
        print(f"      Non-numeric columns: {non_numeric_cols}")
    checks.append({"check": "all_numeric", "passed": all_numeric,
                    "non_numeric": non_numeric_cols})

    # Check 7: Data statistics summary
    print(f"\n   📊 Cleaned dataset statistics:")
    print(f"      Shape: {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
    print(f"      Features: {len(feature_columns)}")
    print(f"      Target: {TARGET_COLUMN}")

    if TARGET_COLUMN in cleaned_df.columns:
        glucose_stats = cleaned_df[TARGET_COLUMN].describe()
        print(f"      Glucose range: {glucose_stats['min']:.1f} - {glucose_stats['max']:.1f} mg/dL")
        print(f"      Glucose mean:  {glucose_stats['mean']:.1f} mg/dL")
        print(f"      Glucose std:   {glucose_stats['std']:.1f} mg/dL")

    all_passed = all(c.get("passed", True) for c in checks if "passed" in c)
    print(f"\n   {'✅ ALL CHECKS PASSED' if all_passed else '⚠️ SOME CHECKS FAILED'}")

    return {"all_passed": all_passed, "checks": checks}


# --------------------------------------------------
# JSON LOG BUILDER (updated with Step 7 cross-reference)
# --------------------------------------------------
def build_cleaning_json_log(
    input_csv_path,
    input_json_path,
    input_folder_path,
    output_csv_path,
    output_json_path,
    output_folder_path,
    original_df,
    cleaned_df,
    step7_json_data,
    nan_analysis,
    nan_handling_log,
    outlier_analysis,
    clipping_log,
    verification_result,
    timestamp_str,
    step7_reference_section,
    step7_detection_result,
):
    """Build comprehensive JSON log for the cleaning pipeline."""

    feature_columns = [col for col in cleaned_df.columns if col != TARGET_COLUMN]
    feature_stats_before = {}
    feature_stats_after  = {}

    for col in feature_columns:
        if col in original_df.columns:
            orig_vals = original_df[col].dropna()
            if len(orig_vals) > 0:
                feature_stats_before[col] = {
                    "mean":   float(orig_vals.mean()),
                    "std":    float(orig_vals.std()),
                    "min":    float(orig_vals.min()),
                    "max":    float(orig_vals.max()),
                    "median": float(orig_vals.median()),
                }

        clean_vals = cleaned_df[col].dropna()
        if len(clean_vals) > 0:
            feature_stats_after[col] = {
                "mean":   float(clean_vals.mean()),
                "std":    float(clean_vals.std()),
                "min":    float(clean_vals.min()),
                "max":    float(clean_vals.max()),
                "median": float(clean_vals.median()),
            }

    full_log = {
        "pipeline_info": {
            "pipeline_name": "Data Cleaning: NaN Handling + Outlier Clipping",
            "pipeline_step": "STEP 8 (Sub-task 1 & 2)",
            "execution_timestamp": timestamp_str,
            "execution_date_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "previous_step": "STEP 7 (Feature Engineering)",
        },

        # NEW: Step 7 cross-reference for full pipeline traceability
        "step7_pipeline_reference": step7_reference_section,

        # NEW: Step 7 folder detection info
        "step7_folder_detection": {
            "description": (
                "Auto-detection scan of Step 7 output folders in the input root. "
                "Informational only — the user manually selected the input folder."
            ),
            "scanned_root":          str(INPUT_ROOT),
            "total_folders_detected": step7_detection_result.get("total_folders", 0),
            "latest_folder_detected": (
                step7_detection_result["latest_folder"]
                if step7_detection_result.get("found") else None
            ),
            "all_detected_folders":  step7_detection_result.get("all_folders", []),
        },

        "file_paths": {
            "input_folder":        str(input_folder_path),
            "input_csv":           str(input_csv_path),
            "input_json_from_step7": str(input_json_path),
            "output_folder":       str(output_folder_path),
            "output_cleaned_csv":  str(output_csv_path),
            "output_json_log":     str(output_json_path),
        },

        "dataset_shape_summary": {
            "input_rows":          int(original_df.shape[0]),
            "input_columns":       int(original_df.shape[1]),
            "output_rows":         int(cleaned_df.shape[0]),
            "output_columns":      int(cleaned_df.shape[1]),
            "rows_dropped":        int(original_df.shape[0] - cleaned_df.shape[0]),
            "columns_unchanged":   int(original_df.shape[1]) == int(cleaned_df.shape[1]),
            "feature_count":       len(feature_columns),
            "target_column":       TARGET_COLUMN,
        },

        "sub_task_1_nan_handling": {
            "description": (
                "NaN values were analyzed across all columns. "
                "Rows with NaN in the target column (Glucose level) were dropped entirely "
                "because the model cannot train without a label. "
                "Remaining NaN values in feature columns were imputed using the MEDIAN "
                "of each respective column. Median is preferred over mean for PPG data "
                "because it is robust to outliers caused by motion artifacts and signal noise."
            ),
            "nan_analysis":  nan_analysis,
            "nan_handling":  nan_handling_log,
        },

        "sub_task_2_outlier_handling": {
            "description": (
                f"Outliers were detected using the IQR (Interquartile Range) method "
                f"with a multiplier of {IQR_MULTIPLIER}. "
                f"Lower bound = Q1 - {IQR_MULTIPLIER} × IQR, "
                f"Upper bound = Q3 + {IQR_MULTIPLIER} × IQR. "
                f"Values outside these bounds were CLIPPED (capped) to the nearest bound. "
                f"The target column (Glucose level) was NOT touched. "
                f"Clipping is preferred over removal because PPG datasets are typically small "
                f"and losing rows would reduce training data."
            ),
            "outlier_analysis": outlier_analysis,
            "clipping_log":     clipping_log,
        },

        "feature_statistics": {
            "before_cleaning": feature_stats_before,
            "after_cleaning":  feature_stats_after,
        },

        "verification_results": verification_result,

        "cleaning_rationale": {
            "why_median_imputation": (
                "Median is robust to outliers and skewed distributions. "
                "PPG signal features often have non-Gaussian distributions due to "
                "motion artifacts, sensor noise, and physiological variability. "
                "Using mean would be distorted by these extreme values."
            ),
            "why_iqr_clipping": (
                "IQR-based clipping caps extreme values without removing entire rows. "
                "This preserves sample count (critical for small PPG datasets) while "
                "preventing extreme values from distorting the scaler in the next step. "
                "The 1.5× IQR multiplier is the standard statistical convention."
            ),
            "why_not_touch_target": (
                "Glucose level is the ground truth measurement from a blood glucometer. "
                "Even extreme glucose values are real clinical measurements and should not "
                "be modified. Clipping glucose would distort the model's learning."
            ),
        },
    }

    return full_log


# --------------------------------------------------
# SAVE OUTPUTS (updated naming)
# --------------------------------------------------
def save_outputs(cleaned_df, json_log, output_folder, timestamp_str):
    """
    Save cleaned dataset and JSON log.
    Folder : Master dataset 24F cleaned <timestamp>
    CSV    : Master dataset 24F cleaned <timestamp>.csv
    JSON   : Master dataset 24F cleaned <timestamp>.json
    """
    folder_name = f"Master dataset 24F cleaned {timestamp_str}"
    output_dir  = output_folder / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_name  = f"Master dataset 24F cleaned {timestamp_str}.csv"
    json_name = f"Master dataset 24F cleaned {timestamp_str}.json"

    csv_path  = output_dir / csv_name
    json_path = output_dir / json_name

    # Check for existing files
    replaced_files = []

    csv_pre = check_existing_file(csv_path)
    if csv_pre["exists"]:
        replaced_files.append({
            "label":       "Cleaned Dataset CSV",
            "path":        str(csv_path),
            "old_size_kb": csv_pre["size_kb"],
        })

    json_pre = check_existing_file(json_path)
    if json_pre["exists"]:
        replaced_files.append({
            "label":       "Cleaning Log JSON",
            "path":        str(json_path),
            "old_size_kb": json_pre["size_kb"],
        })

    # Save CSV
    cleaned_df.to_csv(csv_path, index=False)
    csv_post = check_existing_file(csv_path)
    print(f"\n💾 Saved cleaned dataset : {csv_name}")
    print(f"   📊 Size: {csv_post['size_kb']:.2f} KB")

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_log, f, indent=4, default=str)
    json_post = check_existing_file(json_path)
    print(f"💾 Saved cleaning log    : {json_name}")
    print(f"   📊 Size: {json_post['size_kb']:.2f} KB")

    # Report replaced files
    if replaced_files:
        print(f"\n♻️  REPLACED {len(replaced_files)} existing file(s):")
        for rf in replaced_files:
            new_size = check_existing_file(rf["path"])["size_kb"]
            old_str  = f"{rf['old_size_kb']:.2f} KB" if rf["old_size_kb"] else "N/A"
            new_str  = f"{new_size:.2f} KB"           if new_size          else "N/A"
            print(f"   {rf['label']}: {old_str} → {new_str}")
    else:
        print(f"\n🆕 All output files are newly created.")

    print(f"\n📁 Output folder: {output_dir}")

    return csv_path, json_path, output_dir


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("🧹 STEP 8 (Sub-task 1 & 2): DATA CLEANING PIPELINE")
    print("   Sub-task 1: Handle NaN Values (Median Imputation)")
    print("   Sub-task 2: Handle Outliers (IQR Clipping)")
    print("=" * 70)

    # Validate paths
    if not INPUT_ROOT.exists():
        raise SystemExit(f"❌ Input folder does not exist: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Generate timestamp (consistent format with Step 7)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # ── Step 1: Auto-detect latest Step 7 output folder (informational) ──
    print(f"\n🔍 Scanning for latest Step 7 output folder...")
    step7_detection_result = find_latest_step7_output_folder(INPUT_ROOT)
    print_step7_folder_detection_report(step7_detection_result)

    # ── Step 2: Select input FOLDER ──
    print(f"📂 Opening folder selector at: {INPUT_ROOT}")
    input_folder = popup_folder_selector(INPUT_ROOT)
    print(f"📁 Selected folder : {input_folder.name}")
    print(f"   Full path       : {input_folder}")

    # ── Step 3: Auto-detect CSV and JSON inside the folder ──
    print(f"\n{'─' * 60}")
    print(f"🔍 AUTO-DETECTING FILES INSIDE FOLDER")
    print(f"{'─' * 60}")

    input_csv_path, input_json_path = find_csv_and_json_in_folder(input_folder)
    print(f"   📄 CSV found  : {input_csv_path.name}")
    print(f"   📄 JSON found : {input_json_path.name}")

    # ── Step 4: Load input files ──
    print(f"\n{'─' * 60}")
    print(f"📥 LOADING INPUT FILES")
    print(f"{'─' * 60}")

    original_df    = load_csv(input_csv_path)
    step7_json_data = load_json(input_json_path)

    # Display input column inventory
    print(f"\n📋 Input columns ({len(original_df.columns)}):")
    for i, col in enumerate(original_df.columns, 1):
        if col == TARGET_COLUMN:
            print(f"   {i:2d}. {col}  ← TARGET")
        else:
            print(f"   {i:2d}. {col}")

    # Validate column count
    expected_features = 24
    actual_features   = len(original_df.columns) - 1
    if actual_features != expected_features:
        print(f"\n   ⚠️ Expected {expected_features} features, found {actual_features}")
    else:
        print(f"\n   ✅ Feature count verified: {actual_features} features + 1 target")

    # ── Step 5: Validate this is a Step 7 output ──
    print(f"\n{'─' * 60}")
    print(f"🔍 VALIDATING SELECTED FOLDER IS STEP 7 OUTPUT")
    print(f"{'─' * 60}")

    step7_validation = validate_is_step7_output(input_folder, original_df, step7_json_data)

    # Print warnings
    for warning in step7_validation["warnings"]:
        print(warning)

    # Print errors and abort if any
    if not step7_validation["passed"]:
        print(f"\n❌ STEP 7 OUTPUT VALIDATION FAILED:")
        for error in step7_validation["errors"]:
            print(f"\n   ❌ {error}")
        raise SystemExit(
            "\n❌ Execution aborted: Selected folder is not a valid Step 7 output.\n"
            f"   Please select a '{STEP7_FOLDER_IDENTIFIER}' folder from:\n"
            f"   {INPUT_ROOT}"
        )

    print(f"   ✅ Step 7 output validation passed.")
    print(f"   📊 Columns   : {step7_validation['column_count']}")
    print(f"   📊 Rows      : {step7_validation['row_count']}")
    print(f"   📄 JSON step : {step7_validation['json_pipeline_step']}")

    # ── Step 6: Build Step 7 cross-reference ──
    print(f"\n{'─' * 60}")
    print(f"📋 BUILDING STEP 7 PIPELINE CROSS-REFERENCE")
    print(f"{'─' * 60}")

    step7_reference_section = build_step7_reference_section(
        step7_json_data, input_json_path, input_csv_path
    )

    if step7_reference_section["status"] == "found":
        prov = step7_reference_section.get("pipeline_provenance", {})
        feat = step7_reference_section.get("feature_engineering_summary", {})
        print(f"   ✅ Step 7 cross-reference built successfully.")
        print(f"      Step 7 execution    : {prov.get('step7_execution_date', 'N/A')}")
        print(f"      Total features      : {feat.get('total_features', 'N/A')}")
        print(f"      Total columns       : {feat.get('total_columns', 'N/A')}")

        # Check upstream chain
        upstream = step7_reference_section.get("upstream_pipeline_chain", {})
        step6_ref = upstream.get("step6_reference_from_step7", {})
        if isinstance(step6_ref, dict) and step6_ref.get("status") == "found":
            print(f"      Step 6 reference    : ✅ Found (full pipeline chain preserved)")
        else:
            print(f"      Step 6 reference    : ⚠️  Not found in Step 7 log")
    else:
        print(f"   ⚠️  Step 7 JSON data not available for cross-reference.")

    # ── Step 7: SUB-TASK 1 — Analyze and Handle NaN ──
    nan_analysis = analyze_nan_values(original_df)
    df_after_nan, nan_handling_log = handle_nan_values(original_df, nan_analysis)

    # ── Step 8: SUB-TASK 2 — Analyze and Clip Outliers ──
    outlier_analysis = analyze_outliers(df_after_nan)
    cleaned_df, clipping_log = clip_outliers(df_after_nan, outlier_analysis)

    # ── Step 9: Verify cleaned dataset ──
    verification = verify_cleaned_dataset(cleaned_df, original_df)

    # ── Display before/after comparison ──
    print(f"\n{'─' * 60}")
    print(f"📊 BEFORE vs AFTER COMPARISON")
    print(f"{'─' * 60}")
    feature_columns = [col for col in cleaned_df.columns if col != TARGET_COLUMN]
    print(f"\n   {'Column':<35} {'Before Min':>12} {'After Min':>12} {'Before Max':>12} {'After Max':>12}")
    print(f"   {'─' * 35} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")

    for col in feature_columns:
        if col in original_df.columns:
            b_min = original_df[col].min()
            b_max = original_df[col].max()
        else:
            b_min = b_max = float('nan')
        a_min = cleaned_df[col].min()
        a_max = cleaned_df[col].max()

        changed = ""
        if not (np.isclose(b_min, a_min, rtol=1e-9, equal_nan=True) and
                np.isclose(b_max, a_max, rtol=1e-9, equal_nan=True)):
            changed = " ←"

        print(f"   {col:<35} {b_min:>12.6f} {a_min:>12.6f} {b_max:>12.6f} {a_max:>12.6f}{changed}")

    # ── Step 10: Build JSON log ──
    print(f"\n{'─' * 60}")
    print(f"📝 BUILDING COMPREHENSIVE JSON LOG")
    print(f"{'─' * 60}")

    folder_name = f"Master dataset 24F cleaned {timestamp_str}"
    csv_name    = f"Master dataset 24F cleaned {timestamp_str}.csv"
    json_name   = f"Master dataset 24F cleaned {timestamp_str}.json"
    output_dir  = OUTPUT_ROOT / folder_name

    json_log = build_cleaning_json_log(
        input_csv_path=input_csv_path,
        input_json_path=input_json_path,
        input_folder_path=input_folder,
        output_csv_path=output_dir / csv_name,
        output_json_path=output_dir / json_name,
        output_folder_path=output_dir,
        original_df=original_df,
        cleaned_df=cleaned_df,
        step7_json_data=step7_json_data,
        nan_analysis=nan_analysis,
        nan_handling_log=nan_handling_log,
        outlier_analysis=outlier_analysis,
        clipping_log=clipping_log,
        verification_result=verification,
        timestamp_str=timestamp_str,
        step7_reference_section=step7_reference_section,
        step7_detection_result=step7_detection_result,
    )
    print(f"   ✅ JSON log structure built with {len(json_log)} top-level sections.")

    # ── Step 11: Save outputs ──
    print(f"\n{'─' * 60}")
    print(f"💾 SAVING OUTPUTS")
    print(f"{'─' * 60}")

    csv_path, json_path, output_dir = save_outputs(
        cleaned_df, json_log, OUTPUT_ROOT, timestamp_str
    )

    # ── Final Summary ──
    print(f"\n{'=' * 70}")
    print(f"📌 DATA CLEANING PIPELINE — FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"")
    print(f"   📥 Input folder : {input_folder.name}")
    print(f"      CSV          : {input_csv_path.name}")
    print(f"      JSON         : {input_json_path.name}")
    print(f"      Shape        : {original_df.shape[0]} rows × {original_df.shape[1]} columns")
    print(f"")
    print(f"   📤 Output folder: {output_dir.name}")
    print(f"      CSV          : {csv_path.name}")
    print(f"      JSON         : {json_path.name}")
    print(f"      Shape        : {cleaned_df.shape[0]} rows × {cleaned_df.shape[1]} columns")
    print(f"")
    print(f"   🧹 Sub-task 1 — NaN Handling:")
    print(f"      Total NaN found           : {nan_analysis['total_nan_count']}")
    print(f"      Rows dropped (target NaN) : {nan_handling_log['total_rows_dropped']}")
    print(f"      Values imputed (median)   : {nan_handling_log['total_values_imputed']}")
    print(f"      Imputation method         : Median")
    print(f"")
    print(f"   ✂️  Sub-task 2 — Outlier Clipping:")
    print(f"      IQR multiplier            : {IQR_MULTIPLIER}")
    print(f"      Total outliers detected   : {outlier_analysis['total_outliers_detected']}")
    print(f"      Total values clipped      : {clipping_log['total_values_clipped']}")
    print(f"      Target column touched     : No")
    print(f"")
    print(f"   ✅ Verification : {'ALL PASSED' if verification['all_passed'] else 'SOME CHECKS FAILED'}")
    print(f"")
    print(f"   📁 Output folder : {output_dir}")
    print(f"   📄 Dataset CSV   : {csv_path.name}")
    print(f"   📄 JSON Log      : {json_path.name}")
    print(f"")
    print(f"✅ Data cleaning pipeline completed successfully!")
    print(f"   → Output is ready for Step 8 Sub-task 3 & 4")
    print(f"     (Train/Test Split + RobustScaler Normalization)")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()