# ==========================================
# STEP 7: FEATURE ENGINEERING
# IR Base + Selective RED/Ratio Additions
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
INPUT_MASTER_ROOT = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\07_Final_Data_Set")
OUTPUT_ROOT       = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\08_ML_Data_Set_Processing\Data_set_with_24_features")


# --------------------------------------------------
# FEATURE ENGINEERING CONFIGURATION
# --------------------------------------------------

# TIER 1: All 18 IR base features (keep as-is from master CSV)
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

# TIER 2: Selective RED/Ratio engineered features
# Each entry: (new_column_name, operation, operand_1, operand_2)
# Operations: "ratio" = op1 / op2, "difference" = op1 - op2, "keep" = just copy op1
ENGINEERED_FEATURES = [
    ("Ratio_systolic_amplitude", "ratio",      "Red_systolic amplitude", "IR_systolic amplitude"),
    ("Ratio_TEO_Mean",           "ratio",      "Red_TEO Mean",           "IR_TEO Mean"),
    ("Diff_2nd_Derivative_Mean", "difference",  "Red_2nd_Derivative_Mean","IR_2nd_Derivative_Mean"),
    ("Diff_Spectral_Entropy",    "difference",  "Red_Spectral Entropy",   "IR_Spectral Entropy"),
    ("Diff_Dicrotic_notch",      "difference",  "Red_Dicrotic notch",     "IR_Dicrotic notch"),
]

# Features to keep as-is (already combined or single metric)
KEEP_AS_IS_FEATURES = [
    "Ensemble ratio",
]

# Target column
TARGET_COLUMN = "Glucose level (mg/dl)"

# Expected total: 18 IR + 5 engineered + 1 ensemble = 24 features + 1 target = 25 columns


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def popup_file_selector(initial_dir):
    """
    Opens a file dialog for user to select a CSV file.
    Returns: Path to selected file.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    messagebox.showinfo(
        title="Feature Engineering Pipeline",
        message=(
            "Select the MASTER dataset CSV file to process.\n\n"
            "This should be the combined dataset containing\n"
            "all RED, IR features, Ensemble ratio, and Glucose level.\n\n"
            "Click OK to open the file browser."
        ),
    )

    selected_file = filedialog.askopenfilename(
        initialdir=str(initial_dir),
        title="Select MASTER Dataset CSV File",
        filetypes=[
            ("CSV Files", "*.csv"),
            ("All Files", "*.*"),
        ],
    )

    root.destroy()

    if not selected_file:
        raise SystemExit("❌ User cancelled: No file selected. Execution terminated.")

    return Path(selected_file)


def load_master_csv(file_path):
    """Load the master dataset CSV file."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    if file_path.suffix.lower() != ".csv":
        raise ValueError(f"Expected CSV file, got: {file_path.suffix}")

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"CSV file is empty: {file_path}")

    print(f"✅ Loaded master dataset: {file_path.name}")
    print(f"   📊 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    return df


def validate_required_columns(df):
    """
    Check that all required source columns exist in the master CSV.
    Returns: dict with validation results.
    """
    required_columns = set()

    # IR base features
    for col in IR_BASE_FEATURES:
        required_columns.add(col)

    # Source columns for engineered features
    for new_name, operation, op1, op2 in ENGINEERED_FEATURES:
        required_columns.add(op1)
        if operation in ["ratio", "difference"]:
            required_columns.add(op2)

    # Keep-as-is features
    for col in KEEP_AS_IS_FEATURES:
        required_columns.add(col)

    # Target column
    required_columns.add(TARGET_COLUMN)

    available_columns = set(df.columns.tolist())
    missing_columns = required_columns - available_columns
    found_columns = required_columns & available_columns

    result = {
        "all_found": len(missing_columns) == 0,
        "required_count": len(required_columns),
        "found_count": len(found_columns),
        "missing_count": len(missing_columns),
        "missing_columns": sorted(list(missing_columns)),
        "found_columns": sorted(list(found_columns)),
    }

    if result["all_found"]:
        print(f"✅ Column validation passed: All {result['required_count']} required columns found.")
    else:
        print(f"❌ Column validation FAILED:")
        print(f"   Required: {result['required_count']}")
        print(f"   Found:    {result['found_count']}")
        print(f"   Missing:  {result['missing_count']}")
        for col in result["missing_columns"]:
            print(f"      ❌ {col}")

    return result


def compute_engineered_feature(row, operation, operand_1_col, operand_2_col):
    """
    Compute a single engineered feature value for one row.
    Handles NaN, division by zero safely.
    """
    val1 = row.get(operand_1_col, np.nan)
    val2 = row.get(operand_2_col, np.nan)

    if pd.isna(val1) or pd.isna(val2):
        return np.nan

    if operation == "ratio":
        if val2 == 0:
            return np.nan
        return val1 / val2

    elif operation == "difference":
        return val1 - val2

    elif operation == "keep":
        return val1

    else:
        raise ValueError(f"Unknown operation: {operation}")


def build_engineered_dataset(df):
    """
    Build the final engineered dataset with:
    - 18 IR base features
    - 5 engineered RED/Ratio features
    - 1 Ensemble ratio (keep as-is)
    - 1 Target (Glucose level)
    Total: 24 features + 1 target = 25 columns

    Returns: (engineered_df, engineering_log)
    """
    print(f"\n{'─' * 60}")
    print(f"🔧 FEATURE ENGINEERING PIPELINE")
    print(f"{'─' * 60}")

    engineered_data = {}
    engineering_log = {
        "tier_1_ir_base": [],
        "tier_2_engineered": [],
        "tier_3_keep_as_is": [],
        "target": None,
    }

    # ── TIER 1: Extract all 18 IR base features ──
    print(f"\n📌 TIER 1: Extracting {len(IR_BASE_FEATURES)} IR base features...")
    for col in IR_BASE_FEATURES:
        if col in df.columns:
            engineered_data[col] = df[col].values.copy()
            nan_count = df[col].isna().sum()
            print(f"   ✅ {col}  (NaN: {nan_count})")
            engineering_log["tier_1_ir_base"].append({
                "output_column": col,
                "source_column": col,
                "operation": "direct_copy",
                "description": f"IR base feature copied directly from master dataset",
                "nan_count": int(nan_count),
            })
        else:
            print(f"   ❌ {col}  — MISSING from master dataset")
            engineered_data[col] = np.full(len(df), np.nan)
            engineering_log["tier_1_ir_base"].append({
                "output_column": col,
                "source_column": col,
                "operation": "direct_copy",
                "description": "MISSING — filled with NaN",
                "nan_count": len(df),
            })

    # ── TIER 2: Compute engineered features ──
    print(f"\n📌 TIER 2: Computing {len(ENGINEERED_FEATURES)} engineered features...")
    for new_name, operation, op1, op2 in ENGINEERED_FEATURES:
        computed_values = []
        for idx, row in df.iterrows():
            val = compute_engineered_feature(row, operation, op1, op2)
            computed_values.append(val)

        engineered_data[new_name] = computed_values
        nan_count = sum(1 for v in computed_values if pd.isna(v))

        # Build human-readable formula description
        if operation == "ratio":
            formula = f"{op1} / {op2}"
            description = f"Ratio of {op1} to {op2}. Captures inter-wavelength relationship."
        elif operation == "difference":
            formula = f"{op1} - {op2}"
            description = f"Difference between {op1} and {op2}. Captures wavelength-specific variation."
        else:
            formula = f"copy({op1})"
            description = f"Direct copy of {op1}."

        print(f"   ✅ {new_name}")
        print(f"      Formula: {formula}")
        print(f"      NaN count: {nan_count}")

        engineering_log["tier_2_engineered"].append({
            "output_column": new_name,
            "operation": operation,
            "operand_1": op1,
            "operand_2": op2,
            "formula": formula,
            "description": description,
            "nan_count": int(nan_count),
        })

    # ── TIER 3: Keep-as-is features ──
    print(f"\n📌 TIER 3: Keeping {len(KEEP_AS_IS_FEATURES)} as-is features...")
    for col in KEEP_AS_IS_FEATURES:
        if col in df.columns:
            engineered_data[col] = df[col].values.copy()
            nan_count = df[col].isna().sum()
            print(f"   ✅ {col}  (NaN: {nan_count})")
            engineering_log["tier_3_keep_as_is"].append({
                "output_column": col,
                "source_column": col,
                "operation": "direct_copy",
                "description": f"Already a combined/single metric. Kept unchanged.",
                "nan_count": int(nan_count),
            })
        else:
            print(f"   ❌ {col}  — MISSING from master dataset")
            engineered_data[col] = np.full(len(df), np.nan)
            engineering_log["tier_3_keep_as_is"].append({
                "output_column": col,
                "source_column": col,
                "operation": "direct_copy",
                "description": "MISSING — filled with NaN",
                "nan_count": len(df),
            })

    # ── TARGET: Glucose level ──
    print(f"\n📌 TARGET: Appending {TARGET_COLUMN}...")
    if TARGET_COLUMN in df.columns:
        engineered_data[TARGET_COLUMN] = df[TARGET_COLUMN].values.copy()
        nan_count = df[TARGET_COLUMN].isna().sum()
        print(f"   ✅ {TARGET_COLUMN}  (NaN: {nan_count})")
        engineering_log["target"] = {
            "output_column": TARGET_COLUMN,
            "source_column": TARGET_COLUMN,
            "operation": "direct_copy",
            "description": "Ground truth glucose level. Target variable for ML prediction.",
            "nan_count": int(nan_count),
        }
    else:
        print(f"   ❌ {TARGET_COLUMN}  — MISSING from master dataset")
        engineered_data[TARGET_COLUMN] = np.full(len(df), np.nan)
        engineering_log["target"] = {
            "output_column": TARGET_COLUMN,
            "source_column": TARGET_COLUMN,
            "operation": "direct_copy",
            "description": "MISSING — filled with NaN",
            "nan_count": len(df),
        }

    # ── Build DataFrame ──
    engineered_df = pd.DataFrame(engineered_data)

    return engineered_df, engineering_log


def verify_engineered_dataset(engineered_df, original_df, engineering_log):
    """
    Verify the engineered dataset integrity.
    Returns: verification result dict.
    """
    print(f"\n{'─' * 60}")
    print(f"🔍 VERIFICATION CHECKS")
    print(f"{'─' * 60}")

    checks = []

    # Check 1: Row count preserved
    rows_match = engineered_df.shape[0] == original_df.shape[0]
    print(f"   {'✅' if rows_match else '❌'} Row count: {engineered_df.shape[0]} "
          f"(original: {original_df.shape[0]})")
    checks.append({"check": "row_count", "passed": rows_match})

    # Check 2: Expected column count (24 features + 1 target = 25)
    expected_cols = len(IR_BASE_FEATURES) + len(ENGINEERED_FEATURES) + len(KEEP_AS_IS_FEATURES) + 1
    cols_match = engineered_df.shape[1] == expected_cols
    print(f"   {'✅' if cols_match else '❌'} Column count: {engineered_df.shape[1]} "
          f"(expected: {expected_cols})")
    checks.append({"check": "column_count", "passed": cols_match,
                    "actual": engineered_df.shape[1], "expected": expected_cols})

    # Check 3: IR base features match original values
    ir_mismatches = 0
    for col in IR_BASE_FEATURES:
        if col in original_df.columns and col in engineered_df.columns:
            orig_vals = original_df[col].values
            eng_vals = engineered_df[col].values
            for i in range(len(orig_vals)):
                if pd.isna(orig_vals[i]) and pd.isna(eng_vals[i]):
                    continue
                if pd.isna(orig_vals[i]) != pd.isna(eng_vals[i]):
                    ir_mismatches += 1
                    continue
                if not np.isclose(float(orig_vals[i]), float(eng_vals[i]), rtol=1e-9):
                    ir_mismatches += 1

    ir_match = ir_mismatches == 0
    print(f"   {'✅' if ir_match else '❌'} IR base feature values integrity: "
          f"{'All match' if ir_match else f'{ir_mismatches} mismatches found'}")
    checks.append({"check": "ir_base_integrity", "passed": ir_match,
                    "mismatches": ir_mismatches})

    # Check 4: Engineered feature ratio/difference verification (spot check first row)
    eng_spot_ok = True
    if len(engineered_df) > 0:
        for new_name, operation, op1, op2 in ENGINEERED_FEATURES:
            if op1 in original_df.columns and op2 in original_df.columns:
                v1 = original_df[op1].iloc[0]
                v2 = original_df[op2].iloc[0]
                eng_val = engineered_df[new_name].iloc[0]

                if operation == "ratio":
                    expected_val = v1 / v2 if v2 != 0 else np.nan
                elif operation == "difference":
                    expected_val = v1 - v2
                else:
                    expected_val = v1

                if pd.isna(expected_val) and pd.isna(eng_val):
                    continue
                if pd.isna(expected_val) != pd.isna(eng_val):
                    eng_spot_ok = False
                    continue
                if not np.isclose(float(expected_val), float(eng_val), rtol=1e-9):
                    eng_spot_ok = False

    print(f"   {'✅' if eng_spot_ok else '❌'} Engineered feature spot check (row 0): "
          f"{'Passed' if eng_spot_ok else 'FAILED'}")
    checks.append({"check": "engineered_spot_check", "passed": eng_spot_ok})

    # Check 5: Target values preserved
    target_match = True
    if TARGET_COLUMN in original_df.columns and TARGET_COLUMN in engineered_df.columns:
        for i in range(len(original_df)):
            ov = original_df[TARGET_COLUMN].iloc[i]
            ev = engineered_df[TARGET_COLUMN].iloc[i]
            if pd.isna(ov) and pd.isna(ev):
                continue
            if pd.isna(ov) != pd.isna(ev):
                target_match = False
                break
            if not np.isclose(float(ov), float(ev), rtol=1e-9):
                target_match = False
                break

    print(f"   {'✅' if target_match else '❌'} Target glucose values preserved: "
          f"{'All match' if target_match else 'MISMATCH DETECTED'}")
    checks.append({"check": "target_integrity", "passed": target_match})

    # Check 6: NaN summary
    total_nans = engineered_df.isna().sum().sum()
    print(f"   📊 Total NaN values in engineered dataset: {total_nans}")
    checks.append({"check": "nan_count", "total_nans": int(total_nans)})

    all_passed = all(c.get("passed", True) for c in checks)
    print(f"\n   {'✅ ALL CHECKS PASSED' if all_passed else '⚠️ SOME CHECKS FAILED'}")

    return {"all_passed": all_passed, "checks": checks}


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


def build_full_json_log(
    input_file_path,
    output_csv_path,
    output_json_path,
    output_folder_path,
    original_df,
    engineered_df,
    engineering_log,
    verification_result,
    timestamp_str,
):
    """
    Build comprehensive JSON log explaining:
    - How 24 features + 1 target were derived from 30+ original features
    - Full mapping of every feature
    - Verification results
    """

    # Build the complete feature mapping explanation
    feature_mapping = []
    feature_index = 0

    # TIER 1: IR Base
    for entry in engineering_log["tier_1_ir_base"]:
        feature_index += 1
        feature_mapping.append({
            "index": feature_index,
            "tier": "TIER_1_IR_BASE",
            "output_column": entry["output_column"],
            "source_columns": [entry["source_column"]],
            "operation": entry["operation"],
            "formula": f"direct_copy({entry['source_column']})",
            "description": entry["description"],
            "why_included": (
                "IR (infrared ~940nm) is the primary PPG channel. "
                "Deeper tissue penetration, higher SNR, less skin tone sensitivity, "
                "less motion artifact. Standard reference channel in clinical pulse oximetry."
            ),
            "nan_count": entry["nan_count"],
        })

    # TIER 2: Engineered
    for entry in engineering_log["tier_2_engineered"]:
        feature_index += 1

        # Specific physiological reasoning for each engineered feature
        why_map = {
            "Ratio_systolic_amplitude": (
                "Ratio of pulse amplitude between RED and IR wavelengths. "
                "Core of SpO2 measurement physics. Glucose affects hemoglobin glycation (HbA1c), "
                "which has different absorption at RED vs IR. This ratio directly captures that difference. "
                "Cannot be derived from IR alone."
            ),
            "Ratio_TEO_Mean": (
                "Ratio of Teager Energy between RED and IR channels. "
                "Captures how differently blood absorbs/scatters light at each wavelength. "
                "Glucose changes blood optical density. This ratio captures that optical density difference. "
                "Cannot be derived from IR alone."
            ),
            "Diff_2nd_Derivative_Mean": (
                "Difference of 2nd derivative (acceleration plethysmogram) between RED and IR. "
                "2nd derivative showed the LARGEST difference between channels in the dataset. "
                "Relates to arterial stiffness which changes with glucose levels. "
                "The difference captures depth-dependent stiffness variation between wavelengths."
            ),
            "Diff_Spectral_Entropy": (
                "Difference in spectral entropy (frequency complexity) between RED and IR. "
                "Glucose affects blood viscosity which changes harmonic structure differently at each wavelength. "
                "Captures inter-wavelength frequency complexity variation."
            ),
            "Diff_Dicrotic_notch": (
                "Difference in dicrotic notch position between RED and IR. "
                "Dicrotic notch is sensitive to arterial stiffness, blood viscosity, and vascular compliance. "
                "These are all glucose-affected parameters. "
                "The notch appears differently in RED vs IR due to different tissue penetration depths."
            ),
        }

        feature_mapping.append({
            "index": feature_index,
            "tier": "TIER_2_ENGINEERED",
            "output_column": entry["output_column"],
            "source_columns": [entry["operand_1"], entry["operand_2"]],
            "operation": entry["operation"],
            "formula": entry["formula"],
            "description": entry["description"],
            "why_included": why_map.get(entry["output_column"], "Inter-wavelength engineered feature."),
            "nan_count": entry["nan_count"],
        })

    # TIER 3: Keep-as-is
    for entry in engineering_log["tier_3_keep_as_is"]:
        feature_index += 1
        feature_mapping.append({
            "index": feature_index,
            "tier": "TIER_3_KEEP_AS_IS",
            "output_column": entry["output_column"],
            "source_columns": [entry["source_column"]],
            "operation": entry["operation"],
            "formula": f"direct_copy({entry['source_column']})",
            "description": entry["description"],
            "why_included": (
                "Ensemble ratio is already a combined metric derived from both channels. "
                "It captures the overall relationship between RED and IR signal characteristics. "
                "No further transformation needed."
            ),
            "nan_count": entry["nan_count"],
        })

    # TARGET
    target_entry = engineering_log["target"]
    target_info = {
        "output_column": target_entry["output_column"],
        "source_column": target_entry["source_column"],
        "operation": target_entry["operation"],
        "description": target_entry["description"],
        "nan_count": target_entry["nan_count"],
    }

    # Build the dropped features explanation
    all_original_cols = set(original_df.columns.tolist())
    all_kept_source_cols = set()
    for col in IR_BASE_FEATURES:
        all_kept_source_cols.add(col)
    for _, _, op1, op2 in ENGINEERED_FEATURES:
        all_kept_source_cols.add(op1)
        all_kept_source_cols.add(op2)
    for col in KEEP_AS_IS_FEATURES:
        all_kept_source_cols.add(col)
    all_kept_source_cols.add(TARGET_COLUMN)

    dropped_cols = sorted(list(all_original_cols - all_kept_source_cols))

    dropped_explanation = []
    for col in dropped_cols:
        reason = "Unknown"
        if col.startswith("Red_"):
            feature_name = col.replace("Red_", "")
            ir_counterpart = f"IR_{feature_name}"
            if ir_counterpart in IR_BASE_FEATURES:
                reason = (
                    f"Redundant with IR counterpart '{ir_counterpart}'. "
                    f"RED and IR measure the same physical quantity for this feature. "
                    f"IR is kept as the primary channel (deeper penetration, higher SNR, "
                    f"less motion artifact, standard clinical reference)."
                )
            else:
                reason = (
                    f"RED channel feature. Partially captured through engineered features "
                    f"(ratios or differences). Direct inclusion would add redundancy."
                )
        dropped_explanation.append({
            "dropped_column": col,
            "reason_for_dropping": reason,
        })

    # Assemble full JSON structure
    full_log = {
        "pipeline_info": {
            "pipeline_name": "Feature Engineering: IR Base + Selective RED/Ratio Additions",
            "pipeline_step": "STEP 7",
            "execution_timestamp": timestamp_str,
            "execution_date_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "file_paths": {
            "input_master_csv": str(input_file_path),
            "output_engineered_csv": str(output_csv_path),
            "output_json_log": str(output_json_path),
            "output_folder": str(output_folder_path),
        },
        "dataset_transformation_summary": {
            "original_total_columns": int(original_df.shape[1]),
            "original_total_rows": int(original_df.shape[0]),
            "original_feature_columns": int(original_df.shape[1] - 1),
            "engineered_total_columns": int(engineered_df.shape[1]),
            "engineered_total_rows": int(engineered_df.shape[0]),
            "engineered_feature_columns": int(engineered_df.shape[1] - 1),
            "target_column": TARGET_COLUMN,
            "reduction_summary": (
                f"Reduced from {original_df.shape[1] - 1} features to "
                f"{engineered_df.shape[1] - 1} features. "
                f"Removed {(original_df.shape[1] - 1) - (engineered_df.shape[1] - 1)} "
                f"redundant features."
            ),
        },
        "feature_composition": {
            "tier_1_ir_base_count": len(IR_BASE_FEATURES),
            "tier_1_description": (
                "All 18 IR channel features kept as primary physiological measurements. "
                "IR (~940nm) has deeper tissue penetration, higher SNR, less skin tone sensitivity, "
                "and is the standard reference channel in clinical pulse oximetry."
            ),
            "tier_2_engineered_count": len(ENGINEERED_FEATURES),
            "tier_2_description": (
                "5 engineered features derived from RED-IR relationships. "
                "These capture inter-wavelength physiological information that IR alone cannot provide. "
                "Includes ratios (amplitude, energy) and differences (derivatives, entropy, dicrotic notch)."
            ),
            "tier_3_keep_as_is_count": len(KEEP_AS_IS_FEATURES),
            "tier_3_description": (
                "1 feature (Ensemble ratio) kept unchanged as it is already a combined metric."
            ),
            "total_features": len(IR_BASE_FEATURES) + len(ENGINEERED_FEATURES) + len(KEEP_AS_IS_FEATURES),
            "target_count": 1,
            "total_columns": len(IR_BASE_FEATURES) + len(ENGINEERED_FEATURES) + len(KEEP_AS_IS_FEATURES) + 1,
        },
        "complete_feature_mapping": feature_mapping,
        "target_variable": target_info,
        "dropped_features": {
            "total_dropped": len(dropped_cols),
            "dropped_columns": dropped_explanation,
            "general_dropping_rationale": (
                "RED channel features that are near-duplicates of their IR counterparts "
                "were removed to eliminate multicollinearity and reduce overfitting risk. "
                "Where RED carries unique inter-wavelength information (amplitude ratio, "
                "energy ratio, derivative/entropy/notch differences), that information is "
                "preserved through engineered ratio or difference features."
            ),
        },
        "verification_results": verification_result,
        "physiological_rationale": {
            "why_ir_is_primary": (
                "IR light (~940nm) penetrates deeper into tissue, seeing larger and more stable "
                "blood vessels. It has higher signal-to-noise ratio, is less affected by skin "
                "melanin content, and is less sensitive to motion artifacts. Clinical pulse "
                "oximetry devices use IR as the primary measurement channel."
            ),
            "why_selective_red_additions": (
                "RED light (~660nm) has different absorption characteristics than IR due to "
                "fundamental hemoglobin physics. Oxygenated hemoglobin (HbO2) and deoxygenated "
                "hemoglobin (HHb) absorb RED and IR light differently. The RELATIONSHIP between "
                "RED and IR (ratio or difference) captures physiologically meaningful information "
                "about blood composition, oxygen saturation, and potentially glucose-related "
                "changes in blood optical properties. Keeping both channels separately would be "
                "redundant; capturing their relationship is informative."
            ),
            "glucose_relevance": (
                "Glucose affects blood properties in ways that influence PPG signals: "
                "1) Blood viscosity changes → affects pulse wave morphology, "
                "2) Hemoglobin glycation (HbA1c) → changes optical absorption at different wavelengths, "
                "3) Blood refractive index changes → affects light scattering, "
                "4) Arterial stiffness changes → affects pulse wave derivatives and dicrotic notch. "
                "The selected feature set captures these effects through IR morphological features "
                "and RED-IR inter-wavelength relationships."
            ),
        },
    }

    return full_log


def save_outputs(engineered_df, json_log, output_folder, timestamp_str):
    """
    Save the engineered dataset CSV and JSON log file.
    Folder name: Data_set_with_selected_feature_&_target_YYYYMMDD_HHMMSS
    CSV name:    Data_set_with_selected_feature_&_target_YYYYMMDD_HHMMSS.csv
    JSON name:   Data_set_with_selected_feature_&_target_YYYYMMDD_HHMMSS.json
    """
    # Create output folder
    folder_name = f"Data_set_with_selected_feature_&_target_{timestamp_str}"
    output_dir = output_folder / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # File names
    csv_name = f"Data_set_with_selected_feature_&_target_{timestamp_str}.csv"
    json_name = f"Data_set_with_selected_feature_&_target_{timestamp_str}.json"

    csv_path = output_dir / csv_name
    json_path = output_dir / json_name

    # Check for existing files
    replaced_files = []

    csv_pre = check_existing_file(csv_path)
    if csv_pre["exists"]:
        replaced_files.append({"label": "Engineered Dataset CSV", "path": str(csv_path),
                                "old_size_kb": csv_pre["size_kb"]})

    json_pre = check_existing_file(json_path)
    if json_pre["exists"]:
        replaced_files.append({"label": "Engineering Log JSON", "path": str(json_path),
                                "old_size_kb": json_pre["size_kb"]})

    # Save CSV
    engineered_df.to_csv(csv_path, index=False)
    csv_post = check_existing_file(csv_path)
    print(f"\n💾 Saved engineered dataset: {csv_name}")
    print(f"   📊 Size: {csv_post['size_kb']:.2f} KB")

    # Save JSON
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_log, f, indent=4, default=str)
    json_post = check_existing_file(json_path)
    print(f"💾 Saved engineering log: {json_name}")
    print(f"   📊 Size: {json_post['size_kb']:.2f} KB")

    # Report replaced files
    if replaced_files:
        print(f"\n♻️ REPLACED {len(replaced_files)} existing file(s):")
        for rf in replaced_files:
            new_size = check_existing_file(rf["path"])["size_kb"]
            old_str = f"{rf['old_size_kb']:.2f} KB" if rf["old_size_kb"] else "N/A"
            new_str = f"{new_size:.2f} KB" if new_size else "N/A"
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
    print("🔧 STEP 7: FEATURE ENGINEERING PIPELINE")
    print("   IR Base (18) + Selective RED/Ratio (5) + Ensemble (1) + Target (1)")
    print("   Total: 24 Features + 1 Target = 25 Columns")
    print("=" * 70)

    # Validate paths
    if not INPUT_MASTER_ROOT.exists():
        raise SystemExit(f"❌ Input folder does not exist: {INPUT_MASTER_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Generate timestamp for this run
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Step 1: Select input file via popup ──
    print(f"\n📂 Opening file selector at: {INPUT_MASTER_ROOT}")
    input_file = popup_file_selector(INPUT_MASTER_ROOT)
    print(f"📄 Selected file: {input_file.name}")
    print(f"   Full path: {input_file}")

    # ── Step 2: Load master CSV ──
    print(f"\n{'─' * 60}")
    print(f"📥 LOADING MASTER DATASET")
    print(f"{'─' * 60}")
    original_df = load_master_csv(input_file)

    # Display original column inventory
    print(f"\n📋 Original columns ({len(original_df.columns)}):")
    for i, col in enumerate(original_df.columns, 1):
        print(f"   {i:2d}. {col}")

    # ── Step 3: Validate required columns ──
    print(f"\n{'─' * 60}")
    print(f"🔍 VALIDATING REQUIRED SOURCE COLUMNS")
    print(f"{'─' * 60}")
    validation = validate_required_columns(original_df)

    if not validation["all_found"]:
        print(f"\n❌ Cannot proceed: {validation['missing_count']} required columns missing.")
        print(f"   Missing columns:")
        for col in validation["missing_columns"]:
            print(f"      ❌ {col}")
        raise SystemExit("❌ Feature engineering aborted due to missing columns.")

    # ── Step 4: Build engineered dataset ──
    engineered_df, engineering_log = build_engineered_dataset(original_df)

    # ── Step 5: Verify engineered dataset ──
    verification = verify_engineered_dataset(engineered_df, original_df, engineering_log)

    # Display final column inventory
    print(f"\n📋 Engineered dataset columns ({len(engineered_df.columns)}):")
    feature_count = 0
    for i, col in enumerate(engineered_df.columns, 1):
        if col == TARGET_COLUMN:
            print(f"   {i:2d}. {col}  ← TARGET")
        else:
            feature_count += 1
            print(f"   {i:2d}. {col}")
    print(f"\n   Total features: {feature_count}")
    print(f"   Total columns:  {len(engineered_df.columns)} (features + target)")

    # Display sample data
    print(f"\n📊 Sample values (first row):")
    print(f"{'─' * 60}")
    for col in engineered_df.columns:
        val = engineered_df[col].iloc[0]
        if col == TARGET_COLUMN:
            print(f"   🩸 {col}: {val}")
        else:
            print(f"      {col}: {val}")

    # ── Step 6: Build JSON log ──
    print(f"\n{'─' * 60}")
    print(f"📝 BUILDING COMPREHENSIVE JSON LOG")
    print(f"{'─' * 60}")

    # Prepare output paths for JSON log (before saving)
    folder_name = f"Data_set_with_selected_feature_&_target_{timestamp_str}"
    csv_name = f"Data_set_with_selected_feature_&_target_{timestamp_str}.csv"
    json_name = f"Data_set_with_selected_feature_&_target_{timestamp_str}.json"
    output_dir = OUTPUT_ROOT / folder_name

    json_log = build_full_json_log(
        input_file_path=input_file,
        output_csv_path=output_dir / csv_name,
        output_json_path=output_dir / json_name,
        output_folder_path=output_dir,
        original_df=original_df,
        engineered_df=engineered_df,
        engineering_log=engineering_log,
        verification_result=verification,
        timestamp_str=timestamp_str,
    )
    print(f"   ✅ JSON log structure built with {len(json_log)} top-level sections.")

    # ── Step 7: Save outputs ──
    print(f"\n{'─' * 60}")
    print(f"💾 SAVING OUTPUTS")
    print(f"{'─' * 60}")

    csv_path, json_path, output_dir = save_outputs(
        engineered_df, json_log, OUTPUT_ROOT, timestamp_str
    )

    # ── Final Summary ──
    print(f"\n{'=' * 70}")
    print(f"📌 FEATURE ENGINEERING PIPELINE — FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"")
    print(f"   📥 Input:  {input_file.name}")
    print(f"      Shape:  {original_df.shape[0]} rows × {original_df.shape[1]} columns")
    print(f"      Features: {original_df.shape[1] - 1}")
    print(f"")
    print(f"   📤 Output: {csv_path.name}")
    print(f"      Shape:  {engineered_df.shape[0]} rows × {engineered_df.shape[1]} columns")
    print(f"      Features: {engineered_df.shape[1] - 1}")
    print(f"      Target:   {TARGET_COLUMN}")
    print(f"")
    print(f"   📊 Transformation:")
    print(f"      TIER 1 — IR Base features:        {len(IR_BASE_FEATURES)} (direct copy)")
    print(f"      TIER 2 — Engineered RED/Ratio:     {len(ENGINEERED_FEATURES)} (ratio + difference)")
    print(f"      TIER 3 — Keep-as-is (Ensemble):    {len(KEEP_AS_IS_FEATURES)} (direct copy)")
    print(f"      TARGET — Glucose level:            1")
    print(f"      ─────────────────────────────────────")
    print(f"      TOTAL:                             {engineered_df.shape[1]} columns")
    print(f"")
    dropped_count = (original_df.shape[1] - 1) - (engineered_df.shape[1] - 1)
    print(f"   🗑️ Dropped: {dropped_count} redundant RED features")
    print(f"   ✅ Verification: {'ALL PASSED' if verification['all_passed'] else 'SOME CHECKS FAILED'}")
    print(f"")
    print(f"   📁 Output folder: {output_dir}")
    print(f"   📄 Dataset CSV:   {csv_path.name}")
    print(f"   📄 JSON Log:      {json_path.name}")
    print(f"")
    print(f"✅ Feature engineering pipeline completed successfully!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()