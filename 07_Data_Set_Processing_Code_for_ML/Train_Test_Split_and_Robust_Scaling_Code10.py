# ==========================================
# STEP 8 (Sub-task 3 & 4): TRAIN/TEST SPLIT + ROBUST SCALING
# Input: Cleaned 24-feature dataset from Step 8 (Sub-task 1 & 2)
# Output: Scaled train/test splits ready for XGBoost
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


# --------------------------------------------------
# USER SETTINGS (PASTE YOUR PATHS HERE)
# --------------------------------------------------
INPUT_ROOT  = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\09_Cleaned_dataset_without_(NaN_&_outliers)")
OUTPUT_ROOT = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\10_Robust_Scaled_and_Train_Test_Splitted_Data_Set")

# Train/Test split EXACT COUNTS (change as needed)
# These two numbers MUST add up exactly to your total number of samples
AMOUNT_OF_TRAIN_SAMPLES = 2   # ← PASTE EXACT TRAIN COUNT HERE
AMOUNT_OF_TEST_SAMPLES  = 1    # ← PASTE EXACT TEST COUNT HERE

RANDOM_STATE = 42              # Fixed seed for reproducibility


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
TARGET_COLUMN = "Glucose level (mg/dl)"


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def popup_folder_selector(initial_dir):
    """
    Opens a folder dialog for user to select the Step 8 (Sub-task 1&2) output folder.
    That folder should contain both the CSV and JSON files.
    Returns: Path to selected folder.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    messagebox.showinfo(
        title="Train/Test Split & Scaling Pipeline — Folder Selection",
        message=(
            "Select the Step 8 (Sub-task 1&2) OUTPUT FOLDER to process.\n\n"
            "This folder should contain:\n"
            "  • removed_outliers_&_NaN_24_feature_data_set_XXXXXX.csv\n"
            "  • removed_outliers_&_NaN_24_feature_data_set_XXXXXX.json\n\n"
            "Select the FOLDER itself (not the files inside).\n\n"
            "Click OK to open the folder browser."
        ),
    )

    selected_folder = filedialog.askdirectory(
        initialdir=str(initial_dir),
        title="Select Step 8 (Sub-task 1&2) Output FOLDER (contains CSV + JSON)",
    )

    root.destroy()

    if not selected_folder:
        raise SystemExit("❌ User cancelled: No folder selected. Execution terminated.")

    return Path(selected_folder)


def find_csv_and_json_in_folder(folder_path):
    """
    Automatically find the CSV and JSON files inside the selected folder.
    Returns: (csv_path, json_path)
    """
    folder = Path(folder_path)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found or not a directory: {folder}")

    csv_files = list(folder.glob("*.csv"))
    json_files = list(folder.glob("*.json"))

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No CSV file found inside folder: {folder}")
    if len(json_files) == 0:
        raise FileNotFoundError(f"No JSON file found inside folder: {folder}")

    if len(csv_files) > 1:
        print(f"   ⚠️ Multiple CSV files found. Using first one: {csv_files[0].name}")
    if len(json_files) > 1:
        print(f"   ⚠️ Multiple JSON files found. Using first one: {json_files[0].name}")

    return csv_files[0], json_files[0]


def load_csv(file_path):
    """Load a CSV file and return DataFrame."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

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
# SUB-TASK 3: SEPARATE X AND y + TRAIN/TEST SPLIT
# --------------------------------------------------
def separate_x_y(df):
    """
    Separate features (X) and target (y).
    X = all columns except target
    y = target column only

    Returns: (X, y, feature_columns)
    """
    print(f"\n{'─' * 60}")
    print(f"🔧 SUB-TASK 3a: SEPARATING X (Features) AND y (Target)")
    print(f"{'─' * 60}")

    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    feature_columns = [col for col in df.columns if col != TARGET_COLUMN]
    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    print(f"   ✅ X (Features): {X.shape[0]} rows × {X.shape[1]} columns")
    print(f"   ✅ y (Target):   {y.shape[0]} values")
    print(f"   📊 Feature columns ({len(feature_columns)}):")
    for i, col in enumerate(feature_columns, 1):
        print(f"      {i:2d}. {col}")
    print(f"   🩸 Target: {TARGET_COLUMN}")

    # Verify no NaN
    x_nans = int(X.isna().sum().sum())
    y_nans = int(y.isna().sum())
    print(f"\n   🔍 NaN check: X has {x_nans} NaN(s), y has {y_nans} NaN(s)")

    if x_nans > 0 or y_nans > 0:
        print(f"   ⚠️ WARNING: NaN values detected! This should have been handled in Sub-task 1.")

    return X, y, feature_columns


def perform_train_test_split(X, y):
    """
    Split X and y into train and test sets using EXACT COUNTS.
    Uses AMOUNT_OF_TRAIN_SAMPLES and AMOUNT_OF_TEST_SAMPLES from configuration.

    Returns: (X_train, X_test, y_train, y_test, split_info)
    """
    print(f"\n{'─' * 60}")
    print(f"🔧 SUB-TASK 3b: TRAIN/TEST SPLIT (Exact Counts)")
    print(f"{'─' * 60}")

    total_samples = len(X)

    # ── Validate exact counts ──
    specified_total = AMOUNT_OF_TRAIN_SAMPLES + AMOUNT_OF_TEST_SAMPLES
    if specified_total != total_samples:
        raise ValueError(
            f"❌ Configuration error:\n"
            f"      You specified Train = {AMOUNT_OF_TRAIN_SAMPLES}, Test = {AMOUNT_OF_TEST_SAMPLES}\n"
            f"      Sum = {specified_total}, but your dataset has {total_samples} samples.\n"
            f"      Please adjust AMOUNT_OF_TRAIN_SAMPLES and AMOUNT_OF_TEST_SAMPLES\n"
            f"      so they add up exactly to {total_samples}."
        )

    print(f"   📊 Configuration:")
    print(f"      Total samples:   {total_samples}")
    print(f"      Train samples:   {AMOUNT_OF_TRAIN_SAMPLES}")
    print(f"      Test samples:    {AMOUNT_OF_TEST_SAMPLES}")
    print(f"      Random state:    {RANDOM_STATE}")

    # ── Perform split using exact test count ──
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y,
        test_size=AMOUNT_OF_TEST_SAMPLES,
        random_state=RANDOM_STATE,
    )

    # Capture original indices before resetting
    train_indices = X_train_raw.index.tolist()
    test_indices = X_test_raw.index.tolist()

    # Reset indices for clean output
    X_train = X_train_raw.reset_index(drop=True)
    X_test = X_test_raw.reset_index(drop=True)
    y_train = y_train_raw.reset_index(drop=True)
    y_test = y_test_raw.reset_index(drop=True)

    # Verify exact counts
    assert len(X_train) == AMOUNT_OF_TRAIN_SAMPLES, "Train count mismatch after split"
    assert len(X_test) == AMOUNT_OF_TEST_SAMPLES, "Test count mismatch after split"

    print(f"\n   ✅ Split completed:")
    print(f"      X_train: {X_train.shape[0]} rows × {X_train.shape[1]} columns")
    print(f"      X_test:  {X_test.shape[0]} rows × {X_test.shape[1]} columns")
    print(f"      y_train: {y_train.shape[0]} values")
    print(f"      y_test:  {y_test.shape[0]} values")

    # Verify total row count
    total_after = X_train.shape[0] + X_test.shape[0]
    rows_match = total_after == total_samples
    print(f"\n   🔍 Row count verification: {X_train.shape[0]} + {X_test.shape[0]} = {total_after} "
          f"(original: {total_samples}) {'✅' if rows_match else '❌'}")

    # Display glucose distribution in train and test
    print(f"\n   📊 Glucose distribution:")
    print(f"      Train — min: {y_train.min():.1f}, max: {y_train.max():.1f}, "
          f"mean: {y_train.mean():.1f}, std: {y_train.std():.1f}")
    print(f"      Test  — min: {y_test.min():.1f}, max: {y_test.max():.1f}, "
          f"mean: {y_test.mean():.1f}, std: {y_test.std():.1f}")

    # Build split info for logging
    split_info = {
        "split_method": "exact_count",
        "total_samples": int(total_samples),
        "train_samples_specified": AMOUNT_OF_TRAIN_SAMPLES,
        "test_samples_specified": AMOUNT_OF_TEST_SAMPLES,
        "train_samples_actual": int(X_train.shape[0]),
        "test_samples_actual": int(X_test.shape[0]),
        "random_state": RANDOM_STATE,
        "random_state_note": "Fixed seed ensures the same rows are selected every time the code runs.",
        "train_original_indices": train_indices,
        "test_original_indices": test_indices,
        "glucose_distribution": {
            "train": {
                "min": float(y_train.min()),
                "max": float(y_train.max()),
                "mean": float(y_train.mean()),
                "std": float(y_train.std()),
                "median": float(y_train.median()),
                "count": int(len(y_train)),
            },
            "test": {
                "min": float(y_test.min()),
                "max": float(y_test.max()),
                "mean": float(y_test.mean()),
                "std": float(y_test.std()),
                "median": float(y_test.median()),
                "count": int(len(y_test)),
            },
        },
    }

    return X_train, X_test, y_train, y_test, split_info


# --------------------------------------------------
# SUB-TASK 4: ROBUST SCALING
# --------------------------------------------------
def perform_robust_scaling(X_train, X_test, feature_columns):
    """
    Fit RobustScaler on X_train ONLY.
    Transform both X_train and X_test using the SAME fitted scaler.

    Returns: (X_train_scaled, X_test_scaled, scaler_params)
    """
    print(f"\n{'─' * 60}")
    print(f"📏 SUB-TASK 4: ROBUST SCALING")
    print(f"{'─' * 60}")

    scaler = RobustScaler()

    # ── Fit on X_train ONLY ──
    print(f"\n   🔧 Fitting RobustScaler on X_train ({X_train.shape[0]} samples)...")
    scaler.fit(X_train)
    print(f"   ✅ Scaler fitted successfully.")

    # ── Display scaler parameters ──
    print(f"\n   📊 Scaler Parameters (learned from X_train only):")
    print(f"   {'Column':<35} {'Center (Median)':>18} {'Scale (IQR)':>18}")
    print(f"   {'─' * 35} {'─' * 18} {'─' * 18}")

    scaler_details = []
    for i, col in enumerate(feature_columns):
        center = float(scaler.center_[i])
        scale = float(scaler.scale_[i])
        print(f"   {col:<35} {center:>18.6f} {scale:>18.6f}")
        scaler_details.append({
            "feature": col,
            "center_median": center,
            "scale_iqr": scale,
        })

    # ── Transform X_train ──
    print(f"\n   🔄 Transforming X_train...")
    X_train_scaled_array = scaler.transform(X_train)
    X_train_scaled = pd.DataFrame(X_train_scaled_array, columns=feature_columns)
    print(f"   ✅ X_train transformed: {X_train_scaled.shape[0]} rows × {X_train_scaled.shape[1]} columns")

    # ── Transform X_test using SAME scaler ──
    print(f"\n   🔄 Transforming X_test (using scaler fitted on X_train)...")
    X_test_scaled_array = scaler.transform(X_test)
    X_test_scaled = pd.DataFrame(X_test_scaled_array, columns=feature_columns)
    print(f"   ✅ X_test transformed: {X_test_scaled.shape[0]} rows × {X_test_scaled.shape[1]} columns")

    # ── Display before/after comparison for X_train ──
    print(f"\n   📊 X_train — Before vs After Scaling:")
    print(f"   {'Column':<35} {'Before Min':>12} {'After Min':>12} {'Before Max':>12} {'After Max':>12}")
    print(f"   {'─' * 35} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")

    train_comparison = []
    for col in feature_columns:
        b_min = float(X_train[col].min())
        b_max = float(X_train[col].max())
        a_min = float(X_train_scaled[col].min())
        a_max = float(X_train_scaled[col].max())
        print(f"   {col:<35} {b_min:>12.6f} {a_min:>12.6f} {b_max:>12.6f} {a_max:>12.6f}")
        train_comparison.append({
            "feature": col,
            "before_min": b_min, "before_max": b_max,
            "after_min": a_min, "after_max": a_max,
        })

    # ── Display before/after comparison for X_test ──
    print(f"\n   📊 X_test — Before vs After Scaling:")
    print(f"   {'Column':<35} {'Before Min':>12} {'After Min':>12} {'Before Max':>12} {'After Max':>12}")
    print(f"   {'─' * 35} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 12}")

    test_comparison = []
    for col in feature_columns:
        b_min = float(X_test[col].min())
        b_max = float(X_test[col].max())
        a_min = float(X_test_scaled[col].min())
        a_max = float(X_test_scaled[col].max())
        print(f"   {col:<35} {b_min:>12.6f} {a_min:>12.6f} {b_max:>12.6f} {a_max:>12.6f}")
        test_comparison.append({
            "feature": col,
            "before_min": b_min, "before_max": b_max,
            "after_min": a_min, "after_max": a_max,
        })

    # ── Build scaler parameters log ──
    scaler_params = {
        "scaler_type": "RobustScaler",
        "fitted_on": "X_train only",
        "data_leakage_prevention": "Scaler was fitted exclusively on training data. Test data was only transformed, never used for fitting.",
        "formula": "X_scaled = (X - median) / IQR",
        "feature_scaler_parameters": scaler_details,
        "train_before_after_comparison": train_comparison,
        "test_before_after_comparison": test_comparison,
    }

    return X_train_scaled, X_test_scaled, scaler_params


# --------------------------------------------------
# VERIFICATION
# --------------------------------------------------
def verify_outputs(X_train_scaled, X_test_scaled, y_train, y_test,
                   original_df, feature_columns):
    """
    Verify all outputs for integrity.
    Returns: verification result dict.
    """
    print(f"\n{'─' * 60}")
    print(f"🔍 FINAL VERIFICATION")
    print(f"{'─' * 60}")

    checks = []

    # Check 1: Total row count preserved
    total_split = X_train_scaled.shape[0] + X_test_scaled.shape[0]
    original_rows = original_df.shape[0]
    rows_match = total_split == original_rows
    print(f"   {'✅' if rows_match else '❌'} Total rows: {total_split} "
          f"(train: {X_train_scaled.shape[0]} + test: {X_test_scaled.shape[0]}) "
          f"= original: {original_rows}")
    checks.append({"check": "total_row_count", "passed": rows_match,
                    "train": X_train_scaled.shape[0], "test": X_test_scaled.shape[0],
                    "total": total_split, "original": original_rows})

    # Check 2: Feature count preserved
    train_cols_match = X_train_scaled.shape[1] == len(feature_columns)
    test_cols_match = X_test_scaled.shape[1] == len(feature_columns)
    cols_ok = train_cols_match and test_cols_match
    print(f"   {'✅' if cols_ok else '❌'} Feature count: train={X_train_scaled.shape[1]}, "
          f"test={X_test_scaled.shape[1]} (expected: {len(feature_columns)})")
    checks.append({"check": "feature_count", "passed": cols_ok})

    # Check 3: Column names preserved
    train_names = list(X_train_scaled.columns) == feature_columns
    test_names = list(X_test_scaled.columns) == feature_columns
    names_ok = train_names and test_names
    print(f"   {'✅' if names_ok else '❌'} Column names preserved: {names_ok}")
    checks.append({"check": "column_names", "passed": names_ok})

    # Check 4: No NaN in scaled outputs
    train_nans = int(X_train_scaled.isna().sum().sum())
    test_nans = int(X_test_scaled.isna().sum().sum())
    y_train_nans = int(y_train.isna().sum())
    y_test_nans = int(y_test.isna().sum())
    no_nans = (train_nans + test_nans + y_train_nans + y_test_nans) == 0
    print(f"   {'✅' if no_nans else '❌'} No NaN: X_train={train_nans}, X_test={test_nans}, "
          f"y_train={y_train_nans}, y_test={y_test_nans}")
    checks.append({"check": "no_nan", "passed": no_nans})

    # Check 5: y values not scaled (raw glucose preserved)
    y_train_match = True
    y_test_match = True
    if TARGET_COLUMN in original_df.columns:
        all_glucose = original_df[TARGET_COLUMN].values
        for val in y_train.values:
            if val not in all_glucose:
                y_train_match = False
                break
        for val in y_test.values:
            if val not in all_glucose:
                y_test_match = False
                break

    y_ok = y_train_match and y_test_match
    print(f"   {'✅' if y_ok else '❌'} Target values unmodified (raw glucose): {y_ok}")
    checks.append({"check": "target_unmodified", "passed": y_ok})

    # Check 6: All values are numeric and finite
    all_finite = True
    for col in X_train_scaled.columns:
        if not np.all(np.isfinite(X_train_scaled[col].values)):
            all_finite = False
            break
    for col in X_test_scaled.columns:
        if not np.all(np.isfinite(X_test_scaled[col].values)):
            all_finite = False
            break

    print(f"   {'✅' if all_finite else '❌'} All scaled values finite: {all_finite}")
    checks.append({"check": "all_finite", "passed": all_finite})

    # Summary statistics
    print(f"\n   📊 Scaled data summary:")
    print(f"      X_train_scaled — mean range: [{X_train_scaled.mean().min():.4f}, {X_train_scaled.mean().max():.4f}]")
    print(f"      X_train_scaled — std range:  [{X_train_scaled.std().min():.4f}, {X_train_scaled.std().max():.4f}]")
    print(f"      X_test_scaled  — mean range: [{X_test_scaled.mean().min():.4f}, {X_test_scaled.mean().max():.4f}]")
    print(f"      X_test_scaled  — std range:  [{X_test_scaled.std().min():.4f}, {X_test_scaled.std().max():.4f}]")

    all_passed = all(c.get("passed", True) for c in checks if "passed" in c)
    print(f"\n   {'✅ ALL CHECKS PASSED' if all_passed else '⚠️ SOME CHECKS FAILED'}")

    return {"all_passed": all_passed, "checks": checks}


# --------------------------------------------------
# JSON LOG BUILDER
# --------------------------------------------------
def build_split_scale_json_log(
    input_csv_path,
    input_json_path,
    input_folder_path,
    output_folder_path,
    output_file_paths,
    original_df,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    feature_columns,
    prev_step_json,
    split_info,
    scaler_params,
    verification_result,
    timestamp_str,
):
    """Build comprehensive JSON log for the split and scaling pipeline."""

    full_log = {
        "pipeline_info": {
            "pipeline_name": "Train/Test Split + RobustScaler Normalization",
            "pipeline_step": "STEP 8 (Sub-task 3 & 4)",
            "execution_timestamp": timestamp_str,
            "execution_date_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "previous_step": "STEP 8 (Sub-task 1 & 2) — NaN Handling + Outlier Clipping",
        },
        "file_paths": {
            "input_folder": str(input_folder_path),
            "input_csv": str(input_csv_path),
            "input_json_from_prev_step": str(input_json_path),
            "output_folder": str(output_folder_path),
            "output_files": output_file_paths,
        },
        "dataset_summary": {
            "input_rows": int(original_df.shape[0]),
            "input_columns": int(original_df.shape[1]),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "target_scaled": False,
            "target_note": "Glucose values are kept as raw measurements. Not scaled.",
        },
        "sub_task_3_train_test_split": {
            "description": (
                f"Dataset was split into exact counts: "
                f"{AMOUNT_OF_TRAIN_SAMPLES} training samples and "
                f"{AMOUNT_OF_TEST_SAMPLES} testing samples using sklearn train_test_split. "
                f"Random state {RANDOM_STATE} ensures reproducibility. "
                f"Split was performed BEFORE scaling to prevent data leakage."
            ),
            "split_details": split_info,
        },
        "sub_task_4_robust_scaling": {
            "description": (
                "RobustScaler was fitted ONLY on X_train data. "
                "Both X_train and X_test were transformed using this same fitted scaler. "
                "RobustScaler uses median (center) and IQR (scale) instead of mean and std, "
                "making it robust to outliers common in PPG signal data. "
                "Formula: X_scaled = (X - median) / IQR. "
                "Target variable (glucose) was NOT scaled."
            ),
            "scaler_parameters": scaler_params,
        },
        "output_shapes": {
            "X_train_scaled": {
                "rows": int(X_train_scaled.shape[0]),
                "columns": int(X_train_scaled.shape[1]),
            },
            "X_test_scaled": {
                "rows": int(X_test_scaled.shape[0]),
                "columns": int(X_test_scaled.shape[1]),
            },
            "y_train": {
                "count": int(len(y_train)),
                "min": float(y_train.min()),
                "max": float(y_train.max()),
                "mean": float(y_train.mean()),
            },
            "y_test": {
                "count": int(len(y_test)),
                "min": float(y_test.min()),
                "max": float(y_test.max()),
                "mean": float(y_test.mean()),
            },
        },
        "verification_results": verification_result,
        "previous_step_reference": {
            "prev_step_json_file": str(input_json_path),
            "prev_step_pipeline_info": prev_step_json.get("pipeline_info", {}),
        },
        "important_notes": {
            "data_leakage_prevention": (
                "The dataset was split into train and test BEFORE any scaling was applied. "
                "The RobustScaler was fitted exclusively on the training set. "
                "The test set was only transformed using the already-fitted scaler. "
                "This prevents information from the test set leaking into the training process."
            ),
            "reproducibility": (
                f"Random state {RANDOM_STATE} was used for the train/test split. "
                f"Using the same random state will produce the exact same split every time."
            ),
            "target_not_scaled": (
                "Glucose level (target variable) was intentionally NOT scaled. "
                "XGBoost regression works directly with raw target values. "
                "Scaling the target would complicate interpretation of predictions."
            ),
            "scaler_saved_for_future_use": (
                "The scaler parameters (center/median and scale/IQR for each feature) "
                "are saved in this JSON file. When making predictions on new data in the future, "
                "apply the SAME scaler transformation using these saved parameters."
            ),
        },
    }

    return full_log


# --------------------------------------------------
# SAVE OUTPUTS
# --------------------------------------------------
def save_all_outputs(
    X_train_scaled, X_test_scaled, y_train, y_test,
    X_train_unscaled, X_test_unscaled,
    json_log, output_root, timestamp_str,
):
    """
    Save all output files in organized sub-folders.

    Main folder: Train_test_split_&_scaled_24_Feature_data_YYYYMMDD_HHMMSS/
        ├── train/
        │   ├── X_train_scaled.csv
        │   ├── X_train_unscaled.csv
        │   └── y_train.csv
        ├── test/
        │   ├── X_test_scaled.csv
        │   ├── X_test_unscaled.csv
        │   └── y_test.csv
        └── json/
            └── Train_test_split_&_scaled_24_Feature_data_YYYYMMDD_HHMMSS.json
    """
    # Create main output folder
    main_folder_name = f"Train_test_split_&_scaled_24_Feature_data_{timestamp_str}"
    main_dir = output_root / main_folder_name
    main_dir.mkdir(parents=True, exist_ok=True)

    # Create sub-folders
    train_dir = main_dir / "train"
    test_dir = main_dir / "test"
    json_dir = main_dir / "json"

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)

    replaced_files = []

    # ── Save train files ──
    train_files = {
        "X_train_scaled.csv": X_train_scaled,
        "X_train_unscaled.csv": X_train_unscaled,
        "y_train.csv": y_train.to_frame(name=TARGET_COLUMN),
    }

    print(f"\n   📁 Saving TRAIN files to: {train_dir.name}/")
    for fname, data in train_files.items():
        fpath = train_dir / fname
        pre_info = check_existing_file(fpath)
        if pre_info["exists"]:
            replaced_files.append({"label": fname, "path": str(fpath),
                                    "old_size_kb": pre_info["size_kb"]})

        data.to_csv(fpath, index=False)
        post_info = check_existing_file(fpath)
        print(f"      💾 {fname} — {post_info['size_kb']:.2f} KB "
              f"({data.shape[0]} rows × {data.shape[1]} cols)")

    # ── Save test files ──
    test_files = {
        "X_test_scaled.csv": X_test_scaled,
        "X_test_unscaled.csv": X_test_unscaled,
        "y_test.csv": y_test.to_frame(name=TARGET_COLUMN),
    }

    print(f"\n   📁 Saving TEST files to: {test_dir.name}/")
    for fname, data in test_files.items():
        fpath = test_dir / fname
        pre_info = check_existing_file(fpath)
        if pre_info["exists"]:
            replaced_files.append({"label": fname, "path": str(fpath),
                                    "old_size_kb": pre_info["size_kb"]})

        data.to_csv(fpath, index=False)
        post_info = check_existing_file(fpath)
        print(f"      💾 {fname} — {post_info['size_kb']:.2f} KB "
              f"({data.shape[0]} rows × {data.shape[1]} cols)")

    # ── Save JSON log ──
    json_name = f"Train_test_split_&_scaled_24_Feature_data_{timestamp_str}.json"
    json_path = json_dir / json_name

    pre_info = check_existing_file(json_path)
    if pre_info["exists"]:
        replaced_files.append({"label": "Processing JSON Log", "path": str(json_path),
                                "old_size_kb": pre_info["size_kb"]})

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_log, f, indent=4, default=str)

    post_info = check_existing_file(json_path)
    print(f"\n   📁 Saving JSON log to: {json_dir.name}/")
    print(f"      💾 {json_name} — {post_info['size_kb']:.2f} KB")

    # Report replaced files
    if replaced_files:
        print(f"\n   ♻️ REPLACED {len(replaced_files)} existing file(s):")
        for rf in replaced_files:
            new_size = check_existing_file(rf["path"])["size_kb"]
            old_str = f"{rf['old_size_kb']:.2f} KB" if rf["old_size_kb"] else "N/A"
            new_str = f"{new_size:.2f} KB" if new_size else "N/A"
            print(f"      {rf['label']}: {old_str} → {new_str}")
    else:
        print(f"\n   🆕 All output files are newly created.")

    # Build output file paths dict for JSON log
    output_file_paths = {
        "main_folder": str(main_dir),
        "train_folder": str(train_dir),
        "test_folder": str(test_dir),
        "json_folder": str(json_dir),
        "X_train_scaled": str(train_dir / "X_train_scaled.csv"),
        "X_train_unscaled": str(train_dir / "X_train_unscaled.csv"),
        "y_train": str(train_dir / "y_train.csv"),
        "X_test_scaled": str(test_dir / "X_test_scaled.csv"),
        "X_test_unscaled": str(test_dir / "X_test_unscaled.csv"),
        "y_test": str(test_dir / "y_test.csv"),
        "json_log": str(json_path),
    }

    return main_dir, output_file_paths


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("📏 STEP 8 (Sub-task 3 & 4): TRAIN/TEST SPLIT + ROBUST SCALING")
    print("   Sub-task 3: Separate X/y + Train/Test Split (Exact Counts)")
    print("   Sub-task 4: RobustScaler (fit on train, transform both)")
    print(f"   Configuration: {AMOUNT_OF_TRAIN_SAMPLES} Train / {AMOUNT_OF_TEST_SAMPLES} Test")
    print(f"   Random State: {RANDOM_STATE}")
    print("=" * 70)

    # Validate paths
    if not INPUT_ROOT.exists():
        raise SystemExit(f"❌ Input folder does not exist: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Generate timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Select input FOLDER ──
    print(f"\n📂 Opening folder selector at: {INPUT_ROOT}")
    input_folder = popup_folder_selector(INPUT_ROOT)
    print(f"📁 Selected folder: {input_folder.name}")
    print(f"   Full path: {input_folder}")

    # ── Auto-detect CSV and JSON ──
    print(f"\n{'─' * 60}")
    print(f"🔍 AUTO-DETECTING FILES INSIDE FOLDER")
    print(f"{'─' * 60}")

    input_csv_path, input_json_path = find_csv_and_json_in_folder(input_folder)
    print(f"   📄 CSV found:  {input_csv_path.name}")
    print(f"   📄 JSON found: {input_json_path.name}")

    # ── Load input files ──
    print(f"\n{'─' * 60}")
    print(f"📥 LOADING INPUT FILES")
    print(f"{'─' * 60}")

    original_df = load_csv(input_csv_path)
    prev_step_json = load_json(input_json_path)

    # Display input column inventory
    print(f"\n📋 Input columns ({len(original_df.columns)}):")
    for i, col in enumerate(original_df.columns, 1):
        if col == TARGET_COLUMN:
            print(f"   {i:2d}. {col}  ← TARGET")
        else:
            print(f"   {i:2d}. {col}")

    # Validate column count
    expected_features = 24
    actual_features = len(original_df.columns) - 1
    if actual_features != expected_features:
        print(f"\n   ⚠️ Expected {expected_features} features, found {actual_features}")
    else:
        print(f"\n   ✅ Feature count verified: {actual_features} features + 1 target")

    # ── SUB-TASK 3a: Separate X and y ──
    X, y, feature_columns = separate_x_y(original_df)

    # ── SUB-TASK 3b: Train/Test Split ──
    X_train, X_test, y_train, y_test, split_info = perform_train_test_split(X, y)

    # ── SUB-TASK 4: Robust Scaling ──
    X_train_scaled, X_test_scaled, scaler_params = perform_robust_scaling(
        X_train, X_test, feature_columns
    )

    # ── Verify all outputs ──
    verification = verify_outputs(
        X_train_scaled, X_test_scaled, y_train, y_test,
        original_df, feature_columns
    )

    # ── Build JSON log (initial — will update file paths after saving) ──
    print(f"\n{'─' * 60}")
    print(f"📝 BUILDING COMPREHENSIVE JSON LOG")
    print(f"{'─' * 60}")

    # Placeholder paths — will be updated after save
    temp_json_log = build_split_scale_json_log(
        input_csv_path=input_csv_path,
        input_json_path=input_json_path,
        input_folder_path=input_folder,
        output_folder_path=OUTPUT_ROOT / f"Train_test_split_&_scaled_24_Feature_data_{timestamp_str}",
        output_file_paths={},
        original_df=original_df,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        prev_step_json=prev_step_json,
        split_info=split_info,
        scaler_params=scaler_params,
        verification_result=verification,
        timestamp_str=timestamp_str,
    )
    print(f"   ✅ JSON log structure built with {len(temp_json_log)} top-level sections.")

    # ── Save all outputs ──
    print(f"\n{'─' * 60}")
    print(f"💾 SAVING ALL OUTPUTS")
    print(f"{'─' * 60}")

    main_output_dir, output_file_paths = save_all_outputs(
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        X_train_unscaled=X_train,
        X_test_unscaled=X_test,
        json_log=temp_json_log,
        output_root=OUTPUT_ROOT,
        timestamp_str=timestamp_str,
    )

    # Update JSON log with actual file paths and re-save
    temp_json_log["file_paths"]["output_files"] = output_file_paths
    json_log_path = Path(output_file_paths["json_log"])
    with open(json_log_path, "w", encoding="utf-8") as f:
        json.dump(temp_json_log, f, indent=4, default=str)

    # ── Final Summary ──
    print(f"\n{'=' * 70}")
    print(f"📌 TRAIN/TEST SPLIT & SCALING PIPELINE — FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"")
    print(f"   📥 Input folder: {input_folder.name}")
    print(f"      CSV:   {input_csv_path.name}")
    print(f"      JSON:  {input_json_path.name}")
    print(f"      Shape: {original_df.shape[0]} rows × {original_df.shape[1]} columns")
    print(f"")
    print(f"   📊 Sub-task 3 — Train/Test Split (Exact Counts):")
    print(f"      Train samples:   {AMOUNT_OF_TRAIN_SAMPLES}")
    print(f"      Test samples:    {AMOUNT_OF_TEST_SAMPLES}")
    print(f"      Random state:    {RANDOM_STATE}")
    print(f"      Train glucose:   {y_train.min():.1f} - {y_train.max():.1f} mg/dL (mean: {y_train.mean():.1f})")
    print(f"      Test glucose:    {y_test.min():.1f} - {y_test.max():.1f} mg/dL (mean: {y_test.mean():.1f})")
    print(f"")
    print(f"   📏 Sub-task 4 — RobustScaler:")
    print(f"      Scaler type:     RobustScaler")
    print(f"      Fitted on:       X_train only (no data leakage)")
    print(f"      Target scaled:   No (raw glucose values preserved)")
    print(f"      Features scaled: {len(feature_columns)}")
    print(f"")
    print(f"   ✅ Verification: {'ALL PASSED' if verification['all_passed'] else 'SOME CHECKS FAILED'}")
    print(f"")
    print(f"   📁 Output structure:")
    print(f"      {main_output_dir.name}/")
    print(f"      ├── train/")
    print(f"      │   ├── X_train_scaled.csv     ({X_train_scaled.shape[0]}×{X_train_scaled.shape[1]})")
    print(f"      │   ├── X_train_unscaled.csv   ({X_train.shape[0]}×{X_train.shape[1]})")
    print(f"      │   └── y_train.csv            ({len(y_train)} values)")
    print(f"      ├── test/")
    print(f"      │   ├── X_test_scaled.csv      ({X_test_scaled.shape[0]}×{X_test_scaled.shape[1]})")
    print(f"      │   ├── X_test_unscaled.csv    ({X_test.shape[0]}×{X_test.shape[1]})")
    print(f"      │   └── y_test.csv             ({len(y_test)} values)")
    print(f"      └── json/")
    print(f"          └── {json_log_path.name}")
    print(f"")
    print(f"   📌 IMPORTANT FOR XGBOOST:")
    print(f"      → Use X_train_scaled.csv + y_train.csv for TRAINING")
    print(f"      → Use X_test_scaled.csv  + y_test.csv  for TESTING")
    print(f"      → Scaler parameters saved in JSON for future predictions")
    print(f"")
    print(f"✅ Train/Test split & scaling pipeline completed successfully!")
    print(f"   → Output is READY for XGBoost model training")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()