# ==========================================
# STEP 9: XGBoost MODEL TRAINING + EVALUATION
# Updated:
# - Popup message reflects new naming (Master dataset 24F split scaled)
# - Output folder naming: XGBoost results & Conclusions YYYY-MM-DD HH-MM-SS
# - Timestamp format: YYYY-MM-DD HH-MM-SS (pipeline-consistent)
# - Validation that selected folder is from Step 8 (Sub-task 3&4) — hard-fail
# - Auto-load JSON from json/ subfolder for traceability
# - Concise pipeline chain (Step 6 → 7 → 8a → 8b → 9) — no bloat
# - Auto-detection of latest split-scaled output folder
# - Variable rename: full_report → xgboost_report_data
# - ALL hyperparameters, model logic, metrics, and calculations preserved exactly
# ==========================================

import os
import json
import time
import traceback
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)


# --------------------------------------------------
# USER SETTINGS (PASTE YOUR PATHS HERE)
# --------------------------------------------------
INPUT_ROOT  = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\10_Robust_Scaled_and_Train_Test_Splitted_Data_Set")
OUTPUT_ROOT = Path(r"C:\Users\DELL\Documents\GitHub\fyp\08_Results_and_Visualizations\XGBoost_Results_&_Conclusions")


# --------------------------------------------------
# XGBOOST HYPERPARAMETERS  (⚠️ DO NOT CHANGE WITHOUT INTENT)
# --------------------------------------------------
# Number of boosting trees (rounds). More trees = more complex model.
# Recommended range: 50–500. Start low for small datasets such as 100.
N_ESTIMATORS = 100

# Maximum depth of each tree. Deeper = more complex.
# Recommended range: 2–6. Use 3 for small datasets to avoid overfitting.
MAX_DEPTH = 2

# Step size shrinkage. Lower = slower learning but more robust.
# Recommended range: 0.01–0.3. Default 0.1 is a good start.
LEARNING_RATE = 0.05

# Fraction of training rows randomly sampled per tree.
# Adds randomness to prevent overfitting.
# Recommended range: 0.6–1.0. Default 0.8 works well.
SUBSAMPLE = 0.8

# Fraction of features randomly sampled per tree.
# Forces trees to use different features = reduces overfitting.
# Recommended range: 0.6–1.0. Default 0.8 works well.
COLSAMPLE_BYTREE = 0.7

# L1 regularization (Lasso). Pushes unimportant feature weights to zero.
# Higher value = more features ignored = simpler model.
# Recommended range: 0–10. Default 0 (no L1 penalty).
REG_ALPHA = 0.1

# L2 regularization (Ridge). Smooths feature weights to prevent any single
# feature from dominating. Higher value = smoother, more stable model.
# Recommended range: 0–10. Default 1.
REG_LAMBDA = 2

# Minimum sum of instance weight needed in a child node.
# Higher value = more conservative splits = less overfitting.
# For small datasets keep it low.
# Recommended range: 1–10. Default 1.
MIN_CHILD_WEIGHT = 3

# Minimum loss reduction required to make a split.
# Higher value = fewer splits = simpler tree.
# Acts as a pruning threshold.
# Recommended range: 0–5. Default 0 (no minimum).
GAMMA = 0.1

# Fixed random seed for reproducibility.
RANDOM_STATE = 42


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------
TARGET_COLUMN = "Glucose level (mg/dl)"

# Previous step (Step 8 Sub-task 3&4) identification patterns
PREV_STEP_FOLDER_IDENTIFIER     = "Master dataset 24F split scaled"
PREV_STEP_JSON_PIPELINE_STEP_ID = "STEP 8 (Sub-task 3 & 4)"


# --------------------------------------------------
# AUTO-DETECT LATEST PREVIOUS STEP FOLDER
# --------------------------------------------------
def find_latest_prev_step_folder(root_path):
    """
    Scans INPUT_ROOT for folders matching PREV_STEP_FOLDER_IDENTIFIER.
    Returns info about the most recently modified folder.
    Informational only — user still selects manually.
    """
    root = Path(root_path)
    if not root.exists():
        return {"found": False, "latest_folder": None, "all_folders": [], "total_folders": 0}

    candidates = [
        p for p in root.iterdir()
        if p.is_dir() and PREV_STEP_FOLDER_IDENTIFIER.lower() in p.name.lower()
    ]

    if not candidates:
        return {"found": False, "latest_folder": None, "all_folders": [], "total_folders": 0}

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    folder_info = []
    for p in candidates:
        try:
            mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            mtime = "Unknown"

        has_train = (p / "train").is_dir()
        has_test  = (p / "test").is_dir()
        has_json  = (p / "json").is_dir()

        folder_info.append({
            "folder_name":   p.name,
            "full_path":     str(p),
            "last_modified": mtime,
            "has_train":     has_train,
            "has_test":      has_test,
            "has_json":      has_json,
        })

    return {
        "found":         True,
        "latest_folder": folder_info[0],
        "all_folders":   folder_info,
        "total_folders": len(folder_info),
    }


def print_prev_step_folder_detection_report(detection_result):
    """Prints a clean terminal report of detected previous-step folders."""
    print(f"\n{'─' * 60}")
    print(f"🔍 STEP 8 (Sub-task 3&4) OUTPUT FOLDER AUTO-DETECTION")
    print(f"{'─' * 60}")

    if not detection_result["found"]:
        print(f"   ⚠️  No '{PREV_STEP_FOLDER_IDENTIFIER}' folders found in:")
        print(f"       {INPUT_ROOT}")
        print(f"   Please ensure Step 8 (Sub-task 3&4) has been run first.")
        return

    total  = detection_result["total_folders"]
    latest = detection_result["latest_folder"]

    print(f"   📁 Found {total} split-scaled folder(s) in:")
    print(f"       {INPUT_ROOT}")
    print(f"")
    print(f"   ✅ LATEST (most recently modified):")
    print(f"      📁 {latest['folder_name']}")
    print(f"         Last modified : {latest['last_modified']}")
    print(f"         Has train/    : {'✅' if latest['has_train'] else '❌'}")
    print(f"         Has test/     : {'✅' if latest['has_test']  else '❌'}")
    print(f"         Has json/     : {'✅' if latest['has_json']  else '❌'}")
    print(f"")

    if len(detection_result["all_folders"]) > 1:
        print(f"   📋 All detected folders (newest → oldest):")
        for idx, info in enumerate(detection_result["all_folders"], start=1):
            marker = "← LATEST" if idx == 1 else ""
            print(f"      {idx}. {info['folder_name']}  {marker}")
            print(f"         Modified : {info['last_modified']}")
        print(f"")

    print(f"   ℹ️  Folder browser will open at the root folder.")
    print(f"       Please select the folder listed above.")
    print(f"{'─' * 60}")


# --------------------------------------------------
# FOLDER SELECTOR POPUP (updated message)
# --------------------------------------------------
def popup_folder_selector(initial_dir):
    """
    Opens a folder dialog for user to select the Step 8 (Sub-task 3&4) output folder.
    Returns: Path to selected folder.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    messagebox.showinfo(
        title="Step 9 — XGBoost Training Pipeline",
        message=(
            "Select the Step 8 (Sub-task 3&4) OUTPUT FOLDER.\n\n"
            "──────────────────────────────────────────\n"
            "EXPECTED FOLDER NAME PATTERN:\n"
            "   Master dataset 24F split scaled <timestamp>\n\n"
            "EXPECTED CONTENTS:\n"
            "   Master dataset 24F split scaled <timestamp>/\n"
            "       ├── train/\n"
            "       │     ├── X_train_scaled.csv\n"
            "       │     └── y_train.csv\n"
            "       ├── test/\n"
            "       │     ├── X_test_scaled.csv\n"
            "       │     └── y_test.csv\n"
            "       └── json/\n"
            "             └── Master dataset 24F split scaled ....json\n\n"
            "──────────────────────────────────────────\n"
            "The script will automatically:\n"
            "  1. Find train/, test/, json/ subfolders\n"
            "  2. Validate this is a Step 8 (Sub-task 3&4) output\n"
            "  3. Train XGBoost regressor on glucose data\n"
            "  4. Generate predictions, metrics, and report\n\n"
            "Check the terminal for which folder was\n"
            "detected as the latest one.\n\n"
            "Click OK to open the folder browser."
        ),
    )

    selected_folder = filedialog.askdirectory(
        initialdir=str(initial_dir),
        title="Select Step 8 (Sub-task 3&4) Output FOLDER (Master dataset 24F split scaled ...)",
    )

    root.destroy()

    if not selected_folder:
        raise SystemExit("❌ User cancelled: No folder selected. Execution terminated.")

    return Path(selected_folder)


# --------------------------------------------------
# FILE FINDERS
# --------------------------------------------------
def find_train_test_files(folder_path):
    """
    Auto-detect the 4 required CSV files inside train/ and test/ subfolders.
    Returns: dict with paths to all 4 files.
    """
    folder = Path(folder_path)
    train_dir = folder / "train"
    test_dir  = folder / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"'train/' subfolder not found inside: {folder}")
    if not test_dir.exists():
        raise FileNotFoundError(f"'test/' subfolder not found inside: {folder}")

    required_files = {
        "X_train_scaled": train_dir / "X_train_scaled.csv",
        "y_train":        train_dir / "y_train.csv",
        "X_test_scaled":  test_dir / "X_test_scaled.csv",
        "y_test":         test_dir / "y_test.csv",
    }

    for label, fpath in required_files.items():
        if not fpath.exists():
            raise FileNotFoundError(f"Required file not found: {fpath}")

    return required_files


def find_json_in_folder(folder_path):
    """
    Auto-detect the JSON file inside the json/ subfolder.
    Returns: Path to JSON file, or None if not found.
    """
    folder = Path(folder_path)
    json_dir = folder / "json"

    if not json_dir.exists():
        return None

    json_files = list(json_dir.glob("*.json"))
    if len(json_files) == 0:
        return None

    if len(json_files) > 1:
        print(f"   ⚠️ Multiple JSON files found in json/. Using first one: {json_files[0].name}")

    return json_files[0]


def load_csv(file_path):
    """Load a CSV file and return DataFrame."""
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {file_path}")
    return df


def load_json(file_path):
    """Load a JSON file and return dict."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
# VALIDATION: IS THIS THE STEP 8 (Sub-task 3&4) OUTPUT?
# --------------------------------------------------
def validate_is_prev_step_output(folder_path, prev_step_json_data):
    """
    Validates that the selected folder is a genuine Step 8 (Sub-task 3&4) output.
    Hard-fails on errors (no warnings-only mode).

    Checks:
      1. Folder name contains PREV_STEP_FOLDER_IDENTIFIER
      2. train/, test/, json/ subfolders all exist
      3. Required 4 CSVs exist
      4. JSON has pipeline_step = "STEP 8 (Sub-task 3 & 4)"
    """
    folder_path = Path(folder_path)
    warnings    = []
    errors      = []

    # Check 1: Folder name
    if PREV_STEP_FOLDER_IDENTIFIER.lower() not in folder_path.name.lower():
        warnings.append(
            f"⚠️  Folder name does not match expected pattern.\n"
            f"   Selected : '{folder_path.name}'\n"
            f"   Expected : Contains '{PREV_STEP_FOLDER_IDENTIFIER}'\n"
            f"   Proceeding — but verify this is a split-scaled output."
        )

    # Check 2: Required subfolders
    train_dir = folder_path / "train"
    test_dir  = folder_path / "test"
    json_dir  = folder_path / "json"

    if not train_dir.exists():
        errors.append(f"'train/' subfolder missing inside: {folder_path}")
    if not test_dir.exists():
        errors.append(f"'test/' subfolder missing inside: {folder_path}")
    if not json_dir.exists():
        errors.append(f"'json/' subfolder missing inside: {folder_path}")

    # Check 3: Required CSV files
    required_csvs = [
        train_dir / "X_train_scaled.csv",
        train_dir / "y_train.csv",
        test_dir  / "X_test_scaled.csv",
        test_dir  / "y_test.csv",
    ]
    for csv_path in required_csvs:
        if not csv_path.exists():
            errors.append(f"Required CSV file missing: {csv_path}")

    # Check 4: JSON pipeline step identifier
    pipeline_step = ""
    if prev_step_json_data is not None:
        pipeline_step = prev_step_json_data.get("pipeline_info", {}).get("pipeline_step", "")
        if PREV_STEP_JSON_PIPELINE_STEP_ID.lower() not in pipeline_step.lower():
            warnings.append(
                f"⚠️  JSON log does not identify as the expected previous step.\n"
                f"   pipeline_step found : '{pipeline_step}'\n"
                f"   Expected            : Contains '{PREV_STEP_JSON_PIPELINE_STEP_ID}'"
            )
    else:
        warnings.append(
            "⚠️  No JSON file found in json/ subfolder. Pipeline traceability will be limited."
        )

    passed = len(errors) == 0

    return {
        "passed":             passed,
        "folder_name":        folder_path.name,
        "json_pipeline_step": pipeline_step,
        "warnings":           warnings,
        "errors":             errors,
    }


# --------------------------------------------------
# CONCISE PIPELINE CHAIN BUILDER (short, readable)
# --------------------------------------------------
def build_pipeline_chain_summary(prev_step_json_data, prev_json_path):
    """
    Builds a CONCISE pipeline traceability chain:
        Step 6 → Step 7 → Step 8 (1&2) → Step 8 (3&4) → Step 9
    Only essential identifiers from each upstream step.
    """
    if prev_step_json_data is None:
        return {
            "status": "not_found",
            "message": "Previous step JSON not available.",
        }

    def safe_get(d, *keys, default="Not recorded"):
        current = d
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    # ── Step 8 (Sub-task 3&4) — the directly previous step ──
    s8b_info  = safe_get(prev_step_json_data, "pipeline_info", default={})
    s8b_split = safe_get(prev_step_json_data, "sub_task_3_train_test_split", "split_details", default={})
    s8b_scale = safe_get(prev_step_json_data, "sub_task_4_robust_scaling", "scaler_parameters", default={})

    # ── Pipeline chain from Step 8 (Sub-task 3&4)'s "pipeline_chain_summary" section ──
    upstream_chain = safe_get(prev_step_json_data, "pipeline_chain_summary", default={})

    chain = {
        "description": (
            "Concise pipeline traceability: Step 6 → Step 7 → Step 8 (1&2) → "
            "Step 8 (3&4) → Step 9 (XGBoost). Only key identifiers from each step."
        ),

        # Upstream steps (extracted from prev step's own chain)
        "step_6_combine":             safe_get(upstream_chain, "step_6_combine",             default={}),
        "step_7_feature_engineering": safe_get(upstream_chain, "step_7_feature_engineering", default={}),
        "step_8_cleaning_sub_1_2":    safe_get(upstream_chain, "step_8_cleaning_sub_1_2",    default={}),

        # Direct previous step (Step 8 Sub-task 3&4)
        "step_8_split_scale_sub_3_4": {
            "execution_date":       safe_get(s8b_info, "execution_date_readable"),
            "train_samples":        safe_get(s8b_split, "train_samples_actual"),
            "test_samples":         safe_get(s8b_split, "test_samples_actual"),
            "random_state":         safe_get(s8b_split, "random_state"),
            "scaler_type":          safe_get(s8b_scale, "scaler_type"),
            "scaler_fitted_on":     safe_get(s8b_scale, "fitted_on"),
            "prev_json_used_here":  str(prev_json_path) if prev_json_path else "Not found",
        },
    }

    return chain


# --------------------------------------------------
# PHASE 1: LOAD DATA
# ⚠️ ALL VALIDATION LOGIC PRESERVED
# --------------------------------------------------
def load_all_data(file_paths):
    """Load all 4 CSV files and validate."""
    print(f"\n{'─' * 60}")
    print(f"📥 PHASE 1: LOADING DATA")
    print(f"{'─' * 60}")

    X_train    = load_csv(file_paths["X_train_scaled"])
    y_train_df = load_csv(file_paths["y_train"])
    X_test     = load_csv(file_paths["X_test_scaled"])
    y_test_df  = load_csv(file_paths["y_test"])

    # Extract y as Series
    y_train = y_train_df[TARGET_COLUMN]
    y_test  = y_test_df[TARGET_COLUMN]

    feature_columns = list(X_train.columns)

    print(f"   ✅ X_train: {X_train.shape[0]} rows × {X_train.shape[1]} columns")
    print(f"   ✅ y_train: {len(y_train)} values")
    print(f"   ✅ X_test:  {X_test.shape[0]} rows × {X_test.shape[1]} columns")
    print(f"   ✅ y_test:  {len(y_test)} values")

    # Validate
    checks_passed = True

    # Column match
    if list(X_train.columns) != list(X_test.columns):
        print(f"   ❌ Feature columns mismatch between train and test!")
        checks_passed = False
    else:
        print(f"   ✅ Feature columns match: {len(feature_columns)} features")

    # NaN check
    train_nans = int(X_train.isna().sum().sum() + y_train.isna().sum())
    test_nans  = int(X_test.isna().sum().sum() + y_test.isna().sum())
    if train_nans > 0 or test_nans > 0:
        print(f"   ❌ NaN detected! Train: {train_nans}, Test: {test_nans}")
        checks_passed = False
    else:
        print(f"   ✅ No NaN values found")

    # Row match
    if X_train.shape[0] != len(y_train):
        print(f"   ❌ X_train rows ({X_train.shape[0]}) != y_train length ({len(y_train)})")
        checks_passed = False
    if X_test.shape[0] != len(y_test):
        print(f"   ❌ X_test rows ({X_test.shape[0]}) != y_test length ({len(y_test)})")
        checks_passed = False

    if checks_passed:
        print(f"   ✅ All validation checks passed")
    else:
        print(f"   ⚠️ Some validation checks failed — proceed with caution")

    # Display glucose summary
    print(f"\n   📊 Glucose summary:")
    print(f"      Train — min: {y_train.min():.1f}, max: {y_train.max():.1f}, mean: {y_train.mean():.1f}")
    print(f"      Test  — min: {y_test.min():.1f}, max: {y_test.max():.1f}, mean: {y_test.mean():.1f}")

    return X_train, X_test, y_train, y_test, feature_columns


# --------------------------------------------------
# PHASE 3: TRAIN MODEL
# ⚠️ ALL HYPERPARAMETERS AND MODEL LOGIC PRESERVED EXACTLY
# --------------------------------------------------
def train_xgboost_model(X_train, y_train):
    """
    Create and train XGBoost Regressor.
    Returns: (trained_model, training_time_seconds)
    """
    print(f"\n{'─' * 60}")
    print(f"🚀 PHASE 3: TRAINING XGBOOST MODEL")
    print(f"{'─' * 60}")

    # Display hyperparameters
    print(f"\n   📊 Hyperparameter Configuration:")
    print(f"      n_estimators      = {N_ESTIMATORS}")
    print(f"      max_depth         = {MAX_DEPTH}")
    print(f"      learning_rate     = {LEARNING_RATE}")
    print(f"      subsample         = {SUBSAMPLE}")
    print(f"      colsample_bytree  = {COLSAMPLE_BYTREE}")
    print(f"      reg_alpha (L1)    = {REG_ALPHA}")
    print(f"      reg_lambda (L2)   = {REG_LAMBDA}")
    print(f"      min_child_weight  = {MIN_CHILD_WEIGHT}")
    print(f"      gamma             = {GAMMA}")
    print(f"      random_state      = {RANDOM_STATE}")
    print(f"      objective         = reg:squarederror")
    print(f"      tree_method       = auto")

    model = xgb.XGBRegressor(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        learning_rate=LEARNING_RATE,
        subsample=SUBSAMPLE,
        colsample_bytree=COLSAMPLE_BYTREE,
        reg_alpha=REG_ALPHA,
        reg_lambda=REG_LAMBDA,
        min_child_weight=MIN_CHILD_WEIGHT,
        gamma=GAMMA,
        random_state=RANDOM_STATE,
        objective="reg:squarederror",
        tree_method="auto",
        verbosity=0,
    )

    print(f"\n   🔧 Training model on {X_train.shape[0]} samples, {X_train.shape[1]} features...")

    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    print(f"   ✅ Model trained successfully in {training_time:.3f} seconds")

    return model, training_time


# --------------------------------------------------
# PHASE 4: PREDICTIONS
# ⚠️ PRESERVED EXACTLY
# --------------------------------------------------
def make_predictions(model, X_train, X_test):
    """
    Make predictions on both train and test sets.
    Returns: (y_pred_train, y_pred_test)
    """
    print(f"\n{'─' * 60}")
    print(f"🔮 PHASE 4: MAKING PREDICTIONS")
    print(f"{'─' * 60}")

    y_pred_train = model.predict(X_train)
    print(f"   ✅ Train predictions: {len(y_pred_train)} values")

    y_pred_test = model.predict(X_test)
    print(f"   ✅ Test predictions:  {len(y_pred_test)} values")

    return y_pred_train, y_pred_test


# --------------------------------------------------
# PHASE 5: EVALUATION METRICS
# ⚠️ ALL METRIC CALCULATIONS PRESERVED EXACTLY
# --------------------------------------------------
def calculate_metrics(y_actual, y_predicted, set_name):
    """
    Calculate all evaluation metrics for a given set.
    Returns: metrics dict.
    """
    mae  = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mape = mean_absolute_percentage_error(y_actual, y_predicted) * 100

    # R² needs at least 2 samples
    if len(y_actual) >= 2:
        r2 = r2_score(y_actual, y_predicted)
    else:
        r2 = None

    metrics = {
        "set_name":     set_name,
        "sample_count": int(len(y_actual)),
        "MAE_mg_dL":    round(float(mae), 4),
        "RMSE_mg_dL":   round(float(rmse), 4),
        "R2_score":     round(float(r2), 4) if r2 is not None else "undefined (need ≥2 samples)",
        "MAPE_percent": round(float(mape), 4),
    }

    return metrics


def display_metrics(train_metrics, test_metrics):
    """Display evaluation metrics for both train and test sets."""
    print(f"\n{'─' * 60}")
    print(f"📊 PHASE 5: EVALUATION METRICS")
    print(f"{'─' * 60}")

    print(f"\n   {'Metric':<25} {'TRAIN':>15} {'TEST':>15}")
    print(f"   {'─' * 25} {'─' * 15} {'─' * 15}")

    # MAE
    print(f"   {'MAE (mg/dL)':<25} {train_metrics['MAE_mg_dL']:>15.4f} {test_metrics['MAE_mg_dL']:>15.4f}")

    # RMSE
    print(f"   {'RMSE (mg/dL)':<25} {train_metrics['RMSE_mg_dL']:>15.4f} {test_metrics['RMSE_mg_dL']:>15.4f}")

    # R²
    train_r2 = train_metrics['R2_score']
    test_r2  = test_metrics['R2_score']
    train_r2_str = f"{train_r2:.4f}" if isinstance(train_r2, float) else str(train_r2)
    test_r2_str  = f"{test_r2:.4f}"  if isinstance(test_r2, float)  else str(test_r2)
    print(f"   {'R² Score':<25} {train_r2_str:>15} {test_r2_str:>15}")

    # MAPE
    print(f"   {'MAPE (%)':<25} {train_metrics['MAPE_percent']:>15.4f} {test_metrics['MAPE_percent']:>15.4f}")

    # Sample count
    print(f"   {'Samples':<25} {train_metrics['sample_count']:>15} {test_metrics['sample_count']:>15}")


# --------------------------------------------------
# PHASE 6: OVERFITTING ANALYSIS
# ⚠️ ALL THRESHOLDS AND DIAGNOSIS LOGIC PRESERVED EXACTLY
# --------------------------------------------------
def analyze_overfitting(train_metrics, test_metrics):
    """
    Automatically analyze if model is overfitting.
    Returns: overfitting analysis dict.
    """
    print(f"\n{'─' * 60}")
    print(f"🔍 PHASE 6: OVERFITTING ANALYSIS")
    print(f"{'─' * 60}")

    analysis = {
        "train_mae":  train_metrics["MAE_mg_dL"],
        "test_mae":   test_metrics["MAE_mg_dL"],
        "train_rmse": train_metrics["RMSE_mg_dL"],
        "test_rmse":  test_metrics["RMSE_mg_dL"],
        "diagnosis":  "",
        "details":    [],
    }

    # MAE comparison
    mae_ratio  = test_metrics["MAE_mg_dL"] / train_metrics["MAE_mg_dL"]  if train_metrics["MAE_mg_dL"]  > 0 else float('inf')

    # RMSE comparison
    rmse_ratio = test_metrics["RMSE_mg_dL"] / train_metrics["RMSE_mg_dL"] if train_metrics["RMSE_mg_dL"] > 0 else float('inf')

    print(f"\n   📊 Error Ratio Analysis:")
    print(f"      MAE ratio  (Test/Train): {mae_ratio:.2f}")
    print(f"      RMSE ratio (Test/Train): {rmse_ratio:.2f}")

    # R² comparison
    train_r2 = train_metrics["R2_score"]
    test_r2  = test_metrics["R2_score"]

    if isinstance(train_r2, float) and isinstance(test_r2, float):
        r2_gap = train_r2 - test_r2
        print(f"      R² gap     (Train-Test): {r2_gap:.4f}")
        analysis["r2_gap"] = round(r2_gap, 4)

    # Diagnosis
    print(f"\n   🩺 Diagnosis:")

    if mae_ratio > 3.0:
        diagnosis = "SEVERE OVERFITTING"
        emoji = "🔴"
        details = [
            "Test error is more than 3x the train error.",
            "Model has memorized training data instead of learning patterns.",
            "Recommendations:",
            "  → Reduce max_depth (try 2)",
            "  → Reduce n_estimators (try 50)",
            "  → Increase reg_alpha or reg_lambda",
            "  → Increase min_child_weight",
            "  → Collect more training data",
        ]
    elif mae_ratio > 2.0:
        diagnosis = "MODERATE OVERFITTING"
        emoji = "🟠"
        details = [
            "Test error is 2-3x the train error.",
            "Model is partially overfitting to training data.",
            "Recommendations:",
            "  → Reduce max_depth by 1",
            "  → Increase subsample (try 0.7)",
            "  → Increase reg_lambda (try 3-5)",
            "  → Increase gamma (try 0.1-0.5)",
        ]
    elif mae_ratio > 1.5:
        diagnosis = "MILD OVERFITTING"
        emoji = "🟡"
        details = [
            "Test error is 1.5-2x the train error.",
            "Some overfitting present but may be acceptable.",
            "Model is learning some patterns but not generalizing perfectly.",
            "Consider: slight reduction in max_depth or increase in regularization.",
        ]
    elif mae_ratio >= 0.8:
        diagnosis = "GOOD GENERALIZATION"
        emoji = "🟢"
        details = [
            "Train and test errors are similar.",
            "Model is generalizing well to unseen data.",
            "No significant overfitting detected.",
            "This is the ideal scenario.",
        ]
    else:
        diagnosis = "UNUSUAL — TEST BETTER THAN TRAIN"
        emoji = "🔵"
        details = [
            "Test error is lower than train error.",
            "This is unusual and may indicate:",
            "  → Test set happens to be easier than training set",
            "  → Very small test set (unreliable metrics)",
            "  → Random variation due to small dataset",
        ]

    analysis["diagnosis"]  = diagnosis
    analysis["details"]    = details
    analysis["mae_ratio"]  = round(mae_ratio, 4)
    analysis["rmse_ratio"] = round(rmse_ratio, 4)

    print(f"      {emoji} {diagnosis}")
    for d in details:
        print(f"      {d}")

    # Small dataset warning
    if test_metrics["sample_count"] < 5:
        warning = (
            "⚠️ WARNING: Test set has fewer than 5 samples. "
            "All metrics should be interpreted with extreme caution. "
            "Results may not be statistically reliable."
        )
        print(f"\n      {warning}")
        analysis["small_dataset_warning"] = warning

    return analysis


# --------------------------------------------------
# PHASE 7: FEATURE IMPORTANCE
# ⚠️ PRESERVED EXACTLY
# --------------------------------------------------
def analyze_feature_importance(model, feature_columns):
    """
    Extract and rank feature importance from XGBoost model.
    Returns: feature importance DataFrame and dict.
    """
    print(f"\n{'─' * 60}")
    print(f"📊 PHASE 7: FEATURE IMPORTANCE ANALYSIS")
    print(f"{'─' * 60}")

    importances = model.feature_importances_

    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        "Feature":    feature_columns,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Add rank and percentage
    total_importance = importances.sum()
    importance_df["Rank"]                  = range(1, len(importance_df) + 1)
    importance_df["Percentage"]            = (importance_df["Importance"] / total_importance * 100).round(2)
    importance_df["Cumulative_Percentage"] = importance_df["Percentage"].cumsum().round(2)

    # Display
    print(f"\n   {'Rank':<6} {'Feature':<35} {'Importance':>12} {'%':>8} {'Cumul%':>8}")
    print(f"   {'─' * 6} {'─' * 35} {'─' * 12} {'─' * 8} {'─' * 8}")

    for _, row in importance_df.iterrows():
        bar_len = int(row["Percentage"] / 2)
        bar = "█" * bar_len
        print(f"   {int(row['Rank']):<6} {row['Feature']:<35} {row['Importance']:>12.4f} "
              f"{row['Percentage']:>7.2f}% {row['Cumulative_Percentage']:>7.2f}%  {bar}")

    # Identify zero-importance features
    zero_features = importance_df[importance_df["Importance"] == 0]["Feature"].tolist()
    if zero_features:
        print(f"\n   ⚠️ Features with ZERO importance ({len(zero_features)}):")
        for f in zero_features:
            print(f"      → {f} (contributes nothing — candidate for removal)")

    # Top contributors
    top_80 = importance_df[importance_df["Cumulative_Percentage"] <= 80]
    if len(top_80) > 0:
        print(f"\n   📌 Top features contributing ~80% of importance:")
        for _, row in top_80.iterrows():
            print(f"      → {row['Feature']} ({row['Percentage']:.2f}%)")

    # Build log dict
    importance_log = {
        "total_features":            len(feature_columns),
        "zero_importance_features":  zero_features,
        "zero_importance_count":     len(zero_features),
        "feature_ranking": [
            {
                "rank":                  int(row["Rank"]),
                "feature":               row["Feature"],
                "importance":            float(row["Importance"]),
                "percentage":            float(row["Percentage"]),
                "cumulative_percentage": float(row["Cumulative_Percentage"]),
            }
            for _, row in importance_df.iterrows()
        ],
    }

    return importance_df, importance_log


# --------------------------------------------------
# PHASE 8: ACTUAL VS PREDICTED TABLE
# ⚠️ PRESERVED EXACTLY
# --------------------------------------------------
def build_prediction_tables(y_train, y_pred_train, y_test, y_pred_test):
    """
    Build actual vs predicted comparison tables.
    Returns: (train_table_df, test_table_df, tables_log)
    """
    print(f"\n{'─' * 60}")
    print(f"📋 PHASE 8: ACTUAL vs PREDICTED ANALYSIS")
    print(f"{'─' * 60}")

    # ── Train table ──
    train_table = pd.DataFrame({
        "Sample":                  range(1, len(y_train) + 1),
        "Actual_Glucose_mg_dL":    y_train.values,
        "Predicted_Glucose_mg_dL": np.round(y_pred_train, 2),
        "Error_mg_dL":             np.round(np.abs(y_train.values - y_pred_train), 2),
        "Percent_Error": np.round(
            np.abs(y_train.values - y_pred_train) / y_train.values * 100, 2
        ),
    })

    print(f"\n   📊 TRAIN SET — Actual vs Predicted:")
    print(f"   {'Sample':>8} {'Actual':>12} {'Predicted':>12} {'Error':>10} {'% Error':>10}")
    print(f"   {'─' * 8} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 10}")
    for _, row in train_table.iterrows():
        print(f"   {int(row['Sample']):>8} {row['Actual_Glucose_mg_dL']:>12.1f} "
              f"{row['Predicted_Glucose_mg_dL']:>12.2f} "
              f"{row['Error_mg_dL']:>10.2f} {row['Percent_Error']:>9.2f}%")

    # ── Test table ──
    test_table = pd.DataFrame({
        "Sample":                  range(1, len(y_test) + 1),
        "Actual_Glucose_mg_dL":    y_test.values,
        "Predicted_Glucose_mg_dL": np.round(y_pred_test, 2),
        "Error_mg_dL":             np.round(np.abs(y_test.values - y_pred_test), 2),
        "Percent_Error": np.round(
            np.abs(y_test.values - y_pred_test) / y_test.values * 100, 2
        ),
    })

    print(f"\n   📊 TEST SET — Actual vs Predicted:")
    print(f"   {'Sample':>8} {'Actual':>12} {'Predicted':>12} {'Error':>10} {'% Error':>10}")
    print(f"   {'─' * 8} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 10}")
    for _, row in test_table.iterrows():
        print(f"   {int(row['Sample']):>8} {row['Actual_Glucose_mg_dL']:>12.1f} "
              f"{row['Predicted_Glucose_mg_dL']:>12.2f} "
              f"{row['Error_mg_dL']:>10.2f} {row['Percent_Error']:>9.2f}%")

    # Summary
    print(f"\n   📊 Error Summary:")
    print(f"      Train — Avg error: {train_table['Error_mg_dL'].mean():.2f} mg/dL, "
          f"Avg % error: {train_table['Percent_Error'].mean():.2f}%")
    print(f"      Test  — Avg error: {test_table['Error_mg_dL'].mean():.2f} mg/dL, "
          f"Avg % error: {test_table['Percent_Error'].mean():.2f}%")

    # Build log
    tables_log = {
        "train_predictions":       train_table.to_dict(orient="records"),
        "test_predictions":        test_table.to_dict(orient="records"),
        "train_avg_error_mg_dL":   round(float(train_table["Error_mg_dL"].mean()), 4),
        "train_avg_percent_error": round(float(train_table["Percent_Error"].mean()), 4),
        "test_avg_error_mg_dL":    round(float(test_table["Error_mg_dL"].mean()), 4),
        "test_avg_percent_error":  round(float(test_table["Percent_Error"].mean()), 4),
    }

    return train_table, test_table, tables_log


# --------------------------------------------------
# PHASE 9: BUILD REPORT + SAVE
# --------------------------------------------------
def build_xgboost_report_data(
    input_folder_path,
    output_folder_path,
    output_file_paths,
    X_train,
    X_test,
    y_train,
    y_test,
    feature_columns,
    training_time,
    train_metrics,
    test_metrics,
    overfitting_analysis,
    importance_log,
    tables_log,
    timestamp_str,
    pipeline_chain_summary,
):
    """Build complete JSON report of the XGBoost pipeline."""

    hyperparameters = {
        "n_estimators":     N_ESTIMATORS,
        "max_depth":        MAX_DEPTH,
        "learning_rate":    LEARNING_RATE,
        "subsample":        SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "reg_alpha":        REG_ALPHA,
        "reg_lambda":       REG_LAMBDA,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "gamma":            GAMMA,
        "random_state":     RANDOM_STATE,
        "objective":        "reg:squarederror",
        "tree_method":      "auto",
    }

    hyperparameter_explanations = {
        "n_estimators":     "Number of boosting trees. More trees = more complex model.",
        "max_depth":        "Maximum depth per tree. Deeper = more complex. Low for small data.",
        "learning_rate":    "Step size per boosting round. Lower = slower but more robust.",
        "subsample":        "Fraction of training rows sampled per tree. Adds randomness.",
        "colsample_bytree": "Fraction of features sampled per tree. Diversifies trees.",
        "reg_alpha":        "L1 regularization. Pushes unimportant feature weights to zero.",
        "reg_lambda":       "L2 regularization. Smooths weights to prevent dominance.",
        "min_child_weight": "Minimum weight in child node. Higher = more conservative.",
        "gamma":            "Minimum loss reduction for split. Higher = fewer splits = simpler tree.",
        "random_state":     "Fixed seed for reproducibility.",
        "objective":        "Loss function. reg:squarederror = standard regression.",
        "tree_method":      "Algorithm for tree construction. auto = best available.",
    }

    xgboost_report_data = {
        "pipeline_info": {
            "pipeline_name":          "XGBoost Glucose Prediction Model",
            "pipeline_step":          "STEP 9",
            "execution_timestamp":    timestamp_str,
            "execution_date_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "previous_step":          "STEP 8 (Sub-task 3 & 4) — Train/Test Split + Scaling",
        },

        # NEW: Concise pipeline chain
        "pipeline_chain_summary": pipeline_chain_summary,

        "file_paths": {
            "input_folder":  str(input_folder_path),
            "output_folder": str(output_folder_path),
            "output_files":  output_file_paths,
        },

        "data_summary": {
            "train_samples":   int(X_train.shape[0]),
            "test_samples":    int(X_test.shape[0]),
            "total_samples":   int(X_train.shape[0] + X_test.shape[0]),
            "feature_count":   len(feature_columns),
            "feature_columns": feature_columns,
            "target_column":   TARGET_COLUMN,
        },

        "model_configuration": {
            "model_type":                  "XGBRegressor",
            "library":                     "xgboost",
            "hyperparameters":             hyperparameters,
            "hyperparameter_explanations": hyperparameter_explanations,
            "training_time_seconds":       round(training_time, 4),
        },

        "evaluation_metrics": {
            "train_metrics": train_metrics,
            "test_metrics":  test_metrics,
        },

        "overfitting_analysis": overfitting_analysis,
        "feature_importance":   importance_log,
        "predictions":          tables_log,

        "interpretation_guide": {
            "MAE":  "Mean Absolute Error — average prediction error in mg/dL. Lower is better.",
            "RMSE": "Root Mean Squared Error — penalizes large errors more. Lower is better.",
            "R2":   "R-squared — fraction of variance explained. 1.0 = perfect. Can be negative.",
            "MAPE": "Mean Absolute Percentage Error — error as percentage. Lower is better.",
            "overfitting_diagnosis": (
                "Compares train vs test error. "
                "If test error >> train error = overfitting. "
                "If similar = good generalization."
            ),
        },
    }

    return xgboost_report_data


def save_all_outputs(model, train_table, test_table, importance_df,
                     xgboost_report_data, output_root, timestamp_str):
    """
    Save all output files.

    XGBoost results & Conclusions <timestamp>/
        ├── model/
        │   └── xgboost_glucose_model.json
        ├── predictions/
        │   ├── train_predictions.csv
        │   └── test_predictions.csv
        ├── importance/
        │   └── feature_importance.csv
        └── report/
            └── XGBoost_full_report_<timestamp>.json
    """
    main_folder_name = f"XGBoost results & Conclusions {timestamp_str}"
    main_dir         = output_root / main_folder_name
    main_dir.mkdir(parents=True, exist_ok=True)

    model_dir      = main_dir / "model"
    pred_dir       = main_dir / "predictions"
    importance_dir = main_dir / "importance"
    report_dir     = main_dir / "report"

    model_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    importance_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    replaced_files = []

    # ── Save model ──
    model_path = model_dir / "xgboost_glucose_model.json"
    pre_info = check_existing_file(model_path)
    if pre_info["exists"]:
        replaced_files.append({"label": "Model file", "path": str(model_path),
                                "old_size_kb": pre_info["size_kb"]})

    model.save_model(str(model_path))
    post_info = check_existing_file(model_path)
    print(f"\n   💾 Model saved: {model_path.name} ({post_info['size_kb']:.2f} KB)")

    # ── Save predictions ──
    train_pred_path = pred_dir / "train_predictions.csv"
    test_pred_path  = pred_dir / "test_predictions.csv"

    for fpath, data, label in [
        (train_pred_path, train_table, "Train predictions"),
        (test_pred_path,  test_table,  "Test predictions"),
    ]:
        pre_info = check_existing_file(fpath)
        if pre_info["exists"]:
            replaced_files.append({"label": label, "path": str(fpath),
                                    "old_size_kb": pre_info["size_kb"]})
        data.to_csv(fpath, index=False)
        post_info = check_existing_file(fpath)
        print(f"   💾 {label}: {fpath.name} ({post_info['size_kb']:.2f} KB)")

    # ── Save feature importance ──
    importance_path = importance_dir / "feature_importance.csv"
    pre_info = check_existing_file(importance_path)
    if pre_info["exists"]:
        replaced_files.append({"label": "Feature importance", "path": str(importance_path),
                                "old_size_kb": pre_info["size_kb"]})
    importance_df.to_csv(importance_path, index=False)
    post_info = check_existing_file(importance_path)
    print(f"   💾 Feature importance: {importance_path.name} ({post_info['size_kb']:.2f} KB)")

    # ── Save report JSON ──
    report_path = report_dir / f"XGBoost_full_report_{timestamp_str}.json"
    pre_info = check_existing_file(report_path)
    if pre_info["exists"]:
        replaced_files.append({"label": "Full report JSON", "path": str(report_path),
                                "old_size_kb": pre_info["size_kb"]})

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(xgboost_report_data, f, indent=4, default=str)
    post_info = check_existing_file(report_path)
    print(f"   💾 Full report: {report_path.name} ({post_info['size_kb']:.2f} KB)")

    # Report replaced files
    if replaced_files:
        print(f"\n   ♻️ REPLACED {len(replaced_files)} existing file(s):")
        for rf in replaced_files:
            new_size = check_existing_file(rf["path"])["size_kb"]
            old_str = f"{rf['old_size_kb']:.2f} KB" if rf["old_size_kb"] else "N/A"
            new_str = f"{new_size:.2f} KB"          if new_size          else "N/A"
            print(f"      {rf['label']}: {old_str} → {new_str}")
    else:
        print(f"\n   🆕 All output files are newly created.")

    output_file_paths = {
        "main_folder":        str(main_dir),
        "model":              str(model_path),
        "train_predictions":  str(train_pred_path),
        "test_predictions":   str(test_pred_path),
        "feature_importance": str(importance_path),
        "full_report":        str(report_path),
    }

    return main_dir, output_file_paths


# --------------------------------------------------
# MAIN EXECUTION
# --------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("🎯 STEP 9: XGBoost MODEL TRAINING + EVALUATION")
    print("   Model: XGBRegressor (Gradient Boosted Trees)")
    print("   Target: Glucose Level Prediction (mg/dL)")
    print("=" * 70)

    # Validate paths
    if not INPUT_ROOT.exists():
        raise SystemExit(f"❌ Input folder does not exist: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Generate timestamp (pipeline-consistent format)
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # ── Step 1: Auto-detect latest previous-step folder (informational) ──
    print(f"\n🔍 Scanning for latest split-scaled output folder...")
    prev_detection_result = find_latest_prev_step_folder(INPUT_ROOT)
    print_prev_step_folder_detection_report(prev_detection_result)

    # ── Step 2: Select input FOLDER ──
    print(f"📂 Opening folder selector at: {INPUT_ROOT}")
    input_folder = popup_folder_selector(INPUT_ROOT)
    print(f"📁 Selected folder : {input_folder.name}")
    print(f"   Full path       : {input_folder}")

    # ── Step 3: Auto-detect required CSVs ──
    print(f"\n{'─' * 60}")
    print(f"🔍 AUTO-DETECTING TRAIN/TEST FILES")
    print(f"{'─' * 60}")
    file_paths = find_train_test_files(input_folder)
    for label, fpath in file_paths.items():
        print(f"   📄 {label}: {fpath.name}")

    # ── Step 4: Auto-detect previous JSON ──
    print(f"\n   🔍 Looking for JSON in json/ subfolder...")
    prev_json_path = find_json_in_folder(input_folder)
    prev_step_json_data = None
    if prev_json_path is not None:
        print(f"   📄 JSON found: {prev_json_path.name}")
        try:
            prev_step_json_data = load_json(prev_json_path)
            print(f"   ✅ JSON loaded successfully.")
        except Exception as e:
            print(f"   ⚠️  Could not parse JSON: {e}")
    else:
        print(f"   ⚠️  No JSON file found in json/ subfolder.")

    # ── Step 5: Validate previous-step output ──
    print(f"\n{'─' * 60}")
    print(f"🔍 VALIDATING SELECTED FOLDER IS STEP 8 (Sub-task 3&4) OUTPUT")
    print(f"{'─' * 60}")

    prev_validation = validate_is_prev_step_output(input_folder, prev_step_json_data)

    # Print warnings
    for warning in prev_validation["warnings"]:
        print(warning)

    # Hard-fail on errors
    if not prev_validation["passed"]:
        print(f"\n❌ PREVIOUS STEP OUTPUT VALIDATION FAILED:")
        for error in prev_validation["errors"]:
            print(f"\n   ❌ {error}")
        raise SystemExit(
            "\n❌ Execution aborted: Selected folder is not a valid Step 8 (Sub-task 3&4) output.\n"
            f"   Please select a '{PREV_STEP_FOLDER_IDENTIFIER}' folder from:\n"
            f"   {INPUT_ROOT}"
        )

    print(f"   ✅ Previous-step validation passed.")
    print(f"   📁 Folder name : {prev_validation['folder_name']}")
    print(f"   📄 JSON step   : {prev_validation['json_pipeline_step']}")

    # ── Step 6: Build concise pipeline chain ──
    print(f"\n{'─' * 60}")
    print(f"📋 BUILDING CONCISE PIPELINE CHAIN SUMMARY")
    print(f"{'─' * 60}")

    pipeline_chain_summary = build_pipeline_chain_summary(prev_step_json_data, prev_json_path)

    if pipeline_chain_summary.get("status") != "not_found":
        s6  = pipeline_chain_summary.get("step_6_combine", {})
        s7  = pipeline_chain_summary.get("step_7_feature_engineering", {})
        s8a = pipeline_chain_summary.get("step_8_cleaning_sub_1_2", {})
        s8b = pipeline_chain_summary.get("step_8_split_scale_sub_3_4", {})
        print(f"   ✅ Pipeline chain summary built.")
        print(f"      Step 6  : {s6.get('build_date',     'N/A')}  →  {s6.get('successfully_compiled', '?')} subjects")
        print(f"      Step 7  : {s7.get('execution_date', 'N/A')}  →  {s7.get('total_features',        '?')} features")
        print(f"      Step 8a : {s8a.get('execution_date','N/A')}  →  {s8a.get('output_rows',          '?')} rows after cleaning")
        print(f"      Step 8b : {s8b.get('execution_date','N/A')}  →  Train={s8b.get('train_samples', '?')}, Test={s8b.get('test_samples', '?')}")
    else:
        print(f"   ⚠️  Pipeline chain summary skipped (no previous JSON data).")

    # ── PHASE 1: Load all data ──
    X_train, X_test, y_train, y_test, feature_columns = load_all_data(file_paths)

    # ── PHASE 3: Train model ──
    model, training_time = train_xgboost_model(X_train, y_train)

    # ── PHASE 4: Predictions ──
    y_pred_train, y_pred_test = make_predictions(model, X_train, X_test)

    # ── PHASE 5: Evaluation metrics ──
    train_metrics = calculate_metrics(y_train, y_pred_train, "TRAIN")
    test_metrics  = calculate_metrics(y_test,  y_pred_test,  "TEST")
    display_metrics(train_metrics, test_metrics)

    # ── PHASE 6: Overfitting analysis ──
    overfitting_analysis = analyze_overfitting(train_metrics, test_metrics)

    # ── PHASE 7: Feature importance ──
    importance_df, importance_log = analyze_feature_importance(model, feature_columns)

    # ── PHASE 8: Actual vs Predicted ──
    train_table, test_table, tables_log = build_prediction_tables(
        y_train, y_pred_train, y_test, y_pred_test
    )

    # ── PHASE 9: Build report and save ──
    print(f"\n{'─' * 60}")
    print(f"💾 PHASE 9: SAVING MODEL + RESULTS")
    print(f"{'─' * 60}")

    # Build report (placeholder paths — updated after save)
    xgboost_report_data = build_xgboost_report_data(
        input_folder_path=input_folder,
        output_folder_path=OUTPUT_ROOT,
        output_file_paths={},
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        training_time=training_time,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        overfitting_analysis=overfitting_analysis,
        importance_log=importance_log,
        tables_log=tables_log,
        timestamp_str=timestamp_str,
        pipeline_chain_summary=pipeline_chain_summary,
    )

    main_output_dir, output_file_paths = save_all_outputs(
        model=model,
        train_table=train_table,
        test_table=test_table,
        importance_df=importance_df,
        xgboost_report_data=xgboost_report_data,
        output_root=OUTPUT_ROOT,
        timestamp_str=timestamp_str,
    )

    # Update report with actual file paths and re-save
    xgboost_report_data["file_paths"]["output_files"] = output_file_paths
    report_path = Path(output_file_paths["full_report"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(xgboost_report_data, f, indent=4, default=str)

    # ── Final Summary ──
    print(f"\n{'=' * 70}")
    print(f"📌 XGBOOST PIPELINE — FINAL SUMMARY")
    print(f"{'=' * 70}")
    print(f"")
    print(f"   📥 Input: {input_folder.name}")
    print(f"      Train: {X_train.shape[0]} samples × {X_train.shape[1]} features")
    print(f"      Test:  {X_test.shape[0]} samples × {X_test.shape[1]} features")
    print(f"")
    print(f"   ⚙️ Model Configuration:")
    print(f"      Trees: {N_ESTIMATORS}, Depth: {MAX_DEPTH}, LR: {LEARNING_RATE}")
    print(f"      Subsample: {SUBSAMPLE}, ColSample: {COLSAMPLE_BYTREE}")
    print(f"      L1(α): {REG_ALPHA}, L2(λ): {REG_LAMBDA}")
    print(f"      MinChildWeight: {MIN_CHILD_WEIGHT}, Gamma: {GAMMA}")
    print(f"      Training time: {training_time:.3f} seconds")
    print(f"")
    print(f"   📊 Performance:")
    print(f"      {'':>20} {'TRAIN':>12} {'TEST':>12}")
    print(f"      {'MAE (mg/dL)':>20} {train_metrics['MAE_mg_dL']:>12.4f} {test_metrics['MAE_mg_dL']:>12.4f}")
    print(f"      {'RMSE (mg/dL)':>20} {train_metrics['RMSE_mg_dL']:>12.4f} {test_metrics['RMSE_mg_dL']:>12.4f}")

    train_r2 = train_metrics['R2_score']
    test_r2  = test_metrics['R2_score']
    train_r2_str = f"{train_r2:.4f}" if isinstance(train_r2, float) else str(train_r2)
    test_r2_str  = f"{test_r2:.4f}"  if isinstance(test_r2, float)  else str(test_r2)
    print(f"      {'R² Score':>20} {train_r2_str:>12} {test_r2_str:>12}")
    print(f"      {'MAPE (%)':>20} {train_metrics['MAPE_percent']:>12.4f} {test_metrics['MAPE_percent']:>12.4f}")
    print(f"")
    print(f"   🩺 Overfitting: {overfitting_analysis['diagnosis']}")
    print(f"")
    print(f"   📊 Top 3 Important Features:")
    for _, row in importance_df.head(3).iterrows():
        print(f"      {int(row['Rank'])}. {row['Feature']} ({row['Percentage']:.2f}%)")
    print(f"")
    print(f"   📁 Output structure:")
    print(f"      {main_output_dir.name}/")
    print(f"      ├── model/")
    print(f"      │   └── xgboost_glucose_model.json")
    print(f"      ├── predictions/")
    print(f"      │   ├── train_predictions.csv")
    print(f"      │   └── test_predictions.csv")
    print(f"      ├── importance/")
    print(f"      │   └── feature_importance.csv")
    print(f"      └── report/")
    print(f"          └── {report_path.name}")
    print(f"")
    print(f"✅ XGBoost pipeline completed successfully!")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()