# ==========================================
# STEP 9: XGBoost MODEL TRAINING + EVALUATION
# Full pipeline with:
# - Bug fix for XGBoost save (_estimator_type)
# - Cross-validation on full dataset
# - Per-glucose-range error analysis (clinical bins)
# - Tuning history CSV (appends across all runs)
# - Visualizations: Predicted vs Actual, Clarke Error Grid, Residuals, Importance
# - Hyperparameter sweep mode
# - Sample weighting for high-glucose samples
# - Learning curve (optional)
# ALL hyperparameters and toggles at top for easy tuning.
# ==========================================

import os
import json
import time
import traceback
import pickle
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

import xgboost as xgb
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import KFold, cross_val_score, learning_curve


# ════════════════════════════════════════════════════════════════════
# 📁 USER SETTINGS — PATHS
# ════════════════════════════════════════════════════════════════════

# Root folder where Step 8 (Sub-task 3&4) outputs live.
# This is where your split-scaled train/test data is stored.
INPUT_ROOT  = Path(r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\10_Robust_Scaled_and_Train_Test_Splitted_Data_Set")

# Root folder where XGBoost results & conclusions will be saved.
# Each run creates a timestamped subfolder here.
OUTPUT_ROOT = Path(r"C:\Users\DELL\Documents\GitHub\fyp\08_Results_and_Visualizations\XGBoost_Results_&_Conclusions")


# ════════════════════════════════════════════════════════════════════
# 🎛️ XGBOOST HYPERPARAMETERS  (⚠️ TUNE THESE FOR MODEL PERFORMANCE)
# ════════════════════════════════════════════════════════════════════

N_ESTIMATORS = 50
# Number of boosting trees built sequentially. More trees = more complex model.
# Range: 50-500. Start with 100 for small datasets; reduce if overfitting.

MAX_DEPTH = 2
# Maximum depth of each tree. Deeper trees capture more patterns but risk overfitting.
# Range: 2-6. Use 2-3 for small datasets, 4-6 for larger ones.

LEARNING_RATE = 0.05
# Step size for each tree's contribution. Lower = slower but more robust learning.
# Range: 0.01-0.3. Lower values usually need more n_estimators to compensate.

SUBSAMPLE = 0.8
# Fraction of training rows randomly sampled for each tree. Adds randomness.
# Range: 0.6-1.0. Lower values reduce overfitting but increase training noise.

COLSAMPLE_BYTREE = 0.7
# Fraction of features randomly sampled for each tree. Forces feature diversity.
# Range: 0.6-1.0. Lower values force trees to find different patterns.

REG_ALPHA = 0.1
# L1 regularization. Pushes unimportant feature weights toward zero (feature selection).
# Range: 0-10. Higher values create sparser models with fewer active features.

REG_LAMBDA = 3
# L2 regularization. Smooths feature weights to prevent any single feature dominating.
# Range: 0-10. Higher values create more stable, generalizable models.

MIN_CHILD_WEIGHT = 3
# Minimum sample weight required in a child node. Higher = more conservative splits.
# Range: 1-10. For small datasets keep low (1-3); for noisy data use higher (5-10).

GAMMA = 0.1
# Minimum loss reduction required to make a split. Acts as a pruning threshold.
# Range: 0-5. Higher values create simpler trees by skipping marginal splits.

RANDOM_STATE = 42
# Fixed seed for reproducibility. Same seed = identical results across runs.
# Any integer works; 42 is conventional in ML for historical reasons.


# ════════════════════════════════════════════════════════════════════
# 🔄 PHASE 2 — TUNING AIDS (always on, cheap to compute)
# ════════════════════════════════════════════════════════════════════

CV_FOLDS = 5
# Number of folds for cross-validation. Splits data into K parts, tests on each.
# Range: 3-10. Use 5 for ~50 samples, 10 for ~200+ samples, 3 for <30 samples.

GLUCOSE_BINS = [0, 70, 100, 125, 180, 999]
# Glucose ranges (mg/dL) for per-range error analysis. Edges in ascending order.
# Default = clinical thresholds: hypo/normal/pre-diabetic/diabetic/hyperglycemic.

GLUCOSE_BIN_LABELS = [
    "Hypoglycemic (<70)",
    "Normal (70-100)",
    "Pre-diabetic (100-125)",
    "Diabetic (125-180)",
    "Hyperglycemic (>180)",
]
# Human-readable labels for each glucose bin. Must match length of GLUCOSE_BINS - 1.
# Used in error tables and plots to identify clinical ranges.


# ════════════════════════════════════════════════════════════════════
# 📊 PHASE 3 — VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════

GENERATE_PLOTS = True
# Master toggle for all visualization plots. Set False to skip plot generation.
# Plots include: Predicted vs Actual, Clarke Grid, Residuals, Feature Importance.

PLOT_DPI = 150
# Resolution of saved plots in dots-per-inch. Higher = clearer but bigger files.
# Range: 100-300. Use 150 for normal use, 300 for publication-quality images.


# ════════════════════════════════════════════════════════════════════
# 🎚️ PHASE 4 — ADVANCED TOGGLES (off by default)
# ════════════════════════════════════════════════════════════════════

# --- Hyperparameter Sweep Mode ---
SWEEP_MODE = False
# When True, runs multiple training iterations sweeping ONE hyperparameter.
# Use this to find the optimal value for a parameter (e.g., max_depth).

SWEEP_PARAM = "max_depth"
# Name of the hyperparameter to sweep. Must match a variable name above.
# Examples: "max_depth", "n_estimators", "learning_rate", "reg_lambda".

SWEEP_VALUES = [2, 3, 4, 5]
# List of values to try for the swept parameter. Runs once per value.
# Example: [50, 100, 150, 200] for n_estimators sweep.


# --- Sample Weighting (for class imbalance / high-glucose accuracy) ---
USE_SAMPLE_WEIGHTS = True
# When True, gives higher importance to high-glucose samples during training.
# Use this if your model underestimates high glucose values.

HIGH_GLUCOSE_THRESHOLD = 130
# Glucose value (mg/dL) above which samples are considered "high".
# Samples with glucose >= this value receive HIGH_GLUCOSE_WEIGHT during training.

HIGH_GLUCOSE_WEIGHT = 2.0
# Importance multiplier for high-glucose samples. 1.0 = equal, 2.0 = twice as important.
# Range: 1.5-3.0. Higher values reduce underestimation but may overshoot normal range.


# --- Learning Curve ---
SHOW_LEARNING_CURVE = True
# When True, generates a learning curve plot showing how performance scales with data.
# Slower to compute (trains model multiple times); useful to diagnose data needs.

LEARNING_CURVE_POINTS = 10
# Number of training-set sizes to evaluate in the learning curve.
# Range: 5-20. More points = smoother curve but slower computation.


# ════════════════════════════════════════════════════════════════════
# ⚙️ FIXED CONFIGURATION
# ════════════════════════════════════════════════════════════════════

TARGET_COLUMN = "Glucose level (mg/dl)"
PREV_STEP_FOLDER_IDENTIFIER     = "Master dataset 24F split scaled"
PREV_STEP_JSON_PIPELINE_STEP_ID = "STEP 8 (Sub-task 3 & 4)"

# Tuning history CSV — single file across all runs
TUNING_HISTORY_CSV = OUTPUT_ROOT / "tuning_history.csv"


# ════════════════════════════════════════════════════════════════════
# AUTO-DETECT LATEST PREVIOUS STEP FOLDER
# ════════════════════════════════════════════════════════════════════

def find_latest_prev_step_folder(root_path):
    """Scans for latest Step 8 (Sub-task 3&4) folder. Informational only."""
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
        folder_info.append({
            "folder_name":   p.name,
            "full_path":     str(p),
            "last_modified": mtime,
            "has_train":     (p / "train").is_dir(),
            "has_test":      (p / "test").is_dir(),
            "has_json":      (p / "json").is_dir(),
        })

    return {
        "found":         True,
        "latest_folder": folder_info[0],
        "all_folders":   folder_info,
        "total_folders": len(folder_info),
    }


def print_prev_step_folder_detection_report(detection_result):
    print(f"\n{'─' * 60}")
    print(f"🔍 STEP 8 (Sub-task 3&4) OUTPUT FOLDER AUTO-DETECTION")
    print(f"{'─' * 60}")
    if not detection_result["found"]:
        print(f"   ⚠️  No '{PREV_STEP_FOLDER_IDENTIFIER}' folders found in:\n       {INPUT_ROOT}")
        return
    total  = detection_result["total_folders"]
    latest = detection_result["latest_folder"]
    print(f"   📁 Found {total} split-scaled folder(s)")
    print(f"   ✅ LATEST: {latest['folder_name']}")
    print(f"      Last modified : {latest['last_modified']}")
    print(f"      Has train/    : {'✅' if latest['has_train'] else '❌'}")
    print(f"      Has test/     : {'✅' if latest['has_test']  else '❌'}")
    print(f"      Has json/     : {'✅' if latest['has_json']  else '❌'}")


def popup_folder_selector(initial_dir):
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    messagebox.showinfo(
        title="Step 9 — XGBoost Training Pipeline",
        message=(
            "Select the Step 8 (Sub-task 3&4) OUTPUT FOLDER.\n\n"
            "Expected name: Master dataset 24F split scaled <timestamp>\n\n"
            "Expected contents:\n"
            "   train/ X_train_scaled.csv, y_train.csv\n"
            "   test/  X_test_scaled.csv, y_test.csv\n"
            "   json/  Master dataset 24F split scaled ....json\n\n"
            "Click OK to open the folder browser."
        ),
    )
    selected_folder = filedialog.askdirectory(
        initialdir=str(initial_dir),
        title="Select Step 8 (Sub-task 3&4) Output Folder",
    )
    root.destroy()
    if not selected_folder:
        raise SystemExit("❌ User cancelled: No folder selected.")
    return Path(selected_folder)


# ════════════════════════════════════════════════════════════════════
# FILE FINDERS
# ════════════════════════════════════════════════════════════════════

def find_train_test_files(folder_path):
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
    folder = Path(folder_path)
    json_dir = folder / "json"
    if not json_dir.exists():
        return None
    json_files = list(json_dir.glob("*.json"))
    if not json_files:
        return None
    return json_files[0]


def load_csv(file_path):
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {file_path}")
    return df


def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_existing_file(file_path):
    p = Path(file_path)
    if p.exists() and p.is_file():
        try:
            size_kb = p.stat().st_size / 1024.0
            return {"exists": True, "path": str(p), "size_kb": size_kb}
        except Exception:
            return {"exists": True, "path": str(p), "size_kb": None}
    return {"exists": False, "path": str(p), "size_kb": None}


def validate_is_prev_step_output(folder_path, prev_step_json_data):
    folder_path = Path(folder_path)
    warnings = []
    errors = []
    if PREV_STEP_FOLDER_IDENTIFIER.lower() not in folder_path.name.lower():
        warnings.append(f"⚠️ Folder name does not match expected pattern.")
    for sub in ["train", "test", "json"]:
        if not (folder_path / sub).exists():
            errors.append(f"'{sub}/' subfolder missing inside: {folder_path}")
    required_csvs = [
        folder_path / "train" / "X_train_scaled.csv",
        folder_path / "train" / "y_train.csv",
        folder_path / "test"  / "X_test_scaled.csv",
        folder_path / "test"  / "y_test.csv",
    ]
    for csv_path in required_csvs:
        if not csv_path.exists():
            errors.append(f"Required CSV file missing: {csv_path}")
    pipeline_step = ""
    if prev_step_json_data is not None:
        pipeline_step = prev_step_json_data.get("pipeline_info", {}).get("pipeline_step", "")
    return {
        "passed":             len(errors) == 0,
        "folder_name":        folder_path.name,
        "json_pipeline_step": pipeline_step,
        "warnings":           warnings,
        "errors":             errors,
    }


def build_pipeline_chain_summary(prev_step_json_data, prev_json_path):
    if prev_step_json_data is None:
        return {"status": "not_found", "message": "Previous step JSON not available."}

    def safe_get(d, *keys, default="Not recorded"):
        current = d
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        return current

    s8b_info  = safe_get(prev_step_json_data, "pipeline_info", default={})
    s8b_split = safe_get(prev_step_json_data, "sub_task_3_train_test_split", "split_details", default={})
    s8b_scale = safe_get(prev_step_json_data, "sub_task_4_robust_scaling", "scaler_parameters", default={})
    upstream_chain = safe_get(prev_step_json_data, "pipeline_chain_summary", default={})

    return {
        "description": "Concise pipeline traceability: Step 6 → 7 → 8(1&2) → 8(3&4) → 9 (XGBoost).",
        "step_6_combine":             safe_get(upstream_chain, "step_6_combine",             default={}),
        "step_7_feature_engineering": safe_get(upstream_chain, "step_7_feature_engineering", default={}),
        "step_8_cleaning_sub_1_2":    safe_get(upstream_chain, "step_8_cleaning_sub_1_2",    default={}),
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


# ════════════════════════════════════════════════════════════════════
# PHASE 1: LOAD DATA
# ════════════════════════════════════════════════════════════════════

def load_all_data(file_paths):
    print(f"\n{'─' * 60}\n📥 PHASE 1: LOADING DATA\n{'─' * 60}")
    X_train    = load_csv(file_paths["X_train_scaled"])
    y_train_df = load_csv(file_paths["y_train"])
    X_test     = load_csv(file_paths["X_test_scaled"])
    y_test_df  = load_csv(file_paths["y_test"])

    y_train = y_train_df[TARGET_COLUMN]
    y_test  = y_test_df[TARGET_COLUMN]
    feature_columns = list(X_train.columns)

    print(f"   ✅ X_train: {X_train.shape[0]} rows × {X_train.shape[1]} columns")
    print(f"   ✅ y_train: {len(y_train)} values")
    print(f"   ✅ X_test:  {X_test.shape[0]} rows × {X_test.shape[1]} columns")
    print(f"   ✅ y_test:  {len(y_test)} values")

    if list(X_train.columns) != list(X_test.columns):
        print(f"   ❌ Feature columns mismatch!")
    else:
        print(f"   ✅ Feature columns match: {len(feature_columns)} features")

    train_nans = int(X_train.isna().sum().sum() + y_train.isna().sum())
    test_nans  = int(X_test.isna().sum().sum() + y_test.isna().sum())
    if train_nans > 0 or test_nans > 0:
        print(f"   ❌ NaN detected! Train: {train_nans}, Test: {test_nans}")
    else:
        print(f"   ✅ No NaN values found")

    print(f"\n   📊 Glucose summary:")
    print(f"      Train — min: {y_train.min():.1f}, max: {y_train.max():.1f}, mean: {y_train.mean():.1f}")
    print(f"      Test  — min: {y_test.min():.1f}, max: {y_test.max():.1f}, mean: {y_test.mean():.1f}")

    return X_train, X_test, y_train, y_test, feature_columns


# ════════════════════════════════════════════════════════════════════
# PHASE 3: TRAIN MODEL (with sample weighting support)
# ════════════════════════════════════════════════════════════════════

def build_xgboost_model(hyperparams):
    """Builds an XGBRegressor with the provided hyperparameters dict."""
    model = xgb.XGBRegressor(
        n_estimators=hyperparams["N_ESTIMATORS"],
        max_depth=hyperparams["MAX_DEPTH"],
        learning_rate=hyperparams["LEARNING_RATE"],
        subsample=hyperparams["SUBSAMPLE"],
        colsample_bytree=hyperparams["COLSAMPLE_BYTREE"],
        reg_alpha=hyperparams["REG_ALPHA"],
        reg_lambda=hyperparams["REG_LAMBDA"],
        min_child_weight=hyperparams["MIN_CHILD_WEIGHT"],
        gamma=hyperparams["GAMMA"],
        random_state=hyperparams["RANDOM_STATE"],
        objective="reg:squarederror",
        tree_method="auto",
        verbosity=0,
    )
    return model


def compute_sample_weights(y_train):
    """Computes per-sample weights based on glucose threshold."""
    if not USE_SAMPLE_WEIGHTS:
        return None
    weights = np.where(y_train >= HIGH_GLUCOSE_THRESHOLD, HIGH_GLUCOSE_WEIGHT, 1.0)
    return weights


def train_xgboost_model(X_train, y_train, hyperparams, verbose=True):
    if verbose:
        print(f"\n{'─' * 60}\n🚀 PHASE 3: TRAINING XGBOOST MODEL\n{'─' * 60}")
        print(f"\n   📊 Hyperparameter Configuration:")
        for k, v in hyperparams.items():
            print(f"      {k.lower():<20} = {v}")

    model = build_xgboost_model(hyperparams)

    # Sample weighting setup
    sample_weights = compute_sample_weights(y_train)
    if verbose and sample_weights is not None:
        n_high = int(np.sum(sample_weights == HIGH_GLUCOSE_WEIGHT))
        n_norm = int(np.sum(sample_weights == 1.0))
        print(f"\n   ⚖️  Sample weighting ENABLED:")
        print(f"      Threshold     : {HIGH_GLUCOSE_THRESHOLD} mg/dL")
        print(f"      Normal weight : 1.0  ({n_norm} samples)")
        print(f"      High weight   : {HIGH_GLUCOSE_WEIGHT}  ({n_high} samples)")
        print(f"      Effective n   : {n_norm + n_high * HIGH_GLUCOSE_WEIGHT:.1f}")

    if verbose:
        print(f"\n   🔧 Training on {X_train.shape[0]} samples, {X_train.shape[1]} features...")

    start_time = time.time()
    if sample_weights is not None:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)
    training_time = time.time() - start_time

    if verbose:
        print(f"   ✅ Model trained in {training_time:.3f} seconds")

    return model, training_time


# ════════════════════════════════════════════════════════════════════
# PHASE 4: PREDICTIONS
# ════════════════════════════════════════════════════════════════════

def make_predictions(model, X_train, X_test):
    print(f"\n{'─' * 60}\n🔮 PHASE 4: MAKING PREDICTIONS\n{'─' * 60}")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    print(f"   ✅ Train predictions: {len(y_pred_train)} values")
    print(f"   ✅ Test predictions:  {len(y_pred_test)} values")
    return y_pred_train, y_pred_test


# ════════════════════════════════════════════════════════════════════
# PHASE 5: EVALUATION METRICS
# ════════════════════════════════════════════════════════════════════

def calculate_metrics(y_actual, y_predicted, set_name):
    mae  = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mape = mean_absolute_percentage_error(y_actual, y_predicted) * 100
    r2 = r2_score(y_actual, y_predicted) if len(y_actual) >= 2 else None
    return {
        "set_name":     set_name,
        "sample_count": int(len(y_actual)),
        "MAE_mg_dL":    round(float(mae), 4),
        "RMSE_mg_dL":   round(float(rmse), 4),
        "R2_score":     round(float(r2), 4) if r2 is not None else "undefined (need ≥2 samples)",
        "MAPE_percent": round(float(mape), 4),
    }


def display_metrics(train_metrics, test_metrics):
    print(f"\n{'─' * 60}\n📊 PHASE 5: EVALUATION METRICS\n{'─' * 60}")
    print(f"\n   {'Metric':<25} {'TRAIN':>15} {'TEST':>15}")
    print(f"   {'─' * 25} {'─' * 15} {'─' * 15}")
    print(f"   {'MAE (mg/dL)':<25} {train_metrics['MAE_mg_dL']:>15.4f} {test_metrics['MAE_mg_dL']:>15.4f}")
    print(f"   {'RMSE (mg/dL)':<25} {train_metrics['RMSE_mg_dL']:>15.4f} {test_metrics['RMSE_mg_dL']:>15.4f}")
    train_r2 = train_metrics['R2_score']; test_r2  = test_metrics['R2_score']
    train_r2_str = f"{train_r2:.4f}" if isinstance(train_r2, float) else str(train_r2)
    test_r2_str  = f"{test_r2:.4f}"  if isinstance(test_r2, float)  else str(test_r2)
    print(f"   {'R² Score':<25} {train_r2_str:>15} {test_r2_str:>15}")
    print(f"   {'MAPE (%)':<25} {train_metrics['MAPE_percent']:>15.4f} {test_metrics['MAPE_percent']:>15.4f}")
    print(f"   {'Samples':<25} {train_metrics['sample_count']:>15} {test_metrics['sample_count']:>15}")


# ════════════════════════════════════════════════════════════════════
# 🆕 PHASE 5b: CROSS-VALIDATION (FULL DATASET)
# ════════════════════════════════════════════════════════════════════

def run_cross_validation(X_full, y_full, hyperparams):
    """Performs K-fold CV on the full dataset. Returns dict with per-fold + aggregate metrics."""
    print(f"\n{'─' * 60}\n🔄 PHASE 5b: CROSS-VALIDATION ({CV_FOLDS}-Fold on Full Dataset)\n{'─' * 60}")
    print(f"   Total samples: {len(y_full)}  →  ~{len(y_full)//CV_FOLDS} test per fold\n")

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=hyperparams["RANDOM_STATE"])

    fold_mae, fold_rmse, fold_r2 = [], [], []
    print(f"   {'Fold':<6} {'MAE':>10} {'RMSE':>10} {'R²':>10}")
    print(f"   {'─' * 6} {'─' * 10} {'─' * 10} {'─' * 10}")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_full), start=1):
        X_tr, X_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_tr, y_val = y_full.iloc[train_idx], y_full.iloc[val_idx]

        model_cv = build_xgboost_model(hyperparams)

        # Apply sample weights inside CV if enabled
        if USE_SAMPLE_WEIGHTS:
            sw = np.where(y_tr >= HIGH_GLUCOSE_THRESHOLD, HIGH_GLUCOSE_WEIGHT, 1.0)
            model_cv.fit(X_tr, y_tr, sample_weight=sw)
        else:
            model_cv.fit(X_tr, y_tr)

        y_val_pred = model_cv.predict(X_val)
        mae  = mean_absolute_error(y_val, y_val_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
        r2   = r2_score(y_val, y_val_pred) if len(y_val) >= 2 else float("nan")

        fold_mae.append(mae); fold_rmse.append(rmse); fold_r2.append(r2)
        print(f"   {fold_idx:<6} {mae:>10.4f} {rmse:>10.4f} {r2:>10.4f}")

    fold_mae = np.array(fold_mae); fold_rmse = np.array(fold_rmse); fold_r2 = np.array(fold_r2)

    print(f"   {'─' * 6} {'─' * 10} {'─' * 10} {'─' * 10}")
    print(f"   {'Mean':<6} {fold_mae.mean():>10.4f} {fold_rmse.mean():>10.4f} {fold_r2.mean():>10.4f}")
    print(f"   {'StdDev':<6} {fold_mae.std():>10.4f} {fold_rmse.std():>10.4f} {fold_r2.std():>10.4f}")

    return {
        "n_folds":       CV_FOLDS,
        "fold_mae":      fold_mae.round(4).tolist(),
        "fold_rmse":     fold_rmse.round(4).tolist(),
        "fold_r2":       fold_r2.round(4).tolist(),
        "mae_mean":      round(float(fold_mae.mean()), 4),
        "mae_std":       round(float(fold_mae.std()), 4),
        "rmse_mean":     round(float(fold_rmse.mean()), 4),
        "rmse_std":      round(float(fold_rmse.std()), 4),
        "r2_mean":       round(float(fold_r2.mean()), 4),
        "r2_std":        round(float(fold_r2.std()), 4),
    }


# ════════════════════════════════════════════════════════════════════
# 🆕 PHASE 5c: PER-GLUCOSE-RANGE ERROR ANALYSIS
# ════════════════════════════════════════════════════════════════════

def per_range_error_analysis(y_actual, y_pred, set_name):
    """Breaks down prediction error by clinical glucose ranges."""
    y_actual = np.asarray(y_actual)
    y_pred = np.asarray(y_pred)
    errors = np.abs(y_actual - y_pred)
    pct_errors = np.where(y_actual > 0, errors / y_actual * 100, 0)

    rows = []
    for i in range(len(GLUCOSE_BINS) - 1):
        lo, hi = GLUCOSE_BINS[i], GLUCOSE_BINS[i + 1]
        label = GLUCOSE_BIN_LABELS[i]
        mask = (y_actual >= lo) & (y_actual < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append({
                "Range": label, "Count": 0, "Avg_Error_mg_dL": None,
                "Avg_Pct_Error": None, "Max_Error_mg_dL": None,
            })
        else:
            rows.append({
                "Range": label,
                "Count": n,
                "Avg_Error_mg_dL": round(float(errors[mask].mean()), 2),
                "Avg_Pct_Error":   round(float(pct_errors[mask].mean()), 2),
                "Max_Error_mg_dL": round(float(errors[mask].max()), 2),
            })

    print(f"\n   📊 {set_name} — Per-Glucose-Range Error:")
    print(f"   {'Range':<26} {'Count':>6} {'AvgErr':>10} {'AvgPct%':>10} {'MaxErr':>10}")
    print(f"   {'─' * 26} {'─' * 6} {'─' * 10} {'─' * 10} {'─' * 10}")
    for r in rows:
        cnt = r["Count"]
        ae  = f"{r['Avg_Error_mg_dL']:.2f}" if r['Avg_Error_mg_dL'] is not None else "  -"
        ap  = f"{r['Avg_Pct_Error']:.2f}"   if r['Avg_Pct_Error']  is not None else "  -"
        mx  = f"{r['Max_Error_mg_dL']:.2f}" if r['Max_Error_mg_dL'] is not None else "  -"
        print(f"   {r['Range']:<26} {cnt:>6} {ae:>10} {ap:>10} {mx:>10}")
    return rows


def display_per_range_analysis(y_train, y_pred_train, y_test, y_pred_test):
    print(f"\n{'─' * 60}\n📊 PHASE 5c: PER-GLUCOSE-RANGE ERROR ANALYSIS\n{'─' * 60}")
    train_breakdown = per_range_error_analysis(y_train, y_pred_train, "TRAIN")
    test_breakdown  = per_range_error_analysis(y_test,  y_pred_test,  "TEST")
    return {"train_per_range": train_breakdown, "test_per_range": test_breakdown}


# ════════════════════════════════════════════════════════════════════
# PHASE 6: OVERFITTING ANALYSIS
# ════════════════════════════════════════════════════════════════════

def analyze_overfitting(train_metrics, test_metrics):
    print(f"\n{'─' * 60}\n🔍 PHASE 6: OVERFITTING ANALYSIS\n{'─' * 60}")
    analysis = {
        "train_mae":  train_metrics["MAE_mg_dL"], "test_mae":   test_metrics["MAE_mg_dL"],
        "train_rmse": train_metrics["RMSE_mg_dL"], "test_rmse":  test_metrics["RMSE_mg_dL"],
        "diagnosis":  "", "details": [],
    }
    mae_ratio  = test_metrics["MAE_mg_dL"] / train_metrics["MAE_mg_dL"]  if train_metrics["MAE_mg_dL"]  > 0 else float('inf')
    rmse_ratio = test_metrics["RMSE_mg_dL"] / train_metrics["RMSE_mg_dL"] if train_metrics["RMSE_mg_dL"] > 0 else float('inf')
    print(f"\n   📊 Error Ratio Analysis:")
    print(f"      MAE ratio  (Test/Train): {mae_ratio:.2f}")
    print(f"      RMSE ratio (Test/Train): {rmse_ratio:.2f}")
    train_r2 = train_metrics["R2_score"]; test_r2  = test_metrics["R2_score"]
    if isinstance(train_r2, float) and isinstance(test_r2, float):
        r2_gap = train_r2 - test_r2
        print(f"      R² gap     (Train-Test): {r2_gap:.4f}")
        analysis["r2_gap"] = round(r2_gap, 4)

    print(f"\n   🩺 Diagnosis:")
    if mae_ratio > 3.0:
        diagnosis = "SEVERE OVERFITTING"; emoji = "🔴"
        details = ["Test error >3x train. Model memorized training data.",
                   "Try: max_depth=2, n_estimators=50, more regularization."]
    elif mae_ratio > 2.0:
        diagnosis = "MODERATE OVERFITTING"; emoji = "🟠"
        details = ["Test error 2-3x train. Partial overfitting.",
                   "Try: reduce max_depth by 1, increase reg_lambda."]
    elif mae_ratio > 1.5:
        diagnosis = "MILD OVERFITTING"; emoji = "🟡"
        details = ["Test error 1.5-2x train. Acceptable but improvable.",
                   "Consider: slight reduction in max_depth or regularization."]
    elif mae_ratio >= 0.8:
        diagnosis = "GOOD GENERALIZATION"; emoji = "🟢"
        details = ["Train and test errors similar. Model generalizes well."]
    else:
        diagnosis = "UNUSUAL — TEST BETTER THAN TRAIN"; emoji = "🔵"
        details = ["Test error lower than train. Likely small/lucky test set."]
    analysis["diagnosis"] = diagnosis; analysis["details"] = details
    analysis["mae_ratio"] = round(mae_ratio, 4); analysis["rmse_ratio"] = round(rmse_ratio, 4)
    print(f"      {emoji} {diagnosis}")
    for d in details: print(f"      {d}")
    if test_metrics["sample_count"] < 5:
        warning = "⚠️ WARNING: Test set <5 samples. Interpret with extreme caution."
        print(f"\n      {warning}")
        analysis["small_dataset_warning"] = warning
    return analysis


# ════════════════════════════════════════════════════════════════════
# PHASE 7: FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════

def analyze_feature_importance(model, feature_columns):
    print(f"\n{'─' * 60}\n📊 PHASE 7: FEATURE IMPORTANCE ANALYSIS\n{'─' * 60}")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_columns, "Importance": importances})
    importance_df = importance_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    total = importances.sum()
    importance_df["Rank"] = range(1, len(importance_df) + 1)
    importance_df["Percentage"] = (importance_df["Importance"] / total * 100).round(2)
    importance_df["Cumulative_Percentage"] = importance_df["Percentage"].cumsum().round(2)

    print(f"\n   {'Rank':<6} {'Feature':<35} {'Importance':>12} {'%':>8} {'Cumul%':>8}")
    print(f"   {'─' * 6} {'─' * 35} {'─' * 12} {'─' * 8} {'─' * 8}")
    for _, row in importance_df.iterrows():
        bar = "█" * int(row["Percentage"] / 2)
        print(f"   {int(row['Rank']):<6} {row['Feature']:<35} {row['Importance']:>12.4f} "
              f"{row['Percentage']:>7.2f}% {row['Cumulative_Percentage']:>7.2f}%  {bar}")

    zero_features = importance_df[importance_df["Importance"] == 0]["Feature"].tolist()
    if zero_features:
        print(f"\n   ⚠️ Features with ZERO importance ({len(zero_features)}):")
        for f in zero_features: print(f"      → {f}")

    importance_log = {
        "total_features": len(feature_columns),
        "zero_importance_features": zero_features,
        "zero_importance_count": len(zero_features),
        "feature_ranking": [
            {"rank": int(row["Rank"]), "feature": row["Feature"],
             "importance": float(row["Importance"]), "percentage": float(row["Percentage"]),
             "cumulative_percentage": float(row["Cumulative_Percentage"])}
            for _, row in importance_df.iterrows()
        ],
    }
    return importance_df, importance_log


# ════════════════════════════════════════════════════════════════════
# PHASE 8: ACTUAL VS PREDICTED TABLE
# ════════════════════════════════════════════════════════════════════

def build_prediction_tables(y_train, y_pred_train, y_test, y_pred_test):
    print(f"\n{'─' * 60}\n📋 PHASE 8: ACTUAL vs PREDICTED ANALYSIS\n{'─' * 60}")
    train_table = pd.DataFrame({
        "Sample": range(1, len(y_train) + 1),
        "Actual_Glucose_mg_dL": y_train.values,
        "Predicted_Glucose_mg_dL": np.round(y_pred_train, 2),
        "Error_mg_dL": np.round(np.abs(y_train.values - y_pred_train), 2),
        "Percent_Error": np.round(np.abs(y_train.values - y_pred_train) / y_train.values * 100, 2),
    })
    test_table = pd.DataFrame({
        "Sample": range(1, len(y_test) + 1),
        "Actual_Glucose_mg_dL": y_test.values,
        "Predicted_Glucose_mg_dL": np.round(y_pred_test, 2),
        "Error_mg_dL": np.round(np.abs(y_test.values - y_pred_test), 2),
        "Percent_Error": np.round(np.abs(y_test.values - y_pred_test) / y_test.values * 100, 2),
    })

    print(f"\n   📊 TRAIN SET (top 10 + summary):")
    print(f"   {'Sample':>8} {'Actual':>12} {'Predicted':>12} {'Error':>10} {'% Error':>10}")
    for _, row in train_table.head(10).iterrows():
        print(f"   {int(row['Sample']):>8} {row['Actual_Glucose_mg_dL']:>12.1f} "
              f"{row['Predicted_Glucose_mg_dL']:>12.2f} {row['Error_mg_dL']:>10.2f} "
              f"{row['Percent_Error']:>9.2f}%")
    if len(train_table) > 10:
        print(f"   ... ({len(train_table) - 10} more samples — see CSV)")

    print(f"\n   📊 TEST SET (all):")
    print(f"   {'Sample':>8} {'Actual':>12} {'Predicted':>12} {'Error':>10} {'% Error':>10}")
    for _, row in test_table.iterrows():
        print(f"   {int(row['Sample']):>8} {row['Actual_Glucose_mg_dL']:>12.1f} "
              f"{row['Predicted_Glucose_mg_dL']:>12.2f} {row['Error_mg_dL']:>10.2f} "
              f"{row['Percent_Error']:>9.2f}%")

    print(f"\n   📊 Error Summary:")
    print(f"      Train — Avg error: {train_table['Error_mg_dL'].mean():.2f} mg/dL, "
          f"Avg %: {train_table['Percent_Error'].mean():.2f}%")
    print(f"      Test  — Avg error: {test_table['Error_mg_dL'].mean():.2f} mg/dL, "
          f"Avg %: {test_table['Percent_Error'].mean():.2f}%")

    tables_log = {
        "train_predictions": train_table.to_dict(orient="records"),
        "test_predictions":  test_table.to_dict(orient="records"),
        "train_avg_error_mg_dL":   round(float(train_table["Error_mg_dL"].mean()), 4),
        "train_avg_percent_error": round(float(train_table["Percent_Error"].mean()), 4),
        "test_avg_error_mg_dL":    round(float(test_table["Error_mg_dL"].mean()), 4),
        "test_avg_percent_error":  round(float(test_table["Percent_Error"].mean()), 4),
    }
    return train_table, test_table, tables_log


# ════════════════════════════════════════════════════════════════════
# 🆕 PHASE 9: VISUALIZATIONS
# ════════════════════════════════════════════════════════════════════

def plot_predicted_vs_actual(y_train, y_pred_train, y_test, y_pred_test, save_path):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_train, y_pred_train, alpha=0.6, color="steelblue", label=f"Train (n={len(y_train)})", s=60)
    ax.scatter(y_test, y_pred_test, alpha=0.8, color="crimson", label=f"Test (n={len(y_test)})", s=80, marker="^")
    all_vals = np.concatenate([y_train, y_test, y_pred_train, y_pred_test])
    lo, hi = all_vals.min() - 10, all_vals.max() + 10
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="Perfect prediction (y=x)")
    ax.fill_between([lo, hi], [lo - 15, hi - 15], [lo + 15, hi + 15], alpha=0.15, color="green", label="±15 mg/dL")
    ax.set_xlabel("Actual Glucose (mg/dL)"); ax.set_ylabel("Predicted Glucose (mg/dL)")
    ax.set_title("Predicted vs Actual Glucose Values")
    ax.legend(loc="upper left"); ax.grid(True, alpha=0.3); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    plt.tight_layout(); plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()


def plot_clarke_error_grid(y_test, y_pred_test, save_path):
    """Clarke Error Grid — clinical accuracy standard for glucose meters."""
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.scatter(y_test, y_pred_test, color="crimson", s=80, zorder=5, label=f"Predictions (n={len(y_test)})")
    ax.plot([0, 400], [0, 400], "k-", alpha=0.3)
    # Zone A boundaries (±20%)
    ax.plot([0, 400], [0, 400 * 1.2], "g--", alpha=0.4)
    ax.plot([0, 400], [0, 400 * 0.8], "g--", alpha=0.4)
    # Annotate zones
    ax.text(20, 380, "Zone A: Clinically Accurate (±20%)", color="green", fontsize=10, weight="bold")
    ax.text(20, 360, "Zone B: Benign Error",   color="orange", fontsize=10, weight="bold")
    ax.text(20, 340, "Zone D/E: Clinically Significant Error", color="red", fontsize=10, weight="bold")
    # Count zone-A
    in_zone_a = np.sum(np.abs(y_pred_test - y_test) <= 0.2 * y_test)
    pct_a = in_zone_a / len(y_test) * 100
    ax.text(200, 50, f"Zone A: {in_zone_a}/{len(y_test)} ({pct_a:.1f}%)\n"
                     f"(Clinical target: ≥95%)", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", edgecolor="orange"))
    ax.set_xlabel("Reference Glucose (mg/dL)"); ax.set_ylabel("Predicted Glucose (mg/dL)")
    ax.set_title("Clarke Error Grid — Clinical Accuracy Assessment")
    ax.set_xlim(0, max(400, y_test.max() + 50)); ax.set_ylim(0, max(400, y_pred_test.max() + 50))
    ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
    return {"zone_a_count": int(in_zone_a), "zone_a_percent": round(float(pct_a), 2)}


def plot_residuals(y_train, y_pred_train, y_test, y_pred_test, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    res_train = y_train.values - y_pred_train
    res_test  = y_test.values - y_pred_test
    axes[0].scatter(y_pred_train, res_train, alpha=0.6, color="steelblue", s=60, label="Train")
    axes[0].scatter(y_pred_test, res_test, alpha=0.8, color="crimson", s=80, marker="^", label="Test")
    axes[0].axhline(0, color="k", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Predicted Glucose (mg/dL)"); axes[0].set_ylabel("Residual (Actual - Predicted)")
    axes[0].set_title("Residual Plot"); axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].hist(res_train, bins=15, alpha=0.5, color="steelblue", label="Train", edgecolor="black")
    axes[1].hist(res_test, bins=10, alpha=0.7, color="crimson", label="Test", edgecolor="black")
    axes[1].axvline(0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Residual (mg/dL)"); axes[1].set_ylabel("Count")
    axes[1].set_title("Residual Distribution"); axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()


def plot_feature_importance_bar(importance_df, save_path):
    n_show = min(20, len(importance_df))
    top_df = importance_df.head(n_show).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(6, n_show * 0.3)))
    bars = ax.barh(top_df["Feature"], top_df["Percentage"], color="steelblue", edgecolor="black")
    for bar, pct in zip(bars, top_df["Percentage"]):
        ax.text(pct + 0.1, bar.get_y() + bar.get_height()/2, f"{pct:.2f}%",
                va="center", fontsize=9)
    ax.set_xlabel("Importance (%)"); ax.set_title(f"Top {n_show} Feature Importances")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout(); plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()


def plot_learning_curve_chart(X_full, y_full, hyperparams, save_path):
    """Generates learning curve — train/CV error as data grows."""
    print(f"\n   📈 Computing learning curve...")
    model_lc = build_xgboost_model(hyperparams)
    train_sizes = np.linspace(0.1, 1.0, LEARNING_CURVE_POINTS)
    try:
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model_lc, X_full, y_full, train_sizes=train_sizes,
            cv=min(CV_FOLDS, max(2, len(y_full)//5)),
            scoring="neg_mean_absolute_error", random_state=hyperparams["RANDOM_STATE"],
            n_jobs=1
        )
        train_mae = -train_scores.mean(axis=1); val_mae = -val_scores.mean(axis=1)
        fig, ax = plt.subplots(figsize=(9, 6))
        ax.plot(train_sizes_abs, train_mae, "o-", color="steelblue", label="Training MAE")
        ax.plot(train_sizes_abs, val_mae, "s-", color="crimson", label="CV MAE")
        ax.fill_between(train_sizes_abs, train_mae - (-train_scores.std(axis=1)),
                        train_mae + (-train_scores.std(axis=1)), alpha=0.15, color="steelblue")
        ax.fill_between(train_sizes_abs, val_mae - (-val_scores.std(axis=1)),
                        val_mae + (-val_scores.std(axis=1)), alpha=0.15, color="crimson")
        ax.set_xlabel("Training Samples"); ax.set_ylabel("MAE (mg/dL)")
        ax.set_title("Learning Curve — Train vs CV Error"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(save_path, dpi=PLOT_DPI, bbox_inches="tight"); plt.close()
        print(f"   ✅ Learning curve saved: {save_path.name}")
    except Exception as e:
        print(f"   ⚠️ Learning curve failed: {e}")


def generate_all_plots(y_train, y_pred_train, y_test, y_pred_test,
                       importance_df, X_full, y_full, hyperparams, plots_dir):
    """Generates all visualization plots, returns plot info dict."""
    print(f"\n{'─' * 60}\n🎨 PHASE 9: VISUALIZATIONS\n{'─' * 60}")
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_info = {}

    p1 = plots_dir / "01_predicted_vs_actual.png"
    plot_predicted_vs_actual(y_train, y_pred_train, y_test, y_pred_test, p1)
    print(f"   ✅ Saved: {p1.name}"); plot_info["predicted_vs_actual"] = str(p1)

    p2 = plots_dir / "02_clarke_error_grid.png"
    clarke_info = plot_clarke_error_grid(y_test, y_pred_test, p2)
    print(f"   ✅ Saved: {p2.name}  (Zone A: {clarke_info['zone_a_percent']}%)")
    plot_info["clarke_error_grid"] = str(p2); plot_info["clarke_zone_a"] = clarke_info

    p3 = plots_dir / "03_residual_plot.png"
    plot_residuals(y_train, y_pred_train, y_test, y_pred_test, p3)
    print(f"   ✅ Saved: {p3.name}"); plot_info["residual_plot"] = str(p3)

    p4 = plots_dir / "04_feature_importance.png"
    plot_feature_importance_bar(importance_df, p4)
    print(f"   ✅ Saved: {p4.name}"); plot_info["feature_importance"] = str(p4)

    if SHOW_LEARNING_CURVE:
        p5 = plots_dir / "05_learning_curve.png"
        plot_learning_curve_chart(X_full, y_full, hyperparams, p5)
        plot_info["learning_curve"] = str(p5)

    return plot_info


# ════════════════════════════════════════════════════════════════════
# 🆕 PHASE 10: TUNING HISTORY CSV (appends each run)
# ════════════════════════════════════════════════════════════════════

def append_tuning_history(timestamp_str, hyperparams, train_metrics, test_metrics,
                          cv_results, overfitting_analysis, importance_df,
                          clarke_info=None, run_label="single"):
    """Appends a row to OUTPUT_ROOT/tuning_history.csv tracking all runs."""
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Top 10 features
    top10 = importance_df.head(10)
    top_features_str = "; ".join([
        f"{row['Feature']}({row['Percentage']:.1f}%)" for _, row in top10.iterrows()
    ])

    row = {
        "timestamp": timestamp_str,
        "run_label": run_label,
        # Hyperparameters
        "n_estimators": hyperparams["N_ESTIMATORS"],
        "max_depth": hyperparams["MAX_DEPTH"],
        "learning_rate": hyperparams["LEARNING_RATE"],
        "subsample": hyperparams["SUBSAMPLE"],
        "colsample_bytree": hyperparams["COLSAMPLE_BYTREE"],
        "reg_alpha": hyperparams["REG_ALPHA"],
        "reg_lambda": hyperparams["REG_LAMBDA"],
        "min_child_weight": hyperparams["MIN_CHILD_WEIGHT"],
        "gamma": hyperparams["GAMMA"],
        "random_state": hyperparams["RANDOM_STATE"],
        # Sample weighting
        "use_sample_weights": USE_SAMPLE_WEIGHTS,
        "high_glucose_threshold": HIGH_GLUCOSE_THRESHOLD if USE_SAMPLE_WEIGHTS else None,
        "high_glucose_weight": HIGH_GLUCOSE_WEIGHT if USE_SAMPLE_WEIGHTS else None,
        # Train metrics
        "train_mae": train_metrics["MAE_mg_dL"],
        "train_rmse": train_metrics["RMSE_mg_dL"],
        "train_r2": train_metrics["R2_score"],
        "train_mape": train_metrics["MAPE_percent"],
        "train_samples": train_metrics["sample_count"],
        # Test metrics
        "test_mae": test_metrics["MAE_mg_dL"],
        "test_rmse": test_metrics["RMSE_mg_dL"],
        "test_r2": test_metrics["R2_score"],
        "test_mape": test_metrics["MAPE_percent"],
        "test_samples": test_metrics["sample_count"],
        # CV
        "cv_folds": cv_results["n_folds"],
        "cv_mae_mean": cv_results["mae_mean"],
        "cv_mae_std": cv_results["mae_std"],
        "cv_rmse_mean": cv_results["rmse_mean"],
        "cv_r2_mean": cv_results["r2_mean"],
        # Overfitting
        "mae_ratio_test_train": overfitting_analysis["mae_ratio"],
        "diagnosis": overfitting_analysis["diagnosis"],
        # Clinical
        "clarke_zone_a_pct": clarke_info["zone_a_percent"] if clarke_info else None,
        # Top features
        "top_10_features": top_features_str,
        # User notes (manually editable later)
        "notes": "",
    }

    df_row = pd.DataFrame([row])

    if TUNING_HISTORY_CSV.exists():
        df_existing = pd.read_csv(TUNING_HISTORY_CSV)
        # Align columns (in case schema evolved)
        for col in df_row.columns:
            if col not in df_existing.columns:
                df_existing[col] = None
        for col in df_existing.columns:
            if col not in df_row.columns:
                df_row[col] = None
        df_combined = pd.concat([df_existing, df_row[df_existing.columns]], ignore_index=True)
    else:
        df_combined = df_row

    df_combined.to_csv(TUNING_HISTORY_CSV, index=False)
    print(f"\n   📝 Tuning history updated: {TUNING_HISTORY_CSV.name} (total runs: {len(df_combined)})")


# ════════════════════════════════════════════════════════════════════
# 🆕 SAVE MODEL — WITH BUG FIX (Option A + B combined)
# ════════════════════════════════════════════════════════════════════

def save_model_safely(model, model_path_json, model_path_pkl=None):
    """Saves XGBoost model with the _estimator_type fix + Booster API backup."""
    try:
        model._estimator_type = "regressor"  # Fix for XGBoost ≥ 2.1 bug
        model.save_model(str(model_path_json))
        print(f"   💾 Model saved (sklearn API): {model_path_json.name}")
    except Exception as e:
        print(f"   ⚠️ sklearn save failed: {e}")
        try:
            model.get_booster().save_model(str(model_path_json))
            print(f"   💾 Model saved via Booster API: {model_path_json.name}")
        except Exception as e2:
            print(f"   ❌ Booster save also failed: {e2}")

    # Always save pickle as backup (preserves full sklearn wrapper)
    if model_path_pkl is None:
        model_path_pkl = model_path_json.with_suffix(".pkl")
    try:
        with open(model_path_pkl, "wb") as f:
            pickle.dump(model, f)
        print(f"   💾 Pickle backup saved: {model_path_pkl.name}")
    except Exception as e:
        print(f"   ⚠️ Pickle save failed: {e}")


# ════════════════════════════════════════════════════════════════════
# BUILD REPORT
# ════════════════════════════════════════════════════════════════════

def build_xgboost_report_data(input_folder_path, output_folder_path, output_file_paths,
                              X_train, X_test, y_train, y_test, feature_columns,
                              training_time, train_metrics, test_metrics, cv_results,
                              per_range_results, overfitting_analysis, importance_log,
                              tables_log, plot_info, timestamp_str, pipeline_chain_summary,
                              hyperparams):
    hyperparameter_explanations = {
        "n_estimators":     "Number of boosting trees.",
        "max_depth":        "Maximum depth per tree.",
        "learning_rate":    "Step size per boosting round.",
        "subsample":        "Fraction of rows sampled per tree.",
        "colsample_bytree": "Fraction of features sampled per tree.",
        "reg_alpha":        "L1 regularization.",
        "reg_lambda":       "L2 regularization.",
        "min_child_weight": "Minimum weight in child node.",
        "gamma":            "Minimum loss reduction for split.",
        "random_state":     "Random seed for reproducibility.",
    }
    return {
        "pipeline_info": {
            "pipeline_name": "XGBoost Glucose Prediction Model",
            "pipeline_step": "STEP 9",
            "execution_timestamp": timestamp_str,
            "execution_date_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "previous_step": "STEP 8 (Sub-task 3 & 4) — Train/Test Split + Scaling",
        },
        "pipeline_chain_summary": pipeline_chain_summary,
        "file_paths": {
            "input_folder": str(input_folder_path),
            "output_folder": str(output_folder_path),
            "output_files": output_file_paths,
        },
        "data_summary": {
            "train_samples": int(X_train.shape[0]),
            "test_samples":  int(X_test.shape[0]),
            "total_samples": int(X_train.shape[0] + X_test.shape[0]),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
        },
        "model_configuration": {
            "model_type": "XGBRegressor",
            "library": "xgboost",
            "hyperparameters": hyperparams,
            "hyperparameter_explanations": hyperparameter_explanations,
            "training_time_seconds": round(training_time, 4),
            "sample_weighting": {
                "enabled": USE_SAMPLE_WEIGHTS,
                "threshold_mg_dL": HIGH_GLUCOSE_THRESHOLD if USE_SAMPLE_WEIGHTS else None,
                "weight": HIGH_GLUCOSE_WEIGHT if USE_SAMPLE_WEIGHTS else None,
            }
        },
        "evaluation_metrics": {
            "train_metrics": train_metrics,
            "test_metrics":  test_metrics,
            "cross_validation": cv_results,
            "per_glucose_range": per_range_results,
        },
        "overfitting_analysis": overfitting_analysis,
        "feature_importance":   importance_log,
        "predictions":          tables_log,
        "visualizations":       plot_info,
    }


def save_all_outputs(model, train_table, test_table, importance_df, xgboost_report_data,
                     output_root, timestamp_str):
    main_folder_name = f"XGBoost results & Conclusions {timestamp_str}"
    main_dir = output_root / main_folder_name
    main_dir.mkdir(parents=True, exist_ok=True)

    model_dir = main_dir / "model"; pred_dir = main_dir / "predictions"
    importance_dir = main_dir / "importance"; report_dir = main_dir / "report"
    for d in [model_dir, pred_dir, importance_dir, report_dir]: d.mkdir(parents=True, exist_ok=True)

    # ── Save model (with bug fix) ──
    model_path = model_dir / "xgboost_glucose_model.json"
    save_model_safely(model, model_path)

    # ── Predictions ──
    train_pred_path = pred_dir / "train_predictions.csv"
    test_pred_path  = pred_dir / "test_predictions.csv"
    train_table.to_csv(train_pred_path, index=False)
    test_table.to_csv(test_pred_path, index=False)
    print(f"   💾 Train predictions: {train_pred_path.name}")
    print(f"   💾 Test predictions:  {test_pred_path.name}")

    # ── Feature importance ──
    importance_path = importance_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    print(f"   💾 Feature importance: {importance_path.name}")

    # ── Report JSON ──
    report_path = report_dir / f"XGBoost_full_report_{timestamp_str}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(xgboost_report_data, f, indent=4, default=str)
    print(f"   💾 Full report: {report_path.name}")

    output_file_paths = {
        "main_folder": str(main_dir),
        "model_json":  str(model_path),
        "model_pickle": str(model_path.with_suffix(".pkl")),
        "train_predictions":  str(train_pred_path),
        "test_predictions":   str(test_pred_path),
        "feature_importance": str(importance_path),
        "full_report":        str(report_path),
    }
    return main_dir, output_file_paths


# ════════════════════════════════════════════════════════════════════
# 🆕 HYPERPARAMETER SWEEP MODE
# ════════════════════════════════════════════════════════════════════

def run_sweep_mode(X_train, X_test, y_train, y_test, feature_columns,
                   base_hyperparams, timestamp_str):
    """Sweep mode: quick training per value, full analysis only on winner."""
    print(f"\n{'═' * 60}")
    print(f"🔁 SWEEP MODE — Parameter: {SWEEP_PARAM}")
    print(f"   Values: {SWEEP_VALUES}")
    print(f"{'═' * 60}")

    if SWEEP_PARAM not in base_hyperparams:
        raise ValueError(f"SWEEP_PARAM '{SWEEP_PARAM}' not found in hyperparameters.")

    sweep_main_folder = OUTPUT_ROOT / f"XGBoost SWEEP {SWEEP_PARAM} {timestamp_str}"
    sweep_main_folder.mkdir(parents=True, exist_ok=True)

    sweep_summary = []
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)

    # Quick sweep — train + test metrics + CV mean only
    for idx, value in enumerate(SWEEP_VALUES, start=1):
        print(f"\n   🔧 [{idx}/{len(SWEEP_VALUES)}] {SWEEP_PARAM} = {value}")
        run_hyperparams = base_hyperparams.copy()
        run_hyperparams[SWEEP_PARAM] = value

        model_run, _ = train_xgboost_model(X_train, y_train, run_hyperparams, verbose=False)
        y_pred_tr = model_run.predict(X_train)
        y_pred_te = model_run.predict(X_test)

        tr_mae = mean_absolute_error(y_train, y_pred_tr)
        te_mae = mean_absolute_error(y_test, y_pred_te)

        # Quick CV (single run, no full analysis)
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=base_hyperparams["RANDOM_STATE"])
        cv_maes = []
        for tr_idx, val_idx in kf.split(X_full):
            m = build_xgboost_model(run_hyperparams)
            m.fit(X_full.iloc[tr_idx], y_full.iloc[tr_idx])
            cv_maes.append(mean_absolute_error(y_full.iloc[val_idx], m.predict(X_full.iloc[val_idx])))
        cv_mae_mean = float(np.mean(cv_maes)); cv_mae_std = float(np.std(cv_maes))

        sweep_summary.append({
            "value": value,
            "train_mae": round(tr_mae, 4),
            "test_mae":  round(te_mae, 4),
            "cv_mae_mean": round(cv_mae_mean, 4),
            "cv_mae_std":  round(cv_mae_std, 4),
            "mae_ratio":  round(te_mae / tr_mae if tr_mae > 0 else float("inf"), 4),
        })
        print(f"      Train MAE: {tr_mae:.4f} | Test MAE: {te_mae:.4f} | CV MAE: {cv_mae_mean:.4f} ± {cv_mae_std:.4f}")

    # Display sweep summary
    print(f"\n{'─' * 60}\n📊 SWEEP RESULTS\n{'─' * 60}")
    df_sweep = pd.DataFrame(sweep_summary)
    print(f"\n   {SWEEP_PARAM:<10} {'Train MAE':>12} {'Test MAE':>12} {'CV MAE':>12} {'CV Std':>10} {'Ratio':>8}")
    print(f"   {'─' * 10} {'─' * 12} {'─' * 12} {'─' * 12} {'─' * 10} {'─' * 8}")
    for r in sweep_summary:
        print(f"   {str(r['value']):<10} {r['train_mae']:>12.4f} {r['test_mae']:>12.4f} "
              f"{r['cv_mae_mean']:>12.4f} {r['cv_mae_std']:>10.4f} {r['mae_ratio']:>8.2f}")

    # Pick winner = lowest CV MAE
    winner = min(sweep_summary, key=lambda r: r["cv_mae_mean"])
    print(f"\n   🏆 WINNER: {SWEEP_PARAM} = {winner['value']}  (CV MAE = {winner['cv_mae_mean']:.4f})")

    # Save sweep summary CSV
    sweep_csv = sweep_main_folder / f"sweep_summary_{SWEEP_PARAM}.csv"
    df_sweep.to_csv(sweep_csv, index=False)
    print(f"   💾 Sweep summary saved: {sweep_csv}")

    # Run full pipeline on winner
    print(f"\n{'═' * 60}\n🏆 FULL ANALYSIS ON WINNER ({SWEEP_PARAM}={winner['value']})\n{'═' * 60}")
    winner_hyperparams = base_hyperparams.copy()
    winner_hyperparams[SWEEP_PARAM] = winner["value"]

    return sweep_main_folder, winner_hyperparams, df_sweep


# ════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("🎯 STEP 9: XGBoost MODEL TRAINING + EVALUATION")
    print("   Model: XGBRegressor (Gradient Boosted Trees)")
    print("   Target: Glucose Level Prediction (mg/dL)")
    print("=" * 70)

    if not INPUT_ROOT.exists():
        raise SystemExit(f"❌ Input folder does not exist: {INPUT_ROOT}")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    timestamp_str = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    print(f"\n🔍 Scanning for latest split-scaled output folder...")
    prev_detection_result = find_latest_prev_step_folder(INPUT_ROOT)
    print_prev_step_folder_detection_report(prev_detection_result)

    print(f"\n📂 Opening folder selector at: {INPUT_ROOT}")
    input_folder = popup_folder_selector(INPUT_ROOT)
    print(f"📁 Selected folder : {input_folder.name}")

    print(f"\n{'─' * 60}\n🔍 AUTO-DETECTING TRAIN/TEST FILES\n{'─' * 60}")
    file_paths = find_train_test_files(input_folder)
    for label, fpath in file_paths.items():
        print(f"   📄 {label}: {fpath.name}")

    prev_json_path = find_json_in_folder(input_folder)
    prev_step_json_data = None
    if prev_json_path is not None:
        try:
            prev_step_json_data = load_json(prev_json_path)
            print(f"   ✅ JSON loaded: {prev_json_path.name}")
        except Exception as e:
            print(f"   ⚠️  Could not parse JSON: {e}")

    prev_validation = validate_is_prev_step_output(input_folder, prev_step_json_data)
    if not prev_validation["passed"]:
        for error in prev_validation["errors"]:
            print(f"❌ {error}")
        raise SystemExit("❌ Validation failed.")
    print(f"   ✅ Previous-step validation passed.")

    pipeline_chain_summary = build_pipeline_chain_summary(prev_step_json_data, prev_json_path)

    # PHASE 1
    X_train, X_test, y_train, y_test, feature_columns = load_all_data(file_paths)

    # Build base hyperparameters dict
    base_hyperparams = {
        "N_ESTIMATORS": N_ESTIMATORS, "MAX_DEPTH": MAX_DEPTH,
        "LEARNING_RATE": LEARNING_RATE, "SUBSAMPLE": SUBSAMPLE,
        "COLSAMPLE_BYTREE": COLSAMPLE_BYTREE, "REG_ALPHA": REG_ALPHA,
        "REG_LAMBDA": REG_LAMBDA, "MIN_CHILD_WEIGHT": MIN_CHILD_WEIGHT,
        "GAMMA": GAMMA, "RANDOM_STATE": RANDOM_STATE,
    }

    # ── SWEEP MODE (if enabled) ──
    sweep_main_folder = None
    sweep_df = None
    if SWEEP_MODE:
        sweep_main_folder, hyperparams, sweep_df = run_sweep_mode(
            X_train, X_test, y_train, y_test, feature_columns,
            base_hyperparams, timestamp_str
        )
    else:
        hyperparams = base_hyperparams

    # PHASE 3
    model, training_time = train_xgboost_model(X_train, y_train, hyperparams)

    # PHASE 4
    y_pred_train, y_pred_test = make_predictions(model, X_train, X_test)

    # PHASE 5
    train_metrics = calculate_metrics(y_train, y_pred_train, "TRAIN")
    test_metrics  = calculate_metrics(y_test,  y_pred_test,  "TEST")
    display_metrics(train_metrics, test_metrics)

    # PHASE 5b — CV on FULL dataset
    X_full = pd.concat([X_train, X_test], ignore_index=True)
    y_full = pd.concat([y_train, y_test], ignore_index=True)
    cv_results = run_cross_validation(X_full, y_full, hyperparams)

    # PHASE 5c — Per-range error
    per_range_results = display_per_range_analysis(y_train, y_pred_train, y_test, y_pred_test)

    # PHASE 6
    overfitting_analysis = analyze_overfitting(train_metrics, test_metrics)

    # PHASE 7
    importance_df, importance_log = analyze_feature_importance(model, feature_columns)

    # PHASE 8
    train_table, test_table, tables_log = build_prediction_tables(
        y_train, y_pred_train, y_test, y_pred_test
    )

    # Determine output folder (sweep vs normal)
    if SWEEP_MODE and sweep_main_folder is not None:
        winner_subfolder = sweep_main_folder / f"WINNER_{SWEEP_PARAM}_{hyperparams[SWEEP_PARAM]}"
        output_main_dir = winner_subfolder
        run_label = f"sweep_{SWEEP_PARAM}={hyperparams[SWEEP_PARAM]}"
    else:
        run_label = "single"

    # PHASE 9 — Visualizations
    plot_info = {}
    if GENERATE_PLOTS:
        if SWEEP_MODE and sweep_main_folder is not None:
            plots_dir = sweep_main_folder / f"WINNER_{SWEEP_PARAM}_{hyperparams[SWEEP_PARAM]}" / "plots"
        else:
            plots_dir = OUTPUT_ROOT / f"XGBoost results & Conclusions {timestamp_str}" / "plots"
        plot_info = generate_all_plots(
            y_train, y_pred_train, y_test, y_pred_test,
            importance_df, X_full, y_full, hyperparams, plots_dir
        )

    # PHASE — Save outputs
    print(f"\n{'─' * 60}\n💾 SAVING MODEL + RESULTS\n{'─' * 60}")
    if SWEEP_MODE and sweep_main_folder is not None:
        # Save inside sweep folder
        winner_label = f"WINNER_{SWEEP_PARAM}_{hyperparams[SWEEP_PARAM]}"
        main_output_dir = sweep_main_folder / winner_label
        main_output_dir.mkdir(parents=True, exist_ok=True)
        model_dir = main_output_dir / "model"; pred_dir = main_output_dir / "predictions"
        importance_dir = main_output_dir / "importance"; report_dir = main_output_dir / "report"
        for d in [model_dir, pred_dir, importance_dir, report_dir]: d.mkdir(parents=True, exist_ok=True)

        model_path = model_dir / "xgboost_glucose_model.json"
        save_model_safely(model, model_path)
        train_table.to_csv(pred_dir / "train_predictions.csv", index=False)
        test_table.to_csv(pred_dir / "test_predictions.csv", index=False)
        importance_df.to_csv(importance_dir / "feature_importance.csv", index=False)
        report_path = report_dir / f"XGBoost_full_report_{timestamp_str}.json"

        output_file_paths = {
            "main_folder": str(main_output_dir),
            "sweep_folder": str(sweep_main_folder),
            "model_json": str(model_path),
            "model_pickle": str(model_path.with_suffix(".pkl")),
            "train_predictions": str(pred_dir / "train_predictions.csv"),
            "test_predictions": str(pred_dir / "test_predictions.csv"),
            "feature_importance": str(importance_dir / "feature_importance.csv"),
            "full_report": str(report_path),
        }
    else:
        xgboost_report_data_temp = build_xgboost_report_data(
            input_folder, OUTPUT_ROOT, {}, X_train, X_test, y_train, y_test,
            feature_columns, training_time, train_metrics, test_metrics,
            cv_results, per_range_results, overfitting_analysis, importance_log,
            tables_log, plot_info, timestamp_str, pipeline_chain_summary, hyperparams
        )
        main_output_dir, output_file_paths = save_all_outputs(
            model, train_table, test_table, importance_df,
            xgboost_report_data_temp, OUTPUT_ROOT, timestamp_str
        )
        report_path = Path(output_file_paths["full_report"])

    # Build final report
    xgboost_report_data = build_xgboost_report_data(
        input_folder, main_output_dir, output_file_paths,
        X_train, X_test, y_train, y_test, feature_columns,
        training_time, train_metrics, test_metrics, cv_results,
        per_range_results, overfitting_analysis, importance_log,
        tables_log, plot_info, timestamp_str, pipeline_chain_summary, hyperparams
    )
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(xgboost_report_data, f, indent=4, default=str)

    # PHASE 10 — Append to tuning history
    clarke_info = plot_info.get("clarke_zone_a") if plot_info else None
    append_tuning_history(
        timestamp_str, hyperparams, train_metrics, test_metrics,
        cv_results, overfitting_analysis, importance_df, clarke_info, run_label
    )

    # ── Final Summary ──
    print(f"\n{'=' * 70}\n📌 XGBOOST PIPELINE — FINAL SUMMARY\n{'=' * 70}")
    print(f"\n   📥 Input: {input_folder.name}")
    print(f"      Train: {X_train.shape[0]} × {X_train.shape[1]} | Test: {X_test.shape[0]} × {X_test.shape[1]}")
    print(f"\n   ⚙️  Hyperparameters used:")
    for k, v in hyperparams.items(): print(f"      {k.lower():<20} = {v}")
    if USE_SAMPLE_WEIGHTS:
        print(f"      sample_weighting    = ON (threshold={HIGH_GLUCOSE_THRESHOLD}, weight={HIGH_GLUCOSE_WEIGHT})")
    print(f"\n   📊 Performance:")
    print(f"      {'':>20} {'TRAIN':>12} {'TEST':>12} {'CV':>15}")
    print(f"      {'MAE (mg/dL)':>20} {train_metrics['MAE_mg_dL']:>12.4f} {test_metrics['MAE_mg_dL']:>12.4f} "
          f"{cv_results['mae_mean']:>8.4f} ± {cv_results['mae_std']:.2f}")
    print(f"      {'RMSE (mg/dL)':>20} {train_metrics['RMSE_mg_dL']:>12.4f} {test_metrics['RMSE_mg_dL']:>12.4f} "
          f"{cv_results['rmse_mean']:>8.4f} ± {cv_results['rmse_std']:.2f}")
    print(f"\n   🩺 Overfitting: {overfitting_analysis['diagnosis']}")
    if plot_info.get("clarke_zone_a"):
        print(f"   🏥 Clarke Zone A: {plot_info['clarke_zone_a']['zone_a_percent']:.1f}% (target: ≥95%)")
    print(f"\n   📊 Top 3 Features:")
    for _, row in importance_df.head(3).iterrows():
        print(f"      {int(row['Rank'])}. {row['Feature']} ({row['Percentage']:.2f}%)")
    print(f"\n   📁 Output folder: {main_output_dir.name}")
    print(f"   📝 Tuning history: {TUNING_HISTORY_CSV.name}")
    print(f"\n✅ XGBoost pipeline completed successfully!\n{'=' * 70}\n")


if __name__ == "__main__":
    try:
        main()
    except SystemExit as e:
        print(f"\n{e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        traceback.print_exc()