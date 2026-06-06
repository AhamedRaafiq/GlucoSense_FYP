# ==========================================
# STEP 9: XGBoost MODEL TRAINING + EVALUATION
# Input: Scaled train/test splits from Step 8
# Output: Trained model + predictions + full report
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
# XGBOOST HYPERPARAMETERS
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


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def popup_folder_selector(initial_dir):
    """
    Opens a folder dialog for user to select the Step 8 output folder.
    Returns: Path to selected folder.
    """
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    messagebox.showinfo(
        title="XGBoost Training Pipeline — Folder Selection",
        message=(
            "Select the Step 8 (Sub-task 3&4) OUTPUT FOLDER.\n\n"
            "This folder should contain:\n"
            "  • train/  (X_train_scaled.csv, y_train.csv)\n"
            "  • test/   (X_test_scaled.csv, y_test.csv)\n\n"
            "Select the MAIN FOLDER itself.\n\n"
            "Click OK to open the folder browser."
        ),
    )

    selected_folder = filedialog.askdirectory(
        initialdir=str(initial_dir),
        title="Select Step 8 Output FOLDER (contains train/ and test/ subfolders)",
    )

    root.destroy()

    if not selected_folder:
        raise SystemExit("❌ User cancelled: No folder selected. Execution terminated.")

    return Path(selected_folder)


def find_train_test_files(folder_path):
    """
    Auto-detect the 4 required files inside train/ and test/ subfolders.
    Returns: dict with paths to all 4 files.
    """
    folder = Path(folder_path)
    train_dir = folder / "train"
    test_dir = folder / "test"

    if not train_dir.exists():
        raise FileNotFoundError(f"'train/' subfolder not found inside: {folder}")
    if not test_dir.exists():
        raise FileNotFoundError(f"'test/' subfolder not found inside: {folder}")

    required_files = {
        "X_train_scaled": train_dir / "X_train_scaled.csv",
        "y_train": train_dir / "y_train.csv",
        "X_test_scaled": test_dir / "X_test_scaled.csv",
        "y_test": test_dir / "y_test.csv",
    }

    for label, fpath in required_files.items():
        if not fpath.exists():
            raise FileNotFoundError(f"Required file not found: {fpath}")

    return required_files


def load_csv(file_path):
    """Load a CSV file and return DataFrame."""
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError(f"CSV file is empty: {file_path}")
    return df


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
# PHASE 1: LOAD DATA
# --------------------------------------------------
def load_all_data(file_paths):
    """Load all 4 CSV files and validate."""
    print(f"\n{'─' * 60}")
    print(f"📥 PHASE 1: LOADING DATA")
    print(f"{'─' * 60}")

    X_train = load_csv(file_paths["X_train_scaled"])
    y_train_df = load_csv(file_paths["y_train"])
    X_test = load_csv(file_paths["X_test_scaled"])
    y_test_df = load_csv(file_paths["y_test"])

    # Extract y as Series
    y_train = y_train_df[TARGET_COLUMN]
    y_test = y_test_df[TARGET_COLUMN]

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
    test_nans = int(X_test.isna().sum().sum() + y_test.isna().sum())
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
# --------------------------------------------------
def calculate_metrics(y_actual, y_predicted, set_name):
    """
    Calculate all evaluation metrics for a given set.
    Returns: metrics dict.
    """
    mae = mean_absolute_error(y_actual, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
    mape = mean_absolute_percentage_error(y_actual, y_predicted) * 100

    # R² needs at least 2 samples
    if len(y_actual) >= 2:
        r2 = r2_score(y_actual, y_predicted)
    else:
        r2 = None

    metrics = {
        "set_name": set_name,
        "sample_count": int(len(y_actual)),
        "MAE_mg_dL": round(float(mae), 4),
        "RMSE_mg_dL": round(float(rmse), 4),
        "R2_score": round(float(r2), 4) if r2 is not None else "undefined (need ≥2 samples)",
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
    test_r2 = test_metrics['R2_score']
    train_r2_str = f"{train_r2:.4f}" if isinstance(train_r2, float) else str(train_r2)
    test_r2_str = f"{test_r2:.4f}" if isinstance(test_r2, float) else str(test_r2)
    print(f"   {'R² Score':<25} {train_r2_str:>15} {test_r2_str:>15}")

    # MAPE
    print(f"   {'MAPE (%)':<25} {train_metrics['MAPE_percent']:>15.4f} {test_metrics['MAPE_percent']:>15.4f}")

    # Sample count
    print(f"   {'Samples':<25} {train_metrics['sample_count']:>15} {test_metrics['sample_count']:>15}")


# --------------------------------------------------
# PHASE 6: OVERFITTING ANALYSIS
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
        "train_mae": train_metrics["MAE_mg_dL"],
        "test_mae": test_metrics["MAE_mg_dL"],
        "train_rmse": train_metrics["RMSE_mg_dL"],
        "test_rmse": test_metrics["RMSE_mg_dL"],
        "diagnosis": "",
        "details": [],
    }

    # MAE comparison
    mae_ratio = test_metrics["MAE_mg_dL"] / train_metrics["MAE_mg_dL"] if train_metrics["MAE_mg_dL"] > 0 else float('inf')

    # RMSE comparison
    rmse_ratio = test_metrics["RMSE_mg_dL"] / train_metrics["RMSE_mg_dL"] if train_metrics["RMSE_mg_dL"] > 0 else float('inf')

    print(f"\n   📊 Error Ratio Analysis:")
    print(f"      MAE ratio  (Test/Train): {mae_ratio:.2f}")
    print(f"      RMSE ratio (Test/Train): {rmse_ratio:.2f}")

    # R² comparison
    train_r2 = train_metrics["R2_score"]
    test_r2 = test_metrics["R2_score"]

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

    analysis["diagnosis"] = diagnosis
    analysis["details"] = details
    analysis["mae_ratio"] = round(mae_ratio, 4)
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
        "Feature": feature_columns,
        "Importance": importances,
    }).sort_values("Importance", ascending=False).reset_index(drop=True)

    # Add rank and percentage
    total_importance = importances.sum()
    importance_df["Rank"] = range(1, len(importance_df) + 1)
    importance_df["Percentage"] = (importance_df["Importance"] / total_importance * 100).round(2)
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
        "total_features": len(feature_columns),
        "zero_importance_features": zero_features,
        "zero_importance_count": len(zero_features),
        "feature_ranking": [
            {
                "rank": int(row["Rank"]),
                "feature": row["Feature"],
                "importance": float(row["Importance"]),
                "percentage": float(row["Percentage"]),
                "cumulative_percentage": float(row["Cumulative_Percentage"]),
            }
            for _, row in importance_df.iterrows()
        ],
    }

    return importance_df, importance_log


# --------------------------------------------------
# PHASE 8: ACTUAL VS PREDICTED TABLE
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
        "Sample": range(1, len(y_train) + 1),
        "Actual_Glucose_mg_dL": y_train.values,
        "Predicted_Glucose_mg_dL": np.round(y_pred_train, 2),
        "Error_mg_dL": np.round(np.abs(y_train.values - y_pred_train), 2),
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
        "Sample": range(1, len(y_test) + 1),
        "Actual_Glucose_mg_dL": y_test.values,
        "Predicted_Glucose_mg_dL": np.round(y_pred_test, 2),
        "Error_mg_dL": np.round(np.abs(y_test.values - y_pred_test), 2),
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
        "train_predictions": train_table.to_dict(orient="records"),
        "test_predictions": test_table.to_dict(orient="records"),
        "train_avg_error_mg_dL": round(float(train_table["Error_mg_dL"].mean()), 4),
        "train_avg_percent_error": round(float(train_table["Percent_Error"].mean()), 4),
        "test_avg_error_mg_dL": round(float(test_table["Error_mg_dL"].mean()), 4),
        "test_avg_percent_error": round(float(test_table["Percent_Error"].mean()), 4),
    }

    return train_table, test_table, tables_log


# --------------------------------------------------
# PHASE 9: SAVE MODEL + RESULTS
# --------------------------------------------------
def build_full_report_json(
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
):
    """Build complete JSON report of the XGBoost pipeline."""

    hyperparameters = {
        "n_estimators": N_ESTIMATORS,
        "max_depth": MAX_DEPTH,
        "learning_rate": LEARNING_RATE,
        "subsample": SUBSAMPLE,
        "colsample_bytree": COLSAMPLE_BYTREE,
        "reg_alpha": REG_ALPHA,
        "reg_lambda": REG_LAMBDA,
        "min_child_weight": MIN_CHILD_WEIGHT,
        "gamma": GAMMA,
        "random_state": RANDOM_STATE,
        "objective": "reg:squarederror",
        "tree_method": "auto",
    }

    hyperparameter_explanations = {
        "n_estimators": "Number of boosting trees. More trees = more complex model.",
        "max_depth": "Maximum depth per tree. Deeper = more complex. Low for small data.",
        "learning_rate": "Step size per boosting round. Lower = slower but more robust.",
        "subsample": "Fraction of training rows sampled per tree. Adds randomness.",
        "colsample_bytree": "Fraction of features sampled per tree. Diversifies trees.",
        "reg_alpha": "L1 regularization. Pushes unimportant feature weights to zero.",
        "reg_lambda": "L2 regularization. Smooths weights to prevent dominance.",
        "min_child_weight": "Minimum weight in child node. Higher = more conservative.",
        "gamma": "Minimum loss reduction for split. Higher = fewer splits = simpler tree.",
        "random_state": "Fixed seed for reproducibility.",
        "objective": "Loss function. reg:squarederror = standard regression.",
        "tree_method": "Algorithm for tree construction. auto = best available.",
    }

    full_report = {
        "pipeline_info": {
            "pipeline_name": "XGBoost Glucose Prediction Model",
            "pipeline_step": "STEP 9",
            "execution_timestamp": timestamp_str,
            "execution_date_readable": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "previous_step": "STEP 8 (Sub-task 3 & 4) — Train/Test Split + Scaling",
        },
        "file_paths": {
            "input_folder": str(input_folder_path),
            "output_folder": str(output_folder_path),
            "output_files": output_file_paths,
        },
        "data_summary": {
            "train_samples": int(X_train.shape[0]),
            "test_samples": int(X_test.shape[0]),
            "total_samples": int(X_train.shape[0] + X_test.shape[0]),
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
        },
        "model_configuration": {
            "model_type": "XGBRegressor",
            "library": "xgboost",
            "hyperparameters": hyperparameters,
            "hyperparameter_explanations": hyperparameter_explanations,
            "training_time_seconds": round(training_time, 4),
        },
        "evaluation_metrics": {
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        },
        "overfitting_analysis": overfitting_analysis,
        "feature_importance": importance_log,
        "predictions": tables_log,
        "interpretation_guide": {
            "MAE": "Mean Absolute Error — average prediction error in mg/dL. Lower is better.",
            "RMSE": "Root Mean Squared Error — penalizes large errors more. Lower is better.",
            "R2": "R-squared — fraction of variance explained. 1.0 = perfect. Can be negative.",
            "MAPE": "Mean Absolute Percentage Error — error as percentage. Lower is better.",
            "overfitting_diagnosis": (
                "Compares train vs test error. "
                "If test error >> train error = overfitting. "
                "If similar = good generalization."
            ),
        },
    }

    return full_report


def save_all_outputs(model, train_table, test_table, importance_df,
                     full_report, output_root, timestamp_str):
    """
    Save all output files.

    XGBoost_Model_Results_YYYYMMDD_HHMMSS/
        ├── model/
        │   └── xgboost_glucose_model.json
        ├── predictions/
        │   ├── train_predictions.csv
        │   └── test_predictions.csv
        ├── importance/
        │   └── feature_importance.csv
        └── report/
            └── XGBoost_full_report.json
    """
    main_folder_name = f"XGBoost_Model_Results_{timestamp_str}"
    main_dir = output_root / main_folder_name
    main_dir.mkdir(parents=True, exist_ok=True)

    model_dir = main_dir / "model"
    pred_dir = main_dir / "predictions"
    importance_dir = main_dir / "importance"
    report_dir = main_dir / "report"

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
    test_pred_path = pred_dir / "test_predictions.csv"

    for fpath, data, label in [
        (train_pred_path, train_table, "Train predictions"),
        (test_pred_path, test_table, "Test predictions"),
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
        json.dump(full_report, f, indent=4, default=str)
    post_info = check_existing_file(report_path)
    print(f"   💾 Full report: {report_path.name} ({post_info['size_kb']:.2f} KB)")

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

    output_file_paths = {
        "main_folder": str(main_dir),
        "model": str(model_path),
        "train_predictions": str(train_pred_path),
        "test_predictions": str(test_pred_path),
        "feature_importance": str(importance_path),
        "full_report": str(report_path),
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

    # Generate timestamp
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── PHASE 1: Select and load data ──
    print(f"\n📂 Opening folder selector at: {INPUT_ROOT}")
    input_folder = popup_folder_selector(INPUT_ROOT)
    print(f"📁 Selected folder: {input_folder.name}")

    file_paths = find_train_test_files(input_folder)
    X_train, X_test, y_train, y_test, feature_columns = load_all_data(file_paths)

    # ── PHASE 3: Train model ──
    model, training_time = train_xgboost_model(X_train, y_train)

    # ── PHASE 4: Predictions ──
    y_pred_train, y_pred_test = make_predictions(model, X_train, X_test)

    # ── PHASE 5: Evaluation metrics ──
    train_metrics = calculate_metrics(y_train, y_pred_train, "TRAIN")
    test_metrics = calculate_metrics(y_test, y_pred_test, "TEST")
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
    full_report = build_full_report_json(
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
    )

    main_output_dir, output_file_paths = save_all_outputs(
        model=model,
        train_table=train_table,
        test_table=test_table,
        importance_df=importance_df,
        full_report=full_report,
        output_root=OUTPUT_ROOT,
        timestamp_str=timestamp_str,
    )

    # Update report with actual file paths and re-save
    full_report["file_paths"]["output_files"] = output_file_paths
    report_path = Path(output_file_paths["full_report"])
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=4, default=str)

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
    test_r2 = test_metrics['R2_score']
    train_r2_str = f"{train_r2:.4f}" if isinstance(train_r2, float) else str(train_r2)
    test_r2_str = f"{test_r2:.4f}" if isinstance(test_r2, float) else str(test_r2)
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
    main()