import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from ..config import settings
from .prediction import KEPT_FEATURES

def clean_data_and_clip_outliers(df: pd.DataFrame, target_col: str, iqr_multiplier: float = 1.5) -> pd.DataFrame:
    """Handle NaNs and clip outliers using IQR method for all feature columns."""
    df_clean = df.copy()
    
    # Fill NaN values with median
    for col in df_clean.columns:
        if col != target_col:
            median_val = df_clean[col].median()
            df_clean[col] = df_clean[col].fillna(median_val if not np.isnan(median_val) else 0.0)
            
    # Clip outliers
    for col in df_clean.columns:
        if col != target_col:
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            
    return df_clean

def retrain_model_with_db_data(training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Retrain the XGBoost model using the data stored in the database.
    Each item in training_data contains 'features' (dict of 24 features) and 'glucose' (float target).
    """
    if len(training_data) < 10:
        return {
            "success": False,
            "message": f"Insufficient training data. Need at least 10 samples, got {len(training_data)}."
        }
        
    # Create DataFrame
    records = []
    for item in training_data:
        rec = item['features'].copy()
        rec['Glucose_Level'] = item['glucose']
        records.append(rec)
        
    df = pd.DataFrame(records)
    target_col = 'Glucose_Level'
    
    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 1. Clean data and clip outliers
    df_cleaned = clean_data_and_clip_outliers(df, target_col)
    X_clean = df_cleaned.drop(columns=[target_col])
    y_clean = df_cleaned[target_col]
    
    # 2. Fit RobustScaler
    scaler = RobustScaler()
    scaler.fit(X_clean)
    
    # Save scaler parameters
    scaler_params = {}
    for i, col in enumerate(X_clean.columns):
        scaler_params[col] = {
            "median": float(scaler.center_[i]),
            "iqr": float(scaler.scale_[i])
        }
        
    # Scale X
    X_scaled_arr = scaler.transform(X_clean)
    X_scaled = pd.DataFrame(X_scaled_arr, columns=X_clean.columns)
    
    # 3. Keep only the 15 features selected for the model
    X_model = X_scaled[KEPT_FEATURES]
    
    # 4. Calculate sample weights (higher weight for glucose >= 130)
    sample_weights = np.where(y_clean >= 130.0, 2.0, 1.0)
    
    # 5. Initialize and train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=75,
        max_depth=2,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_alpha=0.5,
        reg_lambda=5,
        min_child_weight=5,
        gamma=0.3,
        random_state=42,
        objective="reg:squarederror"
    )
    
    model.fit(X_model, y_clean, sample_weight=sample_weights)
    
    # Backup current model and scaler
    if os.path.exists(settings.MODEL_PATH):
        os.rename(settings.MODEL_PATH, settings.MODEL_PATH + ".bak")
    if os.path.exists(settings.SCALER_PATH):
        os.rename(settings.SCALER_PATH, settings.SCALER_PATH + ".bak")
        
    # Save new assets
    with open(settings.MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    with open(settings.SCALER_PATH, 'w') as f:
        json.dump(scaler_params, f, indent=4)
        
    # Reload predictor assets in prediction.py (clear global cache)
    from .prediction import load_prediction_assets
    import sys
    # Clear prediction globals
    import Backend.app.services.prediction as pred
    pred._model = None
    pred._scaler_params = None
    load_prediction_assets()
    
    # Calculate training metrics
    preds = model.predict(X_model)
    mae = float(np.mean(np.abs(y_clean - preds)))
    rmse = float(np.sqrt(np.mean((y_clean - preds)**2)))
    
    # R2 Score manually
    y_mean = np.mean(y_clean)
    ss_tot = np.sum((y_clean - y_mean)**2)
    ss_res = np.sum((y_clean - preds)**2)
    r2 = float(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
    
    return {
        "success": True,
        "message": f"Successfully retrained model with {len(training_data)} samples.",
        "metrics": {
            "mae": mae,
            "rmse": rmse,
            "r2": r2
        }
    }
