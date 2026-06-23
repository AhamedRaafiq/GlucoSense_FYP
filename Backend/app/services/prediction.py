import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from ..config import settings

# Load model and scaler parameters
_model = None
_scaler_params = None

# Feature list used in model training (15 features)
KEPT_FEATURES = [
    "IR_Skewness",
    "IR_Spectral Entropy",
    "IR_pulse width",
    "IR_PPI",
    "IR_HRV",
    "IR_TEO Mean",
    "IR_1st_Derivative_Mean",
    "IR_2nd_Derivative_Mean",
    "IR_2nd_Derivative_Skewness",
    "IR_Decay time",
    "IR_Dicrotic notch",
    "Diff_2nd_Derivative_Mean",
    "Diff_Spectral_Entropy",
    "Diff_Dicrotic_notch",
    "Ensemble ratio"
]

def load_prediction_assets():
    """Load model and scaler parameters dynamically."""
    global _model, _scaler_params
    
    # Load model
    if _model is None:
        if os.path.exists(settings.MODEL_PATH):
            with open(settings.MODEL_PATH, 'rb') as f:
                _model = pickle.load(f)
        else:
            raise FileNotFoundError(f"XGBoost model file not found at {settings.MODEL_PATH}")
            
    # Load scaler params
    if _scaler_params is None:
        if os.path.exists(settings.SCALER_PATH):
            with open(settings.SCALER_PATH, 'r') as f:
                _scaler_params = json.load(f)
        else:
            raise FileNotFoundError(f"Scaler parameters not found at {settings.SCALER_PATH}")
            
    return _model, _scaler_params

def scale_features(features: Dict[str, float], scaler_params: Dict[str, Any]) -> Dict[str, float]:
    """
    Scale features using RobustScaler parameters.
    Formula: X_scaled = (X - median) / IQR
    """
    scaled = {}
    for feature_name, value in features.items():
        if feature_name in scaler_params:
            median = scaler_params[feature_name]['median']
            iqr = scaler_params[feature_name]['iqr']
            if iqr == 0:
                scaled[feature_name] = 0.0
            else:
                scaled[feature_name] = (value - median) / iqr
        else:
            scaled[feature_name] = value
    return scaled

def get_glucose_classification(glucose: float) -> str:
    """Classify blood glucose level into clinical categories."""
    if glucose < 70.0:
        return "Hypoglycemic"
    elif 70.0 <= glucose <= 100.0:
        return "Normal"
    elif 100.0 < glucose <= 125.0:
        return "Pre-diabetic"
    elif 125.0 < glucose <= 180.0:
        return "Diabetic"
    else:
        return "Hyperglycemic"

def predict_glucose(averaged_features: Dict[str, float]) -> Tuple[float, str]:
    """
    Perform blood glucose prediction.
    1. Scale features using the saved scaler params.
    2. Extract only the 15 kept features.
    3. Run prediction with XGBoost model.
    """
    model, scaler_params = load_prediction_assets()
    
    # 1. Scale all features
    scaled_features = scale_features(averaged_features, scaler_params)
    
    # 2. Keep only the 15 features selected during training
    # Check if all features exist
    x_input = []
    for feat in KEPT_FEATURES:
        if feat in scaled_features:
            x_input.append(scaled_features[feat])
        else:
            # Fallback to unscaled feature value or 0
            x_input.append(averaged_features.get(feat, 0.0))
            
    # Convert to 2D array for prediction (1 sample, 15 features)
    x_input_arr = np.array([x_input])
    
    # Predict
    predicted_val = float(model.predict(x_input_arr)[0])
    
    # Make sure predictions are reasonable (e.g. at least 40 mg/dL)
    predicted_val = max(40.0, predicted_val)
    
    classification = get_glucose_classification(predicted_val)
    
    return predicted_val, classification
