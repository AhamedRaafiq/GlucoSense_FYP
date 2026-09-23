# 🤖 Machine Learning Models: XGBoost Clinical Training

This directory contains the machine learning pipeline for non-invasive blood glucose estimation. It trains an optimized XGBoost regression model, applying clinical sample weighting and rigorous cross-validation to ensure medical-grade accuracy.

## Pipeline Integration
This step (Step 09) ingest the 24-dimensional feature vectors generated in Step 08, training the final predictive model and outputting serialized models for deployment.

## 🚀 Key Features

- **10-Phase Execution Pipeline:** Automates data ingestion, feature selection, training, inference, and evaluation.
- **Clinical Sample Weighting:** Prioritizes hyperglycemic ranges (glucose ≥130 mg/dL receives 2.0x weight) to penalize dangerous underestimations.
- **Comprehensive Evaluation:** 5-Fold Cross-Validation, Clarke Error Grid Analysis, and Per-Glucose-Range Error Analysis across 5 clinical tiers (<70, 70-100, 100-125, 125-180, >180 mg/dL).
- **Overfitting Diagnosis:** Analyzes test/train MAE ratios (Good <1.5, Mild 1.5-2.0, Moderate 2.0-3.0, Severe >3.0).
- **Feature Importance Tracking:** Ranks features by percentage with cumulative contribution tracking.

## ⚙️ Model Configuration

| Parameter | Value | Description |
|---|---|---|
| `n_estimators` | 75 | Number of boosting rounds |
| `max_depth` | 2 | Maximum tree depth to prevent overfitting |
| `learning_rate` | 0.03 | Step size shrinkage used in update |
| `subsample` | 0.8 | Subsample ratio of the training instances |
| `colsample_bytree` | 0.7 | Subsample ratio of columns when constructing each tree |
| `reg_alpha` | 0.5 | L1 regularization term on weights |
| `reg_lambda` | 5.0 | L2 regularization term on weights |
| `min_child_weight` | 5 | Minimum sum of instance weight needed in a child |
| `gamma` | 0.3 | Minimum loss reduction required for a further partition |

*Top features identified: IR_Skewness (18.12%), IR_Spectral_Entropy (15.04%), IR_pulse_width (12.15%).*

## 📁 Input & Output

**Input Format:** 
- `X_train` (63×24), `y_train` (63)
- `X_test` (12×24), `y_test` (12)

**Output Artifacts:**
- `model/`: XGBoost JSON model, Pickle file, Feature template
- `predictions/`: Train and Test predictions (CSV)
- `importance/`: Feature importance rankings (CSV)
- `report/`: Statistical evaluation report (JSON)
- `plots/`: 5 evaluation plots (PNG) including Clarke Error Grid

## 🚀 Quick Start

**1. Install Dependencies**
```bash
pip install numpy pandas scikit-learn xgboost matplotlib
```

**2. Run the Training Pipeline**
```bash
python XGBoost_ML_Code11.py
```

**3. Extract Tree Topology (Optional)**
```bash
python extract_model_details.py
```

## 🛠️ Troubleshooting

| Issue | Potential Cause | Solution |
|---|---|---|
| **Severe Overfitting (MAE Ratio > 3.0)** | Model memorizing training data | Reduce `max_depth`, increase `reg_lambda` or `reg_alpha`. |
| **Missing Input Data** | Step 08 did not complete | Run Step 08 feature extraction pipeline first. |
| **Feature Mismatch** | Test data missing features | Ensure consistent feature extraction; check `manual_selection` mode. |
