# Machine Learning Models

## Overview
This folder contains machine learning models for predicting diabetes risk from PPG (Photoplethysmography) signal features.

## Purpose
- Train classification/regression models for diabetes prediction
- Store trained model files (.pkl, .h5, .pt)
- Evaluate model performance
- Compare different ML algorithms

## Planned Contents

### Model Types
- **Classical ML**: Random Forest, SVM, XGBoost
- **Deep Learning**: Neural Networks, LSTM, CNN
- **Ensemble Methods**: Stacking, Voting Classifiers

### File Structure (To Be Created)
```
06_Machine_Learning_Models/
├── trained_models/          # Saved model files
├── notebooks/               # Training notebooks
├── evaluation/              # Performance metrics
├── hyperparameters/         # Model configurations
└── README.md               # This file
```

## Input Data
Models will use features from `../04_Data_Storage/_Features_/`

## Expected Outputs
- Trained model files
- Performance metrics (accuracy, precision, recall, F1)
- Confusion matrices
- ROC curves
- Feature importance rankings

## Getting Started
This folder is currently empty. Models will be added as the project progresses.

## Related Folders
- `04_Data_Storage/_Features_/` - Input features for training
- `07_Results_and_Visualizations/` - Model performance visualizations
- `08_Tests/` - Model validation tests
