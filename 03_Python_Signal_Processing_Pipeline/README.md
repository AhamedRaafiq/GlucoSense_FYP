# Signal Processing Pipeline

## Overview
This folder contains Jupyter notebooks for processing raw PPG signals into features suitable for diabetes prediction. The pipeline includes normalization, filtering, feature extraction, and visualization.

## Pipeline Stages

The signal processing follows these sequential stages:

### 1. **Normalization**
- `Normalization.ipynb` - Initial normalization approach
- `Normalization2.ipynb` - Improved normalization with advanced techniques
- Converts raw ADC values to standardized ranges
- Removes DC offset and baseline wander

### 2. **Filtering**
- `Trimming_&_Low_Pass_Filter.pynb` - Signal trimming and noise reduction
- `SAVITZKY-GOLAY_VISUALIZATION.ipynb` - Savitzky-Golay filter visualization
- Removes high-frequency noise
- Preserves important signal characteristics

### 3. **Feature Extraction**
- `x.ipynb`, `x2.ipynb`, `x3.ipynb` - Exploratory feature analysis notebooks
- Extracts time-domain and frequency-domain features
- Calculates physiological parameters (heart rate, SpO2, etc.)

## Notebooks

| Notebook | Purpose | Input | Output |
|----------|---------|-------|--------|
| `Normalization.ipynb` | Basic signal normalization | Raw CSV | Normalized signals |
| `Normalization2.ipynb` | Advanced normalization | Raw CSV | Normalized signals |
| `Trimming_&_Low_Pass_Filter.pynb` | Noise filtering | Normalized | Filtered signals |
| `SAVITZKY-GOLAY_VISUALIZATION.ipynb` | Filter visualization | Any signal | Plots |
| `x.ipynb` | Feature exploration | Filtered | Features |
| `x2.ipynb` | Advanced features | Filtered | Features |
| `x3.ipynb` | Feature analysis | Filtered | Features |

> **Note**: Notebooks `x.ipynb`, `x2.ipynb`, and `x3.ipynb` will be renamed to more descriptive names in the future.

## Getting Started

### Prerequisites
```bash
pip install -r ../requirements.txt
```

Required packages:
- `jupyter` - Notebook environment
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `matplotlib` - Visualization
- `scipy` - Signal processing
- `scikit-learn` - Feature scaling

### Running the Pipeline

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Process data sequentially**:
   - Start with normalization notebooks
   - Apply filtering
   - Extract features

3. **Input data location**: `../04_Data_Storage/Raw/`
4. **Output data location**: `../04_Data_Storage/Cleaned/` or `../04_Data_Storage/_Features_/`

## Signal Processing Techniques

### Normalization Methods
- Min-max scaling
- Z-score normalization
- Robust scaling (median-based)

### Filtering Techniques
- Low-pass filtering (remove high-frequency noise)
- Savitzky-Golay filter (smooth while preserving peaks)
- Baseline wander removal
- Motion artifact reduction

### Feature Types
- **Time-domain**: Peak amplitude, inter-beat intervals, signal variance
- **Frequency-domain**: FFT analysis, power spectral density
- **Morphological**: Pulse width, rise time, fall time
- **Statistical**: Mean, median, standard deviation, skewness

## Data Flow

```
Raw Data (04_Data_Storage/Raw/)
    ↓
Normalization
    ↓
Filtering
    ↓
Feature Extraction
    ↓
Features (04_Data_Storage/_Features_/)
    ↓
Machine Learning (06_Machine_Learning_Models/)
```

## Best Practices

1. **Always work on copies** - Keep raw data untouched
2. **Document parameters** - Note filter settings and thresholds
3. **Visualize intermediate steps** - Check signal quality after each stage
4. **Version control** - Save notebook outputs with meaningful names
5. **Reproducibility** - Set random seeds for consistent results

## Troubleshooting

### Poor Signal Quality
- Check normalization parameters
- Adjust filter cutoff frequencies
- Verify input data quality

### Feature Extraction Errors
- Ensure signals are properly filtered
- Check for NaN or infinite values
- Validate data ranges

## Related Folders
- `02_Python_Data_Logger/` - Captures raw data
- `04_Data_Storage/` - Stores all pipeline data
- `06_Machine_Learning_Models/` - Uses extracted features
