# Signal Processing Pipeline

## Overview
This folder contains the comprehensive signal processing pipeline for PPG signals. The main notebook `Filters_Normalization_SignalQualityCheck.ipynb` implements a complete end-to-end pipeline including data loading, filtering, normalization, and signal quality assessment.

## Main Pipeline Notebook

### `Filters_Normalization_SignalQualityCheck.ipynb`

This is the primary signal processing notebook that implements a configurable pipeline with the following stages:

#### **Configuration Parameters**
- **File & Device Settings**: Input file path, sampling frequency (400 Hz)
- **Data Selection**: Sample range selection or full file processing
- **Plotting Settings**: Customizable subplot heights and grid intervals
- **Validity Thresholds**: Signal quality thresholds for IR and Red channels

#### **Processing Stages**

1. **Signal Inversion** (Optional)
   - Inverts signal polarity if needed
   - Configurable via `INVERT_ENABLE` flag

2. **Spike Removal** (Optional)
   - Removes signal spikes using median filtering
   - Kernel size: 51 samples (configurable)
   - Configurable via `SPIKE_ENABLE` flag

3. **Low-Pass Filtering**
   - Cutoff frequency: 16.0 Hz (configurable)
   - Order: 4 (configurable)
   - Removes high-frequency noise
   - Configurable via `LP_ENABLE` flag

4. **High-Pass Filtering**
   - Cutoff frequency: 0.5 Hz (configurable)
   - Order: 4 (configurable)
   - Removes baseline wander and DC offset
   - Configurable via `HP_ENABLE` flag

5. **Savitzky-Golay Smoothing**
   - Window size: 31 samples (configurable)
   - Polynomial order: 3 (configurable)
   - Preserves signal shape while smoothing
   - Configurable via `SG_ENABLE` flag

6. **Normalization**
   - Method 1: Min-Max normalization (0-1 range)
   - Method 2: Z-Score normalization (mean=0, std=1)
   - Selectable via `NORM_SELECTION` parameter

7. **Signal Quality Index (SQI) Assessment**
   - **Skewness**: Acceptable range 0.0-2.5 (Best: 0.5-1.5)
   - **Kurtosis**: Acceptable range 1.5-7.0 (Best: 2.0-4.0)
   - **Perfusion Index (PI)**: Acceptable range 0.1-10.0 (Best: 0.5-4.0)
   - **Signal-to-Noise Ratio (SNR)**: Acceptable range 5.0-25.0 dB (Best: 12-25 dB)
   - **Zero-Crossing Rate (ZCR)**: Acceptable range 1.0-4.0 Hz (Best: 1.5-2.5 Hz)

## Additional Notebooks

### `SAVITZKY-GOLAY_VISUALIZATION.ipynb`
- Visualizes the effect of Savitzky-Golay filter parameters
- Helps optimize filter settings for signal smoothing

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
- `scipy` - Signal processing (filters, signal analysis)

### Running the Pipeline

1. **Start Jupyter**:
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**:
   - `Filters_Normalization_SignalQualityCheck.ipynb`

3. **Configure parameters** in the first cell:
   - Set input file path
   - Choose processing options (filters, normalization method)
   - Adjust filter parameters if needed

4. **Run all cells** to process the signal

5. **Review outputs**:
   - Visualizations of each processing stage
   - Signal quality metrics
   - Final processed signal

## Data Flow

```
Raw CSV Data (05_Data_Storage/Windowed/)
    ↓
Load & Validate
    ↓
Optional: Spike Removal
    ↓
Optional: Signal Inversion
    ↓
Low-Pass Filter (16 Hz)
    ↓
High-Pass Filter (0.5 Hz)
    ↓
Savitzky-Golay Smoothing
    ↓
Normalization (Min-Max or Z-Score)
    ↓
Signal Quality Assessment
    ↓
Processed Signal Ready for Feature Extraction
```

## Signal Quality Metrics

The pipeline calculates comprehensive SQI metrics to assess signal quality:

- **Skewness**: Measures signal asymmetry
- **Kurtosis**: Measures signal peakedness
- **Perfusion Index**: Ratio of pulsatile to non-pulsatile signal
- **SNR**: Signal-to-noise ratio in dB
- **Zero-Crossing Rate**: Frequency of signal zero crossings

Each metric has defined acceptable ranges and optimal "best" ranges for high-quality signals.

## Best Practices

1. **Start with default parameters** - The default configuration works well for most PPG signals
2. **Visualize each stage** - Check plots to ensure proper filtering
3. **Monitor SQI metrics** - Ensure signals meet quality thresholds
4. **Adjust filters carefully** - Small changes in cutoff frequencies can significantly affect results
5. **Document changes** - Note any parameter modifications for reproducibility

## Troubleshooting

### Poor Signal Quality After Filtering
- Check if spike removal is needed (`SPIKE_ENABLE`)
- Verify signal polarity (`INVERT_ENABLE`)
- Adjust low-pass cutoff frequency (try 12-20 Hz range)
- Modify Savitzky-Golay window size (try 21-41 samples)

### SQI Metrics Out of Range
- Review raw signal quality
- Check for motion artifacts
- Verify sensor placement during data collection
- Consider adjusting validity thresholds

### Visualization Issues
- Adjust `SUBPLOT_HEIGHT` for better plot visibility
- Modify grid line intervals for clearer time/frequency scales

## Related Folders
- `02_Python_Data_Logger/` - Captures raw data
- `03_Python_Data_Processing/` - Windows the raw data
- `05_Data_Storage/Windowed/` - Input windowed data
- `05_Signal_Feature_Learning/` - Feature analysis
- `07_Machine_Learning_Models/` - Uses processed signals
