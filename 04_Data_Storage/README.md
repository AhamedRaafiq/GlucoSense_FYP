# Data Storage

## Overview
This folder contains all data used in the non-invasive diabetes prediction project, organized by processing stage. Data flows through multiple stages from raw sensor readings to extracted features ready for machine learning.

## Folder Structure

### `Raw/`
**Purpose**: Original, unprocessed PPG signals directly from the ESP32 sensor

**Contents**:
- CSV files with timestamp-based naming
- Two columns: IR, RED
- 18-bit ADC values (0-262143)
- May contain noise, artifacts, and baseline wander

**Naming Convention**: `YYYY-MM-DD_HH-MM-SS_raw.csv`

**Do NOT modify files in this folder** - always work on copies

---

### `Cleaned/`
**Purpose**: Normalized and filtered signals ready for analysis

**Contents**:
- Normalized signals (standardized ranges)
- Filtered data (noise removed)
- Baseline-corrected signals
- Same CSV format as Raw

**Processing Applied**:
- Normalization (min-max or z-score)
- Low-pass filtering
- Baseline wander removal
- Motion artifact reduction

**Naming Convention**: `YYYY-MM-DD_HH-MM-SS_cleaned.csv`

---

### `Windowed/`
**Purpose**: Segmented signals split into fixed-duration windows

**Contents**:
- Time-windowed signal segments
- Typically 5-10 second windows
- Used for beat-to-beat analysis
- May include overlap between windows

**Use Cases**:
- Heart rate variability analysis
- Beat detection
- Temporal feature extraction

**Naming Convention**: `YYYY-MM-DD_HH-MM-SS_window_001.csv`

---

### `_Features_/`
**Purpose**: Extracted features ready for machine learning models

**Contents**:
- CSV files with feature columns
- Time-domain features (peak amplitude, intervals)
- Frequency-domain features (FFT, PSD)
- Morphological features (pulse shape)
- Statistical features (mean, std, skewness)

**Format**: Each row represents one signal segment with multiple feature columns

**Naming Convention**: `YYYY-MM-DD_features.csv`

---

### `Normal_PPG_Only_Data_Set_For_Practice/`
**Purpose**: Reference dataset of normal (healthy) PPG signals

**Contents**:
- Baseline PPG signals from healthy individuals
- Used for algorithm development and testing
- Practice data for model training

**Use Cases**:
- Algorithm validation
- Baseline comparison
- Initial model training

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Pipeline                            │
└─────────────────────────────────────────────────────────────┘

ESP32 Sensor
     ↓
[Raw/] ← Original sensor data
     ↓
Normalization & Filtering
     ↓
[Cleaned/] ← Processed signals
     ↓
Windowing (optional)
     ↓
[Windowed/] ← Segmented signals
     ↓
Feature Extraction
     ↓
[_Features_/] ← ML-ready features
     ↓
Machine Learning Models
```

## File Naming Conventions

### Timestamps
All files use ISO 8601 format: `YYYY-MM-DD_HH-MM-SS`

Example: `2026-01-02_14-30-45_raw.csv`

### Suffixes
- `_raw.csv` - Raw sensor data
- `_cleaned.csv` - Processed signals
- `_window_NNN.csv` - Windowed segment (NNN = segment number)
- `_features.csv` - Extracted features

### Subject Identifiers (if applicable)
- `subject_001_YYYY-MM-DD_raw.csv`

## Data Management Best Practices

### Storage
- Keep raw data indefinitely
- Cleaned data can be regenerated from raw
- Features can be regenerated from cleaned
- Archive old experiments in dated subfolders

### Backup
- Regular backups of `Raw/` folder (most critical)
- Version control for processing scripts
- Document processing parameters

### Git Tracking
- **DO NOT** commit large CSV files to Git
- Use `.gitignore` to exclude `*.csv`
- Commit only small sample datasets for testing
- Use Git LFS for large files if necessary

### Organization Tips
1. Create subfolders by date or experiment
2. Document data collection conditions
3. Keep metadata files (JSON/YAML) with parameters
4. Use consistent naming across all stages

## Disk Space Considerations

Typical file sizes:
- Raw data: ~1-5 MB per minute of recording
- Cleaned data: Similar to raw
- Windowed data: Depends on window size and overlap
- Features: Much smaller (~KB per file)

**Recommendation**: Allocate at least 10 GB for data storage

## Related Folders
- `02_Python_Data_Logger/` - Populates `Raw/` folder
- `03_Python_Signal_Processing_Pipeline/` - Processes data through stages
- `06_Machine_Learning_Models/` - Consumes `_Features_/` data
