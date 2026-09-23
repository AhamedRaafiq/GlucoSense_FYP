# 🧠 Signal Feature Learning

> Step 05 & 06: Batch-aware, rejection-safe PPG feature extraction for Machine Learning.

This module extracts a comprehensive suite of 39 physiological features from segmented PPG windows (Step 05) and computes subject-level averages (Step 06) to prepare the data for dataset creation (Step 07).

## 🚀 Key Capabilities

- **Robust Extraction:** Handles anomalies with `safe_array()` (strips NaN/Inf) and `safe_ratio()` (prevents ZeroDivision).
- **Rejection-Aware:** Skips low-quality windows identified in Step 4 (`metadata.status == REJECTED`).
- **State Management:** Automatically purges stale feature directories before execution.
- **Execution Modes:** Supports both `BATCH` (automated pipeline) and `SINGLE` (GUI/headless) modes.
- **Noise Reduction (Step 06):** Applies column-wise arithmetic averaging across valid windows to reduce variance (1/√W).

## 📊 Extracted Features

Extracts 19 features per channel (Red + IR) plus 1 cross-channel ratio, yielding a **39-column feature vector**.

| Group | Features |
|-------|----------|
| **Statistical** | Shannon Entropy (64-bin), Spectral Entropy (Welch PSD) |
| **Waveform Shape** | Skewness, Kurtosis, Pulse Width FWHM, Systolic Amplitude, Rise Time, Decay Time, Dicrotic Notch |
| **Heart Rate** | Peak-to-Peak Interval (PPI), BPM, Heart Rate Variability (SDNN) |
| **Signal Energy** | Teager Energy Operator (TEO) Mean, TEO Standard Deviation |
| **Derivative** | 1st Derivative Mean (VPG), 2nd Derivative Mean (SDPPG), 2nd Derivative Skewness |
| **Spectral** | Harmonic Ratio |
| **Cross-Channel** | Ensemble Ratio (Red/IR AC-DC ratio) |

## 🔄 Input / Output Specification

### Input (From Step 04)
- `Full CSV`: Complete window data
- `Ensemble CSV`: Average heartbeat waveform
- `Configuration JSON`: Metadata and status

### Output (Per Window)
| File | Description |
|------|-------------|
| `Features_Table.csv` | Human-readable 19×3 table of features. |
| `Features_Flat.csv` | Machine-readable 1×39 vector for ML input. |
| `Features.json` | Key-value store of all extracted features. |
| `Configuration.json` | Inherited and updated metadata. |
| `Signal_Overview.png` | Visual summary of the processed window. |

## 🛠️ Quick Start

**Prerequisites:**
```bash
pip install numpy pandas scipy matplotlib
```

**Run the pipeline:**
```bash
# Execute feature extraction (Step 05)
python step05_feature_extraction.py --mode BATCH

# Execute feature averaging (Step 06)
python step06_feature_averaging.py
```