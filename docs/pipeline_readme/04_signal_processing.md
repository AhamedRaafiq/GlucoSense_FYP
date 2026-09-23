# 📈 Automated Signal Processing Pipeline (Step 04)

> 12-Stage DSP engine converting raw 15s PPG windows into clean, ensemble-averaged single-beat templates.

## 📝 Overview
This module serves as the heavy-lifting digital signal processing (DSP) engine in the GlucoSense pipeline. It ingests noisy, raw photoplethysmography (PPG) signals and processes them through a rigorous 12-stage automated pipeline to output high-fidelity, validated ensemble-averaged beats ready for feature extraction.

**Pipeline Position:** Step 04 — Bridges raw windowed data (Step 03) to feature extraction (Step 05).

## ✨ Key Features
- **Comprehensive 12-Stage Processing:** Includes filtering, inversion, smoothing, and normalization.
- **Robust Signal Quality Indexing (SQI):** Evaluates Skewness, Kurtosis, Perfusion Index (PI), Zero Crossing Rate (ZCR), and SNR at every stage.
- **Advanced Ensemble Averaging:** Peak/valley detection, VPG foot refinement, beat validation, and fixed-length resampling (220 points).
- **SDPPG Fiducial Extraction:** Automatically identifies a, b, c, d, and e points on the second derivative PPG.
- **Multiple Execution Modes:** Supports BATCH (all subjects), SINGLE (GUI picker), and MULTI (sequential selections).

## ⚙️ Technical Specifications

| Parameter | Value | Description |
|-----------|-------|-------------|
| **Sampling Rate (FS)** | 400.0 Hz | Original signal sampling frequency |
| **Low-Pass Filter** | 16.0 Hz | 4th-order Butterworth SOS (Zero-phase) |
| **High-Pass Filter** | 0.5 Hz | 4th-order Butterworth |
| **Ensemble Length** | 220 samples | Fixed size for ensemble-averaged beats |
| **Min Valid Beats** | 8 | Required validated beats for ensemble |

### 📊 SQI Acceptable Limits
| Metric | Valid Range |
|--------|-------------|
| **Skewness** | 0 to 2.5 |
| **Kurtosis** | 1.5 to 7.0 |
| **Perfusion Index (PI)** | 0.1% to 10% |
| **Zero Crossing Rate (ZCR)** | 1.0 to 4.0 Hz |
| **SNR** | 5 to 25 dB |

## 🔀 Input & Output Formats

**📥 Inputs:**
- `{subject}_Windowed/{subject}_Win{n}.csv` (Raw 15s window @ 400Hz = 6000 samples)

**📤 Outputs (Per Window):**
- `*_Filtered_Full.csv` (6 columns: processed signal at key stages)
- `*_Filtered_Ensemble.csv` (8 columns, 220 rows: single-beat template + derivatives)
- `*_Configuration.json` (Pipeline hyperparameters)
- Diagnostic plots (ensemble, beat numbering, diagnostic per stage)

**📤 Outputs (Per Subject):**
- `*_Combined_Report.json` (Aggregated SQI and pipeline status)

## 🚀 Quick Start

### 1. Requirements
Ensure you are running **Python 3.10+** with at least 4-8 GB RAM.
Install the required dependencies:
```bash
pip install numpy pandas scipy matplotlib
```

### 2. Execution
Run the main script to initiate the pipeline:
```bash
python Automated_Signal_Processing_Code.py
```
*Select your preferred execution mode (BATCH/SINGLE/MULTI) when prompted.*

## 🧠 Architecture: The 12 Stages
1. **Hyperparameter Loading:** Initializes DSP settings.
2. **CSV Validation & Selection:** Column normalization and threshold checks.
3. **Spike Removal:** Median filter (kernel=3).
4. **Signal Inversion:** Inverts reflective PPG (×-1).
5. **Low-Pass Filter:** Removes high-frequency noise.
6. **Savitzky-Golay Smoothing:** (Optional) poly=3, window=31.
7. **High-Pass Filter:** Removes baseline wander.
8. **Normalization:** MinMax [0,1] or Z-Score.
9. **Signal Quality Index (SQI):** Computes 5 critical quality metrics.
10. **Pipeline Diagnostic:** SQI evaluated at every stage with R-ratio.
11. **Ensemble Detection + SDPPG:** Identifies beats, aligns, resamples to 220pts, computes ensemble average, and extracts SDPPG fiducials.
12. **Golden Standard Features & Verification:** Final validation and data persistence.

## 🛠 Troubleshooting

| Issue | Potential Cause | Solution |
|-------|-----------------|----------|
| **Window fails SQI check** | Extreme noise or motion artifact | Check raw signal plots; ensure window is clean. Pipeline automatically skips bad windows. |
| **Insufficient beats for ensemble** | Min valid beats (<8) not met | The signal might be highly irregular. Check for arrhythmias or sensor disconnections. |
| **Memory errors during BATCH** | Insufficient RAM | Ensure at least 4GB RAM is available. Close other applications or run in MULTI mode. |