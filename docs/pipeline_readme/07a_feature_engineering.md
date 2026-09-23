# ⚙️ Step 08: 24-Feature Engineering

This module transforms the raw 30+ feature master dataset into a curated, 24-feature dataset optimized for machine learning models. 

**Pipeline Integration:** Consumes Step 07 output (`MASTER_Dataset.csv` & `BuildLog.json`) and feeds into Step 09 (Data Cleaning + Train/Test Split).

## ✨ Key Features
- **Dimensionality Reduction:** Drops 7 redundant Red signal features, yielding a streamlined 25-column output (24 features + 1 target).
- **Three-Tier Classification:** Organizes features into Base IR, Engineered, and Keep-As-Is categories.
- **Rigorous Verification:** 5-point integrity checks including row/column counts, IR preservation (`1e-9` tolerance), engineered spot-checks, and glucose mapping.
- **Cross-Pipeline Traceability:** Ingests Step 07 `BuildLog.json` to ensure unbroken provenance.

## 📊 Three-Tier Feature Classification

### Tier 1: IR Base Features (18)
Direct copies of IR signals, chosen for deeper tissue penetration (~5mm) and higher SNR.
*Includes:* `IR_Skewness`, `IR_Kurtosis`, `IR_Shannon_Entropy`, `IR_Spectral_Entropy`, `IR_pulse_width`, `IR_PPI`, `IR_systolic_amplitude`, `IR_BPM`, `IR_HRV`, `IR_TEO_Mean`, `IR_TEO_std_dev`, `IR_1st_Derivative_Mean`, `IR_2nd_Derivative_Mean`, `IR_2nd_Derivative_Skewness`, `IR_Harmonic_ratio`, `IR_Rise_time`, `IR_Decay_time`, `IR_Dicrotic_notch`

### Tier 2: Engineered Features (5)
Mathematical operations combining Red and IR features to capture critical physiological interactions:
| Feature | Operation | Rationale |
|---------|-----------|-----------|
| `Ratio_systolic_amplitude` | Red/IR | SpO2/HbA1c balance |
| `Ratio_TEO_Mean` | Red/IR | Signal energy/viscosity |
| `Diff_2nd_Derivative_Mean` | Red-IR | Arterial stiffness |
| `Diff_Spectral_Entropy` | Red-IR | Frequency complexity |
| `Diff_Dicrotic_notch` | Red-IR | Vascular compliance |

### Tier 3: Keep-As-Is (1) + Target
- **Ensemble ratio**
- **Target:** `Glucose level (mg/dl)`

## ⚙️ Configuration & Tech Specs

| Parameter | Description |
|-----------|-------------|
| `INPUT_MASTER_ROOT` | Path containing Step 07 outputs. |
| `OUTPUT_ROOT` | Target directory for the engineered dataset. |
| `IR_BASE_FEATURES` | List defining Tier 1 extraction. |
| `ENGINEERED_FEATURES` | List defining Tier 2 mathematical operations. |

## 📥 I/O Format Specification

| Input | Output |
|-------|--------|
| `MASTER_Dataset.csv` (Sample: 10 rows × 32 cols) | Engineered Dataset (Sample: 10 rows × 25 cols) |
| `BuildLog.json` | Updated logs |

## 🚀 Quick Start

**Prerequisites:**
```bash
pip install pandas numpy
```

**Run the Script:**
```bash
python Data_set_with_24_Features_creation_08.py
```

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Missing BuildLog** | Script cannot find `BuildLog.json` from Step 07. | Ensure `INPUT_MASTER_ROOT` points to a valid Step 07 output dir. |
| **Verification Failure** | Column count does not equal 25. | Check the input master dataset for missing features. |
| **Feature Name Error** | Column names in the master dataset have changed. | Verify headers in `MASTER_Dataset.csv` match configuration lists. |