# 🗄️ Data Storage Architecture

> Central data dictionary and storage architecture for the GlucoSense PPG pipeline.

This repository defines the centralized data storage schema, ensuring data consistency, immutability of raw recordings, and standardizing the I/O for all processing steps.

## 📂 Storage Tiers

The data is organized into five distinct tiers:

| Tier | Description | Mutability |
|------|-------------|------------|
| `Raw/` | Unprocessed 18-bit ADC values directly from the sensor. | **Immutable** |
| `Cleaned/` | Normalized, filtered, and baseline-corrected data. | Regenerable |
| `Windowed/` | Data segmented into discrete 15-second windows. | Regenerable |
| `_Features_/` | Extracted physiological features (CSV format) for ML. | Regenerable |
| `Normal_PPG_Only_Data_Set_For_Practice/` | Healthy baseline dataset used for algorithm calibration and practice. | Reference |

## ⚙️ Technical Specifications

- **File Naming Convention:** ISO 8601 standard (`YYYY-MM-DD_HH-MM-SS_<type>.csv`)
- **Storage Footprint:** ~1-5 MB per minute of raw data recording.
- **Recommended Workspace:** ≥10 GB for full pipeline execution.
- **Version Control:** Large CSV files are excluded via `.gitignore` to prevent repository bloat.

## 📜 Data Management Guidelines

1. **Never modify `Raw/` data:** Any corrections must be applied programmatically and saved to `Cleaned/`.
2. **Reproducibility:** `Cleaned/`, `Windowed/`, and `_Features_/` directories can be fully reconstructed from `Raw/` using the pipeline scripts.
3. **Data Purging:** Stale outputs in intermediate tiers are automatically purged before generating new outputs.
