# 🧬 Step 07: Master Dataset Creation

This module merges per-subject averaged PPG feature CSVs with ground-truth glucose levels from a metadata Excel sheet to produce a unified master dataset. 

**Pipeline Integration:** Consumes Step 06 outputs (Averaged PPG features) and feeds into Step 08 (24-Feature Engineering).

## ✨ Key Features
- **Dual Execution Modes:** Supports SINGLE (interactive GUI prompt for one subject) or BATCH (all subjects) modes.
- **Robust Subject Matching:** Regex-based ID extraction (stripping `_AveFeatures`/`_Features`), with case-insensitive, whitespace-stripped matching against metadata.
- **Data Integrity:** Strict verification using `np.isclose(rtol=1e-9)` for every merged value.
- **Resilient Batching:** Continue-on-failure processing ensures one bad subject doesn't halt the entire batch.
- **Traceability:** Timestamped output directories with comprehensive build logs.

## ⚙️ Configuration & Tech Specs

| Parameter | Description |
|-----------|-------------|
| `INPUT_FEATURES_ROOT` | Directory containing per-subject averaged feature CSVs. |
| `METADATA_FILE_PATH` | Path to the ground-truth Excel metadata sheet. |
| `OUTPUT_ROOT` | Target directory for generated datasets. |
| `METADATA_ID_COLUMN` | Default: `'ID'` |
| `GLUCOSE_COLUMN` | Default: `'Glucose level (mg/dl)'` |

## 📥 I/O Format Specification

| Input | Output |
|-------|--------|
| Per-subject feature CSVs | `Final_Data.csv` (Per-subject) |
| Metadata Excel sheet | `MASTER_Dataset.csv` (Sample: 10 rows × 25 cols) |
| | `BuildLog.json` |

## 🚀 Quick Start

**Prerequisites:**
```bash
pip install pandas numpy openpyxl
```

**Run the Script:**
```bash
python Data_Set_Creation_Code07.py
```
*A GUI popup will prompt to select Single Subject (YES) or Batch Mode (NO).*

## 🏗️ How It Works
1. Scans `INPUT_FEATURES_ROOT` for feature CSVs.
2. Extracts Subject IDs using regex and matches them to the metadata Excel file.
3. Appends the ground-truth glucose level to the feature row.
4. Verifies data integrity post-merge.
5. Saves individual records and concatenates them into a master CSV in a timestamped `MasterDataset_{source}_{timestamp}/` directory.

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| **Subject Match Failure** | ID mismatch between CSV filename and metadata sheet. | Ensure metadata IDs match CSV names (case-insensitive). |
| **Missing Dependency** | `openpyxl` not installed for reading Excel files. | Run `pip install openpyxl`. |
| **Integrity Check Fails** | Floating-point anomalies during merge. | Verify `rtol` parameters or check input CSVs for NaNs. |