# Data_Set_Creation_Code07.py

> Combines per-subject averaged features (from Step 6) with glucose level metadata to produce the final ML-ready dataset for the non-invasive glucose estimation pipeline.

---

## TL;DR

This Python tool is **Step 7** in the pipeline. It takes the averaged feature CSVs produced by Step 6 (one per subject) and merges them with an Excel metadata file containing reference glucose levels. The output is a master dataset where each row represents one subject with all their features plus the target glucose value — ready for training machine learning models.

**Quick Stats:**
- ~430 lines of Python code
- 2 processing modes (SINGLE / BATCH)
- Excel + CSV metadata file support
- Case-insensitive Subject ID matching
- Per-subject + master dataset CSV outputs
- Build log JSON for full traceability
- Pre-existence snapshot tracking for safe re-runs

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Quick Start](#quick-start)
3. [Background & Motivation](#background--motivation)
4. [Tool Overview](#tool-overview)
5. [Features & Capabilities](#features--capabilities)
6. [Installation & Prerequisites](#installation--prerequisites)
7. [Metadata File Requirements](#metadata-file-requirements)
8. [Input Data Format](#input-data-format)
9. [Output Structure](#output-structure)
10. [Usage Examples](#usage-examples)
11. [Methodology](#methodology)
12. [Configuration Reference](#configuration-reference)
13. [Code Architecture](#code-architecture)
14. [Troubleshooting & Tips](#troubleshooting--tips)
15. [Next Step in Pipeline](#next-step-in-pipeline)
16. [References](#references)

---

## Quick Start

### Minimum Steps to Run

```bash
# 1. Create virtual environment
python -m venv ppg_env

# 2. Activate (Windows)
ppg_env\Scripts\activate

# 3. Install dependencies (includes openpyxl for Excel reading)
pip install pandas numpy openpyxl

# 4. Open the script and set THREE paths at the top:
#    INPUT_FEATURES_ROOT = r"path/to/06_Averaged_Features"
#    METADATA_FILE_PATH  = r"path/to/your/metadata.xlsx"
#    OUTPUT_ROOT         = r"path/where/dataset/saves"

# 5. Run the script
python Data_Set_Creation_Code07.py

# 6. A popup asks: Single subject or BATCH?
#    YES    → Pick ONE subject folder
#    NO     → Pick whole BATCH folder (parent of all subjects)
#    CANCEL → Exit

# 7. Wait for processing
# 8. Find your MASTER dataset CSV in the output folder
```

### Expected First Run Output

```
============================================================
🩸 PPG FINAL DATASET BUILDER
   (Features + Glucose Level → Combined Dataset)
============================================================

📂 Loading metadata: C:\...\PPG Meta Data Collection Sheet.xlsx
✅ Loaded metadata: 75 rows, 12 columns

[Popup appears - user picks BATCH mode]

📌 Mode: BATCH
📁 Source: C:\...\06_Averaged_Features

✅ Found 10 folder(s) in source directory
✅ Valid subject folders after filtering: 10
📁 Output folder: C:\...\07_Final_Data_Set\MasterDataset_06_Averaged_Features_2026-06-21_14-35-42

🔄 Processing: Ali(22-enc-12)v1_AveFeatures
   Subject ID: Ali(22-enc-12)v1
   📄 Source CSV: Ali(22-enc-12)v1_AveFeature.csv
   ✅ Metadata matched.
   🩸 Glucose: 95.0 mg/dL
   ✅ Integrity check passed.
   💾 Saved: Ali(22-enc-12)v1_Final_Data.csv

[... more subjects ...]

💾 MASTER dataset saved: 06_Averaged_Features_MASTER_Dataset.csv
   📊 Rows: 10
   📊 Columns: 25

🎉 PIPELINE COMPLETE
```

### Common First-Time Issues

| Problem | Quick Fix |
|---|---|
| "Metadata file not found" | Verify METADATA_FILE_PATH points to correct .xlsx file |
| "No metadata match for Subject ID" | Subject ID in folder doesn't match metadata exactly |
| "Missing required ID column" | Excel file must have 'ID' column (or update METADATA_ID_COLUMN) |
| Import errors | Run `pip install pandas numpy openpyxl` |

---

## Background & Motivation

### Why This Tool Is Critical

Machine learning models need **two things** to learn:
1. **Features** — Input variables (what we measured)
2. **Labels/Targets** — Output variables (what we want to predict)

Up until Step 6, the pipeline has only produced features (PPG signal characteristics). The actual glucose values (the ML targets) live in a separate Excel sheet that was filled in when you collected the data with a reference glucose meter.

**This tool's job:** Bring these two data sources together into a single ML-ready dataset.

### The "Two Halves of the Puzzle"

```
┌────────────────────────────────┐    ┌─────────────────────────────────┐
│ PPG Features (Step 6 output)   │    │ Reference Glucose (Excel sheet) │
├────────────────────────────────┤    ├─────────────────────────────────┤
│ Subject: Ali_v1                │    │ Subject: Ali_v1                 │
│ IR_pulse_width: 0.234          │    │ Glucose: 95 mg/dL               │
│ IR_PPI: 0.823                  │    │                                 │
│ Red_BPM: 72.5                  │    │ Subject: Jamil_v2               │
│ ... (24 features)              │    │ Glucose: 130 mg/dL              │
│                                │    │                                 │
│ Subject: Jamil_v2              │    │ Subject: Majid_v3               │
│ IR_pulse_width: 0.241          │    │ Glucose: 112 mg/dL              │
│ ...                            │    │ ...                             │
└────────────────────────────────┘    └─────────────────────────────────┘
              ↓                                       ↓
              └───────────────┬───────────────────────┘
                              ↓
                  ┌─────────────────────────────┐
                  │  THIS TOOL combines them    │
                  └─────────────────────────────┘
                              ↓
              ┌────────────────────────────────────┐
              │  MASTER Dataset (ML-Ready)         │
              │  Subject_ID | Features | Glucose   │
              │  Ali_v1     | 0.234... | 95        │
              │  Jamil_v2   | 0.241... | 130       │
              │  Majid_v3   | ...      | 112       │
              └────────────────────────────────────┘
```

### Where This Fits in the Pipeline

```
[Step 4: Signal Processing]
       ↓
[Step 5: Feature Extraction]      <- Per-window features
       ↓
[Step 6: Average Features]        <- Per-subject averages (no labels)
       ↓
[Step 7: THIS TOOL]               <- Add glucose labels = ML-ready data
       ↓
[Step 8: Feature Engineering]     <- Add derived features
       ↓
[Step 9+: ML Training]
```

This is the **last data preparation step** before ML processing begins. After this, all downstream code operates on the master dataset rather than per-subject files.

---

## Tool Overview

### Two Operating Modes

The tool starts with a popup dialog asking which mode to use:

**SINGLE Mode (YES button):**
- Process ONE specific subject folder
- Pick the folder via file dialog
- Output: One combined CSV file
- Best for: Testing, debugging, re-processing one subject

**BATCH Mode (NO button):**
- Process ALL subject folders inside a parent directory
- Pick the parent folder via file dialog
- Output: Per-subject CSVs + MASTER dataset + build log
- Best for: Final dataset assembly, full pipeline runs

### Workflow Steps

For each subject the tool:

1. **Finds** the `*_AveFeature.csv` file in the subject folder
2. **Extracts** the subject ID from the folder name
3. **Searches** the metadata for a row matching that ID (case-insensitive)
4. **Combines** the features with the glucose value
5. **Verifies** data integrity (no values corrupted in transit)
6. **Saves** the combined row as a CSV

In BATCH mode, after processing all subjects:
- Compiles all combined rows into a **MASTER dataset CSV**
- Generates a **build log JSON** with full traceability

### Suggested Diagram to Create

```
DIAGRAM 1: Pipeline Integration Diagram

Create a horizontal flow showing:
  - LEFT: 06_Averaged_Features folder with subject subfolders
  - CENTER-LEFT: Excel metadata file
  - ARROW: Both merge in this tool
  - CENTER-RIGHT: This Python tool
  - RIGHT: Output - MASTER dataset CSV + per-subject CSVs + build log
  - Color coding:
    * BLUE: Feature inputs
    * GREEN: Metadata input
    * ORANGE: This tool
    * PURPLE: Outputs (highlight MASTER dataset)
  - Tool suggestion: draw.io or Excalidraw
  - Size: Landscape, 1920x1080
```

---

## Features & Capabilities

### Core Functionality
- **Excel + CSV metadata support** — Reads both `.xlsx` and `.csv` metadata files
- **Smart folder filtering** — Skips non-subject folders (like `Feature_Plots`)
- **Case-insensitive matching** — Handles ID variations (e.g., "Ali_v1" matches "ali_v1")
- **Integrity verification** — Confirms no data corruption during combination
- **Per-subject AND master output** — Gives you both granular files and the combined dataset

### Mode Flexibility
- **Popup-based mode selection** — Yes/No dialog for SINGLE vs BATCH
- **Folder picker integration** — Easy navigation to source folders
- **Cancel anytime** — Graceful exit if user changes mind

### Safety & Traceability
- **Pre-existence snapshot** — Captures existing files before any wipe
- **Timestamped output folders** — BATCH mode creates unique folders per run
- **Build log JSON** — Records every successful subject and every failure
- **Continue-on-failure** — One bad subject doesn't stop the entire batch
- **File replacement tracking** — Reports which files were overwritten

### Validation
- **Required column checks** — Validates metadata has ID and glucose columns
- **Empty file detection** — Skips empty feature CSVs gracefully
- **Multiple match warnings** — Alerts if metadata has duplicate IDs
- **Integrity checks** — Verifies features and glucose values preserved exactly

---

## Installation & Prerequisites

### System Requirements

| Requirement | Recommended |
|---|---|
| **Python** | 3.10+ |
| **OS** | Windows 10/11, Linux, macOS |
| **RAM** | 4 GB minimum |
| **Disk Space** | Negligible (small CSV outputs) |

### Required Python Packages

```
pandas >= 2.0.0       # DataFrame operations, CSV I/O
numpy >= 1.24.0       # Numerical operations and NaN handling
openpyxl >= 3.1.0     # Excel file reading (.xlsx)
tkinter               # GUI dialogs (included with Python)
```

### Installation Steps

#### Step 1: Create Virtual Environment

```bash
python -m venv ppg_env
```

#### Step 2: Activate Environment

**Windows:**
```bash
ppg_env\Scripts\activate
```

**Linux / macOS:**
```bash
source ppg_env/bin/activate
```

#### Step 3: Install Dependencies

```bash
pip install pandas numpy openpyxl
```

### Why openpyxl?

The `openpyxl` package is required because pandas needs it to read Excel `.xlsx` files. Without it, you'll get:

```
ImportError: Missing optional dependency 'openpyxl'.
```

If you only use CSV metadata files, `openpyxl` is not strictly needed, but installing it doesn't hurt.

### Tkinter on Linux

Tkinter is included with Python on Windows/macOS by default. On some Linux distributions, install separately:

```bash
sudo apt-get install python3-tk     # Ubuntu/Debian
sudo dnf install python3-tkinter    # Fedora
```

---

## Metadata File Requirements

This is the most important section to get right. The metadata file is **external to the pipeline** — you create and maintain it manually based on your data collection sessions.

### Supported Formats

- **Excel (.xlsx, .xls)** — Recommended for ease of editing
- **CSV (.csv)** — Simpler, also supported

### Required Columns

Your metadata file MUST contain at least these two columns:

| Column | Default Name | What It Contains |
|---|---|---|
| Subject ID | `ID` | Matches the folder name pattern (without `_Features` suffix) |
| Glucose Level | `Glucose level (mg/dl)` | Reference glucose value from finger-stick meter |

Column names are configurable — see [Configuration Reference](#configuration-reference).

### Sample Metadata Structure

A minimal valid Excel file:

| ID | Glucose level (mg/dl) |
|---|---|
| Ali(22-enc-12)v1 | 95 |
| Jamil(23-enc-46)v2 | 130 |
| Majid(24-mct-59)v2 | 112 |

You can have **additional columns** (age, gender, BMI, etc.) — they'll be preserved but not used by this tool.

### Subject ID Matching Rules

The tool extracts a subject ID from each folder name and looks it up in the metadata:

**Folder name examples and extracted IDs:**

| Folder Name | Extracted Subject ID |
|---|---|
| `Ali(22-enc-12)v1_AveFeatures` | `Ali(22-enc-12)v1` |
| `Jamil(23-enc-46)v2_Features` | `Jamil(23-enc-46)v2` |
| `Subject_001_AveFeatures` | `Subject_001` |

The matching is:
- **Case-insensitive:** `Ali_v1` matches `ali_v1`
- **Whitespace-stripped:** Leading/trailing spaces ignored
- **Exact otherwise:** Special characters like `(`, `)`, `-` must match

### Common ID Mismatches to Avoid

| Folder Name | Metadata ID | Result |
|---|---|---|
| `Ali(22-enc-12)v1` | `Ali(22-enc-12)v1` | ✅ Match |
| `Ali(22-enc-12)v1` | `ali(22-enc-12)v1` | ✅ Match (case-insensitive) |
| `Ali(22-enc-12)v1` | `Ali(22-enc-12) v1` | ❌ Extra space |
| `Ali(22-enc-12)v1` | `Ali(22-enc-12)V1` | ✅ Match (case-insensitive) |
| `Ali(22-enc-12)v1` | `Ali(22-enc12)v1` | ❌ Missing dash |

When in doubt, copy the exact folder name into the metadata ID column.

---

## Input Data Format

### Expected Folder Structure (from Step 6)

The tool expects the output of **Step 6: Average Feature Extraction**:

```
06_Averaged_Features/
├── Ali(22-enc-12)v1_AveFeatures/                <- Subject folder
│   ├── Ali(22-enc-12)v1_AveFeature.csv          <- REQUIRED (this is what we read)
│   ├── Ali(22-enc-12)v1_AveFeature_Config.json  <- Not used here
│   └── Feature_Plots/                            <- Skipped (not a subject folder)
│
├── Jamil(23-enc-46)v2_AveFeatures/
│   ├── Jamil(23-enc-46)v2_AveFeature.csv
│   └── ...
│
└── Majid(24-mct-59)v2_AveFeatures/
    └── ...
```

### What the Tool Reads

For each subject folder, the tool only needs the `*_AveFeature.csv` file. Other files (configs, plots) are ignored.

### Expected CSV Format

The `*_AveFeature.csv` file should have:
- **Single row** of averaged features
- Multiple columns with feature names

Sample input file:

```csv
Red_Skewness,IR_Skewness,Red_Kurtosis,IR_Kurtosis,...,Ensemble ratio
0.823,0.847,3.215,3.402,...,0.989
```

### Smart Folder Filtering

In BATCH mode, the tool automatically filters out folders that don't contain a feature CSV. This means folders like:
- `Feature_Plots/` (subfolder for plots, not a subject)
- Any other non-subject folders accidentally present

...are silently skipped with a notification in the terminal.

---

## Output Structure

The output structure depends on which mode you choose.

### SINGLE Mode Output

```
07_Final_Data_Set/
└── Ali(22-enc-12)v1_Final_Data.csv     <- One combined CSV
```

Just one file with the features and glucose value for the selected subject.

### BATCH Mode Output

```
07_Final_Data_Set/
└── MasterDataset_06_Averaged_Features_2026-06-21_14-35-42/    <- Timestamped batch folder
    ├── Ali(22-enc-12)v1_Final_Data.csv           <- Per-subject files
    ├── Jamil(23-enc-46)v2_Final_Data.csv
    ├── Majid(24-mct-59)v2_Final_Data.csv
    ├── ... (one per successful subject)
    │
    ├── 06_Averaged_Features_MASTER_Dataset.csv   <- ★ THE FINAL DATASET ★
    └── 06_Averaged_Features_DataPipeline_BuildLog.json
```

The folder name format is:
```
MasterDataset_<source_folder_name>_YYYY-MM-DD_HH-MM-SS
```

Each batch run creates a new timestamped folder, so you keep history of previous runs.

### Output File Contents

#### Per-Subject CSV (`*_Final_Data.csv`)

A single-row CSV combining features + glucose:

```csv
Red_Skewness,IR_Skewness,Red_Kurtosis,IR_Kurtosis,...,Ensemble ratio,Glucose level (mg/dl)
0.823,0.847,3.215,3.402,...,0.989,95.0
```

- All columns from input feature CSV preserved
- One extra column added at the end: `Glucose level (mg/dl)`

#### MASTER Dataset CSV (`*_MASTER_Dataset.csv`)

**This is the main output of the entire pipeline up to this point.**

Multi-row CSV with one row per subject:

```csv
Red_Skewness,IR_Skewness,...,Ensemble ratio,Glucose level (mg/dl)
0.823,0.847,...,0.989,95.0
0.812,0.829,...,0.992,130.0
0.834,0.851,...,0.985,112.0
...
```

- One row per successful subject
- All feature columns plus the glucose target column
- Ready for ML training (after Step 8 feature engineering)

#### Build Log JSON (`*_BuildLog.json`)

Full traceability of the build process:

```json
{
    "build_date": "2026-06-21 14:35:42",
    "processing_mode": "batch",
    "source_features_folder": "C:\\...\\06_Averaged_Features",
    "metadata_file_used": "C:\\...\\PPG Meta Data.xlsx",
    "output_folder": "C:\\...\\MasterDataset_...",
    "format_description": "Features Row Vector + Target Glucose Scalar",
    "total_subjects_found": 10,
    "successful_compilations": 9,
    "failed_compilations": 1,
    "master_dataset_path": "C:\\...\\06_Averaged_Features_MASTER_Dataset.csv",
    "successful_subjects": ["Ali(22-enc-12)v1", "Jamil(23-enc-46)v2", ...],
    "failed_subjects": [
        {"subject_id": "Naufar(22-enc-08)v1", "reason": "no_metadata_match"}
    ]
}
```

This log helps you:
- Verify which subjects made it into the dataset
- Diagnose why specific subjects failed
- Track pipeline runs across time
- Reproduce builds later

### Where to Paste Sample Output Diagrams

To add visual examples to this README:

1. **Create folder:** `images/` next to your `README.md`
2. **Save sample outputs as PNG files:**
   - `sample_master_dataset.png` — Screenshot of MASTER CSV in Excel
   - `sample_folder_structure.png` — Output folder tree screenshot
   - `sample_terminal_output.png` — Terminal showing successful run
3. **Reference them in the README:**
   ```markdown
   ![Sample MASTER Dataset](images/sample_master_dataset.png)
   ```

---

## Usage Examples

### Example 1: SINGLE Mode (Process One Subject)

**Scenario:** You re-ran Step 6 for one subject and want to test combination before running the full batch.

**Steps:**

```bash
# 1. Run the script
python Data_Set_Creation_Code07.py

# 2. Popup appears - click YES (single subject mode)

# 3. Folder picker opens
# 4. Navigate to ONE subject folder (e.g., Ali(22-enc-12)v1_AveFeatures)
# 5. Click "Select Folder"

# 6. Tool processes only that subject
# 7. Output: 07_Final_Data_Set/Ali(22-enc-12)v1_Final_Data.csv
```

### Example 2: BATCH Mode (Process All Subjects)

**Scenario:** Step 6 has finished averaging all subjects. You want to build the final MASTER dataset.

**Steps:**

```bash
# 1. Run the script
python Data_Set_Creation_Code07.py

# 2. Popup appears - click NO (batch mode)

# 3. Folder picker opens
# 4. Navigate to PARENT folder (06_Averaged_Features)
# 5. Click "Select Folder"

# 6. Tool processes all subjects + creates MASTER dataset
# 7. Output: 07_Final_Data_Set/MasterDataset_..._<timestamp>/
```

### Example 3: Re-running After Metadata Update

**Scenario:** You found a typo in the metadata Excel and corrected one glucose value.

**Steps:**

```bash
# 1. Save the corrected Excel file

# 2. Re-run the script
python Data_Set_Creation_Code07.py

# 3. Choose BATCH mode

# 4. New timestamped folder is created (old one stays)
#    Old: MasterDataset_..._2026-06-20_10-15-30/
#    New: MasterDataset_..._2026-06-21_14-35-42/

# 5. Updated values now in new MASTER dataset
# 6. (Optional) Delete old timestamped folder if not needed
```

### Expected Terminal Output Sample

```
============================================================
🩸 PPG FINAL DATASET BUILDER
============================================================

📂 Loading metadata: C:\...\PPG Meta Data Collection Sheet.xlsx
✅ Loaded metadata: 75 rows, 12 columns

📌 Mode: BATCH
📁 Source: C:\...\06_Averaged_Features

✅ Found 10 folder(s) in source directory
✅ Valid subject folders after filtering: 10
📁 Output folder: C:\...\MasterDataset_06_Averaged_Features_2026-06-21_14-35-42

────────────────────────────────────────────────────────────
🔄 Processing: Ali(22-enc-12)v1_AveFeatures
   Subject ID: Ali(22-enc-12)v1
   📄 Source CSV: Ali(22-enc-12)v1_AveFeature.csv
   ✅ Metadata matched.
   🩸 Glucose: 95.0 mg/dL
   ✅ Integrity check passed.
   💾 Saved: Ali(22-enc-12)v1_Final_Data.csv
   📊 Shape: 25 columns (24 features + 1 glucose target)

[... 9 more subjects processed ...]

💾 MASTER dataset saved: 06_Averaged_Features_MASTER_Dataset.csv
   📊 Rows: 10
   📊 Columns: 25
   🧾 Build log saved: 06_Averaged_Features_DataPipeline_BuildLog.json

============================================================
📌 PIPELINE EXECUTION SUMMARY
============================================================
✅ Successful: 10
❌ Failed:     0

📊 Total files written:     12
🆕 Files newly created:     12

============================================================
🎉 PIPELINE COMPLETE
============================================================
```

---

## Methodology

### Overall Algorithm

The tool implements a 7-stage processing pipeline:

```
1. LOAD metadata file (Excel or CSV)
2. PROMPT user for processing mode (SINGLE/BATCH)
3. DISCOVER subject folders
4. FOR each subject:
   a. EXTRACT subject ID from folder name
   b. LOAD feature CSV
   c. MATCH subject ID against metadata
   d. COMBINE features + glucose
   e. VERIFY data integrity
   f. SAVE per-subject CSV
5. AGGREGATE all rows into MASTER dataset (batch mode)
6. SAVE MASTER dataset CSV
7. WRITE build log JSON
```

### Stage 1: Metadata Loading

The tool reads the metadata file using pandas:

```python
def load_metadata(metadata_path):
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(metadata_path)
    elif suffix == ".csv":
        df = pd.read_csv(metadata_path)
```

After loading, it validates that required columns exist (`ID` and `Glucose level (mg/dl)`). If missing, the script exits with a clear error message.

### Stage 2: Mode Selection

A Yes/No/Cancel popup dialog determines the mode:

```python
choice = messagebox.askyesnocancel(...)
# YES   → ("single", folder_path)
# NO    → ("batch", folder_path)  
# CANCEL→ exit
```

### Stage 3: Folder Discovery

In SINGLE mode: just one folder is processed.

In BATCH mode: the parent folder is scanned for subfolders, then filtered:

```python
valid_feature_folders = [
    p for p in feature_folders
    if find_feature_csv_in_folder(p) is not None
]
```

This filtering automatically skips non-subject folders like `Feature_Plots`.

### Stage 4: Subject ID Extraction

The tool extracts the subject ID from the folder name:

```python
def derive_subject_id_from_folder(folder_name):
    name = re.sub(r"_AveFeatures$", "", folder_name)  # Remove "_AveFeatures"
    name = re.sub(r"_Features$", "", name)            # Remove "_Features"
    return name.strip()
```

Examples:
- `Ali(22-enc-12)v1_AveFeatures` → `Ali(22-enc-12)v1`
- `Jamil(23-enc-46)v2_Features` → `Jamil(23-enc-46)v2`

### Stage 5: Metadata Matching

The tool searches the metadata for the matching ID:

```python
sid_clean = str(subject_id).strip().lower()
matches = metadata_df[
    metadata_df[METADATA_ID_COLUMN].astype(str).str.strip().str.lower() == sid_clean
]
```

The matching is:
- **Case-insensitive** (`Ali` matches `ali`)
- **Whitespace-stripped** (handles accidental spaces)
- **Exact otherwise** (preserves special characters)

If multiple matches found, the first is used (with a warning).

### Stage 6: Combination

The features and glucose value are combined into a single dict:

```python
def build_combined_row(feature_row, glucose_value):
    combined = {}
    for col, val in feature_row.items():
        combined[col] = val
    combined[GLUCOSE_COLUMN] = glucose_value
    return combined
```

The glucose column is **appended at the end** so the original feature order is preserved.

### Stage 7: Integrity Verification

After combining, the tool verifies no data was corrupted:

```python
def verify_combined_row(combined_row, feature_row, glucose_value):
    # Check every feature value is preserved exactly
    for col, val in feature_row.items():
        if not np.isclose(float(val), float(combined_row[col]), rtol=1e-9):
            feature_mismatches.append(f"{col}: {val} != {combined_row[col]}")
    
    # Check glucose value is preserved
    if not np.isclose(float(glucose_value), float(combined_row[GLUCOSE_COLUMN]), rtol=1e-9):
        glucose_mismatch = f"Glucose: {glucose_value} != {combined_row[GLUCOSE_COLUMN]}"
```

This uses `np.isclose()` with very tight tolerance (1e-9) to handle floating-point precision while catching real errors.

### Stage 8: MASTER Dataset Assembly (BATCH only)

After all subjects are processed, the tool builds the master dataset:

```python
master_df = pd.DataFrame(all_combined_rows)
master_csv_path = output_folder / f"{source_name}_MASTER_Dataset.csv"
master_df.to_csv(master_csv_path, index=False)
```

This single file contains the entire dataset for ML training.

### Why These Design Choices

**Why case-insensitive matching?**
- Subject IDs are typed by humans and prone to case variations
- Preserves your ability to use mixed case (e.g., "Ali_v1") in folders or metadata
- Reduces frustrating "match not found" errors

**Why integrity verification?**
- Catches silent data corruption (e.g., dtype conversion issues)
- Provides confidence that the ML model sees what was extracted
- Adds <100ms per subject — negligible cost for verification

**Why timestamped output folders?**
- Preserves history of dataset builds
- Allows comparing different runs (e.g., after metadata updates)
- Prevents accidental overwrites
- Enables reproducibility documentation

**Why a separate MASTER dataset CSV?**
- Per-subject CSVs are great for inspection
- ML pipeline needs everything in one file
- Master CSV is the **single source of truth** for downstream steps

---

## Configuration Reference

All configurable settings are at the top of the script.

### Path Settings

```python
INPUT_FEATURES_ROOT = Path(r"C:\...\06_Averaged_Features")
METADATA_FILE_PATH  = Path(r"C:\...\PPG Meta Data Collection Sheet.xlsx")
OUTPUT_ROOT         = Path(r"C:\...\07_Final_Data_Set")
```

| Parameter | Description |
|---|---|
| `INPUT_FEATURES_ROOT` | Folder containing `*_AveFeatures` subfolders from Step 6 |
| `METADATA_FILE_PATH` | Path to the Excel/CSV file with subject IDs and glucose values |
| `OUTPUT_ROOT` | Where the final dataset is saved (folder created if missing) |

### Column Name Settings

```python
METADATA_ID_COLUMN = "ID"
GLUCOSE_COLUMN = "Glucose level (mg/dl)"
```

| Parameter | Description |
|---|---|
| `METADATA_ID_COLUMN` | Name of the column in metadata containing subject IDs |
| `GLUCOSE_COLUMN` | Name of the column containing glucose values (also used in output) |

**If your metadata uses different column names**, just update these constants:

```python
# If your Excel has "Subject_ID" instead of "ID":
METADATA_ID_COLUMN = "Subject_ID"

# If your column is named "Blood Glucose":
GLUCOSE_COLUMN = "Blood Glucose"
```

---

## Code Architecture

### File Structure

```
project_root/
├── Data_Set_Creation_Code07.py     <- All code in single file
└── README.md                        <- This file
```

### Main Imports

```python
import os                            # Path operations, folder handling
import re                            # Subject ID extraction from folder names
import json                          # Build log serialization
import shutil                        # Recursive folder deletion
import traceback                     # Error trace printing
from pathlib import Path             # Modern path operations
from datetime import datetime        # Timestamp for output folder names

import tkinter as tk                 # GUI dialogs
from tkinter import filedialog, messagebox  # Popup mode selector + folder picker

import numpy as np                   # NaN handling, float comparison
import pandas as pd                  # Excel/CSV I/O, DataFrame operations
```

### Function Groups

#### Output Naming
```python
def build_batch_folder_name(source_folder_name):
    """Generates timestamped output folder name for batch runs."""
```

#### File Tracking Helpers
```python
def check_existing_file(file_path):
    """Returns dict with existence + size info."""

def snapshot_folder_files(folder_path):
    """Recursively captures all files before wiping folder."""

def report_replaced_files(replaced_list, location_str):
    """Prints clean terminal report of replaced files."""
```

#### Metadata & Feature Helpers
```python
def load_metadata(metadata_path):
    """Loads Excel or CSV metadata file. Validates required columns."""

def derive_subject_id_from_folder(folder_name):
    """Extracts subject ID by removing _AveFeatures/_Features suffix."""

def find_feature_csv_in_folder(folder_path):
    """Locates the *_AveFeature.csv file in a subject folder."""

def load_feature_row(feature_csv_path):
    """Reads the single-row feature CSV and returns first row."""

def match_metadata_row(metadata_df, subject_id):
    """Case-insensitive lookup of subject ID in metadata."""
```

#### Combination & Verification
```python
def build_combined_row(feature_row, glucose_value):
    """Combines features + glucose into single dict."""

def verify_combined_row(combined_row, feature_row, glucose_value):
    """Validates no data corruption with float tolerance."""
```

#### Mode Selection
```python
def popup_selector():
    """Yes/No/Cancel dialog for SINGLE vs BATCH mode."""
```

#### Main Processing
```python
def process_single_feature_folder(feature_folder, metadata_df, output_folder, snapshot_paths_set):
    """
    Processes ONE subject:
    1. Find feature CSV
    2. Extract subject ID
    3. Match metadata
    4. Combine features + glucose
    5. Verify integrity
    6. Save combined CSV
    Returns status dict.
    """
```

#### Orchestration
```python
def main():
    """
    Top-level orchestrator:
    1. Validate paths
    2. Load metadata
    3. Get mode from user
    4. Process subjects
    5. Build MASTER dataset (batch only)
    6. Save build log
    7. Print summary
    """
```

### Data Flow Diagram

```
+-----------------+      +-----------------+
| Metadata File   |      | Subject Folders |
| (Excel/CSV)     |      | (from Step 6)   |
+--------+--------+      +--------+--------+
         |                        |
         v                        v
+--------+------------------------+--------+
| load_metadata()        find_feature_csv()|
+--------+------------------------+--------+
         |                        |
         v                        v
+--------+------------------------+--------+
|       process_single_feature_folder()    |
|   1. derive_subject_id_from_folder()     |
|   2. match_metadata_row()                |
|   3. build_combined_row()                |
|   4. verify_combined_row()               |
+--------+------------------------+--------+
         |
         v
+-----------------+      +-----------------+
| Per-subject     |      | MASTER Dataset  |
| CSVs            | ---> | (combined)      |
+-----------------+      +-----------------+
                                 |
                                 v
                         +-----------------+
                         | Build Log JSON  |
                         +-----------------+
```

---

## Troubleshooting & Tips

### Common Issues Table

| Symptom | Cause | Fix |
|---|---|---|
| **"Metadata file not found"** | Wrong path | Verify `METADATA_FILE_PATH` points to actual file |
| **"Missing required ID column"** | Wrong column name | Update `METADATA_ID_COLUMN` or rename column in Excel |
| **"Missing glucose column"** | Wrong column name | Update `GLUCOSE_COLUMN` or rename column in Excel |
| **"No metadata match for Subject ID"** | ID mismatch between folder and metadata | Compare exactly — copy folder name to metadata ID |
| **"No feature CSV found"** | Subject folder is empty | Re-run Step 6 for that subject |
| **ImportError: openpyxl** | Missing package | `pip install openpyxl` |
| **Tkinter dialog doesn't open** | python3-tk not installed (Linux) | `sudo apt-get install python3-tk` |
| **All subjects skipped in BATCH** | Selected wrong parent folder | Select the parent of subject folders, not a subject folder |
| **Multiple metadata rows warning** | Duplicate IDs in Excel | Remove duplicate rows from metadata |
| **Integrity check fails** | Float precision edge case | Usually harmless; verify by inspecting output CSV |

### Debugging Workflow

#### Step 1: Verify Metadata Loads

Run this in Python to test metadata loading:

```python
import pandas as pd
df = pd.read_excel(r"path\to\your\metadata.xlsx")
print(df.columns.tolist())
print(df.head())
```

You should see:
- All expected column names
- Subject IDs match your folder names
- Glucose values look reasonable

#### Step 2: Test Subject ID Extraction

For a folder name, verify the extracted ID matches your metadata:

```python
import re
folder_name = "Ali(22-enc-12)v1_AveFeatures"
extracted = re.sub(r"_AveFeatures$", "", folder_name)
extracted = re.sub(r"_Features$", "", extracted)
print(f"Extracted ID: '{extracted}'")
# Now check if this exact ID exists in your metadata
```

#### Step 3: Check Folder Structure

Verify the input folder has the expected structure:

```python
from pathlib import Path
folder = Path(r"path\to\06_Averaged_Features")
for sub in folder.iterdir():
    if sub.is_dir():
        csvs = list(sub.glob("*_AveFeature.csv"))
        print(f"{sub.name}: {len(csvs)} CSV(s)")
```

You should see every subject folder containing exactly one `*_AveFeature.csv`.

### Best Practices

#### Maintain Metadata Consistency
- Use the **exact same naming convention** in folders and metadata
- Avoid typos — copy-paste folder names when filling metadata
- Don't include trailing spaces in IDs

#### Validate Before Batch
- Run SINGLE mode on 1-2 subjects first
- Verify the output CSV looks correct
- Then run BATCH for the full dataset

#### Keep Metadata in Version Control
- Even though it's an Excel file, you can track changes
- Use git LFS for binary files
- Document any changes in a changelog

#### Backup Before Major Re-runs
- BATCH mode creates new timestamped folders, but per-subject mode overwrites
- Always backup the MASTER dataset before re-runs
- Build logs help reconstruct what changed

#### Don't Manually Edit Output Files
- The MASTER dataset is derived from features + metadata
- If you need to change values, fix the source and re-run
- Manual edits break the traceability chain

---

## Next Step in Pipeline

After successfully building the MASTER dataset, your output folder contains the foundation for ML training:

```
07_Final_Data_Set/
└── MasterDataset_..._YYYY-MM-DD_HH-MM-SS/
    ├── (per-subject CSVs)
    ├── *_MASTER_Dataset.csv      <- ★ Input for Step 8 ★
    └── *_BuildLog.json
```

### Next Tool: Step 8 - Feature Engineering

The MASTER dataset feeds into **Step 8: Feature Engineering**, which:
- Reads the MASTER CSV
- Computes engineered features (ratios, differences between RED/IR)
- Adds these to expand the feature set
- Saves enriched dataset for ML training

### Subsequent Steps

After Step 8, the pipeline continues with:
- **Step 9 (Sub-task 1&2):** Data Cleaning (NaN handling, outlier removal)
- **Step 9 (Sub-task 3&4):** Train/Test Split + RobustScaler
- **Step 10:** XGBoost ML Model Training + Evaluation

**See:** `Feature_Engineering_README.md` (Step 8) for the next stage.

---

## References

### Software Libraries

1. The pandas development team, "pandas-dev/pandas: Pandas," Zenodo, 2024. [DOI: 10.5281/zenodo.3509134]. Documentation: https://pandas.pydata.org/docs/

2. openpyxl Project, "openpyxl — A Python library to read/write Excel files," 2024. [Online]. Available: https://openpyxl.readthedocs.io/

3. C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357-362, 2020. [DOI: 10.1038/s41586-020-2649-2]

4. The Tk Toolkit Documentation, "tkinter — Python interface to Tcl/Tk," Python Software Foundation. [Online]. Available: https://docs.python.org/3/library/tkinter.html

---

## Summary

This tool is the **last data preparation step** before ML training begins. By merging features (from your signal processing pipeline) with glucose labels (from your data collection metadata), it creates the supervised learning dataset that powers everything downstream.

Key benefits:
- ✅ Excel + CSV metadata support
- ✅ Smart subject ID matching (case-insensitive)
- ✅ Per-subject + master dataset outputs
- ✅ Build log JSON for full traceability
- ✅ Pre-existence file tracking for safe re-runs
- ✅ Integrity verification prevents silent data corruption
- ✅ Continue-on-failure for robust batch processing

For best results: keep your metadata file accurate, use consistent subject ID naming, and verify outputs before proceeding to ML training.

Happy combining!