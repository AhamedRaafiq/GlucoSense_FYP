# Average_Feature_Extraction_Code06.py

> Aggregates per-window feature CSVs into a single averaged feature vector per subject. This is the bridge between per-window analysis (Step 5) and dataset-level processing (Step 7+).

---

## TL;DR

This Python tool reads all the per-window feature CSV files produced by Step 5 (Feature Extraction), then for each subject it computes the column-wise arithmetic mean to produce one averaged feature row per subject. It also generates per-feature variability plots (one PNG per feature) and validates that all windows in a subject used consistent hyperparameters. Output is one folder per subject containing an averaged CSV, a config JSON, and a Feature_Plots subfolder.

**Quick Stats:**
- ~450 lines of Python code
- 2 processing modes (BATCH / SINGLE)
- Per-subject averaging (column-wise arithmetic mean)
- One averaged row + ~20 variability plots per subject
- Cross-window config validation
- Graceful handling of empty subjects (when all windows were rejected upstream)
- Detailed file replacement tracking with old/new size comparison
- Grand total summary across all processed subjects

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Quick Start](#quick-start)
3. [Background & Motivation](#background--motivation)
4. [Tool Overview](#tool-overview)
5. [Features & Capabilities](#features--capabilities)
6. [Installation & Prerequisites](#installation--prerequisites)
7. [Input Data Format](#input-data-format)
8. [Output Structure](#output-structure)
9. [Usage Examples](#usage-examples)
10. [Methodology](#methodology)
11. [Configuration Reference](#configuration-reference)
12. [Code Architecture](#code-architecture)
13. [Troubleshooting & Tips](#troubleshooting--tips)
14. [Next Step in Pipeline](#next-step-in-pipeline)
15. [References](#references)

---

## Quick Start

### Minimum Steps to Run

```bash
# 1. Create virtual environment
python -m venv ppg_env

# 2. Activate (Windows)
ppg_env\Scripts\activate

# 3. Install dependencies
pip install pandas numpy matplotlib

# 4. Open the script and set your paths at the top:
#    INPUT_ROOT_PATH = r"path/to/Step5/Features/output"
#    OUTPUT_ROOT     = r"path/where/averaged/saves"

# 5. Run the script
python Average_Feature_Extraction_Code06.py

# 6. Choose processing mode when prompted:
#    [1] BATCH  - process all subject folders
#    [2] SINGLE - pick one folder via popup

# 7. Wait for processing
# 8. Check output folder
```

### Expected First Run Output

```
======================================================================
📊 AVERAGING FEATURE VALUES (BATCH-AWARE)
======================================================================

======================================================================
  SELECT AVERAGE FEATURE EXTRACTION MODE
======================================================================
  1) BATCH  — Process ALL *_Features folders inside:
              C:\Users\...\05_Features_
  2) SINGLE — Pop up dialog to choose ONE *_Features folder
======================================================================
  Enter choice [1 or 2]: 1

📦 BATCH MODE — found 10 subject folder(s) to process:
   • Ali(22-enc-12)v1_Features
   • Jamil(23-enc-46)v2_Features
   • Majid(24-mct-59)v2_Features
   ...

📦 [1/10] SUBJECT FOLDER
📂 Processing: ...\Ali(22-enc-12)v1_Features
✅ Found 15 window folders
📋 AVERAGED FEATURE TABLE
   Feature                  RED         IR
   Skewness               0.823       0.892
   Kurtosis               3.215       3.402
   ...
✅ Subject done — 15 windows averaged

🎯 GRAND TOTAL SUMMARY
  Subjects processed       : 10
  ✅ Successful            : 9
  ⚠️  Empty (no features)  : 1
  ❌ Failed                : 0
```

### Common First-Time Issues

| Problem | Quick Fix |
|---|---|
| "No *_Features folders found" | Verify `INPUT_ROOT_PATH` matches Step 5 output location |
| All subjects show EMPTY | Check that Step 5 actually produced features (not all rejected) |
| Import errors | Run `pip install pandas numpy matplotlib` |
| Permission denied saving | Check write access to `OUTPUT_ROOT` folder |

---

## Background & Motivation

### Why Per-Subject Averaging?

In the ML pipeline for glucose estimation:

- **Step 5 extracts features** from each 15-second window separately
- Each window produces 1 row × N features (typically 24 features)
- A typical subject has 10-20 valid windows
- So each subject has **15+ feature rows** (per window) but only **1 reference glucose value** (from finger-stick)

For ML training, the standard rule is:
**One feature vector per measurement = One row per subject**

This tool collapses those 15+ window rows into **one averaged row per subject**, ready for ML training.

### Why Averaging (Not Other Methods)?

Window-level features have natural noise from:
- Slight motion between windows
- Heart rate variability across the recording
- Measurement noise

Averaging across windows:
- Reduces this noise (central limit theorem)
- Produces a more stable "representative" feature value
- Maintains the same feature interpretation across subjects

Alternative approaches considered:
- **Median:** More robust to outliers but loses information about distribution
- **Weighted average:** Requires quality scores per window (complexity not worth it)
- **Use best single window:** Subjective and discards data

Arithmetic mean is the simplest, most interpretable choice and standard in research.

### Where This Fits in the Pipeline

```
[Step 4: Signal Processing]
       |
       v
[Step 5: Feature Extraction]    <- N windows × 24 features per subject
       |
       v
[Step 6: THIS TOOL]             <- 1 averaged row × 24 features per subject
       |
       v
[Step 7: Feature Engineering]   <- Adds ratio/difference features
       |
       v
[Step 8: Train/Test Split + Scaling]
       |
       v
[Step 9: ML Model Training]
```

This tool is the **last per-subject processing step** before features get organized into a master dataset. Everything downstream operates on the dataset as a whole, not per-subject.

---

## Tool Overview

### Two Operating Modes

The tool starts with a console prompt asking which mode you want:

**BATCH Mode:**
- Processes ALL `*_Features` folders inside `INPUT_ROOT_PATH`
- Goes through each subject one by one
- Best for processing the entire dataset
- Provides grand total summary at the end

**SINGLE Mode:**
- Opens a folder picker dialog
- You select ONE `*_Features` folder
- Best for testing or re-processing one subject
- Useful after changing parameters in Step 5

### Per-Subject Processing Flow

For each subject folder, the tool:

1. **Scans** for window subfolders (e.g., `*_Win0_Feature`, `*_Win1_Feature`)
2. **Loads** the `_Features_Flat.csv` from each window
3. **Loads** the `_Filtered_Configuration.json` (if present) for validation
4. **Computes** column-wise mean across all loaded rows
5. **Validates** that all windows used consistent hyperparameters
6. **Generates** one variability plot per feature
7. **Writes** outputs to subject-specific output folder

### Three Possible Status Outcomes

After processing each subject, the tool records one of three statuses:

| Status | Meaning | Action |
|---|---|---|
| **SUCCESS** | Features averaged successfully | Outputs written; subject ready for next step |
| **EMPTY** | No valid feature CSVs found | Skipped (likely all windows rejected in Step 5) |
| **FAILED** | Unexpected error during processing | Error logged; batch continues with next subject |

These statuses appear in the grand total summary, helping you identify which subjects need attention.

### Suggested Diagram to Create

```
DIAGRAM 1: Pipeline Position & Data Flow

Create a horizontal flow diagram showing:
  - LEFT: Multiple window feature CSVs (stacked, labeled Win0...WinN)
  - ARROW: "Column-wise mean"
  - CENTER: Single averaged row CSV
  - RIGHT: Output structure (CSV + JSON + Plots folder)
  - Annotation: "N rows × 24 features → 1 row × 24 features"
  - Color coding:
    * BLUE: Input files
    * GREEN: Averaging operation
    * ORANGE: Output files
  - Tool suggestion: draw.io, Excalidraw, or PowerPoint
  - Size: Landscape, 1920x1080
```

---

## Features & Capabilities

### Core Functionality
- **Arithmetic mean averaging** — Column-wise mean across all valid windows
- **Per-feature variability plots** — One PNG per feature showing window-to-window values
- **Cross-window config validation** — Detects hyperparameter mismatches
- **Backward compatibility** — Supports both new (lowercase) and old (uppercase) JSON keys

### Mode Flexibility
- **BATCH mode** — Process all subjects at once
- **SINGLE mode** — Pick one subject via popup dialog
- **Graceful fallback** — Falls back to manual path entry if popup fails
- **Continue-on-failure** — One subject failing doesn't stop the batch

### Quality Awareness
- **Empty subject detection** — Doesn't crash if all windows were rejected upstream
- **Config validation reports** — Warnings logged in JSON output
- **Per-subject summary** — Shows windows used, plots generated, files replaced
- **Grand total report** — Aggregate statistics across all processed subjects

### Output Management
- **Clean folder replacement** — Wipes old output folder for fresh re-runs
- **File replacement tracking** — Shows old vs new file sizes
- **Pre-existence snapshot** — Captures folder state before wiping
- **Organized per-subject folders** — Each subject gets its own output folder

### Visualization
- **Per-feature variability plots** — Visualize window-to-window consistency
- **Average line overlay** — Each plot shows the averaged value as dashed line
- **Sequential window labels** — X-axis shows Win0, Win1, Win2, etc.
- **Auto-scaled axes** — Handles any feature value range

---

## Installation & Prerequisites

### System Requirements

| Requirement | Recommended |
|---|---|
| **Python** | 3.10+ |
| **OS** | Windows 10/11, Linux, macOS |
| **RAM** | 4 GB minimum |
| **Disk Space** | ~50 MB per subject (mostly plots) |

### Required Python Packages

```
pandas >= 2.0.0       # DataFrame operations, CSV I/O
numpy >= 1.24.0       # Numerical computations
matplotlib >= 3.7.0   # Plot generation (Agg backend, no display)
tkinter               # GUI folder picker (usually included with Python)
```

### Critical Note: Matplotlib Backend

This tool uses the **Agg backend** (non-interactive) so plots are generated as PNG files without opening any windows:

```python
import matplotlib
matplotlib.use("Agg")  # MUST be before importing pyplot
import matplotlib.pyplot as plt
```

This is intentional because:
- Tool can process hundreds of plots without GUI overhead
- Works on headless servers (no display needed)
- Much faster than interactive backends

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
pip install pandas numpy matplotlib
```

### Tkinter on Linux

Tkinter is included with Python on Windows/macOS by default. On some Linux distributions, install it separately:

**Ubuntu / Debian:**
```bash
sudo apt-get install python3-tk
```

**Fedora:**
```bash
sudo dnf install python3-tkinter
```

---

## Input Data Format

### Expected Input Folder Structure

The tool expects the output of **Step 5: Feature Extraction**:

```
INPUT_ROOT_PATH/
├── Ali(22-enc-12)v1_Features/                    <- Subject folder
│   ├── Ali(22-enc-12)v1_Win0_Feature/            <- Per-window subfolder
│   │   ├── Ali(22-enc-12)v1_Win0_Feature_Features_Flat.csv      <- REQUIRED
│   │   ├── Ali(22-enc-12)v1_Win0_Feature_Features_Table.csv     <- Not used here
│   │   ├── Ali(22-enc-12)v1_Win0_Feature_Features.json          <- Not used here
│   │   ├── Ali(22-enc-12)v1_Win0_Filtered_Configuration.json    <- OPTIONAL (validation)
│   │   └── Ali(22-enc-12)v1_Win0_Feature_Signal_Overview.png    <- Not used here
│   ├── Ali(22-enc-12)v1_Win1_Feature/
│   │   └── ...
│   └── ... (more windows)
│
├── Jamil(23-enc-46)v2_Features/                  <- Another subject
│   └── ... (same structure)
│
└── ... (more subjects)
```

### Required Files Per Window

For each window subfolder, the tool looks for:

| File | Required | Used For |
|---|---|---|
| `*_Features_Flat.csv` | ✅ Yes | Source of features to average |
| `*_Filtered_Configuration.json` | ⚠️ Recommended | Cross-window validation |
| Other files | ❌ Ignored | Not used by this tool |

If a window subfolder is missing the `_Features_Flat.csv`, it's silently skipped.

### Expected CSV Format

The `*_Features_Flat.csv` files should have:
- **One row** of feature values
- **Columns named:** `Red_<feature>`, `IR_<feature>`, and `Ensemble ratio`

Sample input file:

```csv
Red_Skewness,IR_Skewness,Red_Kurtosis,IR_Kurtosis,...,Ensemble ratio
0.812,0.834,3.123,3.245,...,0.987
```

### Handling of REJECTED Windows

In Step 5, some windows may have been rejected (poor signal quality, not enough beats, etc.). When a window is rejected:
- **No output folder is created** for that window
- The tool simply doesn't find it during scanning
- The subject's average is computed from remaining valid windows

This is **expected behavior** — rejected windows shouldn't pollute the average.

### Subject Folder Naming

Folder names must end with `_Features` for BATCH mode detection:

✅ **Valid:** `Ali(22-enc-12)v1_Features`, `Subject_001_Features`
❌ **Invalid:** `Ali_data`, `Subject001` (won't be detected)

### Window Folder Naming

Window subfolders must contain `_Win<N>` somewhere in the name:

✅ **Valid:** `Ali_Win0_Feature`, `Ali_Win15_Feature_v2`
❌ **Invalid:** `window0`, `data_1` (window index can't be extracted)

The number after `_Win` determines window ordering in the variability plots.

---

## Output Structure

### Per-Subject Output Folder

For each subject processed, the tool creates this structure:

```
OUTPUT_ROOT/
└── Ali(22-enc-12)v1_AveFeatures/                      <- Subject output folder
    ├── Ali(22-enc-12)v1_AveFeature.csv                <- Averaged features (1 row)
    ├── Ali(22-enc-12)v1_AveFeature_Config.json        <- Metadata + validation
    └── Feature_Plots/                                  <- Variability plots
        ├── Red_Skewness.png
        ├── IR_Skewness.png
        ├── Red_Kurtosis.png
        ├── IR_Kurtosis.png
        ├── ...
        └── Ensemble_ratio.png
```

### Output File Contents

#### 1. `*_AveFeature.csv` (the main output)

A single-row CSV with averaged feature values:

```csv
Red_Skewness,IR_Skewness,Red_Kurtosis,IR_Kurtosis,...,Ensemble ratio
0.823,0.847,3.215,3.402,...,0.989
```

**Row count:** Always 1 (one row per subject)
**Column count:** Same as input (typically 24)

This file is the primary output used by Step 7.

#### 2. `*_AveFeature_Config.json`

A structured JSON containing:

```json
{
    "metadata": {
        "base_name": "Ali(22-enc-12)v1",
        "source_features_folder": "...",
        "sampling_rate_fs": 400,
        "window_duration_sec": 15,
        "number_of_windows_used": 15,
        "date_averaged": "2026-06-21 14:35:42"
    },
    "hyperparameters_reference": {
        "...": "(copied from first window's config)"
    },
    "averaging_info": {
        "averaging_method": "Column-wise arithmetic mean",
        "windows_used": ["Win0", "Win1", "Win2", ...],
        "configuration_validation": "No major configuration mismatches detected",
        "validation_warnings": []
    }
}
```

This provides full traceability for the averaged values.

#### 3. `Feature_Plots/` folder

Contains one PNG per feature showing window-to-window variability:

**Sample plot description:**
```
Plot title: "Red_Skewness"
X-axis: Window labels (Win0, Win1, Win2, ..., Win14)
Y-axis: Feature value
Data: Blue dots connected by line (one point per window)
Overlay: Dashed horizontal line at the average value
Legend: "Window Value" + "Average = 0.823456"
```

This is **the visualization output you'd paste your example diagrams into**.

### Where to Paste Your Example Output Diagrams

Once you generate sample plots from running the tool, place them in:

```
your_README_location/
├── README.md
└── images/                          <- CREATE THIS FOLDER
    ├── sample_variability_plot.png  <- Your sample plot here
    ├── sample_folder_structure.png  <- Annotated output tree
    └── sample_terminal_output.png   <- Screenshot of terminal run
```

Then reference them in the README using:
```markdown
![Sample Variability Plot](images/sample_variability_plot.png)
```

You can replace the suggested diagram descriptions later with actual image references.

### File Replacement Tracking

When re-running on an already-processed subject, the tool:
1. **Captures snapshot** of existing files (paths + sizes)
2. **Wipes the output folder** entirely
3. **Recreates** with fresh files
4. **Reports** which files were replaced and the size changes

Example terminal output:

```
♻️ REPLACED 23 EXISTING FILE(S) in:
   📁 C:\...\Ali(22-enc-12)v1_AveFeatures
   ----------------------------------------
   1. [Averaged Features CSV] Ali(22-enc-12)v1_AveFeature.csv
      ↳ Old size: 1.23 KB  →  New size: 1.25 KB
   2. [Averaged Config JSON] Ali(22-enc-12)v1_AveFeature_Config.json
      ↳ Old size: 3.45 KB  →  New size: 3.48 KB
   3. [Plot (Red_Skewness)] Red_Skewness.png
      ↳ Old size: 32.10 KB  →  New size: 32.15 KB
   ...
```

This audit trail is helpful for understanding what changed between runs.

---

## Usage Examples

### Example 1: BATCH Mode (Process All Subjects)

**Scenario:** Step 5 has finished processing all 10 of your subjects. You want to generate averaged features for all of them.

**Steps:**

```bash
# 1. Run the script
python Average_Feature_Extraction_Code06.py

# 2. When prompted, enter: 1

# 3. Wait for processing (typically ~5 seconds per subject)
```

**Expected behavior:**
- Tool finds all `*_Features` folders inside `INPUT_ROOT_PATH`
- Processes them one by one
- Shows per-subject progress
- Displays grand total summary at the end

**Typical run time:** ~1 minute for 10 subjects

### Example 2: SINGLE Mode (Process One Subject)

**Scenario:** You re-ran Step 5 for one specific subject after fixing a parameter. You only want to re-average that one subject.

**Steps:**

```bash
# 1. Run the script
python Average_Feature_Extraction_Code06.py

# 2. When prompted, enter: 2

# 3. A folder picker dialog opens
# 4. Navigate to the subject's _Features folder (e.g., Ali(22-enc-12)v1_Features)
# 5. Click "Select Folder"

# 6. Tool processes only that subject
```

**Use when:**
- Testing parameter changes
- Re-running after data correction
- Quality-checking one specific subject

### Example 3: Re-running After Step 5 Changes

**Scenario:** You changed hyperparameters in Step 5 and need to re-run Steps 5 + 6 entirely.

**Steps:**

```bash
# 1. First, re-run Step 5 to regenerate features with new parameters
python Feature_Extraction_Code05.py

# 2. Then run Step 6 in BATCH mode
python Average_Feature_Extraction_Code06.py
# Enter: 1

# 3. Tool will:
#    - Detect that output folders already exist
#    - Wipe them clean
#    - Recreate with new averaged values
#    - Report all replaced files
```

**Important:** Always re-run Step 6 after re-running Step 5, otherwise the averaged values will be stale.

### Expected Terminal Output Sample

```
======================================================================
📊 AVERAGING FEATURE VALUES (BATCH-AWARE)
======================================================================

  Enter choice [1 or 2]: 1

📦 BATCH MODE — found 10 subject folder(s) to process:
   • Ali(22-enc-12)v1_Features
   • Hammadh(23-enc-08)v2_Features
   • Jamil(23-enc-46)v2_Features
   ...

======================================================================
📦 [1/10] SUBJECT FOLDER
======================================================================

📂 Processing: C:\...\Ali(22-enc-12)v1_Features
📁 Output folder: C:\...\Ali(22-enc-12)v1_AveFeatures
✅ Found 15 window folders

📋 AVERAGED FEATURE TABLE
   Feature                  RED         IR
   Skewness               0.823       0.847
   Kurtosis               3.215       3.402
   pulse width            0.234       0.247
   PPI                    0.823       0.834
   BPM                   72.345      72.567
   ...

📋 ENSEMBLE RATIO
   Feature           Value
   Ensemble Ratio    0.989

✅ Loaded 15 configuration file(s)
✅ Pipeline configuration checked successfully

🖼️  Generating 24 feature plots...
✅ All 24 feature plots saved.

💾 Averaged CSV saved: C:\...\Ali(22-enc-12)v1_AveFeature.csv
💾 Config JSON saved: C:\...\Ali(22-enc-12)v1_AveFeature_Config.json

✅ Subject done — 15 windows averaged

[... 9 more subjects ...]

======================================================================
🎯 GRAND TOTAL SUMMARY
======================================================================
  Subjects processed       : 10
  ✅ Successful            : 9
  ⚠️  Empty (no features)  : 1
  ❌ Failed                : 0

  📊 Total windows averaged    : 142
  🖼️  Total plots generated    : 216
  💾 Total files written       : 218
  ♻️  Total files replaced     : 0
  🆕 Total files newly created : 218
  🗑️  Folders fully wiped      : 0
  📂 Output root               : C:\...\06_Averaged_Features

──────────────────────────────────────────────────────────────────────
📋 PER-SUBJECT BREAKDOWN
──────────────────────────────────────────────────────────────────────
  ✅ Ali(22-enc-12)v1  →  windows: 15, plots: 24, replaced: 0
  ✅ Hammadh(23-enc-08)v2  →  windows: 14, plots: 24, replaced: 0
  ✅ Jamil(23-enc-46)v2  →  windows: 13, plots: 24, replaced: 0
  ⚠️  Naufar(22-enc-08)v1  →  no valid feature CSVs found (likely all rejected)
  ...

======================================================================
🎉 AVERAGING COMPLETE
======================================================================
```

---

## Methodology

### Overall Algorithm

The tool implements a 7-stage processing pipeline for each subject:

```
1. SCAN subject folder for window subfolders
2. LOAD each window's _Features_Flat.csv (and config JSON)
3. AGGREGATE rows into a single DataFrame
4. AVERAGE column-wise (arithmetic mean, NaN-safe)
5. VALIDATE configs across windows (sampling rate, hyperparameters)
6. PLOT each feature's variability across windows
7. SAVE averaged CSV + config JSON + plots
```

### Stage 1: Window Discovery

The tool scans the subject folder for subfolders matching the pattern:

```python
subfolders = [folder for folder in subject_folder if os.path.isdir(folder)]
```

Then sorts them by window index (`Win0`, `Win1`, `Win2`, ...) using the `get_win_index()` helper:

```python
def get_win_index(name):
    if "_Win" in name:
        return int(name.split("_Win")[1].split("_")[0])
    return 999  # Folders without _Win pattern go to the end
```

This ensures plots show windows in chronological order.

### Stage 2: Feature Loading

For each window subfolder:

```python
# Look for the Features_Flat CSV (the source of features)
flat = next((f for f in files if f.endswith("_Features_Flat.csv")), None)

# Load it as a single-row DataFrame
df = pd.read_csv(flat_path)

# Take the first row as the window's feature vector
feature_rows.append(df.iloc[0])
```

If the file is missing or empty, the window is silently skipped.

### Stage 3: Aggregation

All loaded rows are stacked into a single DataFrame:

```python
df_all = pd.DataFrame(feature_rows)
```

If a subject has 15 valid windows with 24 features each, `df_all` becomes a 15×24 matrix.

### Stage 4: Column-wise Mean (the actual averaging)

The core operation:

```python
df_avg_flat = pd.DataFrame([df_all.mean(numeric_only=True)])
```

This:
- Computes the mean of each column (one value per feature)
- Ignores NaN values (NaN-safe)
- Returns a single-row DataFrame

**Why arithmetic mean?**
- Simple and interpretable
- Standard in research literature
- Works well when windows have similar quality
- Most robust assumption for unknown distribution

### Stage 5: Config Validation (Cross-Window Consistency)

For ML training to be valid, all windows from a subject should be processed with **the same hyperparameters** in Step 5. The tool checks this:

```python
# Take first window's config as reference
ref = configs[0]
ref_metadata = get_metadata_from_config(ref)
ref_hyperparameters = get_hyperparameters_from_config(ref)

# Compare all other configs against the reference
for i, c in enumerate(configs[1:], start=1):
    if c_hyperparameters != ref_hyperparameters:
        warnings.append(f"Hyperparameters mismatch in config {i}")
```

Specific checks:
- **Sampling rate** matches across windows
- **Window duration** matches across windows
- **Hyperparameter dict** matches across windows

Any mismatches are logged as warnings in the output JSON. The averaging still proceeds (warnings are informational), but you should investigate before training ML models on inconsistent data.

### Stage 6: Variability Plots

For each feature, the tool generates a PNG showing window-to-window variability:

```python
for feat in df_all.columns:
    y = df_all[feat].values         # Values from all windows
    avg = np.nanmean(y)             # The averaged value
    
    plt.plot(x, y, marker="o", label="Window Value")
    plt.axhline(avg, linestyle="--", label=f"Average = {avg:.6f}")
    plt.xticks(x, win_labels)       # Win0, Win1, ...
    plt.savefig(plot_file_path, dpi=150)
```

**How to interpret these plots:**
- **Stable line:** Feature is consistent across windows → reliable average
- **Wildly varying line:** Feature is noisy → average may not be representative
- **Outlier window:** One window much higher/lower → may indicate that window had quality issues

These plots are **invaluable for quality control** — if a feature looks unstable for a subject, investigate that subject's signal quality before relying on the averaged value.

### Stage 7: Output Writing

Three outputs are written:

#### 7a. Averaged CSV
```python
df_avg_flat.to_csv(output_csv_path, index=False)
```
Single-row CSV ready for Step 7.

#### 7b. Config JSON
```python
with open(config_path, "w") as f:
    json.dump(average_config, f, indent=4)
```
Contains metadata, hyperparameters reference, and validation warnings.

#### 7c. Plot Folder
```python
plot_paths = {feat: path for feat in features}
# (each plot already saved during Stage 6)
```
One PNG per feature, organized in `Feature_Plots/` subfolder.

### Design Decision: Why Arithmetic Mean (vs Alternatives)

| Method | Pros | Cons | Chosen? |
|---|---|---|---|
| **Arithmetic Mean** | Simple, standard, interpretable | Sensitive to outliers | ✅ Yes |
| **Median** | Robust to outliers | Less informative about distribution | ❌ No |
| **Trimmed Mean** | Robust + interpretable | Adds complexity, edge cases | ❌ No |
| **Weighted Mean** | Could prioritize high-quality windows | Requires quality scores (out of scope) | ❌ No |

Arithmetic mean was chosen for simplicity and alignment with research conventions.

### Design Decision: Why Plot Every Feature

While 24 plots per subject seems excessive, they serve critical purposes:
- **Quality control:** Spot outlier windows immediately
- **Subject characterization:** Compare variability patterns across subjects
- **Debugging:** When a model performs poorly on a subject, plots help identify why
- **Documentation:** Visual record of what the averaging produced

The PNG files are small (~30 KB each), so disk overhead is negligible.

---

## Configuration Reference

All configurable settings are at the top of the script.

### Path Settings

```python
INPUT_ROOT_PATH = r"C:\Users\...\05_Features_"
OUTPUT_ROOT    = r"C:\Users\...\06_Averaged_Features"
```

| Parameter | Default | Description |
|---|---|---|
| `INPUT_ROOT_PATH` | Step 5 output path | Folder containing `*_Features` subject folders from Step 5 |
| `OUTPUT_ROOT` | Step 6 output path | Where averaged outputs will be saved (folder created if missing) |

### Internal Behaviors (Not Configurable, But Important)

#### Folder Replacement Logic
When an output folder already exists, the tool:
1. Captures snapshot of existing files
2. Wipes the entire output folder
3. Recreates with fresh content
4. Reports all replacements

This ensures no stale files remain from previous runs.

#### File Naming Patterns

| Pattern | Example |
|---|---|
| Subject output folder | `{subject_name}_AveFeatures` |
| Averaged CSV | `{subject_name}_AveFeature.csv` |
| Config JSON | `{subject_name}_AveFeature_Config.json` |
| Plot file | `{feature_name_safe}.png` (special chars → `_`) |

#### Backward Compatibility Flags

The tool supports both old and new config JSON formats:

**New format (from updated Step 5):**
- Keys: `metadata`, `hyperparameters`, `extracted_features`
- Sampling rate key: `sampling_rate_fs`
- Window duration key: `duration_sec`

**Old format (legacy):**
- Keys: `Metadata`, `Processing_Settings`, `Ensemble_Method`
- Sampling rate key: `Sampling_Rate_FS`
- Window duration key: `Plot_Duration_Sec`

The `get_*_from_config()` helper functions try both naming conventions automatically.

---

## Code Architecture

### File Structure

```
project_root/
├── Average_Feature_Extraction_Code06.py    <- All code in single file
└── README.md                                <- This file
```

### Main Imports

```python
import pandas as pd                          # DataFrame operations, CSV I/O
import numpy as np                           # Numerical operations
import os                                    # File path handling, folder operations
import json                                  # Config JSON serialization
import shutil                                # Recursive folder deletion
import traceback                             # Error trace printing
import tkinter as tk                         # GUI folder picker
from tkinter import filedialog               # File dialog widget
import matplotlib                            # Plotting library
matplotlib.use("Agg")                        # MUST be before pyplot import
import matplotlib.pyplot as plt              # Plot generation
from datetime import datetime                # Timestamp generation
from pathlib import Path                     # Modern path operations
```

### Function Groups

#### Helpers — File Tracking

```python
def check_existing_file(file_path):
    """Check if file exists and return size info."""

def report_replaced_files(replaced_list, output_folder_path_str):
    """Print clean terminal report of replaced files."""
```

#### Helpers — Config Detection

```python
def load_json_safe(path):
    """Load JSON file, return None on failure."""

def is_structured_config(cfg):
    """Detect structured config JSON (supports old + new formats)."""

def is_flat_feature_json(cfg):
    """Detect flat feature JSON format."""

def get_metadata_from_config(cfg):
    """Extract metadata dict (supports old + new keys)."""

def get_sampling_rate(metadata):
    """Extract sampling rate (supports old + new keys)."""

def get_window_duration(metadata):
    """Extract window duration (supports old + new keys)."""

def get_hyperparameters_from_config(cfg):
    """Extract hyperparameters dict."""

def get_win_index(name):
    """Parse window index from folder name like 'Subject_Win5_Feature'."""
```

#### Mode Selection

```python
def prompt_processing_mode():
    """Console prompt for BATCH vs SINGLE mode."""

def collect_features_folders(mode):
    """Returns list of *_Features folders based on chosen mode."""
```

#### Main Processing

```python
def process_single_features_folder(input_features_folder):
    """
    Processes ONE subject folder:
    1. Sets up output paths
    2. Captures pre-existing file snapshot
    3. Wipes & recreates output folder
    4. Scans for window subfolders
    5. Loads features from each window
    6. Computes column-wise mean
    7. Validates configs across windows
    8. Generates per-feature variability plots
    9. Saves CSV + JSON + plots
    10. Returns status dict (SUCCESS/EMPTY/FAILED)
    """
```

#### Orchestration

```python
def main():
    """
    Top-level orchestrator:
    1. Validates input root path
    2. Prompts for mode
    3. Collects folders to process
    4. Iterates through each subject
    5. Catches per-subject errors
    6. Prints grand total summary
    """
```

### Data Flow Diagram

```
+-------------------+      +-------------------+
| Input: many       |      | Output: one row   |
| windowed feature  | ---> | per subject       |
| CSVs (15 windows) |      | (averaged)        |
+-------------------+      +-------------------+
        |                            ^
        v                            |
+-------------------+      +-------------------+
| pandas DataFrame  | ---> | df.mean()         |
| (stacked rows)    |      | column-wise       |
+-------------------+      +-------------------+
        |                            |
        v                            v
+-------------------+      +-------------------+
| Per-feature plots |      | CSV + JSON +      |
| (variability)     |      | Feature_Plots/    |
+-------------------+      +-------------------+
```

### Why Single-File Design

The entire tool is in one file because:
- **Simplicity:** Easy to share, copy, deploy
- **No package complexity:** No imports across files
- **Single-purpose:** All code serves the averaging task
- **Manageable size:** ~450 lines is still readable

### Suggested Diagram to Create

```
DIAGRAM 2: Function Flow Diagram

Create a flowchart showing:
  - TOP: main() entry point
  - Branch to mode selector (BATCH vs SINGLE)
  - Both branches converge at process_single_features_folder()
  - Show the 10 internal stages of that function
  - End with grand total summary
  - Color coding:
    * BLUE: User interaction
    * GREEN: Data loading
    * ORANGE: Computation
    * PURPLE: Output writing
  - Tool: draw.io or Mermaid flowchart
  - Size: Portrait, 1080x1920
```

---

## Troubleshooting & Tips

### Common Issues Table

| Symptom | Cause | Fix |
|---|---|---|
| **"No *_Features folders found"** | Wrong INPUT_ROOT_PATH | Verify path matches Step 5 output location exactly |
| **All subjects show "EMPTY"** | All windows rejected in Step 5 | Check Step 5 signal quality; may need parameter tuning |
| **Some subjects "EMPTY"** | Specific subjects had bad signal | Acceptable — investigate those subjects' raw data |
| **"Hyperparameters mismatch" warning** | Step 5 was run with different params per subject | Re-run Step 5 with consistent settings for all subjects |
| **"Sampling rate mismatch" warning** | Different windows recorded at different rates | Should not happen — investigate raw data integrity |
| **Plot generation fails** | NaN values in feature | Check upstream feature extraction for issues |
| **Folder permission denied** | OUTPUT_ROOT not writable | Check folder permissions, run as administrator if needed |
| **Tkinter dialog doesn't open (Linux)** | python3-tk not installed | `sudo apt-get install python3-tk` |
| **Import error: matplotlib** | Wrong backend or missing package | Verify `matplotlib.use("Agg")` is before pyplot import |
| **File replacement tracking shows weird sizes** | Disk caching issue | Restart and re-run; sizes should normalize |

### Debugging Workflow

#### Step 1: Verify Input Path

Run a quick check:

```python
import os
INPUT_ROOT_PATH = r"your\path\here"
print("Path exists:", os.path.exists(INPUT_ROOT_PATH))
print("Contents:", [f for f in os.listdir(INPUT_ROOT_PATH) if f.endswith("_Features")])
```

You should see a list of subject folders. If empty, your path is wrong.

#### Step 2: Check Window Folders for One Subject

Pick one subject folder and verify it has window subfolders with the expected files:

```python
subject_folder = r"INPUT_ROOT_PATH\Ali(22-enc-12)v1_Features"
for f in os.listdir(subject_folder):
    full = os.path.join(subject_folder, f)
    if os.path.isdir(full):
        files = os.listdir(full)
        has_flat = any(name.endswith("_Features_Flat.csv") for name in files)
        print(f"{f}: has flat CSV = {has_flat}")
```

Every window subfolder should have `has flat CSV = True`. If not, Step 5 didn't complete properly.

#### Step 3: Inspect Variability Plots

If averaged values look strange:
1. Open the `Feature_Plots/` folder for that subject
2. Look for plots with high variability or outliers
3. Identify which window numbers are problematic
4. Go back to Step 4 (Signal Processing) outputs for those windows
5. Investigate signal quality for those specific windows

### Best Practices

#### Run Step 5 with Consistent Hyperparameters
If hyperparameters change between subjects, the averaged features become incomparable. Always use BATCH mode in Step 5 with locked hyperparameters before running Step 6.

#### Inspect Variability Plots Per Subject
After running this tool, spot-check the variability plots for a few subjects. Consistent features → reliable averages. Variable features → may need quality improvements upstream.

#### Don't Manually Edit Averaged CSVs
The averaged CSV is mathematically derived from window-level features. If you need to change values, fix the source (Step 5 parameters) and re-run, don't edit the output.

#### Use BATCH Mode After Initial Validation
First, run SINGLE mode on 1-2 representative subjects to verify everything works. Then run BATCH mode for the full dataset.

#### Empty Subjects Are Normal
With proper signal quality thresholds in Step 5, some recordings will be rejected entirely. Don't worry if 5-10% of subjects show EMPTY status — that's the quality control working correctly.

#### Re-run Step 6 After Re-running Step 5
The averaged values depend entirely on Step 5's output. If you change Step 5's hyperparameters, you MUST re-run Step 6 or you'll have stale averaged values.

---

## Next Step in Pipeline

After successfully averaging features for all subjects, your output folder contains the foundation for ML training:

```
06_Averaged_Features/
├── Ali(22-enc-12)v1_AveFeatures/
│   ├── Ali(22-enc-12)v1_AveFeature.csv
│   └── ...
├── Jamil(23-enc-46)v2_AveFeatures/
│   └── ...
└── ...
```

### Next Tool: Step 7 - Feature Engineering

The averaged features from this tool feed into **Step 7: Feature Engineering**, which:
- Reads all averaged CSV files from this output
- Combines them into a master dataset (one row per subject, all features)
- Adds engineered features (ratios, differences, interactions)
- Produces the final feature set for ML training

### Subsequent Steps

After Step 7, the pipeline continues with:
- **Step 8:** Data Cleaning + Train/Test Split + Scaling
- **Step 9:** XGBoost ML Model Training + Evaluation

**See:** `Feature_Engineering_README.md` (Step 7) for the next stage.

---

## References

### Software Libraries

1. The pandas development team, "pandas-dev/pandas: Pandas," Zenodo, 2024. [DOI: 10.5281/zenodo.3509134]. Documentation: https://pandas.pydata.org/docs/

2. C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357-362, 2020. [DOI: 10.1038/s41586-020-2649-2]. Documentation: https://numpy.org/doc/stable/

3. J. D. Hunter, "Matplotlib: A 2D graphics environment," *Computing in Science & Engineering*, vol. 9, no. 3, pp. 90-95, 2007. [DOI: 10.1109/MCSE.2007.55]. Documentation: https://matplotlib.org/stable/

### Statistical Concepts

4. R. V. Hogg, J. W. McKean, and A. T. Craig, *Introduction to Mathematical Statistics*, 8th ed. Pearson, 2018. (For arithmetic mean and central limit theorem)

5. The Tk Toolkit Documentation, "tkinter — Python interface to Tcl/Tk," Python Software Foundation. [Online]. Available: https://docs.python.org/3/library/tkinter.html

---

## Summary

This tool performs a critical aggregation step in the glucose estimation pipeline. While conceptually simple (just averaging), it provides essential quality controls and visualizations that catch problems before they propagate into the ML model.

Key benefits:
- ✅ Standard, interpretable averaging method
- ✅ Cross-window consistency validation
- ✅ Per-feature variability visualization for quality control
- ✅ Graceful handling of subjects with no valid windows
- ✅ Detailed file tracking for reproducibility
- ✅ Batch and single mode for flexibility
- ✅ Backward compatibility with old config formats

For best results: run after Step 5 with consistent hyperparameters, inspect variability plots for quality, and don't manually edit the averaged outputs.

Happy averaging!