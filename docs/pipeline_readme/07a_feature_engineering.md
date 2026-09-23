# Data_set_with_24_Features_creation_08.py

> Transforms the raw 30+ feature MASTER dataset (from Step 7) into a curated 24-feature ML-ready dataset using a three-tier feature engineering approach: IR base features + selective RED/Ratio engineering + Ensemble metric.

---

## TL;DR

This Python tool is **Step 8** in the pipeline. It reads the MASTER dataset produced by Step 7 and applies scientifically-justified feature engineering to reduce 30+ features down to exactly 24 carefully chosen features (plus 1 glucose target). The output is an optimized dataset that eliminates redundancy while preserving inter-wavelength physiological information critical for glucose estimation.

**Quick Stats:**
- ~700 lines of Python code
- Single-file processing (one MASTER CSV → one engineered CSV)
- Three-tier feature classification system
- 18 IR base + 5 engineered + 1 ensemble + 1 target = 25 columns total
- Auto-detection of latest Step 7 batch folder
- Cross-pipeline traceability via Step 7 build log integration
- Floating-point integrity verification (1e-9 tolerance)
- Timestamped output folders (preserves run history)

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Quick Start](#quick-start)
3. [Background & Motivation](#background--motivation)
4. [The Three-Tier System](#the-three-tier-system)
5. [Tool Overview](#tool-overview)
6. [Features & Capabilities](#features--capabilities)
7. [Installation & Prerequisites](#installation--prerequisites)
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

# 3. Install dependencies
pip install pandas numpy

# 4. Open the script and set TWO paths at the top:
#    INPUT_MASTER_ROOT = r"path/to/07_Final_Data_Set"
#    OUTPUT_ROOT       = r"path/where/24F/dataset/saves"

# 5. Run the script
python Data_set_with_24_Features_creation_08.py

# 6. A popup asks: Pick a MasterDataset_* folder
# 7. Script auto-finds the *_MASTER_Dataset.csv inside
# 8. Wait for processing
# 9. Find timestamped output folder with CSV + JSON log
```

### Expected First Run Output

```
======================================================================
🔧 STEP 7: FEATURE ENGINEERING PIPELINE
   IR Base (18) + Selective RED/Ratio (5) + Ensemble (1) + Target (1)
   Total: 24 Features + 1 Target = 25 Columns
======================================================================

🔍 Scanning for latest Step 6 batch folder...
✅ LATEST: MasterDataset_06_Averaged_Features_2026-06-21_14-35-42
   Last modified: 2026-06-21 14:35:45
   Has MASTER CSV: ✅ Yes

📂 Opening folder selector...
[User selects the MasterDataset folder via popup]

✅ Auto-detected MASTER CSV: 06_Averaged_Features_MASTER_Dataset.csv
✅ Loaded master dataset: 06_Averaged_Features_MASTER_Dataset.csv
   📊 Shape: 10 rows × 32 columns

✅ File validation passed.
✅ Found Step 6 build log: 06_Averaged_Features_DataPipeline_BuildLog.json
✅ Column validation passed: All 30 required columns found.

🔧 FEATURE ENGINEERING PIPELINE
📌 TIER 1: Extracting 18 IR base features...
📌 TIER 2: Computing 5 engineered features...
📌 TIER 3: Keeping 1 as-is features...
📌 TARGET: Appending Glucose level (mg/dl)...

✅ ALL CHECKS PASSED

💾 Saved engineered dataset: Master_Dataset_With_24F_2026-06-21_14-40-15.csv
💾 Saved engineering log: Master_Dataset_With_24F_2026-06-21_14-40-15.json

✅ Feature engineering pipeline completed successfully!
```

### Common First-Time Issues

| Problem | Quick Fix |
|---|---|
| "Filename does not match MASTER pattern" | Pick `*_MASTER_Dataset.csv`, not `*_Final_Data.csv` |
| "CSV has only 1 row" | You selected a single-subject file instead of MASTER |
| "Missing required columns" | Step 7 didn't complete or produced wrong output |
| Step 6 build log not found | Step 6 was run in single mode (proceeds anyway) |

---

## Background & Motivation

### Why Feature Engineering?

A common mistake in ML is "more features = better model." In reality, raw datasets often contain:

- **Redundant features** — Multiple columns measuring the same thing
- **Highly correlated features** — Cause multicollinearity in models
- **Low-value features** — Add noise without predictive power
- **Untransformed features** — Need mathematical manipulation to be useful

For glucose estimation from PPG, our raw MASTER dataset has 30+ features per subject. Many are duplicates between RED and IR channels (e.g., `Red_Skewness` vs `IR_Skewness`) that measure essentially the same physiological quantity at different wavelengths.

### The Three Core Questions

This tool answers three scientific questions for each feature:

1. **Which features carry unique information?** → Keep them (IR base)
2. **Which are redundant duplicates?** → Drop them (most RED features)
3. **Which need transformation to be useful?** → Engineer them (5 ratio/difference features)

### Where This Fits in the Pipeline

```
[Step 7: Data Set Creation]
       |
       v
   30+ feature MASTER dataset
       |
       v
[Step 8: THIS TOOL]              <- Feature engineering
       |
       v
   24 feature engineered dataset
       |
       v
[Step 9: Cleaning + Splitting]
       |
       v
[Step 10: ML Training]
```

This is the **last feature transformation step** before machine learning preprocessing begins. After this, the dataset is conceptually frozen — downstream steps only clean and split, not modify.

### Why Exactly 24 Features?

The number 24 isn't arbitrary. It's the result of:
- **18 IR base features** — All primary PPG channel features
- **5 engineered features** — Inter-wavelength relationships (RED/IR ratios + differences)
- **1 ensemble metric** — Already-combined dual-channel measure

This balance preserves information while eliminating redundancy.

---

## The Three-Tier System

The core of this tool is a **three-tier feature classification system**, where each tier has a specific scientific purpose.

### Tier 1: IR Base Features (18 features) — Direct Copy

The IR (infrared, ~940nm) channel is treated as the **primary physiological reference**.

**Why IR is primary:**
- Deeper tissue penetration (~5mm vs RED's ~3mm)
- Sees larger, more stable blood vessels
- Higher signal-to-noise ratio (SNR)
- Less sensitive to skin pigmentation
- Less affected by motion artifacts
- Standard channel in clinical pulse oximetry

**All 18 IR features kept:**
```
IR_Skewness, IR_Kurtosis, IR_Shannon Entropy, IR_Spectral Entropy,
IR_pulse width, IR_PPI, IR_systolic amplitude, IR_BPM, IR_HRV,
IR_TEO Mean, IR_TEO std dev, IR_1st_Derivative_Mean,
IR_2nd_Derivative_Mean, IR_2nd_Derivative_Skewness,
IR_Harmonic ratio, IR_Rise time, IR_Decay time, IR_Dicrotic notch
```

These are copied **as-is** from the MASTER dataset.

### Tier 2: Engineered Features (5 features) — Mathematical Transformations

These features capture **inter-wavelength relationships** that cannot be derived from IR alone. They encode how RED and IR signals differ — which carries unique physiological information.

| Feature | Operation | Formula | What It Captures |
|---|---|---|---|
| `Ratio_systolic_amplitude` | Ratio | `Red_systolic_amplitude / IR_systolic_amplitude` | SpO2-related amplitude balance |
| `Ratio_TEO_Mean` | Ratio | `Red_TEO_Mean / IR_TEO_Mean` | Blood optical density difference |
| `Diff_2nd_Derivative_Mean` | Difference | `Red_2nd_Derivative_Mean - IR_2nd_Derivative_Mean` | Arterial stiffness wavelength variation |
| `Diff_Spectral_Entropy` | Difference | `Red_Spectral_Entropy - IR_Spectral_Entropy` | Inter-wavelength frequency complexity |
| `Diff_Dicrotic_notch` | Difference | `Red_Dicrotic_notch - IR_Dicrotic_notch` | Vascular compliance wavelength variation |

**Why these specific 5?**
Each captures a different physiological aspect that's relevant to glucose estimation through the RED-IR relationship, not just one channel alone.

### Tier 3: Keep-As-Is (1 feature) — Already Combined

Some features are already combined metrics that don't need further transformation.

- **`Ensemble ratio`** — Already a dual-channel derived metric. Kept unchanged.

### Plus: The Target Column

**`Glucose level (mg/dl)`** — The ground truth glucose values appended at the end. This is what the ML model will learn to predict.

### Final Composition

```
+─────────────────────────────────────────────────+
│  TIER 1:  18 IR base features                   │
│  TIER 2:   5 engineered features                │
│  TIER 3:   1 keep-as-is feature                 │
│  TARGET:   1 glucose level                      │
│  ─────────────────────────────────────────────  │
│  TOTAL:   25 columns (24 features + 1 target)   │
+─────────────────────────────────────────────────+
```

### Suggested Diagram to Create

```
DIAGRAM 1: Three-Tier System Visualization

Create a horizontal layered diagram showing:
  - LEFT: Original MASTER dataset (30+ feature columns)
  - CENTER: This tool's three-tier processing
    * Tier 1 box (BLUE): 18 IR features → direct copy
    * Tier 2 box (ORANGE): 5 engineered features → math operations
    * Tier 3 box (GREEN): 1 ensemble feature → direct copy
    * Target box (RED): 1 glucose value
  - RIGHT: Output dataset (25 columns)
  - Arrows showing column transformations
  - Color coding:
    * BLUE: IR features (primary reference)
    * ORANGE: Engineered features (transformations)
    * GREEN: Combined metrics
    * RED: Target variable
    * GRAY (struck through): Dropped redundant features
  - Tool suggestion: draw.io or PowerPoint
  - Size: Landscape, 1920x1080
```

---

## Tool Overview

### Folder-Based Selection

Unlike previous pipeline tools that ask for individual files, this tool asks you to **pick a Step 7 batch folder**. The script then automatically finds the `*_MASTER_Dataset.csv` inside.

This is intentional because:
- Step 7 outputs go into timestamped folders
- Each folder contains ONE MASTER CSV
- Picking the folder is more user-friendly than navigating to the deep file
- Allows automatic discovery of associated build log

### Auto-Detection Features

The tool automatically:
- Scans `INPUT_MASTER_ROOT` for `MasterDataset_*` folders
- Identifies the most recently modified one
- Shows you which folder it found (informational)
- Finds the `*_MASTER_Dataset.csv` inside your selected folder
- Locates the matching `*_DataPipeline_BuildLog.json` for traceability

### Three-Tier Feature Processing

For each row in the MASTER dataset, the tool:
1. **Copies** the 18 IR base features directly
2. **Computes** the 5 engineered features using ratio/difference math
3. **Copies** the 1 Ensemble ratio feature directly
4. **Appends** the glucose target value

### Data Integrity Verification

After processing, the tool verifies:
- Row count is preserved (same as input)
- Column count equals 25 (24 features + 1 target)
- IR base feature values match the source exactly (float comparison)
- Engineered feature values match recomputed expected values
- Glucose target values are preserved exactly

Any verification failure aborts the pipeline with a clear error message.

### Cross-Pipeline Traceability

The tool reads Step 7's `DataPipeline_BuildLog.json` and embeds key information in its own JSON output. This creates a **traceable chain** showing:
- Which Step 7 run produced the input
- How many subjects went into the MASTER dataset
- Whether row counts match between Step 7 log and current input

---

## Features & Capabilities

### Core Functionality
- **Three-tier feature engineering** — Scientifically justified feature selection
- **Auto-detection of latest batch folder** — Helps users find recent outputs
- **Smart MASTER CSV finder** — Auto-locates the correct file in selected folder
- **Cross-pipeline traceability** — Reads and embeds Step 7 build log

### Validation & Quality
- **File pattern validation** — Confirms selected file is a MASTER dataset
- **Row count validation** — Rejects single-subject files
- **Column validation** — Verifies all required source columns exist
- **Integrity verification** — Confirms no data corruption with 1e-9 tolerance

### Output & Logging
- **Timestamped output folders** — Each run creates a new folder (preserves history)
- **Comprehensive JSON log** — Documents every feature decision and physiological rationale
- **File replacement tracking** — Reports if any files were overwritten

### Error Handling
- **Continue-on-warning** — Non-critical issues logged but don't stop execution
- **Abort-on-error** — Critical failures stop execution with clear messages
- **Helpful error hints** — Tells you exactly what's wrong and how to fix it

---

## Installation & Prerequisites

### System Requirements

| Requirement | Recommended |
|---|---|
| **Python** | 3.10+ |
| **OS** | Windows 10/11, Linux, macOS |
| **RAM** | 4 GB minimum |
| **Disk Space** | Negligible (small CSV + JSON outputs) |

### Required Python Packages

```
pandas >= 2.0.0       # DataFrame operations, CSV I/O
numpy >= 1.24.0       # Numerical operations and NaN handling
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
pip install pandas numpy
```

### Tkinter on Linux

Tkinter is included with Python on Windows/macOS by default. On some Linux distributions:

```bash
sudo apt-get install python3-tk     # Ubuntu/Debian
```

---

## Input Data Format

### Expected Folder Structure (from Step 7)

The tool expects the output of **Step 7: Data Set Creation**:

```
07_Final_Data_Set/
└── MasterDataset_06_Averaged_Features_2026-06-21_14-35-42/    <- You select this folder
    ├── (per-subject CSVs)
    ├── 06_Averaged_Features_MASTER_Dataset.csv                <- Auto-found
    └── 06_Averaged_Features_DataPipeline_BuildLog.json        <- Auto-found
```

### Sample MASTER CSV Structure

The input MASTER CSV should have this structure (simplified example):

```csv
Red_Skewness,IR_Skewness,Red_Kurtosis,IR_Kurtosis,...,Ensemble ratio,Glucose level (mg/dl)
0.823,0.847,3.215,3.402,...,0.989,95.0
0.812,0.829,3.198,3.412,...,0.992,130.0
0.834,0.851,3.245,3.398,...,0.985,112.0
```

**Key requirements:**
- Multiple rows (one per subject)
- Both `Red_*` and `IR_*` columns for each base feature
- `Ensemble ratio` column
- `Glucose level (mg/dl)` column (the target)

### Required Source Columns

The tool needs ALL of these columns to exist in the MASTER CSV:

**For IR base features (18 columns):**
```
IR_Skewness, IR_Kurtosis, IR_Shannon Entropy, IR_Spectral Entropy,
IR_pulse width, IR_PPI, IR_systolic amplitude, IR_BPM, IR_HRV,
IR_TEO Mean, IR_TEO std dev, IR_1st_Derivative_Mean,
IR_2nd_Derivative_Mean, IR_2nd_Derivative_Skewness,
IR_Harmonic ratio, IR_Rise time, IR_Decay time, IR_Dicrotic notch
```

**For engineered features (5 source pairs):**
```
Red_systolic amplitude  + IR_systolic amplitude
Red_TEO Mean            + IR_TEO Mean
Red_2nd_Derivative_Mean + IR_2nd_Derivative_Mean
Red_Spectral Entropy    + IR_Spectral Entropy
Red_Dicrotic notch      + IR_Dicrotic notch
```

**For keep-as-is and target:**
```
Ensemble ratio
Glucose level (mg/dl)
```

If any required column is missing, the script aborts with a clear list of what's missing.

---

## Output Structure

### Timestamped Output Folder

Each run creates a new folder with a timestamp:

```
08_Data_set_with_24_features/
└── Master_Dataset_With_24F_2026-06-21_14-40-15/    <- Timestamped folder
    ├── Master_Dataset_With_24F_2026-06-21_14-40-15.csv    <- Engineered dataset
    └── Master_Dataset_With_24F_2026-06-21_14-40-15.json   <- Comprehensive log
```

**Folder naming format:**
```
Master_Dataset_With_24F_YYYY-MM-DD_HH-MM-SS
```

This preserves history across multiple runs — each run gets its own folder.

### Output CSV Contents

The output CSV has **exactly 25 columns**:
- 18 IR base features (Tier 1)
- 5 engineered features (Tier 2)
- 1 Ensemble ratio (Tier 3)
- 1 Glucose level (target)

Sample output:

```csv
IR_Skewness,IR_Kurtosis,...,IR_Dicrotic notch,Ratio_systolic_amplitude,Ratio_TEO_Mean,Diff_2nd_Derivative_Mean,Diff_Spectral_Entropy,Diff_Dicrotic_notch,Ensemble ratio,Glucose level (mg/dl)
0.847,3.402,...,0.823,1.024,0.967,-0.123,0.045,-0.012,0.989,95.0
0.829,3.412,...,0.834,1.018,0.972,-0.118,0.052,-0.018,0.992,130.0
```

Row count matches the input MASTER CSV exactly.

### Output JSON Contents

The JSON log is comprehensive and contains:

**Pipeline info:**
- Execution timestamp
- Pipeline name and step

**Step 6 reference section (NEW):**
- Cross-references the Step 7 (Data Set Creation) build log
- Records which subjects were in the source dataset
- Notes if row counts match between Step 7 log and current input

**Batch folder detection:**
- Records what folders were detected
- Identifies which was used

**Dataset transformation summary:**
- Original columns vs engineered columns
- Reduction summary

**Feature composition:**
- Count of features in each tier
- Description of each tier's purpose

**Complete feature mapping:**
- Every feature with its source, operation, formula, and rationale

**Dropped features:**
- List of features removed
- Reasoning for each removal

**Verification results:**
- All integrity checks performed
- Pass/fail status

**Physiological rationale:**
- Scientific justification for the feature engineering decisions

### Where to Paste Sample Output Diagrams

To add visual examples to this README:

1. **Create folder:** `images/` next to your `README.md`
2. **Save sample outputs as PNG files:**
   - `sample_engineered_dataset.png` — Screenshot of output CSV
   - `sample_folder_structure.png` — Output folder tree
   - `sample_terminal_output.png` — Terminal showing successful run
3. **Reference them in the README:**
   ```markdown
   ![Sample Engineered Dataset](images/sample_engineered_dataset.png)
   ```

---

## Usage Examples

### Example 1: Standard Run (After Step 7)

**Scenario:** Step 7 just produced a new MASTER dataset. You want to apply feature engineering.

**Steps:**

```bash
# 1. Run the script
python Data_set_with_24_Features_creation_08.py

# 2. Terminal shows latest detected batch folder
# 3. Popup appears - click OK
# 4. Folder picker opens at INPUT_MASTER_ROOT
# 5. Navigate to the MasterDataset_*_<timestamp> folder
# 6. Click "Select Folder"

# 7. Script auto-finds:
#    - The *_MASTER_Dataset.csv
#    - The *_DataPipeline_BuildLog.json
# 8. Processing completes
# 9. Find output in OUTPUT_ROOT/Master_Dataset_With_24F_<timestamp>/
```

### Example 2: Re-running After Step 7 Update

**Scenario:** You fixed an issue in Step 7 and re-ran it. Now you need to re-process with Step 8.

**Steps:**

```bash
# 1. Run the script
python Data_set_with_24_Features_creation_08.py

# 2. Script detects the newest batch folder automatically
# 3. Pick the newest folder via popup
# 4. New timestamped output folder is created
#    (Old output from previous run stays preserved)
```

### Example 3: Tracing Provenance via JSON

**Scenario:** You want to know which Step 7 run produced a specific Step 8 output.

**Steps:**

```bash
# 1. Open the JSON file in the output folder
# 2. Look in the "step6_pipeline_reference" section
# 3. Find:
#    - step6_build_date: when Step 7 ran
#    - step6_master_dataset_path: which file was used
#    - subjects_successfully_compiled: which subjects were included
#    - full_step6_log_raw: complete Step 7 log preserved
```

### Expected Terminal Output Sample

```
======================================================================
🔧 STEP 7: FEATURE ENGINEERING PIPELINE
======================================================================

🔍 Scanning for latest Step 6 batch folder...
✅ LATEST: MasterDataset_06_Averaged_Features_2026-06-21_14-35-42

📂 Opening folder selector at: C:\...\07_Final_Data_Set
📁 Selected folder: MasterDataset_06_Averaged_Features_2026-06-21_14-35-42

🔎 Searching for MASTER CSV inside selected folder...
✅ Auto-detected MASTER CSV: 06_Averaged_Features_MASTER_Dataset.csv

📥 LOADING MASTER DATASET
✅ Loaded: 10 rows × 32 columns

🔍 VALIDATING SELECTED FILE IS MASTER DATASET
✅ File validation passed.

📋 LOADING STEP 6 BUILD LOG (for traceability)
✅ Found Step 6 build log
   📊 Step 6 Summary:
      Total subjects found    : 10
      Successfully compiled   : 10
      Subjects in MASTER CSV  : 10
   ✅ Row count matches Step 6 successful compilations.

🔍 VALIDATING REQUIRED SOURCE COLUMNS
✅ Column validation passed: All 30 required columns found.

🔧 FEATURE ENGINEERING PIPELINE
📌 TIER 1: Extracting 18 IR base features...
📌 TIER 2: Computing 5 engineered features...
📌 TIER 3: Keeping 1 as-is features...
📌 TARGET: Appending Glucose level (mg/dl)...

🔍 VERIFICATION CHECKS
✅ Row count: 10 (original: 10)
✅ Column count: 25 (expected: 25)
✅ IR base feature values integrity: All match
✅ Engineered feature spot check (row 0): Passed
✅ Target glucose values preserved: All match
✅ ALL CHECKS PASSED

💾 Saved engineered dataset: Master_Dataset_With_24F_2026-06-21_14-40-15.csv
💾 Saved engineering log: Master_Dataset_With_24F_2026-06-21_14-40-15.json

📌 FEATURE ENGINEERING PIPELINE — FINAL SUMMARY
======================================================================
   📥 Input  : 06_Averaged_Features_MASTER_Dataset.csv
      Features: 31
   📤 Output : Master_Dataset_With_24F_2026-06-21_14-40-15.csv
      Features: 24
      Target  : Glucose level (mg/dl)

   📊 Transformation:
      TIER 1 — IR Base features        : 18  (direct copy)
      TIER 2 — Engineered RED/Ratio    :  5  (ratio + difference)
      TIER 3 — Keep-as-is (Ensemble)   :  1  (direct copy)
      TARGET — Glucose level           :  1
      ─────────────────────────────────────
      TOTAL                            : 25  columns

   🗑️  Dropped : 7 redundant RED features
   ✅ Verification : ALL PASSED

✅ Feature engineering pipeline completed successfully!
```

---

## Methodology

### Brief Overview

The tool applies a **three-tier classification system** to transform 30+ raw features into a curated 24-feature set. The methodology is grounded in:
- **Physiological reasoning** — Which channel measures what best
- **Information theory** — Avoid redundant features
- **Statistical practice** — Reduce multicollinearity in ML models

### The Three Tiers (Quick Reference)

**Tier 1 — IR Base (18 features):** Direct copy of all IR channel features. IR is the primary physiological reference due to deeper tissue penetration and higher SNR.

**Tier 2 — Engineered (5 features):** Mathematical transformations that capture inter-wavelength relationships between RED and IR channels.

**Tier 3 — Keep-As-Is (1 feature):** Pre-combined metrics that need no further transformation.

### The 5 Engineered Features Deep Dive

Each engineered feature was selected based on three criteria:
1. **Physiological relevance** to glucose estimation
2. **Inter-wavelength information** not captured by IR alone
3. **Mathematical interpretability** (simple ratio or difference)

#### 1. Ratio_systolic_amplitude

**Formula:** `Red_systolic_amplitude / IR_systolic_amplitude`

**Physiological meaning:** The systolic amplitude ratio is the foundation of pulse oximetry (SpO2) calculation. It captures how blood absorbs RED vs IR light differently.

**Why this transformation:**
- Glucose affects hemoglobin glycation (HbA1c)
- Glycated hemoglobin has different RED vs IR absorption properties
- The amplitude ratio directly captures this absorption difference

**What it adds:** Cannot be derived from IR alone — requires both channels.

#### 2. Ratio_TEO_Mean

**Formula:** `Red_TEO_Mean / IR_TEO_Mean`

**Physiological meaning:** Teager Energy Operator (TEO) measures signal energy. The ratio between channels reflects optical density differences.

**Why this transformation:**
- Glucose changes blood viscosity
- Viscosity changes affect light scattering differently at RED vs IR
- The energy ratio captures this scattering difference

**What it adds:** Inter-wavelength optical density relationship.

#### 3. Diff_2nd_Derivative_Mean

**Formula:** `Red_2nd_Derivative_Mean - IR_2nd_Derivative_Mean`

**Physiological meaning:** The 2nd derivative (acceleration plethysmogram) reflects arterial stiffness. The difference between channels showed the largest inter-wavelength variation in our dataset.

**Why this transformation:**
- Glucose affects arterial wall stiffness
- Different tissue depths (RED vs IR) show different stiffness signatures
- The difference captures depth-dependent stiffness variation

**What it adds:** Depth-resolved vascular elasticity information.

#### 4. Diff_Spectral_Entropy

**Formula:** `Red_Spectral_Entropy - IR_Spectral_Entropy`

**Physiological meaning:** Spectral entropy measures frequency complexity. Differences between channels reveal harmonic structure variations.

**Why this transformation:**
- Glucose affects blood flow dynamics
- Flow changes alter harmonic content differently at each wavelength
- The difference captures inter-wavelength frequency complexity

**What it adds:** Inter-wavelength spectral complexity information.

#### 5. Diff_Dicrotic_notch

**Formula:** `Red_Dicrotic_notch - IR_Dicrotic_notch`

**Physiological meaning:** The dicrotic notch position reflects vascular compliance and blood viscosity — both glucose-affected.

**Why this transformation:**
- The notch appears at slightly different positions in RED vs IR
- Due to different tissue penetration depths
- The position difference captures depth-dependent vascular properties

**What it adds:** Inter-wavelength vascular dynamics information.

### Why Other RED Features Are Dropped

Most RED channel features (e.g., `Red_Skewness`, `Red_Kurtosis`, `Red_BPM`) are **near-duplicates** of their IR counterparts. They measure the same physiological quantity at a different wavelength but provide essentially the same information.

**Consequences of keeping both:**
- **Multicollinearity** — Two highly correlated features confuse ML models
- **Increased dimensionality** — More features, more overfitting risk
- **Reduced generalization** — Model learns redundancy instead of patterns

**The decision:** Keep the IR version (higher quality channel), drop the RED version (redundant). Where RED carries unique information, capture it through engineered features instead.

### The Verification Strategy

After feature engineering, the tool performs 5 integrity checks:

1. **Row count check** — Output rows must equal input rows
2. **Column count check** — Must be exactly 25 (24 features + 1 target)
3. **IR base integrity** — IR values in output must match input within 1e-9 tolerance
4. **Engineered spot check** — Recompute row 0's engineered features and verify
5. **Target preservation** — Glucose values must be preserved exactly

Any failure stops the pipeline and prevents corrupted output.

### Cross-Pipeline Traceability

The tool also reads Step 7's build log (`*_DataPipeline_BuildLog.json`) and embeds key information in its own JSON output. This creates a **traceable provenance chain**:

```
Step 7 Build Log → Step 8 JSON Reference Section
```

The Step 7 reference section includes:
- When Step 7 ran (build date)
- Which subjects were successfully compiled
- The full Step 7 log preserved as-is

This means months later, you can look at a Step 8 JSON and know exactly which Step 7 run produced its input.

---

## Configuration Reference

All configuration is at the top of the script.

### Path Settings

```python
INPUT_MASTER_ROOT = Path(r"C:\...\07_Final_Data_Set")
OUTPUT_ROOT       = Path(r"C:\...\08_Data_set_with_24_features")
```

| Parameter | Description |
|---|---|
| `INPUT_MASTER_ROOT` | Folder containing `MasterDataset_*` subfolders from Step 7 |
| `OUTPUT_ROOT` | Where engineered datasets are saved (folder created if missing) |

### Feature Configuration Lists

#### IR Base Features (18 items)
```python
IR_BASE_FEATURES = [
    "IR_Skewness", "IR_Kurtosis", "IR_Shannon Entropy",
    # ... 15 more
]
```
List of all IR features to copy directly. Change this to add/remove IR base features.

#### Engineered Features (5 entries)
```python
ENGINEERED_FEATURES = [
    ("Ratio_systolic_amplitude", "ratio", "Red_systolic amplitude", "IR_systolic amplitude"),
    # ... 4 more
]
```
Each entry is `(new_name, operation, operand_1, operand_2)`.

**Operations:**
- `"ratio"` — Returns `op1 / op2` (NaN if op2 is zero)
- `"difference"` — Returns `op1 - op2`
- `"keep"` — Just copies op1

#### Keep-As-Is Features
```python
KEEP_AS_IS_FEATURES = ["Ensemble ratio"]
```
Features kept unchanged from the MASTER CSV.

#### Target Column
```python
TARGET_COLUMN = "Glucose level (mg/dl)"
```
The ML target variable. Appended at the end of the output.

### How to Add a New Engineered Feature

To add a new engineered feature, just append to the `ENGINEERED_FEATURES` list:

```python
ENGINEERED_FEATURES = [
    # ... existing entries ...
    ("My_New_Feature", "ratio", "Red_HRV", "IR_HRV"),  # NEW
]
```

The tool will automatically:
- Compute the new feature for every row
- Add it to the output CSV
- Document it in the JSON log
- Include it in the verification spot check

**Note:** Both source columns must exist in the MASTER CSV, otherwise validation will fail.

### Pattern Identifiers

```python
MASTER_FILE_IDENTIFIER = "MASTER_Dataset"
MASTER_BATCH_FOLDER_PREFIX = "MasterDataset_"
```

These constants are used for auto-detection and validation. Don't change unless you've also updated Step 7's naming conventions.

---

## Code Architecture

### File Structure

```
project_root/
├── Data_set_with_24_Features_creation_08.py    <- All code in single file
└── README.md                                     <- This file
```

### Main Imports

```python
import os                             # Path operations
import json                           # JSON I/O for build logs
import traceback                      # Error trace printing
from pathlib import Path              # Modern path operations
from datetime import datetime         # Timestamp generation

import tkinter as tk                  # GUI dialogs
from tkinter import filedialog, messagebox  # Popup folder picker

import numpy as np                    # NaN handling, float comparison
import pandas as pd                   # DataFrame operations, CSV I/O
```

### Function Groups

#### Auto-Detection Helpers
```python
def find_latest_master_batch_folder(root_path):
    """Scans for MasterDataset_* folders, returns the most recent."""

def print_batch_folder_detection_report(detection_result):
    """Prints what was detected to terminal (info only)."""
```

#### Validation Helpers
```python
def validate_is_master_dataset(file_path, df):
    """Confirms selected CSV is a MASTER (not single-subject)."""

def validate_required_columns(df):
    """Checks all source columns exist in MASTER CSV."""
```

#### Cross-Reference Helpers
```python
def try_load_step6_build_log(master_csv_path):
    """Loads Step 7's build log if available alongside CSV."""

def build_step6_reference_section(step6_log, step6_log_path, master_csv_path):
    """Builds structured cross-reference for inclusion in JSON output."""
```

#### File Selection
```python
def popup_folder_selector(initial_dir):
    """Opens GUI dialog for picking the MasterDataset folder."""

def find_master_csv_in_folder(folder_path):
    """Auto-locates the *_MASTER_Dataset.csv inside selected folder."""
```

#### Feature Engineering Core
```python
def compute_engineered_feature(row, operation, operand_1_col, operand_2_col):
    """Computes one engineered feature value for one row."""

def build_engineered_dataset(df):
    """Main feature engineering: builds the 25-column output DataFrame."""
```

#### Verification
```python
def verify_engineered_dataset(engineered_df, original_df, engineering_log):
    """Runs all 5 integrity checks on the engineered output."""
```

#### Output
```python
def build_full_json_log(...):
    """Builds the comprehensive JSON log with all sections."""

def save_outputs(engineered_df, json_log, output_folder, timestamp_str):
    """Saves CSV + JSON to timestamped output folder."""
```

#### Orchestration
```python
def main():
    """
    Top-level orchestrator:
    1. Validate paths
    2. Auto-detect latest batch folder (info)
    3. Get folder via popup
    4. Find MASTER CSV in folder
    5. Load and validate
    6. Load Step 7 build log
    7. Validate required columns
    8. Build engineered dataset
    9. Verify integrity
    10. Build and save outputs
    """
```

### Data Flow Diagram

```
+-----------------------+      +-----------------------+
| MasterDataset_*       |      | DataPipeline_BuildLog |
| folder (from Step 7)  |      | .json (from Step 7)   |
+-----------+-----------+      +-----------+-----------+
            |                              |
            v                              v
+-----------+-----------+      +-----------+-----------+
| find_master_csv_in_   |      | try_load_step6_       |
| folder()              |      | build_log()           |
+-----------+-----------+      +-----------+-----------+
            |                              |
            v                              |
+-----------+-----------+                  |
| load_master_csv()     |                  |
+-----------+-----------+                  |
            |                              |
            v                              |
+-----------+-----------+                  |
| validate_is_master_   |                  |
| dataset()             |                  |
+-----------+-----------+                  |
            |                              |
            v                              |
+-----------+-----------+                  |
| validate_required_    |                  |
| columns()             |                  |
+-----------+-----------+                  |
            |                              |
            v                              |
+-----------+-----------+                  |
| build_engineered_     |                  |
| dataset()             |                  |
+-----------+-----------+                  |
            |                              |
            v                              |
+-----------+-----------+                  |
| verify_engineered_    |                  |
| dataset()             |                  |
+-----------+-----------+                  |
            |                              |
            +------------------------------+
                          |
                          v
            +-----------------------+
            | build_full_json_log() |
            +-----------+-----------+
                        |
                        v
            +-----------+-----------+
            | save_outputs()        |
            +-----------+-----------+
                        |
            +-----------+-----------+
            |           |           |
            v           v           v
        CSV file    JSON log    Output folder
```

---

## Troubleshooting & Tips

### Common Issues Table

| Symptom | Cause | Fix |
|---|---|---|
| **"Filename does not match MASTER pattern"** | Selected wrong file | Pick `*_MASTER_Dataset.csv`, not `*_Final_Data.csv` |
| **"CSV has only 1 data row"** | Selected single-subject file | Pick MASTER CSV (multiple rows) |
| **"Missing required columns"** | Step 7 didn't produce expected output | Verify Step 7 completed; check column names |
| **"No '*_MASTER_Dataset.csv' file found"** | Selected wrong folder | Pick the timestamped MasterDataset_ folder |
| **"Engineered feature spot check FAILED"** | Float precision edge case | Usually harmless; inspect output CSV |
| **Row count mismatch warning** | Step 7 was re-run after build log saved | Verify which Step 7 run is canonical |
| **"No Step 6 build log found"** | Step 7 ran in single mode | Re-run Step 7 in BATCH mode for full traceability |
| **Permission denied saving** | Output folder not writable | Check folder permissions |
| **Tkinter dialog doesn't open** | python3-tk missing (Linux) | `sudo apt-get install python3-tk` |
| **Output values look strange** | Source data has NaN/zeros | Inspect input CSV for data quality issues |

### Debugging Workflow

#### Step 1: Verify Input Folder Contents

Check the selected folder has both required files:

```python
from pathlib import Path
folder = Path(r"path\to\MasterDataset_..._timestamp")
csvs = list(folder.glob("*MASTER_Dataset*.csv"))
jsons = list(folder.glob("*DataPipeline_BuildLog*.json"))
print(f"MASTER CSVs found: {len(csvs)}")
print(f"Build logs found: {len(jsons)}")
```

You should see exactly 1 of each.

#### Step 2: Verify Required Columns

Run this to check your MASTER CSV has all needed columns:

```python
import pandas as pd
df = pd.read_csv(r"path\to\MASTER_Dataset.csv")
print("Total columns:", len(df.columns))
print("Columns starting with 'Red_':", [c for c in df.columns if c.startswith("Red_")])
print("Columns starting with 'IR_':", [c for c in df.columns if c.startswith("IR_")])
print("Has Ensemble ratio:", "Ensemble ratio" in df.columns)
print("Has Glucose:", "Glucose level (mg/dl)" in df.columns)
```

#### Step 3: Check Output JSON for Details

After a run, open the output JSON to see what was actually done:

```python
import json
with open(r"path\to\output\Master_Dataset_With_24F_..._json", "r") as f:
    log = json.load(f)

# See what features were created
for feat in log["complete_feature_mapping"]:
    print(f"{feat['index']}. {feat['output_column']} ({feat['tier']})")
```

### Best Practices

#### Always Re-run Step 8 After Step 7 Changes
The engineered features depend entirely on Step 7's output. If you re-run Step 7, you MUST re-run Step 8 or you'll have stale engineered data.

#### Don't Manually Edit the Output CSV
The output is mathematically derived from the input. Manual edits break the verification chain and provenance tracking.

#### Use Timestamped Output Folders for History
The timestamped folder naming preserves history. Don't manually rename or delete folders without good reason — they're your audit trail.

#### Verify Row Count Match
Check that the Step 7 build log subject count matches your input row count. A mismatch indicates Step 7 was re-run between log save and now.

#### Add New Features Carefully
When adding to `ENGINEERED_FEATURES`, ensure source columns exist in the MASTER CSV and document your reasoning. The pipeline's strength is its scientific justification.

#### Keep the JSON Log
The JSON output is gold for thesis writing — it documents every decision with rationale. Don't delete output JSONs after pipeline runs.

---

## Next Step in Pipeline

After successfully creating the 24-feature dataset, your output folder contains the input for ML preprocessing:

```
08_Data_set_with_24_features/
└── Master_Dataset_With_24F_..._<timestamp>/
    ├── Master_Dataset_With_24F_..._<timestamp>.csv    <- ★ Input for Step 9 ★
    └── Master_Dataset_With_24F_..._<timestamp>.json
```

### Next Tool: Step 9 - Data Cleaning + Train/Test Split

The 24-feature dataset feeds into **Step 9: Data Cleaning + Train/Test Split**, which:
- Reads this engineered CSV
- Handles missing values (NaN imputation)
- Removes outliers (clipping)
- Splits into train/test sets (stratified)
- Applies RobustScaler normalization
- Saves prepared data ready for ML training

### Subsequent Steps

After Step 9, the pipeline continues with:
- **Step 10:** XGBoost ML Model Training + Evaluation
- **Step 11:** Inference Engine (planned)
- **Step 12:** Dashboard (planned)

**See:** Step 9 README for the next stage.

---

## References

### Software Libraries

1. The pandas development team, "pandas-dev/pandas: Pandas," Zenodo, 2024. [DOI: 10.5281/zenodo.3509134]. Documentation: https://pandas.pydata.org/docs/

2. C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357-362, 2020. [DOI: 10.1038/s41586-020-2649-2]

### Feature Engineering Concepts

3. A. Géron, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, 2nd ed. O'Reilly Media, 2019. (Chapter on feature engineering and dimensionality reduction)

4. M. Kuhn and K. Johnson, *Feature Engineering and Selection: A Practical Approach for Predictive Models*. CRC Press, 2019.

### PPG Physiology (Light Reference)

5. J. Allen, "Photoplethysmography and its application in clinical physiological measurement," *Physiological Measurement*, vol. 28, no. 3, pp. R1-R39, 2007.

---

## Summary

This tool is the **scientific transformation step** of the pipeline. Through careful three-tier feature classification, it transforms a redundant raw dataset into a curated, ML-ready dataset that preserves the most valuable information while eliminating noise.

Key benefits:
- ✅ Scientifically-justified feature selection (every choice documented)
- ✅ Three-tier classification reduces multicollinearity
- ✅ Inter-wavelength engineered features capture unique information
- ✅ Floating-point integrity verification prevents data corruption
- ✅ Cross-pipeline traceability via Step 7 build log integration
- ✅ Timestamped outputs preserve full pipeline history
- ✅ Comprehensive JSON documentation for thesis writing

For best results: trust the three-tier system, don't manually modify outputs, and keep your JSON logs as documentation gold.

Happy engineering!