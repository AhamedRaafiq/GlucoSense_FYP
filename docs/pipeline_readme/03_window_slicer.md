# Step 3: Windowing Tool

> Interactive GUI tool for manually selecting fixed-duration windows from raw PPG recordings, designed as Step 3 in the non-invasive glucose estimation pipeline.

---

## TL;DR

This Python tool provides a graphical interface for manually selecting clean 15-second windows from raw PPG recordings. It displays both RED and IR channels simultaneously, lets you scroll through the recording, click to position a fixed-duration selection window, and saves selected segments as individual CSV files for downstream signal processing.

**Quick Stats:**
- ~270 lines of Python code
- Single-file processing (interactive, one recording at a time)
- Dual-channel visualization (RED + IR)
- Configurable window duration (default: 15 seconds)
- Click-to-position window selection
- Add / Undo / Done workflow
- Outputs one CSV per selected window

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
9. [Interactive GUI Walkthrough](#interactive-gui-walkthrough)
10. [Configuration Reference](#configuration-reference)
11. [Code Architecture](#code-architecture)
12. [Troubleshooting & Tips](#troubleshooting--tips)
13. [Next Step in Pipeline](#next-step-in-pipeline)

---

## Quick Start

### Minimum Steps to Run

```bash
# 1. Create virtual environment
python -m venv ppg_env

# 2. Activate (Windows)
ppg_env\Scripts\activate

# 3. Install dependencies
pip install numpy pandas matplotlib

# 4. Open the script and set your paths at the top:
#    RAW_INPUT_DIR   = r"path/to/raw/csv/folder"
#    BASE_OUTPUT_DIR = r"path/where/windows/save"

# 5. Run the script
python Windowing_Tool.py

# 6. File picker opens — select a raw CSV file
# 7. GUI appears showing RED (top) and IR (bottom)
# 8. Scroll, click to position window, click "Add Block"
# 9. Repeat for as many windows as you want
# 10. Click "Done" — all windows save to output folder
```

### Expected First Run Behavior

```
✅ Matplotlib backend: TkAgg
📂 Opening File Dialog...
[file picker appears - select your CSV]
✅ Loaded: subject_recording.csv | Duration: 120.45s | FS=400 Hz

🧭 Controls:
   - Use SLIDER to scroll
   - CLICK on RED(top) or IR(bottom) plot to place BLUE window start
   - 'Add Block' saves EXACTLY the blue-window segment
   - 'Undo Last' removes last saved window
   - 'Done' closes viewer and saves to files

[GUI window opens]
[user interacts...]

✅ Saved Window: 12.500s - 27.500s
✅ Saved Window: 45.300s - 60.300s
✅ Saved Window: 78.100s - 93.100s

💾 Saving 3 selected windows (each = 15s)...
   [0] Saved: 12.500s - 27.500s -> subject_recording_Win0.csv
   [1] Saved: 45.300s - 60.300s -> subject_recording_Win1.csv
   [2] Saved: 78.100s - 93.100s -> subject_recording_Win2.csv

✅ DONE!
📂 Location: C:\...\03_Windowed\subject_recording
```

### Common First-Time Issues

| Problem | Quick Fix |
|---|---|
| GUI doesn't open | Verify `matplotlib.use("TkAgg")` is set before pyplot import |
| File picker doesn't appear | Install tkinter: `sudo apt-get install python3-tk` (Linux) |
| "Required columns missing" | Check CSV has IR and RED columns (any common naming) |
| Window too small/large | Adjust `WINDOW_DURATION` in script (default 15 seconds) |

---

## Background & Motivation

### Why Manual Windowing?

Raw PPG recordings are typically 1-3 minutes long, but **not all of that time contains usable signal**. Common problems include:

- Motion artifacts when subjects shift their finger
- Sensor pressure changes causing baseline jumps
- Brief periods of poor perfusion (cold finger)
- Saturation regions where ADC clips
- Settling time at the start of recording

**Automatic windowing** would either:
- Be too lenient → include bad windows that corrupt the model
- Be too strict → reject usable windows because of one small artifact

**Manual windowing** lets a human expert visually inspect the signal and pick the cleanest segments. This is the **gold standard** for research-grade datasets where data quality matters more than processing speed.

### What is a "Window"?

A window is a fixed-duration slice of continuous PPG signal — typically **15 seconds** in this project. Why 15 seconds?

- **Enough beats:** ~15-25 heartbeats at 60-100 BPM (sufficient for ensemble averaging)
- **Short enough:** Reduces chance of motion artifacts during the window
- **ML-friendly:** Provides consistent input size for feature extraction
- **Matches training data:** All downstream code expects this duration

### Where This Fits in the Pipeline

```
[Step 1: Raw Recording]
       |
       v
[Step 2: Verification]    <- Visual check that signal is usable at all
       |
       v
[Step 3: THIS TOOL]       <- Manual selection of clean 15s windows
       |
       v
[Step 4: Signal Processing]  <- Automated filtering + beat detection
       |
       v
[Step 5: Feature Extraction]
       |
       v
[ML Model Training]
```

This tool is the **last manual step** before everything becomes automated. The quality of windows you select here directly affects model performance downstream.

---

## Tool Overview

### What the GUI Looks Like

```
+-----------------------------------------------------------+
|  Fixed Window Selector: click to place blue window...     |
|                                                            |
|  +-----------------------------------------------------+  |
|  |                                                     |  |
|  |   RED Amplitude                                     |  |
|  |   (red waveform line)                               |  |
|  |        ___    ___    ___    ___                     |  |
|  |   ____/   \__/   \__/   \__/   \___                 |  |
|  |        [::::: BLUE WINDOW :::::]                    |  |
|  |                                                     |  |
|  +-----------------------------------------------------+  |
|  +-----------------------------------------------------+  |
|  |                                                     |  |
|  |   IR Amplitude                                      |  |
|  |   (blue waveform line)                              |  |
|  |        ___    ___    ___    ___                     |  |
|  |   ____/   \__/   \__/   \__/   \___                 |  |
|  |        [::::: BLUE WINDOW :::::]                    |  |
|  |                                                     |  |
|  |   Blue Window: 12.50s -> 27.50s | Saved: 3          |  |
|  +-----------------------------------------------------+  |
|                                                            |
|  Scroll Start (sec) [=========o================]          |
|                                                            |
|  [ Add Block ]  [ Undo Last ]      [   Done   ]           |
+-----------------------------------------------------------+
```

### Two-Panel Layout

- **TOP PANEL (RED channel):** Shows the red light waveform in red color
- **BOTTOM PANEL (IR channel):** Shows the infrared waveform in blue color
- **Shared X-axis:** Time in seconds, synchronized between both panels
- **Blue tinted region:** Your currently-selected window (visible on both panels)

### Sliding View System

Because recordings can be long (60-300 seconds), the GUI shows only a **30-second window** at a time (configurable via `VIEW_WINDOW_SEC`). Use the slider at the bottom to scroll through the entire recording.

The blue selection window is **fixed at 15 seconds** (configurable via `WINDOW_DURATION`) but you can position it anywhere within the visible region.

### Suggested Diagram to Create

```
DIAGRAM 1: GUI Screenshot with Annotations

Create an annotated screenshot showing:
  - Real screenshot of the running GUI
  - Labeled arrows pointing to:
    * RED channel panel (top)
    * IR channel panel (bottom)
    * Blue selection window (with size annotation)
    * Scroll slider
    * Add Block / Undo / Done buttons
    * Info text showing current selection
  - Color coding:
    * RED arrow: RED channel elements
    * BLUE arrow: IR channel elements
    * GREEN arrow: Selection elements
    * ORANGE arrow: Control elements
  - Tool: Screenshot software + Figma/PowerPoint for annotations
  - Size: Landscape, 1920x1080
```

---

## Features & Capabilities

### Interactive Workflow
- **Visual window selection** — see exactly what you're saving
- **Dual-channel display** — verify both RED and IR are clean
- **Click-to-position** — natural way to pick window start
- **Add/Undo system** — easy curation of selections
- **Live info display** — see current window time + saved count

### Smart Data Handling
- **Column name aliasing** — accepts `IR`, `IR_Value`, `infrared`, etc.
- **Auto column detection** — no need to rename your CSV columns
- **Auto-scaling axes** — handles any signal amplitude range
- **Window edge clamping** — prevents selecting past recording end

### Configurability
- **Adjustable window duration** — change `WINDOW_DURATION` for different applications
- **Configurable sample rate** — must match your sensor (default 400 Hz)
- **Customizable view size** — adjust how much signal is visible at once
- **Flexible input/output paths** — set once at the top of the script

### Organized Output
- **Per-recording folders** — each input file gets its own output folder
- **Sequential naming** — `_Win0`, `_Win1`, `_Win2`, etc.
- **Preserves original columns** — saved CSVs have same structure as input
- **Easy downstream processing** — output ready for automated signal pipeline

---

## Installation & Prerequisites

### System Requirements

| Requirement | Recommended |
|---|---|
| **Python** | 3.10+ |
| **OS** | Windows 10/11, Linux, macOS |
| **RAM** | 4 GB minimum |
| **Display** | 1280x720 minimum (1920x1080 recommended) |
| **Storage** | Negligible (just saves small CSVs) |

### Required Python Packages

```
numpy >= 1.24.0
pandas >= 2.0.0
matplotlib >= 3.7.0
tkinter (usually included with Python)
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
pip install numpy pandas matplotlib
```

### Critical Note: Matplotlib Backend

This tool **requires** the TkAgg backend to display the interactive GUI. The script forces this at the top:

```python
import matplotlib
matplotlib.use("TkAgg")  # MUST be before importing pyplot
import matplotlib.pyplot as plt
```

If you remove or change this line, the GUI may not appear, or buttons/sliders won't respond.

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

### Verification

```bash
python -c "import matplotlib; matplotlib.use('TkAgg'); import matplotlib.pyplot as plt; print('GUI backend OK')"
```

Expected output: `GUI backend OK` (no errors)

---

## Input Data Format

### Expected CSV Structure

The tool expects CSV files with two required columns: **IR** (infrared) and **RED**. Column naming is **case-insensitive** and supports multiple common variations.

### Supported Column Name Aliases

| Required Channel | Accepted Names |
|---|---|
| **IR Channel** | `IR_Value`, `IR`, `ir_value`, `infrared`, `Infrared` |
| **RED Channel** | `Red_Value`, `RED`, `red_value` |

The script auto-detects and renames to internal standard names (`IR_Value`, `Red_Value`) during loading.

### Sample Input File

```csv
Timestamp,IR,RED
0,125463,87234
1,125890,87456
2,126123,87689
3,126890,87932
4,127445,88212
...
```

**Notes:**
- `Timestamp` column is optional (not used by this tool)
- Order of columns doesn't matter
- Other columns can exist (they'll be preserved in output)

### File Naming Convention

The recommended pattern for input files:

```
{subject_id}_{session}.csv
```

**Examples:**
- `Ali(22-enc-12)v1.csv`
- `Jamil(23-enc-46)v2.csv`
- `Subject_001_morning.csv`

The output windows will inherit this naming with `_Win{N}` suffix.

### Sample Rate

The tool assumes **400 Hz sampling rate** by default (configurable via `FS` parameter). For accurate window duration calculation, this must match your actual sensor output rate.

If your sensor samples at a different rate:
```python
FS = 100   # Change to your actual sample rate
```

### Recording Duration

Any duration works, but for best results:
- **Minimum:** 30 seconds (allows at least 1 good window)
- **Recommended:** 60-180 seconds (allows 3-10 good windows)
- **Maximum:** No hard limit, but very long recordings (>10 minutes) may slow the GUI

---

## Output Structure

### Per-Recording Output Folder

For each input CSV processed, the tool creates a folder named after the input file:

```
BASE_OUTPUT_DIR/
└── {input_filename}/                       <- Folder named after input
    ├── {input_filename}_Win0.csv           <- First selected window
    ├── {input_filename}_Win1.csv           <- Second selected window
    ├── {input_filename}_Win2.csv           <- Third selected window
    └── ... (one CSV per selected window)
```

### Complete Example

Input file:
```
RAW_INPUT_DIR/
└── Ali(22-enc-12)v1.csv  (120 seconds, 48000 samples)
```

After running tool and selecting 5 windows:

```
BASE_OUTPUT_DIR/
└── Ali(22-enc-12)v1/
    ├── Ali(22-enc-12)v1_Win0.csv           (samples 5000-11000  | 12.5s-27.5s)
    ├── Ali(22-enc-12)v1_Win1.csv           (samples 12000-18000 | 30.0s-45.0s)
    ├── Ali(22-enc-12)v1_Win2.csv           (samples 22000-28000 | 55.0s-70.0s)
    ├── Ali(22-enc-12)v1_Win3.csv           (samples 32000-38000 | 80.0s-95.0s)
    └── Ali(22-enc-12)v1_Win4.csv           (samples 40000-46000 | 100.0s-115.0s)
```

### Output File Contents

Each saved window CSV contains:
- **Same columns** as the input CSV (all preserved)
- **Same data format** (no transformation applied)
- **Exactly `WINDOW_DURATION × FS` rows** (default: 15 × 400 = 6000 rows)

Sample output file structure:
```csv
Timestamp,IR_Value,Red_Value
5000,125463,87234
5001,125890,87456
5002,126123,87689
...
(6000 rows total for 15-second window at 400 Hz)
```

### Window Numbering

- Windows are numbered **starting from 0** (`_Win0`, `_Win1`, `_Win2`, ...)
- Numbering follows the **order you clicked Add Block**, not time order
- You can select windows out of order; they'll be numbered chronologically by save sequence

### Suggested Diagram to Create

```
DIAGRAM 2: Output Folder Tree

Create a tree diagram showing:
  - LEFT: Input folder with single CSV file
  - ARROW: "Running the tool"
  - RIGHT: Output folder with named subfolder
  - Inside subfolder: 5 numbered Win CSV files
  - Annotate each Win file with its time range
  - Color coding:
    * BLUE: Input file
    * GREEN: Output folder
    * ORANGE: Individual window files
  - Tool: TreeView format in PowerPoint or Excalidraw
  - Size: Portrait, 1080x1920
```

---

## Interactive GUI Walkthrough

### Step-by-Step Workflow

#### Step 1: Launch the Tool
Run the script. A **file picker dialog** appears showing your `RAW_INPUT_DIR`.

#### Step 2: Select a CSV File
Navigate to and double-click your raw recording CSV. The file picker closes.

#### Step 3: GUI Window Opens
The main GUI appears showing:
- **RED channel** waveform on the TOP panel (red line)
- **IR channel** waveform on the BOTTOM panel (blue line)
- **First 30 seconds** of the recording visible
- A **blue tinted region** at the start (this is your selection window)
- A **scroll slider** at the bottom for navigation
- **Three buttons:** Add Block, Undo Last, Done

#### Step 4: Scroll to Find Good Signal
Drag the slider to navigate through the recording. The waveforms update in real-time as you scroll. Look for regions with:
- Clean, regular pulses
- No sudden amplitude changes
- No flat or saturated regions
- Consistent baseline

#### Step 5: Click to Position the Window
Click anywhere on either panel. The **blue tinted region jumps** so its left edge aligns with your click position. The window size stays fixed (default 15 seconds).

You can click on either panel:
- Click on RED panel → blue window moves
- Click on IR panel → blue window moves
- Both panels stay synchronized

#### Step 6: Verify the Selection
Look at the **info text** at the bottom of the IR panel:
```
Blue Window: 12.500s -> 27.500s | Saved: 0
```

This shows the exact time range of your current selection.

#### Step 7: Save This Window
Click **"Add Block"** button. The terminal prints:
```
✅ Saved Window: 12.500s - 27.500s
```

The info text updates:
```
Blue Window: 12.500s -> 27.500s | Saved: 1
```

The window itself doesn't move — you can immediately reposition it for another selection.

#### Step 8: Continue Selecting More Windows
Repeat Steps 4-7 for each window you want. Most recordings yield 3-8 good windows.

#### Step 9: Make Corrections if Needed
If you accidentally added a bad window, click **"Undo Last"**. The terminal prints:
```
↩️ Removed: 12.500s - 27.500s
```

#### Step 10: Finish and Save
When you've selected all desired windows, click **"Done"**. The GUI closes and the tool:
1. Lists all saved windows in the terminal
2. Creates the output folder
3. Writes each window as a separate CSV
4. Confirms completion with file paths

### GUI Element Reference

| Element | Function | Notes |
|---|---|---|
| **Top Panel** | Shows RED channel | Click anywhere to set window start |
| **Bottom Panel** | Shows IR channel | Click anywhere to set window start |
| **Blue Tinted Region** | Currently selected 15s window | Updates immediately on click |
| **Scroll Slider** | Navigate through full recording | Drag or click anywhere on track |
| **Info Text** | Shows window times + saved count | Updates after each action |
| **Add Block** | Save current window coordinates | Window stays visible after add |
| **Undo Last** | Remove most recent saved window | Only removes one at a time |
| **Done** | Close GUI and write CSVs | Cannot undo after clicking |

### Auto-Clamp Behavior

If you click near the end of the recording where there's not enough room for a full 15-second window, the tool automatically **clamps** the window to fit:

```
Recording length: 100 seconds
You click at: 95 seconds (only 5 seconds left)
Window auto-adjusts to: 85s → 100s
```

You'll see a brief terminal message if this happens:
```
⚠️ Window exceeded signal end. Adjusting.
```

This prevents incomplete windows from being saved.

### Tips for Good Window Selection

**Visual checklist for a good window:**
- ✅ Smooth, regular pulses (one peak per heartbeat)
- ✅ Consistent peak amplitudes (no large variations)
- ✅ Clean baseline (no sudden jumps)
- ✅ Both RED and IR look similar in shape
- ✅ No flat regions or saturation
- ✅ Visible dicrotic notch (small bump after main peak)

**Red flags (avoid these regions):**
- ❌ Motion artifacts (sudden large spikes)
- ❌ Flat lines (sensor disconnection)
- ❌ Saturated regions (signal clipped at max)
- ❌ Very low amplitude (poor perfusion)
- ❌ Inconsistent peak shapes
- ❌ Baseline drift within the 15-second window

**General strategy:**
- Skip the first 5-10 seconds (sensor settling time)
- Look for 30+ second stretches of stable signal
- Try to pick 5-10 windows per recording (more = better averaging)
- Be consistent across subjects (don't pick noisy windows from one but only pristine from another)

---

## Configuration Reference

All configurable settings are at the top of the script in the **USER CONFIGURATION SECTION**. Change values and re-run.

### Path Settings

```python
RAW_INPUT_DIR   = r"C:\Users\...\02_Verified_Raw"
BASE_OUTPUT_DIR = r"C:\Users\...\03_Windowed"
```

| Parameter | Description |
|---|---|
| `RAW_INPUT_DIR` | Folder containing your verified raw CSV files. The file picker opens here. |
| `BASE_OUTPUT_DIR` | Folder where output subfolders (one per input file) will be created. |

### Window & Sampling Settings

```python
WINDOW_DURATION = 15     # Seconds (blue window width)
FS              = 400    # Hz
VIEW_WINDOW_SEC = 30     # Seconds shown in sliding viewer
```

| Parameter | Default | Description |
|---|---|---|
| `WINDOW_DURATION` | `15` | **Size of each saved window in seconds.** Must match what downstream pipeline expects. Standard is 15s. |
| `FS` | `400` | **Sampling frequency of your sensor.** Must match the actual rate at which data was recorded. Used to convert seconds to sample indices. |
| `VIEW_WINDOW_SEC` | `30` | **How much signal is visible in the GUI at once.** Doesn't affect saved windows — just changes scrolling view. |

### Important Configuration Notes

#### WINDOW_DURATION
- Pipeline standard is **15 seconds**
- Changing this affects all downstream processing (signal processing pipeline expects 15s = 6000 samples at 400Hz)
- Only change if you're rebuilding the entire pipeline with different window sizes

#### FS (Sampling Frequency)
- Must match your sensor's actual output rate
- If your firmware samples at 200 Hz, set `FS = 200`
- Wrong FS → wrong window durations in samples
- Default 400 Hz matches the recommended MAX30102 configuration

#### VIEW_WINDOW_SEC
- Larger value → see more signal at once, but each pixel represents more time (less detail)
- Smaller value → more detail, but more scrolling needed
- 30 seconds is a good balance for most use cases
- Set to ≥ 2× `WINDOW_DURATION` so you can always see the full selection window

### Quick Tuning for Different Use Cases

**Short windows (e.g., HRV analysis):**
```python
WINDOW_DURATION = 10
VIEW_WINDOW_SEC = 20
```

**Long windows (e.g., morphology averaging):**
```python
WINDOW_DURATION = 30
VIEW_WINDOW_SEC = 60
```

**Low sample rate sensor:**
```python
FS = 100
# WINDOW_DURATION stays at 15 (will save 1500 samples)
```

---

## Code Architecture

### File Structure

```
project_root/
├── Windowing_Tool.py       <- All code in single file
└── README.md               <- This file
```

### Main Imports

```python
import os                                  # File path operations, folder creation
import numpy as np                         # Numerical operations, array clipping
import pandas as pd                        # CSV reading/writing, DataFrame slicing

import matplotlib                          # GUI backend control (CRITICAL)
matplotlib.use("TkAgg")                    # MUST be before importing pyplot

import matplotlib.pyplot as plt            # Plot generation, figure management
from matplotlib.widgets import Slider, Button  # Interactive GUI controls
from tkinter import filedialog, Tk         # Native file picker dialog
```

### The Matplotlib Backend Requirement (Important!)

The very first lines after standard imports are:

```python
import matplotlib
matplotlib.use("TkAgg")
```

**Why this is critical:**
- Matplotlib has multiple backends (Agg, TkAgg, Qt5Agg, etc.)
- Default backend may be `Agg` (non-interactive) — buttons/sliders won't work
- `TkAgg` is the only backend that reliably creates **interactive windows** with proper button/slider response
- This call **must happen BEFORE** `import matplotlib.pyplot as plt`
- If you put it after, the backend is already locked and your change is ignored

**Symptoms if backend is wrong:**
- GUI window doesn't appear
- GUI appears but buttons do nothing
- Sliders don't update plot
- Plot freezes after first click

### Key Functions

#### `load_csv_with_aliases(file_path)`
```python
def load_csv_with_aliases(file_path: str) -> pd.DataFrame:
    # Reads CSV, detects column name variations, renames to internal standard.
    # Handles: IR/IR_Value/ir_value/infrared and RED/Red_Value/red_value
    # Returns DataFrame with standardized column names (IR_Value, Red_Value).
```

#### `choose_file(raw_input_dir)`
```python
def choose_file(raw_input_dir: str) -> str:
    # Opens tkinter file picker dialog at the specified directory.
    # Returns path to selected CSV file (or empty string if cancelled).
    # Handles edge case where input is already a direct file path.
```

#### `fixed_window_selector(df, total_duration)`
```python
def fixed_window_selector(df, total_duration):
    # The main GUI function. Creates the dual-panel matplotlib figure with:
    # - RED channel (top) and IR channel (bottom)
    # - Scroll slider for navigation
    # - Click handler for window positioning
    # - Add Block / Undo Last / Done buttons
    # - Auto-scaling axes and synchronized blue window overlay
    # Returns list of (start_sec, end_sec) tuples for saved windows.
```

#### `save_selected_windows(df, file_path, windows)`
```python
def save_selected_windows(df, file_path, windows):
    # Iterates through saved window time ranges:
    # 1. Converts seconds to sample indices using FS
    # 2. Slices the DataFrame for each window
    # 3. Creates output folder named after input file
    # 4. Saves each window as {filename}_Win{N}.csv
    # Prints progress to terminal.
```

#### `slice_data()`
```python
def slice_data():
    # Main orchestrator:
    # 1. Verify matplotlib backend
    # 2. Open file picker → get CSV path
    # 3. Load CSV with column aliasing
    # 4. Compute total duration from row count and FS
    # 5. Print user controls to terminal
    # 6. Launch GUI selector
    # 7. Save selected windows to disk
```

### Function Call Hierarchy

```
slice_data() [Entry Point]
    |
    +-- choose_file()
    |       +-- tkinter file dialog
    |
    +-- load_csv_with_aliases()
    |       +-- pd.read_csv()
    |       +-- column rename logic
    |
    +-- fixed_window_selector() [GUI loop]
    |       +-- Creates matplotlib figure
    |       +-- Sets up sliders + buttons
    |       +-- Registers click handler
    |       +-- Blocks until "Done" clicked
    |       +-- Returns saved windows
    |
    +-- save_selected_windows()
            +-- Creates output folder
            +-- Slices DataFrame per window
            +-- Saves CSVs with proper naming
```

### Suggested Diagram to Create

```
DIAGRAM 3: Function Flow Diagram

Create a flowchart showing:
  - TOP: slice_data() entry box
  - Sequential boxes for each function call
  - Decision diamonds for user interactions (file selected? windows saved?)
  - GUI block shown as a separate "interactive loop" subgraph
  - Color coding:
    * BLUE: Data loading
    * GREEN: GUI interaction
    * ORANGE: File I/O
    * PURPLE: User decisions
  - Tool: draw.io or Mermaid flowchart
  - Size: Portrait, 1080x1920
```

---

## Troubleshooting & Tips

### Common Issues Table

| Symptom | Likely Cause | Fix |
|---|---|---|
| **GUI doesn't open** | Wrong matplotlib backend | Verify `matplotlib.use("TkAgg")` is set BEFORE `import matplotlib.pyplot as plt` |
| **File picker doesn't appear** | Tkinter not installed (Linux only) | `sudo apt-get install python3-tk` |
| **"Required columns missing" error** | CSV uses different column names | Edit `load_csv_with_aliases()` to add your column name as an alias |
| **Buttons don't respond** | Backend issue or display issue | Try restarting Python, verify TkAgg backend |
| **Plot doesn't update on click** | Same backend issue | Force quit and restart with correct backend |
| **Window size wrong in output** | FS mismatch | Verify `FS` matches your actual sensor sample rate |
| **All saved windows are same length but wrong** | Wrong WINDOW_DURATION | Change `WINDOW_DURATION` in script |
| **Blue window jumps to edge** | Click near end of recording (auto-clamp) | Click further from the end |
| **Slider scrolls past end of data** | Recording shorter than VIEW_WINDOW_SEC | Reduce `VIEW_WINDOW_SEC` or use shorter view |
| **Output folder not created** | Permission issue | Check write access to `BASE_OUTPUT_DIR` |
| **GUI freezes after many windows** | Matplotlib memory accumulation | Save and restart for long sessions (>30 windows) |
| **Can't see dicrotic notch in GUI** | Auto-scale hiding small features | Click on a more zoomed region to refresh scale |

### Debugging Workflow

#### Step 1: Verify Matplotlib Backend
Add this at the top of the script (after imports) to confirm:
```python
print("Backend:", matplotlib.get_backend())
```
Should print `Backend: TkAgg`. If not, the backend forcing isn't working.

#### Step 2: Test File Picker Independently
If file picker isn't appearing, test tkinter alone:
```python
import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
print(filedialog.askopenfilename())
```
If this fails, you have a tkinter installation problem.

#### Step 3: Test CSV Loading
If you get "Required columns missing", run this manually:
```python
import pandas as pd
df = pd.read_csv("your_file.csv")
print(df.columns.tolist())
```
Then add any missing aliases to `load_csv_with_aliases()`.

#### Step 4: Verify Output Folder Write Access
Try creating a test file:
```python
import os
test_path = os.path.join(BASE_OUTPUT_DIR, "test.txt")
with open(test_path, "w") as f:
    f.write("test")
print("Write OK")
```
If this fails, fix folder permissions before running the tool.

### Best Practices

#### Quality Over Quantity
Better to save 3 clean windows than 10 noisy ones. The signal processing pipeline can handle fewer windows; it can't fix bad data.

#### Be Consistent Across Subjects
If you're strict for one subject, be strict for all. Inconsistent quality across subjects introduces bias into your ML model.

#### Document Your Selection Criteria
Keep notes on what you consider "good signal" so you can apply the same standard months later or have others reproduce your work.

#### Restart Periodically
Matplotlib can accumulate memory during long sessions. After processing 20-30 recordings, restart Python to keep performance smooth.

#### Backup Original Recordings
Never modify or move the input CSVs. This tool only reads them; if you need to re-window with different settings, you need the originals.

#### Sample Naming Convention
Stick to a consistent naming pattern for input files. This makes batch processing in downstream tools much easier:
- ✅ Good: `Ali(22-enc-12)v1.csv`, `Ali(22-enc-12)v2.csv`
- ❌ Bad: `recording_1.csv`, `Ali_morning.csv`, `subject_2_session_a.csv`

---

## Next Step in Pipeline

After successfully windowing your recordings with this tool, your output folder will contain organized per-subject window files ready for the next stage:

```
03_Windowed/
├── Ali(22-enc-12)v1/
│   ├── Ali(22-enc-12)v1_Win0.csv
│   ├── Ali(22-enc-12)v1_Win1.csv
│   └── ...
├── Jamil(23-enc-46)v2/
│   ├── Jamil(23-enc-46)v2_Win0.csv
│   └── ...
└── ...
```

### Next Tool: Automated Signal Processing Pipeline

The output of this tool feeds directly into **Step 4: Automated Signal Processing** which:

- Reads each windowed CSV
- Applies filtering (low-pass, high-pass, smoothing)
- Detects individual heartbeats
- Performs ensemble averaging
- Extracts signal quality metrics
- Saves processed outputs ready for feature extraction

The Signal Processing pipeline supports **BATCH mode** which automatically processes ALL subject folders inside `03_Windowed/` — so once you've windowed all your recordings here, you can run the next step once and process everything at once.

**See:** `Automated_Signal_Processing_README.md` for details on the next stage.

---

## Summary

This tool is the **manual quality gate** of your data pipeline. While it requires human time and attention, the quality of windows selected here directly affects every downstream stage including final model performance.

Key benefits:
- ✅ Visual inspection prevents bad data from entering the pipeline
- ✅ Dual-channel view confirms both RED and IR are clean
- ✅ Flexible Add/Undo workflow allows easy curation
- ✅ Configurable for different window sizes and sample rates
- ✅ Organized output ready for automated processing
- ✅ Lightweight and easy to use

For best results: take your time, be consistent, and document your criteria. Good data in → good predictions out.

Happy windowing!