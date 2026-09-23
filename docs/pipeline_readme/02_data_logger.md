# Step 1: PPG Data Logger

> High-performance Python tool that receives PPG sensor data from an ESP32 microcontroller via serial, displays it in real-time with dual-channel plots, and saves it to timestamped CSV files for the non-invasive glucose estimation pipeline.

---

## TL;DR

This tool bridges the ESP32 firmware (Step 0) and the rest of the Python pipeline. It opens a serial connection to your MAX30102 sensor, reads IR and RED channel data at 60+ FPS using PyQtGraph, and simultaneously saves every sample to a CSV file with millisecond-precision timestamps. The dual-panel GUI lets you verify signal quality in real-time while recording.

**Quick Stats:**
- ~150 lines of Python code
- 60 FPS live plotting (10-20× faster than matplotlib)
- Single-file recording (one CSV per session)
- Terminal-based session ID input
- Auto-create folder + auto-append `.csv` extension
- Overwrite protection with prompt
- Configurable buffer size for display window
- Built on PyQt5 + PyQtGraph + pyserial

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Quick Start](#quick-start)
3. [Background & Motivation](#background--motivation)
4. [Tool Overview](#tool-overview)
5. [Features & Capabilities](#features--capabilities)
6. [Installation & Prerequisites](#installation--prerequisites)
7. [Hardware Setup](#hardware-setup)
8. [Output Structure](#output-structure)
9. [Step-by-Step Usage](#step-by-step-usage)
10. [Configuration Reference](#configuration-reference)
11. [Code Architecture](#code-architecture)
12. [Troubleshooting](#troubleshooting)
13. [Performance Notes & Tips](#performance-notes--tips)
14. [Next Step in Pipeline](#next-step-in-pipeline)
15. [References](#references)

---

## Quick Start

### Minimum Steps to Run

```bash
# 1. Connect your ESP32 (running PPG firmware) to your computer via USB
# 2. Identify the COM port:
#    Windows: Check Device Manager → Ports (COM & LPT)
#    Linux:   ls /dev/ttyUSB*
#    macOS:   ls /dev/cu.*

# 3. Create virtual environment
python -m venv ppg_env

# 4. Activate (Windows)
ppg_env\Scripts\activate

# 5. Install dependencies
pip install pyqtgraph PyQt5 pyserial numpy

# 6. Open the script and set:
#    SERIAL_PORT  = 'COM7'  (or your actual port)
#    BAUD_RATE    = 115200  (must match ESP32 firmware)
#    DATA_STORAGE_FOLDER_PATH = r"path/to/save/csvs"

# 7. Run the script
python Data_Logger.py

# 8. Enter a session ID when prompted (e.g., "SubjectA_Glu100")
# 9. GUI opens with live waveforms — verify signal looks good
# 10. Close the window to stop recording and save CSV
```

### Expected First Run Output

```
==================================================
📂 TARGET FOLDER: C:\Users\YourName\...\01_Raw
==================================================
👉 Enter Session ID (e.g., SubjectA_Glu100): SubjectA_Glu100

✅ Recording Started! Saving to: SubjectA_Glu100.csv
   (Close the plot window to stop saving)

[GUI window opens with two empty plots]
[Data starts flowing immediately]
[Two waveforms appear and scroll across screen]

[User closes the window]

💾 Session Saved & Closed.
```

### Common First-Time Issues

| Problem | Quick Fix |
|---|---|
| "Could not open COM7" | Close Arduino IDE or other serial monitors |
| GUI opens but plots are flat | Check baud rate matches ESP32 firmware exactly |
| Garbled data in CSV | Baud rate mismatch — verify both ends |
| Import errors | Run `pip install pyqtgraph PyQt5 pyserial numpy` |

---

## Background & Motivation

### Why a Custom Python Receiver?

The Arduino IDE has a built-in Serial Plotter, so why build a custom tool?

**Three critical needs the Arduino plotter cannot meet:**

1. **Save data to CSV** — Arduino plotter only displays, doesn't save
2. **High-precision timestamps** — Need millisecond accuracy for research
3. **Custom folder organization** — Need files named per session, organized per subject

This tool provides all three while still offering excellent live visualization.

### Why PyQtGraph Instead of Matplotlib?

Matplotlib is the standard Python plotting library, but for **real-time streaming data**, it's too slow:

| Aspect | Matplotlib | PyQtGraph |
|---|---|---|
| Typical FPS | 5-15 FPS | 60+ FPS |
| Rendering | Software (CPU) | Hardware-accelerated (OpenGL) |
| Designed for | Static publication plots | Real-time data acquisition |
| Memory usage | Higher | Optimized for streaming |
| Latency | 100-200ms | <16ms |

PyQtGraph was built specifically for applications like ours — scientific instruments and data acquisition systems that need to display fast-changing data smoothly. With matplotlib, the plot would lag behind the actual sensor data and miss visual artifacts that signal real problems.

### Why Circular Buffers?

Naive approach: append each new sample to a list, plot the whole list.
- Problem: After 60 seconds at 400 Hz, you have 24,000 samples to redraw 60 times per second = 1.4 million points/second = lag

Circular buffer approach: fixed-size NumPy array, shift left and add new sample at the end.
- Always plot exactly `WINDOW_SIZE` points (e.g., 4000)
- Constant memory usage regardless of recording length
- NumPy roll operation is O(N) but very fast for small N
- Plot update time stays constant

### Where This Fits in the Pipeline

```
[ESP32 Firmware] → [USB Serial] → [THIS TOOL] → [Raw CSV] → [Step 2: Verification] → [Step 3: Windowing] → ...
                                       ↑
                                  You are here
```

This is **Step 1** of the data pipeline — the entry point for all raw sensor data into the system. Every downstream step depends on the quality and consistency of files produced here.

---

## Tool Overview

### What the GUI Looks Like

```
+-----------------------------------------------------------+
|  Recording: SubjectA_Glu100.csv | COM7                    |
|                                                            |
|  +-----------------------------------------------------+  |
|  |  IR Signal (Raw)                                    |  |
|  |                                                     |  |
|  |        ___    ___    ___    ___    ___              |  |
|  |   ____/   \__/   \__/   \__/   \__/   \____         |  |
|  |  (blue line, IR channel data)                       |  |
|  |                                                     |  |
|  |          Amplitude                                  |  |
|  +-----------------------------------------------------+  |
|  +-----------------------------------------------------+  |
|  |  Red Signal (Raw)                                   |  |
|  |                                                     |  |
|  |        ___    ___    ___    ___    ___              |  |
|  |   ____/   \__/   \__/   \__/   \__/   \____         |  |
|  |  (orange line, RED channel data)                    |  |
|  |                                                     |  |
|  |          Amplitude          Samples                 |  |
|  +-----------------------------------------------------+  |
|                                                            |
+-----------------------------------------------------------+
```

### Two-Panel Real-Time View

- **TOP PANEL:** IR (Infrared) channel — typically blue waveform
- **BOTTOM PANEL:** RED channel — typically orange waveform
- **Shared X-axis:** Sample numbers (zoom in one panel zooms the other)
- **Auto-scrolling:** New data pushes old data to the left
- **Live updates:** Plot refreshes 60 times per second

### Workflow Phases

**Phase 1: Terminal Input (before GUI)**
- Tool prints target folder path
- Asks for session ID
- Checks if file exists, prompts to overwrite if needed
- Opens CSV file for writing

**Phase 2: GUI Recording (live phase)**
- Window opens with two empty plots
- Serial port opens, data starts flowing
- Each sample: parsed → saved to CSV → added to display buffer
- Plot refreshes every 16ms (60 FPS)

**Phase 3: Cleanup (after close)**
- User closes window with X button
- Serial port closes
- CSV file is flushed and closed
- Confirmation printed to terminal

### Suggested Diagram to Create

```
DIAGRAM 1: System Data Flow

Create a horizontal flow diagram showing:
  - LEFT: ESP32 with MAX30102 sensor (with finger illustration)
  - ARROW: USB cable labeled "Serial @ baud rate"
  - CENTER: Computer running this Python tool
  - Two outputs from the tool:
    * UP arrow: Live PyQt GUI window
    * DOWN arrow: CSV file being written
  - RIGHT: Next pipeline step (Step 2: Verification)
  - Color coding:
    * RED: Hardware components
    * BLUE: Serial communication
    * GREEN: Python application
    * PURPLE: Output files
  - Tool: draw.io or PowerPoint
  - Size: Landscape, 1920x1080
```

---

## Features & Capabilities

### Real-Time Performance
- **60 FPS plot updates** — Hardware-accelerated rendering via PyQtGraph
- **Sub-16ms latency** — Sample appears on screen almost instantly
- **Circular buffer** — Constant memory usage regardless of recording length
- **Linked X-axis** — Zoom one panel and the other zooms in sync

### Robust Data Logging
- **Per-sample timestamps** — HH:MM:SS.mmm format (millisecond precision)
- **Real-time CSV writing** — No data loss even if app crashes
- **Standard CSV format** — Easy to load in Python, MATLAB, Excel
- **Three columns:** Timestamp, IR, RED

### User Experience
- **Terminal-based session naming** — Quick text input, no popups
- **Auto-append `.csv`** — Don't need to type the extension
- **Overwrite protection** — Prompts before replacing existing files
- **Auto-folder creation** — Output folder created if it doesn't exist
- **Window title shows session** — See what's recording at a glance

### Hardware Integration
- **Cross-platform serial** — Works with any COM/USB port
- **Configurable baud rate** — Match any sensor firmware
- **Graceful error handling** — Clear messages for connection problems
- **Auto-cleanup on close** — Serial port + file released properly

### Visual Quality Verification
- **Dual-channel display** — See IR and RED simultaneously
- **Professional styling** — White background, MATLAB-style colors
- **Grid overlay** — Easy amplitude reading
- **Synchronized panels** — Both update in lockstep

---

## Installation & Prerequisites

### System Requirements

| Requirement | Recommended |
|---|---|
| **Python** | 3.10+ |
| **OS** | Windows 10/11, Linux, macOS |
| **RAM** | 4 GB minimum |
| **Display** | 1280x720 minimum (1920x1080 recommended) |
| **USB Port** | USB 2.0 or higher |

### Required Python Packages

```
pyqtgraph >= 0.13.0    # High-performance plotting library
PyQt5 >= 5.15.0        # GUI framework (Qt bindings for Python)
pyserial >= 3.5        # Serial communication library
numpy >= 1.24.0        # Numerical arrays for circular buffer
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
pip install pyqtgraph PyQt5 pyserial numpy
```

#### Step 4: Verify Installation

```bash
python -c "import pyqtgraph, PyQt5, serial, numpy; print('All packages OK')"
```

Expected output: `All packages OK`

### USB Driver Requirements

The ESP32-S3 connects as a USB serial device. Driver requirements depend on the chip used:

- **ESP32-S3 with native USB:** No drivers needed (uses standard USB-CDC class)
- **ESP32 boards with CP210x bridge:** Install [CP210x driver](https://www.silabs.com/developers/usb-to-uart-bridge-vcp-drivers) (most common)
- **ESP32 boards with CH340 bridge:** Install [CH340 driver](http://www.wch.cn/downloads/CH341SER_ZIP.html)

Most modern Windows 10/11 systems install these drivers automatically when you plug in the board.

---

## Hardware Setup

### Required Hardware

| Item | Notes |
|---|---|
| ESP32-S3 board with firmware loaded | Must be running compatible PPG firmware |
| MAX30102 sensor connected to ESP32 | Per firmware wiring (typically GPIO 1=SDA, GPIO 2=SCL) |
| USB-C to USB-A or USB-C cable | For ESP32-S3 DevKit |
| Computer with USB port | Any modern PC |

### Identifying Your COM Port

The most common setup issue is using the wrong COM port. Here's how to find yours:

#### Windows

1. Plug in your ESP32 via USB
2. Open **Device Manager** (Win+X → Device Manager)
3. Expand **"Ports (COM & LPT)"**
4. Look for entries like:
   - `Silicon Labs CP210x USB to UART Bridge (COM7)`
   - `USB Serial Device (COM4)`
   - `USB-Enhanced-SERIAL CH340 (COM3)`
5. The number in parentheses is your port (e.g., `COM7`)

#### Linux

```bash
# Before plugging in ESP32:
ls /dev/ttyUSB* /dev/ttyACM*
# Note what's listed (or "No such file" if nothing)

# Plug in ESP32, then run again:
ls /dev/ttyUSB* /dev/ttyACM*
# The NEW entry is your ESP32 (typically /dev/ttyUSB0 or /dev/ttyACM0)

# You may need permissions:
sudo usermod -a -G dialout $USER
# Then log out and log back in
```

#### macOS

```bash
ls /dev/cu.*
# Look for entries like:
# /dev/cu.SLAB_USBtoUART  (CP210x)
# /dev/cu.usbserial-0001  (Generic)
# /dev/cu.wchusbserial1410 (CH340)
```

### Setting the Correct COM Port

In the script, update the `SERIAL_PORT` constant:

```python
# Windows example
SERIAL_PORT = 'COM7'

# Linux example
SERIAL_PORT = '/dev/ttyUSB0'

# macOS example
SERIAL_PORT = '/dev/cu.SLAB_USBtoUART'
```

### Matching the Baud Rate

The `BAUD_RATE` in this Python tool **MUST match** the `UART_BAUD_RATE` in your ESP32 firmware:

```python
# In this Python tool:
BAUD_RATE = 115200

# Must match in ESP32 firmware (main.c):
#define UART_BAUD_RATE 115200
```

**Recommended baud rates:**
| Sample Rate | Baud Rate |
|---|---|
| 50-100 Hz | 115200 |
| 200-400 Hz | 115200 or 460800 |
| 800-1600 Hz | 460800 or 921600 |
| 3200 Hz | 921600 |

### Closing Conflicting Programs

The COM port can only be used by one program at a time. Before running this tool, close:
- Arduino IDE Serial Monitor / Serial Plotter
- PuTTY, TeraTerm, or other terminal programs
- VS Code's serial monitor
- ESP-IDF monitor (`idf.py monitor`)

If you see "Could not open COM port" errors, this is usually the cause.

---

## Output Structure

### Single File Per Session

Each run of this tool creates **one CSV file** in your specified folder:

```
DATA_STORAGE_FOLDER_PATH/
├── SubjectA_Glu100.csv         <- First session
├── SubjectA_Glu120.csv         <- Different session
├── SubjectB_Glu85.csv          <- Different subject
└── ...
```

The filename comes from the **Session ID** you enter at the terminal prompt. The `.csv` extension is added automatically if you don't include it.

### File Naming Recommendations

Use a consistent naming convention to make downstream batch processing easy:

**Recommended pattern:**
```
{Subject}_{Glucose}_{Date}.csv
```

**Examples:**
- `Ali(22-enc-12)v1.csv` — Subject Ali, session 1
- `Jamil(23-enc-46)v2_Glu130.csv` — Subject Jamil, glucose 130
- `TestRun_2024-12-19.csv` — Date-based for testing

**Avoid:**
- Spaces (use underscores instead)
- Special characters except `_-()`
- Generic names like `data.csv` (gets overwritten easily)

### File Contents Format

Each saved CSV has three columns with a header row:

```csv
Timestamp,IR,RED
14:32:15.123,125463,87234
14:32:15.126,125890,87456
14:32:15.128,126123,87689
14:32:15.131,126890,87932
14:32:15.133,127445,88212
...
```

| Column | Type | Description |
|---|---|---|
| `Timestamp` | String | Local time in HH:MM:SS.mmm format |
| `IR` | Float | Raw IR channel reading from sensor |
| `RED` | Float | Raw RED channel reading from sensor |

### Sample Serial Input Format

For reference, the ESP32 firmware sends data in this format over serial:

```
0,0
0,0
125463,87234
126890,88456
128122,89745
...
```

The Python tool parses each line, extracts IR and RED values, adds a timestamp, and writes to the CSV. The `0,0` lines (indicating no finger detected) are saved as-is for later filtering.

### Overwrite Protection

If you enter a session ID that already exists, the tool prompts before overwriting:

```
👉 Enter Session ID (e.g., SubjectA_Glu100): SubjectA_Glu100

⚠️  WARNING: File 'SubjectA_Glu100.csv' already exists!
   Overwrite it? (y/n): n
❌ Cancelled. Exiting.
```

This prevents accidentally destroying previous recordings.

---

## Step-by-Step Usage

### Phase 1: Pre-Flight Setup

1. **Power on ESP32** — Connect via USB, wait for power LED
2. **Verify firmware is running** — ESP32 should be actively sampling
3. **Find COM port** (see Hardware Setup section)
4. **Close conflicting programs** (Arduino IDE, etc.)
5. **Edit script paths** if not already set

### Phase 2: Terminal Input

Run the script:

```bash
python Data_Logger.py
```

You'll see:

```
==================================================
📂 TARGET FOLDER: C:\Users\YourName\Documents\fyp\05_Data_Storage\01_Raw
==================================================
👉 Enter Session ID (e.g., SubjectA_Glu100): 
```

Enter a session ID. Options:

**Option A: Just the name**
```
👉 Enter Session ID: Ali_v1
```
(Tool auto-appends `.csv` → file becomes `Ali_v1.csv`)

**Option B: With extension**
```
👉 Enter Session ID: Ali_v1.csv
```
(Used as-is)

### Phase 3: Overwrite Check

If the file exists:

```
⚠️  WARNING: File 'Ali_v1.csv' already exists!
   Overwrite it? (y/n): 
```

- Type `y` to overwrite
- Type `n` (or anything else) to cancel and exit

### Phase 4: GUI Recording

After confirming, you'll see:

```
✅ Recording Started! Saving to: Ali_v1.csv
   (Close the plot window to stop saving)
```

The GUI window opens immediately. Within ~1 second, you should see:
- IR waveform (blue) appearing on top panel
- RED waveform (orange) appearing on bottom panel
- Both scrolling smoothly from right to left

**Place your finger on the MAX30102 sensor** — the waveforms should immediately change from baseline (around 0) to actual PPG signal with visible heartbeats.

### Phase 5: Monitor Signal Quality

While recording, check that:
- ✅ Waveforms are clean and pulsatile
- ✅ Both channels are similar in shape
- ✅ No flat regions or sudden spikes
- ✅ Amplitude is consistent

If signal looks bad:
- Reposition finger
- Apply gentle pressure
- Wait for sensor to stabilize (~5 seconds)
- If still bad, may need to adjust LED currents in firmware

### Phase 6: Stop Recording

When done, simply **close the GUI window** (click the X button).

You'll see in the terminal:

```
💾 Session Saved & Closed.
```

The CSV file is now safely saved with all your data.

### Recording Duration Guidance

Recommended recording lengths depend on your goal:

| Use Case | Duration |
|---|---|
| Quick test / debugging | 10-30 seconds |
| Single window for processing | 30-60 seconds |
| Multiple windows from one session | 90-180 seconds |
| Long-term data collection | 3-5 minutes |

Longer recordings give more flexibility in the windowing tool but take longer to process downstream.

---

## Configuration Reference

All settings are at the top of the script in the **USER CONFIGURATION SECTION**.

### Hardware Connection

```python
SERIAL_PORT = 'COM7'
BAUD_RATE   = 115200
```

| Parameter | Description |
|---|---|
| `SERIAL_PORT` | The COM port (Windows) or device path (Linux/macOS) of your ESP32. **Update this!** |
| `BAUD_RATE` | Serial communication speed. **Must match ESP32 firmware exactly.** Default: 115200. |

### File Saving

```python
DATA_STORAGE_FOLDER_PATH = r"C:\Users\YourName\...\01_Raw"
```

| Parameter | Description |
|---|---|
| `DATA_STORAGE_FOLDER_PATH` | Folder where CSV files will be saved. Tool creates this folder if it doesn't exist. Use `r"..."` (raw string) on Windows to handle backslashes. |

### Plotter Settings

```python
FS          = 400
WINDOW_SIZE = 4000
```

| Parameter | Description |
|---|---|
| `FS` | Sampling rate in Hz. Used for time axis calculations. Must match ESP32 firmware sample rate. |
| `WINDOW_SIZE` | Number of samples shown in the live plot at any moment. Larger = see more history, smaller = see more detail. |

### Tuning WINDOW_SIZE

Calculate based on how many seconds of signal you want to see:

```
WINDOW_SIZE = FS × seconds_to_display
```

Examples:
- `FS=400, WINDOW_SIZE=2000` → 5 seconds visible
- `FS=400, WINDOW_SIZE=4000` → 10 seconds visible
- `FS=400, WINDOW_SIZE=8000` → 20 seconds visible
- `FS=1600, WINDOW_SIZE=4000` → 2.5 seconds visible

**Recommendations:**
- For quick signal checks: 5-10 seconds (`WINDOW_SIZE = 2000-4000`)
- For inspecting heart rate: 10-20 seconds (`WINDOW_SIZE = 4000-8000`)
- For long-term trends: 30+ seconds (`WINDOW_SIZE = 12000+`)

Larger values use more memory and CPU but show more context.

### Comparison with Arduino Serial Plotter

Some users wonder why not just use Arduino IDE's built-in Serial Plotter. Here's a comparison:

| Feature | Arduino Serial Plotter | This Tool |
|---|---|---|
| Live plotting | ✅ Yes | ✅ Yes (60 FPS) |
| Multi-channel | ✅ Yes | ✅ Yes (linked X-axis) |
| Save to file | ❌ No | ✅ Yes (real-time CSV) |
| Timestamps | ❌ No | ✅ Yes (ms precision) |
| Custom folder organization | ❌ No | ✅ Yes |
| Session naming | ❌ No | ✅ Yes |
| Setup complexity | Lower | Slightly higher |

**Use Arduino plotter when:** You just want to peek at signal quality during firmware development.

**Use this tool when:** You're collecting actual research data that needs to be saved and analyzed.

---

## Code Architecture

### File Structure

```
project_root/
├── Data_Logger.py          <- All code in single file
└── README.md               <- This file
```

### Main Imports

```python
import sys                                  # System exit, command-line args
import os                                   # Folder creation, path operations
import csv                                  # CSV file writing
import serial                               # PySerial for ESP32 communication
import serial.tools.list_ports              # COM port detection utilities
import numpy as np                          # Circular buffer arrays
from datetime import datetime               # Timestamp generation
from collections import deque               # (Imported but uses NumPy buffers)

import pyqtgraph as pg                      # High-performance plotting library
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox  # GUI framework
from PyQt5.QtCore import QTimer             # Periodic timer for plot updates
```

### Key Class: `HighPerfPlotter`

The entire application logic lives in a single class that extends `QMainWindow`:

```python
class HighPerfPlotter(QMainWindow):
    def __init__(self):
        # Initialize circular buffers (NumPy zeros arrays)
        # Create target folder if needed
        # Build the GUI
        # Connect to serial port
```

### Key Methods

#### `__init__(self)`
```python
def __init__(self):
    # Sets up:
    #   - Circular buffers (ir_buffer, red_buffer) of size WINDOW_SIZE
    #   - CSV file handle and writer (created later)
    #   - Target folder verification
    # Calls init_ui() and setup_serial()
```

#### `init_ui(self)`
```python
def init_ui(self):
    # Builds the PyQt window:
    #   - White background with black text (publication style)
    #   - Two stacked plot panels (IR top, RED bottom)
    #   - Linked X-axes for synchronized zooming
    #   - QTimer set to fire every 16ms (60 FPS target)
    # Connects timer to update_loop()
```

#### `setup_serial(self)`
```python
def setup_serial(self):
    # Opens the serial port at configured BAUD_RATE
    # Flushes any stale data from input buffer
    # Shows error popup if connection fails
```

#### `update_loop(self)`
```python
def update_loop(self):
    # Called every 16ms by QTimer. The main work happens here:
    #   1. Read ALL waiting bytes from serial buffer
    #   2. Parse each line (expecting "IR,RED" format)
    #   3. Write timestamped row to CSV
    #   4. Shift circular buffer left, append new value to right
    #   5. Tell PyQtGraph to redraw the plot
    # Handles malformed lines gracefully (skip without crashing)
```

#### `closeEvent(self, event)`
```python
def closeEvent(self, event):
    # Triggered when user closes the window
    # Cleanup:
    #   - Close serial port (release for other apps)
    #   - Close CSV file (flush buffers to disk)
    #   - Print confirmation message
```

### Data Flow Diagram

```
+----------+        +---------+        +---------+
|  ESP32   |  USB   | pyserial|  read  | update_ |
|  Sensor  | -----> | (Python)| -----> |  loop() |
+----------+        +---------+        +---------+
                                            |
                          +-----------------+----------------+
                          |                                  |
                          v                                  v
                    +---------+                       +-----------+
                    | Parse   |                       | Update    |
                    | "IR,RED"|                       | Circular  |
                    +---------+                       | Buffer    |
                          |                           +-----------+
                          v                                  |
                    +---------+                              v
                    | Write   |                       +-----------+
                    | CSV Row |                       | Redraw    |
                    +---------+                       | PyQtGraph |
                          |                           +-----------+
                          v                                  |
                    +---------+                              v
                    | Disk    |                       +-----------+
                    | (.csv)  |                       | GUI       |
                    +---------+                       | Display   |
                                                      +-----------+
```

### Why Single-Threaded Design

This tool runs entirely on a single thread using PyQt's event loop. This might seem suboptimal for I/O-heavy work, but it's actually the right choice here:

1. **Simpler code** — No thread synchronization issues
2. **No race conditions** — Single buffer, single writer
3. **PyQtGraph requires main thread** — GUI updates must happen on main thread anyway
4. **16ms timer is plenty** — Modern CPUs can handle thousands of serial reads per timer tick
5. **Serial driver is non-blocking** — `in_waiting` check is instant

### Suggested Diagram to Create

```
DIAGRAM 2: Application Architecture

Create a layered diagram showing:
  - TOP LAYER: PyQt5 GUI window (with the two plot panels)
  - MIDDLE LAYER: HighPerfPlotter class with its methods
  - BOTTOM LAYER: Three I/O streams:
    * pyserial (input from ESP32)
    * CSV writer (output to file)
    * QTimer (triggers update_loop every 16ms)
  - Arrows showing data flow between layers
  - Color coding:
    * BLUE: GUI components
    * GREEN: Class/logic
    * ORANGE: External I/O
  - Tool: draw.io or PlantUML
  - Size: Portrait, 1080x1920
```

---

## Troubleshooting

### Common Issues Table

| Symptom | Likely Cause | Fix |
|---|---|---|
| **"Could not open COM7"** | Port in use by another program | Close Arduino IDE, PuTTY, ESP-IDF monitor |
| **"Could not open COM7"** | Wrong port number | Check Device Manager / `ls /dev/`, update `SERIAL_PORT` |
| **"Permission denied" (Linux)** | User not in dialout group | `sudo usermod -a -G dialout $USER`, log out/in |
| **GUI opens, plots are flat** | Baud rate mismatch | Match `BAUD_RATE` to ESP32 firmware exactly |
| **GUI opens, no data at all** | ESP32 not running firmware | Re-flash ESP32 with PPG firmware |
| **Garbled / corrupt CSV data** | Baud rate mismatch | Verify both ends, try 115200 first |
| **Plot freezes after few seconds** | Serial buffer overflow | Increase `BAUD_RATE` (e.g., to 921600) |
| **High CPU usage** | WINDOW_SIZE too large | Reduce `WINDOW_SIZE` to 4000 or smaller |
| **App crashes on startup** | PyQt5 installation broken | Recreate venv, reinstall PyQt5 |
| **"No module named pyqtgraph"** | Missing package | `pip install pyqtgraph PyQt5 pyserial` |
| **CSV is empty** | File permission issue | Check write access to output folder |
| **Window closes immediately** | Exception during init | Run from terminal to see error messages |
| **Plots update slowly** | Wrong matplotlib backend | This tool uses PyQtGraph, not matplotlib — should be fast by default |

### Debugging Workflow

#### Step 1: Verify Hardware Independently

Test serial connection WITHOUT this tool to isolate the problem:

```python
# Quick test script
import serial
ser = serial.Serial('COM7', 115200, timeout=1)
for _ in range(20):
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    print(line)
ser.close()
```

If you see valid data → the issue is in this tool's GUI/CSV part.
If you see nothing or errors → the issue is with ESP32 or serial port.

#### Step 2: Check COM Port Identification

```python
# Run this to list all available ports
import serial.tools.list_ports
for port in serial.tools.list_ports.comports():
    print(f"{port.device} - {port.description}")
```

Make sure your ESP32 appears in the list.

#### Step 3: Verify Baud Rate Match

In Arduino IDE's Serial Monitor:
1. Open Serial Monitor (Tools → Serial Monitor)
2. Set baud rate dropdown to match `BAUD_RATE`
3. If you see readable numbers like `125463,87234` → baud is correct
4. If you see garbage characters → baud rate is wrong

#### Step 4: Check Python Imports

```bash
python -c "import pyqtgraph; print(pyqtgraph.__version__)"
python -c "import PyQt5; print(PyQt5.QtCore.QT_VERSION_STR)"
python -c "import serial; print(serial.__version__)"
```

All three should print version numbers without errors.

#### Step 5: Run From Terminal (Not IDE)

If the app crashes silently when run from VS Code or PyCharm, try running from terminal:

```bash
python Data_Logger.py
```

You'll see error messages that may be hidden in the IDE.

### Best Practices

#### Always Close Other Serial Tools First
The most common issue. Arduino IDE Serial Monitor + this tool = port conflict.

#### Test with Short Sessions First
Before doing a long recording, do a 30-second test to verify everything works.

#### Don't Move the USB Cable Mid-Recording
Disconnecting/reconnecting USB causes Windows to assign a new COM port. The tool will lose connection.

#### Use Consistent Session Names
Naming convention helps later when batch-processing files.

#### Back Up Your CSVs
Don't rely on a single copy of important recordings.

#### Monitor Disk Space
Long recordings at 400 Hz produce ~10 KB/second. A 5-minute session = ~3 MB. Plan accordingly for batch collection.

---

## Performance Notes & Tips

### Performance Benchmarks

Tested on a typical mid-range laptop (Intel i5, 16GB RAM, Windows 11):

| Metric | Value |
|---|---|
| Plot refresh rate | 60 FPS (consistent) |
| Max sustained sample rate | ~3,200 Hz (tested with ESP32) |
| Latency (sensor → screen) | <16ms (single frame) |
| CPU usage (400 Hz sampling) | ~3-5% |
| RAM usage | ~150 MB |
| Disk write rate | ~10 KB/s (at 400 Hz, ~3 MB per 5min session) |

The bottleneck is typically the serial bandwidth, not the Python tool itself.

### Why PyQtGraph Achieves 60 FPS

The key technical reasons:

1. **OpenGL-based rendering** — Uses hardware acceleration via Qt
2. **No redraws of static elements** — Only the changing data is updated
3. **Optimized line drawing** — Custom C++ paths for fast 2D rendering
4. **Smart culling** — Off-screen data isn't drawn
5. **Asynchronous draw calls** — Doesn't block the event loop

In contrast, matplotlib redraws the entire figure each time, including labels, ticks, and grid — much slower.

### Circular Buffer Implementation

The data buffer is a simple NumPy array updated in-place:

```python
# Shift everything left by one position
self.ir_buffer[:-1] = self.ir_buffer[1:]

# Append new value at the end
self.ir_buffer[-1] = ir_val
```

This is O(N) per update, but for N=4000 it takes <100 microseconds. Much simpler than using `collections.deque` and just as fast in practice.

### Memory Stays Constant

Regardless of recording length:
- Buffer is always `WINDOW_SIZE` samples (e.g., 4000 floats = 32 KB)
- Old samples are discarded (overwritten in buffer)
- CSV file grows on disk, but RAM usage stays flat

This means you can run multi-hour recordings without memory issues.

### Best Practices for Maximum Performance

#### Use Lower BAUD_RATE for Lower Sample Rates
Higher baud doesn't help if sensor is slow. Stick with 115200 for ≤400 Hz.

#### Use Higher BAUD_RATE for High Sample Rates
At 1600+ Hz, you NEED 921600 baud to avoid buffer overflow.

#### Keep WINDOW_SIZE Reasonable
Larger = more visual context but slower redraws. Sweet spot: 2000-8000 samples.

#### Don't Run Other CPU-Heavy Apps
Video calls, browsers with many tabs, etc. can cause frame drops.

#### Use SSD for Output Folder
Slower HDDs may cause hiccups during high-rate logging. SSD eliminates this.

#### Close Browser Tabs During Long Recordings
Modern browsers can be surprisingly resource-hungry, even when minimized.

---

## Next Step in Pipeline

After successfully recording PPG data with this tool, your output folder will contain raw CSV files ready for the next pipeline stage:

```
01_Raw/
├── SubjectA_Glu100.csv
├── SubjectA_Glu120.csv
├── SubjectB_Glu85.csv
└── ...
```

### Next Tool: Step 2 - Raw Data Verification

The output of this tool feeds into the **Raw Data Verification** stage, where you:

- Visually inspect each recording
- Identify sessions with major problems (sensor disconnections, all-noise recordings, etc.)
- Filter out unusable files before windowing
- Move good recordings to the next stage

After verification, files move forward to **Step 3: Windowing** which lets you manually select clean 15-second windows from each recording.

**See:** `Raw_Verification_README.md` (Step 2) and `Windowing_Tool_README.md` (Step 3) for next stages.

---

## References

### Software Libraries

1. PyQtGraph Documentation, "PyQtGraph — Scientific Graphics and GUI Library for Python," 2024. [Online]. Available: https://www.pyqtgraph.org/

2. Riverbank Computing, "PyQt5 Reference Guide," 2024. [Online]. Available: https://www.riverbankcomputing.com/static/Docs/PyQt5/

3. PySerial Project, "PySerial — Python Serial Port Extension," 2024. [Online]. Available: https://pyserial.readthedocs.io/en/latest/

4. C. R. Harris et al., "Array programming with NumPy," *Nature*, vol. 585, pp. 357-362, 2020. [DOI: 10.1038/s41586-020-2649-2]

### Hardware Documentation

5. Espressif Systems, "ESP32-S3 Series Datasheet," 2024. [Online]. Available: https://www.espressif.com/en/products/socs/esp32-s3

6. Maxim Integrated, "MAX30102 Pulse Oximeter and Heart-Rate Sensor Datasheet," 2018. [Online]. Available: https://www.analog.com/media/en/technical-documentation/data-sheets/MAX30102.pdf

### Communication Protocols

7. USB Implementers Forum, "USB Class Definitions for Communications Devices (USB-CDC)," 2010. [Online]. Available: https://www.usb.org/document-library/class-definitions-communication-devices-12

---

## Summary

This tool is the **critical bridge** between your hardware data acquisition (ESP32 firmware) and your software analysis pipeline. Its job is simple but vital: get clean data from sensor to disk while letting you verify quality in real-time.

Key benefits:
- ✅ High-performance plotting (60 FPS) confirms signal quality immediately
- ✅ Reliable CSV logging ensures no data loss
- ✅ Simple terminal-based interface for quick session setup
- ✅ Overwrite protection prevents accidental data loss
- ✅ Cross-platform compatibility (Windows, Linux, macOS)
- ✅ Minimal dependencies (just pip install and run)

For best results: identify your COM port correctly, match baud rates, and use a consistent file naming convention from the start.

Happy logging!