# ESP32-S3 PPG Signal Acquisition Firmware

> Embedded firmware for high-resolution photoplethysmography (PPG) data acquisition using the MAX30102 sensor on ESP32-S3, designed as the data source for non-invasive blood glucose estimation research.

---

## TL;DR

This C firmware runs on an ESP32-S3 microcontroller and continuously samples a MAX30102 pulse oximeter sensor at configurable rates (50 Hz to 3200 Hz). It outputs raw 18-bit RED and IR channel data via UART in CSV format (`IR,RED\n`) for real-time visualization or capture into Python pipelines.

**Quick Stats:**
- ~250 lines of C code
- 18-bit ADC resolution (best for glucose feature extraction)
- Configurable sample rates: 50 Hz to 3200 Hz
- Independent RED/IR LED current control
- Up to 921,600 baud UART output
- I2C @ 400 kHz for fast FIFO reads
- FreeRTOS task-based architecture

---

## Table of Contents

1. [TL;DR](#tldr)
2. [Quick Start](#quick-start)
3. [Background & Motivation](#background--motivation)
4. [System Overview](#system-overview)
5. [Features & Capabilities](#features--capabilities)
6. [Hardware Requirements & Wiring](#hardware-requirements--wiring)
7. [Sensor Theory & Register Map](#sensor-theory--register-map)
8. [Installation & Build Environment](#installation--build-environment)
9. [Flashing & First Run](#flashing--first-run)
10. [Configuration Reference](#configuration-reference)
11. [Output Format & Integration](#output-format--integration)
12. [Tuning History (Real Experiments)](#tuning-history-real-experiments)
13. [Code Architecture](#code-architecture)
14. [Troubleshooting & Tuning Guide](#troubleshooting--tuning-guide)
15. [Future Enhancements](#future-enhancements)

---

## Quick Start

### Hardware Shopping List

| Item | Quantity | Notes |
|---|---|---|
| ESP32-S3 DevKit | 1 | DevKitC-1 or compatible |
| MAX30102 breakout module | 1 | Most common: GY-MAX30102 |
| USB-C cable | 1 | For flashing and power |
| Jumper wires (female-female) | 4 | For I2C connection |

### 4-Wire Connection

```
ESP32-S3      MAX30102
--------      --------
3.3V    --->  VIN
GND     --->  GND
GPIO 1  --->  SDA
GPIO 2  --->  SCL
```

### Software Flashing

```bash
# 1. Set target chip
idf.py set-target esp32s3

# 2. Build the firmware
idf.py build

# 3. Flash and start monitoring
idf.py flash monitor
```

### Expected First Serial Output

```
I (XXX) PPG_TUNER: Sensor Configured:
I (XXX) PPG_TUNER: - I2C Speed: 400000 Hz
I (XXX) PPG_TUNER: - LED RED Current: 10.0 mA (Reg: 0x32)
I (XXX) PPG_TUNER: - LED IR  Current: 7.0 mA (Reg: 0x23)
I (XXX) PPG_TUNER: - SpO2 Config Reg (0x0A): 0x2F
0,0
0,0
125463,87234     <- Finger placed!
126890,88456
128122,89745
...
```

If you see `0,0` even with finger on sensor, increase LED currents or lower `FINGER_THRESHOLD`.

### Common First-Time Issues

| Problem | Quick Fix |
|---|---|
| No serial output | Verify baud rate matches (default 115200) |
| Always shows `0,0` | Lower `FINGER_THRESHOLD` from 50000 to 30000 |
| Signal saturated (262143) | Reduce `LED_CURRENT_RED_MA` and `LED_CURRENT_IR_MA` |
| Build errors | Check ESP-IDF version is v5.5.1 |

---

## Background & Motivation

### Why PPG Sensing?

Photoplethysmography (PPG) is a non-invasive optical technique that detects blood volume changes in tissue. By illuminating the finger with RED (660 nm) and Infrared (880 nm) light and measuring how much is reflected back, we can extract information about:

- Heart rate and rhythm
- Blood oxygen saturation (SpO2)
- Pulse waveform morphology
- Vascular characteristics
- Potentially: glucose concentration (active research area)

### Why MAX30102?

The MAX30102 is purpose-built for PPG applications:
- Integrated RED + IR LEDs + photodiode in one package
- 18-bit ADC resolution (high dynamic range)
- Built-in FIFO buffer (32 samples deep)
- I2C interface (simple wiring)
- Industry-standard for wearable health devices

### Why ESP32-S3?

The ESP32-S3 was chosen for:
- Native USB support (no extra USB-to-Serial chip needed)
- Dual-core for future expansion (one core for sensor, one for processing)
- WiFi + Bluetooth for future wireless features
- FreeRTOS for clean task management
- Powerful enough to handle real-time signal processing locally (future use case)
- Affordable and well-documented

### Where This Fits in the Glucose Pipeline

```
[MAX30102] -> [THIS FIRMWARE] -> [UART Stream] -> [Python Receiver] -> [Signal Processing] -> [ML Model] -> [Glucose Prediction]
                    ^
               You are here
```

This firmware is the **data acquisition layer** — the foundation of the entire pipeline. Without clean, consistent raw data from this stage, all downstream processing and ML becomes unreliable.

---

## System Overview

### Hardware Block Diagram

```
   +-------------+        +-------------+        +-------------+
   |   Finger    |        |  MAX30102   |        |  ESP32-S3   |
   | (on sensor) | ===>   |   Sensor    | <===>  |    MCU      |
   +-------------+        +-------------+   I2C  +------+------+
                            ^      |                    |
                            |      v                    | UART
                          RED   IR LED              (Serial)
                          LED   Photodiode               v
                                                  +-----------+
                                                  |    PC     |
                                                  | (Plotter  |
                                                  |  / Python)|
                                                  +-----------+
```

### Software Architecture

```
+---------------------------------------------------+
|                  app_main()                       |
|  +---------------------------------------------+  |
|  | 1. I2C Driver Initialization               |  |
|  | 2. UART Baud Rate Configuration            |  |
|  | 3. MAX30102 Sensor Setup                   |  |
|  | 4. Create FreeRTOS Task (ppg_task)         |  |
|  +---------------------------------------------+  |
|                       |                           |
|                       v                           |
|  +---------------------------------------------+  |
|  |          ppg_task() [FreeRTOS Task]        |  |
|  |  +----------------------------------------+ |  |
|  |  | Loop forever:                          | |  |
|  |  |  1. Read FIFO pointers (I2C)           | |  |
|  |  |  2. Calculate samples available        | |  |
|  |  |  3. Read sample bytes from FIFO (I2C)  | |  |
|  |  |  4. Parse 18-bit RED + IR values       | |  |
|  |  |  5. Check finger presence threshold    | |  |
|  |  |  6. Output via printf to UART          | |  |
|  |  |  7. Delay 10ms, repeat                 | |  |
|  |  +----------------------------------------+ |  |
|  +---------------------------------------------+  |
+---------------------------------------------------+
```

### Mermaid Data Flow

```mermaid
flowchart LR
    A[Finger on Sensor] --> B[MAX30102 LEDs + Photodiode]
    B --> C[18-bit ADC]
    C --> D[Internal FIFO Buffer]
    D --> E[I2C @ 400kHz]
    E --> F[ESP32-S3 FreeRTOS Task]
    F --> G[printf to UART]
    G --> H[USB-CDC to PC]
    H --> I[Serial Plotter / Python Receiver]
```

### Suggested Diagram to Create

```
DIAGRAM 1: Complete System Block Diagram

Create a colorful block diagram showing:
  - LEFT: Finger illustration with MAX30102 sensor
  - CENTER: ESP32-S3 board with labeled GPIO pins
  - RIGHT: PC/Laptop with serial plotter screenshot
  - Arrows showing data flow (left to right)
  - Color coding:
    * BLUE: Power lines (3.3V, GND)
    * GREEN: I2C lines (SDA, SCL)
    * ORANGE: USB/UART connection
    * RED: Signal flow
  - Include voltage/frequency labels
  - Tool suggestion: Fritzing, draw.io, or Visio
  - Size: Landscape, 1920x1080
```

---

## Features & Capabilities

### Core Functionality

- **Real-time PPG acquisition** — Continuous sampling with no data loss
- **Dual-channel output** — RED and IR simultaneously
- **18-bit resolution** — Maximum sensitivity for subtle waveform features
- **FIFO-based buffering** — Hardware buffer prevents missed samples
- **Configurable everything** — Sample rate, LED currents, ADC range, baud rate

### Configurability (at compile time)

- **Independent RED/IR LED currents** (0.0 to 51.0 mA each)
- **Four ADC sensitivity ranges** (2048 to 16384 nA)
- **Eight sample rates** (50 Hz to 3200 Hz)
- **Four pulse widths / resolutions** (15-bit to 18-bit)
- **Six FIFO averaging modes** (1 to 32 samples)
- **Three UART baud rates** (115200 to 921600)

### Quality & Reliability

- **Finger detection** via amplitude threshold
- **I2C error logging** to terminal
- **FreeRTOS task isolation** — sensor task can't crash main thread
- **FIFO rollover enabled** — handles overflow gracefully

### Output Format

- **Plain CSV stream** — `IR,RED\n` per sample
- **Compatible with Arduino Serial Plotter** — Instant visualization
- **Python-friendly** — Simple `pyserial` parsing
- **Zero overhead protocol** — No framing, no checksums (assumes clean USB connection)

---

## Hardware Requirements & Wiring

### Bill of Materials (BOM)

| Component | Part Number | Notes |
|---|---|---|
| Microcontroller | ESP32-S3-DevKitC-1 | Or any ESP32-S3 board with USB |
| PPG Sensor | MAX30102 breakout (GY-MAX30102) | Most common module |
| USB Cable | USB Type-C to USB Type-A | For flashing + power + serial |
| Jumper Wires | Female-Female, 4 pieces | 10cm length recommended |
| (Optional) Breadboard | Half-size | If not soldering directly |

### Pin Assignments (ESP32-S3 Side)

| Function | ESP32-S3 Pin | Reason |
|---|---|---|
| I2C SDA | GPIO 1 | Configurable via `I2C_MASTER_SDA_IO` |
| I2C SCL | GPIO 2 | Configurable via `I2C_MASTER_SCL_IO` |
| 3.3V Power | 3V3 pin | Powers MAX30102 |
| Ground | GND pin | Common ground |

### MAX30102 Breakout Pinout

| Pin Label | Function | Connect To |
|---|---|---|
| VIN | Power input (3.3V or 5V tolerant) | ESP32 3V3 |
| GND | Ground | ESP32 GND |
| SDA | I2C Data | ESP32 GPIO 1 |
| SCL | I2C Clock | ESP32 GPIO 2 |
| INT | Interrupt (not used) | Leave disconnected |
| RD | Red LED enable (auto) | Leave disconnected |
| IRD | IR LED enable (auto) | Leave disconnected |

### Complete Wiring Table

```
+-----------+----------+-----------+
| ESP32-S3  | Wire     | MAX30102  |
+-----------+----------+-----------+
| 3V3       | RED      | VIN       |
| GND       | BLACK    | GND       |
| GPIO 1    | GREEN    | SDA       |
| GPIO 2    | YELLOW   | SCL       |
+-----------+----------+-----------+
```

### Pull-Up Resistors

The MAX30102 breakout module **typically includes built-in 4.7kΩ pull-up resistors** on SDA and SCL. The ESP32 firmware also enables internal pull-ups via:

```c
.sda_pullup_en = GPIO_PULLUP_ENABLE,
.scl_pullup_en = GPIO_PULLUP_ENABLE,
```

**For most users:** No external pull-ups needed.

**If you experience I2C errors:** Add external 2.2kΩ or 4.7kΩ pull-ups from SDA → 3V3 and SCL → 3V3.

### Power Considerations

- MAX30102 max current draw: ~30 mA (depends on LED settings)
- ESP32-S3 supplies up to 500 mA on 3V3 pin (more than enough)
- Total system draw: ~80-150 mA via USB
- No external power supply needed for development

### Suggested Diagram to Create

```
DIAGRAM 2: Wiring Diagram (Fritzing-style)

Create a realistic breadboard-style wiring diagram showing:
  - ESP32-S3 DevKit (with USB-C port visible)
  - MAX30102 breakout board
  - 4 colored wires (red, black, green, yellow) connecting them
  - Pin labels on both boards
  - Top-down view, no perspective
  - Tool: Fritzing (free), or draw.io with electronics shapes
  - Size: Landscape, 1920x1080
```

---

## Sensor Theory & Register Map

### How the MAX30102 Works

The sensor contains three key components in a tiny package:

1. **Red LED** (660 nm wavelength) — Penetrates ~3 mm into tissue
2. **Infrared LED** (880 nm wavelength) — Penetrates ~5 mm into tissue
3. **Photodiode** — Measures reflected light intensity

During each measurement cycle:
1. Red LED pulses on for `PULSE_WIDTH` microseconds
2. Photodiode captures reflected red light → ADC reading
3. Red LED off, IR LED pulses on
4. Photodiode captures reflected IR light → ADC reading
5. Both readings pushed into 32-sample FIFO buffer
6. Repeat at configured `SAMPLE_RATE`

When blood pulses through capillaries, more light is absorbed → less reflected → ADC reading decreases. This produces the inverted pulsatile waveform we call PPG.

### FIFO Architecture

```
+-------------------------------------------------------+
|              MAX30102 FIFO Buffer (32 deep)           |
|  +------+------+------+------+      +------+         |
|  |      |      |      |      | ...  |      |         |
|  | RD+IR| RD+IR| RD+IR| RD+IR|      | RD+IR|         |
|  +------+------+------+------+      +------+         |
|     ^                                   ^             |
|  Write Ptr                          Read Ptr          |
+-------------------------------------------------------+
                       |
                       | I2C reads
                       v
                 +----------+
                 | ESP32-S3 |
                 +----------+
```

The ESP32 reads FIFO pointers via register 0x04 to calculate how many new samples are available, then reads sample data from register 0x07.

### Key Register Summary

| Register | Address | Purpose |
|---|---|---|
| Mode Configuration | 0x09 | Sets operating mode (SpO2 = both LEDs) |
| SpO2 Configuration | 0x0A | Sets ADC range, sample rate, pulse width |
| LED1 PA | 0x0C | Red LED current (PA = Pulse Amplitude) |
| LED2 PA | 0x0D | IR LED current |
| FIFO Config | 0x08 | Sample averaging, rollover behavior |
| FIFO Write Ptr | 0x04 | Tells us how many samples available |
| FIFO Read Ptr | 0x06 | Position we've read up to |
| FIFO Data | 0x07 | Sample data (6 bytes per sample: 3 RED + 3 IR) |

### Register 0x0A Bit Layout (SpO2 Configuration)

```
+--------+--------+--------+--------+--------+--------+--------+--------+
|  Bit 7 |  Bit 6 |  Bit 5 |  Bit 4 |  Bit 3 |  Bit 2 |  Bit 1 |  Bit 0 |
+--------+--------+--------+--------+--------+--------+--------+--------+
|        |     ADC Range   |       Sample Rate        |  Pulse Width    |
|  RSVD  |     (2 bits)    |        (3 bits)          |    (2 bits)     |
+--------+-----------------+--------------------------+-----------------+
```

**Example calculation:** With `ADC_RANGE_OPT=1`, `SAMPLE_RATE_OPT=3`, `PULSE_WIDTH_OPT=3`:
```
spo2_conf = (1 << 5) | (3 << 2) | 3 = 0x20 | 0x0C | 0x03 = 0x2F
```

### Register 0x08 Bit Layout (FIFO Configuration)

```
+--------+--------+--------+--------+--------+--------+--------+--------+
|  Bit 7 |  Bit 6 |  Bit 5 |  Bit 4 |  Bit 3 |  Bit 2 |  Bit 1 |  Bit 0 |
+--------+--------+--------+--------+--------+--------+--------+--------+
|     Sample Average        | RLO_EN |     FIFO Almost Full Threshold   |
|        (3 bits)           |        |             (4 bits)             |
+---------------------------+--------+----------------------------------+

RLO_EN = FIFO Rollover Enable (1 = overwrite oldest when full)
```

### LED Current Conversion

The firmware converts mA to register value using:

```
Register Value = Current_mA / 0.2
```

| Desired Current | Register Value (hex) |
|---|---|
| 0.4 mA | 0x02 |
| 5.0 mA | 0x19 |
| 7.0 mA | 0x23 |
| 10.0 mA | 0x32 |
| 15.0 mA | 0x4B |
| 25.0 mA | 0x7D |
| 51.0 mA (max) | 0xFF |

---

## Installation & Build Environment

### Prerequisites

| Requirement | Recommended Version |
|---|---|
| ESP-IDF | v5.5.1 |
| Python | 3.8+ (auto-installed with ESP-IDF) |
| Git | Latest |
| USB Drivers | CP210x or CH340 (depends on board) |

### Installing ESP-IDF v5.5.1

#### Windows

1. Download the ESP-IDF Windows Installer (v5.5.1) from:
   ```
   https://dl.espressif.com/dl/esp-idf/
   ```

2. Run installer, select:
   - Install path: `C:\esp\esp-idf`
   - Python and tools: Install bundled versions
   - Add `idf.py` to PATH: Yes

3. After install, open "ESP-IDF 5.5.1 PowerShell" from Start Menu.

#### Linux / macOS

```bash
# 1. Clone ESP-IDF
mkdir -p ~/esp
cd ~/esp
git clone -b v5.5.1 --recursive https://github.com/espressif/esp-idf.git

# 2. Install tools
cd esp-idf
./install.sh esp32s3

# 3. Set up environment (add to ~/.bashrc or ~/.zshrc)
alias get_idf='. $HOME/esp/esp-idf/export.sh'

# 4. Activate environment
get_idf
```

### Project Structure Expected

```
your_project_folder/
├── CMakeLists.txt
├── main/
│   ├── CMakeLists.txt
│   └── main.c              <- The firmware code
└── sdkconfig (auto-generated)
```

### Build Commands

```bash
# Navigate to project folder
cd your_project_folder

# Set target chip (first time only)
idf.py set-target esp32s3

# (Optional) Open configuration menu
idf.py menuconfig

# Build firmware
idf.py build

# Combined flash + monitor
idf.py flash monitor

# Just flash (no monitor)
idf.py flash

# Just monitor existing firmware
idf.py monitor

# Clean build artifacts
idf.py fullclean
```

### Optional: VS Code Setup

Install the **ESP-IDF VS Code Extension** for:
- One-click build/flash/monitor buttons
- Integrated terminal with ESP-IDF environment
- Syntax highlighting for ESP-IDF APIs
- Built-in serial monitor with timestamps

### Common Build Errors

| Error | Cause | Fix |
|---|---|---|
| `idf.py: command not found` | Environment not sourced | Run `get_idf` or open ESP-IDF terminal |
| `CMake Error: target esp32s3` | Wrong target set | Run `idf.py set-target esp32s3` |
| `uart.h not found` | Missing component | Add `driver` to `REQUIRES` in CMakeLists |
| `Failed to flash` | Wrong port | Specify port: `idf.py -p COM5 flash` |

---

## Flashing & First Run

### Step 1: Connect Hardware

1. Wire the MAX30102 to ESP32-S3 per the wiring table
2. Connect USB-C cable from ESP32 to PC
3. Power LED on ESP32 should illuminate

### Step 2: Identify COM Port

**Windows:**
```powershell
# In ESP-IDF PowerShell
mode | findstr COM
```

**Linux:**
```bash
ls /dev/ttyUSB* /dev/ttyACM*
# Example output: /dev/ttyUSB0
```

**macOS:**
```bash
ls /dev/cu.*
# Example output: /dev/cu.usbserial-0001
```

### Step 3: Flash Firmware

```bash
# Auto-detect port
idf.py flash monitor

# Specify port manually if needed
idf.py -p COM5 flash monitor          # Windows
idf.py -p /dev/ttyUSB0 flash monitor  # Linux
idf.py -p /dev/cu.usbserial-0001 flash monitor  # macOS
```

You should see compilation messages, then flashing progress:

```
Connecting...
Chip is ESP32-S3 (revision v0.1)
Flash size: 8MB
Compressed 142336 bytes to 96872...
Wrote 142336 bytes (96872 compressed) at 0x00010000 in 1.2 seconds...
Hash of data verified.

Leaving...
Hard resetting via RTS pin...
```

### Step 4: Live Serial Output

After flashing, the monitor starts automatically. You'll see:

```
I (315) cpu_start: Application information:
I (321) cpu_start: Project name:     ppg_acquisition
I (326) cpu_start: ESP-IDF:          v5.5.1
I (XXX) PPG_TUNER: Sensor Configured:
I (XXX) PPG_TUNER: - I2C Speed: 400000 Hz
I (XXX) PPG_TUNER: - LED RED Current: 10.0 mA (Reg: 0x32)
I (XXX) PPG_TUNER: - LED IR  Current: 7.0 mA (Reg: 0x23)
I (XXX) PPG_TUNER: - SpO2 Config Reg (0x0A): 0x2F

0,0
0,0
[place finger on sensor]
85234,124612
86120,125892
87445,127103
...
```

### Step 5: Exit Monitor

Press `Ctrl + ]` to exit the serial monitor.

---

## Configuration Reference

All tunable parameters are at the top of `main.c` in the **USER CONFIGURATION SECTION**. Change values and rebuild (`idf.py build flash`) to apply.

### 1. I2C Speed

```c
#define I2C_SPEED_HZ 400000
```

| Value | Description |
|---|---|
| `100000` | Standard mode (100 kHz) — works but slow |
| `400000` | **Fast mode (400 kHz) — RECOMMENDED** |
| `1000000` | Fast-mode Plus (1 MHz) — may not work with all sensors |

**When to change:** Keep at 400 kHz. Higher speeds risk corruption with longer wires.

### 2. LED Currents (Independent Channels)

```c
#define LED_CURRENT_RED_MA 10.0
#define LED_CURRENT_IR_MA  7.0
```

| Value | Description |
|---|---|
| `0.0 - 5.0` | Very low — for very thin skin or signal saturation testing |
| `5.0 - 15.0` | **Normal range — RECOMMENDED** |
| `15.0 - 30.0` | High — for darker skin tones or thick fingers |
| `30.0 - 51.0` | Maximum — only if signal is too weak otherwise |

**When to change:**
- Signal saturated (all 262143) → Lower LED current
- Signal too weak (< 50000) → Raise LED current
- RED and IR can be tuned independently for balance

### 3. ADC Range (Sensitivity)

```c
#define ADC_RANGE_OPT 1
```

| Value | Range | Sensitivity |
|---|---|---|
| `0` | 2048 nA | Most sensitive (easy to saturate) |
| `1` | 4096 nA | **RECOMMENDED start** |
| `2` | 8192 nA | Less sensitive |
| `3` | 16384 nA | Least sensitive (hard to saturate) |

**When to change:** Increase if signal saturates even at low LED currents.

### 4. Sample Rate

```c
#define SAMPLE_RATE_OPT 3
```

| Value | Sample Rate | Use Case |
|---|---|---|
| `0` | 50 Hz | Minimum (heart rate only) |
| `1` | 100 Hz | Basic morphology |
| `2` | 200 Hz | Good detail |
| `3` | **400 Hz** | **RECOMMENDED — matches training data** |
| `4` | 800 Hz | High detail (uses more bandwidth) |
| `5` | 1000 Hz | Very high (resolution may drop) |
| `6` | 1600 Hz | Maximum useful (17-bit resolution) |
| `7` | 3200 Hz | Maximum (low resolution mode) |

**When to change:** Match your training pipeline. The accompanying Python pipeline expects 400 Hz.

### 5. Pulse Width (Resolution)

```c
#define PULSE_WIDTH_OPT 3
```

| Value | Pulse Width | Resolution |
|---|---|---|
| `0` | 69 µs | 15-bit (lowest detail) |
| `1` | 118 µs | 16-bit |
| `2` | 215 µs | 17-bit |
| `3` | **411 µs** | **18-bit — RECOMMENDED for glucose features** |

**When to change:** Keep at 18-bit. Lower resolutions lose subtle dicrotic notch features critical for glucose estimation.

### 6. FIFO Averaging

```c
#define FIFO_AVG_OPT 0
```

| Value | Samples Averaged | Effect |
|---|---|---|
| `0` | 1 (no averaging) | **RECOMMENDED — raw data for downstream processing** |
| `1` | 2 | Mild smoothing |
| `2` | 4 | Moderate smoothing |
| `3` | 8 | Heavy smoothing |
| `4` | 16 | Very heavy smoothing |
| `5` | 32 | Maximum smoothing |

**When to change:** Keep at 0. Smoothing should happen in software pipeline (Step 5-7 of signal processing) for full control.

### 7. UART Baud Rate

```c
#define UART_BAUD_RATE 115200
```

| Value | Speed | Use Case |
|---|---|---|
| `115200` | **Standard — RECOMMENDED for compatibility** |
| `460800` | Fast — 4× speed |
| `921600` | Very fast — supports highest sample rates |

**When to change:**
- High sample rates (800+ Hz) need 460800 or 921600 to avoid serial buffer overflow
- Python receiver must match this exact value

### 8. Finger Detection Threshold

```c
#define FINGER_THRESHOLD 50000
```

**What it does:** If IR reading is below this value, output `0,0` instead of actual readings (indicates no finger on sensor).

**When to change:**
- Always showing `0,0` with finger on → Lower to 30000 or 20000
- Showing readings without finger → Raise to 70000 or 80000

---

## Output Format & Integration

### Serial Output Specification

**Format:** `<IR_value>,<RED_value>\n`

**Example stream:**
```
0,0                    <- No finger (or below threshold)
125463,87234           <- Finger detected, normal reading
126890,88456
128122,89745
0,0                    <- Finger removed
```

### Value Ranges

| Field | Min | Max | Notes |
|---|---|---|---|
| IR value | 0 | 262143 | 18-bit unsigned (2^18 - 1) |
| RED value | 0 | 262143 | 18-bit unsigned |

**Typical resting values (good signal):**
- IR: 80000 - 200000
- RED: 60000 - 180000

**Saturation indicator:** Value at 262143 means ADC saturated — reduce LED current or increase ADC range.

### Compatible Tools

#### Arduino Serial Plotter

1. Open Arduino IDE
2. Tools → Serial Plotter
3. Set baud rate to match `UART_BAUD_RATE` (default 115200)
4. Two waveforms appear automatically (IR blue, RED orange)

#### Python (Brief Reference)

You'll need a Python receiver script using `pyserial` to:
- Open the COM port at matching baud rate
- Read lines, split by comma
- Parse IR and RED as integers
- Store/visualize/forward to processing pipeline

**Note:** A complete Python receiver implementation will be documented in a separate README (Python Receiver / Streaming module).

#### ESP-IDF Built-in Monitor

```bash
idf.py monitor
# Or specific port:
idf.py -p COM5 monitor
```

Note: ESP-IDF monitor prints with timestamps which may interfere with plain CSV plotting. Use Arduino Serial Plotter for cleaner visualization.

### Suggested Diagram to Create

```
DIAGRAM 3: Serial Output Visualization

Create a screenshot/mockup showing:
  - Arduino Serial Plotter window
  - Two PPG waveforms (IR + RED) over time
  - Clear pulsatile heartbeats visible
  - Time axis showing ~5 seconds of data
  - Labels for IR (blue) and RED (orange) channels
  - Annotation: "Finger placed here" with arrow
  - Tool: Real screenshot from Arduino IDE, or mockup in Figma
  - Size: Landscape, 1280x720
```

---

## Tuning History (Real Experiments)

The following table documents real tuning experiments conducted during firmware development. Each row represents a different parameter combination tested for signal quality.

### Tuning Test Log

| Test # | Sample Rate (Hz) | LED Current (mA) | ADC Range (nA) | Pulse Width (µs) | FIFO Average | I2C Speed (kHz) | Result |
|---|---|---|---|---|---|---|---|
| T1 | 100 | 7.2 | 4096 | 411 | 4 | 400 | Selected |
| T2 | 100 | 10 | 4096 | 411 | 4 | 400 | — |
| T3 | 100 | 15 | 4096 | 411 | 4 | 400 | — |
| T4 | 200 | 10 | 4096 | 411 | 4 | 400 | — |
| T5 | 200 | 10 | 4096 | 411 | 2 | 400 | — |
| T6 | 200 | 10 | 4096 | 411 | 8 | 400 | — |
| T7 | 200 | 10 | 8192 | 411 | 4 | 400 | — |
| T8 | 200 | 15 | 8192 | 411 | 4 | 400 | — |
| T9 | 200 | 15 | 2048 | 411 | 4 | 400 | — |
| T10 | 200 | 10 | 2048 | 411 | 4 | 400 | — |
| T11 | 100 | 10 | 2048 | 411 | 4 | 400 | — |
| T12 | **400** | **10** | **4096** | **411** | **4** | **400** | **Selected** |
| T13 | 400 | 15 | 4096 | 411 | 4 | 400 | Selected |
| T14 | 200 | 7.2 | 4096 | 411 | 4 | 400 | — |
| T15 | 200 | 10 | — | — | — | — | (incomplete) |
| T16 | **400** | **7.2** | **4096** | **411** | **4** | **400** | **Selected** |
| T17 | 800 | 15 | 4096 | 411 | 4 | 400 | — |
| T18 | 1600 | 10 | 4096 | 69 | 4 | 400 | — |
| T19 | 1600 | 10 | 4096 | 69 | 0 | 400 | — |
| T20 | 800 | 10 | 4096 | 215 | 0 | 400 | — |
| T21 | **400** | **R=13/10, IR=7** | **4096** | **411** | **0** | **400** | **Selected (Final)** |

### Key Findings From Tuning

- **Sample rate 400 Hz** consistently produced best results (selected in T12, T13, T16, T21)
- **Pulse width 411 µs** (18-bit resolution) preserved morphological detail
- **ADC range 4096 nA** balanced sensitivity vs saturation risk
- **FIFO averaging 0** (no hardware averaging) preserved raw data for downstream processing
- **I2C @ 400 kHz** sufficient bandwidth for all tested sample rates
- **Independent RED/IR currents** (T21) gave best signal balance: RED=13mA, IR=7mA (or RED=10mA per current config)

### Recommended Configuration (Current Firmware Defaults)

Based on tuning experiments, the current firmware ships with:

```c
#define SAMPLE_RATE_OPT     3       // 400 Hz
#define LED_CURRENT_RED_MA  10.0    // Independent control
#define LED_CURRENT_IR_MA   7.0     // Independent control
#define ADC_RANGE_OPT       1       // 4096 nA
#define PULSE_WIDTH_OPT     3       // 411 µs (18-bit)
#define FIFO_AVG_OPT        0       // No averaging
#define I2C_SPEED_HZ        400000  // 400 kHz
#define UART_BAUD_RATE      115200  // Standard
```

---

## Code Architecture

### File Structure

```
project_root/
├── main/
│   ├── main.c              <- All firmware code in single file
│   └── CMakeLists.txt      <- Component build configuration
├── CMakeLists.txt          <- Project-level build configuration
└── sdkconfig               <- ESP-IDF configuration (auto-generated)
```

### Main Imports / Includes

```c
#include "driver/i2c.h"          // I2C master driver
#include "driver/uart.h"         // UART baud rate control
#include "esp_log.h"             // ESP_LOGI/ESP_LOGE for logging
#include "freertos/task.h"       // FreeRTOS task creation
#include <stdbool.h>             // bool type
#include <stdio.h>               // printf for serial output
#include <string.h>              // String utilities
```

### Key Functions

#### `static esp_err_t max30102_write_reg(uint8_t reg, uint8_t val)`
```c
// Writes a single byte to a specified MAX30102 register via I2C.
// Used for configuring sensor mode, LED currents, ADC settings.
```

#### `static esp_err_t max30102_read_reg(uint8_t reg, uint8_t *data, size_t len)`
```c
// Reads multiple bytes starting from a specified register.
// Used for FIFO pointer reads and FIFO data reads.
```

#### `static void max30102_init_sensor(void)`
```c
// Configures the MAX30102 with user-specified settings:
// 1. Soft reset the sensor
// 2. Configure FIFO (averaging, rollover)
// 3. Set mode to SpO2 (both LEDs active)
// 4. Set ADC range, sample rate, pulse width
// 5. Set independent LED currents (RED + IR)
// 6. Log configuration to terminal
```

#### `void ppg_task(void *arg)`
```c
// FreeRTOS task that runs continuously:
// 1. Read FIFO write/read pointers
// 2. Calculate samples available (handles wrap-around)
// 3. Read sample bytes from FIFO (6 bytes = 3 RED + 3 IR)
// 4. Parse 18-bit values from raw bytes
// 5. Apply finger detection threshold
// 6. Output via printf in IR,RED format
// 7. Delay 10ms and repeat
```

#### `void app_main(void)`
```c
// Entry point — runs once at boot:
// 1. Configure I2C peripheral (pins, speed, pull-ups)
// 2. Install I2C driver
// 3. Set UART baud rate to user-specified value
// 4. Initialize MAX30102 sensor
// 5. Create the ppg_task with 4KB stack
```

### FreeRTOS Task Architecture

```
+--------------------------------+
| Boot                           |
+--------------------------------+
              |
              v
+--------------------------------+
| app_main() runs once           |
| - I2C init                     |
| - UART config                  |
| - Sensor init                  |
| - xTaskCreate(ppg_task)        |
+--------------------------------+
              |
              v
+--------------------------------+
| ppg_task() runs forever        |
| Priority: 5                    |
| Stack: 4096 bytes              |
|                                |
| while(1) {                     |
|   read FIFO                    |
|   parse samples                |
|   printf output                |
|   vTaskDelay(10ms)             |
| }                              |
+--------------------------------+
```

The 10 ms polling delay (`vTaskDelay(pdMS_TO_TICKS(10))`) keeps CPU usage low while ensuring no FIFO overflow at sample rates up to ~3000 Hz (32-sample FIFO at 3000 Hz = 10.6 ms before overflow).

### Suggested Diagram to Create

```
DIAGRAM 4: Function Call Hierarchy

Create a hierarchical diagram showing:
  - TOP: app_main()
  - SECOND LEVEL: i2c init, uart config, sensor init, task create
  - THIRD LEVEL (under sensor init): write_reg called for each register
  - PARALLEL: ppg_task running independently
  - UNDER ppg_task: read_reg (pointers), read_reg (FIFO data), printf
  - Arrows showing function calls
  - Color code:
    * BLUE: One-time setup (app_main)
    * GREEN: Continuous loop (ppg_task)
    * ORANGE: I2C helpers
    * PURPLE: Sensor-specific functions
  - Tool: draw.io or PlantUML
  - Size: Portrait, 1080x1920
```

---

## Troubleshooting & Tuning Guide

### Common Issues Table

| Symptom | Likely Cause | Parameter to Adjust |
|---|---|---|
| All zeros output even with finger | Threshold too high | Lower `FINGER_THRESHOLD` (50000 → 30000) |
| Signal saturated (always 262143) | LED current too high or ADC too sensitive | Lower LED current OR increase `ADC_RANGE_OPT` |
| Signal too weak (< 50000) | LED current too low | Increase LED current 5 mA at a time |
| Choppy/dropped samples | Baud rate too slow for sample rate | Increase `UART_BAUD_RATE` to 460800+ |
| Inconsistent sample timing | `vTaskDelay` too long | Reduce delay from 10ms to 5ms |
| Serial output garbled | Baud rate mismatch | Match plotter/receiver baud exactly |
| I2C errors in log | Wiring issue | Check connections, try external pull-ups |
| Sensor not detected at startup | Wrong I2C address | Verify MAX30102_ADDR = 0x57 |
| Crashes on startup | Stack overflow | Increase task stack in xTaskCreate (4096 → 8192) |
| Compilation: `driver/uart.h not found` | ESP-IDF version mismatch | Verify ESP-IDF v5.5.1 installed |

### Debugging Workflow

#### Step 1: Verify Hardware Connection

Run this code to detect the sensor:

```c
uint8_t part_id;
if (max30102_read_reg(0xFF, &part_id, 1) == ESP_OK) {
    ESP_LOGI(TAG, "Part ID: 0x%02X (should be 0x15)", part_id);
}
```

Expected output: `Part ID: 0x15`

If you get an I2C error, the sensor isn't reachable. Check wiring.

#### Step 2: Check LED Activity

In a dark room, look directly at the sensor while it's running. You should see a faint **red glow** from the RED LED. The IR LED is invisible to human eyes but can be seen with a phone camera.

#### Step 3: Verify Raw Reading Range

Place finger on sensor and observe values in serial output:

- **Both at 262143 constantly** → Saturation, reduce LED current
- **Both at 0-5000** → No signal, increase LED current
- **IR around 80k-200k, RED around 60k-180k with visible pulsation** → 

#### Step 4: Inspect Pulsatile Component

Use Arduino Serial Plotter. You should see:
- Smooth waveforms with regular peaks (your heartbeat)
- ~1 peak per second for resting (60 BPM = 1 Hz)
- Both channels moving together (in phase)

If signals are flat or noisy:
- Press finger more firmly
- Ensure finger covers BOTH LEDs and photodiode
- Try a different finger
- Stay still during measurement

### Tuning Tips by Use Case

#### For Cold Fingers / Poor Perfusion
```c
#define LED_CURRENT_RED_MA  15.0   // Increase from 10.0
#define LED_CURRENT_IR_MA   12.0   // Increase from 7.0
#define ADC_RANGE_OPT       0      // More sensitive
```

#### For Dark Skin Tones
```c
#define LED_CURRENT_RED_MA  20.0   // Higher penetration needed
#define LED_CURRENT_IR_MA   15.0
#define ADC_RANGE_OPT       1      // Keep default
```

#### For High-Detail Capture (Research)
```c
#define SAMPLE_RATE_OPT     3      // 400 Hz
#define PULSE_WIDTH_OPT     3      // 18-bit
#define FIFO_AVG_OPT        0      // No averaging (raw data)
#define UART_BAUD_RATE      921600 // Avoid serial bottleneck
```

#### For Long-Duration Stable Logging
```c
#define SAMPLE_RATE_OPT     1      // 100 Hz (lower bandwidth)
#define FIFO_AVG_OPT        2      // 4-sample averaging (smoother)
#define UART_BAUD_RATE      115200 // Standard reliable rate
```

### When to Accept Limitations

Not every signal can be improved. If after all tuning:
- Signal remains noisy → Subject may have poor peripheral perfusion
- Always saturated at minimum LED → Sensor placement issue, not parameter issue
- Erratic readings → Subject is moving / finger pressure varies → cannot fix in firmware

These are **data collection issues**, not firmware issues.

---

## Future Enhancements

The current firmware is a foundation. Planned future expansions include:

### Battery Power System

- **Battery integration:** LiPo battery (500-1000 mAh) with TP4056 charging circuit
- **Power management:** Low-quiescent voltage regulator (e.g., MCP1700)
- **Charge status LED:** Visual indicator for battery state
- **USB-C charging:** Use existing USB port for both data and charging
- **Battery monitoring:** ADC-based voltage measurement, report via serial

### Wireless Connectivity

- **WiFi mode:** Send data to dashboard via HTTP or WebSocket
- **BLE mode:** Stream to phone/laptop without cables
- **MQTT publishing:** Send to cloud broker for remote monitoring
- **Configurable mode switch:** USB / WiFi / BLE selectable at boot

### Power Management

- **Deep sleep between measurements:** Wake on user button
- **Adaptive sample rate:** Reduce when battery low
- **Sensor shutdown when finger absent:** Save power
- **Estimated battery life:** Calculate based on usage patterns

### Smart Features

- **Onboard signal quality check:** Reject measurements before transmission
- **Auto-calibration of LED currents:** Tune based on observed signal
- **Sample timestamping:** Include precise timestamps in output
- **Multiple sensor support:** Coordinate readings from multiple body sites

### Better Diagnostics

- **OLED display:** Show heart rate, signal quality, battery status
- **Buzzer/LED status:** Audible/visual indicators for finger placement
- **Web-based configuration:** Change parameters without recompilation
- **Firmware OTA updates:** Update over WiFi

---

## Summary

This firmware provides a solid, configurable foundation for PPG data acquisition. Its strengths are:

- ✅ **Highly tunable** for different subjects and conditions
- ✅ **Simple integration** via standard CSV serial format
- ✅ **High-quality data** at 18-bit resolution
- ✅ **Reliable operation** via FreeRTOS task isolation
- ✅ **Documented configuration** from real tuning experiments
- ✅ **Ready for upgrades** (battery, wireless, power management)

The output stream feeds directly into the Python signal processing pipeline that filters, detects beats, extracts features, and predicts glucose levels via machine learning.

For the complete glucose monitoring system documentation, refer to the corresponding README files for each pipeline component.



---

## References & Documentation

### Official Hardware Documentation

**ESP32-S3 Microcontroller:**

1. Espressif Systems, "ESP32-S3 Series Datasheet," Version 1.6, 2024. [Online]. Available: https://www.espressif.com/sites/default/files/documentation/esp32-s3_datasheet_en.pdf

2. Espressif Systems, "ESP32-S3 Technical Reference Manual," Version 1.2, 2024. [Online]. Available: https://www.espressif.com/sites/default/files/documentation/esp32-s3_technical_reference_manual_en.pdf

3. Espressif Systems, "ESP32-S3-DevKitC-1 Hardware Guide," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/stable/esp32s3/hw-reference/esp32s3/user-guide-devkitc-1.html

**MAX30102 Pulse Oximeter & Heart-Rate Sensor:**

4. Maxim Integrated (now Analog Devices), "MAX30102 — High-Sensitivity Pulse Oximeter and Heart-Rate Sensor for Wearable Health," Rev. 1, 2018. [Online]. Available: https://www.analog.com/media/en/technical-documentation/data-sheets/MAX30102.pdf

5. Maxim Integrated, "MAX30102EVKIT — Evaluation Kit for the MAX30102," 2018. [Online]. Available: https://www.analog.com/en/resources/evaluation-hardware-and-software/evaluation-boards-kits/max30102evkit.html

### Software Frameworks

**ESP-IDF (Espressif IoT Development Framework):**

6. Espressif Systems, "ESP-IDF Programming Guide v5.5.1," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/index.html

7. Espressif Systems, "ESP-IDF I2C Driver Documentation," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/api-reference/peripherals/i2c.html

8. Espressif Systems, "ESP-IDF UART Driver Documentation," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/api-reference/peripherals/uart.html

9. Espressif Systems, "ESP-IDF Logging Library," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/api-reference/system/log.html

10. Espressif Systems, "ESP-IDF Get Started Guide," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/get-started/index.html

**FreeRTOS:**

11. Amazon Web Services, "FreeRTOS Kernel Documentation," 2024. [Online]. Available: https://www.freertos.org/Documentation/RTOS_book.html

12. R. Barry, *Mastering the FreeRTOS Real Time Kernel — A Hands-On Tutorial Guide*. Real Time Engineers Ltd., 2016. [Available free: https://www.freertos.org/Documentation/161204_Mastering_the_FreeRTOS_Real_Time_Kernel-A_Hands-On_Tutorial_Guide.pdf]

13. Espressif Systems, "FreeRTOS in ESP-IDF," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/api-reference/system/freertos.html

### Protocol Standards

**I²C (Inter-Integrated Circuit):**

14. NXP Semiconductors, "I2C-Bus Specification and User Manual," Rev. 7.0, 2021. [Online]. Available: https://www.nxp.com/docs/en/user-guide/UM10204.pdf

15. NXP Semiconductors, "I2C Bus Pull-Up Resistor Calculation," Application Note AN10658, 2009. [Online]. Available: https://www.nxp.com/docs/en/application-note/AN10658.pdf

**UART / USB-CDC Communication:**

16. USB Implementers Forum, "Universal Serial Bus Class Definitions for Communications Devices," Rev. 1.2, 2010.

### Scientific Background

**Photoplethysmography (PPG) for Health Monitoring:**

17. J. Allen, "Photoplethysmography and its application in clinical physiological measurement," *Physiological Measurement*, vol. 28, no. 3, pp. R1-R39, 2007. [DOI: 10.1088/0967-3334/28/3/R01]

18. T. Tamura et al., "Wearable photoplethysmographic sensors—past and present," *Electronics*, vol. 3, no. 2, pp. 282-302, 2014. [DOI: 10.3390/electronics3020282]

**Pulse Oximetry Principles:**

19. J. G. Webster (Ed.), *Design of Pulse Oximeters*. Bristol, UK: Institute of Physics Publishing, 1997.

20. Y. Mendelson and B. D. Ochs, "Noninvasive pulse oximetry utilizing skin reflectance photoplethysmography," *IEEE Transactions on Biomedical Engineering*, vol. 35, no. 10, pp. 798-805, 1988. [DOI: 10.1109/10.7286]

### Development Tools & Tutorials

**ESP-IDF Installation:**

21. Espressif Systems, "ESP-IDF Tools Installer for Windows," 2024. [Online]. Available: https://dl.espressif.com/dl/esp-idf/

22. Espressif Systems, "ESP-IDF VS Code Extension," 2024. [Online]. Available: https://github.com/espressif/vscode-esp-idf-extension

**Visualization Tools:**

23. Arduino, "Arduino IDE Serial Plotter Documentation," 2024. [Online]. Available: https://docs.arduino.cc/software/ide-v2/tutorials/ide-v2-serial-plotter

24. Fritzing GmbH, "Fritzing — Electronics Made Easy," 2024. [Online]. Available: https://fritzing.org/

### Community Resources

**Reference Implementations & Libraries:**

25. SparkFun Electronics, "SparkFun MAX3010x Pulse and Proximity Sensor Library," GitHub Repository, 2024. [Online]. Available: https://github.com/sparkfun/SparkFun_MAX3010x_Sensor_Library

26. Espressif Systems, "ESP-IDF Component Manager," 2024. [Online]. Available: https://components.espressif.com/

27. Adafruit Industries, "MAX30105 Pulse Sensor Module Tutorial," 2024. [Online]. Available: https://learn.adafruit.com/max30105-pulse-sensor-breakout

### Useful Application Notes

**Maxim Integrated MAX30102:**

28. Maxim Integrated, "AN6409 — Guidelines for SpO2 Measurement Using the Maxim MAX32664 Sensor Hub," 2019. [Online]. Available: https://pdfserv.maximintegrated.com/en/an/AN6409.pdf

29. Maxim Integrated, "Recommended Configurations and Operating Profiles for MAX30102," Application Note 6271, 2017.

**Espressif:**

30. Espressif Systems, "ESP32-S3 Peripheral Best Practices Guide," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/api-reference/peripherals/

### Programming References

**Embedded C Programming:**

31. M. Barr and A. Massa, *Programming Embedded Systems: With C and GNU Development Tools*, 2nd ed. Sebastopol, CA: O'Reilly Media, 2006.

32. J. Yiu, *The Definitive Guide to ARM® Cortex®-M3 and Cortex®-M4 Processors*, 3rd ed. Oxford, UK: Newnes, 2013.

**Real-Time Systems:**

33. J. W. S. Liu, *Real-Time Systems*. Upper Saddle River, NJ: Prentice Hall, 2000.

### Future Reference Topics

For planned upgrades documented in the "Future Enhancements" section, useful references include:

**Battery Management:**

34. Microchip Technology, "MCP1700/MCP1700-EXP Low Quiescent Current LDO," Datasheet, 2020. [Online]. Available: https://www.microchip.com/wwwproducts/en/MCP1700

35. NanJing Top Power ASIC Corp., "TP4056 — 1A Standalone Linear Li-Lon Battery Charger," Datasheet, 2014.

**BLE & WiFi on ESP32:**

36. Espressif Systems, "ESP-IDF Bluetooth Low Energy Guide," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/api-guides/bt-architecture/index.html

37. Espressif Systems, "ESP-IDF Wi-Fi Driver Documentation," 2024. [Online]. Available: https://docs.espressif.com/projects/esp-idf/en/v5.5.1/esp32s3/api-reference/network/esp_wifi.html

---

*Last updated: [Add your date]*
*Project: Non-Invasive Glucose Estimation — Hardware Acquisition Module*
*Author: [Your name]*

Happy sensing!