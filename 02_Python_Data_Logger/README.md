# 📈 Python PPG Data Logger

A high-performance Python GUI application for real-time visualization and logging of dual-channel Photoplethysmography (PPG) data.

This module captures raw Infrared (IR) and Red PPG signals from a microcontroller via a serial interface. It visualizes the data stream in real-time using hardware-accelerated graphics and logs the readings to timestamped CSV files for downstream processing. 

**Pipeline Position:** Step 02 — Receives serial data from ESP32 (Step 01) and feeds raw CSVs to the Windowing Tool (Step 03).

## ✨ Key Features

- **Real-Time Visualization:** Dual-channel plots (IR in blue, RED in orange) at 60+ FPS.
- **Hardware Acceleration:** Powered by PyQtGraph with OpenGL rendering (10-20x faster than Matplotlib).
- **Efficient Memory Management:** Fixed-size NumPy circular buffer (~32KB) prevents memory leaks.
- **Precision Logging:** Millisecond-accurate timestamping in CSV outputs.
- **Session Management:** Prompts for Session ID, auto-creates directories, and prevents file overwriting.
- **Low Overhead:** Single-threaded `QTimer` design ensures race-condition-free operation at 3-5% CPU usage.

## ⚙️ Technical Specifications & Configuration

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| `SERIAL_PORT` | `COM7` | Target serial port for the ESP32 |
| `BAUD_RATE` | `115200` | Serial baud rate |
| `FS` | `400 Hz` | Sampling frequency |
| `WINDOW_SIZE` | `4000` | Buffer size (10 seconds at 400 Hz) |
| Latency | `< 16 ms` | UI update interval |
| Max Sustained | `~3200 Hz` | Maximum supported sampling rate |
| Storage | `~10 KB/s` | Disk write speed (~3MB per 5-min session) |

### Baud Rate Matrix
| Sampling Rate (Hz) | Recommended Baud Rate |
|-------------------|-----------------------|
| 50 - 100          | 115200                |
| 200 - 400         | 115200 or 460800      |
| 800 - 1600        | 460800 or 921600      |
| 3200              | 921600                |

## 📥 Input / Output Format

- **Input:** Serial string stream from ESP32 (e.g., `IR_Value,RED_Value\n`).
- **Output:** Stored in `05_Data_Storage/01_Raw/<Session_ID>.csv`
  - Columns: `Timestamp`, `IR`, `RED`

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install pyqtgraph PyQt5 pyserial numpy
   ```

2. **Run the Logger:**
   ```bash
   python Data_Logger.py
   ```
   *You will be prompted to enter a Session ID to begin logging.*

## 🏗️ Architecture

The `HighPerfPlotter` class inherits from `QMainWindow` and uses a single-threaded design driven by a `QTimer` (16ms interval) for UI updates.
- `init_ui()`: Sets up the PyQtGraph windows and plot items.
- `setup_serial()`: Initializes the PySerial connection.
- `update_loop()`: Reads serial data, appends to the NumPy circular buffer, writes to CSV, and updates plots.
- `closeEvent()`: Safely closes the serial port and file handles on exit.

## 🔧 Troubleshooting

| Issue | Potential Cause | Solution |
|-------|----------------|----------|
| **No Data / Empty Plots** | Wrong COM port or ESP32 not connected. | Verify `SERIAL_PORT` matches your system's device manager. |
| **SerialException** | Port is busy or baud rate mismatch. | Close other serial monitors (e.g., Arduino IDE) and check `BAUD_RATE`. |
| **High CPU Usage / Lag** | Sample rate exceeds baud rate capacity. | Refer to the Baud Rate Matrix and increase baud rate if using >400Hz. |