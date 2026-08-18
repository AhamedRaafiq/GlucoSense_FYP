# 📡 ESP32-S3 PPG Signal Acquisition Firmware

> **GlucoSense FYP - Step 01: Hardware Acquisition Layer**

This module provides the embedded C firmware for the ESP32-S3 microcontroller to interface with a MAX30102 pulse oximeter sensor. It captures high-resolution, dual-channel (Red + IR) photoplethysmography (PPG) signals from the fingertip and streams them over UART/USB-CDC in real-time.

**Pipeline Position:** This is the first step in the pipeline. It captures raw PPG data and feeds it into the `02_Python_Data_Logger` module.

## ✨ Key Features
- **High-Speed I2C Communication:** Interfaces with MAX30102 at 400 kHz Fast Mode.
- **18-Bit Dual-Channel Acquisition:** Captures both Red (660nm) and IR (880nm) signals.
- **FreeRTOS Integration:** Dedicated `ppg_task` running at priority 5 for reliable 10ms polling.
- **Real-Time Streaming:** Outputs data in clean CSV format (`IR,RED\n`).
- **Smart Finger Detection:** Emits `0,0` if no finger is detected (IR threshold: 50000).

## ⚙️ Technical Specifications & Configuration

| Parameter | Configuration |
| :--- | :--- |
| **Microcontroller** | ESP32-S3 (Dual-core Xtensa LX7 @ 240 MHz) |
| **Framework** | ESP-IDF v5.5.1 |
| **Sensor** | MAX30102 (I2C Address: `0x57`) |
| **I2C Pins** | SDA: `GPIO 1`, SCL: `GPIO 2` |
| **Sample Rate** | 400 Hz |
| **Pulse Width** | 411 µs (18-bit ADC resolution) |
| **ADC Range** | 4096 nA |
| **LED Currents** | RED: 10.0 mA, IR: 7.0 mA |
| **FIFO Averaging** | None (Raw data output) |

*Note: This configuration was optimized after 21 experimental test runs (T1-T21).*

## 🔌 Wiring Guide

| ESP32-S3 Pin | MAX30102 Pin |
| :--- | :--- |
| `3V3` | `VIN` |
| `GND` | `GND` |
| `GPIO 1` | `SDA` |
| `GPIO 2` | `SCL` |

## 🚀 Quick Start

1. **Set up the ESP-IDF environment** (v5.5.1 recommended).
2. **Connect the hardware** according to the wiring guide.
3. **Build and Flash:**
   ```bash
   idf.py set-target esp32s3
   idf.py build
   idf.py -p COM5 flash monitor
   ```
   *(Change `COM5` to your ESP32's actual COM port).*

## 📤 Output Format
The firmware streams data over UART/USB-CDC in standard CSV format:
```csv
<IR_Value>,<RED_Value>
134502,125430
134510,125435
0,0
```
*(Outputs `0,0` when no finger is detected)*

## 🛠️ Architecture
The system is built on FreeRTOS with a task-based architecture. Key components:
- `app_main()`: Initializes I2C, configures the sensor, and creates the RTOS task.
- `max30102_init_sensor()`: Writes to control registers (Mode Config `0x09`, SpO2 Config `0x0A`, LEDs `0x0C`/`0x0D`, FIFO `0x08`).
- `ppg_task()`: Polls the FIFO Data register (`0x07`) every 10ms and handles UART streaming.
- **Key Files:** `main/main.c`, `CMakeLists.txt`, `sdkconfig`.

## 🆘 Troubleshooting

| Issue | Possible Cause & Solution |
| :--- | :--- |
| **All Zeros Output** | I2C communication failure. Check SDA/SCL wiring and pull-up resistors. |
| **Signal Saturation (Flatline at Max)** | Ambient light interference or LED current too high. Lower LED mA values in `0x0C`/`0x0D`. |
| **Dropped Samples** | Baud rate mismatch or USB bottleneck. Ensure receiving script matches baud rate. |
| **ESP32 Core Panic / Reset** | Stack overflow in `ppg_task`. Ensure stack size is at least 4096 bytes. |