# ESP32 Firmware

## Overview
This folder contains the firmware for the ESP32-S3 microcontroller that interfaces with the MAX30102 sensor to acquire PPG (Photoplethysmography) signals for non-invasive diabetes prediction.

## Hardware Requirements

### Microcontroller
- **ESP32-S3** (any variant with I2C support)

### Sensor
- **MAX30102** - Integrated Pulse Oximetry and Heart-Rate Monitor Module
  - Dual LED (Red + IR)
  - 18-bit ADC resolution
  - I2C interface

### Pin Connections
| ESP32-S3 Pin | MAX30102 Pin | Description |
|--------------|--------------|-------------|
| GPIO 1       | SDA          | I2C Data    |
| GPIO 2       | SCL          | I2C Clock   |
| 3.3V         | VIN          | Power       |
| GND          | GND          | Ground      |

## Configuration Parameters

The firmware includes user-configurable parameters in `main/main.c`:

### Key Settings
- **I2C Speed**: 400kHz (recommended for high sample rates)
- **LED Current**: 
  - Red: 10.0 mA (adjustable 0-51 mA)
  - IR: 7.0 mA (adjustable 0-51 mA)
- **Sample Rate**: 400Hz (configurable: 50Hz - 3200Hz)
- **ADC Range**: 4096nA (4 sensitivity levels available)
- **Pulse Width**: 411μs (18-bit resolution)
- **UART Baud Rate**: 921600 (for high-speed data transmission)

## Build Instructions

### Prerequisites
1. Install [ESP-IDF](https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/) (v5.0 or later)
2. Set up ESP-IDF environment:
   ```bash
   . $HOME/esp/esp-idf/export.sh
   ```

### Build Steps
```bash
cd 01_Firmware_ESP32
idf.py set-target esp32s3
idf.py build
```

### Flash to Device
```bash
idf.py -p <PORT> flash monitor
```
Replace `<PORT>` with your serial port (e.g., `COM3` on Windows, `/dev/ttyUSB0` on Linux)

## Output Format

The firmware outputs comma-separated values via UART:
```
IR_value,RED_value
```

Example:
```
125430,98234
127891,99012
0,0          # No finger detected
```

## Troubleshooting

### I2C Communication Errors
- Check wiring connections
- Verify pull-up resistors on SDA/SCL (usually built-in on ESP32-S3)
- Ensure MAX30102 is powered with 3.3V

### Signal Saturation
- Reduce LED current values
- Increase ADC range setting
- Ensure proper finger placement (not too tight)

### Low Signal Quality
- Increase LED current
- Ensure good finger contact
- Check for ambient light interference

## Documentation

Additional documentation can be found in the `docs/` subfolder:
- Hardware schematics
- Sensor datasheets
- Calibration procedures

## Related Folders
- `02_Python_Data_Logger/` - Tools to capture and log this firmware's output
- `03_Python_Signal_Processing_Pipeline/` - Process the raw PPG signals
