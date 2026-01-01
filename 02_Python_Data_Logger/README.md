# Python Data Logger

## Overview
This folder contains Python scripts for capturing, logging, and visualizing PPG (Photoplethysmography) data from the ESP32-S3 + MAX30102 sensor system.

## Files

### `data_logger.py`
Main data logging script that captures serial data from the ESP32 and saves it to CSV files.

**Features:**
- Real-time serial data capture
- Automatic CSV file generation with timestamps
- Configurable baud rate (default: 921600)
- Data validation and error handling

**Usage:**
```bash
python data_logger.py --port COM3 --output data.csv
```

**Arguments:**
- `--port`: Serial port (e.g., COM3, /dev/ttyUSB0)
- `--output`: Output CSV filename
- `--baud`: Baud rate (default: 921600)
- `--duration`: Recording duration in seconds (optional)

### `Ex_plots.py`
Example plotting script for visualizing captured PPG signals.

**Features:**
- Real-time plotting of IR and RED channels
- Signal quality visualization
- Export plots as images

**Usage:**
```bash
python Ex_plots.py --input data.csv
```

## Requirements

Install required Python packages:
```bash
pip install -r ../requirements.txt
```

Key dependencies:
- `pyserial` - Serial communication
- `pandas` - Data handling
- `matplotlib` - Visualization
- `numpy` - Numerical operations

## Quick Start

1. **Connect ESP32** to your computer via USB
2. **Identify the serial port**:
   - Windows: Check Device Manager (e.g., COM3)
   - Linux/Mac: `ls /dev/tty*` (e.g., /dev/ttyUSB0)
3. **Run the data logger**:
   ```bash
   python data_logger.py --port COM3 --output my_data.csv
   ```
4. **Place finger on sensor** and wait for data collection
5. **Visualize the data**:
   ```bash
   python Ex_plots.py --input my_data.csv
   ```

## Data Format

Output CSV files contain two columns:
```csv
IR,RED
125430,98234
127891,99012
126543,98765
```

- **IR**: Infrared LED readings (18-bit values, 0-262143)
- **RED**: Red LED readings (18-bit values, 0-262143)
- Values of `0,0` indicate no finger detected

## Troubleshooting

### Port Access Denied
- **Windows**: Close any other programs using the serial port
- **Linux**: Add user to dialout group: `sudo usermod -a -G dialout $USER`

### No Data Received
- Verify correct baud rate (must match firmware setting)
- Check USB cable connection
- Ensure ESP32 firmware is running (`idf.py monitor`)

### Garbled Data
- Incorrect baud rate - update to match firmware (default: 921600)
- USB cable quality - try a different cable

## Data Storage

Captured data is automatically organized in `04_Data_Storage/Raw/` for further processing.

## Related Folders
- `01_Firmware_ESP32/` - ESP32 firmware that generates the data
- `03_Python_Signal_Processing_Pipeline/` - Process the logged data
- `04_Data_Storage/` - Organized storage for all data
