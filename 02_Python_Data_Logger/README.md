# Python High-Performance PPG Data Logger

## 1. Architectural Overview
This application acts as a high-speed data acquisition (DAQ) program that reads raw serial data from the ESP32-S3 microcontroller, saves it to a structured CSV file, and provides real-time visualization. It uses **PyQt5** for the GUI, **PyQtGraph** (a high-performance graphics library built on Qt) for hardware-accelerated charting, and **NumPy** for fast array indexing.

### Data Flow Diagram

```
+--------------------------------------------------------------------------+
|                                ESP32-S3                                  |
|                 Sends "IR_val,RED_val\n" via UART                        |
+--------------------------------------------------------------------------+
                                     |
                                     | USB Serial (115200 Baud)
                                     v
+--------------------------------------------------------------------------+
|                        data_logger_code02.py                             |
|                                                                          |
|  1. Main Thread Input Loop:                                              |
|     - Terminal prompts for Session ID                                    |
|     - Validates and creates CSV file                                     |
|                                                                          |
|  2. Start PyQt5 GUI (HighPerfPlotter):                                   |
|     - Initialize PyQtGraph UI (2 Linked Plots)                           |
|     - Open Serial connection (PySerial)                                  |
|     - Starts 16ms Update Timer (QTimer)                                  |
|                                                                          |
|  3. Update Loop (Every 16ms / ~60 FPS):                                  |
|     - Polls Serial Input Buffer (ser.in_waiting)                         |
|     - Reads all waiting bytes & decodes "IR,RED"                         |
|     - Appends "Timestamp, IR, RED" to CSV                                |
|     - Rolls NumPy arrays & inserts new samples                           |
|     - Updates plots using PyQtGraph setData()                            |
|                                                                          |
|  4. Close Event:                                                         |
|     - Safely closes Serial Port                                          |
|     - Flushes and closes CSV file                                        |
+--------------------------------------------------------------------------+
```

---

## 2. Core Modules & Step-by-Step Execution

### Module 1: Pre-Execution Setup & Initialization (`__main__` entry point)
The script begins execution in the standard Python main block:
1. **Interactive Session Naming**: Prompts the user via the terminal to input a Session ID (e.g., `SubjectA_Glucose100`).
2. **File Check & Overwrite Protection**:
   - Ensures the file ends with the `.csv` extension.
   - If the file already exists in `DATA_STORAGE_FOLDER_PATH`, it warns the user and asks for overwrite confirmation (`y/n`).
3. **CSV Preparation**: Opens the CSV file, instantiates a `csv.writer`, and writes the header:
   ```python
   ["Timestamp", "IR", "RED"]
   ```
4. **GUI Loop Handover**: Instantiates the `QApplication` event loop and opens the `HighPerfPlotter` GUI window.

### Module 2: High-Performance Graphics Initialization (`init_ui`)
To achieve a smooth 60 FPS refresh rate without lagging the incoming serial data, the GUI uses **PyQtGraph** instead of Matplotlib.
* **Plot Configurations**: Sets up a `GraphicsLayoutWidget` with a clean white background.
* **Dual Plots**:
  * **Plot 1**: Displays the raw IR signal in Matlab Blue (`#0072bd`).
  * **Plot 2**: Displays the raw RED signal in Matlab Orange-Red (`#d95319`).
* **Axis Linking**: The X-axes of both plots are linked:
  ```python
  self.p2.setXLink(self.p1)
  ```
  This links zooming and panning on one plot to the other automatically.
* **GUI Refresh Timer**: Starts a `QTimer` configured with a **16ms interval** (corresponds to $62.5\,\text{Hz}$ screen redraw rate) to trigger the processing loop.

### Module 3: Serial Communication Configuration (`setup_serial`)
Establishes connection to the hardware using `PySerial`:
```python
self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
self.ser.flushInput()
```
* **Input Flushing**: Flushes the input buffer on startup to clear any old, partial, or corrupted bytes sent while the serial port was closed.

### Module 4: High-Speed Serial Polling & Data Parsing (`update_loop`)
The `update_loop` function runs every 16ms. It handles incoming serial data and updates the real-time plots:

1. **Non-Blocking Serial Reading**:
   Reads all waiting data in the buffer at once to prevent latency:
   ```python
   while self.ser.in_waiting:
       line = self.ser.readline().decode('utf-8', errors='ignore').strip()
   ```
2. **Packet Parsing**:
   Splits the line by the comma delimiter:
   ```python
   parts = line.split(',')
   if len(parts) == 2:
       ir_val = float(parts[0])
       red_val = float(parts[1])
   ```
3. **High-Resolution Timestamp Logging**:
   Captures the system clock time down to millisecond precision and writes it to the CSV file:
   ```python
   t_now = datetime.now().strftime('%H:%M:%S.%f')[:-3] # Truncates to milliseconds
   self.csv_writer.writerow([t_now, ir_val, red_val])
   ```
4. **NumPy Rolling Buffers**:
   Instead of using standard Python lists, the plot data is stored in fixed-size NumPy arrays (`self.ir_buffer` and `self.red_buffer` of size `WINDOW_SIZE = 4000`). When a new sample arrives, the array values are rolled left:
   ```python
   self.ir_buffer[:-1] = self.ir_buffer[1:] # Shift elements left by 1 index
   self.ir_buffer[-1] = ir_val              # Insert new value at the end
   ```
   This array rolling operation is implemented in C within NumPy, making it much faster than appending and popping from Python lists.

### Module 5: Interface Redraw & Clean Closure
* **Screen Redraw**: Once all pending serial bytes are processed, the graph curves are updated:
  ```python
  self.curve_ir.setData(self.ir_buffer)
  self.curve_red.setData(self.red_buffer)
  ```
* **Resource Cleanup (`closeEvent`)**: When the user closes the plot window, the PyQt event loop intercepts the event, safely closes the serial port connection, flushes the file buffer, and closes the CSV file to prevent data loss.

---

## 3. Dependencies & Libraries (Imports)
* **`sys`**: Accesses system-specific functions, primarily used for exiting the script gracefully on errors (`sys.exit`).
* **`os`**: Interacts with the operating system to check directory existence and automatically create output folders.
* **`csv`**: Writes tabular logs to disk, formatting timestamps and raw LED readings into comma-separated values.
* **`serial` (PySerial)**: Manages connection to the COM port to stream incoming binary data packets from the microcontroller.
* **`serial.tools.list_ports`**: Scans and identifies available system serial ports (COM ports) connected to the computer.
* **`numpy` (`np`)**: Handles high-performance numerical array operations, facilitating fast memory shifting for rolling charts.
* **`datetime`**: Generates high-resolution, millisecond-precision timestamps associated with each incoming data sample.
* **`collections.deque`**: A container offering fast appends and pops from both ends (imported but not actively used in the final loop).
* **`pyqtgraph` (`pg`)**: Provides hardware-accelerated, lightweight charting utilities optimized for high-speed data display.
* **`PyQt5.QtWidgets`**: Renders basic desktop GUI elements (application layout, window management, and error popups).
* **`PyQt5.QtCore` (`QTimer`)**: Triggers execution loops at regular millisecond intervals to redraw plots and parse data.

---

## 4. Input & Output Files

### Input
* **Source Stream**: Real-time USB serial data stream from the ESP32-S3 microcontroller (e.g., `COM7` on Windows, `/dev/ttyUSB0` on Linux).
* **Baud Rate**: `115200` bps (or matches the hardware transmitter baud rate).

### Output
* **Output Path**: `05_Data_Storage/01_Raw/`
* **Filename Structure**: `{Session_ID}.csv` (e.g., `SubjectA_Glu100.csv` based on user terminal input).
* **Column Headers**: `Timestamp,IR,RED` (where Timestamp contains `%H:%M:%S.%f` millisecond resolution, and IR/RED are raw 18-bit integers).

---

## 5. Hyperparameters & Configuration
* **`SERIAL_PORT` (`COM7`)**: The active COM port identifier where the ESP32 hardware is connected.
* **`BAUD_RATE` (`115200`)**: Serial communication speed; must match the controller rate to prevent data corruption.
* **`DATA_STORAGE_FOLDER_PATH`**: Absolute folder destination on disk where raw signal CSV logs are saved.
* **`FS` (400 Hz)**: The expected acquisition sampling frequency, defining the data collection time axis.
* **`WINDOW_SIZE` (`4000`)**: The graphical window size; at 400 Hz, 4000 points display a rolling window of exactly 10 seconds.

---

## 6. Report & Documentation Guidelines

### Why PyQtGraph Over Matplotlib for Real-Time Plots?
* **Low CPU Overhead**: Matplotlib is designed for static publishing figures and redraws the whole canvas on update, causing high CPU load. PyQtGraph uses Qt's GraphicsView framework to update only the modified line coordinate memory buffers, enabling smooth 60 FPS real-time rendering.
* **Non-Blocking Execution**: Efficient rendering prevents GUI lags that could cause the serial interface to drop incoming bytes during high-frequency sampling (400Hz).

### Why Fixed-Size NumPy Arrays for Circular Buffers?
* **C-Speed Array Rolling**: Appending and popping from standard Python lists triggers frequent memory re-allocations. Fixed-size NumPy arrays shifted via vector operations:
  ```python
  self.ir_buffer[:-1] = self.ir_buffer[1:]
  ```
  execute at compile-speed in C, minimizing memory thrashing.

### Why Millisecond-Precision Timestamps?
* **Acquisition Timing Validation**: Physical communication delays, USB bridge latency, or OS multitasking delays can cause jitters in sample timing. Logging real-time clock timestamps down to millisecond precision (`%H:%M:%S.%f`) allows post-acquisition analysis to calculate the exact time intervals between consecutive samples and verify sampling rate stability.
