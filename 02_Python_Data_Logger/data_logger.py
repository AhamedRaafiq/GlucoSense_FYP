# ==============================================================================
# ⚙️ USER CONFIGURATION SECTION
# ==============================================================================
# 1. Hardware Connection
SERIAL_PORT = 'COM6'          # CHECK DEVICE MANAGER! (e.g., COM3 on Windows, /dev/ttyUSB0 on Linux)
BAUD_RATE   = 921600          # MUST match the ESP32 Code (921600 Recommended)

# 2. File Saving Location
# Tip: Use 'r' before the string to handle backslashes on Windows safely.
# Example: r"C:\Users\YourName\Desktop\FYP_Data_Raw"
DATA_STORAGE_FOLDER_PATH = r"C:\Users\DELL\Documents\GitHub\fyp\04_Data_Storage\Normal_PPG_Only_Data_Set_For_Practice\ali.csv" 

# 3. Plotter Settings
FS          = 100             # Sampling Rate (Hz) - Used for time axis
WINDOW_SIZE = 2000            # How many points to show (2000 points @ 100Hz = 20 seconds)
# ==============================================================================

import sys
import os
import csv
import serial
import serial.tools.list_ports
import numpy as np
from datetime import datetime
from collections import deque

# High-Performance Graphics & GUI Libraries
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QMessageBox, QInputDialog
from PyQt5.QtCore import QTimer

class HighPerfPlotter(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 1. Setup Data Structures (Circular Buffers for Speed)
        self.ir_buffer = np.zeros(WINDOW_SIZE)
        self.red_buffer = np.zeros(WINDOW_SIZE)
        self.ptr = 0
        self.csv_file = None
        self.csv_writer = None
        
        # 2. Create Target Folder if it doesn't exist
        if not os.path.exists(DATA_STORAGE_FOLDER_PATH):
            try:
                os.makedirs(DATA_STORAGE_FOLDER_PATH)
                print(f"✅ Created data folder: {DATA_STORAGE_FOLDER_PATH}")
            except OSError as e:
                print(f"❌ Error creating folder: {e}")
                sys.exit(1)
            
        # 3. GUI Setup
        self.init_ui()
        
        # 4. Connect Serial
        self.setup_serial()

        # 5. Start Session (Popup Dialog)
        self.ask_filename()

    def init_ui(self):
        self.setWindowTitle("PPG Signal Acquisition (High-Performance)")
        self.resize(1200, 700)
        
        # Central Layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # PyQtGraph Config (Professional White Background)
        pg.setConfigOption('background', 'w')
        pg.setConfigOption('foreground', 'k')
        
        # Graphics Layout
        self.plot_widget = pg.GraphicsLayoutWidget()
        layout.addWidget(self.plot_widget)
        
        # --- PLOT 1: IR SIGNAL (Blue) ---
        self.p1 = self.plot_widget.addPlot(title="IR Signal (Raw)")
        self.p1.setLabel('left', 'Amplitude')
        self.p1.showGrid(x=True, y=True, alpha=0.3)
        self.curve_ir = self.p1.plot(pen=pg.mkPen('#0072bd', width=2)) # Matlab Blue
        
        self.plot_widget.nextRow()
        
        # --- PLOT 2: RED SIGNAL (Red) ---
        self.p2 = self.plot_widget.addPlot(title="Red Signal (Raw)")
        self.p2.setLabel('left', 'Amplitude')
        self.p2.setLabel('bottom', 'Samples')
        self.p2.showGrid(x=True, y=True, alpha=0.3)
        self.curve_red = self.p2.plot(pen=pg.mkPen('#d95319', width=2)) # Matlab Red
        
        # Link X-Axis (Zooming one zooms both)
        self.p2.setXLink(self.p1)

        # Update Timer (Targeting ~60 FPS)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(16) # 16ms interval

    def setup_serial(self):
        try:
            self.ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
            self.ser.flushInput()
            print(f"✅ Serial Connected: {SERIAL_PORT} @ {BAUD_RATE}")
        except Exception as e:
            QMessageBox.critical(self, "Serial Error", f"Could not open {SERIAL_PORT}.\n\nCheck connection or close Arduino IDE.\nError: {e}")
            sys.exit(1)

    def ask_filename(self):
        # Popup Dialog for Session ID
        filename, ok = QInputDialog.getText(self, "New Session", "Enter Session ID (e.g., SubjectA_Glu100):")
        
        if ok and filename:
            filename = filename.strip()
            # Auto-append .csv if missing
            if not filename.lower().endswith('.csv'):
                filename += ".csv"
                
            self.filepath = os.path.join(DATA_STORAGE_FOLDER_PATH, filename)
            
            # Check overwrite protection
            if os.path.exists(self.filepath):
                reply = QMessageBox.question(self, "Warning", f"File '{filename}' already exists.\nOverwrite it?",
                                             QMessageBox.Yes | QMessageBox.No)
                if reply == QMessageBox.No:
                    self.ask_filename() # Ask again
                    return

            # Open File for Writing
            try:
                self.csv_file = open(self.filepath, mode='w', newline='')
                self.csv_writer = csv.writer(self.csv_file)
                self.csv_writer.writerow(["Timestamp", "IR", "RED"]) # Header
                
                self.setWindowTitle(f"Recording: {filename} | {SERIAL_PORT}")
                print(f"📂 Recording Started: {self.filepath}")
            except Exception as e:
                QMessageBox.critical(self, "File Error", f"Cannot create file: {e}")
                sys.exit(1)
        else:
            sys.exit(0) # User cancelled

    def update_loop(self):
        if not self.ser.is_open:
            return

        try:
            # High-Speed Read: Read ALL waiting bytes at once
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line: continue
                
                parts = line.split(',')
                if len(parts) == 2:
                    try:
                        # Parse Data (Expected format: IR, RED)
                        ir_val = float(parts[0])
                        red_val = float(parts[1])
                        
                        # Save to CSV
                        if self.csv_writer:
                            t_now = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                            self.csv_writer.writerow([t_now, ir_val, red_val])

                        # Update Circular Buffer (Numpy roll is very fast)
                        self.ir_buffer[:-1] = self.ir_buffer[1:]
                        self.ir_buffer[-1] = ir_val
                        
                        self.red_buffer[:-1] = self.red_buffer[1:]
                        self.red_buffer[-1] = red_val
                        
                    except ValueError:
                        pass # Skip corrupt packets

            # Redraw Plot (Once per timer tick)
            self.curve_ir.setData(self.ir_buffer)
            self.curve_red.setData(self.red_buffer)

        except Exception as e:
            print(f"Serial Loop Error: {e}")

    def closeEvent(self, event):
        # Cleanup when window is closed
        if self.ser.is_open:
            self.ser.close()
        if self.csv_file:
            self.csv_file.close()
            print("\n💾 Session Saved & Closed.")
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HighPerfPlotter()
    window.show()
    sys.exit(app.exec_())