import serial
import time
import csv
import os
import sys
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

# ==========================================
# --- CONFIGURATION ---
# ==========================================
SERIAL_PORT = 'COM6'   # Ensure this matches your ESP32 Port
BAUD_RATE = 115200     # Must match the ESP32 C code

# --- PLOTTER CONFIGURATION (NEW!) ---
VISIBLE_SAMPLES = 5000      # How many data points to show on screen at once
PLOT_REFRESH_RATE = 30     # Update plot every N samples (Higher = Faster Performance)
# ==========================================

def get_valid_filename():
    """Asks user for a filename and checks for duplicates."""
    while True:
        filename = input("\n📂 Enter filename for this session (e.g., 'SubjectA_Session1'): ").strip()
        
        # Auto-add .csv extension
        if not filename.lower().endswith('.csv'):
            filename += ".csv"
            
        # Check if file exists to prevent overwriting
        if os.path.exists(filename):
            print(f"⚠️  File '{filename}' already exists!")
            choice = input("   Overwrite it? (y/n): ").strip().lower()
            if choice == 'y':
                return filename
            else:
                print("   Okay, try a different name.")
        else:
            return filename

def log_data():
    print("\n==================================================")
    print("      GLUCOSE DATA LOGGER (SIGNAL ONLY MODE)      ")
    print("      (Timestamp + IR + Red Values Only)          ")
    print("==================================================")

    # --- STEP 1: INPUTS ---
    target_filename = get_valid_filename()
    
    # [REMOVED] Glucose Input
    # [REMOVED] Subject Name Input
    
    print("\n--------------------------------------------------")
    print(f"   Target File:  {target_filename}")
    print("--------------------------------------------------")
    
    input("👉 Press ENTER to start recording (Ctrl+C to stop)...")

    # --- PLOTTER SETUP (NEW!) ---
    plt.ion()  # Turn on interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Buffers to hold data for plotting
    ir_buffer = deque([0] * VISIBLE_SAMPLES, maxlen=VISIBLE_SAMPLES)
    red_buffer = deque([0] * VISIBLE_SAMPLES, maxlen=VISIBLE_SAMPLES)
    
    # Initialize Empty Lines
    line_ir, = ax1.plot(ir_buffer, 'k-', label='IR Signal')
    line_red, = ax2.plot(red_buffer, 'r-', label='Red Signal')
    
    # Formatting
    ax1.set_title("Real-Time IR Signal")
    ax1.set_ylabel("Amplitude")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title("Real-Time Red Signal")
    ax2.set_ylabel("Amplitude")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    sample_counter = 0 # To track when to update plot

    # --- STEP 2: CONNECT & RECORD ---
    ser = None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"\n✅ Connected to {SERIAL_PORT}. Streaming raw signals...")

        with open(target_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            # --- UPDATED HEADER (Removed Glucose_Label) ---
            writer.writerow([
                "Timestamp", 
                "IR_Value", 
                "Red_Value"
            ]) 

            # DATA LOOP
            while True:
                try:
                    # Read line from ESP32
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line:
                        parts = line.split(',')
                        
                        # --- FILTER: Accept only pairs of numbers ---
                        if len(parts) == 2:
                            ir_str = parts[0].strip()
                            red_str = parts[1].strip()
                            
                            if ir_str.isdigit() and red_str.isdigit():
                                # Generate Timestamp
                                current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                                
                                # Convert to int for plotting
                                ir_val = int(ir_str)
                                red_val = int(red_str)
                                
                                # --- WRITE DATA ROW (Removed Glucose Input) ---
                                writer.writerow([
                                    current_time, 
                                    ir_str, 
                                    red_str
                                ])
                                
                                # --- PLOTTER UPDATE LOGIC (NEW!) ---
                                ir_buffer.append(ir_val)
                                red_buffer.append(red_val)
                                sample_counter += 1
                                
                                # Only redraw plot every N samples to prevent lag
                                if sample_counter % PLOT_REFRESH_RATE == 0:
                                    line_ir.set_ydata(ir_buffer)
                                    line_red.set_ydata(red_buffer)
                                    
                                    # Auto-scale Y axis to fit the waves
                                    ax1.set_ylim(min(ir_buffer)-100, max(ir_buffer)+100)
                                    ax2.set_ylim(min(red_buffer)-100, max(red_buffer)+100)
                                    
                                    plt.pause(0.001) # Update the GUI
                                
                                print(f"Saved: IR={ir_str} | Red={red_str}")
                            else:
                                pass # Ignore junk/boot logs

                except UnicodeDecodeError:
                    pass # Ignore serial glitches

    except serial.SerialException:
        print(f"\n❌ ERROR: Could not open {SERIAL_PORT}.")
        print("   -> Close Arduino IDE/Plotter and try again.")
        
    except KeyboardInterrupt:
        print(f"\n\n🛑 STOPPED. Data saved to '{target_filename}'.")
        plt.close() # Close plot window
        
    finally:
        if ser and ser.is_open:
            ser.close()

if __name__ == "__main__":
    log_data()