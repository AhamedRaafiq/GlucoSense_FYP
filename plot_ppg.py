import serial
import matplotlib.pyplot as plt
from collections import deque
import re # Import the regular expression library

# --- CONFIGURATION ---
SERIAL_PORT = 'COM6'      # ★★★ EDIT THIS to your ESP32-S3's COM port ★★★
BAUD_RATE = 115200
MAX_DATA_POINTS = 200     # How many points to show on the x-axis
# --- END CONFIGURATION ---

# Regex to parse the specific output format: "IR:12345, RED:67890, HR:75.0, SPO2:98.5"
# (\d+) matches an integer (for IR/RED)
# ([\d\.]+) matches a number that can be an integer or a float (for HR/SPO2)
DATA_REGEX = re.compile(r"IR:(\d+), RED:(\d+), HR:([\d\.]+), SPO2:([\d\.]+)")

# Create deques (double-ended queues) to store data
red_data = deque(maxlen=MAX_DATA_POINTS)
ir_data = deque(maxlen=MAX_DATA_POINTS)
spo2_data = deque(maxlen=MAX_DATA_POINTS)
hr_data = deque(maxlen=MAX_DATA_POINTS)

# Turn on Matplotlib's "interactive mode"
plt.ion()

# Create the figure and 2 subplots
# fig.set_figheight(9)
# fig.set_figwidth(15)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
fig.suptitle('Real-Time MAX30102 Data', fontsize=18)

# --- Setup Plot 1 (RED & IR) ---
ax1.set_title('Raw PPG Signals')
ax1.set_ylabel('ADC Value')
ax1.grid(True)
line_red, = ax1.plot(red_data, color='red', label='RED')
line_ir, = ax1.plot(ir_data, color='purple', label='IR')
ax1.legend(loc='upper left')

# --- Setup Plot 2 (SpO2 & HR) ---
ax2.set_title('Calculated Vitals')
ax2.set_xlabel('Time (Samples)')

# Create the first Y-axis (ax2) for SpO2
ax2.set_ylabel('SpO2 (%)', color='blue')
ax2.set_ylim(90, 101) # Fix the Y-axis for SpO2
ax2.tick_params(axis='y', labelcolor='blue')
ax2.grid(True, linestyle='--', alpha=0.6)
line_spo2, = ax2.plot(spo2_data, color='blue', label='SpO2')

# Create a *second* Y-axis (ax_hr) that shares the *same* X-axis
ax_hr = ax2.twinx()
ax_hr.set_ylabel('Heart Rate (BPM)', color='green')
ax_hr.set_ylim(50, 130) # Fix the Y-axis for HR
ax_hr.tick_params(axis='y', labelcolor='green')
line_hr, = ax_hr.plot(hr_data, color='green', label='Heart Rate')

# Combine legends for the second plot
lines, labels = ax2.get_legend_handles_labels()
lines2, labels2 = ax_hr.get_legend_handles_labels()
ax_hr.legend(lines + lines2, labels + labels2, loc='upper left')

plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle

# --- Main Loop ---
try:
    # Connect to the serial port
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
    print("Close the plot window to stop the script.")

    while plt.fignum_exists(fig.number): # Loop as long as the plot window is open
        try:
            # Read one line of data
            line = ser.readline().decode('utf-8').strip()

            # Try to match the line with our regex
            match = DATA_REGEX.search(line)
            
            # If we have a match, parse the data
            if match:
                ir_val = float(match.group(1))
                red_val = float(match.group(2))
                hr_val = float(match.group(3))
                spo2_val = float(match.group(4))

                # Add data to our deques
                ir_data.append(ir_val)
                red_data.append(red_val)
                hr_data.append(hr_val)
                spo2_data.append(spo2_val)
                
                # --- Update Plot 1 (RED & IR) ---
                x_axis_data = range(len(ir_data)) # Create a common X-axis
                line_red.set_ydata(red_data)
                line_red.set_xdata(x_axis_data)
                line_ir.set_ydata(ir_data)
                line_ir.set_xdata(x_axis_data)
                
                # --- Update Plot 2 (SpO2 & HR) ---
                line_spo2.set_ydata(spo2_data)
                line_spo2.set_xdata(x_axis_data)
                line_hr.set_ydata(hr_data)
                line_hr.set_xdata(x_axis_data)

                # --- Rescale Axes ---
                ax1.relim()
                ax1.autoscale_view()
                ax2.relim()
                ax2.autoscale_view(scaley=False) # scaley=False because we fixed the Y-limit
                ax_hr.relim()
                ax_hr.autoscale_view(scaley=False) # scaley=False because we fixed the Y-limit

                # Redraw the plot
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01) # Small pause to allow plot to update

        except Exception as e:
            print(f"Warning: Error reading or parsing data: {e}")
            print(f"Problematic line: '{line}'")

except serial.SerialException as e:
    print(f"Fatal Error: {e}")
    print(f"Could not open port '{SERIAL_PORT}'.")
    print("1. Is the ESP32-S3 plugged in?")
    print("2. Is the port correct? Check Device Manager.")
    print("3. ★ Is `idf.py monitor` (or any other serial program) CLOSED? ★")
except KeyboardInterrupt:
    print("Script stopped by user.")
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial port closed.")
    plt.ioff()
    plt.show() # Show the final plot state
    print("Plotting stopped.")