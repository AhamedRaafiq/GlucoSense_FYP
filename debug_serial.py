import serial
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# --------- USER CONFIG ------------
PORT = "COM6"
BAUD = 115200
WINDOW_SIZE = 200  # samples in window
# ---------------------------------

ser = serial.Serial(PORT, BAUD)
print(f"Connected to {PORT}")

# Buffers
red_vals = deque(maxlen=WINDOW_SIZE)
ir_vals = deque(maxlen=WINDOW_SIZE)
hr_vals = []
spo2_vals = []

plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
line_red, = ax1.plot([], [], 'r-', label='RED')
line_ir, = ax1.plot([], [], 'b-', label='IR')
ax1.set_title("MAX30102 Raw PPG Signal")
ax1.legend()
ax2.set_title("Heart Rate & SpO₂ over Time")
line_hr, = ax2.plot([], [], 'g-', label='HR (bpm)')
line_spo2, = ax2.plot([], [], 'm-', label='SpO₂ (%)')
ax2.legend()

def estimate_hr_spo2(ir):
    # Simple placeholder algorithm – replace with your own filter later
    ir = np.array(ir)
    if len(ir) < 30:
        return 0.0, 0.0

    # Remove DC offset
    ir_mean = np.mean(ir)
    ir_centered = ir - ir_mean

    # Simple zero-crossing heart rate estimate
    crossings = np.where(np.diff(np.sign(ir_centered)))[0]
    if len(crossings) < 2:
        return 0.0, 0.0

    avg_period = np.mean(np.diff(crossings))
    hr = 60.0 / (avg_period * 0.5 * 0.5)  # assuming ~2Hz sampling
    spo2 = 95 + 5 * np.random.random()    # dummy SpO₂ (for now)
    return hr, spo2

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
        if not line:
            continue

        # Expecting "RED:xxxx,IR:yyyy"
        if line.startswith("RED:"):
            try:
                parts = line.replace("RED:", "").split(",IR:")
                red = int(parts[0])
                ir = int(parts[1])
                red_vals.append(red)
                ir_vals.append(ir)

                # Estimate HR and SpO₂
                hr, spo2 = estimate_hr_spo2(ir_vals)
                hr_vals.append(hr)
                spo2_vals.append(spo2)

                # Update plots
                line_red.set_data(range(len(red_vals)), list(red_vals))
                line_ir.set_data(range(len(ir_vals)), list(ir_vals))
                ax1.relim()
                ax1.autoscale_view()

                line_hr.set_data(range(len(hr_vals)), hr_vals)
                line_spo2.set_data(range(len(spo2_vals)), spo2_vals)
                ax2.relim()
                ax2.autoscale_view()

                plt.pause(0.01)
                print(f"HR: {hr:.1f} bpm, SpO₂: {spo2:.1f}%")

            except Exception as e:
                print("Parse error:", e, line)
                continue

    except KeyboardInterrupt:
        print("Exiting...")
        break

ser.close()
