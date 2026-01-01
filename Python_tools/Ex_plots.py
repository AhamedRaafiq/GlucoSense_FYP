# --- CELL 1: Imports and Setup ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuration
FILENAME = 'ppg_data_A0014_rms01_0014_T21.csv'  # Ensure this matches your file name
FS_ORIGINAL = 400  # Input Hz (from ESP32)
DURATION_SEC = 30  # Seconds to plot

print(f"Configuration: Plotting at full {FS_ORIGINAL}Hz (No downsampling)")

# --- CELL 2: Robust Data Loading ---
try:
    # 1. Load Data with 'error_bad_lines' handling logic
    # We read everything as strings first to avoid immediate crashes
    df_raw = pd.read_csv(FILENAME, header=None, names=['IR', 'Red'], dtype=str)
    
    print(f"Raw Lines Read: {len(df_raw)}")

    # 2. Force Convert to Numbers (Coerce errors to NaN)
    # This turns "Sensor Configured" or "I2C Error" into NaN (Not a Number)
    df_raw['IR'] = pd.to_numeric(df_raw['IR'], errors='coerce')
    df_raw['Red'] = pd.to_numeric(df_raw['Red'], errors='coerce')

    # 3. Drop Clean-up
    # Remove any rows that turned into NaN (the text lines)
    df = df_raw.dropna().reset_index(drop=True)
    
    print(f"Cleaned Valid Samples: {len(df)} (Removed {len(df_raw) - len(df)} bad lines)")

    if len(df) < 100:
        raise ValueError("Not enough valid data! Check your CSV file content.")

    # 4. Explicitly convert to Numpy Float Arrays (Fixes 'No matching signature' error)
    ir_signal = df['IR'].values.astype(float)
    red_signal = df['Red'].values.astype(float)

    # 5. Downsample (Decimate)
    # Now passing pure float arrays, which decimate loves
    ir_50hz = decimate(ir_signal, DOWN_FACTOR, ftype='fir')
    red_50hz = decimate(red_signal, DOWN_FACTOR, ftype='fir')

    # 6. Create Time Axis
    num_points_new = len(ir_50hz)
    time_axis_50hz = np.linspace(0, num_points_new / FS_TARGET, num_points_new)

    print("Processing Complete.")

    # --- CELL 3: Plotting ---
    # Slice data to requested duration
    samples_to_plot = DURATION_SEC * FS_TARGET
    limit = min(samples_to_plot, len(ir_50hz))
    
    y_ir = ir_50hz[:limit]
    y_red = red_50hz[:limit]
    x_time = time_axis_50hz[:limit]

    plt.figure(figsize=(12, 6))
    plt.plot(x_time, y_ir, label='IR (50Hz)', color='blue', linewidth=1.5)
    plt.plot(x_time, y_red, label='Red (50Hz)', color='red', linewidth=1.5, alpha=0.8)

    plt.title(f'Filtered PPG Signal ({FS_TARGET}Hz) - First {DURATION_SEC}s')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()

except FileNotFoundError:
    print(f"ERROR: Could not find '{FILENAME}'. Make sure the file is in the same folder as this script.")
except Exception as e:
    print(f"An error occurred: {e}")