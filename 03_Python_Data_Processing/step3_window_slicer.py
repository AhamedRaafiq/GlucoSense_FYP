# ==============================================================================
# ⚙️ USER CONFIGURATION SECTION
# ==============================================================================
# 1. INPUT: FOLDER (for popup) OR FILE PATH (to skip popup)
RAW_INPUT_DIR = r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\Raw"

# 2. BASE OUTPUT FOLDER
BASE_OUTPUT_DIR = r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\Windowed"

# 3. Slicing Settings
WINDOW_DURATION = 10     # Seconds
WINDOW_GAP      = 1      # Seconds (Gap between windows)
MIN_REMAINING   = 6      # Seconds (Capture tail if >= 6s remains)
FS              = 400    # Sampling Rate (Hz)
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import filedialog, Tk

# --- HELPER FUNCTIONS ---

def get_valid_input(prompt, allow_back=True):
    while True:
        user_in = input(prompt).strip().lower()
        if allow_back and (user_in == 'b' or user_in == 'back'):
            return 'BACK'
        try:
            return float(user_in)
        except ValueError:
            print("      ❌ Invalid input. Enter a number or 'b' to go back.")

def plot_signal(df, start_sec, end_sec, title_prefix="Signal"):
    s_idx = max(0, int(start_sec * FS))
    e_idx = min(len(df), int(end_sec * FS))
    
    sub_df = df.iloc[s_idx:e_idx]
    time_axis = np.linspace(start_sec, end_sec, len(sub_df))
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, sub_df['IR_Value'], label='IR (Blue)', color='blue', alpha=0.7)
    plt.plot(time_axis, sub_df['Red_Value'], label='Red (Red)', color='red', alpha=0.7)
    plt.title(f"{title_prefix} ({start_sec}s to {end_sec}s)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    print(f"\n📊 Showing Plot: {title_prefix}")
    print("   👉 CLOSE the plot window to continue...")
    plt.show()

# --- MAIN LOGIC ---

def slice_data():
    # 1. LOAD DATA
    file_path = ""
    if os.path.isfile(RAW_INPUT_DIR) and RAW_INPUT_DIR.lower().endswith('.csv'):
        print(f"📄 Direct file detected: {os.path.basename(RAW_INPUT_DIR)}")
        file_path = RAW_INPUT_DIR
    elif os.path.isdir(RAW_INPUT_DIR):
        print("📂 Opening File Dialog...")
        root = Tk()
        root.withdraw() 
        root.attributes('-topmost', True)
        file_path = filedialog.askopenfilename(initialdir=RAW_INPUT_DIR, title="Select Raw CSV", filetypes=[("CSV files", "*.csv")])
    else:
        print(f"❌ Error: Path '{RAW_INPUT_DIR}' not found.")
        return

    if not file_path: return

    try:
        df = pd.read_csv(file_path)
        df.columns = [c.strip() for c in df.columns]
        total_duration = len(df)/FS
        print(f"✅ Loaded. Duration: {total_duration:.2f} seconds.")
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return

    # 2. INTERACTIVE LOOP
    all_ranges = [] 
    
    while True:
        print("\n" + "="*60)
        print(f"   MAIN MENU | Collected Signal Blocks: {len(all_ranges)}")
        print("="*60)
        print("1. Start New Zoom & Selection")
        print("2. Finish & Save Files")
        print("3. Exit (No Save)")
        
        choice = input("👉 Select (1/2/3): ").strip()
        
        if choice == '3': return
        elif choice == '2': break 
            
        elif choice == '1':
            plot_signal(df, 0, total_duration, "FULL VIEW")
            
            print("\n🔎 SETUP ZOOM (Type 'b' to go back)")
            z_start = get_valid_input("   Zoom Start (sec): ")
            if z_start == 'BACK': continue
            z_end = get_valid_input("   Zoom End   (sec): ")
            if z_end == 'BACK': continue
            
            if z_start >= z_end:
                print("   ❌ Error: Start < End.")
                continue

            while True:
                plot_signal(df, z_start, z_end, "ZOOMED VIEW - Select Windows")
                print(f"\n➕ ADD SIGNAL BLOCK inside {z_start}s - {z_end}s")
                print("   (Type 'b' to finish this Zoom)")
                
                w_start = get_valid_input("   Block Start: ")
                if w_start == 'BACK': break
                
                w_end = get_valid_input("   Block End:   ")
                if w_end == 'BACK': continue 
                
                if w_end <= w_start:
                    print("   ❌ Error: End > Start.")
                    continue
                    
                all_ranges.append((w_start, w_end))
                print(f"   ✅ Added block: {w_start}s - {w_end}s")

    # 3. PROCESSING
    if not all_ranges:
        print("❌ No ranges selected. Exiting.")
        return

    file_name_clean = os.path.basename(file_path).replace('.csv', '')
    specific_output_folder = os.path.join(BASE_OUTPUT_DIR, file_name_clean)
    
    if not os.path.exists(specific_output_folder):
        os.makedirs(specific_output_folder)

    print(f"\n✂️ Processing {len(all_ranges)} blocks...")
    global_count = 0 
    
    for i, (start_sec, end_sec) in enumerate(all_ranges):
        current_time = start_sec
        
        # Standard Tiling Loop
        while (current_time + WINDOW_DURATION) <= end_sec:
            start_idx = int(current_time * FS)
            end_idx   = int((current_time + WINDOW_DURATION) * FS)
            
            chunk = df.iloc[start_idx:end_idx].copy()
            out_name = f"{file_name_clean}_Win{global_count}.csv"
            out_path = os.path.join(specific_output_folder, out_name)
            chunk.to_csv(out_path, index=False)
            
            print(f"   [{global_count}] Standard: {current_time:.1f}s - {current_time+WINDOW_DURATION:.1f}s")
            global_count += 1
            current_time += (WINDOW_DURATION + WINDOW_GAP)
            
        # Backfill Logic
        last_window_end = current_time - WINDOW_GAP
        # Correct calculation: The "gap" is technically unused signal until the end
        # We check the remaining usable tail from the *actual end* of the selection
        
        # If we just finished a window at T=20, and end_sec=27. 
        # Current_time is now 21.
        # Ideally, we just check: Is (end_sec - last_window_end) >= MIN_REMAINING?
        # Note: If no windows fit at all (short range), current_time == start_sec.
        
        if global_count > 0:
            remaining = end_sec - last_window_end 
        else:
            remaining = end_sec - start_sec # Special case for short blocks

        if remaining >= MIN_REMAINING:
            # Create Backfill Window aligned to end_sec
            bf_start = end_sec - WINDOW_DURATION
            bf_end   = end_sec
            
            start_idx = int(bf_start * FS)
            end_idx   = int(bf_end * FS)
            
            chunk = df.iloc[start_idx:end_idx].copy()
            
            # --- RENAMING LOGIC HERE ---
            out_name = f"{file_name_clean}_Win{global_count}_Backfill.csv"
            out_path = os.path.join(specific_output_folder, out_name)
            
            chunk.to_csv(out_path, index=False)
            print(f"   [{global_count}] Backfill: {bf_start:.1f}s - {bf_end:.1f}s (Saved as _Backfill)")
            global_count += 1

    print("\n" + "="*50)
    print(f"✅ DONE! Generated {global_count} windows.")
    print(f"📂 Location: {specific_output_folder}")
    print("="*50)

if __name__ == "__main__":
    slice_data()