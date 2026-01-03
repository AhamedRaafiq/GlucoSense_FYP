"""
CSV Column Converter
Converts old format (IR, RED) to new format (IR_Value, Red_Value)
"""

import os
import pandas as pd
from pathlib import Path

# Configuration
WINDOWED_FOLDER = r"C:\Users\DELL\Documents\GitHub\fyp\05_Data_Storage\Windowed"

def convert_csv_columns(file_path):
    """Convert a single CSV file to new column format"""
    try:
        # Read CSV
        df = pd.read_csv(file_path)
        
        # Check if conversion is needed
        if 'IR' in df.columns and 'RED' in df.columns:
            # Rename columns
            df = df.rename(columns={'IR': 'IR_Value', 'RED': 'Red_Value'})
            
            # Save back to file
            df.to_csv(file_path, index=False)
            return True, "Converted"
        elif 'IR_Value' in df.columns and 'Red_Value' in df.columns:
            return False, "Already in new format"
        else:
            return False, f"Unknown format: {df.columns.tolist()}"
            
    except Exception as e:
        return False, f"Error: {e}"

def main():
    print("="*60)
    print("CSV Column Format Converter")
    print("="*60)
    print(f"Scanning folder: {WINDOWED_FOLDER}\n")
    
    # Find all CSV files recursively
    csv_files = list(Path(WINDOWED_FOLDER).rglob("*.csv"))
    
    if not csv_files:
        print("❌ No CSV files found.")
        return
    
    print(f"Found {len(csv_files)} CSV file(s)\n")
    
    converted_count = 0
    skipped_count = 0
    error_count = 0
    
    for csv_file in csv_files:
        success, message = convert_csv_columns(str(csv_file))
        
        if success:
            print(f"✅ {csv_file.name}: {message}")
            converted_count += 1
        elif "Already" in message:
            print(f"⏭️  {csv_file.name}: {message}")
            skipped_count += 1
        else:
            print(f"❌ {csv_file.name}: {message}")
            error_count += 1
    
    print("\n" + "="*60)
    print(f"✅ Converted: {converted_count}")
    print(f"⏭️  Skipped: {skipped_count}")
    print(f"❌ Errors: {error_count}")
    print("="*60)

if __name__ == "__main__":
    main()
