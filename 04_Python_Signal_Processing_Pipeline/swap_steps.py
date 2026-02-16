import json
import os

nb_path = r"c:/Users/DELL/Documents/GitHub/fyp/04_Python_Signal_Processing_Pipeline/Filters_Normalization_SignalQualityCheck.ipynb"

def swap_steps():
    if not os.path.exists(nb_path):
        print(f"Error: Notebook not found at {nb_path}")
        return

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    cells = nb['cells']
    step10_idx = -1
    step11_idx = -1

    # Locate cells
    for i, cell in enumerate(cells):
        source = "".join(cell.get('source', []))
        if "STEP 10: GLUCOSE FEATURE EXTRACTION" in source:
            step10_idx = i
        elif "STEP 11: FULL PIPELINE DIAGNOSTIC REPORT" in source:
            step11_idx = i

    if step10_idx == -1 or step11_idx == -1:
        print(f"Error: Could not find both steps. Step 10 found: {step10_idx != -1}, Step 11 found: {step11_idx != -1}")
        return

    print(f"Found Step 10 at index {step10_idx}")
    print(f"Found Step 11 at index {step11_idx}")

    # Extract cells
    cell10 = cells[step10_idx]
    cell11 = cells[step11_idx]

    # Modify Headers
    # Rename Step 10 -> Step 11
    new_source10 = []
    for line in cell10['source']:
        new_source10.append(line.replace("STEP 10:", "STEP 11:"))
    cell10['source'] = new_source10

    # Rename Step 11 -> Step 10
    new_source11 = []
    for line in cell11['source']:
        new_source11.append(line.replace("STEP 11:", "STEP 10:"))
    cell11['source'] = new_source11

    # Swap in the list
    # We need to be careful with indices if we pop/insert.
    # Easiest way: create a new list or assign by index if mapped 1:1
    
    # Assuming standard order 10 then 11:
    if step10_idx < step11_idx:
        cells[step10_idx] = cell11
        cells[step11_idx] = cell10
    else:
        # If they are already out of order (unlikely given the request), just swap
        cells[step11_idx] = cell10
        cells[step10_idx] = cell11
        
    print("Swapped cells and updated headers.")

    # Save
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    
    print("Notebook saved successfully.")

if __name__ == "__main__":
    swap_steps()
