# ✂️ Interactive PPG Windowing Tool

> **Manual Quality Gate for Photoplethysmography (PPG) Data**

This interactive graphical tool allows researchers to visually inspect raw PPG recordings and manually extract clean, artifact-free 15-second windows. It acts as the critical manual quality gate in the data processing pipeline before automated signal processing.

## 🔄 Pipeline Position
**Step 03** — Raw Data Curation (precedes automated signal processing in Step 04).

## ✨ Key Features
- **Dual-Channel Viewer**: Synchronized RED and IR signal visualization with linked X-axes.
- **Interactive Sliding Viewport**: 30-second sliding viewport controlled by a horizontal scroll slider.
- **Point-and-Click Curation**: Click to drop a translucent 15-second selection overlay window.
- **Smart Column Detection**: Automatic mapping for various column name aliases (e.g., `IR_Value`, `ir_value`, `infrared`).
- **Precision Output**: Slices exact 6000-row subsets (15 seconds × 400 Hz) while preserving all metadata.
- **GUI Controls**: Intuitive buttons for adding blocks, undoing last selection, and batch writing.

## ⚙️ Configuration Parameters

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `WINDOW_DURATION` | `15` | Fixed duration of the extracted window in seconds. |
| `FS` | `400` | Sampling frequency of the PPG data in Hz. |
| `VIEW_WINDOW_SEC` | `30` | Width of the visible sliding viewport in seconds. |
| `RAW_INPUT_DIR` | (User defined) | Directory containing raw subject CSV files. |
| `BASE_OUTPUT_DIR` | (User defined) | Root directory for outputting extracted windows. |

## 📥 Input → Output Format

**Input Format:** Raw CSV files containing RED and IR PPG signals with associated metadata.
**Output Format:** Extracted CSV subsets representing exact 15-second chunks (6000 rows).

**Output Directory Structure:**
```text
BASE_OUTPUT_DIR/
├── {Subject_ID}/
│   ├── {Subject_ID}_Win0.csv
│   ├── {Subject_ID}_Win1.csv
│   └── ...
```

## 🚀 Quick Start

1. **Install Dependencies**
```bash
pip install numpy pandas matplotlib
# Note: On Linux, you may also need to install python3-tk
```

2. **Run the Tool**
```bash
# Note: Replace with the actual filename if different
python main.py
```

> [!WARNING]
> Ensure you use the `TkAgg` backend for Matplotlib. The script must execute `matplotlib.use('TkAgg')` **before** importing `matplotlib.pyplot`.

## 🧠 Acceptance Criteria for Selection
When selecting windows, visually inspect for the following criteria:

- **Accept**: Regular pulses, stable baseline, visible dicrotic notch, and matched RED/IR morphology.
- **Reject**: Motion artifacts/spikes, flatlines, sensor clipping, or severe baseline drift.

## 🛠️ Troubleshooting

| Issue | Solution |
| :--- | :--- |
| **GUI Backend Error** | Ensure `python3-tk` is installed and `matplotlib.use('TkAgg')` is called first. |
| **KeyError on Columns** | Verify input CSV column names. The tool auto-maps standard aliases. |
| **Window Out of Bounds** | The tool auto-clamps selections to prevent out-of-bounds errors at the recording's end. |