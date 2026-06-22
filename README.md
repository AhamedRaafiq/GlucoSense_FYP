# Non-Invasive Diabetes Prediction using PPG Signals

## 🎯 Project Overview

This project develops a non-invasive diabetes prediction system using **Photoplethysmography (PPG)** signals captured from a **MAX30102** sensor interfaced with an **ESP32-S3** microcontroller. The system processes PPG waveforms to extract physiological features and applies machine learning algorithms to predict diabetes risk.

## 🏗️ System Architecture

```
┌─────────────────┐
│   ESP32-S3      │
│   + MAX30102    │  ──► Raw PPG Signals (IR, RED)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  Data Logger    │  ──► CSV Files
│  (Python)       │
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Signal          │  ──► Filtered Signals
│ Processing      │      + Features
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ Machine         │  ──► Diabetes Risk
│ Learning        │      Prediction
└─────────────────┘
```

## 📁 Project Structure

```
fyp/
├── 01_Firmware_ESP32/              # ESP32 firmware for PPG acquisition
├── 02_Python_Data_Logger/          # Data capture and logging tools
├── 03_Python_Signal_Processing_Pipeline/  # Signal processing notebooks
├── 04_Data_Storage/                # Organized data storage (Raw, Cleaned, Features)
├── 05_Docs_&_R_Papers/             # Documentation and research papers
├── 06_Machine_Learning_Models/     # ML models for diabetes prediction
├── 07_Results_and_Visualizations/  # Results, plots, and reports
├── 08_Tests/                       # Test suites
└── scripts/                        # Utility scripts
```

## 🚀 Quick Start

### Hardware Requirements
- **ESP32-S3** development board
- **MAX30102** pulse oximeter sensor
- USB cable for programming and data transfer
- Jumper wires for connections

### Pin Connections
| ESP32-S3 | MAX30102 |
|----------|----------|
| GPIO 1   | SDA      |
| GPIO 2   | SCL      |
| 3.3V     | VIN      |
| GND      | GND      |

### Software Requirements
- **ESP-IDF** (v5.0+) for firmware development
- **Python 3.8+** for data processing
- **Jupyter Notebook** for signal processing

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/AhamedRaafiq/fyp_new.git
   cd fyp
   ```

2. **Install Python dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up ESP-IDF**:
   ```bash
   # Follow ESP-IDF installation guide
   # https://docs.espressif.com/projects/esp-idf/en/latest/esp32/get-started/
   ```

4. **Build and flash firmware**:
   ```bash
   cd 01_Firmware_ESP32
   idf.py set-target esp32s3
   idf.py build
   idf.py -p COM3 flash monitor  # Replace COM3 with your port
   ```

5. **Capture data**:
   ```bash
   cd ../02_Python_Data_Logger
   python data_logger.py --port COM3 --output data.csv
   ```

6. **Process signals**:
   ```bash
   cd ../03_Python_Signal_Processing_Pipeline
   jupyter notebook
   # Open and run notebooks sequentially
   ```

## 🔬 Methodology

### 1. Signal Acquisition
- Dual-wavelength PPG (Red: 660nm, IR: 880nm)
- 400 Hz sampling rate
- 18-bit ADC resolution
- Configurable LED currents

### 2. Signal Processing
- **Normalization**: Min-max and z-score scaling
- **Filtering**: Low-pass and Savitzky-Golay filters
- **Feature Extraction**: Time-domain, frequency-domain, and morphological features

### 3. Machine Learning
- Classification algorithms (Random Forest, SVM, Neural Networks)
- Feature selection and dimensionality reduction
- Cross-validation and performance evaluation

## 📊 Features Extracted

- **Time-Domain**: Peak amplitude, inter-beat intervals, pulse width
- **Frequency-Domain**: FFT components, power spectral density
- **Morphological**: Rise time, fall time, pulse shape characteristics
- **Statistical**: Mean, variance, skewness, kurtosis

## 🛠️ Technologies Used

- **Hardware**: ESP32-S3, MAX30102
- **Firmware**: C, ESP-IDF, FreeRTOS
- **Data Processing**: Python, NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn, TensorFlow/PyTorch
- **Development**: Jupyter Notebook, Git, VS Code

## 📖 Documentation

Each folder contains a detailed README:
- [Firmware Documentation](01_Firmware_ESP32/README.md)
- [Data Logger Guide](02_Python_Data_Logger/README.md)
- [Signal Processing Pipeline](03_Python_Signal_Processing_Pipeline/README.md)
- [Data Storage Organization](04_Data_Storage/README.md)

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Ahamed Raafiq** - *Initial work* - [AhamedRaafiq](https://github.com/AhamedRaafiq)

## 🙏 Acknowledgments

- Research papers and references in `05_Docs_&_R_Papers/`
- ESP-IDF framework by Espressif Systems
- MAX30102 sensor documentation by Maxim Integrated

## 📧 Contact

For questions or collaboration opportunities, please open an issue or contact the project maintainer.

---

**Note**: This is an academic research project for Final Year Project (FYP). Results should be validated before clinical use.













I need you to create a comprehensive README.md file for my Python code. This README serves two purposes: (1) a practical guide for code users and collaborators, and (2) reference material I can later extend into my final year project (FYP) thesis.

## ABOUT THE PROJECT

This code is part of a multi-step pipeline for non-invasive blood glucose estimation using PPG (photoplethysmography) signals. The full pipeline has these steps:

Step 1-3: Raw data collection → Verification → Windowing
Step 4: Automated Signal Processing (filtering, beat detection, ensemble averaging)
Step 5: Feature Extraction (19 features per channel)
Step 6: Average Feature Extraction (per-subject averaging)
Step 7: Feature Engineering (ratio/difference features)
Step 8: Data Cleaning + Train/Test Split + Scaling
Step 9: XGBoost ML Model Training + Evaluation

The code I need a README for is: [PASTE YOUR CODE'S STEP NAME AND BRIEF DESCRIPTION HERE]

## THE CODE

[PASTE YOUR FULL CODE HERE — or describe it in detail: what it does, inputs, outputs, key functions, hyperparameters]

## README SPECIFICATIONS

Follow these exact specifications:

### Length & Detail
- Target: 3000-5000 words (collaborator-level, not thesis-deep)
- Average detail — practical and code-friendly
- Can be extended to thesis-level later

### Writing Style
- Professional but approachable
- Light emoji usage (section headers only, not everywhere)
- Use action verbs and clear descriptions
- Include "why" explanations (1-2 sentences per design decision)
- Example: Don't just say "applies Butterworth filter" — say "applies 4th-order Butterworth filter because it provides optimal flatness in passband, important for preserving PPG morphology"

### Code Snippets
- Include main imports with short usage comment beside each import
- Include function signatures only (no function bodies)
- Group functions by category (processing, quality, ensemble, helpers, etc.)

### Diagrams & Flowcharts
- Include both text-based ASCII flowcharts AND Mermaid syntax flowcharts
- For complex visuals, include text descriptions of what diagrams to create (labeled as "Suggested Diagram to Create" with color coding, tools, and size recommendations)

### Hyperparameters
- Include ALL hyperparameters in a dedicated section
- Group by category (File I/O, Filters, Beat Detection, Quality, etc.)
- Use tables with columns: Parameter | Default | Description
- Each description should be 1-2 lines — short but understandable

### No Citations
- Skip references/citations section (I will add later)

### No Version History
- Skip changelog/version history section

## REQUIRED SECTIONS (in this exact order)

### Section 1: Title & TL;DR (~200 words)
- Project title as H1 heading
- One-line description as blockquote
- 3-sentence summary of what the code does
- Quick stats box: lines of code, processing time, key metrics

### Section 2: Table of Contents
- Auto-navigable clickable links to all sections

### Section 3: Quick Start (~300 words)
- Minimum steps to run (numbered bash commands)
- Expected first-run output description
- "If things go wrong, see Troubleshooting section" note

### Section 4: Background & Motivation (~400 words)
- What the problem is (brief, non-thesis)
- Why this code exists
- Where it fits in the larger pipeline (show with simple text diagram)

### Section 5: Pipeline/Process Overview (~400 words)
- Text-based ASCII flowchart of all major stages
- Mermaid flowchart (renders on GitHub)
- Stage summary table (# | Stage | What It Does)
- Suggested diagram description for visual creation

### Section 6: Features & Capabilities (~300 words)
- Core functionality bullet list
- Processing modes (if any)
- Quality assurance features
- Configurability highlights
- Output traceability features
- Failure handling approach

### Section 7: Installation & Prerequisites (~300 words)
- System requirements table (Python, OS, RAM, Disk)
- Dependencies list with minimum versions
- Virtual environment setup (Windows + Linux/Mac)
- Verification command

### Section 8: Input Data Format (~400 words)
- Expected file format with sample snippet
- Required columns/fields
- File naming convention
- Expected folder structure (text tree)
- Data quality requirements

### Section 9: Output Structure (~400 words)
- Complete output folder tree (text-based)
- Description of each output file (what it contains, column names)
- Plot file naming convention
- Suggested diagram for folder structure

### Section 10: Usage Examples (~400 words)
- 3-4 concrete step-by-step examples covering different modes/scenarios
- Expected terminal output for each example
- A "tuning/debugging" example showing how to adjust parameters

### Section 11: Detailed Methodology (~1500 words) — LARGEST SECTION
- For each major processing stage/step in the code:
  * Purpose (1 sentence)
  * Method (1-2 sentences with brief justification of why this approach)
  * Key hyperparameters with defaults
  * Input → Output
  * Visualization filename (if any)
- This section should explain the "what" and "why" of each stage
- Don't go thesis-deep but give enough context that a collaborator understands the design decisions

### Section 12: Hyperparameter Reference (~600 words)
- Grouped tables by category
- Columns: Parameter | Default | Description
- Each description: 1-2 lines, short but clear
- Include tuning notes where relevant (e.g., "Lower this for weak signals")

### Section 13: Key Functions & Architecture (~400 words)
- Main imports with usage comments
- Function signatures grouped by category
- Brief description of what each function does (1 line each)
- Suggested diagram for function call graph

### Section 14: Quality Assessment (~300 words)
- Quality metrics explained briefly
- How rejection/validation works
- Sample terminal output showing quality table

### Section 15: Troubleshooting & Tuning Guide (~400 words)
- Common symptoms table: Symptom | Likely Cause | Parameter to Adjust
- Step-by-step debugging workflow
- Tips for different data types/edge cases
- When to accept a failure vs when to tune

## OUTPUT FORMAT

- Deliver as a SINGLE continuous Markdown file
- Everything between one pair of code fence markers
- Ready to copy-paste directly into a README.md file
- Use GitHub-flavored Markdown
- Tables should render properly on GitHub
- Mermaid blocks should use ```mermaid fencing
- Code blocks should use appropriate language tags (python, bash, csv, etc.)

## IMPORTANT RULES

1. DO NOT split into multiple batches — deliver everything in ONE response
2. DO NOT add sections I didn't ask for
3. DO NOT include citations or references
4. DO NOT include version history or changelog
5. DO preserve the section numbering and order exactly as specified
6. DO make all table of contents links work with GitHub anchor format
7. DO include diagram creation suggestions (not actual images)
8. DO keep explanations practical, not academic
9. DO ensure all code snippets use proper syntax highlighting
10. DO make the README standalone — a new collaborator should understand the code without reading the source