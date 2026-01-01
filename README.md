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
