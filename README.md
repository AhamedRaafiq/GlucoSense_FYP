<div align="center">

# 🩸 GlucoSense

### Non-Invasive Blood Glucose Estimation via PPG Optical Signal Analysis

[![License: Source-Available](https://img.shields.io/badge/License-Source--Available-orange.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://python.org)
[![ESP-IDF](https://img.shields.io/badge/ESP--IDF-v5.x-E7352C?logo=espressif&logoColor=white)](https://docs.espressif.com/projects/esp-idf/)
[![React 18](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-FF6600?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io)
[![PostgreSQL 15](https://img.shields.io/badge/PostgreSQL-15-4169E1?logo=postgresql&logoColor=white)](https://www.postgresql.org)

<br/>

<img src="GlucoSense_Hardware.png" alt="GlucoSense Hardware — Exploded View" width="720"/>

<br/>

*An end-to-end research platform that captures dual-wavelength PPG signals from a fingertip sensor, processes them through a robust 12-stage DSP pipeline, and predicts blood glucose concentration using machine learning — achieving up to **100% Clarke Error Grid Zone A** compliance.*

---

**[Explore the Pipeline ⟶](PIPELINE_FLOWCHART.md)** · **[Contributing ⟶](CONTRIBUTING.md)** · **[Project Report ⟶](GlucoSense_Project_Report.html)**

</div>

<br/>

## 🔬 The Problem

Millions of diabetics worldwide rely on **painful finger-prick blood tests** multiple times a day. GlucoSense explores a fundamentally different approach: shining light through the fingertip and reading glucose levels from the way blood absorbs it — **no needles, no blood, no pain.**

## 💡 How It Works

GlucoSense exploits the **Beer-Lambert Law** — changes in blood glucose alter plasma osmolarity, blood viscosity, and hemoglobin glycation (HbA₁c), which in turn modulate how Red (660 nm) and Infrared (940 nm) light is absorbed by arterial blood. By capturing these subtle photoplethysmographic (PPG) waveform variations at high resolution and extracting clinically meaningful features, a trained XGBoost model can estimate glucose concentration in mg/dL.

<br/>

## 🏗️ System Architecture

```
┌─────────────────┐     UART      ┌──────────────────┐     CSV       ┌──────────────────────────┐
│  ESP32-S3       │──── 400 Hz ──▶│  Python Data      │────────────▶│  12-Stage DSP Pipeline    │
│  + MAX30102     │   IR & RED    │  Logger (PyQt5)   │  Timestamped │  (Butterworth, Peak Det,  │
│  Optical Sensor │               │  60 FPS Real-Time │  Raw Data    │   SQI Gating, Ensemble)   │
└─────────────────┘               └──────────────────┘              └────────────┬─────────────┘
                                                                                 │
                                                              39 Morphological Features
                                                                                 ▼
┌─────────────────┐    REST API   ┌──────────────────┐  Prediction  ┌──────────────────────────┐
│  React 18       │◀────────────▶│  FastAPI Backend  │◀────────────│  XGBoost Regression       │
│  Dashboard      │   + WebSocket │  + PostgreSQL 15  │   mg/dL +   │  (24 Engineered Features, │
│  (Vite, Charts) │               │  (SQLAlchemy 2.0) │   Zone A-E  │   RobustScaler, 5-Fold CV)│
└─────────────────┘               └──────────────────┘              └──────────────────────────┘
```

<br/>

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Hardware** | ESP32-S3 (Dual-Core LX7 @ 240 MHz) · MAX30102 PPG Sensor (18-bit ADC) |
| **Firmware** | Embedded C · ESP-IDF v5.x · FreeRTOS · I²C @ 400 kHz |
| **Signal Processing** | Python · SciPy (Butterworth Filters, Welch PSD) · NumPy · Matplotlib |
| **Machine Learning** | XGBoost · Scikit-Learn (RobustScaler, Stratified K-Fold, Clarke Grid) |
| **Backend** | FastAPI · Uvicorn · SQLAlchemy 2.0 · Pydantic v2 · WebSockets |
| **Frontend** | React 18 · Vite · React Router v6 · Recharts · Lucide Icons |
| **Database & DevOps** | PostgreSQL 15 · pgAdmin 4 · Docker Compose |

<br/>

## 📊 Pipeline at a Glance

The data flows through **4 phases and 11 sequential steps** — from photons hitting the fingertip to a glucose prediction on screen:

| Phase | Steps | What Happens |
|---|---|---|
| **A — Acquisition** | `01` → `02` | ESP32 captures raw IR/RED photocurrent at 400 Hz; Python logger streams and timestamps it to CSV in real-time |
| **B — Signal Processing** | `03` → `04` | Artifact-free 15s windows are extracted, then run through a 12-stage DSP engine (bandpass filtering, peak detection, beat validation, ensemble averaging, 5 SQI quality gates) |
| **C — Feature Engineering** | `05` → `08` | 19 physiological features per channel (time-domain, spectral, derivative) are extracted, averaged across windows, fused with reference glucometer readings, and engineered into a final 24-feature vector |
| **D — Machine Learning** | `09` → `11` | Dataset is cleaned (NaN + IQR outlier removal), split 80/20 with RobustScaler normalisation, and trained with XGBoost including clinical sample weighting for hyperglycemic cases |

> 📌 **Full interactive flowchart →** [`PIPELINE_FLOWCHART.md`](PIPELINE_FLOWCHART.md)

<br/>

## 🏆 Key Results

| Metric | Value |
|---|---|
| **Clarke Error Grid — Zone A** | **91.67% – 100.0%** ✅ |
| **5-Fold CV MAE** | 11.23 – 11.60 mg/dL |
| **Test MAE** | 11.86 – 12.45 mg/dL |
| **Test RMSE** | 13.61 – 14.24 mg/dL |
| **Test MAPE** | 11.40% – 12.07% |

> Zone A compliance meets or exceeds the **ISO 15197** clinical tolerance standard for non-invasive glucose monitoring devices.

**Top predictive features:** `IR_pulse_width` · `IR_PPI` · `IR_HRV` · `Diff_Spectral_Entropy` · `Ratio_TEO_Mean`

<br/>

## 🚀 Quick Start

### Prerequisites

- **Python 3.10+** · **Node.js 18+** · **Docker & Docker Compose** · **ESP-IDF v5.x** (for firmware)

### 1 · Clone & Install

```bash
git clone https://github.com/AhamedRaafiq/GlucoSense_FYP.git
cd GlucoSense_FYP
pip install -r requirements.txt
```

### 2 · Spin Up the Database

```bash
docker-compose up -d
```
> PostgreSQL 15 on `localhost:5432` · pgAdmin at `localhost:5050`

### 3 · Launch the Backend

```bash
cd Backend
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4 · Launch the Frontend

```bash
cd Frontend
npm install
npm run dev
```

### 5 · Run the Inference Engine (Standalone)

```bash
# Mode 1: From an existing CSV
python GlucoSense_Inference_Engine.py

# Mode 2: Live from ESP32
python GlucoSense_Inference_Engine.py   # Select live acquisition when prompted
```

<br/>

## 📁 Project Structure

```
GlucoSense/
├── 01_Firmware_ESP32/            # ESP32-S3 + MAX30102 embedded C firmware
├── 02_Python_Data_Logger/        # Real-time PyQt5 dual-channel waveform logger
├── 03_Python_Data_Processing/    # Interactive window slicer tool
├── 04_Python_Signal_Processing/  # 12-stage automated DSP engine (~3,000 lines)
├── 05_Signal_Feature_Learning/   # 19-feature per-channel extraction + averaging
├── 06_Data_Set_Creation/         # PPG ↔ glucometer metadata fusion
├── 07_Data_Set_Processing/       # 24-feature engineering & dataset cleaning
├── 08_Machine_Learning_Models/   # XGBoost training, tuning & Clarke Grid eval
├── 08_Results_and_Visualizations/# 17 experiment runs + tuning history
├── 09_Tests/                     # pytest unit tests
├── 10_Docs_&_R_Papers/           # Research papers & documentation
├── Backend/                      # FastAPI + SQLAlchemy + WebSocket server
├── Frontend/                     # React 18 + Vite clinical dashboard
├── GlucoSense_Inference_Engine.py  # Unified end-to-end runtime (~1,800 lines)
├── docker-compose.yml            # PostgreSQL 15 + pgAdmin containers
└── requirements.txt              # Python dependencies
```

<br/>

## 🤝 Contributing

Contributions are welcome! Please read the [Contributing Guidelines](CONTRIBUTING.md) before submitting a PR. We follow standard GitHub flow with semantic commit prefixes (`feat:`, `fix:`, `docs:`, `refactor:`, `test:`).

<br/>

## 📄 License

This project is released under a **Source-Available License** — see the [LICENSE](LICENSE) file for details.

> 👁️ You are free to **view, review, and provide feedback** on this repository.  
> 🔒 **Copying, modification, distribution, or use** of this code requires **written permission** from the author.

> ⚠️ **Disclaimer:** GlucoSense is an academic research prototype. It has **not** been validated for clinical use and must **not** be used for medical diagnosis or treatment decisions.

<br/>

<div align="center">

---

Built with 🔬 and ☕ by **Ahamed Raafiq** · Final Year Project · 2026

</div>
