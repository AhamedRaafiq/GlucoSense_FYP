# 🧪 Tests & Validation

This directory contains the comprehensive testing suite for the GlucoSense pipeline. It validates all system components, ensuring the reliability of the firmware, signal processing algorithms, machine learning models, and system integration.

## 🚀 Key Features

- **Firmware Testing:** Validates I2C communication, sensor initialization routines, and UART data transmission (using ESP-IDF testing frameworks).
- **Data Processing Validation:** Unit tests for signal normalization, filtering algorithms, and feature extraction correctness.
- **ML Model Verification:** Tests for training reproducibility, evaluation metric accuracy, cross-validation logic, and edge-case handling.
- **Integration Tests:** End-to-end data flow validation and system benchmark testing.

## 📁 Directory Structure

- `test_firmware/`: Hardware and microcontroller unit tests.
- `test_data_processing/`: Validation for signal cleaning and feature generation.
- `test_ml_models/`: ML pipeline reproducibility and scoring checks.
- `test_integration/`: End-to-end pipeline execution tests.
- `test_data/`: Mock datasets and fixtures for test isolation.

## 🎯 Coverage Goals

- **General Code Coverage:** > 80%
- **Critical Paths:** 100% (Clinical weighting logic, vital signal processing).

## 🚀 Quick Start

**Run All Tests**
```bash
pytest 09_Tests/
```

**Run Specific Test Suite**
```bash
pytest 09_Tests/test_data_processing/test_normalization.py
```
*(Continuous Integration via GitHub Actions is planned for automated test execution).*
