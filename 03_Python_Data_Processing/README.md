# 03_Python_Data_Processing

## Overview
This directory contains Python scripts and notebooks for processing raw PPG (Photoplethysmography) data collected from the MAX30102 sensor.

## Purpose
- Data validation and quality checks
- Initial data exploration and analysis
- Data format conversion
- Preprocessing before signal processing pipeline

## Workflow Position
```
Data Logger (02) → **Data Processing (03)** → Signal Processing (04) → ML Models (07)
```

## Contents
*This folder is currently empty. Add your data processing scripts here.*

## Typical Processing Steps
1. Load raw CSV files from `05_Data_Storage/Raw/`
2. Validate data integrity
3. Handle missing values
4. Remove artifacts and outliers
5. Save processed data to `05_Data_Storage/Cleaned/`

## Usage
Add your Python scripts or Jupyter notebooks here for data processing tasks.
