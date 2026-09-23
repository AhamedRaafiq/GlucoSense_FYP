# 🛠️ Project Automation Scripts

Automated tools for environment setup, data management, device flashing, and batch processing.

This module provides cross-platform utility scripts to streamline repetitive tasks such as configuring the development environment, managing data lifecycles, and deploying firmware to hardware components.

**Pipeline Position:** Cross-cutting automation and utility layer.

## ✨ Key Features
- **Environment Setup:** Automated dependency installation and system configuration for Linux, Mac, and Windows.
- **Data Lifecycle Management:** Automated backups, data reorganization, and stale file cleanup.
- **Hardware Deployment:** Streamlined ESP32 firmware flashing and model deployment.
- **General Utilities:** System health checks, status reporting, and batch processing.

## 📂 Structure
- `setup/`: Environment initialization scripts.
- `data_management/`: Backup, organization, and cleanup scripts.
- `deployment/`: Firmware and model deployment tools.
- `utilities/`: System checks, reporting, and batch execution.

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Linux/Mac
./scripts/setup/setup_environment.sh

# Windows/Cross-platform (Python)
python scripts/setup/setup_environment.py
```

### 2. Data Management (Backup)
```bash
python scripts/data_management/backup_data.py --source 04_Data_Storage/Raw/ --dest backup/
```

### 3. Firmware Deployment
```bash
./scripts/deployment/flash_firmware.sh --port COM3
```

> **Note:** All Python scripts support the `--help` flag for detailed usage instructions and argument validation.

## ⚙️ Core Scripts Reference

| Category | Script | Description |
|---|---|---|
| **Setup** | `setup_environment.[sh\|py]` | Initializes workspace and installs dependencies |
| **Setup** | `install_requirements.py` | Installs Python package requirements |
| **Data** | `backup_data.py` | Copies data from source to backup destination |
| **Data** | `clean_old_data.py` | Removes stale or intermediate data files |
| **Data** | `organize_data.py` | Reorganizes raw and processed data directories |
| **Deploy** | `flash_firmware.sh` | Flashes firmware to ESP32 devices |
| **Deploy** | `deploy_model.py` | Deploys trained ML models to target environments |
| **Utils** | `check_system.py` | Verifies system requirements and hardware status |
| **Utils** | `generate_report.py` | Creates automated project status reports |
| **Utils** | `batch_process.py` | Executes batch processing tasks over datasets |

## 🛠️ Troubleshooting

| Issue | Cause | Solution |
|---|---|---|
| **Permission Denied (.sh scripts)** | Missing executable flags on Linux/Mac | Run `chmod +x scripts/**/*.sh` before executing |
| **COM Port Not Found** | Incorrect port or driver issue during flashing | Verify the port in Device Manager/lsusb and update the `--port` argument |
| **Import Errors in Python Scripts** | Environment not activated or missing dependencies | Ensure your virtual environment is active and run `install_requirements.py` |
