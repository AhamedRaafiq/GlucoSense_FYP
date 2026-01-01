# Contributing to Non-Invasive Diabetes Prediction Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## 🎯 Project Goals

This is an academic research project (Final Year Project) focused on developing a non-invasive diabetes prediction system using PPG signals. Contributions should align with this goal.

## 🤝 How to Contribute

### Reporting Bugs

If you find a bug, please open an issue with:
- Clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, ESP-IDF version)
- Relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Check if the enhancement has already been suggested
- Provide a clear use case
- Explain why this would be useful
- Consider implementation complexity

### Code Contributions

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
4. **Test your changes** thoroughly
5. **Commit with clear messages**:
   ```bash
   git commit -m "Add feature: brief description"
   ```
6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
7. **Open a Pull Request**

## 📝 Coding Standards

### Python Code
- Follow **PEP 8** style guide
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep functions focused and modular
- Add comments for complex logic

Example:
```python
def extract_features(signal, sampling_rate=400):
    """
    Extract time-domain features from PPG signal.
    
    Args:
        signal (np.array): PPG signal array
        sampling_rate (int): Sampling rate in Hz
        
    Returns:
        dict: Dictionary of extracted features
    """
    # Implementation
    pass
```

### C/C++ Code (Firmware)
- Follow ESP-IDF coding style
- Use descriptive variable names
- Add comments for hardware-specific code
- Document configuration parameters

### Jupyter Notebooks
- Clear markdown explanations between code cells
- Remove unnecessary outputs before committing
- Use meaningful cell execution order
- Include visualization outputs

## 🗂️ File Organization

- Place firmware code in `01_Firmware_ESP32/`
- Python scripts in `02_Python_Data_Logger/`
- Notebooks in `03_Python_Signal_Processing_Pipeline/`
- Do NOT commit large data files (use `.gitignore`)
- Update relevant README files

## ✅ Testing

- Test your code before submitting
- Add unit tests for new functions (in `08_Tests/`)
- Verify firmware builds successfully
- Check that notebooks run without errors

## 📊 Data Guidelines

- **Do NOT commit** large CSV files or datasets
- Use sample data (< 1MB) for testing
- Document data collection procedures
- Respect privacy and ethical guidelines

## 🔄 Pull Request Process

1. Ensure your code follows the style guidelines
2. Update documentation (README, docstrings)
3. Add tests if applicable
4. Ensure all tests pass
5. Update CHANGELOG.md if significant changes
6. Request review from maintainers

### PR Title Format
- `feat: Add new feature`
- `fix: Fix bug description`
- `docs: Update documentation`
- `refactor: Code refactoring`
- `test: Add tests`

## 🚫 What NOT to Contribute

- Unrelated features outside project scope
- Code without proper testing
- Large binary files or datasets
- Plagiarized code or content
- Breaking changes without discussion

## 📧 Questions?

If you have questions about contributing:
- Open a discussion issue
- Check existing documentation
- Contact project maintainers

## 🙏 Recognition

Contributors will be acknowledged in:
- Project README
- Research paper acknowledgments (if applicable)
- Git commit history

## 📜 Code of Conduct

- Be respectful and professional
- Provide constructive feedback
- Focus on the project goals
- Help others learn and grow

---

Thank you for contributing to advancing non-invasive diabetes detection research!
