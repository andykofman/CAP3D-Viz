# Contributing to cap3d_view

Thank you for your interest in contributing to cap3d_view! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Pull Request Process](#pull-request-process)
- [Code Style Guidelines](#code-style-guidelines)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Suggesting Enhancements](#suggesting-enhancements)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code.

## How Can I Contribute?

### Reporting Bugs

- Use the GitHub issue tracker
- Include detailed steps to reproduce the bug
- Provide system information and error messages
- Check if the bug has already been reported

### Suggesting Enhancements

- Use the GitHub issue tracker
- Clearly describe the enhancement
- Explain why this enhancement would be useful
- Include mockups or examples if applicable

### Pull Requests

- Fork the repository
- Create a feature branch
- Make your changes
- Add tests for new functionality
- Ensure all tests pass
- Update documentation as needed
- Submit a pull request

## Development Setup

1. **Fork and clone the repository**

   ```bash
   git clone https://github.com/andykofman/RWCap_view
   cd cap3d_view
   ```
2. **Set up a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
4. **Install development dependencies**

   ```bash
   pip install pytest pytest-cov black flake8 mypy
   ```

## Pull Request Process

1. **Create a feature branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```
2. **Make your changes**

   - Follow the code style guidelines
   - Add tests for new functionality
   - Update documentation
3. **Run tests**

   ```bash
   pytest
   ```
4. **Check code quality**

   ```bash
   black src/ tests/
   flake8 src/ tests/
   mypy src/
   ```
5. **Commit your changes**

   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```
6. **Push and create a pull request**

   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints for function parameters and return values
- Keep functions small and focused
- Use descriptive variable and function names
- Add docstrings for all public functions and classes

### Example Code Style

```python
from typing import List, Optional, Tuple
import numpy as np


def process_3d_data(
    data: np.ndarray, 
    threshold: float = 0.5
) -> Tuple[np.ndarray, bool]:
    """
    Process 3D data with optional thresholding.
  
    Args:
        data: Input 3D array
        threshold: Threshold value for processing
      
    Returns:
        Tuple of processed data and success flag
    """
    if data.ndim != 3:
        raise ValueError("Data must be 3-dimensional")
  
    processed = data.copy()
    success = True
  
    try:
        processed[processed < threshold] = 0
    except Exception as e:
        success = False
        print(f"Processing failed: {e}")
  
    return processed, success
```

## Testing Guidelines

- Write tests for all new functionality
- Use pytest for testing
- Aim for high test coverage
- Include both unit tests and integration tests
- Test edge cases and error conditions

### Example Test

```python
import pytest
import numpy as np
from src.processor import process_3d_data


def test_process_3d_data_success():
    """Test successful 3D data processing."""
    data = np.random.rand(10, 10, 10)
    threshold = 0.5
  
    result, success = process_3d_data(data, threshold)
  
    assert success is True
    assert result.shape == data.shape
    assert np.all(result >= 0)


def test_process_3d_data_invalid_input():
    """Test processing with invalid input."""
    data = np.random.rand(10, 10)  # 2D instead of 3D
  
    with pytest.raises(ValueError, match="Data must be 3-dimensional"):
        process_3d_data(data)
```

## Documentation Guidelines

- Update README.md for user-facing changes
- Add docstrings to all public functions and classes
- Include examples in docstrings
- Update any relevant documentation in the `docs/` directory

## Commit Message Format

Use conventional commit format:

- `feat:` for new features
- `fix:` for bug fixes
- `docs:` for documentation changes
- `style:` for formatting changes
- `refactor:` for code refactoring
- `test:` for adding or updating tests
- `chore:` for maintenance tasks

Example: `feat: add 3D data visualization component`

## Getting Help

If you need help with contributing:

1. Check existing issues and pull requests
2. Read the documentation in the `docs/` directory
3. Open an issue for questions or discussions
4. Join our community discussions

## License

By contributing to cap3d_view, you agree that your contributions will be licensed under the same license as the project.

Thank you for contributing to cap3d_view! 🚀
