# Installation Guide

## System Requirements

- Python ≥3.8
- NumPy ≥1.19.0
- Plotly ≥5.0.0
- Matplotlib ≥3.3.0

## Installation Methods

### From PyPI (Recommended)

```bash
pip install cap3d-viz
```

### Development Installation

For development or to access the latest features:

```bash
git clone https://github.com/your-repo/cap3d-viz.git
cd cap3d-viz
pip install -e .[dev]
```

### Optional Dependencies

For enhanced performance:
```bash
pip install cap3d-viz[performance]
```

For documentation building:
```bash
pip install cap3d-viz[docs]
```

For development tools:
```bash
pip install cap3d-viz[dev]
```

All optional dependencies:
```bash
pip install cap3d-viz[all]
```

## Verification

Test your installation:

```python
import cap3d_viz
print(cap3d_viz.__version__)

# Quick test
from cap3d_viz import OptimizedCap3DVisualizer
visualizer = OptimizedCap3DVisualizer()
print("Installation successful!")
```

## Troubleshooting

### Common Issues

1. **ImportError: No module named 'cap3d_viz'**
   - Ensure you have activated the correct Python environment
   - Reinstall with `pip install --upgrade cap3d-viz`

2. **Plotly visualization not showing**
   - For Jupyter notebooks, ensure you have: `pip install jupyter`
   - For standalone scripts, visualizations open in your default browser

3. **Memory issues with large files**
   - Use the streaming parser: `StreamingCap3DParser`
   - Consider the `quick_preview` function for very large files

### Performance Optimization

For optimal performance with large datasets:

```bash
pip install cap3d-viz[performance]
```

This installs additional packages like NumPy and Numba for accelerated computation.
