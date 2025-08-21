# Performance Guide

## Benchmarks

CAP3D-Viz is optimized for high-performance parsing and visualization of large CAP3D files.

### Parser Performance

| Dataset Size | Parse Time | Memory Usage | Blocks/Second |
|-------------|------------|--------------|---------------|
| 1k blocks   | 0.1s       | 2MB          | 10,000        |
| 10k blocks  | 1.0s       | 7.25MB       | 9,882         |
| 50k blocks  | 5.2s       | 35MB         | 9,615         |

### Visualization Performance

| Feature | Performance | Notes |
|---------|-------------|-------|
| Interactive Filtering | <500ms | For 10k blocks |
| Z-slice Operations | 1.06s | For 10k blocks |
| Batched Rendering | 50k+ blocks | Smooth interaction |
| First Frame Render | 14.9-20.8s | For 10k blocks |

## Optimization Strategies

### 1. State-Machine Parser

The parser uses a state-machine architecture that provides:
- 70-80% reduction in condition checking
- Context-aware parsing (only relevant logic executed)
- Direct dispatch tables instead of conditional chains
- Pre-compiled patterns for common operations

### 2. Memory Efficiency

For large files:
```python
from cap3d_viz import StreamingCap3DParser

# Memory-efficient streaming
parser = StreamingCap3DParser("large_file.cap3d")
data = parser.parse_complete()  # <8MB for 10k blocks
```

### 3. Visualization Optimization

For large datasets:
```python
from cap3d_viz import OptimizedCap3DVisualizer

visualizer = OptimizedCap3DVisualizer(
    max_blocks_display=50000  # Automatic LOD
)

# Use batched rendering
fig = visualizer.create_optimized_visualization(
    use_batched_rendering=True,  # Groups blocks efficiently
    show_mediums=False,          # Reduce complexity
    opacity_conductors=0.8       # Improve rendering speed
)
```

### 4. Level of Detail (LOD)

The LOD system automatically:
- Prioritizes larger, more important blocks
- Reduces complexity for distant objects
- Maintains interactive performance

### 5. Filtering for Performance

Quick filtering operations:
```python
# Fast conductor-only view
filtered = visualizer.filter_blocks(
    show_mediums=False,
    show_conductors=True,
    volume_threshold=0.001  # Hide small blocks
)

# Z-slice for focused analysis  
z_slice = visualizer.filter_blocks(
    z_min=4.0, z_max=6.0
)
```

## Performance Tips

### Do's
- ✅ Use `StreamingCap3DParser` for files >1MB
- ✅ Enable batched rendering for >1k blocks
- ✅ Use volume thresholds to hide insignificant blocks
- ✅ Filter by type (conductor/medium) when appropriate
- ✅ Use z-slicing for focused analysis

### Don'ts
- ❌ Load entire large files into memory at once
- ❌ Create individual traces for thousands of blocks
- ❌ Render transparent objects unnecessarily
- ❌ Skip the LOD system for large datasets

## Benchmarking Your Data

Run performance tests on your data:

```python
import time
from cap3d_viz import StreamingCap3DParser

# Benchmark parsing
start = time.time()
parser = StreamingCap3DParser("your_file.cap3d")
data = parser.parse_complete()
parse_time = time.time() - start

print(f"Parsed {len(data.blocks)} blocks in {parse_time:.2f}s")
print(f"Performance: {len(data.blocks)/parse_time:.0f} blocks/second")
```

## Hardware Recommendations

### Minimum Requirements
- RAM: 4GB
- CPU: Dual-core 2GHz
- Python: 3.8+

### Recommended for Large Datasets
- RAM: 16GB+
- CPU: Quad-core 3GHz+
- SSD storage for fast file I/O
- Modern GPU for Plotly WebGL rendering
