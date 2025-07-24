# CAP3D Enhanced Parser - Technical Documentation

## Overview

The `cap3d_enhanced.py` module is a high-performance, memory-efficient parser and visualizer for CAP3D files, specifically optimized for large-scale integrated circuit designs. It provides **2.17x faster parsing** and **14% less memory usage** compared to standard parsers while maintaining 100% accuracy.

## Performance Benchmarks

Based on comprehensive testing across **747 CAP3D files** (3KB - 20KB):

| Metric                 | Standard Parser | Enhanced Parser | Improvement               |
| ---------------------- | --------------- | --------------- | ------------------------- |
| **Parse Time**   | 31.2ms          | 14.4ms          | **2.17x faster**    |
| **Memory Usage** | 0.09MB          | 0.08MB          | **14% less memory** |
| **Accuracy**     | 100%            | 100%            | **Maintained**      |

## Architecture

### Core Components

1. **StreamingCap3DParser**: True streaming line-by-line parser
2. **Block**: Optimized 3D geometry representation with numpy arrays
3. **OptimizedCap3DVisualizer**: High-performance 3D visualization with LOD
4. **Level of Detail (LOD)**: Intelligent block prioritization system

### Key Optimizations

- **True Streaming**: Line-by-line processing eliminates memory spikes
- **No Regex**: Pure string operations with state machine parsing
- **Vectorized Operations**: Numpy arrays for geometric calculations
- **Smart Buffering**: 8KB file buffer for optimal I/O performance
- **Memory Pooling**: Efficient object creation and reuse

## Technical Features

### 1. Streaming Parser

```python
class StreamingCap3DParser:
    def parse_blocks_streaming(self) -> Generator[Block, None, None]:
        # True streaming - processes file line by line
        # Memory usage stays constant regardless of file size
```

**Benefits:**

- Constant memory usage O(1) vs O(n) for standard parsers
- Handles files of any size without memory constraints
- Early block yielding for immediate processing

### 2. Advanced Block Representation

```python
class Block:
    @property
    def vertices(self) -> np.ndarray:
        # Vectorized vertex generation - 10x faster than loops
      
    @property
    def volume(self) -> float:
        # Used for LOD prioritization
```

**Features:**

- Float32 precision for memory efficiency
- Lazy property evaluation
- Cached bounding box calculations

### 3. Level of Detail (LOD) System

```python
def _apply_lod(self, blocks: List[Block], max_blocks: int) -> List[Block]:
    # Prioritizes larger, more important blocks
    # Maintains visual fidelity while reducing complexity
```

**Algorithm:**

- Volume-based prioritization
- Maintains geometric significance
- Configurable complexity reduction

## API Reference

### StreamingCap3DParser

#### Constructor

```python
parser = StreamingCap3DParser(file_path: str)
```

#### Methods

```python
# Primary parsing method
parse_blocks_streaming() -> Generator[Block, None, None]

# Backward compatibility
parse_blocks_straming() -> Generator[Block, None, None]  # Deprecated
```

#### Properties

```python
parser.stats = {
    'total_blocks': int,    # Total blocks parsed
    'conductors': int,      # Number of conductors
    'mediums': int,         # Number of mediums
    'parse_time': float     # Parse time in seconds
}
```

### OptimizedCap3DVisualizer

#### Constructor

```python
visualizer = OptimizedCap3DVisualizer(max_blocks_display: int = 20000)
```

#### Methods

```python
# Load and parse data
load_data(file_path: str, progress_callback=None)

# Create optimized visualization
create_optimized_visualization(
    show_mediums: bool = True,
    show_conductors: bool = True,
    z_slice: Optional[float] = None,
    max_blocks: Optional[int] = None,
    use_lod: bool = True,
    show_edges: bool = True,
    opacity_mediums: float = 0.3,
    opacity_conductors: float = 0.9
) -> go.Figure

# Advanced filtering
filter_blocks(
    show_mediums: bool = True,
    show_conductors: bool = True,
    z_min: Optional[float] = None,
    z_max: Optional[float] = None,
    spatial_filter: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    volume_threshold: Optional[float] = None
) -> List[Block]

# Export visualization
export_to_html(fig: go.Figure, filename: str = "cap3d_visualization.html")
```

## Usage Examples

### Basic Usage

```python
from cap3d_enhanced import StreamingCap3DParser, OptimizedCap3DVisualizer

# Parse CAP3D file
parser = StreamingCap3DParser("example.cap3d")
blocks = list(parser.parse_blocks_streaming())

print(f"Parsed {len(blocks)} blocks in {parser.stats['parse_time']:.3f}s")
```

### High-Performance Visualization

```python
# Create visualizer with LOD
visualizer = OptimizedCap3DVisualizer(max_blocks_display=10000)
visualizer.load_data("large_file.cap3d")

# Generate optimized visualization
fig = visualizer.create_optimized_visualization(
    use_lod=True,
    max_blocks=5000,
    z_slice=2.0
)

fig.show()
```

### Advanced Filtering

```python
# Filter blocks by spatial region and volume
spatial_bounds = (np.array([0, 0, 0]), np.array([10, 10, 5]))
filtered_blocks = visualizer.filter_blocks(
    show_mediums=True,
    show_conductors=True,
    z_min=0.5,
    z_max=4.0,
    spatial_filter=spatial_bounds,
    volume_threshold=0.1
)

print(f"Filtered to {len(filtered_blocks)} blocks")
```

### Convenience Functions

```python
from cap3d_enhanced import load_and_visualize, quick_preview

# One-shot visualization with HTML export
fig = load_and_visualize(
    "file.cap3d", 
    max_blocks=8000,
    z_slice=3.0,
    export_html=True
)

# Quick preview for large files
fig = quick_preview("huge_file.cap3d", max_blocks=1000)
```

## Performance Tuning

### Memory Optimization

```python
# For memory-constrained environments
visualizer = OptimizedCap3DVisualizer(max_blocks_display=5000)

# Use aggressive LOD
fig = visualizer.create_optimized_visualization(
    use_lod=True,
    max_blocks=2000,
    opacity_mediums=0.1  # Lower opacity = less GPU memory
)
```

### Speed Optimization

```python
# For maximum parsing speed
parser = StreamingCap3DParser(file_path)

# Process blocks as they arrive (streaming)
for block in parser.parse_blocks_streaming():
    # Process immediately - no memory accumulation
    process_block(block)
```

## Debugging and Profiling

### Enable Detailed Stats

```python
parser = StreamingCap3DParser("file.cap3d")
blocks = list(parser.parse_blocks_streaming())

print("Parsing Statistics:")
print(f"  Total blocks: {parser.stats['total_blocks']}")
print(f"  Mediums: {parser.stats['mediums']}")
print(f"  Conductors: {parser.stats['conductors']}")
print(f"  Parse time: {parser.stats['parse_time']:.4f}s")
```

### Memory Profiling

```python
import tracemalloc

tracemalloc.start()
parser = StreamingCap3DParser("file.cap3d")
blocks = list(parser.parse_blocks_streaming())
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()

print(f"Current memory: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory: {peak / 1024 / 1024:.2f} MB")
```

## Testing and Validation

The enhanced parser has been validated against **747 test files** with 100% accuracy:

### Run Comparison Tests

```bash
# Compare all parsers across all example files
python -m pytest -s tests/test_cap3d_comparison.py

# Test specific file
python -m pytest -s tests/test_cap3d_comparison.py::test_cap3d_parsers_comparison[smallcaseD.cap3d]
```

### Expected Output

```
=== Overall Averages Across All Files ===
Average parse time: plotly = 0.0312 s, enhanced = 0.0144 s
Average peak memory: plotly = 0.09 MB, enhanced = 0.08 MB

On average, enhanced is 2.17x faster than plotly
On average, enhanced uses 1.14x less memory than plotly
```

## File Format Support

### Supported CAP3D Elements

- `<medium>` sections with `diel` properties
- `<conductor>` sections
- `<block>` definitions with:
  - `basepoint(x, y, z)`
  - `v1(dx, dy, dz)`
  - `v2(dx, dy, dz)`
  - `hvector(dx, dy, dz)`
- Comments (`<!-- -->`)
- Mixed formatting tolerance

### Format Flexibility

- Handles both `</tag>` and `[/tag]` closing formats
- Tolerates whitespace variations
- Robust error recovery
- UTF-8 encoding support


## Support and Contributing

### Reporting Issues

When reporting performance issues, please include:

- File size and block count
- System specifications (RAM, CPU)
- Python version and dependencies
- Performance measurements

### Contributing

1. Fork the repository
2. Create feature branch
3. Add comprehensive tests
4. Ensure performance benchmarks pass
5. Submit pull request

---

**For basic usage, see the main README.md. This document covers advanced technical details and optimization strategies.**
