# Enhanced Cache Memory Module

This directory contains the refactored version of the `ehnanced_Cache_memory.py` module, organized into a modular structure for better scalability and maintainability. This modular architecture provides the same high-performance CAP3D parsing and visualization capabilities while improving code organization and future development potential.

## Module Structure

```
enhanced_cache_memory/
├── __init__.py              # Main module entry point
├── data_models.py           # Core data structures and models
├── parser.py               # Parsing logic and state machine
├── visualizer.py           # 3D visualization and caching
├── utils.py                # Convenience functions
├── test_refactored.py      # Test suite
└── README.md               # This file
```

## Components

### 1. Data Models (`data_models.py`)

Contains all the core data structures used throughout the system:

- **Block**: Represents a 3D block in the CAP3D file
- **CachedMesh**: Cached mesh data for efficient rendering
- **Layer**: Layer definition with name and type
- **PolyElement**: Polygonal element with custom geometry
- **Window**: Simulation window/boundary definition
- **Task**: Simulation task definition
- **ParsedCap3DData**: Complete parsed CAP3D data structure

### 2. Parser (`parser.py`)

Contains the parsing logic for CAP3D files:

- **ParserState**: Optimized parser state for state-machine based parsing
- **StreamingCap3DParser**: Memory-efficient streaming parser for large files

### 3. Visualizer (`visualizer.py`)

Contains the visualization logic:

- **OptimizedCap3DVisualizer**: High-performance 3D visualizer with LOD and caching

### 4. Utils (`utils.py`)

Contains convenience functions:

- `load_and_visualize()`: One-shot function to load and visualize a CAP3D file
- `quick_preview()`: Quick preview for large files
- `create_interactive_dashboard()`: Create interactive dashboard with controls

## Usage

### Basic Usage

```python
from src.enhanced_cache_memory import load_and_visualize, quick_preview

# Load and visualize a file
fig = load_and_visualize("your_file.cap3d")

# Quick preview
fig = quick_preview("your_file.cap3d", max_blocks=50000)
```

### Advanced Usage

```python
from src.enhanced_cache_memory import OptimizedCap3DVisualizer, StreamingCap3DParser

# Create visualizer
visualizer = OptimizedCap3DVisualizer(max_blocks_display=100000)

# Load data
visualizer.load_data("your_file.cap3d")

# Create visualization
fig = visualizer.create_optimized_visualization(
    show_mediums=True,
    show_conductors=True,
    z_slice=100.0,
    use_batched_rendering=True
)

# Export to HTML
visualizer.export_to_html(fig, "output.html")
```

### Direct Parser Usage

```python
from src.enhanced_cache_memory import StreamingCap3DParser

# Parse file
parser = StreamingCap3DParser("your_file.cap3d")
parsed_data = parser.parse_complete()

# Access parsed data
print(f"Blocks: {len(parsed_data.blocks)}")
print(f"Poly elements: {len(parsed_data.poly_elements)}")
print(f"Layers: {len(parsed_data.layers)}")
```

## Features

### Performance Optimizations

- **Batched Rendering**: Combines multiple meshes into single traces for better performance
- **Caching System**: Caches mesh data to avoid regeneration
- **Level of Detail (LOD)**: Prioritizes larger, more important blocks
- **Streaming Parser**: Memory-efficient parsing for large files

### Visualization Features

- **3D Interactive Plots**: Using Plotly for interactive 3D visualization
- **Color Cycling**: Automatic color assignment for different block types
- **Filtering**: Filter by block type, Z-slice, spatial bounds, and volume
- **Window Boundaries**: Visualize simulation boundaries
- **Export to HTML**: Save visualizations as interactive HTML files

### Data Support

- **Blocks**: Standard rectangular blocks (medium/conductor)
- **Poly Elements**: Custom polygonal geometry with coordinate definitions
- **Layers**: Layer definitions and types
- **Windows**: Simulation boundaries
- **Tasks**: Capacitance calculation targets

## Testing

Run the test suite to verify the refactored module works correctly:

```bash
cd src/enhanced_cache_memory
python test_refactored.py
```

## Migration from Original Module

The refactored module maintains full backward compatibility. You can replace:

```python
# Old import
from src.ehnanced_Cache_memory import OptimizedCap3DVisualizer

# New import
from src.enhanced_cache_memory import OptimizedCap3DVisualizer
```

All function signatures and behavior remain the same. The original `ehnanced_Cache_memory.py` file is still available for backward compatibility.

## Benefits of Refactoring

1. **Modularity**: Each component has a single responsibility
2. **Maintainability**: Easier to locate and fix issues
3. **Scalability**: New features can be added to specific modules
4. **Testability**: Each module can be tested independently
5. **Reusability**: Components can be imported individually
6. **Documentation**: Better organization of code and documentation
7. **Performance**: Same high-performance capabilities as the original
8. **Backward Compatibility**: Original file still available and functional

## Future Enhancements

The modular structure makes it easy to add new features:

- **New Parser Types**: Add to `parser.py`
- **New Visualization Methods**: Add to `visualizer.py`
- **New Data Models**: Add to `data_models.py`
- **New Utility Functions**: Add to `utils.py`

## Dependencies

- `numpy`: For numerical operations
- `plotly`: For 3D visualization
- `dataclasses`: For data structure definitions
- `typing`: For type hints
- `concurrent.futures`: For parallel processing (optional)

## License

This module is part of the CAP3D Enhanced Parser project and follows the same license terms.
