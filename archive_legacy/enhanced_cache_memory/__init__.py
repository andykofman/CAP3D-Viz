"""
CAP3D Enhanced Parser - Optimized for Large Files

This module provides enhanced parsing and visualization capabilities
for large CAP3D files that may be too memory-intensive for the standard parser.

Refactored into modular components for better scalability and maintainability.
"""

# Import all the main classes and functions for backward compatibility
from .data_models import (
    Block, 
    CachedMesh, 
    Layer, 
    PolyElement, 
    Window, 
    Task, 
    ParsedCap3DData
)

from .parser import (
    ParserState,
    StreamingCap3DParser
)

from .visualizer import (
    OptimizedCap3DVisualizer
)

from .utils import (
    load_and_visualize,
    quick_preview,
    create_interactive_dashboard
)

# Export all the main components
__all__ = [
    # Data models
    'Block',
    'CachedMesh', 
    'Layer',
    'PolyElement',
    'Window',
    'Task',
    'ParsedCap3DData',
    
    # Parser components
    'ParserState',
    'StreamingCap3DParser',
    
    # Visualizer
    'OptimizedCap3DVisualizer',
    
    # Utility functions
    'load_and_visualize',
    'quick_preview',
    'create_interactive_dashboard'
]

# Version info
__version__ = "2.0.0"
__author__ = "CAP3D Enhanced Team" 