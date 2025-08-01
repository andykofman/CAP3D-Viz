"""
Utility Functions for CAP3D Enhanced Parser

This module contains convenience functions and utility functions
for the enhanced cache memory system.
"""

import plotly.graph_objects as go
from typing import Optional

from .visualizer import OptimizedCap3DVisualizer


def load_and_visualize(file_path: str, 
                      max_blocks: int = 100000,  # Increased default with batched rendering
                      z_slice: Optional[float] = None,
                      export_html: bool = True,
                      use_cache: bool = True,
                      use_batched: bool = True) -> go.Figure:
    """One-shot function to load and visualize a cap3d file"""
    visualizer = OptimizedCap3DVisualizer(max_blocks_display=max_blocks)
    visualizer.load_data(file_path)
    
    fig = visualizer.create_optimized_visualization(
        z_slice=z_slice,
        max_blocks=max_blocks,
        use_lod=True,
        use_cache=use_cache,
        use_batched_rendering=use_batched
    )
    
    if export_html:
        visualizer.export_to_html(fig, f"{file_path.replace('.cap3d', '_visualization.html')}")
    
    return fig


def quick_preview(file_path: str, max_blocks: int = 50000) -> go.Figure:
    """Quick preview - now handles much larger files with batched rendering"""
    return load_and_visualize(file_path, max_blocks=max_blocks, export_html=False)


def create_interactive_dashboard(file_path: str, max_blocks: int = 100000) -> go.Figure:
    """Create an interactive dashboard with real-time controls"""
    visualizer = OptimizedCap3DVisualizer(max_blocks_display=max_blocks)
    visualizer.load_data(file_path)
    return visualizer.create_interactive_dashboard(use_batched=True) 