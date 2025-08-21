"""
Tests for the CAP3D visualization functionality
"""

import pytest
from cap3d_viz import OptimizedCap3DVisualizer, load_and_visualize


class TestVisualization:
    """Test cases for visualization components"""
    
    def test_visualizer_imports(self):
        """Test that visualizer imports correctly"""
        from cap3d_viz import OptimizedCap3DVisualizer
        assert OptimizedCap3DVisualizer is not None
    
    def test_visualizer_creation(self):
        """Test visualizer object creation"""
        visualizer = OptimizedCap3DVisualizer()
        assert visualizer is not None
        assert hasattr(visualizer, 'load_data')
        assert hasattr(visualizer, 'create_optimized_visualization')
    
    def test_utility_functions(self):
        """Test utility function imports"""
        from cap3d_viz import load_and_visualize, quick_preview
        assert load_and_visualize is not None
        assert quick_preview is not None


if __name__ == "__main__":
    pytest.main([__file__])
