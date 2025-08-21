"""
Tests for the CAP3D parser functionality
"""

import pytest
import numpy as np
from cap3d_viz import StreamingCap3DParser, Block


class TestStreamingParser:
    """Test cases for the streaming CAP3D parser"""
    
    def test_parser_imports(self):
        """Test that parser imports correctly"""
        from cap3d_viz import StreamingCap3DParser
        assert StreamingCap3DParser is not None
    
    def test_block_creation(self):
        """Test basic block creation"""
        block = Block(
            name="test_block",
            type="conductor", 
            parent_name="test_medium",
            base=[0, 0, 0],
            v1=[1, 0, 0],
            v2=[0, 1, 0], 
            hvec=[0, 0, 1]
        )
        
        assert block.name == "test_block"
        assert block.type == "conductor"
        assert block.parent_name == "test_medium"
        
        # Test vertex generation
        vertices = block.vertices
        assert vertices.shape == (8, 3)
        assert isinstance(vertices, np.ndarray)
    
    def test_block_bounds(self):
        """Test block bounding box calculation"""
        block = Block(
            name="test_block",
            type="conductor",
            parent_name="test_medium", 
            base=[1, 1, 1],
            v1=[2, 0, 0],
            v2=[0, 2, 0],
            hvec=[0, 0, 2]
        )
        
        min_bounds, max_bounds = block.bounds
        
        assert np.array_equal(min_bounds, [1, 1, 1])
        assert np.array_equal(max_bounds, [3, 3, 3])


if __name__ == "__main__":
    pytest.main([__file__])
