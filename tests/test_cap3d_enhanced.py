"""
Tests for src/cap3d_enhanced.py: Block and StreamingCap3DParser classes.
Covers block geometry, parsing, and memory/performance aspects.
"""
import os
import tempfile
import numpy as np
import pytest

from src.cap3d_enhanced import Block, StreamingCap3DParser

def test_block_properties():
    """Test Block class: vertices, bounds, volume, and center calculations for a unit cube."""
    base = [0, 0, 0]
    v1 = [1, 0, 0]
    v2 = [0, 1, 0]
    hvec = [0, 0, 1]
    block = Block("test_block", "medium", "parent", base, v1, v2, hvec, diel=2.5)
    # Vertices
    expected_vertices = np.array([
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ], dtype=np.float32)
    np.testing.assert_allclose(block.vertices, expected_vertices)
    # Bounds
    min_bound, max_bound = block.bounds
    np.testing.assert_allclose(min_bound, [0,0,0])
    np.testing.assert_allclose(max_bound, [1,1,1])
    # Volume (center of box)
    np.testing.assert_allclose(block.volume, [0.5,0.5,0.5])
    # Center (should be 1.0 for unit cube)
    assert np.isclose(block.center, 1.0)

def test_streamingcap3dparser_parses_blocks():
    """Test StreamingCap3DParser: parses a minimal cap3d file and yields correct Block objects."""
    cap3d_content = """
<medium>
name test_medium
 diel 2.5
 <block>
  name block1
  basepoint(0,0,0)
  v1(1,0,0)
  v2(0,1,0)
  hvector(0,0,1)
 </block>
</medium>
"""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(cap3d_content)
        tmp.flush()
        parser = StreamingCap3DParser(tmp.name)
        blocks = list(parser.parse_blocks_straming())
    os.unlink(tmp.name)
    assert len(blocks) == 1
    block = blocks[0]
    assert block.name == "block1"
    assert block.type == "medium"
    assert block.parent_name == "test_medium"
    np.testing.assert_allclose(block.base, [0,0,0])
    np.testing.assert_allclose(block.v1, [1,0,0])
    np.testing.assert_allclose(block.v2, [0,1,0])
    np.testing.assert_allclose(block.hvec, [0,0,1])
    assert block.diel == 2.5 