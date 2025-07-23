"""
Tests for src/cap3d_matplotlib.py: parsing, geometry, and plotting functions.
Covers window/medium/conductor parsing, block vertex creation, plotting, and performance/memory.
"""
import pytest
import numpy as np
import tracemalloc
import time
import tempfile
import os
from unittest import mock
from src.cap3d_matplotlib import parse_cap3d, create_block_vertices, draw_components

def test_parse_cap3d_window_and_blocks():
    """Test parse_cap3d: parses window, mediums, and conductors from a minimal cap3d file."""
    cap3d_content = """
<window>
v1(0,0,0)
v2(2,2,2)
</window>
<medium>
name m1
 diel 2.5
 <block>
  name b1
  basepoint(0,0,0)
  v1(1,0,0)
  v2(0,1,0)
  hvector(0,0,1)
 </block>
</medium>
<conductor>
name c1
 <block>
  name b2
  basepoint(1,1,1)
  v1(1,0,0)
  v2(0,1,0)
  hvector(0,0,1)
 </block>
</conductor>
"""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(cap3d_content)
        tmp.flush()
        window, mediums, conductors = parse_cap3d(tmp.name)
    os.unlink(tmp.name)
    print(f"Parsed window: {window}, mediums: {len(mediums)}, conductors: {len(conductors)}")
    assert window == (0.0,0.0,0.0,2.0,2.0,2.0)
    assert len(mediums) == 1
    assert len(conductors) == 1
    m = mediums[0]
    c = conductors[0]
    print(f"Medium: {m}, Conductor: {c}")
    assert m['name'] == 'm1'
    assert c['name'] == 'c1'
    assert m['block_name'] == 'b1'
    assert c['block_name'] == 'b2'
    np.testing.assert_allclose(m['base'], (0,0,0))
    np.testing.assert_allclose(c['base'], (1,1,1))

def test_create_block_vertices():
    """Test create_block_vertices: returns correct 8 vertices for a unit cube block."""
    base = (0,0,0)
    v1 = (1,0,0)
    v2 = (0,1,0)
    hvec = (0,0,1)
    verts = create_block_vertices(base, v1, v2, hvec)
    expected = [
        (0,0,0), (1,0,0), (1,1,0), (0,1,0),
        (0,0,1), (1,0,1), (1,1,1), (0,1,1)
    ]
    print(f"Block vertices: {verts}")
    assert verts == expected

def test_draw_components_runs():
    """Test draw_components: runs without error and calls plt.show (mocked)."""
    mediums = [
        {'name': 'm1', 'block_name': 'b1', 'base': (0,0,0), 'v1': (1,0,0), 'v2': (0,1,0), 'hvec': (0,0,1), 'type': 'medium', 'diel': 2.5}
    ]
    conductors = [
        {'name': 'c1', 'block_name': 'b2', 'base': (1,1,1), 'v1': (1,0,0), 'v2': (0,1,0), 'hvec': (0,0,1), 'type': 'conductor', 'diel': None}
    ]
    with mock.patch('matplotlib.pyplot.show') as mock_show:
        draw_components(mediums, conductors)
        print("plt.show was called")
        mock_show.assert_called_once()

@pytest.mark.slow
def test_parse_cap3d_performance():
    """Test parse_cap3d: parsing a small file completes within 2 seconds."""
    cap3d_content = """
<window>
v1(0,0,0)
v2(2,2,2)
</window>
<medium>
name m1
 diel 2.5
 <block>
  name b1
  basepoint(0,0,0)
  v1(1,0,0)
  v2(0,1,0)
  hvector(0,0,1)
 </block>
</medium>
<conductor>
name c1
 <block>
  name b2
  basepoint(1,1,1)
  v1(1,0,0)
  v2(0,1,0)
  hvector(0,0,1)
 </block>
</conductor>
"""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(cap3d_content)
        tmp.flush()
        start = time.time()
        window, mediums, conductors = parse_cap3d(tmp.name)
        elapsed = time.time() - start
    os.unlink(tmp.name)
    print(f"Parsing time: {elapsed:.4f} seconds")
    assert elapsed < 2.0

@pytest.mark.slow
def test_parse_cap3d_memory():
    """Test parse_cap3d: memory usage for parsing a small file stays under 1MB."""
    cap3d_content = """
<window>
v1(0,0,0)
v2(2,2,2)
</window>
<medium>
name m1
 diel 2.5
 <block>
  name b1
  basepoint(0,0,0)
  v1(1,0,0)
  v2(0,1,0)
  hvector(0,0,1)
 </block>
</medium>
<conductor>
name c1
 <block>
  name b2
  basepoint(1,1,1)
  v1(1,0,0)
  v2(0,1,0)
  hvector(0,0,1)
 </block>
</conductor>
"""
    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp:
        tmp.write(cap3d_content)
        tmp.flush()
        tracemalloc.start()
        window, mediums, conductors = parse_cap3d(tmp.name)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    os.unlink(tmp.name)
    print(f"Parsing memory usage: current={current} bytes, peak={peak} bytes")
    assert peak < 1 * 1024 * 1024 