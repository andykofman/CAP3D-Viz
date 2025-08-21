"""
Tests for src/cap3d_plotly.py: Plotly-based visualization and dashboard logic.
Covers figure creation, dashboard structure, performance, and memory usage.
"""
import pytest
import plotly.graph_objects as go
import tracemalloc
import time
import tempfile
import os
from src.cap3d_plotly import draw_components_plotly, create_interactive_dashboard

def test_draw_components_plotly_basic():
    """Test draw_components_plotly: returns a Plotly Figure with traces for given mediums and conductors."""
    mediums = [
        {'name': 'm1', 'block_name': 'b1', 'base': (0,0,0), 'v1': (1,0,0), 'v2': (0,1,0), 'hvec': (0,0,1), 'type': 'medium', 'diel': 2.5}
    ]
    conductors = [
        {'name': 'c1', 'block_name': 'b2', 'base': (1,1,1), 'v1': (1,0,0), 'v2': (0,1,0), 'hvec': (0,0,1), 'type': 'conductor', 'diel': None}
    ]
    fig = draw_components_plotly(mediums, conductors)
    print(f"draw_components_plotly returned type: {type(fig)} with {len(fig.data)} traces")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0

def test_create_interactive_dashboard_structure():
    """Test create_interactive_dashboard: returns a dict of Plotly Figures for different views."""
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
        figs, z_range, n_med, n_cond = create_interactive_dashboard(tmp.name)
    os.unlink(tmp.name)
    print(f"Dashboard keys: {figs.keys()}, n_med: {n_med}, n_cond: {n_cond}")
    assert set(figs.keys()) == {'full', 'mediums', 'conductors', 'sliced'}
    for fig in figs.values():
        print(f"Dashboard view type: {type(fig)}")
        assert isinstance(fig, go.Figure)
    assert n_med == 1
    assert n_cond == 1

@pytest.mark.slow
def test_dashboard_performance():
    """Test dashboard creation: completes within 2 seconds for a small file."""
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
        figs, z_range, n_med, n_cond = create_interactive_dashboard(tmp.name)
        elapsed = time.time() - start
    os.unlink(tmp.name)
    print(f"Dashboard creation time: {elapsed:.4f} seconds")
    assert elapsed < 2.0  # Should parse and build dashboard quickly

@pytest.mark.slow
def test_dashboard_memory_usage():
    """Test dashboard creation: memory usage stays under 1MB for a small file."""
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
        figs, z_range, n_med, n_cond = create_interactive_dashboard(tmp.name)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    os.unlink(tmp.name)
    print(f"Dashboard memory usage: current={current} bytes, peak={peak} bytes")
    assert peak < 1 * 1024 * 1024 