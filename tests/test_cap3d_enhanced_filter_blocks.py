"""
Tests for OptimizedCap3DVisualizer.filter_blocks in src/cap3d_enhanced.py.
"""
import numpy as np
from src.cap3d_enhanced import Block, OptimizedCap3DVisualizer

def make_block(name, typ, base, v1, v2, hvec, diel=None):
    return Block(name, typ, "parent", base, v1, v2, hvec, diel)

def test_filter_blocks_type_and_z():
    vis = OptimizedCap3DVisualizer()
    vis.blocks = [
        make_block("m1", "medium", [0,0,0], [1,0,0], [0,1,0], [0,0,1]),
        make_block("c1", "conductor", [0,0,2], [1,0,0], [0,1,0], [0,0,1])
    ]
    # Only show mediums
    filtered = vis.filter_blocks(show_mediums=True, show_conductors=False)
    print(f"Filtered (mediums only): {[b.name for b in filtered]}")
    assert all(b.type == "medium" for b in filtered)
    # Only show conductors
    filtered = vis.filter_blocks(show_mediums=False, show_conductors=True)
    print(f"Filtered (conductors only): {[b.name for b in filtered]}")
    assert all(b.type == "conductor" for b in filtered)
    # Z filter: only blocks with max z >= 2
    filtered = vis.filter_blocks(z_min=2)
    print(f"Filtered (z_min=2): {[b.name for b in filtered]}")
    assert all(b.bounds[1][2] >= 2 for b in filtered)

def test_filter_blocks_spatial_and_volume():
    vis = OptimizedCap3DVisualizer()
    vis.blocks = [
        make_block("b1", "medium", [0,0,0], [1,0,0], [0,1,0], [0,0,1]),
        make_block("b2", "medium", [10,10,10], [1,0,0], [0,1,0], [0,0,1])
    ]
    
    min_xyz = np.array([0,0,0])
    max_xyz = np.array([5,5,5])
    filtered = vis.filter_blocks(spatial_filter=(min_xyz, max_xyz))
    print(f"Filtered (spatial): {[b.name for b in filtered]}")
    assert all((b.bounds[0] >= min_xyz).all() and (b.bounds[1] <= max_xyz).all() for b in filtered)
    # Only blocks with volume >= 1 (all blocks here are cubes of volume 1)
    filtered = vis.filter_blocks(volume_threshold=1)
    print(f"Filtered (volume >= 1): {[b.name for b in filtered]}")
    assert all(b.volume >= 1 for b in filtered)