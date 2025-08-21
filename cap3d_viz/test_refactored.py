"""
Test script for the refactored enhanced_cache_memory module

This script verifies that all functionality from the original module
is preserved in the new modular structure.
"""

import time
from typing import List

from cap3d_viz import (
    # Data models
    Block, CachedMesh, Layer, PolyElement, Window, Task, ParsedCap3DData,
    
    # Parser components
    ParserState, StreamingCap3DParser,
    
    # Visualizer
    OptimizedCap3DVisualizer,
    
    # Utility functions
    load_and_visualize, quick_preview, create_interactive_dashboard
)


def test_data_models() -> bool:
    """Test that all data models work correctly"""
    print("Testing data models...")
    
    # Test Block
    block = Block(
        name="test_block",
        type="medium",
        parent_name="test_parent",
        base=[0, 0, 0],
        v1=[1, 0, 0],
        v2=[0, 1, 0],
        hvec=[0, 0, 1],
        diel=1.0
    )
    
    assert block.name == "test_block"
    assert block.type == "medium"
    assert block.volume > 0
    assert len(block.vertices) == 8
    assert len(block.bounds) == 2
    
    # Test CachedMesh
    cached_mesh = CachedMesh(
        x=[0, 1, 1, 0],
        y=[0, 0, 1, 1],
        z=[0, 0, 0, 0],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        block_type="medium",
        block_index=0,
        center=[0.5, 0.5, 0],
        volume=1.0,
        bounds=([0, 0, 0], [1, 1, 0])
    )
    
    assert cached_mesh.block_type == "medium"
    assert len(cached_mesh.x) == 4
    
    # Test other data models
    layer = Layer(name="test_layer", type="interconnect")
    assert layer.name == "test_layer"
    
    window = Window(name="test_window", v1=[0, 0, 0], v2=[1, 1, 1])
    assert window.name == "test_window"
    
    task = Task(capacitance_targets=["conductor1", "conductor2"])
    assert len(task.capacitance_targets) == 2
    
    print("✓ All data models working correctly")
    return True


def test_parser() -> bool:
    """Test that the parser works correctly"""
    print("Testing parser...")
    
    # Test ParserState
    state = ParserState()
    assert state.current_section is None
    assert not state.in_block
    assert not state.in_poly
    
    # Test StreamingCap3DParser initialization
    # Note: We can't test actual parsing without a real file, but we can test initialization
    parser = StreamingCap3DParser("dummy_file.cap3d")
    assert parser.file_path == "dummy_file.cap3d"
    assert parser.stats['total_blocks'] == 0
    
    print("✓ Parser components working correctly")
    return True


def test_visualizer() -> bool:
    """Test that the visualizer works correctly"""
    print("Testing visualizer...")
    
    # Test OptimizedCap3DVisualizer initialization
    visualizer = OptimizedCap3DVisualizer(max_blocks_display=10000)
    assert visualizer.max_blocks_display == 10000
    assert len(visualizer.blocks) == 0
    assert len(visualizer.medium_colors) > 0
    assert len(visualizer.conductor_colors) > 0
    
    # Test filtering
    filtered_blocks = visualizer.filter_blocks(
        show_mediums=True,
        show_conductors=True
    )
    assert isinstance(filtered_blocks, list)
    
    # Test cache clearing
    visualizer.clear_cache()
    assert len(visualizer._cached_meshes) == 0
    assert not visualizer._mesh_cache_valid
    
    print("✓ Visualizer components working correctly")
    return True


def test_utility_functions() -> bool:
    """Test that utility functions work correctly"""
    print("Testing utility functions...")
    
    # Test function signatures (we can't test actual execution without files)
    # But we can verify they exist and are callable
    assert callable(load_and_visualize)
    assert callable(quick_preview)
    assert callable(create_interactive_dashboard)
    
    print("✓ Utility functions accessible")
    return True


def test_imports() -> bool:
    """Test that all imports work correctly"""
    print("Testing imports...")
    
    # Test that we can import everything from the main module
    try:
        import cap3d_viz  # noqa: F401
        print("✓ All imports working correctly")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    return True


def test_module_structure() -> bool:
    """Test that the module structure is correct"""
    print("Testing module structure...")
    
    # Check that the module has the expected attributes
    import cap3d_viz as ecm
    
    expected_attributes = [
        'Block', 'CachedMesh', 'Layer', 'PolyElement', 'Window', 'Task', 'ParsedCap3DData',
        'ParserState', 'StreamingCap3DParser',
        'OptimizedCap3DVisualizer',
        'load_and_visualize', 'quick_preview', 'create_interactive_dashboard'
    ]
    
    for attr in expected_attributes:
        if hasattr(ecm, attr):
            print(f"✓ {attr} available")
        else:
            print(f"✗ {attr} missing")
            return False
    
    return True


def main() -> bool:
    """Run all tests"""
    print("=== Testing Refactored Enhanced Cache Memory Module ===\n")
    
    tests = [
        test_imports,
        test_module_structure,
        test_data_models,
        test_parser,
        test_visualizer,
        test_utility_functions
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                print(f"✗ {test.__name__} failed")
        except Exception as e:
            print(f"✗ {test.__name__} failed with error: {e}")
    
    print(f"\n=== Test Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All tests passed! The refactored module is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()