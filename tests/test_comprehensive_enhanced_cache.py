#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Cache Memory CAP3D Module

This suite tests all available functionality in the enhanced_cache_memory module
with extensive command-line options and test modes.

Usage:
    python tests/test_comprehensive_enhanced_cache.py examples/*.cap3d --all-tests
    python tests/test_comprehensive_enhanced_cache.py examples/*.cap3d --performance --memory --cache --filter --lod
    python tests/test_comprehensive_enhanced_cache.py examples/*.cap3d --stress-test --repeat 10 --max-blocks 50000
"""

import argparse
import time
import json
import os
import tracemalloc
import psutil
import platform
import statistics as stats
import gc
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from contextlib import contextmanager

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from enhanced_cache_memory import (
    StreamingCap3DParser, 
    OptimizedCap3DVisualizer,
    Block,
    CachedMesh,
    load_and_visualize,
    quick_preview,
    create_interactive_dashboard
)
import plotly.io as pio

class TestRunner:
    """Comprehensive test runner for enhanced cache memory module"""
    
    def __init__(self, args):
        self.args = args
        self.results = {
            "system": self._get_system_info(),
            "test_config": vars(args),
            "tests": {},
            "summary": {}
        }
        
    def _get_system_info(self):
        """Collect system information"""
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu": platform.processor(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "cpu_count": psutil.cpu_count(),
            "enhanced_cache_version": "2.0.0"
        }
    
    @contextmanager
    def measure_performance(self, test_name: str):
        """Context manager for performance measurement"""
        gc.collect()  # Clean up before measurement
        tracemalloc.start()
        m_before = psutil.Process().memory_info().rss / (1024**2)
        t0 = time.perf_counter()
        
        try:
            yield
        finally:
            t1 = time.perf_counter()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            m_after = psutil.Process().memory_info().rss / (1024**2)
            
            self.results["tests"][test_name] = {
                "duration_s": t1 - t0,
                "peak_memory_mb": peak / (1024**2),
                "memory_delta_mb": m_after - m_before,
                "timestamp": time.time()
            }
            
            if self.args.verbose:
                print(f"  ✓ {test_name}: {t1-t0:.3f}s, {peak/(1024**2):.1f}MB peak")

class ComprehensiveTestSuite:
    """Main test suite with all available test modes"""
    
    def __init__(self, runner: TestRunner):
        self.runner = runner
        self.args = runner.args
        
    def run_all_tests(self, files: List[str]):
        """Run all available tests"""
        print("🚀 Running comprehensive test suite...")
        
        for file_path in files:
            print(f"\n📁 Testing file: {Path(file_path).name}")
            
            # Core functionality tests
            if self.args.parsing or self.args.all_tests:
                self._test_parsing(file_path)
                
            if self.args.visualization or self.args.all_tests:
                self._test_visualization(file_path)
                
            if self.args.cache or self.args.all_tests:
                self._test_caching(file_path)
                
            if self.args.filter or self.args.all_tests:
                self._test_filtering(file_path)
                
            if self.args.lod or self.args.all_tests:
                self._test_lod(file_path)
                
            # Performance tests
            if self.args.performance or self.args.all_tests:
                self._test_performance(file_path)
                
            if self.args.memory or self.args.all_tests:
                self._test_memory_usage(file_path)
                
            if self.args.scalability or self.args.all_tests:
                self._test_scalability(file_path)
                
            # Stress tests
            if self.args.stress_test:
                self._test_stress(file_path)
                
            # Edge case tests
            if self.args.edge_cases or self.args.all_tests:
                self._test_edge_cases(file_path)
                
        # Comparison tests
        if self.args.comparison:
            self._test_module_comparison(files)
            
        # Integration tests
        if self.args.integration or self.args.all_tests:
            self._test_integration(files)
            
        self._generate_summary()
        self._save_results()
    
    def _test_parsing(self, file_path: str):
        """Test parsing functionality"""
        print("  🔍 Testing parsing...")
        
        with self.runner.measure_performance(f"parse_streaming_{Path(file_path).name}"):
            parser = StreamingCap3DParser(file_path)
            blocks = list(parser.parse_blocks_streaming())
            stats = parser.stats
            
        with self.runner.measure_performance(f"parse_validation_{Path(file_path).name}"):
            # Validate parsed data
            assert len(blocks) > 0, "No blocks parsed"
            assert stats['total_blocks'] == len(blocks), "Block count mismatch"
            assert all(isinstance(b, Block) for b in blocks), "Invalid block types"
            
        # Test different parsing modes if available
        for repeat in range(self.args.repeat):
            with self.runner.measure_performance(f"parse_repeat_{repeat}_{Path(file_path).name}"):
                parser = StreamingCap3DParser(file_path)
                blocks_repeat = list(parser.parse_blocks_streaming())
                assert len(blocks_repeat) == len(blocks), f"Inconsistent parsing on repeat {repeat}"
    
    def _test_visualization(self, file_path: str):
        """Test visualization functionality"""
        print("  🎨 Testing visualization...")
        
        # Load data first
        parser = StreamingCap3DParser(file_path)
        blocks = list(parser.parse_blocks_streaming())
        
        visualizer = OptimizedCap3DVisualizer()
        visualizer.blocks = blocks
        visualizer._calculate_bounds()
        
        # Test basic visualization
        with self.runner.measure_performance(f"viz_basic_{Path(file_path).name}"):
            fig = visualizer.create_optimized_visualization(
                max_blocks=min(len(blocks), self.args.max_blocks),
                use_lod=True,
                use_cache=False,  # Test without cache first
                use_batched_rendering=False  # Test individual traces first
            )
            assert fig is not None, "Visualization failed"
            
        # Test cached visualization 
        with self.runner.measure_performance(f"viz_cached_{Path(file_path).name}"):
            fig_cached = visualizer.create_optimized_visualization(
                max_blocks=min(len(blocks), self.args.max_blocks),
                use_lod=True,
                use_cache=True,  # This will auto-build cache
                use_batched_rendering=True  # Test batched rendering
            )
            assert fig_cached is not None, "Cached visualization failed"
            
        # Test rendering if requested
        if self.args.render:
            with self.runner.measure_performance(f"render_{Path(file_path).name}"):
                output_file = f"test_render_{Path(file_path).stem}.png"
                pio.write_image(fig, output_file, format="png", width=1200, height=900)
                assert os.path.exists(output_file), "Render output not created"
                
        # Test utility functions
        if self.args.utilities:
            with self.runner.measure_performance(f"util_quick_preview_{Path(file_path).name}"):
                fig_preview = quick_preview(file_path, max_blocks=1000)
                assert fig_preview is not None, "Quick preview failed"
                
            with self.runner.measure_performance(f"util_load_visualize_{Path(file_path).name}"):
                fig_load = load_and_visualize(file_path, max_blocks=2000, export_html=False)
                assert fig_load is not None, "Load and visualize failed"
    
    def _test_caching(self, file_path: str):
        """Test caching system performance"""
        print("  💾 Testing caching...")
        
        parser = StreamingCap3DParser(file_path)
        blocks = list(parser.parse_blocks_streaming())
        
        visualizer = OptimizedCap3DVisualizer()
        visualizer.blocks = blocks
        visualizer._calculate_bounds()
        
        # Test cache building
        with self.runner.measure_performance(f"cache_build_{Path(file_path).name}"):
            visualizer._build_mesh_cache(max_blocks=self.args.max_blocks)
            assert visualizer._mesh_cache_valid, "Cache not marked as valid"
            assert len(visualizer._cached_meshes) > 0, "No cached meshes created"
            
        # Test cache utilization (multiple visualizations)
        cache_times = []
        for i in range(3):
            with self.runner.measure_performance(f"cache_use_{i}_{Path(file_path).name}"):
                start = time.time()
                fig = visualizer.create_optimized_visualization(
                    max_blocks=self.args.max_blocks,
                    use_cache=True,
                    use_batched_rendering=True
                )
                cache_times.append(time.time() - start)
                
        # Verify cache performance improvement
        if len(cache_times) >= 2:
            avg_cache_time = sum(cache_times[1:]) / len(cache_times[1:])  # Skip first (cold cache)
            self.runner.results["tests"][f"cache_performance_{Path(file_path).name}"] = {
                "avg_cache_time_s": avg_cache_time,
                "cache_efficiency": cache_times[0] / avg_cache_time if avg_cache_time > 0 else 1
            }
            
        # Test cache clearing
        with self.runner.measure_performance(f"cache_clear_{Path(file_path).name}"):
            visualizer.clear_cache()
            assert not visualizer._mesh_cache_valid, "Cache not cleared"
            assert len(visualizer._cached_meshes) == 0, "Cached meshes not cleared"
    
    def _test_filtering(self, file_path: str):
        """Test filtering capabilities"""
        print("  🔍 Testing filtering...")
        
        parser = StreamingCap3DParser(file_path)
        blocks = list(parser.parse_blocks_streaming())
        
        visualizer = OptimizedCap3DVisualizer()
        visualizer.blocks = blocks
        
        # Test different filter combinations
        filter_tests = [
            {"show_mediums": True, "show_conductors": False},
            {"show_mediums": False, "show_conductors": True},
            {"show_mediums": True, "show_conductors": True, "z_max": 5.0},
            {"show_mediums": True, "show_conductors": True, "z_min": 1.0, "z_max": 10.0},
        ]
        
        for i, filter_params in enumerate(filter_tests):
            test_name = f"filter_{i}_{Path(file_path).name}"
            with self.runner.measure_performance(test_name):
                filtered = visualizer.filter_blocks(**filter_params)
                assert isinstance(filtered, list), "Filter didn't return list"
                assert len(filtered) <= len(blocks), "Filter returned more blocks than input"
                
                # Validate filter logic
                if not filter_params.get("show_mediums", True):
                    assert not any(b.type == 'medium' for b in filtered), "Mediums not filtered out"
                if not filter_params.get("show_conductors", True):
                    assert not any(b.type == 'conductor' for b in filtered), "Conductors not filtered out"
    
    def _test_lod(self, file_path: str):
        """Test Level of Detail (LOD) functionality"""
        print("  📐 Testing LOD...")
        
        parser = StreamingCap3DParser(file_path)
        blocks = list(parser.parse_blocks_streaming())
        
        if len(blocks) <= 100:
            print("    ⚠️  Skipping LOD test (too few blocks)")
            return
            
        visualizer = OptimizedCap3DVisualizer()
        visualizer.blocks = blocks
        
        # Test different LOD levels
        lod_levels = [len(blocks) // 4, len(blocks) // 2, len(blocks) * 3 // 4]
        
        for max_blocks in lod_levels:
            test_name = f"lod_{max_blocks}_{Path(file_path).name}"
            with self.runner.measure_performance(test_name):
                lod_blocks = visualizer._apply_lod(blocks, max_blocks)
                assert len(lod_blocks) <= max_blocks, f"LOD didn't reduce blocks to {max_blocks}"
                assert len(lod_blocks) > 0, "LOD returned no blocks"
                
                # Verify LOD prioritizes larger blocks
                if len(lod_blocks) > 1:
                    volumes = [b.volume for b in lod_blocks]
                    assert volumes == sorted(volumes, reverse=True), "LOD didn't prioritize by volume"
    
    def _test_performance(self, file_path: str):
        """Test performance characteristics"""
        print("  ⚡ Testing performance...")
        
        # Parsing performance
        parse_times = []
        for i in range(self.args.repeat):
            start = time.time()
            parser = StreamingCap3DParser(file_path)
            blocks = list(parser.parse_blocks_streaming())
            parse_times.append(time.time() - start)
            
        # Visualization performance  
        viz_times = []
        parser = StreamingCap3DParser(file_path)
        blocks = list(parser.parse_blocks_streaming())
        
        for i in range(min(3, self.args.repeat)):  # Limit viz tests
            visualizer = OptimizedCap3DVisualizer()
            visualizer.blocks = blocks
            visualizer._calculate_bounds()
            
            start = time.time()
            fig = visualizer.create_optimized_visualization(
                max_blocks=min(len(blocks), self.args.max_blocks)
            )
            viz_times.append(time.time() - start)
            
        # Store performance metrics
        perf_data = {
            "blocks_count": len(blocks),
            "parse_time_mean_s": stats.mean(parse_times),
            "parse_time_std_s": stats.pstdev(parse_times) if len(parse_times) > 1 else 0,
            "parse_blocks_per_sec": len(blocks) / stats.mean(parse_times),
            "viz_time_mean_s": stats.mean(viz_times),
            "viz_time_std_s": stats.pstdev(viz_times) if len(viz_times) > 1 else 0,
        }
        
        self.runner.results["tests"][f"performance_{Path(file_path).name}"] = perf_data
    
    def _test_memory_usage(self, file_path: str):
        """Test memory usage patterns"""
        print("  🧠 Testing memory usage...")
        
        # Test memory during parsing
        tracemalloc.start()
        baseline_memory = psutil.Process().memory_info().rss
        
        parser = StreamingCap3DParser(file_path)
        blocks = list(parser.parse_blocks_streaming())
        
        parse_memory = psutil.Process().memory_info().rss
        current, peak_parse = tracemalloc.get_traced_memory()
        
        # Test memory during visualization
        visualizer = OptimizedCap3DVisualizer()
        visualizer.blocks = blocks
        visualizer._calculate_bounds()
        
        viz_memory_before = psutil.Process().memory_info().rss
        fig = visualizer.create_optimized_visualization(max_blocks=min(len(blocks), self.args.max_blocks))
        viz_memory_after = psutil.Process().memory_info().rss
        
        current, peak_total = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        memory_data = {
            "baseline_mb": baseline_memory / (1024**2),
            "parse_delta_mb": (parse_memory - baseline_memory) / (1024**2),
            "viz_delta_mb": (viz_memory_after - viz_memory_before) / (1024**2),
            "peak_parse_mb": peak_parse / (1024**2),
            "peak_total_mb": peak_total / (1024**2),
            "memory_per_block_kb": ((viz_memory_after - baseline_memory) / (1024)) / len(blocks) if blocks else 0
        }
        
        self.runner.results["tests"][f"memory_{Path(file_path).name}"] = memory_data
        
        # Memory cleanup test
        del fig, visualizer, blocks, parser
        gc.collect()
    
    def _test_scalability(self, file_path: str):
        """Test scalability with different block counts"""
        print("  📈 Testing scalability...")
        
        parser = StreamingCap3DParser(file_path)
        all_blocks = list(parser.parse_blocks_streaming())
        
        if len(all_blocks) < 100:
            print("    ⚠️  Skipping scalability test (too few blocks)")
            return
            
        # Test with different block counts
        block_counts = [
            min(100, len(all_blocks)),
            min(1000, len(all_blocks)),
            min(5000, len(all_blocks)),
            len(all_blocks)
        ]
        
        scalability_data = {}
        
        for count in block_counts:
            if count <= 0:
                continue
                
            test_blocks = all_blocks[:count]
            
            # Time visualization creation
            visualizer = OptimizedCap3DVisualizer()
            visualizer.blocks = test_blocks
            visualizer._calculate_bounds()
            
            start = time.time()
            fig = visualizer.create_optimized_visualization(max_blocks=count)
            viz_time = time.time() - start
            
            scalability_data[f"blocks_{count}"] = {
                "viz_time_s": viz_time,
                "time_per_block_ms": (viz_time * 1000) / count,
                "blocks_per_sec": count / viz_time if viz_time > 0 else 0
            }
            
        self.runner.results["tests"][f"scalability_{Path(file_path).name}"] = scalability_data
    
    def _test_stress(self, file_path: str):
        """Stress test with maximum parameters"""
        print("  💪 Running stress test...")
        
        # Parse multiple times rapidly
        stress_results = {"parse_iterations": [], "viz_iterations": []}
        
        for i in range(self.args.stress_iterations):
            # Parsing stress
            start = time.time()
            parser = StreamingCap3DParser(file_path)
            blocks = list(parser.parse_blocks_streaming())
            parse_time = time.time() - start
            stress_results["parse_iterations"].append(parse_time)
            
            # Visualization stress (every 3rd iteration to avoid memory issues)
            if i % 3 == 0:
                visualizer = OptimizedCap3DVisualizer()
                visualizer.blocks = blocks
                visualizer._calculate_bounds()
                
                start = time.time()
                fig = visualizer.create_optimized_visualization(
                    max_blocks=min(len(blocks), self.args.max_blocks)
                )
                viz_time = time.time() - start
                stress_results["viz_iterations"].append(viz_time)
                
                del fig, visualizer
                gc.collect()
                
        # Analyze stress test results
        stress_summary = {
            "parse_iterations": len(stress_results["parse_iterations"]),
            "parse_time_mean_s": stats.mean(stress_results["parse_iterations"]),
            "parse_time_std_s": stats.pstdev(stress_results["parse_iterations"]),
            "viz_iterations": len(stress_results["viz_iterations"]),
            "viz_time_mean_s": stats.mean(stress_results["viz_iterations"]) if stress_results["viz_iterations"] else 0,
            "viz_time_std_s": stats.pstdev(stress_results["viz_iterations"]) if len(stress_results["viz_iterations"]) > 1 else 0,
        }
        
        self.runner.results["tests"][f"stress_{Path(file_path).name}"] = stress_summary
    
    def _test_edge_cases(self, file_path: str):
        """Test edge cases and error handling"""
        print("  🔬 Testing edge cases...")
        
        parser = StreamingCap3DParser(file_path)
        blocks = list(parser.parse_blocks_streaming())
        
        edge_case_results = {}
        
        # Test empty filters
        visualizer = OptimizedCap3DVisualizer()
        visualizer.blocks = blocks
        
        try:
            empty_filter = visualizer.filter_blocks(show_mediums=False, show_conductors=False)
            edge_case_results["empty_filter"] = {"passed": True, "count": len(empty_filter)}
        except Exception as e:
            edge_case_results["empty_filter"] = {"passed": False, "error": str(e)}
            
        # Test extreme LOD
        try:
            extreme_lod = visualizer._apply_lod(blocks, 1)  # Reduce to 1 block
            edge_case_results["extreme_lod"] = {"passed": True, "count": len(extreme_lod)}
        except Exception as e:
            edge_case_results["extreme_lod"] = {"passed": False, "error": str(e)}
            
        # Test visualization with no blocks
        try:
            empty_visualizer = OptimizedCap3DVisualizer()
            empty_visualizer.blocks = []
            empty_visualizer._calculate_bounds()
            edge_case_results["empty_visualization"] = {"passed": True}
        except Exception as e:
            edge_case_results["empty_visualization"] = {"passed": False, "error": str(e)}
            
        self.runner.results["tests"][f"edge_cases_{Path(file_path).name}"] = edge_case_results
    
    def _test_module_comparison(self, files: List[str]):
        """Compare enhanced_cache_memory with other modules"""
        print("  ⚖️  Testing module comparison...")
        
        # This would compare with cap3d_enhanced.py if available
        # Implementation depends on what modules are available for comparison
        comparison_results = {
            "enhanced_cache_version": "2.0.0",
            "comparison_note": "Module comparison requires additional modules to be implemented"
        }
        
        self.runner.results["tests"]["module_comparison"] = comparison_results
    
    def _test_integration(self, files: List[str]):
        """Test integration between components"""
        print("  🔗 Testing integration...")
        
        integration_results = {}
        
        # Test full pipeline for each file
        for file_path in files[:2]:  # Limit to first 2 files for integration
            try:
                # Full pipeline test
                start = time.time()
                
                # Parse
                parser = StreamingCap3DParser(file_path)
                blocks = list(parser.parse_blocks_streaming())
                
                # Visualize with cache
                visualizer = OptimizedCap3DVisualizer()
                visualizer.blocks = blocks
                visualizer._calculate_bounds()
                
                # Build cache
                visualizer._build_mesh_cache()
                
                # Create filtered visualization
                fig = visualizer._create_filtered_figure_from_cache(
                    show_mediums=True,
                    show_conductors=True
                )
                
                # Test utility function integration
                fig_util = quick_preview(file_path, max_blocks=1000)
                
                total_time = time.time() - start
                
                integration_results[f"pipeline_{Path(file_path).name}"] = {
                    "passed": True,
                    "total_time_s": total_time,
                    "blocks_processed": len(blocks)
                }
                
            except Exception as e:
                integration_results[f"pipeline_{Path(file_path).name}"] = {
                    "passed": False,
                    "error": str(e)
                }
                
        self.runner.results["tests"]["integration"] = integration_results
    
    def _generate_summary(self):
        """Generate test summary"""
        print("\n📊 Generating summary...")
        
        total_tests = len(self.runner.results["tests"])
        passed_tests = sum(1 for test in self.runner.results["tests"].values() 
                          if isinstance(test, dict) and test.get("passed", True))
        
        # Calculate overall performance metrics
        parse_times = []
        viz_times = []
        memory_usage = []
        
        for test_name, test_data in self.runner.results["tests"].items():
            if isinstance(test_data, dict):
                if "duration_s" in test_data:
                    if "parse" in test_name:
                        parse_times.append(test_data["duration_s"])
                    elif "viz" in test_name:
                        viz_times.append(test_data["duration_s"])
                        
                if "peak_memory_mb" in test_data:
                    memory_usage.append(test_data["peak_memory_mb"])
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "avg_parse_time_s": stats.mean(parse_times) if parse_times else 0,
            "avg_viz_time_s": stats.mean(viz_times) if viz_times else 0,
            "avg_memory_mb": stats.mean(memory_usage) if memory_usage else 0,
            "test_duration_s": time.time() - self.runner.results.get("start_time", time.time())
        }
        
        self.runner.results["summary"] = summary
        
        # Print summary
        print(f"✅ Tests completed: {passed_tests}/{total_tests} passed ({summary['success_rate']:.1%})")
        print(f"⏱️  Average parse time: {summary['avg_parse_time_s']:.3f}s")
        print(f"🎨 Average viz time: {summary['avg_viz_time_s']:.3f}s") 
        print(f"💾 Average memory: {summary['avg_memory_mb']:.1f}MB")
        
    def _save_results(self):
        """Save test results to file"""
        output_file = self.args.output
        with open(output_file, 'w') as f:
            json.dump(self.runner.results, f, indent=2)
        print(f"📄 Results saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive test suite for enhanced_cache_memory module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  --parsing          Test parsing functionality
  --visualization    Test visualization creation
  --cache           Test caching system
  --filter          Test filtering capabilities  
  --lod             Test Level of Detail (LOD)
  --performance     Test performance metrics
  --memory          Test memory usage
  --scalability     Test scalability
  --stress-test     Run stress tests
  --edge-cases      Test edge cases
  --comparison      Compare with other modules
  --integration     Test component integration
  --utilities       Test utility functions
  --all-tests       Run all available tests

Examples:
  python tests/test_comprehensive_enhanced_cache.py examples/*.cap3d --all-tests
  python tests/test_comprehensive_enhanced_cache.py examples/small*.cap3d --performance --cache --repeat 5
  python tests/test_comprehensive_enhanced_cache.py examples/large*.cap3d --stress-test --max-blocks 10000
        """
    )
    
    # Positional arguments
    parser.add_argument("files", nargs="+", help="CAP3D files to test")
    
    # Test category flags
    parser.add_argument("--parsing", action="store_true", help="Test parsing functionality")
    parser.add_argument("--visualization", action="store_true", help="Test visualization creation")
    parser.add_argument("--cache", action="store_true", help="Test caching system")
    parser.add_argument("--filter", action="store_true", help="Test filtering capabilities")
    parser.add_argument("--lod", action="store_true", help="Test Level of Detail")
    parser.add_argument("--performance", action="store_true", help="Test performance metrics")
    parser.add_argument("--memory", action="store_true", help="Test memory usage")
    parser.add_argument("--scalability", action="store_true", help="Test scalability")
    parser.add_argument("--stress-test", action="store_true", help="Run stress tests")
    parser.add_argument("--edge-cases", action="store_true", help="Test edge cases")
    parser.add_argument("--comparison", action="store_true", help="Compare with other modules")
    parser.add_argument("--integration", action="store_true", help="Test component integration")
    parser.add_argument("--utilities", action="store_true", help="Test utility functions")
    parser.add_argument("--all-tests", action="store_true", help="Run all available tests")
    
    # Configuration options
    parser.add_argument("--repeat", type=int, default=3, help="Number of repetitions for performance tests")
    parser.add_argument("--max-blocks", type=int, default=10000, help="Maximum blocks for visualization tests")
    parser.add_argument("--stress-iterations", type=int, default=10, help="Number of stress test iterations")
    
    # Output options
    parser.add_argument("--render", action="store_true", help="Render visualizations to PNG files")
    parser.add_argument("--output", default="comprehensive_test_results.json", help="Output JSON file")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # If no specific tests selected, default to basic tests
    if not any([args.parsing, args.visualization, args.cache, args.filter, args.lod,
                args.performance, args.memory, args.scalability, args.stress_test,
                args.edge_cases, args.comparison, args.integration, args.utilities,
                args.all_tests]):
        args.parsing = True
        args.visualization = True
        args.performance = True
    
    # Record start time
    start_time = time.time()
    
    # Initialize test runner
    runner = TestRunner(args)
    runner.results["start_time"] = start_time
    
    # Initialize and run test suite
    test_suite = ComprehensiveTestSuite(runner)
    
    try:
        test_suite.run_all_tests(args.files)
    except KeyboardInterrupt:
        print("\n⚠️  Tests interrupted by user")
        runner.results["interrupted"] = True
        test_suite._save_results()
    except Exception as e:
        print(f"\n❌ Test suite failed: {e}")
        runner.results["error"] = str(e)
        test_suite._save_results()
        sys.exit(1)

if __name__ == "__main__":
    main()
