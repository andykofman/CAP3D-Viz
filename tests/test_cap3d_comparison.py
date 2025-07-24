import os
import sys
# Add src directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import time
import tracemalloc
import pytest

from cap3d_plotly import parse_cap3d
from cap3d_enhanced import StreamingCap3DParser

# Test files: all .cap3d in examples/
EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), '..', 'examples')
TEST_FILES = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith('.cap3d')]

# For collecting overall stats
all_results = []

def run_plotly_parser(filepath):
    start = time.time()
    tracemalloc.start()
    window, mediums, conductors = parse_cap3d(filepath)
    mem_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    elapsed = time.time() - start
    return {
        'time': elapsed,
        'mem_peak': mem_peak,
        'mediums': len(mediums),
        'conductors': len(conductors),
        'total_blocks': len(mediums) + len(conductors),
    }

def run_enhanced_parser(filepath):
    start = time.time()
    tracemalloc.start()
    parser = StreamingCap3DParser(filepath)
    mediums = []
    conductors = []
    for block in parser.parse_blocks_streaming():
        if block.type == 'medium':
            mediums.append(block)
        elif block.type == 'conductor':
            conductors.append(block)
    mem_peak = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    elapsed = time.time() - start
    return {
        'time': elapsed,
        'mem_peak': mem_peak,
        'mediums': len(mediums),
        'conductors': len(conductors),
        'total_blocks': len(mediums) + len(conductors),
    }

@pytest.mark.parametrize('filename', TEST_FILES)
def test_cap3d_parsers_comparison(filename):
    filepath = os.path.join(EXAMPLES_DIR, filename)
    print(f"\n\n=== Comparing parsers for {filename} ===")
    print(f"File size: {os.path.getsize(filepath)/1024:.1f} KB")

    print("\n[cap3d_plotly.py parser]")
    plotly_stats = run_plotly_parser(filepath)
    print(f"Parse time: {plotly_stats['time']:.4f} s")
    print(f"Peak memory: {plotly_stats['mem_peak']/1024/1024:.2f} MB")
    print(f"Mediums: {plotly_stats['mediums']}, Conductors: {plotly_stats['conductors']}, Total blocks: {plotly_stats['total_blocks']}")

    print("\n[cap3d_enhanced.py parser]")
    enhanced_stats = run_enhanced_parser(filepath)
    print(f"Parse time: {enhanced_stats['time']:.4f} s")
    print(f"Peak memory: {enhanced_stats['mem_peak']/1024/1024:.2f} MB")
    print(f"Mediums: {enhanced_stats['mediums']}, Conductors: {enhanced_stats['conductors']}, Total blocks: {enhanced_stats['total_blocks']}")

    # Assert block counts match
    assert plotly_stats['total_blocks'] == enhanced_stats['total_blocks'], "Block count mismatch!"
    assert plotly_stats['mediums'] == enhanced_stats['mediums'], "Medium count mismatch!"
    assert plotly_stats['conductors'] == enhanced_stats['conductors'], "Conductor count mismatch!"

    # Print which parser is faster/more memory efficient
    faster = 'plotly' if plotly_stats['time'] < enhanced_stats['time'] else 'enhanced'
    less_mem = 'plotly' if plotly_stats['mem_peak'] < enhanced_stats['mem_peak'] else 'enhanced'
    print(f"\nFaster parser: {faster}")
    print(f"More memory efficient: {less_mem}")

    # Collect for overall stats
    all_results.append({
        'filename': filename,
        'plotly_time': plotly_stats['time'],
        'enhanced_time': enhanced_stats['time'],
        'plotly_mem': plotly_stats['mem_peak'],
        'enhanced_mem': enhanced_stats['mem_peak'],
    })


def teardown_module(module):
    # Compute and print overall averages and ratios
    if not all_results:
        return
    n = len(all_results)
    plotly_time_avg = sum(r['plotly_time'] for r in all_results) / n
    enhanced_time_avg = sum(r['enhanced_time'] for r in all_results) / n
    plotly_mem_avg = sum(r['plotly_mem'] for r in all_results) / n / 1024 / 1024
    enhanced_mem_avg = sum(r['enhanced_mem'] for r in all_results) / n / 1024 / 1024
    time_ratio = plotly_time_avg / enhanced_time_avg if enhanced_time_avg > 0 else float('inf')
    mem_ratio = plotly_mem_avg / enhanced_mem_avg if enhanced_mem_avg > 0 else float('inf')
    print("\n\n=== Overall Averages Across All Files ===")
    print(f"Average parse time: plotly = {plotly_time_avg:.4f} s, enhanced = {enhanced_time_avg:.4f} s")
    print(f"Average peak memory: plotly = {plotly_mem_avg:.2f} MB, enhanced = {enhanced_mem_avg:.2f} MB")
    print(f"\nOn average, enhanced is {time_ratio:.2f}x faster than plotly")
    print(f"On average, enhanced uses {mem_ratio:.2f}x less memory than plotly") 