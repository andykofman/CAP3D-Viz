#!/usr/bin/env python3
"""
Benchmark CAP3D Enhanced parsing & rendering.
Usage:
    python scripts/benchmark_cap3d.py data/*.cap3d --repeat 5 --render
"""

import argparse, time, json, os, tracemalloc, psutil, platform, statistics as stats
from pathlib import Path
import sys

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ehnanced_Cache_memory import StreamingCap3DParser, OptimizedCap3DVisualizer
import plotly.io as pio

def mem_mb():
    return psutil.Process().memory_info().rss / (1024**2)

def time_block(fn, *args, **kwargs):
    tracemalloc.start()
    m_before = mem_mb()
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, t1 - t0, (peak / (1024**2)), (mem_mb() - m_before)

def parse_cap3d_enhanced(file_path):
    """Parse CAP3D file using enhanced parser"""
    parser = StreamingCap3DParser(str(file_path))
    blocks = list(parser.parse_blocks_streaming())
    return blocks, parser.stats

def build_fig_enhanced(blocks):
    """Build figure using enhanced visualizer"""
    visualizer = OptimizedCap3DVisualizer()
    visualizer.blocks = blocks
    visualizer._calculate_bounds()
    fig = visualizer.create_optimized_visualization(max_blocks=len(blocks), use_lod=False)
    return fig

def apply_filter_enhanced(blocks, mediums=True, conductors=True, z_min=None, z_max=None):
    """Apply filters using enhanced visualizer"""
    visualizer = OptimizedCap3DVisualizer()
    visualizer.blocks = blocks
    filtered = visualizer.filter_blocks(
        show_mediums=mediums,
        show_conductors=conductors,
        z_min=z_min,
        z_max=z_max
    )
    return filtered

def measure_interactions(fig, blocks):
    """Measure interaction latencies with enhanced API"""
    latencies = {}
    ops = {
        "filter_conductors": lambda: apply_filter_enhanced(blocks, mediums=False, conductors=True),
        "filter_mediums":   lambda: apply_filter_enhanced(blocks, mediums=True, conductors=False),
        "z_slice":          lambda: apply_filter_enhanced(blocks, z_min=0.5, z_max=1.5),
    }
    for name, op in ops.items():
        _, t, _, _ = time_block(op)
        latencies[name] = t
    return latencies

def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--render", action="store_true", help="Render first frame to PNG")
    p.add_argument("--out", default="bench_results_enhanced.json")
    args = p.parse_args()

    sysinfo = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "repeat": args.repeat,
        "enhanced_version": "ehnanced_cap3d.py"
    }


    results = {"system": sysinfo, "runs": []}

    for f in args.files:
        fpath = Path(f)
        print(f"Benchmarking {fpath.name}...")
        
        for run_idx in range(args.repeat):
            print(f"  Run {run_idx + 1}/{args.repeat}")
            
            # Parse with enhanced parser
            (blocks, parse_stats), t_parse, peak_tracemalloc_mb, delta_rss = time_block(parse_cap3d_enhanced, fpath)

            # Build figure with enhanced visualizer
            fig, t_build, _, _ = time_block(build_fig_enhanced, blocks)
            
            # Render if requested
            t_render = None
            if args.render:
                t0 = time.perf_counter()
                pio.write_image(fig, f"bench_enhanced_{fpath.stem}.png", format="png", width=1200, height=900)
                t_render = time.perf_counter() - t0

            # Measure interactions (headless functions)
            lat = measure_interactions(fig, blocks)
            t_first_frame = t_parse + t_build + (t_render or 0)

            results["runs"].append({
                "file": fpath.name,
                "blocks": len(blocks),
                "conductors": parse_stats['conductors'],
                "mediums": parse_stats['mediums'],
                "t_parse_s": t_parse,
                "t_build_s": t_build,
                "t_render_s": t_render,
                "t_first_frame_s": t_first_frame,
                "peak_tracemalloc_mb": peak_tracemalloc_mb,
                "delta_rss_mb": delta_rss,
                "latencies_s": lat,
                "parse_stats": parse_stats
            })

    # Aggregate summary per file
    by_file = {}
    for r in results["runs"]:
        by_file.setdefault(r["file"], []).append(r)

    summary = []
    for fname, runs in by_file.items():
        get = lambda k: [r[k] for r in runs if r[k] is not None]
        lat_keys = runs[0]["latencies_s"].keys()
        row = {
            "file": fname,
            "blocks": runs[0]["blocks"],
            "conductors": runs[0]["conductors"],
            "mediums": runs[0]["mediums"],
            "t_parse_mean_s": stats.mean(get("t_parse_s")),
            "t_parse_std_s":  stats.pstdev(get("t_parse_s")) if len(get("t_parse_s")) > 1 else 0,
            "t_first_mean_s": stats.mean(get("t_first_frame_s")),
            "peak_mb_mean":   stats.mean(get("peak_tracemalloc_mb")),
            "rss_delta_mb":   stats.mean(get("delta_rss_mb")),
        }
        for lk in lat_keys:
            lat_vals = [r["latencies_s"][lk] for r in runs]
            row[f"{lk}_ms_mean"] = 1000*stats.mean(lat_vals)
        row["blocks_per_sec_parse"] = row["blocks"]/row["t_parse_mean_s"] if row["t_parse_mean_s"] > 0 else 0
        summary.append(row)

    results["summary"] = summary

    with open(args.out, "w") as fp:
        json.dump(results, fp, indent=2)

    print(f"\nBenchmark Results Summary:")
    print(f"{'File':<30} {'Blocks':<8} {'Parse(s)':<10} {'Total(s)':<10} {'Blk/sec':<10} {'Memory(MB)':<12}")
    print("-" * 90)
    
    for row in summary:
        print(f"{row['file']:<30} {row['blocks']:<8} {row['t_parse_mean_s']:<10.3f} "
              f"{row['t_first_mean_s']:<10.3f} {row['blocks_per_sec_parse']:<10.0f} "
              f"{row['peak_mb_mean']:<12.1f}")

    print(f"\nDetailed results written to {args.out}")

if __name__ == "__main__":
    main()
