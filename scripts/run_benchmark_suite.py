#!/usr/bin/env python3
"""
Phase 2 Slice 3: Automated Benchmark Suite

Runs complete benchmark across all test images and modes.
Generates comparison tables, charts, and final markdown report.

Usage:
    python scripts/run_benchmark_suite.py \
        --input-dir input_images/750_Picacho \
        --output-dir output_benchmark \
        --images Pool Aerial GreatRoom Kitchen \
        --runs 3

Features:
    - Runs all modes (baseline, tiled, tiled_atomic, full_optimized)
    - Generates aggregated comparison tables
    - Computes performance gains vs baseline
    - Auto-populates PHASE2_SLICE3_PERFORMANCE_RESULTS.md
    - Creates CSV for further analysis
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))


MODES = ["baseline", "tiled", "tiled_atomic", "full_optimized"]


def run_single_benchmark(
    input_path: Path,
    output_dir: Path,
    mode: str,
    runs: int = 3,
    scratch_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Run benchmark for single image + mode combination.
    
    Returns:
        Parsed JSON results
    """
    cmd = [
        sys.executable,
        "scripts/benchmark_export.py",
        "--input", str(input_path),
        "--output", str(output_dir),
        "--mode", mode,
        "--runs", str(runs),
    ]
    
    if scratch_dir and mode == "full_optimized":
        cmd.extend(["--scratch", str(scratch_dir)])
    
    print(f"\n{'='*80}")
    print(f"Running: {input_path.stem} [{mode}]")
    print(f"{'='*80}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        
        # Load results
        results_path = output_dir / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
        else:
            print(f"Warning: Results file not found: {results_path}")
            return {}
    
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return {}


def compute_performance_gain(baseline: float, optimized: float) -> float:
    """Compute percentage reduction (positive = faster)."""
    if baseline == 0:
        return 0.0
    return ((baseline - optimized) / baseline) * 100


def compute_compression_ratio(baseline_mb: float, compressed_mb: float) -> float:
    """Compute compression ratio."""
    if compressed_mb == 0:
        return 0.0
    return baseline_mb / compressed_mb


def generate_comparison_table(
    image_name: str,
    results_by_mode: Dict[str, Dict[str, Any]],
) -> str:
    """
    Generate markdown comparison table for single image.
    
    Args:
        image_name: Name of the image (e.g., "Pool")
        results_by_mode: Dict mapping mode to results
    
    Returns:
        Markdown table string
    """
    baseline = results_by_mode.get("baseline", {})
    tiled = results_by_mode.get("tiled", {})
    tiled_atomic = results_by_mode.get("tiled_atomic", {})
    full = results_by_mode.get("full_optimized", {})
    
    # Extract key metrics
    def get_metric(results: dict, key_path: List[str], default: str = "N/A") -> str:
        try:
            val = results
            for key in key_path:
                val = val[key]
            return f"{val:.2f}" if isinstance(val, (int, float)) else str(val)
        except (KeyError, TypeError):
            return default
    
    # Build table
    table = f"\n### {image_name}\n\n"
    table += "| Metric | Baseline | Tiled | Tiled+Atomic | Full Optimized |\n"
    table += "|--------|----------|-------|--------------|----------------|\n"
    
    # Export timing
    table += "| **Export Time (s)** | | | | |\n"
    table += f"| Master TIFF | {get_metric(baseline, ['timing_avg', 'export_master'])} | "
    table += f"{get_metric(tiled, ['timing_avg', 'export_master'])} | "
    table += f"{get_metric(tiled_atomic, ['timing_avg', 'export_master'])} | "
    table += f"{get_metric(full, ['timing_avg', 'export_master'])} |\n"
    
    table += f"| Upscaled TIFF | {get_metric(baseline, ['timing_avg', 'export_upscaled'])} | "
    table += f"{get_metric(tiled, ['timing_avg', 'export_upscaled'])} | "
    table += f"{get_metric(tiled_atomic, ['timing_avg', 'export_upscaled'])} | "
    table += f"{get_metric(full, ['timing_avg', 'export_upscaled'])} |\n"
    
    table += f"| Total Export | {get_metric(baseline, ['timing_avg', 'total_export'])} | "
    table += f"{get_metric(tiled, ['timing_avg', 'total_export'])} | "
    table += f"{get_metric(tiled_atomic, ['timing_avg', 'total_export'])} | "
    table += f"{get_metric(full, ['timing_avg', 'total_export'])} |\n"
    
    # File size
    table += "| **File Size (MB)** | | | | |\n"
    table += f"| Master 16-bit | {get_metric(baseline, ['file_size', 'master_mb'])} | "
    table += f"{get_metric(tiled, ['file_size', 'master_mb'])} | "
    table += f"{get_metric(tiled_atomic, ['file_size', 'master_mb'])} | "
    table += f"{get_metric(full, ['file_size', 'master_mb'])} |\n"
    
    table += f"| Upscaled 16-bit | {get_metric(baseline, ['file_size', 'upscaled_mb'])} | "
    table += f"{get_metric(tiled, ['file_size', 'upscaled_mb'])} | "
    table += f"{get_metric(tiled_atomic, ['file_size', 'upscaled_mb'])} | "
    table += f"{get_metric(full, ['file_size', 'upscaled_mb'])} |\n"
    
    table += f"| Total | {get_metric(baseline, ['file_size', 'total_mb'])} | "
    table += f"{get_metric(tiled, ['file_size', 'total_mb'])} | "
    table += f"{get_metric(tiled_atomic, ['file_size', 'total_mb'])} | "
    table += f"{get_metric(full, ['file_size', 'total_mb'])} |\n"
    
    # Memory
    table += "| **Memory (MB)** | | | | |\n"
    table += f"| Peak RSS | {get_metric(baseline, ['memory_avg', 'peak_rss_mb'])} | "
    table += f"{get_metric(tiled, ['memory_avg', 'peak_rss_mb'])} | "
    table += f"{get_metric(tiled_atomic, ['memory_avg', 'peak_rss_mb'])} | "
    table += f"{get_metric(full, ['memory_avg', 'peak_rss_mb'])} |\n"
    
    # Throughput
    table += "| **Throughput** | | | | |\n"
    table += f"| Images/hour | {get_metric(baseline, ['throughput_avg', 'images_per_hour'])} | "
    table += f"{get_metric(tiled, ['throughput_avg', 'images_per_hour'])} | "
    table += f"{get_metric(tiled_atomic, ['throughput_avg', 'images_per_hour'])} | "
    table += f"{get_metric(full, ['throughput_avg', 'images_per_hour'])} |\n"
    
    # Compute gains
    if baseline and tiled:
        baseline_time = baseline.get("timing_avg", {}).get("total_export", 0)
        tiled_time = tiled.get("timing_avg", {}).get("total_export", 0)
        if baseline_time > 0 and tiled_time > 0:
            gain = compute_performance_gain(baseline_time, tiled_time)
            table += f"\n**Performance Gain (Tiled)**: {gain:.1f}% faster\n"
        
        baseline_size = baseline.get("file_size", {}).get("total_mb", 0)
        tiled_size = tiled.get("file_size", {}).get("total_mb", 0)
        if baseline_size > 0 and tiled_size > 0:
            size_reduction = compute_performance_gain(baseline_size, tiled_size)
            compression_ratio = compute_compression_ratio(baseline_size, tiled_size)
            table += f"**File Size Reduction**: {size_reduction:.1f}% smaller (compression ratio: {compression_ratio:.2f}x)\n"
    
    return table


def generate_aggregate_table(all_results: Dict[str, Dict[str, Dict]]) -> str:
    """
    Generate aggregate statistics table across all images.
    
    Args:
        all_results: Nested dict {image_name: {mode: results}}
    
    Returns:
        Markdown table string
    """
    table = "\n## Aggregate Statistics\n\n"
    table += "### Export Latency Reduction\n\n"
    table += "| Scene | Image Size | Baseline (s) | Tiled (s) | Reduction (%) | Target Met? |\n"
    table += "|-------|------------|--------------|-----------|---------------|-------------|\n"
    
    total_baseline = 0.0
    total_tiled = 0.0
    count = 0
    
    for image_name, results_by_mode in all_results.items():
        baseline = results_by_mode.get("baseline", {})
        tiled = results_by_mode.get("tiled", {})
        
        size_mp = baseline.get("input", {}).get("size_mp", 0)
        baseline_time = baseline.get("timing_avg", {}).get("total_export", 0)
        tiled_time = tiled.get("timing_avg", {}).get("total_export", 0)
        
        if baseline_time > 0 and tiled_time > 0:
            reduction = compute_performance_gain(baseline_time, tiled_time)
            target_met = "✅" if reduction >= 30 else "❌"
            
            table += f"| {image_name} | {size_mp:.1f} MP | {baseline_time:.2f} | {tiled_time:.2f} | {reduction:.1f}% | {target_met} |\n"
            
            total_baseline += baseline_time
            total_tiled += tiled_time
            count += 1
    
    # Average
    if count > 0:
        avg_reduction = compute_performance_gain(total_baseline / count, total_tiled / count)
        target_met = "✅" if avg_reduction >= 30 else "❌"
        table += f"| **Average** | - | {total_baseline/count:.2f} | {total_tiled/count:.2f} | {avg_reduction:.1f}% | {target_met} |\n"
    
    table += "\n**Target**: 30-50% reduction on 50MP+ images\n"
    
    # File size table
    table += "\n### File Size Reduction\n\n"
    table += "| Scene | Baseline (MB) | Tiled+LZW (MB) | Reduction (%) | Compression Ratio | Target Met? |\n"
    table += "|-------|---------------|----------------|---------------|-------------------|-------------|\n"
    
    total_baseline_size = 0.0
    total_tiled_size = 0.0
    count = 0
    
    for image_name, results_by_mode in all_results.items():
        baseline = results_by_mode.get("baseline", {})
        tiled = results_by_mode.get("tiled", {})
        
        baseline_size = baseline.get("file_size", {}).get("total_mb", 0)
        tiled_size = tiled.get("file_size", {}).get("total_mb", 0)
        
        if baseline_size > 0 and tiled_size > 0:
            reduction = compute_performance_gain(baseline_size, tiled_size)
            compression = compute_compression_ratio(baseline_size, tiled_size)
            target_met = "✅" if reduction >= 20 else "❌"
            
            table += f"| {image_name} | {baseline_size:.1f} | {tiled_size:.1f} | {reduction:.1f}% | {compression:.2f}x | {target_met} |\n"
            
            total_baseline_size += baseline_size
            total_tiled_size += tiled_size
            count += 1
    
    # Average
    if count > 0:
        avg_reduction = compute_performance_gain(total_baseline_size / count, total_tiled_size / count)
        avg_compression = compute_compression_ratio(total_baseline_size / count, total_tiled_size / count)
        target_met = "✅" if avg_reduction >= 20 else "❌"
        table += f"| **Average** | {total_baseline_size/count:.1f} | {total_tiled_size/count:.1f} | {avg_reduction:.1f}% | {avg_compression:.2f}x | {target_met} |\n"
    
    table += "\n**Target**: 20-40% reduction with compression\n"
    
    return table


def run_benchmark_suite(
    input_dir: Path,
    output_dir: Path,
    images: List[str],
    runs: int = 3,
    scratch_dir: Path | None = None,
) -> Dict[str, Dict[str, Dict]]:
    """
    Run complete benchmark suite.
    
    Returns:
        Nested dict {image_name: {mode: results}}
    """
    all_results = {}
    
    for image_name in images:
        input_path = input_dir / f"{image_name}.tif"
        if not input_path.exists():
            print(f"Warning: Input not found: {input_path}")
            continue
        
        results_by_mode = {}
        
        for mode in MODES:
            mode_output = output_dir / f"{image_name.lower()}_{mode}"
            results = run_single_benchmark(
                input_path=input_path,
                output_dir=mode_output,
                mode=mode,
                runs=runs,
                scratch_dir=scratch_dir,
            )
            results_by_mode[mode] = results
        
        all_results[image_name] = results_by_mode
    
    return all_results


def save_results(
    all_results: Dict[str, Dict[str, Dict]],
    output_dir: Path,
):
    """Save aggregated results to JSON and CSV."""
    # JSON
    json_path = output_dir / "all_results.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to: {json_path}")
    
    # CSV (flatten for spreadsheet analysis)
    csv_path = output_dir / "comparison.csv"
    with open(csv_path, "w") as f:
        f.write("Image,Mode,Export_Time_s,File_Size_MB,Peak_RSS_MB,Images_Per_Hour\n")
        for image_name, results_by_mode in all_results.items():
            for mode, results in results_by_mode.items():
                export_time = results.get("timing_avg", {}).get("total_export", 0)
                file_size = results.get("file_size", {}).get("total_mb", 0)
                peak_rss = results.get("memory_avg", {}).get("peak_rss_mb", 0)
                throughput = results.get("throughput_avg", {}).get("images_per_hour", 0)
                f.write(f"{image_name},{mode},{export_time:.2f},{file_size:.2f},{peak_rss:.2f},{throughput:.1f}\n")
    print(f"✅ CSV saved to: {csv_path}")


def update_results_markdown(
    all_results: Dict[str, Dict[str, Dict]],
    output_path: Path,
):
    """Update PHASE2_SLICE3_PERFORMANCE_RESULTS.md with actual data."""
    content = "# Phase 2 Slice 3: Performance Validation Results\n\n"
    content += "**Status**: ✅ Complete\n"
    content += f"**Date**: {Path(__file__).stat().st_mtime}\n\n"
    
    content += "---\n\n"
    content += "## Executive Summary\n\n"
    content += "Performance validation of Slice 3 PR-2 export optimizations complete.\n\n"
    
    # Detailed per-image results
    content += "---\n\n## Detailed Results\n\n"
    for image_name, results_by_mode in all_results.items():
        content += generate_comparison_table(image_name, results_by_mode)
        content += "\n"
    
    # Aggregate statistics
    content += generate_aggregate_table(all_results)
    
    # Write
    with open(output_path, "w") as f:
        f.write(content)
    
    print(f"\n✅ Results markdown updated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run complete benchmark suite")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with test images")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for results")
    parser.add_argument("--images", nargs="+", required=True, help="List of image names (without .tif)")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs per benchmark")
    parser.add_argument("--scratch", type=Path, help="Scratch directory for tiered storage tests")
    parser.add_argument(
        "--results-md",
        type=Path,
        default=Path("docs/guides/PHASE2_SLICE3_PERFORMANCE_RESULTS.md"),
        help="Path to results markdown file",
    )
    
    args = parser.parse_args()
    
    # Create output dir
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("PHASE 2 SLICE 3: BENCHMARK SUITE")
    print(f"{'='*80}")
    print(f"Input dir: {args.input_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Images: {', '.join(args.images)}")
    print(f"Runs per benchmark: {args.runs}")
    print(f"{'='*80}\n")
    
    # Run benchmarks
    all_results = run_benchmark_suite(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        images=args.images,
        runs=args.runs,
        scratch_dir=args.scratch,
    )
    
    # Save results
    save_results(all_results, args.output_dir)
    
    # Update markdown
    update_results_markdown(all_results, args.results_md)
    
    print(f"\n{'='*80}")
    print("✅ BENCHMARK SUITE COMPLETE")
    print(f"{'='*80}")
    print(f"\nResults:")
    print(f"  - JSON: {args.output_dir}/all_results.json")
    print(f"  - CSV: {args.output_dir}/comparison.csv")
    print(f"  - Markdown: {args.results_md}")
    print(f"\nNext steps:")
    print(f"  1. Review {args.results_md}")
    print(f"  2. Decide rollout strategy based on data")
    print(f"  3. Implement gradual rollout (>80MP → >50MP → all)")


if __name__ == "__main__":
    main()
