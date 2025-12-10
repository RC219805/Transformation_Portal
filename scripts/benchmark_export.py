#!/usr/bin/env python3
"""
Phase 2 Slice 3: Export Performance Benchmark (Single Image)

Detailed profiling of export optimizations on a single image.
Captures timing, file size, memory usage, and throughput metrics.

Usage:
    python scripts/benchmark_export.py \
        --input input_images/750_Picacho/Pool.tif \
        --output output_benchmark/pool \
        --mode baseline \
        --runs 3

Modes:
    baseline        - All optimizations OFF (current behavior)
    tiled           - tiff_tile_size=512, compression="lzw"
    tiled_atomic    - tiled + use_atomic_image_writes=True
    full_optimized  - tiled + atomic + tiered storage
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import psutil

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lux_depth_v2.pipeline import LuxPipelineV2, PipelineConfig
from transformation_portal.core.storage.export_manager import ExportConfig


def get_mode_config(mode: str, output_dir: Path, scratch_dir: Path | None = None) -> ExportConfig:
    """
    Build ExportConfig for the specified mode.
    
    Args:
        mode: One of "baseline", "tiled", "tiled_atomic", "full_optimized"
        output_dir: Output directory
        scratch_dir: Scratch directory (for tiered storage)
    
    Returns:
        ExportConfig with appropriate flags
    """
    if mode == "baseline":
        return ExportConfig(output_dir=output_dir)
    
    elif mode == "tiled":
        return ExportConfig(
            output_dir=output_dir,
            tiff_tile_size=512,
            tiff_compression="lzw",
        )
    
    elif mode == "tiled_atomic":
        return ExportConfig(
            output_dir=output_dir,
            tiff_tile_size=512,
            tiff_compression="lzw",
            use_atomic_image_writes=True,
            use_atomic_report_writes=True,
        )
    
    elif mode == "full_optimized":
        if scratch_dir is None:
            scratch_dir = output_dir / ".scratch"
        return ExportConfig(
            output_dir=output_dir,
            tiff_tile_size=512,
            tiff_compression="lzw",
            use_atomic_image_writes=True,
            use_atomic_report_writes=True,
            enable_tiered_storage=True,
            scratch_dir=scratch_dir,
        )
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def get_file_size_mb(path: Path) -> float:
    """Get file size in MB."""
    if not path.exists():
        return 0.0
    return path.stat().st_size / (1024 * 1024)


def get_image_dimensions(path: Path) -> tuple[int, int]:
    """Get image dimensions from file."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size  # (width, height)
    except Exception:
        return (0, 0)


def benchmark_single_run(
    input_path: Path,
    output_dir: Path,
    mode: str,
    scratch_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Run benchmark for a single image with specified mode.
    
    Returns:
        Dictionary with timing, file size, memory, throughput metrics
    """
    # Get baseline memory
    process = psutil.Process()
    baseline_rss_mb = process.memory_info().rss / (1024 * 1024)
    
    # Get input info
    width, height = get_image_dimensions(input_path)
    input_size_mp = (width * height) / 1_000_000 if width > 0 else 0
    
    # Build config
    export_config = get_mode_config(mode, output_dir, scratch_dir)
    
    # TODO: Wire ExportConfig into PipelineConfig
    # For now, use default PipelineConfig and note that export optimizations
    # will need to be wired through the pipeline
    pipeline_config = PipelineConfig(
        output_dir=str(output_dir),
        write_outputs=True,
    )
    
    # Create pipeline
    pipeline = LuxPipelineV2(pipeline_config)
    
    # Run processing
    start_time = time.time()
    result = pipeline.process_one(input_path)
    end_time = time.time()
    
    # Get peak memory
    peak_rss_mb = process.memory_info().rss / (1024 * 1024)
    delta_rss_mb = peak_rss_mb - baseline_rss_mb
    
    # Extract timing from result
    timing_stages = result.get("timing_stages_s", {})
    export_master = timing_stages.get("export_master", 0.0)
    export_upscaled = timing_stages.get("export_upscaled", 0.0)
    export_preview = timing_stages.get("export_preview", 0.0)
    export_marketing = timing_stages.get("export_marketing", 0.0)
    export_report = timing_stages.get("export_report", 0.0)
    total_export = export_master + export_upscaled + export_preview + export_marketing + export_report
    
    # Get file sizes (approximate paths)
    stem = input_path.stem
    master_path = output_dir / f"{stem}_master16.tif"
    upscaled_path = output_dir / f"{stem}_upscaled16.tif"
    
    master_size_mb = get_file_size_mb(master_path)
    upscaled_size_mb = get_file_size_mb(upscaled_path)
    total_size_mb = master_size_mb + upscaled_size_mb
    
    # Calculate throughput
    total_time = end_time - start_time
    images_per_hour = 3600 / total_time if total_time > 0 else 0
    mb_per_second = total_size_mb / total_time if total_time > 0 else 0
    
    return {
        "input": {
            "path": str(input_path),
            "size_mp": round(input_size_mp, 2),
            "dimensions": [width, height],
        },
        "timing": {
            "export_master": round(export_master, 3),
            "export_upscaled": round(export_upscaled, 3),
            "export_preview": round(export_preview, 3),
            "export_marketing": round(export_marketing, 3),
            "export_report": round(export_report, 3),
            "total_export": round(total_export, 3),
            "total_pipeline": round(total_time, 3),
        },
        "file_size": {
            "master_mb": round(master_size_mb, 2),
            "upscaled_mb": round(upscaled_size_mb, 2),
            "total_mb": round(total_size_mb, 2),
        },
        "memory": {
            "peak_rss_mb": round(peak_rss_mb, 2),
            "baseline_rss_mb": round(baseline_rss_mb, 2),
            "delta_rss_mb": round(delta_rss_mb, 2),
        },
        "throughput": {
            "images_per_hour": round(images_per_hour, 1),
            "mb_per_second": round(mb_per_second, 2),
        },
    }


def run_benchmark(
    input_path: Path,
    output_dir: Path,
    mode: str,
    runs: int = 3,
    scratch_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Run benchmark multiple times and aggregate results.
    
    Args:
        input_path: Input image path
        output_dir: Output directory
        mode: Benchmark mode
        runs: Number of runs (for averaging)
        scratch_dir: Scratch directory (optional)
    
    Returns:
        Dictionary with aggregated results
    """
    print(f"Running benchmark: mode={mode}, runs={runs}")
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    
    results = []
    for i in range(runs):
        print(f"\nRun {i+1}/{runs}...")
        run_output = output_dir / f"run_{i+1}"
        result = benchmark_single_run(input_path, run_output, mode, scratch_dir)
        results.append(result)
        
        # Print timing
        timing = result["timing"]
        print(f"  Export time: {timing['total_export']:.2f}s")
        print(f"  Total time: {timing['total_pipeline']:.2f}s")
        print(f"  File size: {result['file_size']['total_mb']:.1f} MB")
    
    # Aggregate results (average)
    def avg(key_path: List[str]) -> float:
        values = [result]
        for key in key_path:
            values = [v[key] for v in values if isinstance(v, dict) and key in v]
        if not values or not all(isinstance(v, (int, float)) for v in values):
            return 0.0
        return sum(values) / len(values)
    
    aggregated = {
        "test_id": f"{mode}_{input_path.stem}",
        "mode": mode,
        "config": get_mode_config(mode, output_dir, scratch_dir).__dict__,
        "runs": runs,
        "input": results[0]["input"],  # Same for all runs
        "timing_avg": {
            "export_master": round(avg(["timing", "export_master"]), 3),
            "export_upscaled": round(avg(["timing", "export_upscaled"]), 3),
            "total_export": round(avg(["timing", "total_export"]), 3),
            "total_pipeline": round(avg(["timing", "total_pipeline"]), 3),
        },
        "file_size": results[0]["file_size"],  # Same for all runs
        "memory_avg": {
            "peak_rss_mb": round(avg(["memory", "peak_rss_mb"]), 2),
            "delta_rss_mb": round(avg(["memory", "delta_rss_mb"]), 2),
        },
        "throughput_avg": {
            "images_per_hour": round(avg(["throughput", "images_per_hour"]), 1),
            "mb_per_second": round(avg(["throughput", "mb_per_second"]), 2),
        },
        "all_runs": results,
    }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Benchmark export optimizations")
    parser.add_argument("--input", type=Path, required=True, help="Input image path")
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument(
        "--mode",
        choices=["baseline", "tiled", "tiled_atomic", "full_optimized"],
        required=True,
        help="Benchmark mode",
    )
    parser.add_argument("--runs", type=int, default=3, help="Number of runs for averaging")
    parser.add_argument("--scratch", type=Path, help="Scratch directory (for full_optimized mode)")
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Run benchmark
    results = run_benchmark(
        input_path=args.input,
        output_dir=args.output,
        mode=args.mode,
        runs=args.runs,
        scratch_dir=args.scratch,
    )
    
    # Save results
    results_path = args.output / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Benchmark complete!")
    print(f"Results saved to: {results_path}")
    print(f"\nSummary:")
    print(f"  Mode: {results['mode']}")
    print(f"  Export time (avg): {results['timing_avg']['total_export']:.2f}s")
    print(f"  Total time (avg): {results['timing_avg']['total_pipeline']:.2f}s")
    print(f"  File size: {results['file_size']['total_mb']:.1f} MB")
    print(f"  Throughput: {results['throughput_avg']['images_per_hour']:.1f} images/hour")


if __name__ == "__main__":
    main()
