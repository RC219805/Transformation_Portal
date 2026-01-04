#!/usr/bin/env python3
"""
Detailed V3+V2 Pipeline Profiler
Tracks timing for each stage with optimization recommendations.
"""

import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List
import sys


def run_with_timing(cmd: List[str], stage_name: str) -> tuple[float, str]:
    """Run command and return (elapsed_time, output)."""
    print(f"\n{'=' * 80}")
    print(f"STAGE: {stage_name}")
    print(f"{'=' * 80}")
    print(f"Command: {' '.join(cmd)}\n")

    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)

    print(f"\n⏱️  {stage_name}: {elapsed:.2f}s ({elapsed * 1000:.0f}ms)")

    return elapsed, result.stdout


def main():
    # Use local test images
    input_dir = Path("data/validation_expanded")
    output_dir = Path("output/profile_detailed")

    # Select 3 representative images for profiling
    test_images = [
        input_dir / "750Picacho_Aerial.jpg",
        input_dir / "750Picacho_PrimaryBathroom.jpg",
        input_dir / "800-picacho-11.jpg",
    ]

    # Verify images exist
    for img in test_images:
        if not img.exists():
            print(f"ERROR: {img} not found")
            sys.exit(1)

    # Create temp input dir with just these 3 images
    temp_input = Path("output/profile_input_temp")
    temp_input.mkdir(parents=True, exist_ok=True)

    import shutil

    for img in test_images:
        shutil.copy2(img, temp_input / img.name)

    print("\n" + "=" * 80)
    print("V3+V2 PIPELINE PERFORMANCE PROFILER")
    print("=" * 80)
    print(f"\nTest Images: {len(test_images)}")
    for img in test_images:
        print(f"  - {img.name}")
    print(f"\nOutput Dir: {output_dir}")

    timings = {}

    # ====================
    # V3 ONLY (Stage A)
    # ====================

    print("\n\n" + "=" * 80)
    print("PART 1: V3 DEPTH GENERATION ONLY")
    print("=" * 80)

    v3_output = output_dir / "v3_only"

    # Process command (generates depth maps)
    elapsed, output = run_with_timing(
        [
            "python",
            "-m",
            "lux_depth_v3.cli",
            "process",
            "--input-dir",
            str(temp_input),
            "--output-dir",
            str(v3_output),
            "--model",
            "metric-large",  # Commercial license, fast model
        ],
        "V3: Depth Generation (process command)",
    )
    timings["v3_depth_generation_total"] = elapsed
    timings["v3_per_image"] = elapsed / len(test_images)
    timings["v3_throughput_per_hour"] = (len(test_images) / elapsed) * 3600

    # ====================
    # V3+V2 INTEGRATED
    # ====================

    print("\n\n" + "=" * 80)
    print("PART 2: V3+V2 INTEGRATED PIPELINE")
    print("=" * 80)

    v3v2_output = output_dir / "v3_v2_integrated"

    # Full enhance command (V3 + V2) - V2 is enabled by default
    elapsed, output = run_with_timing(
        [
            "python",
            "-m",
            "lux_depth_v3.cli",
            "enhance",
            "--input-dir",
            str(temp_input),
            "--output-dir",
            str(v3v2_output),
            "--preset",
            "interior_luxury",
            "--non-commercial-ok",
        ],
        "V3+V2: Full Integrated Pipeline (enhance command)",
    )
    timings["v3v2_total"] = elapsed
    timings["v3v2_per_image"] = elapsed / len(test_images)
    timings["v3v2_throughput_per_hour"] = (len(test_images) / elapsed) * 3600

    # Calculate V2 overhead
    timings["v2_overhead_total"] = timings["v3v2_total"] - timings["v3_depth_generation_total"]
    timings["v2_overhead_per_image"] = timings["v2_overhead_total"] / len(test_images)
    timings["v2_overhead_percentage"] = (timings["v2_overhead_total"] / timings["v3v2_total"]) * 100

    # ====================
    # RESULTS SUMMARY
    # ====================

    print("\n\n" + "=" * 80)
    print("PERFORMANCE SUMMARY")
    print("=" * 80)

    print(f"\n{'Stage':<40} {'Total (s)':<12} {'Per Image (s)':<15} {'Images/Hour':<15}")
    print("-" * 80)

    print(
        f"{'V3 Depth Generation (Stage A)':<40} {timings['v3_depth_generation_total']:>10.2f}s  {timings['v3_per_image']:>13.2f}s  {timings['v3_throughput_per_hour']:>13.0f}"
    )
    print(
        f"{'V3+V2 Integrated (Stage A+B)':<40} {timings['v3v2_total']:>10.2f}s  {timings['v3v2_per_image']:>13.2f}s  {timings['v3v2_throughput_per_hour']:>13.0f}"
    )
    print(
        f"{'V2 Enhancement Only (Stage B)':<40} {timings['v2_overhead_total']:>10.2f}s  {timings['v2_overhead_per_image']:>13.2f}s  {(len(test_images) / timings['v2_overhead_total']) * 3600:>13.0f}"
    )

    print(f"\n{'Metric':<50} {'Value':<20}")
    print("-" * 70)
    print(f"{'V2 Overhead as % of Total':<50} {timings['v2_overhead_percentage']:>18.1f}%")
    print(f"{'V3:V2 Time Ratio':<50} {timings['v3_depth_generation_total'] / timings['v2_overhead_total']:>18.2f}:1")

    # ====================
    # OPTIMIZATION OPPORTUNITIES
    # ====================

    print("\n\n" + "=" * 80)
    print("OPTIMIZATION OPPORTUNITIES (Ranked by Impact)")
    print("=" * 80)

    opportunities = []

    # Analyze V2 overhead
    if timings["v2_overhead_percentage"] > 50:
        opportunities.append(
            {
                "priority": 1,
                "stage": "V2 Enhancement (Stage B)",
                "impact": "HIGH",
                "current": f"{timings['v2_overhead_per_image']:.2f}s per image ({timings['v2_overhead_percentage']:.1f}% of total)",
                "optimization": "Convert V2 to in-process library (avoid subprocess overhead)",
                "expected_improvement": "15-25% reduction in V2 stage time",
                "implementation": "Refactor lux_depth_v2 as importable module instead of subprocess call",
            }
        )

    # Check if V3 is bottleneck
    if timings["v3_per_image"] > 2.0:
        opportunities.append(
            {
                "priority": 2,
                "stage": "V3 Depth Inference",
                "impact": "MEDIUM",
                "current": f"{timings['v3_per_image']:.2f}s per image",
                "optimization": "Use CoreML-optimized model on Apple Silicon",
                "expected_improvement": "3-5x speedup (target: <0.5s per image)",
                "implementation": "Export DA3 model to CoreML format with ANE optimization",
            }
        )

    # Check throughput
    if timings["v3v2_throughput_per_hour"] < 300:
        opportunities.append(
            {
                "priority": 1,
                "stage": "Overall Pipeline",
                "impact": "HIGH",
                "current": f"{timings['v3v2_throughput_per_hour']:.0f} images/hour",
                "optimization": "Implement parallel processing with process pool",
                "expected_improvement": "2-4x throughput (target: 800+ images/hour)",
                "implementation": "Use multiprocessing.Pool for batch processing with 4-8 workers",
            }
        )

    # Add GPU batch processing opportunity
    opportunities.append(
        {
            "priority": 3,
            "stage": "V2 Upscaling",
            "impact": "MEDIUM",
            "current": "Sequential image processing",
            "optimization": "GPU batch processing for upscaling stage",
            "expected_improvement": "2-3x speedup for upscaling operations",
            "implementation": "Batch TorchUpscaler operations with tensor batching",
        }
    )

    for i, opp in enumerate(sorted(opportunities, key=lambda x: x["priority"]), 1):
        print(f"\n{i}. [{opp['impact']} IMPACT - Priority {opp['priority']}] {opp['stage']}")
        print(f"   Current: {opp['current']}")
        print(f"   Optimization: {opp['optimization']}")
        print(f"   Expected: {opp['expected_improvement']}")
        print(f"   Implementation: {opp['implementation']}")

    # ====================
    # SAVE RESULTS
    # ====================

    results_file = output_dir / "profiling_results.json"
    results_file.parent.mkdir(parents=True, exist_ok=True)

    with open(results_file, "w") as f:
        json.dump(
            {
                "test_config": {
                    "num_images": len(test_images),
                    "images": [img.name for img in test_images],
                    "model": "da3-base-v1.1",
                    "preset": "interior_luxury",
                },
                "timings": timings,
                "optimization_opportunities": opportunities,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )

    print(f"\n\n{'=' * 80}")
    print(f"Detailed results saved to: {results_file}")
    print(f"{'=' * 80}\n")

    # Cleanup
    shutil.rmtree(temp_input)

    return timings


if __name__ == "__main__":
    main()
