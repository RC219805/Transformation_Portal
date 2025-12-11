#!/usr/bin/env python3
"""
Analyze heavy quality benchmark results.

Compares baseline vs heavy-quality stage timings to identify performance ceilings.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import statistics


def analyze_stage_breakdown(reports: List[Dict[str, Any]], config_name: str) -> Dict[str, Any]:
    """Analyze stage timings for a given configuration."""
    
    stage_times = defaultdict(list)
    total_times = []
    memory_peaks = []
    
    for report in reports:
        # Extract stage timings
        stages = report.get("stage_times_sec", {})
        for stage, time_val in stages.items():
            stage_times[stage].append(time_val)
        
        # Extract totals
        timing = report.get("timing_s", 0)
        if timing:
            total_times.append(timing)
        
        # Extract memory
        memory = report.get("memory_usage_mb", {})
        if "peak_rss" in memory:
            memory_peaks.append(memory["peak_rss"])
    
    # Compute medians
    results = {
        "config": config_name,
        "n_samples": len(reports),
        "total_time_median_s": statistics.median(total_times) if total_times else 0,
        "peak_memory_median_mb": statistics.median(memory_peaks) if memory_peaks else 0,
        "stages": {}
    }
    
    for stage, times in stage_times.items():
        if times:
            results["stages"][stage] = {
                "median_s": statistics.median(times),
                "p75_s": statistics.quantiles(times, n=4)[2] if len(times) >= 4 else statistics.median(times),
                "p95_s": statistics.quantiles(times, n=20)[18] if len(times) >= 20 else statistics.median(times),
            }
    
    return results


def compare_configs(baseline: Dict[str, Any], heavy: Dict[str, Any]) -> None:
    """Print comparison between baseline and heavy configurations."""
    
    print("=" * 100)
    print("HEAVY QUALITY BENCHMARK ANALYSIS")
    print("=" * 100)
    print()
    
    # Overall comparison
    print("## Overall Performance")
    print()
    
    baseline_total = baseline["total_time_median_s"]
    heavy_total = heavy["total_time_median_s"]
    delta_total = heavy_total - baseline_total
    pct_total = 100 * delta_total / baseline_total if baseline_total > 0 else 0
    
    print(f"Baseline (current production):")
    print(f"  Total time: {baseline_total:.1f}s")
    print(f"  Peak memory: {baseline['peak_memory_median_mb']:.0f} MB")
    print(f"  N samples: {baseline['n_samples']}")
    print()
    
    print(f"Heavy (max quality):")
    print(f"  Total time: {heavy_total:.1f}s")
    print(f"  Peak memory: {heavy['peak_memory_median_mb']:.0f} MB")
    print(f"  N samples: {heavy['n_samples']}")
    print()
    
    print(f"Delta:")
    print(f"  Time: {delta_total:+.1f}s ({pct_total:+.1f}%)")
    memory_delta = heavy['peak_memory_median_mb'] - baseline['peak_memory_median_mb']
    print(f"  Memory: {memory_delta:+.0f} MB")
    print()
    
    # Stage-by-stage comparison
    print("=" * 100)
    print("## Stage-by-Stage Breakdown")
    print("=" * 100)
    print()
    
    all_stages = set(baseline["stages"].keys()) | set(heavy["stages"].keys())
    
    for stage in sorted(all_stages):
        baseline_stage = baseline["stages"].get(stage, {})
        heavy_stage = heavy["stages"].get(stage, {})
        
        baseline_time = baseline_stage.get("median_s", 0)
        heavy_time = heavy_stage.get("median_s", 0)
        
        if baseline_time == 0 and heavy_time == 0:
            continue
        
        delta = heavy_time - baseline_time
        pct = 100 * delta / baseline_time if baseline_time > 0 else float('inf')
        
        # Percentage of total
        baseline_pct_of_total = 100 * baseline_time / baseline_total if baseline_total > 0 else 0
        heavy_pct_of_total = 100 * heavy_time / heavy_total if heavy_total > 0 else 0
        
        print(f"{stage}:")
        print(f"  Baseline: {baseline_time:.1f}s ({baseline_pct_of_total:.1f}% of total)")
        print(f"  Heavy:    {heavy_time:.1f}s ({heavy_pct_of_total:.1f}% of total)")
        print(f"  Delta:    {delta:+.1f}s ({pct:+.1f}%)")
        print()
    
    # Summary
    print("=" * 100)
    print("## SUMMARY")
    print("=" * 100)
    print()
    
    if delta_total > 0:
        overhead_pct = 100 * delta_total / baseline_total
        print(f"Heavy quality adds {delta_total:.1f}s ({overhead_pct:.1f}%) overhead.")
        print()
        print("Top cost increases:")
        
        # Find stages with biggest increases
        deltas = []
        for stage in all_stages:
            baseline_time = baseline["stages"].get(stage, {}).get("median_s", 0)
            heavy_time = heavy["stages"].get(stage, {}).get("median_s", 0)
            if heavy_time > baseline_time:
                deltas.append((stage, heavy_time - baseline_time))
        
        deltas.sort(key=lambda x: x[1], reverse=True)
        for i, (stage, delta) in enumerate(deltas[:5], 1):
            pct_of_increase = 100 * delta / delta_total
            print(f"  {i}. {stage}: +{delta:.1f}s ({pct_of_increase:.1f}% of total increase)")
    else:
        print("Heavy quality is faster than baseline (unexpected!)")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze heavy quality benchmark results."
    )
    parser.add_argument(
        "benchmark_dir",
        type=Path,
        help="Benchmark directory containing baseline_* and heavy_* subdirectories"
    )
    args = parser.parse_args()
    
    if not args.benchmark_dir.exists():
        print(f"Error: {args.benchmark_dir} does not exist!")
        return 1
    
    # Load baseline reports
    baseline_reports = []
    for report_path in args.benchmark_dir.glob("baseline_*/*_report.json"):
        try:
            with open(report_path) as f:
                baseline_reports.append(json.load(f))
        except Exception as e:
            print(f"Warning: Failed to read {report_path}: {e}")
    
    # Load heavy reports
    heavy_reports = []
    for report_path in args.benchmark_dir.glob("heavy_*/*_report.json"):
        try:
            with open(report_path) as f:
                heavy_reports.append(json.load(f))
        except Exception as e:
            print(f"Warning: Failed to read {report_path}: {e}")
    
    if not baseline_reports:
        print("Error: No baseline reports found!")
        return 1
    
    if not heavy_reports:
        print("Error: No heavy reports found!")
        return 1
    
    print(f"Loaded {len(baseline_reports)} baseline and {len(heavy_reports)} heavy reports")
    print()
    
    # Analyze
    baseline_analysis = analyze_stage_breakdown(baseline_reports, "baseline")
    heavy_analysis = analyze_stage_breakdown(heavy_reports, "heavy")
    
    # Compare
    compare_configs(baseline_analysis, heavy_analysis)
    
    return 0


if __name__ == "__main__":
    exit(main())
