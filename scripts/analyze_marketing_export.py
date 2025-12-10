#!/usr/bin/env python3
"""
Analyze marketing export performance across different settings.

Usage:
    python scripts/analyze_marketing_export.py benchmark_dir1/ benchmark_dir2/ ...
"""

import argparse
import json
import statistics
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def analyze_marketing(reports: List[Dict[str, Any]]) -> None:
    """Analyze marketing export metrics with median-based comparison."""
    
    if not reports:
        print("No reports found!")
        return
    
    # Group by encoder and compression level
    by_setting = defaultdict(lambda: {"times": [], "sizes": [], "cpus": [], "images": []})
    
    for report in reports:
        mkt = report.get("marketing_export", {})
        if not mkt:
            continue
        
        encoder = mkt.get("encoder", "unknown")
        level = mkt.get("compression_level", "unknown")
        time = mkt.get("write_time_s", 0)
        size = mkt.get("bytes_written", 0)
        cpu = mkt.get("cpu_percent_delta", 0)
        img_name = report.get("image", "unknown")
        
        key = (encoder, level)
        by_setting[key]["times"].append(time)
        by_setting[key]["sizes"].append(size)
        by_setting[key]["cpus"].append(cpu)
        by_setting[key]["images"].append(Path(img_name).stem if isinstance(img_name, str) else "unknown")
    
    if not by_setting:
        print("No marketing export data found in reports!")
        return
    
    # Print median-based comparison
    print("=" * 100)
    print("MARKETING EXPORT ANALYSIS (Median-Based)")
    print("=" * 100)
    print()
    
    # Sort by encoder then level
    sorted_settings = sorted(by_setting.items(), key=lambda x: (x[0][0], x[0][1]))
    
    # Find baseline (level 6) for comparison
    baseline_time = None
    baseline_size = None
    for (encoder, level), data in sorted_settings:
        if level == 6:
            baseline_time = statistics.median(data["times"]) if data["times"] else None
            baseline_size = statistics.median(data["sizes"]) if data["sizes"] else None
            break
    
    for (encoder, level), data in sorted_settings:
        times = data["times"]
        sizes = data["sizes"]
        cpus = data["cpus"]
        images = data["images"]
        
        if not times:
            continue
        
        median_time = statistics.median(times)
        median_size = statistics.median(sizes)
        median_cpu = statistics.median(cpus) if cpus else 0
        
        # Compute p75/p95 if we have enough samples
        try:
            p75_time = statistics.quantiles(times, n=4)[2] if len(times) >= 4 else median_time
            p95_time = statistics.quantiles(times, n=20)[18] if len(times) >= 20 else median_time
        except statistics.StatisticsError:
            p75_time = median_time
            p95_time = median_time
        
        # Compute vs baseline
        time_vs_baseline = ""
        size_vs_baseline = ""
        if baseline_time and baseline_time > 0:
            time_diff = median_time - baseline_time
            time_pct = 100 * time_diff / baseline_time
            time_vs_baseline = f" (vs level 6: {time_diff:+.1f}s / {time_pct:+.1f}%)"
        
        if baseline_size and baseline_size > 0:
            size_diff = median_size - baseline_size
            size_pct = 100 * size_diff / baseline_size
            size_vs_baseline = f" (vs level 6: {size_diff/1024/1024:+.1f}MB / {size_pct:+.1f}%)"
        
        print(f"{encoder} level {level}:")
        print(f"  Time (median): {median_time:.1f}s{time_vs_baseline}")
        if len(times) > 1:
            print(f"         (p75): {p75_time:.1f}s, (p95): {p95_time:.1f}s")
        print(f"  Size (median): {median_size / 1024 / 1024:.1f} MB{size_vs_baseline}")
        print(f"  CPU delta (median): {median_cpu:.1f}%")
        print(f"  N samples: {len(times)}")
        print(f"  Images: {', '.join(set(images))}")
        print()
    
    # Summary recommendations
    print("=" * 100)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 100)
    print()
    
    if baseline_time:
        print("Baseline (level 6):")
        print(f"  Time: {baseline_time:.1f}s")
        print(f"  Size: {baseline_size / 1024 / 1024:.1f} MB")
        print()
        
        print("Potential optimizations:")
        for (encoder, level), data in sorted_settings:
            if level == 6 or not data["times"]:
                continue
            
            median_time = statistics.median(data["times"])
            median_size = statistics.median(data["sizes"])
            
            time_savings = baseline_time - median_time
            time_savings_pct = 100 * time_savings / baseline_time
            size_increase_pct = 100 * (median_size - baseline_size) / baseline_size
            
            if time_savings > 5:  # More than 5s savings
                verdict = "✅ RECOMMENDED" if size_increase_pct <= 20 else "⚠️  CONSIDER"
                print(f"  Level {level}: {time_savings:.1f}s savings ({time_savings_pct:.1f}%), "
                      f"size +{size_increase_pct:.1f}% {verdict}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze marketing export performance from pipeline reports."
    )
    parser.add_argument(
        "output_dirs",
        nargs="+",
        type=Path,
        help="Output directories containing *_report.json files"
    )
    args = parser.parse_args()
    
    reports = []
    for output_dir in args.output_dirs:
        if not output_dir.exists():
            print(f"Warning: {output_dir} does not exist, skipping")
            continue
        
        for report_path in output_dir.glob("**/*_report.json"):
            try:
                with open(report_path) as f:
                    report = json.load(f)
                    reports.append(report)
            except Exception as e:
                print(f"Warning: Failed to read {report_path}: {e}")
    
    if not reports:
        print("Error: No valid reports found!")
        return 1
    
    print(f"Loaded {len(reports)} reports from {len(args.output_dirs)} directories")
    print()
    
    analyze_marketing(reports)
    return 0


if __name__ == "__main__":
    exit(main())
