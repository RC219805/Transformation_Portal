#!/usr/bin/env python3
"""
Analyze Autotune Production Results

Analyzes processing reports from autotune-enabled production runs to validate
performance gains and identify anomalies.

Usage:
    python scripts/analyze_autotune_production.py output_autotune/aerial_batch/
"""

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Any


def load_reports(output_dir: Path) -> List[Dict[str, Any]]:
    """Load all processing reports from output directory."""
    report_paths = list(output_dir.glob("**/*_report.json"))
    
    if not report_paths:
        print(f"No reports found in {output_dir}", file=sys.stderr)
        sys.exit(1)
    
    reports = []
    for report_path in report_paths:
        try:
            with open(report_path) as f:
                report = json.load(f)
                report["_path"] = str(report_path)
                reports.append(report)
        except Exception as e:
            print(f"Warning: Failed to load {report_path}: {e}", file=sys.stderr)
    
    return reports


def analyze_timing(reports: List[Dict[str, Any]]) -> None:
    """Analyze timing metrics across all reports."""
    total_pipeline_times = []
    total_export_times = []
    export_master_times = []
    export_upscaled_times = []
    export_marketing_times = []
    
    autotune_enabled_count = 0
    tiled_atomic_count = 0
    
    for report in reports:
        timing_s = report.get("timing_s", {})
        timing_stages_s = report.get("timing_stages_s", {})
        export_autotune = report.get("export_autotune", {})
        
        total_pipeline_times.append(timing_s.get("total_pipeline", 0))
        total_export_times.append(timing_s.get("total_export", 0))
        
        export_master_times.append(timing_stages_s.get("export_master", 0))
        export_upscaled_times.append(timing_stages_s.get("export_upscaled", 0))
        export_marketing_times.append(timing_stages_s.get("export_marketing", 0))
        
        if export_autotune.get("enabled"):
            autotune_enabled_count += 1
            
            final_cfg = export_autotune.get("final_export_config", {})
            if final_cfg.get("tiff_tile_size") == 512:
                tiled_atomic_count += 1
    
    print("=" * 80)
    print("AUTOTUNE PRODUCTION ANALYSIS")
    print("=" * 80)
    print()
    
    print(f"Total Images: {len(reports)}")
    print(f"Autotune Enabled: {autotune_enabled_count} ({100*autotune_enabled_count/len(reports):.1f}%)")
    print(f"Tiled Atomic Used: {tiled_atomic_count} ({100*tiled_atomic_count/len(reports) if reports else 0:.1f}%)")
    print()
    
    print("TIMING SUMMARY")
    print("-" * 80)
    print(f"Total Pipeline:")
    print(f"  Mean:   {statistics.mean(total_pipeline_times):.1f}s")
    print(f"  Median: {statistics.median(total_pipeline_times):.1f}s")
    print(f"  StdDev: {statistics.stdev(total_pipeline_times) if len(total_pipeline_times) > 1 else 0:.1f}s")
    print(f"  Range:  {min(total_pipeline_times):.1f}s - {max(total_pipeline_times):.1f}s")
    print()
    
    print(f"Total Export:")
    print(f"  Mean:   {statistics.mean(total_export_times):.1f}s")
    print(f"  Median: {statistics.median(total_export_times):.1f}s")
    print()
    
    # Export breakdown
    mean_export = statistics.mean(total_export_times)
    mean_master = statistics.mean(export_master_times)
    mean_upscaled = statistics.mean(export_upscaled_times)
    mean_marketing = statistics.mean(export_marketing_times)
    
    tiff_critical = mean_master + mean_upscaled
    
    print(f"Export Breakdown:")
    print(f"  Master TIFF:   {mean_master:.1f}s ({100*mean_master/mean_export:.1f}%)")
    print(f"  Upscaled TIFF: {mean_upscaled:.1f}s ({100*mean_upscaled/mean_export:.1f}%)")
    print(f"  Marketing PNG: {mean_marketing:.1f}s ({100*mean_marketing/mean_export:.1f}%)")
    print(f"  TIFF Critical: {tiff_critical:.1f}s ({100*tiff_critical/mean_export:.1f}%)")
    print()


def analyze_complexity(reports: List[Dict[str, Any]]) -> None:
    """Analyze scene complexity distribution and autotune decisions."""
    complexity_values = []
    megapixels_values = []
    
    for report in reports:
        export_autotune = report.get("export_autotune", {})
        if not export_autotune.get("enabled"):
            continue
        
        image_stats = export_autotune.get("image_stats", {})
        complexity = image_stats.get("scene_complexity")
        megapixels = image_stats.get("megapixels")
        
        if complexity is not None:
            complexity_values.append(complexity)
        if megapixels is not None:
            megapixels_values.append(megapixels)
    
    if not complexity_values:
        print("SCENE COMPLEXITY")
        print("-" * 80)
        print("No complexity data available")
        print()
        return
    
    print("SCENE COMPLEXITY")
    print("-" * 80)
    print(f"Mean:   {statistics.mean(complexity_values):.3f}")
    print(f"Median: {statistics.median(complexity_values):.3f}")
    print(f"Range:  {min(complexity_values):.3f} - {max(complexity_values):.3f}")
    print()
    
    print("IMAGE SIZE")
    print("-" * 80)
    print(f"Mean:   {statistics.mean(megapixels_values):.1f} MP")
    print(f"Median: {statistics.median(megapixels_values):.1f} MP")
    print(f"Range:  {min(megapixels_values):.1f} - {max(megapixels_values):.1f} MP")
    print()


def identify_anomalies(reports: List[Dict[str, Any]]) -> None:
    """Identify images that are significantly slower or have other anomalies."""
    total_pipeline_times = [r.get("timing_s", {}).get("total_pipeline", 0) for r in reports]
    mean_time = statistics.mean(total_pipeline_times)
    
    # Anomaly: >15% slower than mean
    threshold = mean_time * 1.15
    
    anomalies = []
    for report in reports:
        time = report.get("timing_s", {}).get("total_pipeline", 0)
        if time > threshold:
            anomalies.append((report["_path"], time, (time/mean_time - 1) * 100))
    
    if anomalies:
        print("⚠️  ANOMALIES DETECTED")
        print("-" * 80)
        print(f"Images >15% slower than mean ({mean_time:.1f}s):")
        print()
        for path, time, pct_slower in sorted(anomalies, key=lambda x: x[1], reverse=True):
            print(f"  {Path(path).parent.name}: {time:.1f}s (+{pct_slower:.1f}%)")
        print()
        print("ACTION: Review these images for complexity misclassification")
        print()
    else:
        print("✅ NO ANOMALIES")
        print("-" * 80)
        print(f"All images within 15% of mean ({mean_time:.1f}s)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze autotune production results")
    parser.add_argument("output_dir", type=Path, help="Output directory with reports")
    parser.add_argument("--baseline", type=Path, help="Baseline directory for comparison")
    
    args = parser.parse_args()
    
    if not args.output_dir.exists():
        print(f"Error: Directory not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)
    
    reports = load_reports(args.output_dir)
    
    analyze_timing(reports)
    analyze_complexity(reports)
    identify_anomalies(reports)
    
    # TODO: Baseline comparison if provided
    if args.baseline:
        print("BASELINE COMPARISON")
        print("-" * 80)
        print("Baseline comparison not implemented yet")
        print()


if __name__ == "__main__":
    main()
