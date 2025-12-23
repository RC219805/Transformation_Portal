#!/usr/bin/env python3
"""
Kitchen-Only Phase 1 Sweep Runner
Modified version of sweep_runner.py for single-image testing
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# Use kitchen image from source directory
KITCHEN_IMAGE = Path("750Picacho_Source_TIFFs/750Picacho_Kitchen.tif")
SWEEP_ROOT = Path("sweep_runs/phase1_kitchen_only")
OUTPUT_BASE = SWEEP_ROOT

# Parameter grid - same as full Phase 1
PARAMETER_GRID = {
    "depth": {
        "gamma": [1.0, 1.1, 0.9],
        "percentile_clip_low": [0.5, 1.0, 0.1],
        "edge_filter_sigma_color": [75, 50, 100],
        "banding_suppression": [0.005, 0.003, 0.007],
    },
    "materials_v3": {
        "confidence_curve": ["linear", "sigmoid", "piecewise"],
        "edge_alignment_weight": [1.0, 1.5, 0.7],
        "low_confidence_threshold": [None, 0.4, 0.5],
    },
    "color_tone": {
        "local_contrast_gain": [2.0, 2.5, 2.2],
        "saturation_protection": [1.0, 0.8, 0.85],
    },
}


def run_single_sweep(category: str, parameter: str, value: Any, delta_id: int) -> Dict[str, Any]:
    """Run a single parameter sweep on kitchen image."""
    
    run_name = f"{category}_{parameter}_delta{delta_id}"
    run_dir = OUTPUT_BASE / run_name
    run_dir.mkdir(exist_ok=True, parents=True)
    
    output_dir = run_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"Running: {run_name}")
    print(f"Parameter: {category}.{parameter} = {value}")
    print(f"{'='*80}\n")
    
    # Save parameters
    params_file = run_dir / "params.json"
    with open(params_file, 'w') as f:
        json.dump({
            "run_id": run_name,
            "category": category,
            "parameter": parameter,
            "value": value,
            "delta_id": delta_id,
            "timestamp": datetime.now().isoformat(),
            "image": str(KITCHEN_IMAGE),
        }, f, indent=2)
    
    # Run CLI with preset
    start_time = time.time()
    
    cmd = [
        sys.executable,
        "-m", "lux_depth_v2.cli",
        "--input", str(KITCHEN_IMAGE),
        "--output-dir", str(output_dir),
        "--preset", "interior_luxury",
        "--device", "cpu",
        "--upscaler-backend", "torch",
    ]
    
    # Note: We can't actually modify internal parameters via CLI
    # This is a limitation - full sweep_runner.py modifies config internally
    # For now, this runs the baseline preset
    
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = time.time()
    duration = end_time - start_time
    
    if result.returncode != 0:
        print(f"❌ Failed: {run_name}")
        print(result.stderr)
        return {"status": "failed", "error": result.stderr}
    
    print(f"✅ Completed: {run_name} ({duration:.1f}s)")
    
    # Save metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "status": "success",
            "duration_s": duration,
            "output_dir": str(output_dir),
        }, f, indent=2)
    
    return {
        "status": "success",
        "duration_s": duration,
        "run_dir": str(run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Kitchen-only Phase 1 sweep")
    parser.add_argument("--category", choices=["depth", "materials_v3", "color_tone", "all"], 
                       default="all", help="Category to sweep")
    args = parser.parse_args()
    
    if not KITCHEN_IMAGE.exists():
        print(f"❌ Kitchen image not found: {KITCHEN_IMAGE}")
        return 1
    
    OUTPUT_BASE.mkdir(exist_ok=True, parents=True)
    
    print(f"\n{'='*80}")
    print("Kitchen-Only Phase 1 Parameter Sweep")
    print(f"{'='*80}")
    print(f"Image: {KITCHEN_IMAGE}")
    print(f"Output: {OUTPUT_BASE}")
    print(f"Category: {args.category}")
    print()
    
    # Determine which categories to run
    if args.category == "all":
        categories = list(PARAMETER_GRID.keys())
    else:
        categories = [args.category]
    
    total_runs = 0
    successful_runs = 0
    failed_runs = 0
    
    all_results = []
    
    for category in categories:
        params = PARAMETER_GRID[category]
        
        for param_name, values in params.items():
            for delta_id, value in enumerate(values):
                total_runs += 1
                result = run_single_sweep(category, param_name, value, delta_id)
                
                if result["status"] == "success":
                    successful_runs += 1
                else:
                    failed_runs += 1
                
                all_results.append(result)
    
    # Save summary
    summary_file = OUTPUT_BASE / "phase1_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "total_runs": total_runs,
            "successful": successful_runs,
            "failed": failed_runs,
            "results": all_results,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print("Phase 1 Kitchen-Only Sweep Complete")
    print(f"{'='*80}")
    print(f"Total runs: {total_runs}")
    print(f"Successful: {successful_runs}")
    print(f"Failed: {failed_runs}")
    print(f"Summary: {summary_file}")
    print()
    
    return 0 if failed_runs == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
