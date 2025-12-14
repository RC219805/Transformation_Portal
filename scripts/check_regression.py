#!/usr/bin/env python3
"""
CI regression checker for water detection validation reports.

Usage:
    python scripts/check_regression.py \\
        --baseline data/water_v0/baseline_v0.json \\
        --current water_validation_ci.json \\
        --mode warning

Exit codes:
    0: No regression (or warning mode)
    1: Regression detected (error mode only)
"""

import argparse
import json
import sys
from pathlib import Path


def check_regression(baseline_path: Path, current_path: Path, mode: str) -> bool:
    """
    Check for quality regression between baseline and current validation reports.
    
    Args:
        baseline_path: Path to baseline validation report (JSON)
        current_path: Path to current validation report (JSON)
        mode: "warning" (print warnings, exit 0) or "error" (print errors, exit 1)
    
    Returns:
        True if no regression, False if regression detected
    """
    # Load reports
    with open(baseline_path) as f:
        baseline = json.load(f)
    
    with open(current_path) as f:
        current = json.load(f)
    
    baseline_summary = baseline["summary"]
    current_summary = current["summary"]
    
    warnings = []
    
    # Constants
    EPSILON = 1e-6
    ABSOLUTE_DRIFT_THRESHOLD = 0.05
    
    # 1. Recall drops >10% (absolute)
    pool_recall_drop = baseline_summary["pool_recall"] - current_summary["pool_recall"]
    if pool_recall_drop > 0.10:
        warnings.append(
            f"Pool recall dropped {pool_recall_drop:.1%} "
            f"({baseline_summary['pool_recall']:.1%} → {current_summary['pool_recall']:.1%})"
        )
    
    ocean_recall_drop = baseline_summary["ocean_recall"] - current_summary["ocean_recall"]
    if ocean_recall_drop > 0.10:
        warnings.append(
            f"Ocean recall dropped {ocean_recall_drop:.1%} "
            f"({baseline_summary['ocean_recall']:.1%} → {current_summary['ocean_recall']:.1%})"
        )
    
    # 2. Edge alignment drops >0.1 (absolute)
    pool_edge_drop = (
        baseline_summary["pool_avg_edge_alignment"] - 
        current_summary["pool_avg_edge_alignment"]
    )
    if pool_edge_drop > 0.1:
        warnings.append(
            f"Pool edge alignment dropped {pool_edge_drop:.2f} "
            f"({baseline_summary['pool_avg_edge_alignment']:.2f} → "
            f"{current_summary['pool_avg_edge_alignment']:.2f})"
        )
    
    ocean_edge_drop = (
        baseline_summary["ocean_avg_edge_alignment"] - 
        current_summary["ocean_avg_edge_alignment"]
    )
    if ocean_edge_drop > 0.1:
        warnings.append(
            f"Ocean edge alignment dropped {ocean_edge_drop:.2f} "
            f"({baseline_summary['ocean_avg_edge_alignment']:.2f} → "
            f"{current_summary['ocean_avg_edge_alignment']:.2f})"
        )
    
    # 3. Coverage drift (median changed by >2x or <0.5x, with epsilon guard)
    pool_baseline_median = baseline_summary.get("pool_median_coverage", 0.0)
    pool_current_median = current_summary.get("pool_median_coverage", 0.0)
    
    if pool_baseline_median < EPSILON:
        # Baseline near zero: use absolute drift check
        if pool_current_median > ABSOLUTE_DRIFT_THRESHOLD:
            warnings.append(
                f"Pool median coverage jumped from ~0 to {pool_current_median:.2%}"
            )
    else:
        # Normal case: ratio test
        pool_cov_ratio = pool_current_median / pool_baseline_median
        if pool_cov_ratio > 2.0:
            warnings.append(
                f"Pool median coverage increased {pool_cov_ratio:.2f}x "
                f"({pool_baseline_median:.2%} → {pool_current_median:.2%})"
            )
        elif pool_cov_ratio < 0.5:
            warnings.append(
                f"Pool median coverage decreased {1/pool_cov_ratio:.2f}x "
                f"({pool_baseline_median:.2%} → {pool_current_median:.2%})"
            )
    
    ocean_baseline_median = baseline_summary.get("ocean_median_coverage", 0.0)
    ocean_current_median = current_summary.get("ocean_median_coverage", 0.0)
    
    if ocean_baseline_median < EPSILON:
        if ocean_current_median > ABSOLUTE_DRIFT_THRESHOLD:
            warnings.append(
                f"Ocean median coverage jumped from ~0 to {ocean_current_median:.2%}"
            )
    else:
        ocean_cov_ratio = ocean_current_median / ocean_baseline_median
        if ocean_cov_ratio > 2.0:
            warnings.append(
                f"Ocean median coverage increased {ocean_cov_ratio:.2f}x "
                f"({ocean_baseline_median:.2%} → {ocean_current_median:.2%})"
            )
        elif ocean_cov_ratio < 0.5:
            warnings.append(
                f"Ocean median coverage decreased {1/ocean_cov_ratio:.2f}x "
                f"({ocean_baseline_median:.2%} → {ocean_current_median:.2%})"
            )
    
    # 4. False trigger rate increased (absolute delta >15%)
    baseline_ft = baseline_summary.get("false_trigger_rate", 0.0)
    current_ft = current_summary.get("false_trigger_rate", 0.0)
    ft_increase = current_ft - baseline_ft
    
    if ft_increase > 0.15:  # Absolute: +0.15 (e.g., 0.05 → 0.20)
        warnings.append(
            f"False trigger rate increased by {ft_increase:.1%} "
            f"({baseline_ft:.1%} → {current_ft:.1%})"
        )
    
    # Report results
    if warnings:
        prefix = "❌ REGRESSION DETECTED" if mode == "error" else "⚠️  WARNING: Regression detected"
        print(f"\n{prefix}")
        print(f"Baseline: {baseline_path}")
        print(f"Current:  {current_path}\n")
        
        for i, w in enumerate(warnings, 1):
            print(f"  {i}. {w}")
        
        print()
        
        if mode == "error":
            print("❌ Build failed due to quality regression")
            return False
        else:
            print("⚠️  Warning mode: build continues despite regression")
            return True
    else:
        print("✅ No regression detected")
        print(f"Baseline: {baseline_path}")
        print(f"Current:  {current_path}")
        return True


def main():
    parser = argparse.ArgumentParser(
        description="Check for quality regression in water detection validation reports"
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        required=True,
        help="Path to baseline validation report (JSON)"
    )
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current validation report (JSON)"
    )
    parser.add_argument(
        "--mode",
        choices=["warning", "error"],
        default="warning",
        help="Failure mode: 'warning' (exit 0) or 'error' (exit 1)"
    )
    
    args = parser.parse_args()
    
    # Validate files exist
    if not args.baseline.exists():
        print(f"❌ Baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    
    if not args.current.exists():
        print(f"❌ Current file not found: {args.current}", file=sys.stderr)
        sys.exit(1)
    
    # Run regression check
    no_regression = check_regression(args.baseline, args.current, args.mode)
    
    if not no_regression and args.mode == "error":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
