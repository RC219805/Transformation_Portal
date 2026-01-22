#!/usr/bin/env python3
"""
Throughput validation script for CI regression detection.

Compares current throughput benchmark results against baseline thresholds.
Fails CI if throughput degrades beyond acceptable limits.

Usage:
    python scripts/validate_throughput.py \\
        --baseline bench/baselines/throughput_baseline.json \\
        --current throughput_results.json \\
        --max-regression 20

Exit codes:
    0: No regression (meets baseline)
    1: Regression detected (fails baseline)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def validate_standard_quality(
    current: Dict[str, Any], baseline: Dict[str, Any], max_regression_pct: float
) -> Tuple[bool, List[str]]:
    """Validate standard quality throughput.

    Args:
        current: Current benchmark results
        baseline: Baseline thresholds
        max_regression_pct: Maximum allowed regression percentage

    Returns:
        (passed, warnings) tuple
    """
    passed = True
    warnings = []

    baseline_config = baseline["baselines"]["standard_quality_cpu"]

    # Extract metrics from current results
    # Current format: direct metrics dict or pytest-benchmark format
    if "images_per_hour" in current:
        throughput = current["images_per_hour"]
        memory_mb = current.get("rss_final_mb", current.get("memory_peak_mb", 0))
    else:
        # Fallback for different formats
        warnings.append("⚠️  Could not parse throughput from current results")
        return False, warnings

    # Check throughput
    min_throughput = baseline_config["min_images_per_hour"]
    if throughput < min_throughput:
        passed = False
        warnings.append(f"❌ Throughput {throughput:.1f} images/hour below baseline {min_throughput} images/hour")
    else:
        warnings.append(f"✅ Throughput {throughput:.1f} images/hour meets baseline ({min_throughput} images/hour)")

    # Check memory
    max_memory = baseline_config["max_memory_mb"]
    if memory_mb > max_memory:
        passed = False
        warnings.append(f"❌ Memory {memory_mb:.1f}MB exceeds baseline {max_memory}MB")
    else:
        warnings.append(f"✅ Memory {memory_mb:.1f}MB within baseline ({max_memory}MB)")

    return passed, warnings


def validate_max_quality(
    current: Dict[str, Any],
    baseline: Dict[str, Any],
    max_regression_pct: float,
    has_gpu: bool = False,
) -> Tuple[bool, List[str]]:
    """Validate max quality throughput.

    Args:
        current: Current benchmark results
        baseline: Baseline thresholds
        max_regression_pct: Maximum allowed regression percentage
        has_gpu: Whether GPU is available

    Returns:
        (passed, warnings) tuple
    """
    passed = True
    warnings = []

    baseline_key = "max_quality_gpu" if has_gpu else "max_quality_cpu"
    baseline_config = baseline["baselines"][baseline_key]

    # Extract metrics
    if "images_per_hour" in current:
        throughput = current["images_per_hour"]
        memory_mb = current.get("rss_final_mb", current.get("memory_peak_mb", 0))
    else:
        warnings.append("⚠️  Could not parse throughput from current results")
        return False, warnings

    # Check throughput
    min_throughput = baseline_config["min_images_per_hour"]
    if throughput < min_throughput:
        passed = False
        warnings.append(
            f"❌ Throughput {throughput:.1f} images/hour below baseline "
            f"{min_throughput} images/hour ({'GPU' if has_gpu else 'CPU'} mode)"
        )
    else:
        warnings.append(
            f"✅ Throughput {throughput:.1f} images/hour meets baseline "
            f"({min_throughput} images/hour, {'GPU' if has_gpu else 'CPU'} mode)"
        )

    # Check memory
    max_memory = baseline_config["max_memory_mb"]
    if memory_mb > max_memory:
        passed = False
        warnings.append(f"❌ Memory {memory_mb:.1f}MB exceeds baseline {max_memory}MB")
    else:
        warnings.append(f"✅ Memory {memory_mb:.1f}MB within baseline ({max_memory}MB)")

    return passed, warnings


def compare_against_production_targets(current: Dict[str, Any], baseline: Dict[str, Any]) -> List[str]:
    """Compare against production targets (informational only).

    Returns list of informational messages.
    """
    messages = []

    targets = baseline.get("production_targets", {})
    if not targets:
        return messages

    if "images_per_hour" in current:
        throughput = current["images_per_hour"]

        # Compare to CPU target
        cpu_target = targets.get("cpu_standard", {}).get("target_images_per_hour", 127)
        if throughput >= cpu_target:
            messages.append(f"🎯 Meets CPU production target: {throughput:.1f} >= {cpu_target} images/hour")
        else:
            pct_of_target = (throughput / cpu_target) * 100
            messages.append(f"ℹ️  {pct_of_target:.1f}% of CPU production target ({throughput:.1f}/{cpu_target} images/hour)")

        # Compare to GPU target (aspirational)
        gpu_target = targets.get("gpu_max", {}).get("target_images_per_hour", 400)
        if throughput >= gpu_target:
            messages.append(f"🎯 Meets GPU production target: {throughput:.1f} >= {gpu_target} images/hour!")
        else:
            pct_of_target = (throughput / gpu_target) * 100
            messages.append(f"ℹ️  {pct_of_target:.1f}% of GPU production target ({throughput:.1f}/{gpu_target} images/hour)")

    return messages


def main():
    parser = argparse.ArgumentParser(description="Validate throughput benchmarks against baseline")
    parser.add_argument("--baseline", type=Path, required=True, help="Path to baseline throughput JSON")
    parser.add_argument(
        "--current",
        type=Path,
        required=True,
        help="Path to current benchmark results JSON",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=20,
        help="Maximum allowed throughput regression percentage (default: 20)",
    )
    parser.add_argument(
        "--quality",
        choices=["standard", "max"],
        default="standard",
        help="Quality level to validate (default: standard)",
    )
    parser.add_argument("--gpu", action="store_true", help="Validate for GPU configuration")

    args = parser.parse_args()

    # Load files
    try:
        baseline = load_json(args.baseline)
        current = load_json(args.current)
    except FileNotFoundError as e:
        print(f"❌ File not found: {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
        sys.exit(1)

    print("=" * 80)
    print("🔍 Throughput Validation Report")
    print("=" * 80)
    print(f"Baseline: {args.baseline}")
    print(f"Current:  {args.current}")
    print(f"Quality:  {args.quality}")
    print(f"Mode:     {'GPU' if args.gpu else 'CPU'}")
    print(f"Max Regression: {args.max_regression}%")
    print("=" * 80)

    # Validate based on quality level
    if args.quality == "standard":
        passed, warnings = validate_standard_quality(current, baseline, args.max_regression)
    else:  # max
        passed, warnings = validate_max_quality(current, baseline, args.max_regression, args.gpu)

    # Print validation results
    print("\n📊 Validation Results:")
    for warning in warnings:
        print(f"  {warning}")

    # Print production target comparison (informational)
    prod_messages = compare_against_production_targets(current, baseline)
    if prod_messages:
        print("\n🎯 Production Target Comparison:")
        for msg in prod_messages:
            print(f"  {msg}")

    # Exit with appropriate code
    print("\n" + "=" * 80)
    if passed:
        print("✅ PASS: Throughput meets baseline requirements")
        print("=" * 80)
        sys.exit(0)
    else:
        print("❌ FAIL: Throughput regression detected")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
