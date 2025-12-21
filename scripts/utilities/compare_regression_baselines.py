#!/usr/bin/env python3
"""Compare test outputs against regression baselines.

This script compares current test results against saved baselines
to detect regressions in the MaterialsV3 pipeline.

Usage:
    python compare_regression_baselines.py [--baseline BASELINE] [--current CURRENT]

Examples:
    # Compare with default paths
    python compare_regression_baselines.py

    # Compare with custom paths
    python compare_regression_baselines.py \
        --baseline regression_baselines/phase1/edge_cases.json \
        --current test-results/current/edge_cases.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any


def load_json(path: Path) -> Dict[str, Any]:
    """Load JSON file and return data."""
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ File not found: {path}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {path}: {e}")
        sys.exit(1)


def compare_baselines(baseline_path: Path, current_path: Path) -> bool:
    """Compare current test results against baseline.
    
    Args:
        baseline_path: Path to baseline JSON file
        current_path: Path to current results JSON file
        
    Returns:
        True if comparison passed, False otherwise
    """
    print(f"Comparing baseline: {baseline_path}")
    print(f"Against current:    {current_path}")
    print()
    
    baseline = load_json(baseline_path)
    current = load_json(current_path)
    
    passed = True
    
    # Compare test counts
    if 'tests' in baseline and 'tests' in current:
        baseline_total = baseline['tests'].get('total', 0)
        current_total = current['tests'].get('total', 0)
        
        if baseline_total != current_total:
            print(f"❌ Test count mismatch: {baseline_total} → {current_total}")
            passed = False
        else:
            print(f"✅ Test count consistent: {current_total}")
        
        # Compare pass rates
        baseline_passed = baseline['tests'].get('passed', 0)
        current_passed = current['tests'].get('passed', 0)
        
        if baseline_passed != current_passed:
            print(f"⚠️  Pass count changed: {baseline_passed} → {current_passed}")
            if current_passed < baseline_passed:
                print("   WARNING: Pass count decreased - possible regression!")
                passed = False
        else:
            print(f"✅ Pass count consistent: {current_passed}")
        
        # Compare failure rates
        baseline_failed = baseline['tests'].get('failed', 0)
        current_failed = current['tests'].get('failed', 0)
        
        if current_failed > baseline_failed:
            print(f"❌ Failure count increased: {baseline_failed} → {current_failed}")
            passed = False
        elif current_failed < baseline_failed:
            print(f"✅ Failure count improved: {baseline_failed} → {current_failed}")
        else:
            print(f"✅ Failure count consistent: {current_failed}")
    
    # Compare execution time (if available)
    if 'duration' in baseline and 'duration' in current:
        baseline_duration = baseline['duration']
        current_duration = current['duration']
        
        duration_diff = current_duration - baseline_duration
        duration_pct = (duration_diff / baseline_duration) * 100 if baseline_duration > 0 else 0
        
        print(f"\n⏱️  Execution time: {baseline_duration:.2f}s → {current_duration:.2f}s")
        
        if abs(duration_pct) > 20:
            if duration_pct > 0:
                print(f"   ⚠️  Execution time increased by {duration_pct:.1f}%")
            else:
                print(f"   ✅ Execution time improved by {abs(duration_pct):.1f}%")
    
    print()
    if passed:
        print("✅ Regression comparison PASSED")
        return True
    else:
        print("❌ Regression comparison FAILED")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare test outputs against regression baselines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--baseline',
        type=Path,
        default=Path('regression_baselines/phase1/edge_cases.json'),
        help='Path to baseline JSON file (default: regression_baselines/phase1/edge_cases.json)'
    )
    
    parser.add_argument(
        '--current',
        type=Path,
        default=Path('test-results/current/edge_cases.json'),
        help='Path to current results JSON file (default: test-results/current/edge_cases.json)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Verify baseline exists
    if not args.baseline.exists():
        print(f"❌ Baseline file not found: {args.baseline}")
        print("\nTo create a baseline, run:")
        print("  pytest tests/test_materials_v3_edge_cases.py --json-report \\")
        print(f"    --json-report-file={args.baseline}")
        sys.exit(1)
    
    # Verify current results exist
    if not args.current.exists():
        print(f"❌ Current results file not found: {args.current}")
        print("\nTo generate current results, run:")
        print("  pytest tests/test_materials_v3_edge_cases.py --json-report \\")
        print(f"    --json-report-file={args.current}")
        sys.exit(1)
    
    # Run comparison
    if compare_baselines(args.baseline, args.current):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
