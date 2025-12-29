#!/usr/bin/env python3
"""Validate pytest-benchmark results against performance budgets.

Usage:
    python scripts/validate_performance_budgets.py \
        --budgets bench/config/performance_budgets.yaml \
        --bench-json test-results/benchmark-perf.json \
        [--baseline-json bench/baselines/perf_bench_baseline.json] \
        [--metric median]

Exit codes:
    0 - All budgets satisfied
    2 - Budget violations detected (CI should fail)
"""
import argparse
import fnmatch
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError as e:
    print("ERROR: Missing PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    raise


def load_yaml(p: Path) -> dict:
    """Load YAML configuration file."""
    return yaml.safe_load(p.read_text())


def load_json(p: Path) -> dict:
    """Load JSON benchmark results."""
    return json.loads(p.read_text())


def bench_index(bench_json: dict) -> dict:
    """Build index of benchmarks by name.
    
    pytest-benchmark JSON structure:
    {"benchmarks": [{"name": ..., "fullname": ..., "stats": {...}}, ...]}
    """
    idx = {}
    for b in bench_json.get("benchmarks", []):
        # Use fullname for matching (includes module::class::test)
        key = b.get("fullname", b.get("name"))
        idx[key] = b
    return idx


def get_latency_s(b: dict, metric: str) -> float:
    """Extract latency in seconds from benchmark stats."""
    stats = b.get("stats", {})
    if metric not in stats:
        # fallback: mean if median missing, etc.
        for k in ("median", "mean", "min", "max"):
            if k in stats:
                return float(stats[k])
        raise KeyError(f"No stats metric found in benchmark: {b.get('name')}")
    return float(stats[metric])


def match_names(all_names: list[str], patterns: list[str]) -> list[str]:
    """Match benchmark names against glob patterns."""
    matched = []
    for pat in patterns:
        matched.extend([n for n in all_names if fnmatch.fnmatch(n, pat)])
    # de-dupe preserve order
    seen = set()
    out = []
    for n in matched:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate pytest-benchmark results against performance budgets"
    )
    ap.add_argument("--budgets", required=True, help="Path to performance_budgets.yaml")
    ap.add_argument("--bench-json", required=True, help="Path to pytest-benchmark JSON output")
    ap.add_argument(
        "--baseline-json",
        default=None,
        help="Optional baseline benchmark JSON for regression checks"
    )
    ap.add_argument(
        "--metric",
        default="median",
        choices=["median", "mean", "min", "max"],
        help="Metric to use for validation (default: median)"
    )
    args = ap.parse_args()

    # Load configuration and results
    budgets_doc = load_yaml(Path(args.budgets))
    bench_doc = load_json(Path(args.bench_json))
    bench_by_name = bench_index(bench_doc)
    all_names = list(bench_by_name.keys())

    # Extract settings
    settings = budgets_doc.get("settings", {})
    max_reg_pct = float(settings.get("max_regression_percent", 0))
    max_reg = max_reg_pct / 100.0 if max_reg_pct else None
    fail_on_unmatched = bool(settings.get("fail_on_unmatched_patterns", False))  # default tolerant

    budget_groups = budgets_doc.get("budgets", {})
    bench_map = budgets_doc.get("benchmark_map", {})

    # Load baseline if provided
    baseline_by_name = None
    if args.baseline_json:
        base_doc = load_json(Path(args.baseline_json))
        baseline_by_name = bench_index(base_doc)

    violations = []
    warnings = []

    # Validate each budget group
    for group, cfg in budget_groups.items():
        # Only validate groups that have max_latency_s (skip throughput section etc.)
        max_latency = cfg.get("max_latency_s", None)
        if max_latency is None:
            continue

        # Get benchmark patterns for this group
        patterns = bench_map.get(group)
        if not patterns:
            warnings.append(f"⚠️  No benchmark_map entry for '{group}' (skipping).")
            continue

        # Match benchmarks to this group
        matched = match_names(all_names, patterns)
        if not matched:
            msg = f"No benchmarks matched patterns {patterns}"
            if fail_on_unmatched:
                violations.append((group, "-", "-", msg))
            else:
                warnings.append(f"⚠️  [{group}] {msg} (fail_on_unmatched_patterns=false)")
            continue

        # Validate each matched benchmark
        for name in matched:
            b = bench_by_name[name]
            t = get_latency_s(b, args.metric)

            # Check budget threshold
            if t > float(max_latency):
                violations.append(
                    (group, name, f"{t:.6f}s", f"exceeds max_latency_s={max_latency}s")
                )

            # Check regression vs baseline (if provided)
            if baseline_by_name and max_reg is not None:
                if name in baseline_by_name:
                    t0 = get_latency_s(baseline_by_name[name], args.metric)
                    if t0 > 0 and t > t0 * (1 + max_reg):
                        regression_pct = (t / t0 - 1) * 100
                        violations.append(
                            (
                                group,
                                name,
                                f"{t:.6f}s",
                                f"regressed {regression_pct:.1f}% > {max_reg_pct:.1f}%"
                            )
                        )
                else:
                    warnings.append(
                        f"⚠️  Baseline missing benchmark '{name}' (no regression check)."
                    )

    # Print warnings
    for w in warnings:
        print(w)

    # Report violations
    if violations:
        print("\n❌ PERFORMANCE BUDGET VIOLATIONS:")
        for group, name, val, why in violations:
            print(f"  [{group}] {name}")
            print(f"    {val} -> {why}")
        return 2

    print("✅ OK: Performance budgets satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
