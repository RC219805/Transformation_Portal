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
    1 - Budget violations detected (CI should fail)
    2 - Configuration / usage / input error (missing deps, unreadable/malformed files)
"""
import argparse
import fnmatch
import json
import math
import sys
from pathlib import Path
from typing import NoReturn

# Exit codes
EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_ERROR = 2

try:
    import yaml
except ImportError as e:
    print(f"ERROR: Missing PyYAML ({e}). Install with: pip install pyyaml", file=sys.stderr)
    raise SystemExit(EXIT_ERROR)


def die(msg: str, code: int = EXIT_ERROR) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _to_finite_float(x: object, ctx: str) -> float:
    """Convert value to finite float, rejecting NaN/inf."""
    try:
        v = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{ctx} must be numeric, got {x!r}") from e
    if not math.isfinite(v):
        raise ValueError(f"{ctx} must be finite, got {v!r}")
    return v


def load_yaml(p: Path) -> dict:
    """Load YAML configuration file."""
    try:
        return yaml.safe_load(p.read_text())
    except FileNotFoundError:
        die(f"YAML config file not found: {p}")
    except PermissionError:
        die(f"Permission denied reading YAML config file: {p}")
    except OSError as e:
        die(f"Failed to read YAML config file {p}: {e}")
    except yaml.YAMLError as e:
        die(f"Failed to parse YAML config file {p}: {e}")


def load_json(p: Path) -> dict:
    """Load JSON benchmark results."""
    try:
        return json.loads(p.read_text())
    except FileNotFoundError:
        die(f"JSON file not found: {p}")
    except PermissionError:
        die(f"Permission denied reading JSON file: {p}")
    except OSError as e:
        die(f"Failed to read JSON file {p}: {e}")
    except json.JSONDecodeError as e:
        die(f"Failed to parse JSON file {p}: {e}")


def bench_index(bench_json: dict) -> dict:
    """Build index of benchmarks by name.

    pytest-benchmark JSON structure:
    {"benchmarks": [{"name": ..., "fullname": ..., "stats": {...}}, ...]}
    """
    benchmarks = bench_json.get("benchmarks", [])
    if not isinstance(benchmarks, list):
        die(f"Expected 'benchmarks' to be a list, got {type(benchmarks).__name__}")

    idx = {}
    for i, b in enumerate(benchmarks):
        if not isinstance(b, dict):
            print(f"WARNING: Skipping non-dict benchmark entry at index {i}", file=sys.stderr)
            continue
        # Use fullname for matching (includes module::class::test)
        key = b.get("fullname", b.get("name"))
        if not key:
            print(f"WARNING: Skipping benchmark at index {i} (no name/fullname)", file=sys.stderr)
            continue
        if key in idx:
            print(f"WARNING: Duplicate benchmark key '{key}' (overwriting)", file=sys.stderr)
        idx[key] = b
    return idx


def get_latency_s(b: dict, metric: str) -> float:
    """Extract latency in seconds from benchmark stats (rejects NaN/inf)."""
    stats = b.get("stats", {})
    if not isinstance(stats, dict):
        raise TypeError("benchmark 'stats' must be a mapping")
    if metric not in stats:
        # fallback: mean if median missing, etc.
        for k in ("median", "mean", "min", "max"):
            if k in stats:
                return _to_finite_float(stats[k], f"benchmark stats[{k}]")
        raise KeyError(f"No stats metric found in benchmark: {b.get('name')}")
    return _to_finite_float(stats[metric], f"benchmark stats[{metric}]")


def match_names(all_names: list[str], patterns: list[str]) -> list[str]:
    """Match benchmark names against glob patterns (case-sensitive)."""
    matched = []
    for pat in patterns:
        matched.extend([n for n in all_names if fnmatch.fnmatchcase(n, pat)])
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
    if budgets_doc is None or not isinstance(budgets_doc, dict):
        die(f"YAML root must be a mapping (dict), got {type(budgets_doc).__name__}")

    bench_doc = load_json(Path(args.bench_json))
    if not isinstance(bench_doc, dict):
        die(f"Benchmark JSON root must be a dict, got {type(bench_doc).__name__}")

    bench_by_name = bench_index(bench_doc)
    all_names = list(bench_by_name.keys())

    if not all_names:
        die("No benchmarks found in pytest-benchmark JSON output.", EXIT_ERROR)

    # Extract settings
    settings = budgets_doc.get("settings", {})
    if not isinstance(settings, dict):
        die(f"'settings' must be a dict, got {type(settings).__name__}")

    raw_max_reg = settings.get("max_regression_percent", None)
    # Note: max_regression_percent <= 0 disables baseline regression checks
    if raw_max_reg is None:
        max_reg_pct = 0.0
        max_reg = None
    else:
        max_reg_pct = _to_finite_float(raw_max_reg, "max_regression_percent")
        max_reg = (max_reg_pct / 100.0) if max_reg_pct > 0 else None
    fail_on_unmatched = bool(settings.get("fail_on_unmatched_patterns", False))  # default tolerant

    budget_groups = budgets_doc.get("budgets", {})
    if not isinstance(budget_groups, dict):
        die(f"'budgets' must be a dict, got {type(budget_groups).__name__}")

    bench_map = budgets_doc.get("benchmark_map", {})
    if not isinstance(bench_map, dict):
        die(f"'benchmark_map' must be a dict, got {type(bench_map).__name__}")

    # Load baseline if provided
    baseline_by_name = None
    if args.baseline_json:
        base_doc = load_json(Path(args.baseline_json))
        if not isinstance(base_doc, dict):
            die(f"Baseline benchmark JSON root must be a dict, got {type(base_doc).__name__}")
        baseline_by_name = bench_index(base_doc)

    violations = []
    warnings = []

    # Validate each budget group
    for group, cfg in budget_groups.items():
        if not isinstance(cfg, dict):
            die(f"Budget entry [{group}] must be a dict, got {type(cfg).__name__}")
        # Only validate groups that have max_latency_s (skip throughput section etc.)
        max_latency = cfg.get("max_latency_s", None)
        if max_latency is None:
            continue

        max_latency_f = _to_finite_float(max_latency, f"[{group}] max_latency_s")

        # Get benchmark patterns for this group
        patterns = bench_map.get(group)
        if not patterns:
            warnings.append(f"⚠️  No benchmark_map entry for '{group}' (skipping).")
            continue

        # Normalize patterns: string → [string]
        if isinstance(patterns, str):
            patterns = [patterns]
        if not isinstance(patterns, list) or not all(isinstance(p, str) for p in patterns):
            die(f"benchmark_map['{group}'] must be a string or list of strings, got {patterns!r}")

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
            try:
                t = get_latency_s(b, args.metric)
            except (KeyError, TypeError, ValueError) as e:
                die(f"Invalid benchmark stats for '{name}' (metric={args.metric}): {e}")

            # Check budget threshold
            if t > max_latency_f:
                msg = f"exceeds max_latency_s={max_latency_f}s"
                if max_latency != max_latency_f:
                    msg += f" (configured: {max_latency!r})"
                violations.append((group, name, f"{t:.6f}s", msg))

            # Check regression vs baseline (if provided)
            if baseline_by_name and max_reg is not None:
                if name in baseline_by_name:
                    try:
                        t0 = get_latency_s(baseline_by_name[name], args.metric)
                    except (KeyError, TypeError, ValueError) as e:
                        warnings.append(f"⚠️  Baseline benchmark '{name}' has invalid stats: {e} (no regression check).")
                        continue
                    if t0 <= 0:
                        warnings.append(f"⚠️  Baseline metric is 0 for '{name}' (no regression check).")
                        continue
                    if t > t0 * (1 + max_reg):
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
        print(w, file=sys.stderr)

    # Report violations
    if violations:
        print("\n❌ PERFORMANCE BUDGET VIOLATIONS:")
        for group, name, val, why in violations:
            print(f"  [{group}] {name}")
            print(f"    {val} -> {why}")
        return EXIT_VIOLATIONS

    print("✅ OK: Performance budgets satisfied.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
