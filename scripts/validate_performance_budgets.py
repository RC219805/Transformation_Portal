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
import sys
from pathlib import Path
from typing import Any, NoReturn

EXIT_OK = 0
EXIT_VIOLATIONS = 1
EXIT_ERROR = 2


try:
    import yaml
except ImportError as e:
    print(
        f"ERROR: Missing PyYAML ({e}). Install with: pip install pyyaml",
        file=sys.stderr,
    )
    raise SystemExit(EXIT_ERROR)

def die(msg: str, code: int = EXIT_ERROR) -> NoReturn:
    print(f"ERROR: {msg}", file=sys.stderr)
    raise SystemExit(code)

def load_yaml(p: Path) -> dict | None:
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


def load_json(p: Path) -> Any:
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


def warn(msg: str) -> None:
    print(f"WARNING: {msg}", file=sys.stderr)


def _as_mapping(value: object, ctx: str) -> dict:
    """Validate value is a mapping. Empty/None becomes {} for tolerance."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        die(f"{ctx} must be a mapping (YAML dict), got: {type(value).__name__}", EXIT_ERROR)
    return value


def _as_list_of_str(value: object, ctx: str) -> list[str]:
    """Validate value is a list[str]. If given a single string, wrap it."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list) or not all(isinstance(p, str) for p in value):
        die(f"{ctx} must be a list of strings, got: {value!r}", EXIT_ERROR)
    return value


def bench_index(bench_json: dict) -> dict:
    """Build index of benchmarks by name.

    pytest-benchmark JSON structure:
    {"benchmarks": [{"name": ..., "fullname": ..., "stats": {...}}, ...]}
    """
    idx: dict[str, dict] = {}
    dropped = 0
    benches = bench_json.get("benchmarks", [])
    if not isinstance(benches, list):
        die(
            "pytest-benchmark JSON must contain 'benchmarks' as a list.",
            EXIT_ERROR,
        )
    for b in benches:
        if not isinstance(b, dict):
            dropped += 1
            continue
        # Use fullname for matching (includes module::class::test)
        key = b.get("fullname") or b.get("name")
        if not key:
            dropped += 1
            continue
        idx[key] = b
    if dropped:
        warn(f"Skipped {dropped} malformed benchmark entries (missing name/fullname or wrong type).")
    return idx


def get_latency_s(b: dict, metric: str) -> float:
    """Extract latency in seconds from benchmark stats."""
    stats = b.get("stats", {})
    if not isinstance(stats, dict):
        raise TypeError("benchmark 'stats' must be a mapping")
    if metric not in stats:
        # Fallback: use the first available metric in a stable preference order.
        for k in ("median", "mean", "min", "max"):
            if k in stats:
                return float(stats[k])
        raise KeyError(f"No stats metric found in benchmark: {b.get('name')}")
    return float(stats[metric])


def match_names(all_names: list[str], patterns: list[str]) -> list[str]:
    """Match benchmark names against glob patterns.
    
    Uses fnmatchcase for case-sensitive, platform-deterministic matching
    (no OS-specific case-folding surprises).
    """
    matched: list[str] = []
    for pat in patterns:
        matched.extend([n for n in all_names if fnmatch.fnmatchcase(n, pat)])

    # De-dupe while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for n in matched:
        if n not in seen:
            out.append(n)
            seen.add(n)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate pytest-benchmark results against performance budgets",
    )
    ap.add_argument("--budgets", required=True, help="Path to performance_budgets.yaml")
    ap.add_argument(
        "--bench-json",
        required=True,
        help="Path to pytest-benchmark JSON output",
    )
    ap.add_argument(
        "--baseline-json",
        default=None,
        help="Optional baseline benchmark JSON for regression checks",
    )
    ap.add_argument(
        "--metric",
        default="median",
        choices=["median", "mean", "min", "max"],
        help="Metric to use for validation (default: median)",
    )
    args = ap.parse_args()

    # Load configuration and results
    budgets_doc_raw = load_yaml(Path(args.budgets))
    budgets_doc = _as_mapping(budgets_doc_raw, "top-level YAML document")
    bench_doc_raw = load_json(Path(args.bench_json))
    if not isinstance(bench_doc_raw, dict):
        die(
            f"Top-level benchmark JSON must be an object, got: {type(bench_doc_raw).__name__}",
            EXIT_ERROR,
        )
    bench_doc = bench_doc_raw
    bench_by_name = bench_index(bench_doc)
    all_names = list(bench_by_name.keys())
    if not all_names:
        die("No benchmarks found in pytest-benchmark JSON output.", EXIT_ERROR)

    # Extract settings
    settings = _as_mapping(budgets_doc.get("settings", {}), "settings")
    raw_max_reg = settings.get("max_regression_percent")
    if raw_max_reg is None:
        max_reg_pct = 0.0
        max_reg = None
    else:
        try:
            max_reg_pct = float(raw_max_reg)
        except (TypeError, ValueError):
            die(
                f"settings.max_regression_percent must be numeric, got: {raw_max_reg!r}",
                EXIT_ERROR,
            )
        max_reg = (max_reg_pct / 100.0) if max_reg_pct > 0 else None

    fail_on_unmatched = bool(settings.get("fail_on_unmatched_patterns", False))

    budget_groups = _as_mapping(budgets_doc.get("budgets", {}), "budgets")
    bench_map = _as_mapping(budgets_doc.get("benchmark_map", {}), "benchmark_map")

    # Load baseline if provided
    baseline_by_name = None
    if args.baseline_json:
        base_doc_raw = load_json(Path(args.baseline_json))
        if not isinstance(base_doc_raw, dict):
            die(
                f"Baseline benchmark JSON must be an object, got: {type(base_doc_raw).__name__}",
                EXIT_ERROR,
            )
        base_doc = base_doc_raw
        baseline_by_name = bench_index(base_doc)

    violations: list[tuple[str, str, str, str]] = []
    warn_msgs: list[str] = []

    # Validate each budget group
    for group, cfg in budget_groups.items():
        if not isinstance(cfg, dict):
            die(
                f"[{group}] budget entry must be a mapping (YAML dict), got: {type(cfg).__name__}",
                EXIT_ERROR,
            )

        # Only validate groups that have max_latency_s (skip throughput section etc.)
        max_latency = cfg.get("max_latency_s")
        if max_latency is None:
            continue
        try:
            max_latency_f = float(max_latency)
        except (TypeError, ValueError):
            die(
                f"[{group}] max_latency_s must be a number (seconds), got: {max_latency!r}",
                EXIT_ERROR,
            )

        # Get benchmark patterns for this group
        patterns = _as_list_of_str(bench_map.get(group), f"benchmark_map[{group!r}]")
        if not patterns:
            warn_msgs.append(f"No benchmark_map entry for '{group}' (skipping).")
            continue

        # Match benchmarks to this group
        matched = match_names(all_names, patterns)
        if not matched:
            msg = f"No benchmarks matched patterns {patterns}"
            if fail_on_unmatched:
                violations.append((group, "-", "-", msg))
            else:
                warn_msgs.append(f"[{group}] {msg} (fail_on_unmatched_patterns=false)")
            continue

        # Validate each matched benchmark
        for name in matched:
            b = bench_by_name[name]
            try:
                t = get_latency_s(b, args.metric)
            except (KeyError, TypeError, ValueError) as e:
                die(f"Invalid benchmark stats for '{name}' (metric={args.metric}): {e}", EXIT_ERROR)

            # Check budget threshold
            if t > max_latency_f:
                violations.append(
                    (group, name, f"{t:.6f}s", f"exceeds max_latency_s={max_latency}s")
                )

            # Check regression vs baseline (if provided)
            if baseline_by_name and max_reg is not None:
                if name in baseline_by_name:
                    try:
                        t0 = get_latency_s(baseline_by_name[name], args.metric)
                    except (KeyError, TypeError, ValueError) as e:
                        warn_msgs.append(
                            f"Baseline benchmark '{name}' has invalid stats: {e} (no regression check)."
                        )
                        continue

                    if t0 <= 0:
                        warn_msgs.append(f"Baseline metric is 0 for '{name}' (no regression check).")
                        continue

                    if t > t0 * (1 + max_reg):
                        regression_pct = (t / t0 - 1) * 100.0
                        violations.append(
                            (
                                group,
                                name,
                                f"{t:.6f}s",
                                f"regressed {regression_pct:.1f}% > {max_reg_pct:.1f}%",
                            )
                        )
                else:
                    warn_msgs.append(f"Baseline missing benchmark '{name}' (no regression check).")

    for w in warn_msgs:
        warn(w)

    if violations:
        print("\nPERFORMANCE BUDGET VIOLATIONS:")
        for group, name, val, why in violations:
            print(f"  [{group}] {name}")
            print(f"    {val} -> {why}")
        return EXIT_VIOLATIONS

    print("OK: Performance budgets satisfied.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
