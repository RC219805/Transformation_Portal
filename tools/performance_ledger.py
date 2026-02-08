#!/usr/bin/env python3
"""Performance ledger tool for pipeline regression detection (legacy).

NOTE: For CI/CD performance gating, the APEX Performance Observability Platform
is now authoritative (see .github/workflows/apex_performance.yml and ADR-024).
This tool remains available for historical baseline queries and ad-hoc analysis.

Parses manifests from batch runs, computes statistics, and compares against
baselines to detect performance regressions.

Usage:
    # Capture baseline
    python tools/performance_ledger.py \\
        --manifests-dir ./output/prod_run/manifests \\
        --output ./docs/performance/baselines/baseline_v2.0.0_da3_apex.json

    # Compare against baseline
    python tools/performance_ledger.py \\
        --baseline ./docs/performance/baselines/baseline_v2.0.0_da3_apex.json \\
        --compare ./output/experimental_run/manifests \\
        --output ./output/perf_report.md \\
        --emit-json ./output/perf_current.json

v1.7: Optional NumPy, bootstrap CI, backward compatibility, input validation.
Phase 2 implementation per ADR-023.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import random
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# NumPy is optional in v1.7 (Condition #2)
try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

__version__ = "1.7.0"

logger = logging.getLogger(__name__)

# Exit codes (v1.7 expansion)
EXIT_SUCCESS = 0
EXIT_REGRESSION = 1
EXIT_BACKEND_MISMATCH = 2
EXIT_INSUFFICIENT_DATA = 3

# Input validation bounds (Condition #6)
MAX_BOOTSTRAP_ITERATIONS = 10000
MAX_HIST_BINS = 100
MAX_TOP_N = 100
MIN_SAMPLES_FOR_COMPARISON = 3

# Regression thresholds (per ADR-023)
DEFAULT_REGRESSION_THRESHOLDS = {
    "p95_worsening_pct": 10.0,
    "mean_worsening_pct": 15.0,
    "failure_rate_increase": 0.0,
}


@dataclass
class EnvironmentMetadata:
    """Environment metadata for baseline reproducibility."""

    python: str
    torch: Optional[str]
    device: str
    os: str
    cpu: Optional[str] = None
    memory_gb: Optional[int] = None


@dataclass
class Statistics:
    """Runtime statistics for a batch run."""

    count: int
    mean_sec: float
    median_sec: float
    p90_sec: float
    p95_sec: float
    min_sec: float
    max_sec: float
    success_rate: float
    std_sec: Optional[float] = None
    total_sec: Optional[float] = None
    overhead_sec: Optional[float] = None
    bootstrap_ci_95_lower: Optional[float] = None
    bootstrap_ci_95_upper: Optional[float] = None


@dataclass
class Baseline:
    """Performance baseline schema."""

    version: str
    backend: str
    quality_tier: str
    environment: EnvironmentMetadata
    statistics: Statistics
    captured_at: str
    captured_by: str = f"tools/performance_ledger.py v{__version__}"
    notes: Optional[str] = None
    backend_compliance: Optional[Dict[str, Any]] = None


@dataclass
class Regression:
    """Detected regression."""

    metric: str
    baseline: float
    current: float
    change_pct: float
    threshold_pct: float
    status: str


# Pure Python statistics (Condition #2: NumPy optional)


def _pure_python_mean(values: List[float]) -> float:
    """Compute mean using pure Python."""
    if not values:
        raise ValueError("Cannot compute mean of empty list")
    return sum(values) / len(values)


def _pure_python_std(values: List[float], mean: Optional[float] = None) -> float:
    """Compute standard deviation using pure Python."""
    if not values:
        raise ValueError("Cannot compute std of empty list")
    if len(values) == 1:
        return 0.0
    if mean is None:
        mean = _pure_python_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance**0.5


def _pure_python_percentile(values: List[float], percentile: float) -> float:
    """Compute percentile using pure Python (linear interpolation)."""
    if not values:
        raise ValueError("Cannot compute percentile of empty list")
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]

    # Linear interpolation (numpy default)
    index = (percentile / 100.0) * (n - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, n - 1)
    weight = index - lower_index

    return sorted_values[lower_index] * (1 - weight) + sorted_values[upper_index] * weight


def _bootstrap_confidence_interval(
    values: List[float], iterations: int = 1000, confidence: float = 0.95, seed: int = 42
) -> Tuple[float, float]:
    """Compute bootstrap confidence interval for mean."""
    if len(values) < 2:
        return (values[0], values[0]) if values else (0.0, 0.0)

    random.seed(seed)
    bootstrap_means = []

    for _ in range(iterations):
        sample = [random.choice(values) for _ in range(len(values))]
        if HAS_NUMPY:
            bootstrap_means.append(float(np.mean(sample)))
        else:
            bootstrap_means.append(_pure_python_mean(sample))

    alpha = (1 - confidence) / 2
    lower_percentile = alpha * 100
    upper_percentile = (1 - alpha) * 100

    if HAS_NUMPY:
        return (
            float(np.percentile(bootstrap_means, lower_percentile)),
            float(np.percentile(bootstrap_means, upper_percentile)),
        )
    else:
        return (
            _pure_python_percentile(bootstrap_means, lower_percentile),
            _pure_python_percentile(bootstrap_means, upper_percentile),
        )


def parse_manifests(manifests_dir: Path) -> List[Dict[str, Any]]:
    """Load all manifest JSONs from directory."""
    if not manifests_dir.exists():
        raise FileNotFoundError(f"Manifests directory not found: {manifests_dir}")

    manifest_files = list(manifests_dir.glob("*.json"))
    if not manifest_files:
        raise ValueError(f"No JSON manifests found in {manifests_dir}")

    manifests = []
    for manifest_file in manifest_files:
        try:
            with open(manifest_file) as f:
                manifests.append(json.load(f))
        except json.JSONDecodeError as e:
            logger.warning(f"Skipping invalid JSON {manifest_file}: {e}")
            continue

    if not manifests:
        raise ValueError(f"No valid manifests parsed from {manifests_dir}")

    logger.info(f"Loaded {len(manifests)} manifests from {manifests_dir}")
    return manifests


def extract_timings(manifests: List[Dict[str, Any]]) -> Tuple[List[float], int, int]:
    """Extract timing data from manifests."""
    timings = []
    success_count = 0
    failure_count = 0

    for manifest in manifests:
        timing_metadata = manifest.get("timing", {})
        total_sec = timing_metadata.get("total_seconds")

        if total_sec is not None:
            timings.append(total_sec)
            depth_meta = manifest.get("depth")
            if depth_meta is not None:
                success_count += 1
            else:
                failure_count += 1

    return timings, success_count, failure_count


def extract_backends(manifests: List[Dict[str, Any]]) -> List[str]:
    """Extract backend identifiers from manifests."""
    backends = []
    for manifest in manifests:
        depth_meta = manifest.get("depth", {})
        backend = depth_meta.get("model", "unknown")
        backends.append(backend)
    return backends


def compute_statistics(timings: List[float], bootstrap_iterations: int = 1000, enable_bootstrap: bool = True) -> Statistics:
    """Compute runtime statistics with optional bootstrap CI."""
    if not timings:
        raise ValueError("No timings provided")

    if HAS_NUMPY:
        timings_array = np.array(timings)
        mean_val = float(np.mean(timings_array))
        std_val = float(np.std(timings_array, ddof=1)) if len(timings) > 1 else 0.0
        median_val = float(np.median(timings_array))
        p90_val = float(np.percentile(timings_array, 90))
        p95_val = float(np.percentile(timings_array, 95))
        min_val = float(np.min(timings_array))
        max_val = float(np.max(timings_array))
        total_val = float(np.sum(timings_array))
    else:
        mean_val = _pure_python_mean(timings)
        std_val = _pure_python_std(timings, mean_val) if len(timings) > 1 else 0.0
        sorted_timings = sorted(timings)
        median_val = _pure_python_percentile(sorted_timings, 50)
        p90_val = _pure_python_percentile(sorted_timings, 90)
        p95_val = _pure_python_percentile(sorted_timings, 95)
        min_val = min(timings)
        max_val = max(timings)
        total_val = sum(timings)

    # Bootstrap CI
    bootstrap_ci_lower = None
    bootstrap_ci_upper = None
    if enable_bootstrap and bootstrap_iterations > 0:
        bootstrap_ci_lower, bootstrap_ci_upper = _bootstrap_confidence_interval(timings, iterations=bootstrap_iterations)

    return Statistics(
        count=len(timings),
        mean_sec=mean_val,
        median_sec=median_val,
        p90_sec=p90_val,
        p95_sec=p95_val,
        min_sec=min_val,
        max_sec=max_val,
        std_sec=std_val,
        success_rate=1.0,
        total_sec=total_val,
        bootstrap_ci_95_lower=bootstrap_ci_lower,
        bootstrap_ci_95_upper=bootstrap_ci_upper,
    )


def capture_environment() -> EnvironmentMetadata:
    """Capture current environment metadata."""
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    torch_version = None
    try:
        import torch

        if hasattr(torch, "__version__"):
            torch_version = torch.__version__
    except (ImportError, AttributeError):
        pass

    device = "unknown"
    if torch_version:
        try:
            import torch

            if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            elif hasattr(torch, "cuda") and torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        except Exception:
            device = "cpu"

    os_info = f"{platform.system()}-{platform.release()}-{platform.machine()}"

    return EnvironmentMetadata(
        python=python_version,
        torch=torch_version,
        device=device,
        os=os_info,
    )


def detect_backend_mismatch(baseline: Baseline, current_backends: List[str]) -> Tuple[bool, str]:
    """Detect backend mismatches between baseline and current run."""
    baseline_backend = baseline.backend

    # Normalize backend names (e.g., "da3" == "depth-anything-v3")
    backend_aliases = {
        "da3": ["da3", "depth-anything-v3", "depth_anything_v3"],
        "depth-pro": ["depth-pro", "depth_pro", "depthpro"],
    }

    normalized_baseline = baseline_backend.lower()
    for canonical, aliases in backend_aliases.items():
        if normalized_baseline in aliases:
            normalized_baseline = canonical
            break

    mismatches = []
    for backend in current_backends:
        normalized_current = backend.lower()
        for canonical, aliases in backend_aliases.items():
            if normalized_current in aliases:
                normalized_current = canonical
                break

        if normalized_current != normalized_baseline:
            mismatches.append(backend)

    if mismatches:
        mismatch_pct = (len(mismatches) / len(current_backends)) * 100
        return (
            True,
            f"Backend mismatch: {mismatch_pct:.1f}% samples ({len(mismatches)}/{len(current_backends)}) differ from baseline '{baseline_backend}'",
        )

    return False, ""


def detect_regressions(
    baseline: Baseline, current_stats: Statistics, thresholds: Dict[str, float], strict: bool = False
) -> List[Regression]:
    """Compare current stats against baseline and detect regressions."""
    regressions = []

    # Check p95 regression
    p95_change_pct = ((current_stats.p95_sec - baseline.statistics.p95_sec) / baseline.statistics.p95_sec) * 100.0
    if strict or p95_change_pct > thresholds["p95_worsening_pct"]:
        status = "regression" if p95_change_pct > thresholds["p95_worsening_pct"] else "potential_regression"
        regressions.append(
            Regression(
                metric="p95_sec",
                baseline=baseline.statistics.p95_sec,
                current=current_stats.p95_sec,
                change_pct=p95_change_pct,
                threshold_pct=thresholds["p95_worsening_pct"],
                status=status,
            )
        )

    # Check mean regression
    mean_change_pct = ((current_stats.mean_sec - baseline.statistics.mean_sec) / baseline.statistics.mean_sec) * 100.0
    if strict or mean_change_pct > thresholds["mean_worsening_pct"]:
        status = "regression" if mean_change_pct > thresholds["mean_worsening_pct"] else "potential_regression"
        regressions.append(
            Regression(
                metric="mean_sec",
                baseline=baseline.statistics.mean_sec,
                current=current_stats.mean_sec,
                change_pct=mean_change_pct,
                threshold_pct=thresholds["mean_worsening_pct"],
                status=status,
            )
        )

    # Check failure rate increase
    failure_rate_change = current_stats.success_rate - baseline.statistics.success_rate
    if failure_rate_change < -thresholds["failure_rate_increase"]:
        regressions.append(
            Regression(
                metric="success_rate",
                baseline=baseline.statistics.success_rate,
                current=current_stats.success_rate,
                change_pct=failure_rate_change * 100.0,
                threshold_pct=thresholds["failure_rate_increase"],
                status="regression",
            )
        )

    # Filter to only confirmed regressions if not strict
    if not strict:
        regressions = [r for r in regressions if r.status == "regression"]

    return regressions


def format_markdown(
    baseline: Baseline, current_stats: Statistics, regressions: List[Regression], env: EnvironmentMetadata
) -> str:
    """Generate markdown report."""
    lines = [
        "# Performance Comparison Report",
        "",
        f"**Baseline:** {baseline.version} ({baseline.backend}, {baseline.quality_tier})",
        f"**Current:** ({current_stats.count} images)",
        f"**Environment:** {env.os}, Python {env.python}, torch {env.torch or 'N/A'}, device={env.device}",
        f"**Math Backend:** {'NumPy' if HAS_NUMPY else 'Pure Python'}",
        "",
        "## Statistics",
        "",
        "| Metric | Baseline | Current | Change | Status |",
        "|--------|----------|---------|--------|--------|",
    ]

    metrics = [
        ("Mean", baseline.statistics.mean_sec, current_stats.mean_sec),
        ("Median", baseline.statistics.median_sec, current_stats.median_sec),
        ("p90", baseline.statistics.p90_sec, current_stats.p90_sec),
        ("p95", baseline.statistics.p95_sec, current_stats.p95_sec),
        ("Min", baseline.statistics.min_sec, current_stats.min_sec),
        ("Max", baseline.statistics.max_sec, current_stats.max_sec),
        ("Success Rate", baseline.statistics.success_rate * 100, current_stats.success_rate * 100),
    ]

    for name, baseline_val, current_val in metrics:
        if "Rate" in name:
            change_pct = current_val - baseline_val
            status = "✅ OK" if change_pct >= 0 else "⚠️ REGRESSION"
            lines.append(f"| {name} | {baseline_val:.1f}% | {current_val:.1f}% | {change_pct:+.1f}% | {status} |")
        else:
            change_pct = ((current_val - baseline_val) / baseline_val) * 100.0
            status = "✅ OK"
            for reg in regressions:
                if name.lower().replace(" ", "_") in reg.metric:
                    status = "⚠️ REGRESSION"
                    break
            lines.append(f"| {name} | {baseline_val:.2f}s | {current_val:.2f}s | {change_pct:+.1f}% | {status} |")

    # Add bootstrap CI if available
    if current_stats.bootstrap_ci_95_lower is not None:
        lines.extend(
            [
                "",
                f"**Bootstrap 95% CI for mean:** [{current_stats.bootstrap_ci_95_lower:.2f}s, {current_stats.bootstrap_ci_95_upper:.2f}s]",
            ]
        )

    if regressions:
        lines.extend(["", "## Regressions Detected", ""])
        for reg in regressions:
            lines.append(
                f"⚠️ **{reg.metric} regression:** {reg.baseline:.2f} → {reg.current:.2f} "
                f"({reg.change_pct:+.1f}%, threshold {reg.threshold_pct:.1f}%)"
            )
        lines.extend(
            [
                "",
                "## Recommendation",
                "",
                "**DO NOT MERGE** - Performance regression detected.",
                "Investigate slowdown before merging changes.",
            ]
        )
    else:
        lines.extend(["", "## Recommendation", "", "✅ **OK TO MERGE** - No performance regressions detected."])

    return "\n".join(lines)


def load_baseline(path: Path) -> Baseline:
    """Load baseline from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")

    with open(path) as f:
        data = json.load(f)

    env = EnvironmentMetadata(**data["environment"])
    stats = Statistics(**data["statistics"])

    return Baseline(
        version=data["version"],
        backend=data["backend"],
        quality_tier=data["quality_tier"],
        environment=env,
        statistics=stats,
        captured_at=data["captured_at"],
        captured_by=data.get("captured_by", "tools/performance_ledger.py v1.0"),
        notes=data.get("notes"),
        backend_compliance=data.get("backend_compliance"),
    )


def save_baseline(baseline: Baseline, path: Path):
    """Save baseline to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict and filter None values for cleaner JSON
    data = asdict(baseline)
    data = {k: v for k, v in data.items() if v is not None}

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Baseline saved to {path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Performance ledger for regression detection (v1.7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exit Codes:
  0 - Success (no regression or potential regression without --strict)
  1 - Significant regression detected
  2 - Backend mismatch between baseline and current run
  3 - Insufficient data for comparison
        """,
    )

    # Core arguments
    parser.add_argument("--manifests-dir", type=Path, help="Directory containing manifest JSONs")
    parser.add_argument("--output", type=Path, required=True, help="Output file (JSON for baseline, MD for report)")
    parser.add_argument("--baseline", type=Path, help="Baseline JSON for comparison")
    parser.add_argument("--compare", type=Path, help="Manifests directory to compare against baseline")
    parser.add_argument("--emit-json", type=Path, help="Emit current stats as JSON")

    # Condition #1: Backward compatibility - add --version as alias for --baseline-version
    # Log deprecation warning when used
    parser.add_argument(
        "--baseline-version", dest="baseline_version", default="v2.0.0-post-pr841", help="Version identifier for baseline"
    )
    parser.add_argument(
        "--version", dest="baseline_version_deprecated", help=argparse.SUPPRESS  # DEPRECATED alias  # Hidden deprecated flag
    )

    parser.add_argument("--backend", default="da3", help="Backend identifier")
    parser.add_argument("--quality-tier", default="standard", help="Quality tier")

    # Threshold arguments
    parser.add_argument("--p95-threshold", type=float, default=10.0, help="p95 regression threshold (default: 10%%)")
    parser.add_argument("--mean-threshold", type=float, default=15.0, help="mean regression threshold (default: 15%%)")
    parser.add_argument(
        "--failure-rate-threshold", type=float, default=0.0, help="Failure rate increase threshold (default: 0.0%%)"
    )

    # v1.7 new arguments
    parser.add_argument("--strict", action="store_true", help="Strict mode: fail on any potential regression")
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1000,
        help=f"Bootstrap iterations for CI (default: 1000, max: {MAX_BOOTSTRAP_ITERATIONS})",
    )
    parser.add_argument("--no-bootstrap", action="store_true", help="Disable bootstrap confidence intervals")

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    logger.info(f"Performance ledger tool v{__version__} (NumPy: {'available' if HAS_NUMPY else 'unavailable'})")

    # Condition #1: Handle deprecated --version flag
    if args.baseline_version_deprecated is not None:
        warnings.warn(
            "Flag '--version' is deprecated and will be removed in v2.0. Use '--baseline-version' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.warning("⚠️  DEPRECATED: --version flag will be removed in v2.0. Use --baseline-version instead.")
        args.baseline_version = args.baseline_version_deprecated

    # Condition #6: Input validation
    if args.bootstrap_iterations > MAX_BOOTSTRAP_ITERATIONS:
        logger.error(f"Bootstrap iterations exceeds maximum ({MAX_BOOTSTRAP_ITERATIONS})")
        return EXIT_INSUFFICIENT_DATA
    if args.bootstrap_iterations < 0:
        logger.error("Bootstrap iterations must be non-negative")
        return EXIT_INSUFFICIENT_DATA

    try:
        # Mode 1: Capture baseline from manifests
        if args.manifests_dir and not args.baseline:
            logger.info(f"Capturing baseline from {args.manifests_dir}")

            manifests = parse_manifests(args.manifests_dir)
            timings, success_count, failure_count = extract_timings(manifests)

            if not timings:
                logger.error("No valid timings extracted from manifests")
                return EXIT_INSUFFICIENT_DATA

            stats = compute_statistics(
                timings,
                bootstrap_iterations=args.bootstrap_iterations if not args.no_bootstrap else 0,
                enable_bootstrap=not args.no_bootstrap,
            )
            total_count = success_count + failure_count
            stats.success_rate = success_count / total_count if total_count > 0 else 0.0

            env = capture_environment()
            backends = extract_backends(manifests)

            # Backend compliance
            backend_compliance = {
                "expected": args.backend,
                "actual": list(set(backends)),
                "mismatch_count": sum(1 for b in backends if b.lower() != args.backend.lower()),
            }

            baseline = Baseline(
                version=args.baseline_version,
                backend=args.backend,
                quality_tier=args.quality_tier,
                environment=env,
                statistics=stats,
                captured_at=datetime.now(timezone.utc).isoformat(),
                backend_compliance=backend_compliance if backend_compliance["mismatch_count"] > 0 else None,
            )

            save_baseline(baseline, args.output)
            logger.info(f"Captured baseline: {stats.count} images, mean={stats.mean_sec:.2f}s, p95={stats.p95_sec:.2f}s")
            return EXIT_SUCCESS

        # Mode 2: Compare against baseline
        elif args.baseline and args.compare:
            logger.info(f"Comparing {args.compare} against baseline {args.baseline}")

            baseline = load_baseline(args.baseline)
            manifests = parse_manifests(args.compare)

            # Condition #6: Check minimum samples
            if len(manifests) < MIN_SAMPLES_FOR_COMPARISON:
                logger.error(f"Insufficient data: need at least {MIN_SAMPLES_FOR_COMPARISON} samples, got {len(manifests)}")
                return EXIT_INSUFFICIENT_DATA

            timings, success_count, failure_count = extract_timings(manifests)
            backends = extract_backends(manifests)

            if not timings:
                logger.error("No valid timings extracted from comparison manifests")
                return EXIT_INSUFFICIENT_DATA

            # Check backend mismatch
            has_mismatch, mismatch_msg = detect_backend_mismatch(baseline, backends)
            if has_mismatch:
                logger.error(mismatch_msg)
                return EXIT_BACKEND_MISMATCH

            current_stats = compute_statistics(
                timings,
                bootstrap_iterations=args.bootstrap_iterations if not args.no_bootstrap else 0,
                enable_bootstrap=not args.no_bootstrap,
            )
            total_count = success_count + failure_count
            current_stats.success_rate = success_count / total_count if total_count > 0 else 0.0

            env = capture_environment()

            thresholds = {
                "p95_worsening_pct": args.p95_threshold,
                "mean_worsening_pct": args.mean_threshold,
                "failure_rate_increase": args.failure_rate_threshold,
            }

            regressions = detect_regressions(baseline, current_stats, thresholds, strict=args.strict)

            # Generate markdown report
            report = format_markdown(baseline, current_stats, regressions, env)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                f.write(report)
            logger.info(f"Report written to {args.output}")

            # Emit JSON if requested
            if args.emit_json:
                current_baseline = Baseline(
                    version="current",
                    backend=args.backend,
                    quality_tier=args.quality_tier,
                    environment=env,
                    statistics=current_stats,
                    captured_at=datetime.now(timezone.utc).isoformat(),
                )
                save_baseline(current_baseline, args.emit_json)

            # Return exit code
            if regressions:
                logger.warning(f"⚠️  {len(regressions)} regression(s) detected")
                return EXIT_REGRESSION
            else:
                logger.info("✅ No regressions detected")
                return EXIT_SUCCESS

        else:
            logger.error(
                "Invalid arguments. Use --manifests-dir --output for baseline capture, or --baseline --compare --output for comparison"
            )
            return EXIT_INSUFFICIENT_DATA

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return EXIT_INSUFFICIENT_DATA


if __name__ == "__main__":
    sys.exit(main())
