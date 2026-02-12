#!/usr/bin/env python3
"""Performance ledger tool for pipeline regression detection.

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

Phase 2 implementation per ADR-023.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Regression thresholds (per ADR-023)
DEFAULT_REGRESSION_THRESHOLDS = {
    "p95_worsening_pct": 10.0,  # p95 > 10% slower = regression
    "mean_worsening_pct": 15.0,  # mean > 15% slower = regression
    "failure_rate_increase": 0.0,  # Any failures = regression
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
    total_sec: Optional[float] = None
    overhead_sec: Optional[float] = None


@dataclass
class Baseline:
    """Performance baseline schema."""

    version: str
    backend: str
    quality_tier: str
    environment: EnvironmentMetadata
    statistics: Statistics
    captured_at: str
    captured_by: str = "tools/performance_ledger.py v1.0"
    notes: Optional[str] = None


@dataclass
class Regression:
    """Detected regression."""

    metric: str
    baseline: float
    current: float
    change_pct: float
    threshold_pct: float
    status: str  # "ok" or "regression"


def parse_manifests(manifests_dir: Path) -> List[Dict[str, Any]]:
    """Load all manifest JSONs from directory.

    Args:
        manifests_dir: Directory containing manifest JSON files

    Returns:
        List of manifest dictionaries

    Raises:
        FileNotFoundError: If manifests directory doesn't exist
        ValueError: If no valid manifests found
    """
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
    """Extract timing data from manifests.

    Args:
        manifests: List of manifest dictionaries

    Returns:
        Tuple of (timings_sec, success_count, failure_count)
    """
    timings = []
    success_count = 0
    failure_count = 0

    for manifest in manifests:
        timing_metadata = manifest.get("timing", {})
        total_sec = timing_metadata.get("total_seconds")

        if total_sec is not None:
            timings.append(total_sec)
            # Determine success/failure from depth status
            depth_meta = manifest.get("depth")
            if depth_meta is not None:
                success_count += 1
            else:
                failure_count += 1

    return timings, success_count, failure_count


def compute_statistics(timings: List[float]) -> Statistics:
    """Compute runtime statistics.

    Args:
        timings: List of timing values in seconds

    Returns:
        Statistics dataclass with computed values
    """
    if not timings:
        raise ValueError("No timings provided")

    timings_array = np.array(timings)

    return Statistics(
        count=len(timings),
        mean_sec=float(np.mean(timings_array)),
        median_sec=float(np.median(timings_array)),
        p90_sec=float(np.percentile(timings_array, 90)),
        p95_sec=float(np.percentile(timings_array, 95)),
        min_sec=float(np.min(timings_array)),
        max_sec=float(np.max(timings_array)),
        success_rate=1.0,  # Updated by caller if failures exist
        total_sec=float(np.sum(timings_array)),
    )


def capture_environment() -> EnvironmentMetadata:
    """Capture current environment metadata.

    Returns:
        EnvironmentMetadata with current system info
    """
    # Get Python version
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Get torch version (optional)
    torch_version = None
    try:
        import torch

        # Handle mocked torch (e.g., in CI tests without ML dependencies)
        if hasattr(torch, "__version__"):
            torch_version = torch.__version__
    except (ImportError, AttributeError):
        pass

    # Get device (placeholder - would need to detect actual device)
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

    # Get OS info
    os_info = f"{platform.system()}-{platform.release()}-{platform.machine()}"

    return EnvironmentMetadata(
        python=python_version,
        torch=torch_version,
        device=device,
        os=os_info,
    )


def detect_regressions(baseline: Baseline, current_stats: Statistics, thresholds: Dict[str, float]) -> List[Regression]:
    """Compare current stats against baseline and detect regressions.

    Args:
        baseline: Baseline to compare against
        current_stats: Current run statistics
        thresholds: Regression threshold configuration

    Returns:
        List of detected regressions
    """
    regressions = []

    # Check p95 regression
    p95_change_pct = ((current_stats.p95_sec - baseline.statistics.p95_sec) / baseline.statistics.p95_sec) * 100.0
    if p95_change_pct > thresholds["p95_worsening_pct"]:
        regressions.append(
            Regression(
                metric="p95_sec",
                baseline=baseline.statistics.p95_sec,
                current=current_stats.p95_sec,
                change_pct=p95_change_pct,
                threshold_pct=thresholds["p95_worsening_pct"],
                status="regression",
            )
        )

    # Check mean regression
    mean_change_pct = ((current_stats.mean_sec - baseline.statistics.mean_sec) / baseline.statistics.mean_sec) * 100.0
    if mean_change_pct > thresholds["mean_worsening_pct"]:
        regressions.append(
            Regression(
                metric="mean_sec",
                baseline=baseline.statistics.mean_sec,
                current=current_stats.mean_sec,
                change_pct=mean_change_pct,
                threshold_pct=thresholds["mean_worsening_pct"],
                status="regression",
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

    return regressions


def format_markdown(
    baseline: Baseline, current_stats: Statistics, regressions: List[Regression], env: EnvironmentMetadata
) -> str:
    """Generate markdown report.

    Args:
        baseline: Baseline for comparison
        current_stats: Current run statistics
        regressions: List of detected regressions
        env: Current environment metadata

    Returns:
        Formatted markdown report
    """
    lines = [
        "# Performance Comparison Report",
        "",
        f"**Baseline:** {baseline.version} ({baseline.backend}, {baseline.quality_tier})",
        f"**Current:** ({current_stats.count} images)",
        f"**Environment:** {env.os}, Python {env.python}, torch {env.torch or 'N/A'}, device={env.device}",
        "",
        "## Statistics",
        "",
        "| Metric | Baseline | Current | Change | Status |",
        "|--------|----------|---------|--------|--------|",
    ]

    # Add rows for each metric
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
    """Load baseline from JSON file.

    Args:
        path: Path to baseline JSON

    Returns:
        Baseline instance

    Raises:
        FileNotFoundError: If baseline file doesn't exist
        ValueError: If baseline is invalid
    """
    if not path.exists():
        raise FileNotFoundError(f"Baseline not found: {path}")

    with open(path) as f:
        data = json.load(f)

    # Reconstruct nested dataclasses
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
    )


def save_baseline(baseline: Baseline, path: Path):
    """Save baseline to JSON file.

    Args:
        baseline: Baseline to save
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(asdict(baseline), f, indent=2)
    logger.info(f"Baseline saved to {path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Performance ledger for regression detection")
    parser.add_argument("--manifests-dir", type=Path, help="Directory containing manifest JSONs")
    parser.add_argument("--output", type=Path, required=True, help="Output file (JSON for baseline, MD for report)")
    parser.add_argument("--baseline", type=Path, help="Baseline JSON for comparison")
    parser.add_argument("--compare", type=Path, help="Manifests directory to compare against baseline")
    parser.add_argument("--emit-json", type=Path, help="Emit current stats as JSON")
    parser.add_argument("--p95-threshold", type=float, default=10.0, help="p95 regression threshold (default: 10%%)")
    parser.add_argument("--mean-threshold", type=float, default=15.0, help="mean regression threshold (default: 15%%)")
    parser.add_argument("--version", default="v2.0.0-post-pr841", help="Version identifier for baseline")
    parser.add_argument("--backend", default="da3", help="Backend identifier")
    parser.add_argument("--quality-tier", default="standard", help="Quality tier")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Performance ledger tool (v1.0)")

    try:
        # Mode 1: Capture baseline from manifests
        if args.manifests_dir and not args.baseline:
            logger.info(f"Capturing baseline from {args.manifests_dir}")

            manifests = parse_manifests(args.manifests_dir)
            timings, success_count, failure_count = extract_timings(manifests)

            if not timings:
                logger.error("No valid timings extracted from manifests")
                return 1

            stats = compute_statistics(timings)
            total_count = success_count + failure_count
            stats.success_rate = success_count / total_count if total_count > 0 else 0.0

            env = capture_environment()

            baseline = Baseline(
                version=args.version,
                backend=args.backend,
                quality_tier=args.quality_tier,
                environment=env,
                statistics=stats,
                captured_at=datetime.now(timezone.utc).isoformat(),
            )

            save_baseline(baseline, args.output)
            logger.info(f"Captured baseline: {stats.count} images, mean={stats.mean_sec:.2f}s, p95={stats.p95_sec:.2f}s")
            return 0

        # Mode 2: Compare against baseline
        elif args.baseline and args.compare:
            logger.info(f"Comparing {args.compare} against baseline {args.baseline}")

            baseline = load_baseline(args.baseline)
            manifests = parse_manifests(args.compare)
            timings, success_count, failure_count = extract_timings(manifests)

            if not timings:
                logger.error("No valid timings extracted from comparison manifests")
                return 1

            current_stats = compute_statistics(timings)
            total_count = success_count + failure_count
            current_stats.success_rate = success_count / total_count if total_count > 0 else 0.0

            env = capture_environment()

            thresholds = {
                "p95_worsening_pct": args.p95_threshold,
                "mean_worsening_pct": args.mean_threshold,
                "failure_rate_increase": DEFAULT_REGRESSION_THRESHOLDS["failure_rate_increase"],
            }

            regressions = detect_regressions(baseline, current_stats, thresholds)

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
                return 1
            else:
                logger.info("✅ No regressions detected")
                return 0

        else:
            logger.error(
                "Invalid arguments. Use --manifests-dir --output for baseline capture, or --baseline --compare --output for comparison"
            )
            return 1

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
