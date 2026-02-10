"""Batch processing statistics computation.

Provides runtime statistics and outlier detection for APEX batch processing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def compute_batch_runtime_stats(runtimes: List[float]) -> Dict[str, Any]:
    """Compute statistics from batch processing runtimes.

    Args:
        runtimes: List of runtime values in seconds

    Returns:
        Dictionary with min, max, mean, median, total
    """
    if not runtimes:
        return {
            "count": 0,
            "total": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
        }

    sorted_runtimes = sorted(runtimes)
    n = len(sorted_runtimes)

    return {
        "count": n,
        "total": sum(sorted_runtimes),
        "mean": sum(sorted_runtimes) / n,
        "min": sorted_runtimes[0],
        "max": sorted_runtimes[-1],
        "median": sorted_runtimes[n // 2] if n % 2 == 1 else (sorted_runtimes[n // 2 - 1] + sorted_runtimes[n // 2]) / 2,
    }


def detect_runtime_outliers(
    image_name: str,
    runtime_s: float,
    runtimes: List[float],
    threshold_multiplier: float = 5.0,
    median: Optional[float] = None,
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Detect if an image runtime is an outlier compared to batch median.

    Logs a warning if runtime exceeds threshold_multiplier × median.

    PERFORMANCE FIX (#3): Accept pre-computed median to avoid O(n²) complexity.
    Caller should compute stats once and pass median for all outlier checks.

    Args:
        image_name: Name of the image being processed
        runtime_s: Runtime for this specific image
        runtimes: List of all runtimes in the batch (for median calculation if needed)
        threshold_multiplier: Multiplier for outlier threshold (default: 5.0x)
        median: Pre-computed median runtime (optional, computed if None)

    Returns:
        Tuple of (warning_message, outlier_metadata) if outlier detected, None otherwise
    """
    if not runtimes or len(runtimes) < 2:
        return None  # Need at least 2 samples for meaningful comparison

    # Use pre-computed median if provided, otherwise compute (for backward compatibility)
    if median is None:
        stats = compute_batch_runtime_stats(runtimes)
        median = stats["median"]

    if median == 0:
        return None  # Avoid division by zero

    ratio = runtime_s / median

    if ratio > threshold_multiplier:
        warning_msg = (
            f"⚠️  Runtime outlier detected: {image_name} took {runtime_s:.2f}s "
            f"({ratio:.1f}× median of {median:.2f}s). "
            f"Investigate for resolution, aspect ratio, or dynamic range issues."
        )
        logger.warning(warning_msg)

        outlier_metadata = {
            "is_outlier": True,
            "runtime_s": runtime_s,
            "median_runtime_s": median,
            "ratio_to_median": ratio,
            "threshold_multiplier": threshold_multiplier,
        }

        return warning_msg, outlier_metadata

    return None
