"""APEX workflow gate for CI/CD quality enforcement.

This module implements the gating logic that blocks PRs/releases when:
1. Worst-zone p95 exceeds threshold (user experience gate)
2. Regression vs baseline exceeds threshold (quality regression gate)
3. Any bucket fails Quality Firewall threshold (contract violation gate)

Design principles:
- Explicit, testable rules (no magic)
- Shadow mode support (warn but don't block)
- Clear explanation of why gate blocked
- Support per-workflow-version gating (V1 vs V2)

Usage:
    from transformation_portal.metrics.gate import should_block, evaluate_gate

    # Check if judgement should block
    block, reason = should_block(judgement, mode="enforce")

    # Full gate evaluation
    result = evaluate_gate(
        judgement=judgement,
        worst_zone_p95_threshold=15.0,
        max_regression_threshold=0.15,
        mode="enforce",
    )

Version: 1.0.0
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from transformation_portal.metrics.contracts import Judgement

__version__ = "1.0.0"

logger = logging.getLogger(__name__)


@dataclass
class GateResult:
    """Result of gate evaluation.

    Attributes:
        should_block: Whether the gate should block release
        mode: Gate mode ("enforce", "shadow", "disabled")
        reasons: List of blocking reasons (may be non-empty in shadow mode)
        explanation: Human-readable summary
        worst_zone_p95: Worst-zone p95 value (if available)
        worst_zone_name: Name of worst zone (if available)
    """

    should_block: bool
    mode: Literal["enforce", "shadow", "disabled"]
    reasons: list[str]
    explanation: str
    worst_zone_p95: Optional[float] = None
    worst_zone_name: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dict for serialization."""
        return {
            "should_block": self.should_block,
            "mode": self.mode,
            "reasons": self.reasons,
            "explanation": self.explanation,
            "worst_zone_p95": self.worst_zone_p95,
            "worst_zone_name": self.worst_zone_name,
        }


def should_block(
    judgement: Judgement,
    worst_zone_p95_threshold: Optional[float] = None,
    max_regression_threshold: float = 0.15,
    mode: Literal["enforce", "shadow", "disabled"] = "enforce",
) -> Tuple[bool, str]:
    """Determine if judgement should block release (simple API).

    Args:
        judgement: Judgement to evaluate
        worst_zone_p95_threshold: Maximum allowed worst-zone p95 (seconds)
        max_regression_threshold: Maximum allowed regression (fraction)
        mode: Gate mode ("enforce", "shadow", "disabled")

    Returns:
        Tuple of (should_block, reason)
    """
    result = evaluate_gate(
        judgement=judgement,
        worst_zone_p95_threshold=worst_zone_p95_threshold,
        max_regression_threshold=max_regression_threshold,
        mode=mode,
    )

    main_reason = result.reasons[0] if result.reasons else "No blocking reasons"
    return result.should_block, main_reason


def evaluate_gate(
    judgement: Optional[Judgement] = None,
    bucket_stats: Optional[Dict] = None,
    regression_report: Optional[Any] = None,
    worst_zone_p95_threshold: Optional[float] = None,
    max_regression_threshold: float = 0.15,
    mode: Literal["enforce", "shadow", "disabled"] = "enforce",
    min_samples: int = 20,
) -> GateResult:
    """Evaluate full gate logic with detailed results.

    Gate rules (evaluated in order):
    0. If mode is "disabled", always pass
    1. Check minimum sample size (insufficient_data if too small)
    2. Check if bucket_stats indicate "fail" (bucket threshold violation)
    3. Check if worst_zone_p95 > threshold (user experience gate)
    4. Check if regression > threshold (quality regression gate)

    Args:
        judgement: Judgement to evaluate (legacy API)
        bucket_stats: Dict of BucketStats (direct API)
        regression_report: RegressionReport (direct API)
        worst_zone_p95_threshold: Maximum allowed worst-zone p95 (seconds)
        max_regression_threshold: Maximum allowed regression (fraction)
        mode: Gate mode:
            - "enforce": Block if any rule fails
            - "shadow": Log warnings but don't block
            - "disabled": Always pass
        min_samples: Minimum samples required for statistical validity (default: 20)

    Returns:
        GateResult with verdict and detailed reasoning
    """
    reasons = []

    # Support both old API (judgement) and new API (bucket_stats)
    if judgement is not None:
        bucket_stats = judgement.bucket_stats
        regression_report = judgement.regression_report
        worst_zone_p95 = judgement.worst_zone_p95
        worst_zone_name = getattr(judgement, "worst_zone_name", None)
    else:
        worst_zone_p95 = None
        worst_zone_name = None

    # Rule 0: Disabled mode always passes
    if mode == "disabled":
        return GateResult(
            should_block=False,
            mode=mode,
            reasons=[],
            explanation="Gate is disabled",
            worst_zone_p95=worst_zone_p95,
            worst_zone_name=worst_zone_name,
        )

    # Rule 1: Check minimum sample size for statistical validity
    # Filter out insufficient-data buckets BEFORE gate evaluation
    if bucket_stats:
        # Separate valid buckets from insufficient-data buckets
        valid_buckets = {
            name: stats
            for name, stats in bucket_stats.items()
            if not getattr(stats, "is_insufficient_data", False) and stats.count >= min_samples
        }

        insufficient_buckets = {
            name: stats
            for name, stats in bucket_stats.items()
            if getattr(stats, "is_insufficient_data", False) or stats.count < min_samples
        }

        if insufficient_buckets:
            logger.info(
                f"Skipping {len(insufficient_buckets)} buckets with insufficient data: {list(insufficient_buckets.keys())}"
            )

        # If ALL buckets have insufficient data, never block
        if not valid_buckets:
            explanation = f"Insufficient data: all {len(bucket_stats)} buckets below minimum n={min_samples} samples"
            logger.warning(explanation)
            return GateResult(
                should_block=False,
                mode=mode,
                reasons=[],
                explanation=explanation,
                worst_zone_p95=worst_zone_p95,
                worst_zone_name=worst_zone_name,
            )

        # Continue with only valid buckets
        bucket_stats = valid_buckets

    # Rule 2: Check bucket threshold violations
    failing_buckets = []
    if bucket_stats:
        for name, stats in bucket_stats.items():
            if stats.pass_fail == "fail":
                failing_buckets.append(name)

    if failing_buckets:
        reasons.append(f"Bucket threshold violation: {', '.join(failing_buckets)} exceeded p95 threshold")

    # Rule 3: Check worst-zone p95
    if worst_zone_p95_threshold is not None and worst_zone_p95 is not None:
        if worst_zone_p95 > worst_zone_p95_threshold:
            zone_info = f" (zone: {worst_zone_name})" if worst_zone_name else ""
            reasons.append(f"Worst-zone p95 exceeded: {worst_zone_p95:.2f}s > {worst_zone_p95_threshold:.2f}s{zone_info}")

    # Rule 4: Check regression threshold
    if regression_report is not None:
        if regression_report.max_regression > max_regression_threshold:
            reasons.append(
                f"Regression threshold exceeded: {regression_report.max_regression * 100:.1f}% > "
                f"{max_regression_threshold * 100:.1f}% (bucket: {regression_report.max_regression_bucket})"
            )

    # Determine verdict and should_block based on mode
    if len(reasons) > 0:
        if mode == "enforce":
            should_block = True
            explanation = f"Gate BLOCKED: {'; '.join(reasons)}"
        else:  # shadow
            should_block = False
            explanation = f"Gate SHADOW (would block): {'; '.join(reasons)}"
    else:
        should_block = False
        explanation = "Gate PASSED"

    return GateResult(
        should_block=should_block,
        mode=mode,
        reasons=reasons,
        explanation=explanation,
        worst_zone_p95=worst_zone_p95,
        worst_zone_name=worst_zone_name,
    )


def should_block_v1_v2_comparison(
    v1_judgement: Judgement,
    v2_judgement: Judgement,
    v2_regression_threshold: float = 0.15,
    mode: Literal["enforce", "shadow", "disabled"] = "shadow",
) -> GateResult:
    """Special gate for V1 vs V2 comparison (dual-run).

    This is used when running V1 and V2 on the same commit to ensure
    V2 doesn't regress compared to V1.

    Args:
        v1_judgement: V1 workflow judgement
        v2_judgement: V2 workflow judgement (with regression_report comparing to V1)
        v2_regression_threshold: Maximum allowed V2 regression vs V1
        mode: Gate mode (default: "shadow" for V2 rollout)

    Returns:
        GateResult for V2 workflow
    """
    # For now, use standard gate on V2 judgement
    # In the future, this could apply special rules for V2 adoption
    return evaluate_gate(
        judgement=v2_judgement,
        max_regression_threshold=v2_regression_threshold,
        mode=mode,
    )
