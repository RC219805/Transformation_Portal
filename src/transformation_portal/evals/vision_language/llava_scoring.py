"""Deterministic scoring layer for structured VQA outputs.

This module provides functions to compute quality scores from VQA results
using a consistent severity penalty system.
"""

from __future__ import annotations

from transformation_portal.evals.vision_language.llava_schema import VQAResult


# Severity penalties for quality issues
_SEVERITY_PENALTIES = {
    "low": 0.10,
    "medium": 0.25,
    "high": 0.50,
}


def recompute_summary_score(result: VQAResult) -> float:
    """Recompute summary score from issues using deterministic penalties.

    This function provides a consistent way to compute quality scores
    independent of the model's own scoring, based purely on detected issues.

    Args:
        result: VQA result with issues list

    Returns:
        Computed score in range [0.0, 1.0]

    Example:
        >>> result = VQAResult(
        ...     passes_basic_quality=True,
        ...     summary_score=0.8,
        ...     issues=[VQAIssue("mask_leak", "medium", "edge bleeding")],
        ... )
        >>> score = recompute_summary_score(result)
        >>> assert 0.0 <= score <= 1.0
    """
    score = 1.0
    for issue in result.issues:
        penalty = _SEVERITY_PENALTIES.get(issue.severity, 0.50)
        score -= penalty

    # Clamp to valid range
    if score < 0.0:
        score = 0.0
    if score > 1.0:
        score = 1.0

    return score


def compute_quality_gate_pass(
    result: VQAResult,
    *,
    min_score: float = 0.75,
    max_high_severity_issues: int = 0,
    max_medium_severity_issues: int = 2,
) -> bool:
    """Determine if a VQA result passes quality gate criteria.

    Args:
        result: VQA result to evaluate
        min_score: Minimum acceptable summary score (default: 0.75)
        max_high_severity_issues: Maximum allowed high severity issues (default: 0)
        max_medium_severity_issues: Maximum allowed medium severity issues (default: 2)

    Returns:
        True if the result passes all quality gate criteria
    """
    # Check summary score threshold
    if result.summary_score < min_score:
        return False

    # Count issues by severity
    high_count = sum(1 for i in result.issues if i.severity == "high")
    medium_count = sum(1 for i in result.issues if i.severity == "medium")

    # Check issue count thresholds
    if high_count > max_high_severity_issues:
        return False
    if medium_count > max_medium_severity_issues:
        return False

    return True


def severity_to_numeric(severity: str) -> float:
    """Convert severity string to numeric value for sorting/comparison.

    Args:
        severity: Severity string ("low", "medium", "high")

    Returns:
        Numeric value (0.0-1.0)
    """
    mapping = {
        "low": 0.25,
        "medium": 0.50,
        "high": 1.00,
    }
    return mapping.get(severity.lower(), 0.50)
