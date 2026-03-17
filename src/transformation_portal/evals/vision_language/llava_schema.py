"""VQA result schema for structured LLaVA outputs.

This module defines the schema for structured visual quality assessment
results, parsing JSON responses from LLaVA into typed dataclasses.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


class VQAParseError(ValueError):
    """Raised when VQA response parsing fails."""


@dataclass(frozen=True)
class VQAIssue:
    """Single quality issue detected in an image.

    Attributes:
        issue_type: Category of issue (e.g., "mask_leakage", "texture_seam")
        severity: Severity level ("low", "medium", "high")
        evidence: Description of the evidence for this issue
    """

    issue_type: str
    severity: str
    evidence: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VQAIssue:
        """Create issue from dictionary."""
        return cls(
            issue_type=str(data.get("issue_type", data.get("type", "unknown"))),
            severity=str(data.get("severity", "medium")).lower(),
            evidence=str(data.get("evidence", "")),
        )


@dataclass
class VQAResult:
    """Structured VQA result from LLaVA quality assessment.

    Attributes:
        passes_basic_quality: Whether the image passes basic quality checks
        summary_score: Numeric quality score (0.0-1.0)
        issues: List of detected quality issues
        raw_text: Original model output text
        model_key: Model key used for inference
        parse_error: Error message if parsing failed
    """

    passes_basic_quality: bool
    summary_score: float
    issues: list[VQAIssue] = field(default_factory=list)
    raw_text: Optional[str] = None
    model_key: Optional[str] = None
    parse_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passes_basic_quality": self.passes_basic_quality,
            "summary_score": self.summary_score,
            "issues": [
                {
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "evidence": issue.evidence,
                }
                for issue in self.issues
            ],
            "model_key": self.model_key,
            "parse_error": self.parse_error,
        }


def _extract_json_from_text(text: str) -> str:
    """Extract JSON object from text that may contain markdown or other content.

    Args:
        text: Raw model output text

    Returns:
        Extracted JSON string

    Raises:
        VQAParseError: If no JSON object found
    """
    # Try to find JSON block in markdown code fence
    code_fence_pattern = r"```(?:json)?\s*(\{[\s\S]*?\})\s*```"
    match = re.search(code_fence_pattern, text)
    if match:
        return match.group(1)

    # Try to find bare JSON object
    json_pattern = r"\{[\s\S]*\}"
    match = re.search(json_pattern, text)
    if match:
        return match.group(0)

    raise VQAParseError(f"No JSON object found in response: {text[:200]}...")


def parse_vqa_result(
    model_key: str,
    raw_text: str,
) -> VQAResult:
    """Parse VQA result from raw model output text.

    Args:
        model_key: Model key used for inference
        raw_text: Raw text output from the model

    Returns:
        Parsed VQAResult (with parse_error set if parsing failed)
    """
    try:
        json_str = _extract_json_from_text(raw_text)
        data = json.loads(json_str)
    except (VQAParseError, json.JSONDecodeError) as exc:
        logger.warning("Failed to parse VQA response: %s", exc)
        return VQAResult(
            passes_basic_quality=False,
            summary_score=0.0,
            issues=[],
            raw_text=raw_text,
            model_key=model_key,
            parse_error=str(exc),
        )

    # Extract fields with defaults
    passes_basic_quality = bool(data.get("passes_basic_quality", False))

    summary_score_raw = data.get("summary_score", 0.0)
    try:
        summary_score = float(summary_score_raw)
        # Clamp to [0, 1]
        summary_score = max(0.0, min(1.0, summary_score))
    except (TypeError, ValueError):
        summary_score = 0.0

    # Parse issues
    issues = []
    raw_issues = data.get("issues", [])
    if isinstance(raw_issues, list):
        for issue_data in raw_issues:
            if isinstance(issue_data, dict):
                try:
                    issues.append(VQAIssue.from_dict(issue_data))
                except Exception as exc:
                    logger.warning("Failed to parse issue: %s", exc)

    return VQAResult(
        passes_basic_quality=passes_basic_quality,
        summary_score=summary_score,
        issues=issues,
        raw_text=raw_text,
        model_key=model_key,
        parse_error=None,
    )
