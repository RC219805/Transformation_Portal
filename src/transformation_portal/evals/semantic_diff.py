"""Semantic diff service using LLaVA for AI-powered change analysis.

This module provides semantic comparison between two artifacts (images/renders)
using the LLaVA vision-language model to generate structured, actionable
explanations of differences.

Integrates with:
- manifest-aware LLaVA backend
- time-travel history
- diff UI
- APEX evaluation harness
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformation_portal.evals.vision_language.llava_backend import (
        LlavaQualityBackend,
    )

logger = logging.getLogger(__name__)


class SemanticDiffError(RuntimeError):
    """Error during semantic diff analysis."""


@dataclass(frozen=True)
class SemanticChange:
    """A single semantic change detected between artifacts."""

    type: str  # geometry|texture|missing|artifact|semantic
    severity: str  # low|medium|high
    description: str
    location: str | None = None  # Optional spatial location hint
    confidence: float = 1.0


@dataclass(frozen=True)
class SemanticDiffResult:
    """Result of semantic diff analysis between two artifacts."""

    summary: str
    changes: tuple[SemanticChange, ...]
    raw_text: str
    structured: dict[str, Any]
    image_a_hash: str = ""
    image_b_hash: str = ""

    @property
    def has_changes(self) -> bool:
        """Whether any changes were detected."""
        return len(self.changes) > 0

    @property
    def high_severity_count(self) -> int:
        """Count of high-severity changes."""
        return sum(1 for c in self.changes if c.severity == "high")

    @property
    def change_types(self) -> set[str]:
        """Set of change types detected."""
        return {c.type for c in self.changes}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "summary": self.summary,
            "changes": [
                {
                    "type": c.type,
                    "severity": c.severity,
                    "description": c.description,
                    "location": c.location,
                    "confidence": c.confidence,
                }
                for c in self.changes
            ],
            "image_a_hash": self.image_a_hash,
            "image_b_hash": self.image_b_hash,
            "high_severity_count": self.high_severity_count,
            "has_changes": self.has_changes,
        }


# Prompt template for semantic diff analysis
_DIFF_PROMPT = """Return only valid JSON.

Compare the TWO images and describe differences.

Focus on:
1. geometry changes (shape, structure, proportions)
2. texture/material changes (color, roughness, reflectivity, patterns)
3. missing or added regions (objects, areas, details)
4. artifacts (noise, seams, distortions, aliasing)
5. semantic differences (object identity, plausibility, context)

For each change, assess severity:
- low: minor visual difference, acceptable
- medium: noticeable difference, may need attention
- high: significant issue requiring fix

Schema:
{
  "summary": string (2-3 sentence overview),
  "changes": [
    {
      "type": "geometry|texture|missing|artifact|semantic",
      "severity": "low|medium|high",
      "description": string (specific, actionable description),
      "location": string (optional, e.g. "upper-left quadrant", "center")
    }
  ]
}

If images are identical or nearly identical, return:
{
  "summary": "No significant differences detected.",
  "changes": []
}
"""

_SYSTEM_PROMPT = (
    "You are a precise visual diff analyst specializing in "
    "computer graphics, 3D rendering, and image quality assessment. "
    "Your analysis should be technical, specific, and actionable."
)


def _build_messages(img_a: Path, img_b: Path) -> list[dict[str, Any]]:
    """Build chat messages for semantic diff analysis.

    Args:
        img_a: Path to first image
        img_b: Path to second image

    Returns:
        List of chat messages for the model
    """
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(img_a)},
                {"type": "image", "image": str(img_b)},
                {"type": "text", "text": _DIFF_PROMPT},
            ],
        },
    ]


def _parse_response(raw: str) -> dict[str, Any]:
    """Parse model response into structured format.

    Args:
        raw: Raw text response from model

    Returns:
        Parsed JSON payload
    """
    # Try to extract JSON from response
    text = raw.strip()

    # Handle markdown code blocks
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (code block markers)
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("Failed to parse semantic diff response: %s", e)
        return {
            "summary": "Failed to parse semantic diff response",
            "changes": [],
            "parse_error": str(e),
            "raw_text": raw[:500],  # Include truncated raw for debugging
        }


def _validate_change(change: dict[str, Any]) -> SemanticChange | None:
    """Validate and convert a change dict to SemanticChange.

    Args:
        change: Raw change dictionary from model

    Returns:
        Validated SemanticChange or None if invalid
    """
    valid_types = {"geometry", "texture", "missing", "artifact", "semantic"}
    valid_severities = {"low", "medium", "high"}

    ctype = change.get("type", "").lower()
    severity = change.get("severity", "").lower()
    description = change.get("description", "")

    # Validate required fields
    if ctype not in valid_types:
        logger.warning("Invalid change type: %s", ctype)
        ctype = "semantic"  # Default fallback

    if severity not in valid_severities:
        logger.warning("Invalid severity: %s", severity)
        severity = "medium"  # Default fallback

    if not description:
        return None

    return SemanticChange(
        type=ctype,
        severity=severity,
        description=description,
        location=change.get("location"),
        confidence=float(change.get("confidence", 1.0)),
    )


def semantic_diff(
    *,
    backend: "LlavaQualityBackend",
    image_a: Path,
    image_b: Path,
) -> SemanticDiffResult:
    """Perform semantic diff between two images using LLaVA.

    This function uses a vision-language model to analyze and explain
    the differences between two images in a structured, actionable format.

    Args:
        backend: LLaVA backend instance for inference
        image_a: Path to first image (before/reference)
        image_b: Path to second image (after/comparison)

    Returns:
        SemanticDiffResult with structured analysis

    Raises:
        SemanticDiffError: If analysis fails critically

    Example:
        >>> result = semantic_diff(
        ...     backend=llava,
        ...     image_a=Path("render_v1.png"),
        ...     image_b=Path("render_v2.png"),
        ... )
        >>> print(result.summary)
        "Texture quality improved, minor geometry artifacts remain."
    """
    # Validate inputs
    if not image_a.exists():
        raise SemanticDiffError(f"Image A not found: {image_a}")
    if not image_b.exists():
        raise SemanticDiffError(f"Image B not found: {image_b}")

    logger.info("Running semantic diff: %s vs %s", image_a.name, image_b.name)

    # Build messages and run inference
    messages = _build_messages(image_a, image_b)

    try:
        raw = backend._run_inference(messages=messages)
    except Exception as e:
        logger.error("Semantic diff inference failed: %s", e)
        raise SemanticDiffError(f"Inference failed: {e}") from e

    # Parse response
    payload = _parse_response(raw)

    # Extract and validate changes
    raw_changes = payload.get("changes", [])
    validated_changes: list[SemanticChange] = []

    for raw_change in raw_changes:
        change = _validate_change(raw_change)
        if change is not None:
            validated_changes.append(change)

    # Compute hashes for lineage
    import hashlib

    def file_hash(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()[:16]

    return SemanticDiffResult(
        summary=payload.get("summary", "Analysis complete."),
        changes=tuple(validated_changes),
        raw_text=raw,
        structured=payload,
        image_a_hash=file_hash(image_a),
        image_b_hash=file_hash(image_b),
    )


def semantic_diff_from_hashes(
    *,
    backend: "LlavaQualityBackend",
    hash_a: str,
    hash_b: str,
    cas_root: Path,
) -> SemanticDiffResult:
    """Perform semantic diff using CAS object hashes.

    Convenience function that resolves CAS hashes to paths before
    performing semantic diff.

    Args:
        backend: LLaVA backend instance
        hash_a: SHA-256 hash of first image in CAS
        hash_b: SHA-256 hash of second image in CAS
        cas_root: Root directory of CAS storage

    Returns:
        SemanticDiffResult with structured analysis
    """

    def resolve(h: str) -> Path:
        return cas_root / "objects" / h[:2] / h

    path_a = resolve(hash_a)
    path_b = resolve(hash_b)

    return semantic_diff(backend=backend, image_a=path_a, image_b=path_b)


# Severity scoring for integration with APEX harness
SEVERITY_WEIGHTS = {
    "low": 0.1,
    "medium": 0.3,
    "high": 0.6,
}


def compute_diff_penalty(result: SemanticDiffResult) -> float:
    """Compute quality penalty score from semantic diff result.

    Args:
        result: Semantic diff result

    Returns:
        Penalty score (0.0 = no issues, higher = more issues)
    """
    penalty = 0.0
    for change in result.changes:
        weight = SEVERITY_WEIGHTS.get(change.severity, 0.3)
        penalty += weight * change.confidence
    return min(penalty, 1.0)  # Cap at 1.0
