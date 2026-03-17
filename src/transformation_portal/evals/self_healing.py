"""Self-healing pipeline: Fix suggestion engine.

This module provides structured fix suggestions based on semantic diff
analysis and evaluation metrics. It maps detected issues to actionable
pipeline configuration changes.

Integrates with:
- semantic_diff module
- APEX evaluation harness
- DAG patcher
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FixSuggestion:
    """A suggested fix for a detected issue.

    Attributes:
        type: Category of fix (segmentation, reconstruction, material, etc.)
        target_node: Node ID to apply fix to
        action: Specific action to take
        params: Parameters for the action
        confidence: Confidence in this fix (0.0-1.0)
        rationale: Explanation of why this fix is suggested
        priority: Priority level (higher = more important)
        reversible: Whether this fix can be easily reverted
    """

    type: str
    target_node: str
    action: str
    params: dict[str, Any]
    confidence: float
    rationale: str
    priority: int = 0
    reversible: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "type": self.type,
            "target_node": self.target_node,
            "action": self.action,
            "params": self.params,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "priority": self.priority,
            "reversible": self.reversible,
        }


@dataclass
class FixSuggestionSet:
    """Collection of fix suggestions with metadata."""

    suggestions: list[FixSuggestion] = field(default_factory=list)
    source_diff_summary: str = ""
    source_metrics: dict[str, float] = field(default_factory=dict)

    def add(self, suggestion: FixSuggestion) -> None:
        """Add a suggestion to the set."""
        self.suggestions.append(suggestion)

    @property
    def high_confidence(self) -> list[FixSuggestion]:
        """Get high-confidence suggestions (>= 0.7)."""
        return [s for s in self.suggestions if s.confidence >= 0.7]

    @property
    def by_priority(self) -> list[FixSuggestion]:
        """Get suggestions sorted by priority (highest first)."""
        return sorted(self.suggestions, key=lambda s: -s.priority)

    @property
    def by_node(self) -> dict[str, list[FixSuggestion]]:
        """Group suggestions by target node."""
        result: dict[str, list[FixSuggestion]] = {}
        for s in self.suggestions:
            result.setdefault(s.target_node, []).append(s)
        return result

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "suggestions": [s.to_dict() for s in self.suggestions],
            "source_diff_summary": self.source_diff_summary,
            "source_metrics": self.source_metrics,
            "count": len(self.suggestions),
            "high_confidence_count": len(self.high_confidence),
        }


# Fix suggestion rules
# These map (change_type, severity) -> suggested fixes


def _suggest_for_missing(
    change: dict[str, Any],
    severity: str,
) -> list[FixSuggestion]:
    """Suggest fixes for missing regions."""
    fixes = []

    if severity in ("medium", "high"):
        # Increase segmentation coverage
        fixes.append(
            FixSuggestion(
                type="segmentation",
                target_node="sam2",
                action="increase_mask_coverage",
                params={"threshold": 0.3, "iou_threshold": 0.5},
                confidence=0.8,
                rationale=change.get("description", "Missing region detected"),
                priority=8,
            )
        )

        # Try different segmentation prompts
        fixes.append(
            FixSuggestion(
                type="segmentation",
                target_node="sam2",
                action="expand_prompt_set",
                params={"include_negative": True},
                confidence=0.6,
                rationale="Expand prompts to capture missing regions",
                priority=5,
            )
        )

    return fixes


def _suggest_for_artifact(
    change: dict[str, Any],
    severity: str,
) -> list[FixSuggestion]:
    """Suggest fixes for artifacts (noise, seams, distortions)."""
    fixes = []

    description = change.get("description", "").lower()

    if severity == "high":
        # Increase reconstruction quality
        fixes.append(
            FixSuggestion(
                type="reconstruction",
                target_node="nvdiffrec",
                action="increase_iterations",
                params={"steps": 500, "refine": True},
                confidence=0.75,
                rationale=change.get("description", "Reconstruction artifact"),
                priority=9,
            )
        )

    if "seam" in description or "edge" in description:
        fixes.append(
            FixSuggestion(
                type="postprocess",
                target_node="postprocess",
                action="enable_seam_blending",
                params={"blend_radius": 8},
                confidence=0.7,
                rationale="Seam artifact detected",
                priority=7,
            )
        )

    if "noise" in description or "grain" in description:
        fixes.append(
            FixSuggestion(
                type="postprocess",
                target_node="postprocess",
                action="apply_denoising",
                params={"strength": 0.5},
                confidence=0.65,
                rationale="Noise artifact detected",
                priority=6,
            )
        )

    return fixes


def _suggest_for_texture(
    change: dict[str, Any],
    severity: str,
) -> list[FixSuggestion]:
    """Suggest fixes for texture/material issues."""
    fixes = []

    description = change.get("description", "").lower()

    if severity in ("medium", "high"):
        # Adjust material priors
        fixes.append(
            FixSuggestion(
                type="material",
                target_node="material_backend",
                action="adjust_roughness_prior",
                params={"bias": 0.1},
                confidence=0.6,
                rationale=change.get("description", "Texture issue"),
                priority=5,
            )
        )

    if "color" in description or "tone" in description:
        fixes.append(
            FixSuggestion(
                type="color",
                target_node="color_grading",
                action="adjust_tone_curve",
                params={"contrast": 1.05},
                confidence=0.55,
                rationale="Color/tone issue detected",
                priority=4,
            )
        )

    if "metallic" in description or "reflective" in description:
        fixes.append(
            FixSuggestion(
                type="material",
                target_node="material_backend",
                action="adjust_metalness_prior",
                params={"bias": -0.1},
                confidence=0.5,
                rationale="Metalness issue detected",
                priority=4,
            )
        )

    return fixes


def _suggest_for_geometry(
    change: dict[str, Any],
    severity: str,
) -> list[FixSuggestion]:
    """Suggest fixes for geometry issues."""
    fixes = []

    if severity == "high":
        # Re-run depth estimation with higher quality
        fixes.append(
            FixSuggestion(
                type="depth",
                target_node="depth_backend",
                action="increase_resolution",
                params={"scale": 2.0},
                confidence=0.7,
                rationale=change.get("description", "Geometry error"),
                priority=8,
            )
        )

        # Increase mesh refinement
        fixes.append(
            FixSuggestion(
                type="reconstruction",
                target_node="nvdiffrec",
                action="increase_mesh_resolution",
                params={"subdivisions": 2},
                confidence=0.65,
                rationale="Increase mesh detail to fix geometry",
                priority=7,
            )
        )

    return fixes


def _suggest_for_semantic(
    change: dict[str, Any],
    severity: str,
) -> list[FixSuggestion]:
    """Suggest fixes for semantic issues."""
    fixes = []

    if severity == "high":
        # Flag for manual review
        fixes.append(
            FixSuggestion(
                type="review",
                target_node="quality_gate",
                action="flag_for_review",
                params={"reason": change.get("description", "Semantic issue")},
                confidence=0.9,
                rationale="Semantic issue requires human review",
                priority=10,
                reversible=True,
            )
        )

    return fixes


def suggest_fixes(
    semantic_diff: dict[str, Any],
    metrics: dict[str, float] | None = None,
) -> FixSuggestionSet:
    """Generate fix suggestions from semantic diff and metrics.

    Args:
        semantic_diff: Structured semantic diff output
        metrics: Optional evaluation metrics (PSNR, SSIM, etc.)

    Returns:
        FixSuggestionSet with prioritized suggestions

    Example:
        >>> diff = {"changes": [{"type": "missing", "severity": "high", ...}]}
        >>> suggestions = suggest_fixes(diff, {"psnr": 25.0})
        >>> for s in suggestions.high_confidence:
        ...     print(f"{s.target_node}: {s.action}")
    """
    metrics = metrics or {}
    result = FixSuggestionSet(
        source_diff_summary=semantic_diff.get("summary", ""),
        source_metrics=metrics,
    )

    # Map change types to suggestion functions
    type_handlers = {
        "missing": _suggest_for_missing,
        "artifact": _suggest_for_artifact,
        "texture": _suggest_for_texture,
        "geometry": _suggest_for_geometry,
        "semantic": _suggest_for_semantic,
    }

    # Process each change
    for change in semantic_diff.get("changes", []):
        ctype = change.get("type", "").lower()
        severity = change.get("severity", "medium").lower()

        handler = type_handlers.get(ctype)
        if handler:
            fixes = handler(change, severity)
            for fix in fixes:
                result.add(fix)

    # Add metric-based suggestions
    if metrics:
        _add_metric_suggestions(result, metrics)

    logger.info(
        "Generated %d fix suggestions (%d high-confidence)",
        len(result.suggestions),
        len(result.high_confidence),
    )

    return result


def _add_metric_suggestions(
    result: FixSuggestionSet,
    metrics: dict[str, float],
) -> None:
    """Add suggestions based on evaluation metrics."""
    # Low PSNR -> increase quality
    psnr = metrics.get("psnr", 100.0)
    if psnr < 25.0:
        result.add(
            FixSuggestion(
                type="quality",
                target_node="nvdiffrec",
                action="increase_iterations",
                params={"steps": 1000},
                confidence=0.7,
                rationale=f"Low PSNR ({psnr:.1f} dB) indicates reconstruction issues",
                priority=8,
            )
        )

    # Low SSIM -> structural issues
    ssim = metrics.get("ssim", 1.0)
    if ssim < 0.85:
        result.add(
            FixSuggestion(
                type="structure",
                target_node="depth_backend",
                action="increase_resolution",
                params={"scale": 1.5},
                confidence=0.65,
                rationale=f"Low SSIM ({ssim:.3f}) indicates structural differences",
                priority=7,
            )
        )

    # High LPIPS -> perceptual issues
    lpips = metrics.get("lpips", 0.0)
    if lpips > 0.3:
        result.add(
            FixSuggestion(
                type="perceptual",
                target_node="material_backend",
                action="adjust_texture_quality",
                params={"detail_level": "high"},
                confidence=0.6,
                rationale=f"High LPIPS ({lpips:.3f}) indicates perceptual differences",
                priority=6,
            )
        )
