"""Self-healing pipeline: Policy and safety layer.

This module defines policies for which fixes can be automatically applied
versus which require human approval. It provides guardrails for the
self-healing system.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from transformation_portal.evals.self_healing import FixSuggestion

logger = logging.getLogger(__name__)


class ApprovalLevel(Enum):
    """Approval level required for a fix."""

    AUTO = "auto"  # Can be applied automatically
    REVIEW = "review"  # Requires human review before apply
    MANUAL = "manual"  # Must be applied manually


@dataclass(frozen=True)
class PolicyDecision:
    """Decision from policy evaluation."""

    can_auto_apply: bool
    approval_level: ApprovalLevel
    reason: str
    warnings: tuple[str, ...] = ()


@dataclass
class SelfHealPolicy:
    """Policy configuration for self-healing pipeline.

    Attributes:
        safe_actions: Actions that can be auto-applied
        confidence_threshold: Minimum confidence for auto-apply
        max_auto_fixes_per_run: Limit on auto-applied fixes
        require_validation: Whether to re-run validation after fix
        allowed_nodes: Nodes that can be modified (None = all)
        blocked_nodes: Nodes that cannot be modified
    """

    safe_actions: set[str] = field(
        default_factory=lambda: {
            "increase_mask_coverage",
            "increase_iterations",
            "expand_prompt_set",
            "enable_seam_blending",
            "apply_denoising",
            "adjust_roughness_prior",
            "adjust_metalness_prior",
            "adjust_tone_curve",
            "increase_resolution",
        }
    )
    confidence_threshold: float = 0.75
    max_auto_fixes_per_run: int = 5
    require_validation: bool = True
    allowed_nodes: set[str] | None = None
    blocked_nodes: set[str] = field(
        default_factory=lambda: {
            "quality_gate",  # Review nodes should not be auto-modified
            "export",  # Export nodes are sensitive
        }
    )

    def evaluate(self, fix: FixSuggestion) -> PolicyDecision:
        """Evaluate whether a fix can be auto-applied.

        Args:
            fix: Fix suggestion to evaluate

        Returns:
            PolicyDecision with approval level and reason
        """
        warnings: list[str] = []

        # Check blocked nodes
        if fix.target_node in self.blocked_nodes:
            return PolicyDecision(
                can_auto_apply=False,
                approval_level=ApprovalLevel.MANUAL,
                reason=f"Node '{fix.target_node}' is blocked from auto-modification",
            )

        # Check allowed nodes (if specified)
        if self.allowed_nodes is not None:
            if fix.target_node not in self.allowed_nodes:
                return PolicyDecision(
                    can_auto_apply=False,
                    approval_level=ApprovalLevel.REVIEW,
                    reason=f"Node '{fix.target_node}' not in allowed list",
                )

        # Check if action is safe
        if fix.action not in self.safe_actions:
            return PolicyDecision(
                can_auto_apply=False,
                approval_level=ApprovalLevel.REVIEW,
                reason=f"Action '{fix.action}' is not in safe actions list",
            )

        # Check confidence threshold
        if fix.confidence < self.confidence_threshold:
            return PolicyDecision(
                can_auto_apply=False,
                approval_level=ApprovalLevel.REVIEW,
                reason=(f"Confidence {fix.confidence:.2f} below threshold " f"{self.confidence_threshold:.2f}"),
            )

        # Check reversibility
        if not fix.reversible:
            warnings.append("Fix is not easily reversible")

        # All checks passed
        return PolicyDecision(
            can_auto_apply=True,
            approval_level=ApprovalLevel.AUTO,
            reason="Fix meets all policy requirements",
            warnings=tuple(warnings),
        )


# Default policy instance
DEFAULT_POLICY = SelfHealPolicy()


def can_auto_apply(fix: FixSuggestion, policy: SelfHealPolicy | None = None) -> bool:
    """Quick check if a fix can be auto-applied.

    Args:
        fix: Fix suggestion to check
        policy: Policy to use (defaults to DEFAULT_POLICY)

    Returns:
        True if fix can be auto-applied
    """
    policy = policy or DEFAULT_POLICY
    decision = policy.evaluate(fix)
    return decision.can_auto_apply


def filter_auto_applicable(
    fixes: list[FixSuggestion],
    policy: SelfHealPolicy | None = None,
    max_fixes: int | None = None,
) -> list[FixSuggestion]:
    """Filter fixes to only those that can be auto-applied.

    Args:
        fixes: List of fix suggestions
        policy: Policy to use
        max_fixes: Maximum number of fixes to return

    Returns:
        List of auto-applicable fixes, sorted by priority
    """
    policy = policy or DEFAULT_POLICY
    max_fixes = max_fixes or policy.max_auto_fixes_per_run

    applicable = []
    for fix in fixes:
        decision = policy.evaluate(fix)
        if decision.can_auto_apply:
            applicable.append(fix)
            if decision.warnings:
                logger.warning(
                    "Fix %s has warnings: %s",
                    fix.action,
                    ", ".join(decision.warnings),
                )

    # Sort by priority and limit
    applicable.sort(key=lambda f: -f.priority)
    return applicable[:max_fixes]


@dataclass
class PolicyReport:
    """Report of policy evaluation for a set of fixes."""

    total_fixes: int
    auto_applicable: int
    needs_review: int
    blocked: int
    decisions: list[tuple[FixSuggestion, PolicyDecision]]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_fixes": self.total_fixes,
            "auto_applicable": self.auto_applicable,
            "needs_review": self.needs_review,
            "blocked": self.blocked,
            "breakdown": [
                {
                    "fix": fix.to_dict(),
                    "decision": {
                        "can_auto_apply": dec.can_auto_apply,
                        "approval_level": dec.approval_level.value,
                        "reason": dec.reason,
                        "warnings": list(dec.warnings),
                    },
                }
                for fix, dec in self.decisions
            ],
        }


def evaluate_all(
    fixes: list[FixSuggestion],
    policy: SelfHealPolicy | None = None,
) -> PolicyReport:
    """Evaluate all fixes against policy.

    Args:
        fixes: List of fix suggestions
        policy: Policy to use

    Returns:
        PolicyReport with detailed breakdown
    """
    policy = policy or DEFAULT_POLICY

    decisions = [(fix, policy.evaluate(fix)) for fix in fixes]

    return PolicyReport(
        total_fixes=len(fixes),
        auto_applicable=sum(1 for _, d in decisions if d.can_auto_apply),
        needs_review=sum(1 for _, d in decisions if d.approval_level == ApprovalLevel.REVIEW),
        blocked=sum(1 for _, d in decisions if d.approval_level == ApprovalLevel.MANUAL),
        decisions=decisions,
    )
