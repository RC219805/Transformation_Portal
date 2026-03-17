"""RL policy guard: Safety gates for action selection.

This module provides safety policies to filter actions before
application, ensuring the RL optimizer doesn't apply dangerous
or invalid modifications.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from transformation_portal.rl.action_space import RLAction

logger = logging.getLogger(__name__)


# Safe action types that can be auto-applied
SAFE_ACTIONS = {
    "increase_mask_coverage",
    "expand_prompt_set",
    "increase_iterations",
    "enable_seam_blending",
    "apply_denoising",
    "adjust_roughness_prior",
    "adjust_metalness_prior",
    "adjust_tone_curve",
    "increase_resolution",
}

# Risky actions that require extra validation
RISKY_ACTIONS = {
    "increase_mesh_resolution",  # Can be expensive
    "adjust_texture_quality",  # May change output significantly
}

# Blocked actions (never allowed)
BLOCKED_ACTIONS = {
    "delete_node",
    "reset_config",
}


@dataclass
class PolicyDecision:
    """Decision from policy evaluation."""

    allowed: bool
    reason: str
    warnings: list[str] = field(default_factory=list)


@dataclass
class RLPolicyConfig:
    """Configuration for RL policy guard."""

    safe_actions: set[str] = field(default_factory=lambda: SAFE_ACTIONS.copy())
    risky_actions: set[str] = field(default_factory=lambda: RISKY_ACTIONS.copy())
    blocked_actions: set[str] = field(default_factory=lambda: BLOCKED_ACTIONS.copy())
    allow_risky: bool = False
    max_iterations_increase: int = 1000
    max_resolution_scale: float = 3.0
    max_bias_magnitude: float = 0.5


def is_safe(action: RLAction, config: RLPolicyConfig | None = None) -> bool:
    """Check if action is safe to apply.

    Args:
        action: RLAction to check
        config: Policy configuration

    Returns:
        True if action is safe
    """
    config = config or RLPolicyConfig()
    return action.action_type in config.safe_actions


def is_blocked(action: RLAction, config: RLPolicyConfig | None = None) -> bool:
    """Check if action is blocked.

    Args:
        action: RLAction to check
        config: Policy configuration

    Returns:
        True if action is blocked
    """
    config = config or RLPolicyConfig()
    return action.action_type in config.blocked_actions


def validate_params(action: RLAction, config: RLPolicyConfig | None = None) -> PolicyDecision:
    """Validate action parameters.

    Args:
        action: RLAction to validate
        config: Policy configuration

    Returns:
        PolicyDecision with validation result
    """
    config = config or RLPolicyConfig()
    warnings = []

    # Check iterations
    if "steps" in action.params:
        steps = action.params["steps"]
        if steps > config.max_iterations_increase:
            return PolicyDecision(
                allowed=False,
                reason=f"Steps {steps} exceeds max {config.max_iterations_increase}",
            )
        if steps > 500:
            warnings.append(f"High iteration count: {steps}")

    # Check resolution
    if "scale" in action.params:
        scale = action.params["scale"]
        if scale > config.max_resolution_scale:
            return PolicyDecision(
                allowed=False,
                reason=f"Scale {scale} exceeds max {config.max_resolution_scale}",
            )

    # Check bias values
    if "bias" in action.params:
        bias = abs(action.params["bias"])
        if bias > config.max_bias_magnitude:
            return PolicyDecision(
                allowed=False,
                reason=f"Bias magnitude {bias} exceeds max {config.max_bias_magnitude}",
            )

    return PolicyDecision(
        allowed=True,
        reason="Parameters valid",
        warnings=warnings,
    )


def evaluate_action(
    action: RLAction,
    config: RLPolicyConfig | None = None,
) -> PolicyDecision:
    """Evaluate action against policy.

    Args:
        action: RLAction to evaluate
        config: Policy configuration

    Returns:
        PolicyDecision with evaluation result
    """
    config = config or RLPolicyConfig()

    # Check blocked
    if is_blocked(action, config):
        return PolicyDecision(
            allowed=False,
            reason=f"Action '{action.action_type}' is blocked",
        )

    # Check risky
    if action.action_type in config.risky_actions:
        if not config.allow_risky:
            return PolicyDecision(
                allowed=False,
                reason=f"Action '{action.action_type}' is risky and risky actions not allowed",
            )

    # Check safe
    if not is_safe(action, config):
        return PolicyDecision(
            allowed=False,
            reason=f"Action '{action.action_type}' not in safe actions list",
        )

    # Validate parameters
    return validate_params(action, config)


def filter_actions(
    actions: list[RLAction],
    config: RLPolicyConfig | None = None,
) -> list[RLAction]:
    """Filter actions to only those allowed by policy.

    Args:
        actions: List of actions to filter
        config: Policy configuration

    Returns:
        Filtered list of allowed actions
    """
    config = config or RLPolicyConfig()
    allowed = []

    for action in actions:
        decision = evaluate_action(action, config)
        if decision.allowed:
            allowed.append(action)
            if decision.warnings:
                logger.warning(
                    "Action %s has warnings: %s",
                    action.action_type,
                    ", ".join(decision.warnings),
                )
        else:
            logger.debug(
                "Filtered action %s: %s",
                action.action_type,
                decision.reason,
            )

    return allowed


def create_action_mask(
    actions: list[RLAction],
    config: RLPolicyConfig | None = None,
) -> list[bool]:
    """Create boolean mask of allowed actions.

    Args:
        actions: List of all actions
        config: Policy configuration

    Returns:
        Boolean mask (True = allowed)
    """
    config = config or RLPolicyConfig()
    return [evaluate_action(a, config).allowed for a in actions]
