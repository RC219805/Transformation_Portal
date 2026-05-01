"""Tests for rl.policy_guard — safety-critical allowlist/blocklist enforcement."""

from __future__ import annotations

import pytest

from transformation_portal.rl.action_space import RLAction
from transformation_portal.rl.policy_guard import (
    BLOCKED_ACTIONS,
    RISKY_ACTIONS,
    SAFE_ACTIONS,
    PolicyDecision,
    RLPolicyConfig,
    create_action_mask,
    evaluate_action,
    filter_actions,
    is_blocked,
    is_safe,
    validate_params,
)

pytestmark = pytest.mark.unit


def _action(action_type: str, params: dict | None = None, index: int = 0) -> RLAction:
    return RLAction(node="sam2", action_type=action_type, params=params or {}, index=index)


class TestConstants:
    def test_safe_actions_is_set(self):
        """SAFE_ACTIONS is a non-empty set."""
        assert isinstance(SAFE_ACTIONS, set)
        assert len(SAFE_ACTIONS) > 0

    def test_blocked_actions_disjoint_from_safe(self):
        """Blocked and safe action sets are mutually exclusive."""
        assert BLOCKED_ACTIONS & SAFE_ACTIONS == set()

    def test_risky_actions_disjoint_from_blocked(self):
        """Risky and blocked action sets are mutually exclusive."""
        assert RISKY_ACTIONS & BLOCKED_ACTIONS == set()

    def test_known_blocked_action_present(self):
        """delete_node and reset_config are always blocked."""
        assert "delete_node" in BLOCKED_ACTIONS
        assert "reset_config" in BLOCKED_ACTIONS


class TestPolicyDecision:
    def test_allowed_true(self):
        """PolicyDecision stores allowed=True."""
        d = PolicyDecision(allowed=True, reason="ok")
        assert d.allowed is True

    def test_allowed_false(self):
        """PolicyDecision stores allowed=False."""
        d = PolicyDecision(allowed=False, reason="blocked")
        assert d.allowed is False

    def test_default_warnings_empty(self):
        """Warnings list defaults to empty."""
        d = PolicyDecision(allowed=True, reason="ok")
        assert d.warnings == []

    def test_warnings_stored(self):
        """Warnings list is stored correctly."""
        d = PolicyDecision(allowed=True, reason="ok", warnings=["high steps"])
        assert d.warnings == ["high steps"]


class TestRLPolicyConfig:
    def test_defaults_match_module_constants(self):
        """Default config mirrors module-level constants."""
        cfg = RLPolicyConfig()
        assert cfg.safe_actions == SAFE_ACTIONS
        assert cfg.blocked_actions == BLOCKED_ACTIONS
        assert cfg.risky_actions == RISKY_ACTIONS

    def test_allow_risky_default_false(self):
        """allow_risky defaults to False."""
        assert RLPolicyConfig().allow_risky is False

    def test_custom_safe_actions(self):
        """Custom safe_actions replaces default."""
        cfg = RLPolicyConfig(safe_actions={"my_action"})
        assert cfg.safe_actions == {"my_action"}


class TestIsSafe:
    def test_known_safe_action_returns_true(self):
        """A standard safe action is recognised."""
        assert is_safe(_action("increase_mask_coverage")) is True

    def test_unknown_action_returns_false(self):
        """An unknown action is not safe."""
        assert is_safe(_action("unknown_action")) is False

    def test_blocked_action_not_safe(self):
        """A blocked action is not safe."""
        assert is_safe(_action("delete_node")) is False

    def test_custom_config_safe_set(self):
        """Custom config expands the safe set."""
        cfg = RLPolicyConfig(safe_actions={"custom_action"})
        assert is_safe(_action("custom_action"), cfg) is True
        assert is_safe(_action("increase_mask_coverage"), cfg) is False


class TestIsBlocked:
    def test_blocked_action_returns_true(self):
        """delete_node is blocked."""
        assert is_blocked(_action("delete_node")) is True

    def test_reset_config_is_blocked(self):
        """reset_config is blocked."""
        assert is_blocked(_action("reset_config")) is True

    def test_safe_action_not_blocked(self):
        """A safe action is not blocked."""
        assert is_blocked(_action("increase_mask_coverage")) is False

    def test_custom_config_blocked_set(self):
        """Custom config changes blocked set."""
        cfg = RLPolicyConfig(blocked_actions={"my_blocked"})
        assert is_blocked(_action("my_blocked"), cfg) is True
        assert is_blocked(_action("delete_node"), cfg) is False


class TestValidateParams:
    def test_no_relevant_params_allowed(self):
        """Empty params are always valid."""
        d = validate_params(_action("apply_denoising", {}))
        assert d.allowed is True

    def test_steps_within_limit_allowed(self):
        """steps=500 is within default max."""
        d = validate_params(_action("increase_iterations", {"steps": 500}))
        assert d.allowed is True

    def test_steps_exceeds_limit_blocked(self):
        """steps exceeding max_iterations_increase is rejected."""
        d = validate_params(_action("increase_iterations", {"steps": 1001}))
        assert d.allowed is False
        assert "1001" in d.reason

    def test_steps_over_500_generates_warning(self):
        """steps=600 is allowed but generates a warning."""
        d = validate_params(_action("increase_iterations", {"steps": 600}))
        assert d.allowed is True
        assert len(d.warnings) > 0

    def test_scale_within_limit_allowed(self):
        """scale=2.0 is within default max."""
        d = validate_params(_action("increase_resolution", {"scale": 2.0}))
        assert d.allowed is True

    def test_scale_exceeds_limit_blocked(self):
        """scale exceeding max_resolution_scale is rejected."""
        d = validate_params(_action("increase_resolution", {"scale": 4.0}))
        assert d.allowed is False

    def test_bias_within_limit_allowed(self):
        """bias=0.3 is within default max_bias_magnitude."""
        d = validate_params(_action("adjust_roughness_prior", {"bias": 0.3}))
        assert d.allowed is True

    def test_bias_exceeds_limit_blocked(self):
        """bias magnitude exceeding max is rejected."""
        d = validate_params(_action("adjust_roughness_prior", {"bias": 0.6}))
        assert d.allowed is False

    def test_negative_bias_magnitude_checked(self):
        """Negative bias uses abs() for comparison."""
        d = validate_params(_action("adjust_roughness_prior", {"bias": -0.6}))
        assert d.allowed is False

    def test_custom_config_limits(self):
        """Custom config changes limits."""
        cfg = RLPolicyConfig(max_iterations_increase=50)
        d = validate_params(_action("increase_iterations", {"steps": 51}), cfg)
        assert d.allowed is False


class TestEvaluateAction:
    def test_blocked_action_denied(self):
        """delete_node is always denied."""
        d = evaluate_action(_action("delete_node"))
        assert d.allowed is False
        assert "blocked" in d.reason.lower()

    def test_risky_action_denied_by_default(self):
        """Risky actions are denied when allow_risky=False."""
        d = evaluate_action(_action("increase_mesh_resolution"))
        assert d.allowed is False

    def test_risky_action_allowed_with_flag(self):
        """Risky actions pass the risky gate when allow_risky=True (and also in safe_actions)."""
        cfg = RLPolicyConfig(
            allow_risky=True,
            safe_actions=SAFE_ACTIONS | {"increase_mesh_resolution"},
        )
        d = evaluate_action(_action("increase_mesh_resolution"), cfg)
        assert d.allowed is True

    def test_safe_action_permitted(self):
        """A standard safe action with valid params is permitted."""
        d = evaluate_action(_action("apply_denoising"))
        assert d.allowed is True

    def test_unknown_action_denied(self):
        """An action not in any list is denied."""
        d = evaluate_action(_action("mystery_action"))
        assert d.allowed is False

    def test_safe_action_bad_params_denied(self):
        """A safe action with out-of-range params is denied."""
        d = evaluate_action(_action("increase_iterations", {"steps": 9999}))
        assert d.allowed is False

    def test_reset_config_always_denied(self):
        """reset_config is always denied regardless of config."""
        cfg = RLPolicyConfig(allow_risky=True)
        d = evaluate_action(_action("reset_config"), cfg)
        assert d.allowed is False


class TestFilterActions:
    def test_empty_list_returns_empty(self):
        """filter_actions([]) → []."""
        assert filter_actions([]) == []

    def test_keeps_safe_actions(self):
        """All safe actions survive filtering."""
        actions = [_action(a, index=i) for i, a in enumerate(sorted(SAFE_ACTIONS))]
        result = filter_actions(actions)
        assert len(result) == len(actions)

    def test_removes_blocked_actions(self):
        """Blocked actions are removed."""
        actions = [_action("delete_node", index=0), _action("apply_denoising", index=1)]
        result = filter_actions(actions)
        assert len(result) == 1
        assert result[0].action_type == "apply_denoising"

    def test_removes_risky_by_default(self):
        """Risky actions are removed by default."""
        actions = [_action("increase_mesh_resolution", index=0), _action("apply_denoising", index=1)]
        result = filter_actions(actions)
        assert all(a.action_type != "increase_mesh_resolution" for a in result)

    def test_mixed_list_filters_correctly(self):
        """Mixed list: only safe actions with valid params pass."""
        actions = [
            _action("delete_node", index=0),
            _action("apply_denoising", index=1),
            _action("increase_mask_coverage", index=2),
        ]
        result = filter_actions(actions)
        types = {a.action_type for a in result}
        assert "delete_node" not in types
        assert "apply_denoising" in types
        assert "increase_mask_coverage" in types

    def test_all_blocked_returns_empty(self):
        """All blocked actions → empty result."""
        actions = [_action("delete_node", index=0), _action("reset_config", index=1)]
        assert filter_actions(actions) == []


class TestCreateActionMask:
    def test_mask_length_matches_input(self):
        """Mask has same length as input list."""
        actions = [_action(a, index=i) for i, a in enumerate(["apply_denoising", "delete_node", "increase_mask_coverage"])]
        mask = create_action_mask(actions)
        assert len(mask) == 3

    def test_blocked_action_is_false(self):
        """delete_node entry is False."""
        actions = [_action("delete_node")]
        mask = create_action_mask(actions)
        assert mask[0] is False

    def test_safe_action_is_true(self):
        """apply_denoising entry is True."""
        actions = [_action("apply_denoising")]
        mask = create_action_mask(actions)
        assert mask[0] is True

    def test_mask_contains_only_booleans(self):
        """Every mask entry is a bool."""
        actions = [_action(a, index=i) for i, a in enumerate(["apply_denoising", "delete_node"])]
        mask = create_action_mask(actions)
        assert all(isinstance(v, bool) for v in mask)

    def test_empty_list_returns_empty_mask(self):
        """Empty input → empty mask."""
        assert create_action_mask([]) == []
