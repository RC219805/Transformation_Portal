#!/usr/bin/env python3
"""Behavior-slice tests for app.py feature-flag and rollout helpers.

Covers env parsing helpers (_env_bool/_env_int/_env_float/_env_csv/_env_rollout_percent),
stable rollout bucketing, cohort-key resolution, and the portal feature-flag
predicates. These were previously exercised only indirectly via integration tests
(see docs/testing/test_coverage_improvement_plan.md Phase 1).
"""

from __future__ import annotations

import importlib

import pytest

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")


# ---------------------------------------------------------------------------
# _env_bool
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["1", "true", "TRUE", "Yes", "on", " on "])
def test_env_bool_truthy_values(monkeypatch, raw):
    monkeypatch.setenv("TP_TEST_FLAG", raw)
    assert orchestrator_app._env_bool("TP_TEST_FLAG", default=False) is True


@pytest.mark.parametrize("raw", ["0", "false", "no", "off", "", "garbage"])
def test_env_bool_falsy_values(monkeypatch, raw):
    monkeypatch.setenv("TP_TEST_FLAG", raw)
    assert orchestrator_app._env_bool("TP_TEST_FLAG", default=True) is False


def test_env_bool_uses_default_when_unset(monkeypatch):
    monkeypatch.delenv("TP_TEST_FLAG", raising=False)
    assert orchestrator_app._env_bool("TP_TEST_FLAG", default=True) is True
    assert orchestrator_app._env_bool("TP_TEST_FLAG", default=False) is False


# ---------------------------------------------------------------------------
# _env_int / _env_float
# ---------------------------------------------------------------------------


def test_env_int_clamps_to_minimum(monkeypatch):
    monkeypatch.setenv("TP_TEST_INT", "-5")
    assert orchestrator_app._env_int("TP_TEST_INT", default=0, minimum=0) == 0


def test_env_int_invalid_string_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("TP_TEST_INT", "not-a-number")
    assert orchestrator_app._env_int("TP_TEST_INT", default=42) == 42


def test_env_int_unset_returns_default(monkeypatch):
    monkeypatch.delenv("TP_TEST_INT", raising=False)
    assert orchestrator_app._env_int("TP_TEST_INT", default=7) == 7


def test_env_float_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("TP_TEST_FLOAT", "abc")
    assert orchestrator_app._env_float("TP_TEST_FLOAT", default=1.5) == 1.5


def test_env_float_clamps_to_minimum(monkeypatch):
    monkeypatch.setenv("TP_TEST_FLOAT", "-2.0")
    assert orchestrator_app._env_float("TP_TEST_FLOAT", default=0.0, minimum=0.0) == 0.0


# ---------------------------------------------------------------------------
# _env_csv
# ---------------------------------------------------------------------------


def test_env_csv_parses_and_trims(monkeypatch):
    monkeypatch.setenv("TP_TEST_CSV", " a , b ,, c")
    assert orchestrator_app._env_csv("TP_TEST_CSV", default=["fallback"]) == ["a", "b", "c"]


def test_env_csv_unset_returns_default(monkeypatch):
    monkeypatch.delenv("TP_TEST_CSV", raising=False)
    assert orchestrator_app._env_csv("TP_TEST_CSV", default=["x"]) == ["x"]


# ---------------------------------------------------------------------------
# _env_rollout_percent
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [("0", 0), ("50", 50), ("100", 100), ("250", 100), ("-10", 0)],
)
def test_env_rollout_percent_is_clamped_to_0_100(monkeypatch, raw, expected):
    monkeypatch.setenv("TP_TEST_ROLLOUT", raw)
    assert orchestrator_app._env_rollout_percent("TP_TEST_ROLLOUT", default=0) == expected


# ---------------------------------------------------------------------------
# _stable_rollout_bucket
# ---------------------------------------------------------------------------


def test_stable_rollout_bucket_is_in_range():
    for key in ("alice", "bob", "carol", "user-1234", "TEAM:ops"):
        bucket = orchestrator_app._stable_rollout_bucket(key)
        assert 0 <= bucket < 100, f"bucket out of range for {key}: {bucket}"


def test_stable_rollout_bucket_is_deterministic():
    a = orchestrator_app._stable_rollout_bucket("operator@example.com")
    b = orchestrator_app._stable_rollout_bucket("operator@example.com")
    assert a == b


def test_stable_rollout_bucket_is_case_insensitive():
    assert orchestrator_app._stable_rollout_bucket("Alice") == orchestrator_app._stable_rollout_bucket("ALICE")


def test_stable_rollout_bucket_empty_key_falls_outside_any_rollout():
    # Empty key returns 100 so no rollout (which checks `bucket < percent`) ever
    # includes a missing cohort key; this is a security-relevant invariant.
    assert orchestrator_app._stable_rollout_bucket("") == 100
    assert orchestrator_app._stable_rollout_bucket("   ") == 100


# ---------------------------------------------------------------------------
# _portal_rollout_cohort_key
# ---------------------------------------------------------------------------


def test_portal_rollout_cohort_key_prefers_username():
    actor = {"username": "Alice", "accessEmail": "alice@example.com", "role": "viewer"}
    assert orchestrator_app._portal_rollout_cohort_key(actor) == "alice"


def test_portal_rollout_cohort_key_falls_back_to_access_email():
    actor = {"accessEmail": "Bob@Example.com", "role": "operator"}
    assert orchestrator_app._portal_rollout_cohort_key(actor) == "bob@example.com"


def test_portal_rollout_cohort_key_falls_back_to_role():
    actor = {"role": "OpeRator"}
    assert orchestrator_app._portal_rollout_cohort_key(actor) == "operator"


def test_portal_rollout_cohort_key_uses_env_default(monkeypatch):
    monkeypatch.setenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", "Forced-Key")
    assert orchestrator_app._portal_rollout_cohort_key(actor=None) == "forced-key"


def test_portal_rollout_cohort_key_handles_non_mapping_actor(monkeypatch):
    monkeypatch.delenv("TP_PORTAL_DIRECT_DEBUG_COHORT_KEY", raising=False)
    # Not a Mapping -> falls through to env default ("direct-debug")
    assert orchestrator_app._portal_rollout_cohort_key(actor="not-a-mapping") == "direct-debug"


# ---------------------------------------------------------------------------
# _portal_rollout_enabled and feature-flag wrappers
# ---------------------------------------------------------------------------


def test_portal_rollout_disabled_when_percent_zero(monkeypatch):
    monkeypatch.setenv("TP_PORTAL_TEST_ROLLOUT", "0")
    assert orchestrator_app._portal_rollout_enabled("TP_PORTAL_TEST_ROLLOUT", actor={"username": "alice"}) is False


def test_portal_rollout_enabled_at_100_percent_includes_everyone(monkeypatch):
    monkeypatch.setenv("TP_PORTAL_TEST_ROLLOUT", "100")
    for username in ("alice", "bob", "carol", "x"):
        assert orchestrator_app._portal_rollout_enabled("TP_PORTAL_TEST_ROLLOUT", actor={"username": username}) is True


def test_portal_rollout_disabled_for_unset_percent(monkeypatch):
    monkeypatch.delenv("TP_PORTAL_TEST_ROLLOUT", raising=False)
    assert orchestrator_app._portal_rollout_enabled("TP_PORTAL_TEST_ROLLOUT", actor={"username": "alice"}) is False


def test_portal_rollout_partial_includes_only_low_buckets(monkeypatch):
    monkeypatch.setenv("TP_PORTAL_TEST_ROLLOUT", "50")
    # With a 50% rollout, exactly the cohorts whose stable bucket < 50 are in.
    decisions = {
        username: orchestrator_app._portal_rollout_enabled("TP_PORTAL_TEST_ROLLOUT", actor={"username": username})
        for username in ("alice", "bob", "carol", "dave", "erin")
    }
    # Property: each decision matches the bucket < percent rule.
    for username, decision in decisions.items():
        bucket = orchestrator_app._stable_rollout_bucket(username)
        assert decision is (bucket < 50), f"{username} bucket={bucket} decision={decision}"


def test_portal_rum_requires_both_master_switch_and_rollout(monkeypatch):
    actor = {"username": "alice"}

    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "false")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")
    assert orchestrator_app._portal_rum_enabled(actor) is False

    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "true")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "0")
    assert orchestrator_app._portal_rum_enabled(actor) is False

    monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "true")
    monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")
    assert orchestrator_app._portal_rum_enabled(actor) is True


def test_portal_staged_uploads_requires_both_master_switch_and_rollout(monkeypatch):
    actor = {"username": "alice"}

    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "false")
    monkeypatch.setenv("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", "100")
    assert orchestrator_app._portal_staged_uploads_enabled(actor) is False

    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "true")
    monkeypatch.setenv("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", "0")
    assert orchestrator_app._portal_staged_uploads_enabled(actor) is False

    monkeypatch.setenv("TP_PORTAL_UPLOAD_STAGING_ENABLED", "true")
    monkeypatch.setenv("TP_PORTAL_STAGED_UPLOADS_ROLLOUT_PERCENT", "100")
    assert orchestrator_app._portal_staged_uploads_enabled(actor) is True


def test_portal_artifact_viewer_modal_flag_passes_through_rollout(monkeypatch):
    monkeypatch.setenv("TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT", "100")
    assert orchestrator_app._portal_artifact_viewer_modal_enabled({"username": "alice"}) is True

    monkeypatch.setenv("TP_PORTAL_ARTIFACT_VIEWER_MODAL_ROLLOUT_PERCENT", "0")
    assert orchestrator_app._portal_artifact_viewer_modal_enabled({"username": "alice"}) is False


def test_portal_review_surface_deferred_flag_passes_through_rollout(monkeypatch):
    monkeypatch.setenv("TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT", "100")
    assert orchestrator_app._portal_review_surface_deferred_enabled({"username": "alice"}) is True

    monkeypatch.setenv("TP_PORTAL_REVIEW_SURFACE_DEFER_ROLLOUT_PERCENT", "0")
    assert orchestrator_app._portal_review_surface_deferred_enabled({"username": "alice"}) is False
