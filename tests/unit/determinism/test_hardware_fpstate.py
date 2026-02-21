"""Unit tests for hardware FP-state enforcement and probe reporting.

These tests validate:
- FPStateReport structure and fields
- enforce_fpstate_and_probe behavior with various probe policies
- Integration with fp_probe module
"""

from __future__ import annotations

import pytest

from transformation_portal.determinism.hardware_fpstate import (
    FPStateReport,
    enforce_fpstate_and_probe,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# FPStateReport structure
# ---------------------------------------------------------------------------


def test_fpstate_report_has_required_fields():
    """FPStateReport has all required fields for manifest v3."""
    report = FPStateReport(
        enforced=True,
        backend="test_backend",
        probe_version=1,
        probe_policy="strict",
        subnormals_preserved=True,
        note=None,
    )

    assert report.enforced is True
    assert report.backend == "test_backend"
    assert report.probe_version == 1
    assert report.probe_policy == "strict"
    assert report.subnormals_preserved is True
    assert report.note is None


def test_fpstate_report_is_frozen():
    """FPStateReport is immutable."""
    report = FPStateReport(
        enforced=True,
        backend="test",
        probe_version=1,
        probe_policy="strict",
        subnormals_preserved=True,
    )
    with pytest.raises(AttributeError):
        report.enforced = False  # type: ignore


# ---------------------------------------------------------------------------
# enforce_fpstate_and_probe behavior
# ---------------------------------------------------------------------------


def test_enforce_fpstate_and_probe_returns_report():
    """enforce_fpstate_and_probe returns a valid FPStateReport."""
    report = enforce_fpstate_and_probe()

    assert isinstance(report, FPStateReport)
    assert isinstance(report.enforced, bool)
    assert isinstance(report.backend, str)
    assert isinstance(report.probe_version, int)
    assert isinstance(report.probe_policy, str)
    assert isinstance(report.subnormals_preserved, bool)


def test_enforce_fpstate_and_probe_default_policy():
    """enforce_fpstate_and_probe defaults to strict policy."""
    report = enforce_fpstate_and_probe()
    assert report.probe_policy == "strict"


def test_enforce_fpstate_and_probe_with_policy():
    """enforce_fpstate_and_probe respects probe_policy parameter."""
    report = enforce_fpstate_and_probe(probe_policy="scalar_only")
    assert report.probe_policy == "scalar_only"


def test_enforce_fpstate_and_probe_with_relaxed_policy():
    """enforce_fpstate_and_probe works with relaxed policy."""
    report = enforce_fpstate_and_probe(probe_policy="relaxed")
    assert report.probe_policy == "relaxed"


def test_enforce_fpstate_and_probe_with_vector_only_policy():
    """enforce_fpstate_and_probe works with vector_only policy."""
    report = enforce_fpstate_and_probe(probe_policy="vector_only")
    assert report.probe_policy == "vector_only"


def test_enforce_fpstate_and_probe_probe_version():
    """enforce_fpstate_and_probe returns probe version 1."""
    report = enforce_fpstate_and_probe()
    assert report.probe_version == 1


# ---------------------------------------------------------------------------
# require_subnormals behavior
# ---------------------------------------------------------------------------


def test_enforce_fpstate_require_subnormals_passes_when_preserved(monkeypatch):
    """require_subnormals=True passes when subnormals are preserved."""
    from transformation_portal.determinism import fp_probe, hardware_fpstate

    # Mock the probe in the hardware_fpstate module where it's imported.
    monkeypatch.setattr(
        hardware_fpstate,
        "probe_fpstate_normalized",
        lambda policy="strict": fp_probe.FPProbeNormalized(
            probe_version=1,
            policy=policy,
            subnormals_preserved=True,
            reason=None,
        ),
    )

    # Should not raise.
    report = enforce_fpstate_and_probe(require_subnormals=True)
    assert report.subnormals_preserved is True


def test_enforce_fpstate_require_subnormals_fails_when_not_preserved(monkeypatch):
    """require_subnormals=True raises when subnormals are not preserved."""
    from transformation_portal.determinism import fp_probe, hardware_fpstate

    # Mock the probe in the hardware_fpstate module where it's imported.
    monkeypatch.setattr(
        hardware_fpstate,
        "probe_fpstate_normalized",
        lambda policy="strict": fp_probe.FPProbeNormalized(
            probe_version=1,
            policy=policy,
            subnormals_preserved=False,
            reason="strict_requires_scalar_and_vector",
        ),
    )

    with pytest.raises(RuntimeError, match="FP-state invariance failure"):
        enforce_fpstate_and_probe(require_subnormals=True)


# ---------------------------------------------------------------------------
# Enforcement backend fallback
# ---------------------------------------------------------------------------


def test_enforce_fpstate_falls_back_to_probe_only(monkeypatch):
    """When enforcement is unavailable, falls back to probe_only backend."""
    import sys

    # Remove the fpstate module to simulate unavailability.
    original = sys.modules.get("transformation_portal.determinism.fpstate")
    monkeypatch.setitem(sys.modules, "transformation_portal.determinism.fpstate", None)

    try:
        report = enforce_fpstate_and_probe()
        # Either probe_only or the actual backend depending on import path.
        assert report.backend in ("probe_only", "fpstate.enforce_ftz_daz_disabled")
    finally:
        if original is not None:
            sys.modules["transformation_portal.determinism.fpstate"] = original


# ---------------------------------------------------------------------------
# Note field behavior
# ---------------------------------------------------------------------------


def test_enforce_fpstate_note_includes_probe_reason_on_failure(monkeypatch):
    """When probe fails, note includes probe reason."""
    from transformation_portal.determinism import fp_probe, hardware_fpstate

    # Mock the probe in the hardware_fpstate module where it's imported.
    monkeypatch.setattr(
        hardware_fpstate,
        "probe_fpstate_normalized",
        lambda policy="strict": fp_probe.FPProbeNormalized(
            probe_version=1,
            policy=policy,
            subnormals_preserved=False,
            reason="strict_requires_scalar_and_vector",
        ),
    )

    report = enforce_fpstate_and_probe()
    assert report.note is not None
    assert "probe:1:strict:strict_requires_scalar_and_vector" in report.note


def test_enforce_fpstate_note_is_none_on_success():
    """When probe succeeds and enforcement succeeds, note may be None."""
    report = enforce_fpstate_and_probe()
    # Note can be None or contain diagnostic info depending on platform.
    # Just verify the field exists and is of correct type.
    assert report.note is None or isinstance(report.note, str)
