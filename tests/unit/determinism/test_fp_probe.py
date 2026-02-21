"""Unit tests for cross-ISA FP-state probe normalization layer.

These tests validate:
- Normalization logic (policy semantics)
- Reason propagation
- Probe version invariants
- Boolean normalization behavior

These tests are deterministic and do not depend on real hardware FP state.
They use simulated raw probe inputs to validate policy logic.
"""

from __future__ import annotations

import pytest

from transformation_portal.determinism.fp_probe import (
    FPProbeNormalized,
    FPProbeRaw,
    normalize_fp_probe,
    probe_fpstate_normalized,
    probe_fpstate_raw,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_probe_version_locked():
    from transformation_portal.determinism.fp_probe import probe_fpstate_raw

    raw = probe_fpstate_raw()
    assert raw.probe_version == 1


def make_raw(
    *,
    smallest: bool = True,
    scalar: bool = True,
    vector: bool = True,
    version: int = 1,
    note: str | None = None,
) -> FPProbeRaw:
    """Create a FPProbeRaw for testing with simplified parameters."""
    return FPProbeRaw(
        probe_version=version,
        smallest_subnormal_nonzero=smallest,
        scalar_mul_preserved=scalar,
        scalar_add_preserved=scalar,
        vector_mul_preserved=vector,
        vector_add_preserved=vector,
        note=note,
    )


# ---------------------------------------------------------------------------
# Policy Semantics - Strict
# ---------------------------------------------------------------------------


def test_strict_requires_scalar_and_vector():
    """Strict policy passes when both scalar and vector tests pass."""
    raw = make_raw(scalar=True, vector=True)
    out = normalize_fp_probe(raw, policy="strict")
    assert out.subnormals_preserved is True
    assert out.reason is None
    assert out.policy == "strict"


def test_strict_fails_if_vector_fails():
    """Strict policy fails when vector tests fail."""
    raw = make_raw(scalar=True, vector=False)
    out = normalize_fp_probe(raw, policy="strict")
    assert out.subnormals_preserved is False
    assert out.reason == "strict_requires_scalar_and_vector"


def test_strict_fails_if_scalar_fails():
    """Strict policy fails when scalar tests fail."""
    raw = make_raw(scalar=False, vector=True)
    out = normalize_fp_probe(raw, policy="strict")
    assert out.subnormals_preserved is False
    assert out.reason == "strict_requires_scalar_and_vector"


# ---------------------------------------------------------------------------
# Policy Semantics - Relaxed
# ---------------------------------------------------------------------------


def test_relaxed_allows_scalar_only():
    """Relaxed policy passes when scalar tests pass, even if vector fails."""
    raw = make_raw(scalar=True, vector=False)
    out = normalize_fp_probe(raw, policy="relaxed")
    assert out.subnormals_preserved is True
    assert out.reason is None


def test_relaxed_allows_vector_only():
    """Relaxed policy passes when vector tests pass, even if scalar fails."""
    raw = make_raw(scalar=False, vector=True)
    out = normalize_fp_probe(raw, policy="relaxed")
    assert out.subnormals_preserved is True
    assert out.reason is None


def test_relaxed_fails_if_both_fail():
    """Relaxed policy fails when both scalar and vector tests fail."""
    raw = make_raw(scalar=False, vector=False)
    out = normalize_fp_probe(raw, policy="relaxed")
    assert out.subnormals_preserved is False
    assert out.reason == "relaxed_requires_scalar_or_vector"


# ---------------------------------------------------------------------------
# Policy Semantics - Scalar Only
# ---------------------------------------------------------------------------


def test_scalar_only_policy_passes_with_scalar():
    """Scalar-only policy passes when scalar tests pass."""
    raw = make_raw(scalar=True, vector=False)
    out = normalize_fp_probe(raw, policy="scalar_only")
    assert out.subnormals_preserved is True
    assert out.reason is None


def test_scalar_only_policy_fails_without_scalar():
    """Scalar-only policy fails when scalar tests fail, regardless of vector."""
    raw = make_raw(scalar=False, vector=True)
    out = normalize_fp_probe(raw, policy="scalar_only")
    assert out.subnormals_preserved is False
    assert out.reason == "scalar_only_failed"


# ---------------------------------------------------------------------------
# Policy Semantics - Vector Only
# ---------------------------------------------------------------------------


def test_vector_only_policy_passes_with_vector():
    """Vector-only policy passes when vector tests pass."""
    raw = make_raw(scalar=False, vector=True)
    out = normalize_fp_probe(raw, policy="vector_only")
    assert out.subnormals_preserved is True
    assert out.reason is None


def test_vector_only_policy_fails_without_vector():
    """Vector-only policy fails when vector tests fail, regardless of scalar."""
    raw = make_raw(scalar=True, vector=False)
    out = normalize_fp_probe(raw, policy="vector_only")
    assert out.subnormals_preserved is False
    assert out.reason == "vector_only_failed"


# ---------------------------------------------------------------------------
# Smallest subnormal invariant
# ---------------------------------------------------------------------------


def test_smallest_subnormal_zero_forces_failure():
    """If nextafter returns zero, all policies fail."""
    raw = make_raw(smallest=False, scalar=True, vector=True, note="nextafter_returned_zero")
    out = normalize_fp_probe(raw, policy="strict")
    assert out.subnormals_preserved is False
    assert out.reason == "nextafter_returned_zero"


def test_smallest_subnormal_zero_forces_failure_all_policies():
    """Smallest subnormal nonzero=False causes failure under all policies."""
    raw = make_raw(smallest=False, scalar=True, vector=True, note="test_note")

    for policy in ["strict", "relaxed", "scalar_only", "vector_only"]:
        out = normalize_fp_probe(raw, policy=policy)  # type: ignore
        assert out.subnormals_preserved is False, f"Failed for policy={policy}"


# ---------------------------------------------------------------------------
# Probe version propagation
# ---------------------------------------------------------------------------


def test_probe_version_propagates():
    """Probe version is preserved through normalization."""
    raw = make_raw(version=7)
    out = normalize_fp_probe(raw, policy="strict")
    assert out.probe_version == 7


def test_probe_version_one_in_raw_probe():
    """Current raw probe implementation returns version 1."""
    raw = probe_fpstate_raw()
    assert raw.probe_version == 1


# ---------------------------------------------------------------------------
# Probe version locking (governance enforcement)
# ---------------------------------------------------------------------------


def test_probe_version_locked():
    """Enforce conscious probe version increments (governance lock).

    This test MUST be updated when probe_version is intentionally bumped.
    It prevents accidental algorithm changes without version coordination.

    See fp_probe.py docstring for version bump criteria.

    When bumping probe_version:
    1. Review the PROBE VERSION GOVERNANCE section in fp_probe.py
    2. Verify the change meets increment criteria
    3. Update this test's expected version
    4. Update any documentation referencing the probe version
    """
    raw = probe_fpstate_raw()
    assert raw.probe_version == 1, (
        "probe_version changed unexpectedly. If this is intentional, "
        "update this test and verify version bump criteria in fp_probe.py docstring."
    )


# ---------------------------------------------------------------------------
# Deterministic output types
# ---------------------------------------------------------------------------


def test_output_types_are_python_primitives():
    """Normalized output types must be Python primitives for JCS/JSON."""
    raw = make_raw()
    out = normalize_fp_probe(raw, policy="strict")

    assert isinstance(out.subnormals_preserved, bool)
    assert isinstance(out.probe_version, int)
    assert isinstance(out.policy, str)


def test_raw_probe_returns_python_bool_types():
    """Raw probe boolean fields must be Python bool, not numpy.bool_."""
    raw = probe_fpstate_raw()

    assert isinstance(raw.smallest_subnormal_nonzero, bool)
    assert isinstance(raw.scalar_mul_preserved, bool)
    assert isinstance(raw.scalar_add_preserved, bool)
    assert isinstance(raw.vector_mul_preserved, bool)
    assert isinstance(raw.vector_add_preserved, bool)


# ---------------------------------------------------------------------------
# Invalid policy handling
# ---------------------------------------------------------------------------


def test_unknown_policy_raises_value_error():
    """Unknown policy string raises ValueError."""
    raw = make_raw()
    with pytest.raises(ValueError, match="Unknown policy"):
        normalize_fp_probe(raw, policy="invalid_policy")  # type: ignore


# ---------------------------------------------------------------------------
# Integration: probe_fpstate_normalized
# ---------------------------------------------------------------------------


def test_probe_fpstate_normalized_default_policy():
    """probe_fpstate_normalized defaults to strict policy."""
    result = probe_fpstate_normalized()
    assert result.policy == "strict"
    assert result.probe_version == 1


def test_probe_fpstate_normalized_with_policy():
    """probe_fpstate_normalized respects policy parameter."""
    result = probe_fpstate_normalized(policy="scalar_only")
    assert result.policy == "scalar_only"


# ---------------------------------------------------------------------------
# Dataclass frozen semantics
# ---------------------------------------------------------------------------


def test_fp_probe_raw_is_immutable():
    """FPProbeRaw is a frozen dataclass."""
    raw = make_raw()
    with pytest.raises(AttributeError):
        raw.probe_version = 2  # type: ignore


def test_fp_probe_normalized_is_immutable():
    """FPProbeNormalized is a frozen dataclass."""
    out = normalize_fp_probe(make_raw(), policy="strict")
    with pytest.raises(AttributeError):
        out.subnormals_preserved = False  # type: ignore
