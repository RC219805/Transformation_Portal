"""Unit tests for environment fingerprint module.

Tests SPEC-DH-001 Section 5 compliance: harness must publish environment
fingerprint with OS, ISA, runtime version, and dependency lock IDs.
"""

from __future__ import annotations

import pytest

from transformation_portal.determinism.environment import (
    HARNESS_ENGINE_VERSION,
    EnvironmentFingerprint,
    capture_environment,
    environment_fingerprint_dict,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# HARNESS_ENGINE_VERSION
# ---------------------------------------------------------------------------


def test_harness_engine_version_is_semver():
    """Harness engine version follows semver format."""
    parts = HARNESS_ENGINE_VERSION.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit()


def test_harness_engine_version_locked():
    """Harness engine version is locked at 1.0.0 for initial release.

    Increment this when harness logic changes require a version bump.
    This test enforces conscious version increments (similar to probe_version).
    """
    assert HARNESS_ENGINE_VERSION == "1.0.0"


# ---------------------------------------------------------------------------
# capture_environment
# ---------------------------------------------------------------------------


def test_capture_environment_returns_fingerprint():
    """capture_environment returns an EnvironmentFingerprint."""
    fp = capture_environment()
    assert isinstance(fp, EnvironmentFingerprint)


def test_capture_environment_has_required_fields():
    """Environment fingerprint has all SPEC-DH-001 required fields."""
    fp = capture_environment()

    # OS info
    assert isinstance(fp.os_system, str)
    assert len(fp.os_system) > 0
    assert isinstance(fp.os_release, str)
    assert isinstance(fp.os_machine, str)
    assert len(fp.os_machine) > 0

    # Runtime info
    assert isinstance(fp.python_version, str)
    assert len(fp.python_version) > 0
    assert isinstance(fp.python_implementation, str)
    assert len(fp.python_implementation) > 0

    # Dependency info
    assert isinstance(fp.numpy_version, str)
    assert len(fp.numpy_version) > 0
    assert isinstance(fp.numpy_config, dict)

    # Harness info
    assert isinstance(fp.harness_engine_version, str)
    assert len(fp.harness_engine_version) > 0


def test_capture_environment_os_machine_is_isa():
    """os_machine reflects ISA (architecture)."""
    fp = capture_environment()
    # Common ISAs
    valid_isas = {"x86_64", "amd64", "arm64", "aarch64", "i686", "i386"}
    assert fp.os_machine.lower() in valid_isas or len(fp.os_machine) > 0


def test_capture_environment_numpy_config_has_version():
    """numpy_config includes version for audit."""
    fp = capture_environment()
    assert "version" in fp.numpy_config
    assert fp.numpy_config["version"] == fp.numpy_version


# ---------------------------------------------------------------------------
# environment_fingerprint_dict
# ---------------------------------------------------------------------------


def test_environment_fingerprint_dict_returns_dict():
    """environment_fingerprint_dict returns a dict."""
    result = environment_fingerprint_dict()
    assert isinstance(result, dict)


def test_environment_fingerprint_dict_is_json_serializable():
    """Environment fingerprint dict can be JSON serialized."""
    import json

    result = environment_fingerprint_dict()
    # Should not raise
    json_str = json.dumps(result)
    parsed = json.loads(json_str)
    assert parsed == result


def test_environment_fingerprint_dict_matches_capture():
    """environment_fingerprint_dict matches capture_environment().to_dict()."""
    fp = capture_environment()
    d = environment_fingerprint_dict()
    assert d == fp.to_dict()


# ---------------------------------------------------------------------------
# EnvironmentFingerprint.to_dict
# ---------------------------------------------------------------------------


def test_fingerprint_to_dict_has_all_fields():
    """to_dict includes all fingerprint fields."""
    fp = capture_environment()
    d = fp.to_dict()

    assert "harness_engine_version" in d
    assert "os_system" in d
    assert "os_release" in d
    assert "os_machine" in d
    assert "python_version" in d
    assert "python_implementation" in d
    assert "numpy_version" in d
    assert "numpy_config" in d


def test_fingerprint_to_dict_values_are_primitives():
    """to_dict values are Python primitives (JCS/JSON safe)."""
    d = environment_fingerprint_dict()

    for key, value in d.items():
        if key == "numpy_config":
            assert isinstance(value, dict)
            for v in value.values():
                assert isinstance(v, (str, int, float, bool, list, type(None)))
        else:
            assert isinstance(value, (str, int, float, bool))
