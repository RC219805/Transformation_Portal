"""Unit tests for deterministic artifact manifest (schema v3).

These tests validate:
- Manifest builder functionality
- Schema version management
- JSON serialization determinism
- Migration from v2 to v3
"""

from __future__ import annotations

import json

import pytest

from transformation_portal.determinism.manifest import (
    MANIFEST_SCHEMA_VERSION,
    build_artifact_manifest,
    is_manifest_v3_compatible,
    migrate_manifest_v2_to_v3,
    stable_manifest_json,
)


pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


def test_current_schema_version_is_3():
    """Current manifest schema version is 3."""
    assert MANIFEST_SCHEMA_VERSION == 3


# ---------------------------------------------------------------------------
# build_artifact_manifest
# ---------------------------------------------------------------------------


def test_build_artifact_manifest_basic():
    """build_artifact_manifest creates a valid v3 manifest."""
    manifest = build_artifact_manifest(
        artifact_id="sha256:abc123",
        tensor_role="xyz_d50_linear_fp32",
        tensor_hash="abc123",
        raw_hash="def456",
        fingerprint_hash="ghi789",
        fpstate_enforced=True,
        fpstate_backend="fpstate.enforce_ftz_daz_disabled",
        probe_version=1,
        probe_policy="strict",
        subnormals_preserved=True,
    )

    assert manifest["schema_version"] == 3
    assert manifest["artifact_id"] == "sha256:abc123"
    assert manifest["tensor_role"] == "xyz_d50_linear_fp32"
    assert manifest["tensor_hash"] == "abc123"
    assert manifest["raw_input_hash"] == "def456"
    assert manifest["fingerprint_hash"] == "ghi789"

    fpstate = manifest["fpstate"]
    assert fpstate["enforced"] is True
    assert fpstate["backend"] == "fpstate.enforce_ftz_daz_disabled"
    assert fpstate["probe_version"] == 1
    assert fpstate["probe_policy"] == "strict"
    assert fpstate["subnormals_preserved"] is True
    assert "note" not in fpstate


def test_build_artifact_manifest_with_note():
    """build_artifact_manifest includes note when provided."""
    manifest = build_artifact_manifest(
        artifact_id="sha256:abc123",
        tensor_role="xyz_d50_linear_fp32",
        tensor_hash="abc123",
        raw_hash="def456",
        fingerprint_hash="ghi789",
        fpstate_enforced=False,
        fpstate_backend="probe_only",
        probe_version=1,
        probe_policy="strict",
        subnormals_preserved=False,
        fpstate_note="strict_requires_scalar_and_vector",
    )

    assert manifest["fpstate"]["note"] == "strict_requires_scalar_and_vector"


def test_build_artifact_manifest_types_are_python_primitives():
    """build_artifact_manifest fields are Python primitives for JCS/JSON."""
    manifest = build_artifact_manifest(
        artifact_id="sha256:abc123",
        tensor_role="xyz_d50_linear_fp32",
        tensor_hash="abc123",
        raw_hash="def456",
        fingerprint_hash="ghi789",
        fpstate_enforced=True,
        fpstate_backend="test",
        probe_version=1,
        probe_policy="strict",
        subnormals_preserved=True,
    )

    assert isinstance(manifest["schema_version"], int)
    assert isinstance(manifest["fpstate"]["enforced"], bool)
    assert isinstance(manifest["fpstate"]["probe_version"], int)
    assert isinstance(manifest["fpstate"]["probe_policy"], str)
    assert isinstance(manifest["fpstate"]["subnormals_preserved"], bool)


# ---------------------------------------------------------------------------
# stable_manifest_json
# ---------------------------------------------------------------------------


def test_stable_manifest_json_sorted_keys():
    """stable_manifest_json produces sorted keys."""
    manifest = {"z": 1, "a": 2, "m": 3}
    result = stable_manifest_json(manifest)
    parsed = json.loads(result)

    # Keys should be in sorted order when iterating.
    keys = list(parsed.keys())
    assert keys == sorted(keys)


def test_stable_manifest_json_compact():
    """stable_manifest_json produces compact JSON (no extra whitespace)."""
    manifest = {"key": "value", "nested": {"inner": True}}
    result = stable_manifest_json(manifest)

    assert " " not in result  # No spaces after separators.
    assert "\n" not in result  # No newlines.


def test_stable_manifest_json_deterministic():
    """stable_manifest_json is deterministic for same input."""
    manifest = build_artifact_manifest(
        artifact_id="sha256:test",
        tensor_role="xyz_d50_linear_fp32",
        tensor_hash="test",
        raw_hash="test",
        fingerprint_hash="test",
        fpstate_enforced=True,
        fpstate_backend="test",
        probe_version=1,
        probe_policy="strict",
        subnormals_preserved=True,
    )

    json1 = stable_manifest_json(manifest)
    json2 = stable_manifest_json(manifest)

    assert json1 == json2


# ---------------------------------------------------------------------------
# is_manifest_v3_compatible
# ---------------------------------------------------------------------------


def test_is_manifest_v3_compatible_true():
    """is_manifest_v3_compatible returns True for valid v3 manifest."""
    manifest = {
        "schema_version": 3,
        "fpstate": {
            "probe_version": 1,
            "probe_policy": "strict",
        },
    }
    assert is_manifest_v3_compatible(manifest) is True


def test_is_manifest_v3_compatible_false_for_v2():
    """is_manifest_v3_compatible returns False for v2 manifest."""
    manifest = {
        "schema_version": 2,
        "fpstate": {
            "enforced": True,
            "subnormals_preserved": True,
        },
    }
    assert is_manifest_v3_compatible(manifest) is False


def test_is_manifest_v3_compatible_false_missing_probe_version():
    """is_manifest_v3_compatible returns False when probe_version is missing."""
    manifest = {
        "schema_version": 3,
        "fpstate": {
            "probe_policy": "strict",
        },
    }
    assert is_manifest_v3_compatible(manifest) is False


def test_is_manifest_v3_compatible_false_missing_probe_policy():
    """is_manifest_v3_compatible returns False when probe_policy is missing."""
    manifest = {
        "schema_version": 3,
        "fpstate": {
            "probe_version": 1,
        },
    }
    assert is_manifest_v3_compatible(manifest) is False


# ---------------------------------------------------------------------------
# migrate_manifest_v2_to_v3
# ---------------------------------------------------------------------------


def test_migrate_manifest_v2_to_v3_basic():
    """migrate_manifest_v2_to_v3 upgrades schema version and adds probe fields."""
    v2_manifest = {
        "schema_version": 2,
        "artifact_id": "sha256:test",
        "fpstate": {
            "enforced": True,
            "backend": "test",
            "subnormals_preserved": True,
        },
    }

    v3_manifest = migrate_manifest_v2_to_v3(v2_manifest)

    assert v3_manifest["schema_version"] == 3
    assert v3_manifest["fpstate"]["probe_version"] == 0
    assert v3_manifest["fpstate"]["probe_policy"] == "legacy"
    # Original fields preserved.
    assert v3_manifest["fpstate"]["enforced"] is True
    assert v3_manifest["fpstate"]["backend"] == "test"


def test_migrate_manifest_v2_to_v3_idempotent():
    """migrate_manifest_v2_to_v3 is idempotent for v3 manifests."""
    v3_manifest = {
        "schema_version": 3,
        "fpstate": {
            "probe_version": 1,
            "probe_policy": "strict",
        },
    }

    result = migrate_manifest_v2_to_v3(v3_manifest)

    assert result["schema_version"] == 3
    assert result["fpstate"]["probe_version"] == 1
    assert result["fpstate"]["probe_policy"] == "strict"


def test_migrate_manifest_v2_to_v3_does_not_modify_original():
    """migrate_manifest_v2_to_v3 returns a copy, not modifying original."""
    v2_manifest = {
        "schema_version": 2,
        "fpstate": {
            "enforced": True,
        },
    }

    migrate_manifest_v2_to_v3(v2_manifest)

    # Original should still be v2.
    assert v2_manifest["schema_version"] == 2
    assert "probe_version" not in v2_manifest["fpstate"]


# ---------------------------------------------------------------------------
# Full integration
# ---------------------------------------------------------------------------


def test_full_manifest_roundtrip():
    """Build, serialize, and parse manifest roundtrip."""
    manifest = build_artifact_manifest(
        artifact_id="sha256:abc123def456",
        tensor_role="xyz_d50_linear_fp32",
        tensor_hash="abc123def456",
        raw_hash="raw123",
        fingerprint_hash="fp456",
        fpstate_enforced=True,
        fpstate_backend="fpstate.enforce_ftz_daz_disabled",
        probe_version=1,
        probe_policy="strict",
        subnormals_preserved=True,
        fpstate_note="test_note",
    )

    json_str = stable_manifest_json(manifest)
    parsed = json.loads(json_str)

    assert parsed["schema_version"] == 3
    assert parsed["fpstate"]["probe_version"] == 1
    assert parsed["fpstate"]["probe_policy"] == "strict"
    assert parsed["fpstate"]["note"] == "test_note"
    assert is_manifest_v3_compatible(parsed) is True
