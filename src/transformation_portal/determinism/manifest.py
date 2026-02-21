from __future__ import annotations

"""
Deterministic artifact manifest for ADR-030 CAS artifacts.

This module provides a versioned, deterministic manifest builder for CAS artifacts.
Schema v3 promotes probe_version and probe_policy to first-class fields for
cross-ISA auditability.

Schema version history:
- v1: Initial schema (deprecated)
- v2: Added fpstate with enforced/backend/subnormals_preserved
- v3: Added probe_version and probe_policy to fpstate section

Design constraints:
- No timestamps
- No environment / host identifiers
- Stable key ordering (use stable_manifest_json for serialization)
- Only content-derived + deterministic probe/enforcement outcomes
"""

from typing import Any, Dict, Optional

from .jcs import dumps

# Current schema version for new manifests.
MANIFEST_SCHEMA_VERSION = 3


def build_artifact_manifest(
    *,
    artifact_id: str,
    tensor_role: str,
    tensor_hash: str,
    raw_hash: str,
    fingerprint_hash: str,
    fpstate_enforced: bool,
    fpstate_backend: str,
    probe_version: int,
    probe_policy: str,
    subnormals_preserved: bool,
    fpstate_note: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a deterministic artifact manifest (schema v3).

    This manifest is content-derived and contains no timestamps or host identifiers.
    All fields are Python primitives for JCS/JSON serialization safety.
    """

    m: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_id": str(artifact_id),
        "tensor_role": str(tensor_role),
        "tensor_hash": str(tensor_hash),
        "raw_input_hash": str(raw_hash),
        "fingerprint_hash": str(fingerprint_hash),
        "fpstate": {
            "enforced": bool(fpstate_enforced),
            "backend": str(fpstate_backend),
            "probe_version": int(probe_version),
            "probe_policy": str(probe_policy),
            "subnormals_preserved": bool(subnormals_preserved),
        },
    }

    if fpstate_note:
        # Deterministic + short; no host/environment data.
        m["fpstate"]["note"] = str(fpstate_note)

    return m


def stable_manifest_json(manifest: Dict[str, Any]) -> str:
    """Serialize manifest to deterministic JSON string.

    Uses RFC 8785 (JCS) canonical serialization for deterministic ordering.
    """
    return dumps(manifest)


def is_manifest_v3_compatible(manifest: Dict[str, Any]) -> bool:
    """Check if manifest is compatible with schema v3."""
    if manifest.get("schema_version") != 3:
        return False
    fpstate = manifest.get("fpstate", {})
    return "probe_version" in fpstate and "probe_policy" in fpstate


def migrate_manifest_v2_to_v3(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a v2 manifest to v3 schema.

    For backward compatibility, assumes probe_version=0 and probe_policy="legacy"
    for manifests without these fields.
    """

    if manifest.get("schema_version") == 3:
        return manifest

    migrated = manifest.copy()
    migrated["schema_version"] = 3

    if "fpstate" in migrated:
        fpstate = migrated["fpstate"].copy()
        fpstate.setdefault("probe_version", 0)
        fpstate.setdefault("probe_policy", "legacy")
        migrated["fpstate"] = fpstate

    return migrated