"""Deterministic artifact manifest for ADR-030 CAS artifacts.

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

from __future__ import annotations

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
    """Build a deterministic artifact manifest.

    This manifest is content-derived and contains no timestamps or host identifiers.
    All fields are Python primitives for JCS/JSON serialization safety.

    Args:
        artifact_id: Content-addressed artifact identifier (sha256:...).
        tensor_role: Certified tensor role (e.g., "xyz_d50_linear_fp32").
        tensor_hash: SHA-256 hash of the tensor payload.
        raw_hash: SHA-256 hash of the raw input file.
        fingerprint_hash: SHA-256 hash of the ingest fingerprint.
        fpstate_enforced: Whether FTZ/DAZ enforcement succeeded.
        fpstate_backend: Enforcement backend used.
        probe_version: Version of the FP-state probe algorithm.
        probe_policy: Policy used for normalizing probe results.
        subnormals_preserved: Whether subnormals are preserved after probe.
        fpstate_note: Optional diagnostic note (no timestamps/host IDs).

    Returns:
        Deterministic manifest dictionary (use stable_manifest_json to serialize).
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
        # Keep deterministic + short; no host data.
        m["fpstate"]["note"] = str(fpstate_note)
    return m


def stable_manifest_json(manifest: Dict[str, Any]) -> str:
    """Serialize manifest to deterministic JSON string.

    Uses RFC 8785 (JCS) canonical serialization for deterministic ordering.
    """
    return dumps(manifest)


def is_manifest_v3_compatible(manifest: Dict[str, Any]) -> bool:
    """Check if manifest is compatible with schema v3.

    This validates the presence of probe_version and probe_policy fields.
    """
    if manifest.get("schema_version") != 3:
        return False
    fpstate = manifest.get("fpstate", {})
    return "probe_version" in fpstate and "probe_policy" in fpstate


def migrate_manifest_v2_to_v3(manifest: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate a v2 manifest to v3 schema.

    For backward compatibility, assumes probe_version=0 and probe_policy="legacy"
    for manifests without these fields.

    Args:
        manifest: Input manifest (v2 or v3).

    Returns:
        Manifest with schema_version=3 and probe fields populated.
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
