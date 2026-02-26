"""Validation helpers for detached attestations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformation_portal.ingest.canonical_json import TP_CANONICAL_JSON_PROFILE
from transformation_portal.ingest.evidence import EVIDENCE_SCHEMA_VERSION

from .detached import ATTESTATION_SCHEMA_VERSION, _validate_sha256


def validate_detached_attestation_surface(attestation: Mapping[str, Any]) -> None:
    """Validate the attestation shape needed for evidence binding and signature checks."""
    if attestation.get("schema") != ATTESTATION_SCHEMA_VERSION:
        found = attestation.get("schema")
        raise ValueError(f"attestation schema must be {ATTESTATION_SCHEMA_VERSION}, got {found!r}")

    if attestation.get("canonicalization") != TP_CANONICAL_JSON_PROFILE:
        found = attestation.get("canonicalization")
        raise ValueError(f"attestation canonicalization must be {TP_CANONICAL_JSON_PROFILE}, got {found!r}")

    subject = attestation.get("subject")
    if not isinstance(subject, Mapping):
        raise ValueError("attestation subject must be an object")
    if subject.get("schema") != EVIDENCE_SCHEMA_VERSION:
        found = subject.get("schema")
        raise ValueError(f"attestation subject schema must be {EVIDENCE_SCHEMA_VERSION}, got {found!r}")
    _validate_sha256(subject.get("evidence_sha256"), field="subject.evidence_sha256")

    for field in ("file_sha256", "bundle_root_sha256"):
        optional_sha = subject.get(field)
        if optional_sha is not None:
            _validate_sha256(optional_sha, field=f"subject.{field}")

    signature = attestation.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("attestation signature must be an object")
    if not isinstance(signature.get("algorithm"), str) or not signature.get("algorithm"):
        raise ValueError("signature.algorithm must be a non-empty string")
    if not isinstance(signature.get("key_id"), str) or not signature.get("key_id"):
        raise ValueError("signature.key_id must be a non-empty string")
    if not isinstance(signature.get("signature"), str) or not signature.get("signature"):
        raise ValueError("signature.signature must be a non-empty string")

    attestation_sha256 = attestation.get("attestation_sha256")
    if attestation_sha256 is not None:
        _validate_sha256(attestation_sha256, field="attestation_sha256")


def bind_attestation_to_evidence(attestation: Mapping[str, Any], evidence: Mapping[str, Any]) -> None:
    """Assert attestation subject hash binds to the provided evidence payload."""
    if evidence.get("schema") != EVIDENCE_SCHEMA_VERSION:
        found = evidence.get("schema")
        raise ValueError(f"evidence schema must be {EVIDENCE_SCHEMA_VERSION}, got {found!r}")

    validate_detached_attestation_surface(attestation)

    att_sha = attestation["subject"]["evidence_sha256"]
    ev_sha = evidence.get("evidence_sha256")
    _validate_sha256(ev_sha, field="evidence.evidence_sha256")

    if att_sha != ev_sha:
        raise ValueError("attestation does not bind to this evidence payload: evidence_sha256 mismatch")
