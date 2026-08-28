"""Validation helpers for detached attestations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from transformation_portal.ingest.canonical_json import TP_CANONICAL_JSON_PROFILE
from transformation_portal.ingest.evidence import EVIDENCE_SCHEMA_VERSION

from .detached import (
    ATTESTATION_SCHEMA_VERSION,
    _recompute_evidence_sha256_from_projected_envelope,
    _validate_sha256,
    compute_attestation_sha256,
)


def _optional_sha256(value: Any, *, field: str) -> str | None:
    if value is None:
        return None
    return _validate_sha256(value, field=field)


def _projected_file_sha256(projected_envelope: Mapping[str, Any]) -> str | None:
    for container, field in (
        (projected_envelope, "projected_envelope.file_integrity.sha256"),
        (projected_envelope.get("data"), "projected_envelope.data.file_integrity.sha256"),
    ):
        if not isinstance(container, Mapping):
            continue
        file_integrity = container.get("file_integrity")
        if isinstance(file_integrity, Mapping) and "sha256" in file_integrity:
            return _validate_sha256(file_integrity.get("sha256"), field=field)
    return None


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
    """Assert the attestation subject binds to the canonical projected evidence."""
    if evidence.get("schema") != EVIDENCE_SCHEMA_VERSION:
        found = evidence.get("schema")
        raise ValueError(f"evidence schema must be {EVIDENCE_SCHEMA_VERSION}, got {found!r}")

    validate_detached_attestation_surface(attestation)

    att_sha = _validate_sha256(
        attestation["subject"]["evidence_sha256"],
        field="subject.evidence_sha256",
    )
    ev_sha = _validate_sha256(
        evidence.get("evidence_sha256"),
        field="evidence.evidence_sha256",
    )

    projected_envelope = evidence.get("projected_envelope")
    if not isinstance(projected_envelope, Mapping):
        raise ValueError("evidence projected_envelope must be an object")
    recomputed_sha = _recompute_evidence_sha256_from_projected_envelope(evidence)
    if recomputed_sha != ev_sha:
        raise ValueError("evidence_sha256 mismatch: projected_envelope does not reproduce stored evidence_sha256")

    if att_sha != recomputed_sha:
        raise ValueError("attestation does not bind to this evidence payload: evidence_sha256 mismatch")

    subject = attestation["subject"]
    evidence_file_sha = _optional_sha256(
        evidence.get("file_sha256"),
        field="evidence.file_sha256",
    )
    projected_file_sha = _projected_file_sha256(projected_envelope)
    if evidence_file_sha != projected_file_sha:
        raise ValueError("evidence file_sha256 does not match projected_envelope file digest")

    for anchor in ("file_sha256", "bundle_root_sha256"):
        subject_sha = _optional_sha256(
            subject.get(anchor),
            field=f"subject.{anchor}",
        )
        evidence_sha = _optional_sha256(
            evidence.get(anchor),
            field=f"evidence.{anchor}",
        )
        if subject_sha != evidence_sha:
            raise ValueError(f"attestation does not bind to this evidence payload: {anchor} mismatch")


def verify_attestation_self_hash(attestation: Mapping[str, Any], *, require_digest: bool = True) -> None:
    """Verify `attestation_sha256` matches the canonicalized attestation payload."""
    stored_digest = attestation.get("attestation_sha256")
    if stored_digest is None:
        if require_digest:
            raise ValueError("attestation_sha256 must be set for integrity verification")
        return

    normalized_stored_digest = _validate_sha256(stored_digest, field="attestation_sha256")
    recomputed_digest = compute_attestation_sha256(attestation)
    if recomputed_digest != normalized_stored_digest:
        raise ValueError("attestation_sha256 mismatch: payload does not match canonical digest")
