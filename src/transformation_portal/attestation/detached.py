"""Detached attestation payload builders for ``tp.meta.evidence.v1`` bindings."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from transformation_portal.ingest.canonical_json import TP_CANONICAL_JSON_PROFILE, canonicalize_json
from transformation_portal.ingest.evidence import EVIDENCE_SCHEMA_VERSION

ATTESTATION_SCHEMA_VERSION = "tp.attestation.detached.v1"
PREIMAGE_SCHEMA_VERSION = "tp.attestation.detached.v1.preimage"
_HEX_CHARS = frozenset("0123456789abcdef")


def _validate_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    if len(value) != 64:
        raise ValueError(f"{field} must be a 64-character sha256 digest")
    lowered = value.lower()
    if any(char not in _HEX_CHARS for char in lowered):
        raise ValueError(f"{field} must be hex (0-9a-f)")
    return lowered


def _validate_evidence_payload_surface(evidence_payload: Mapping[str, Any]) -> None:
    if evidence_payload.get("schema") != EVIDENCE_SCHEMA_VERSION:
        found = evidence_payload.get("schema")
        raise ValueError(f"evidence payload schema must be {EVIDENCE_SCHEMA_VERSION}, got {found!r}")

    evidence_sha256 = evidence_payload.get("evidence_sha256")
    _validate_sha256(evidence_sha256, field="evidence_sha256")

    projected = evidence_payload.get("projected_envelope")
    if not isinstance(projected, Mapping):
        raise ValueError("evidence payload projected_envelope must be an object")


def _recompute_evidence_sha256_from_projected_envelope(evidence_payload: Mapping[str, Any]) -> str:
    projected = evidence_payload["projected_envelope"]
    return hashlib.sha256(canonicalize_json(projected)).hexdigest()


def canonical_attestation_bytes(attestation_payload: Mapping[str, Any]) -> bytes:
    """Serialize detached attestation payload with canonical JSON profile."""
    return canonicalize_json(dict(attestation_payload))


def build_detached_attestation_preimage(evidence_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the signing preimage for detached evidence attestation."""
    _validate_evidence_payload_surface(evidence_payload)

    evidence_sha256 = _validate_sha256(evidence_payload.get("evidence_sha256"), field="evidence_sha256")
    file_sha256 = evidence_payload.get("file_sha256")
    if file_sha256 is not None:
        file_sha256 = _validate_sha256(file_sha256, field="file_sha256")
    bundle_root_sha256 = evidence_payload.get("bundle_root_sha256")
    if bundle_root_sha256 is not None:
        bundle_root_sha256 = _validate_sha256(bundle_root_sha256, field="bundle_root_sha256")

    return {
        "schema": PREIMAGE_SCHEMA_VERSION,
        "subject": {
            "schema": EVIDENCE_SCHEMA_VERSION,
            "evidence_sha256": evidence_sha256,
            "file_sha256": file_sha256,
            "bundle_root_sha256": bundle_root_sha256,
        },
    }


def canonical_attestation_preimage_bytes(evidence_payload: Mapping[str, Any]) -> bytes:
    """Serialize detached attestation preimage with canonical JSON profile."""
    return canonicalize_json(build_detached_attestation_preimage(evidence_payload))


def compute_attestation_sha256(attestation_payload: Mapping[str, Any]) -> str:
    """Compute sha256 of canonical attestation payload excluding ``attestation_sha256`` itself."""
    payload = dict(attestation_payload)
    payload["attestation_sha256"] = None
    return hashlib.sha256(canonicalize_json(payload)).hexdigest()


def _validate_signature(signature: Mapping[str, Any]) -> dict[str, str]:
    algorithm = signature.get("algorithm")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("signature.algorithm must be a non-empty string")

    key_id = signature.get("key_id")
    if not isinstance(key_id, str) or not key_id:
        raise ValueError("signature.key_id must be a non-empty string")

    signature_text = signature.get("signature")
    if not isinstance(signature_text, str) or not signature_text:
        raise ValueError("signature.signature must be a non-empty string")

    return {
        "algorithm": algorithm,
        "key_id": key_id,
        "signature": signature_text,
    }


def build_detached_attestation_payload(
    evidence_payload: Mapping[str, Any],
    *,
    signature: Mapping[str, Any],
    signed_at: str | None = None,
    toolchain: Mapping[str, Any] | None = None,
    claims: Mapping[str, Any] | None = None,
    enforce_recompute_match: bool = True,
) -> dict[str, Any]:
    """Build a ``tp.attestation.detached.v1`` payload binding to evidence hash."""
    if not isinstance(signature, Mapping):
        raise ValueError("signature must be an object")
    if toolchain is not None and not isinstance(toolchain, Mapping):
        raise ValueError("toolchain must be an object or null")
    if claims is not None and not isinstance(claims, Mapping):
        raise ValueError("claims must be an object or null")
    if signed_at is not None and not isinstance(signed_at, str):
        raise ValueError("signed_at must be a string or null")

    _validate_evidence_payload_surface(evidence_payload)

    stored_evidence_sha256 = _validate_sha256(evidence_payload["evidence_sha256"], field="evidence_sha256")
    if enforce_recompute_match:
        recomputed = _recompute_evidence_sha256_from_projected_envelope(evidence_payload)
        if recomputed != stored_evidence_sha256:
            raise ValueError("evidence_sha256 mismatch: projected_envelope does not reproduce stored evidence_sha256")

    file_sha256 = evidence_payload.get("file_sha256")
    if file_sha256 is not None:
        file_sha256 = _validate_sha256(file_sha256, field="file_sha256")

    bundle_root_sha256 = evidence_payload.get("bundle_root_sha256")
    if bundle_root_sha256 is not None:
        bundle_root_sha256 = _validate_sha256(bundle_root_sha256, field="bundle_root_sha256")

    attestation: dict[str, Any] = {
        "schema": ATTESTATION_SCHEMA_VERSION,
        "canonicalization": TP_CANONICAL_JSON_PROFILE,
        "subject": {
            "schema": EVIDENCE_SCHEMA_VERSION,
            "evidence_sha256": stored_evidence_sha256,
            "file_sha256": file_sha256,
            "bundle_root_sha256": bundle_root_sha256,
        },
        "signature": _validate_signature(signature),
        "signed_at": signed_at,
        "toolchain": dict(toolchain) if toolchain is not None else None,
        "claims": dict(claims) if claims is not None else None,
        "attestation_sha256": None,
    }

    attestation["attestation_sha256"] = compute_attestation_sha256(attestation)
    return attestation
