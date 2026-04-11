"""Detached attestation payload builders for Lux run cards."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

from transformation_portal.ingest.canonical_json import TP_CANONICAL_JSON_PROFILE, canonicalize_json

RUN_CARD_ATTESTATION_SCHEMA_VERSION = "tp.run_card.attestation.detached.v1"
RUN_CARD_ATTESTATION_PREIMAGE_SCHEMA_VERSION = "tp.run_card.attestation.detached.v1.preimage"
RUN_CARD_V2_SCHEMA_URI = (
    "https://rc219805.github.io/Transformation_Portal/docs/schemas/run_card/run_card.v2.schema.json"
)
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


def compute_run_card_sha256(run_card_bytes: bytes) -> str:
    """Return the sha256 of the exact run-card bytes on disk."""
    return hashlib.sha256(run_card_bytes).hexdigest()


def validate_run_card_v2_surface(run_card_payload: Mapping[str, Any]) -> None:
    """Validate the run-card fields required for detached trust binding."""
    batch_id = run_card_payload.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("run_card.batch_id must be a non-empty string")
    artifact_tree = run_card_payload.get("artifact_tree")
    if not isinstance(artifact_tree, Mapping):
        raise ValueError("run_card.artifact_tree must be present for v2 detached attestation")
    if run_card_payload.get("artifact_merkle_root") is not None:
        raise ValueError("run_card detached attestation only supports v2 payloads without artifact_merkle_root")
    root_sha256 = artifact_tree.get("root_sha256")
    _validate_sha256(root_sha256, field="run_card.artifact_tree.root_sha256")


def build_run_card_detached_attestation_preimage(
    run_card_payload: Mapping[str, Any],
    *,
    run_card_bytes: bytes,
) -> dict[str, Any]:
    """Build the signing preimage for a detached run-card attestation."""
    validate_run_card_v2_surface(run_card_payload)

    return {
        "schema": RUN_CARD_ATTESTATION_PREIMAGE_SCHEMA_VERSION,
        "subject": {
            "run_card_schema": RUN_CARD_V2_SCHEMA_URI,
            "batch_id": run_card_payload["batch_id"],
            "run_card_sha256": compute_run_card_sha256(run_card_bytes),
            "artifact_tree_root_sha256": run_card_payload["artifact_tree"]["root_sha256"],
        },
    }


def canonical_run_card_attestation_preimage_bytes(
    run_card_payload: Mapping[str, Any],
    *,
    run_card_bytes: bytes,
) -> bytes:
    return canonicalize_json(build_run_card_detached_attestation_preimage(run_card_payload, run_card_bytes=run_card_bytes))


def canonical_run_card_attestation_bytes(attestation_payload: Mapping[str, Any]) -> bytes:
    return canonicalize_json(dict(attestation_payload))


def compute_run_card_attestation_sha256(attestation_payload: Mapping[str, Any]) -> str:
    payload = dict(attestation_payload)
    payload.pop("attestation_sha256", None)
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


def build_run_card_detached_attestation_payload(
    run_card_payload: Mapping[str, Any],
    *,
    run_card_bytes: bytes,
    signature: Mapping[str, Any],
    signed_at: str | None = None,
    toolchain: Mapping[str, Any] | None = None,
    claims: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a detached attestation binding to a run-card v2 payload."""
    validate_run_card_v2_surface(run_card_payload)
    attestation = {
        "schema": RUN_CARD_ATTESTATION_SCHEMA_VERSION,
        "canonicalization": TP_CANONICAL_JSON_PROFILE,
        "subject": {
            "run_card_schema": RUN_CARD_V2_SCHEMA_URI,
            "batch_id": run_card_payload["batch_id"],
            "run_card_sha256": compute_run_card_sha256(run_card_bytes),
            "artifact_tree_root_sha256": run_card_payload["artifact_tree"]["root_sha256"],
        },
        "signature": _validate_signature(signature),
        "signed_at": signed_at,
        "toolchain": dict(toolchain) if toolchain is not None else None,
        "claims": dict(claims) if claims is not None else None,
        "attestation_sha256": None,
    }
    attestation["attestation_sha256"] = compute_run_card_attestation_sha256(attestation)
    return attestation


def validate_run_card_detached_attestation_surface(attestation: Mapping[str, Any]) -> None:
    """Validate detached run-card attestation surface fields."""
    if attestation.get("schema") != RUN_CARD_ATTESTATION_SCHEMA_VERSION:
        raise ValueError(f"attestation schema must be {RUN_CARD_ATTESTATION_SCHEMA_VERSION}")
    if attestation.get("canonicalization") != TP_CANONICAL_JSON_PROFILE:
        raise ValueError(f"attestation canonicalization must be {TP_CANONICAL_JSON_PROFILE}")
    subject = attestation.get("subject")
    if not isinstance(subject, Mapping):
        raise ValueError("attestation subject must be an object")
    if subject.get("run_card_schema") != RUN_CARD_V2_SCHEMA_URI:
        raise ValueError(f"attestation subject run_card_schema must be {RUN_CARD_V2_SCHEMA_URI}")
    batch_id = subject.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("attestation subject batch_id must be a non-empty string")
    _validate_sha256(subject.get("run_card_sha256"), field="subject.run_card_sha256")
    _validate_sha256(subject.get("artifact_tree_root_sha256"), field="subject.artifact_tree_root_sha256")
    signature = attestation.get("signature")
    if not isinstance(signature, Mapping):
        raise ValueError("attestation signature must be an object")
    _validate_signature(signature)
    attestation_sha256 = attestation.get("attestation_sha256")
    if attestation_sha256 is not None:
        _validate_sha256(attestation_sha256, field="attestation_sha256")


def bind_run_card_detached_attestation(
    attestation: Mapping[str, Any],
    run_card_payload: Mapping[str, Any],
    *,
    run_card_bytes: bytes,
) -> None:
    """Assert that the detached attestation binds to the provided run card."""
    validate_run_card_detached_attestation_surface(attestation)
    validate_run_card_v2_surface(run_card_payload)
    subject = attestation["subject"]
    if subject["batch_id"] != run_card_payload["batch_id"]:
        raise ValueError("attestation does not bind to this run card: batch_id mismatch")
    if subject["run_card_sha256"] != compute_run_card_sha256(run_card_bytes):
        raise ValueError("attestation does not bind to this run card: run_card_sha256 mismatch")
    if subject["artifact_tree_root_sha256"] != run_card_payload["artifact_tree"]["root_sha256"]:
        raise ValueError("attestation does not bind to this run card: artifact_tree_root_sha256 mismatch")


def verify_run_card_attestation_self_hash(attestation: Mapping[str, Any], *, require_digest: bool = True) -> None:
    stored_digest = attestation.get("attestation_sha256")
    if stored_digest is None:
        if require_digest:
            raise ValueError("attestation_sha256 must be set for integrity verification")
        return
    normalized_stored_digest = _validate_sha256(stored_digest, field="attestation_sha256")
    recomputed_digest = compute_run_card_attestation_sha256(attestation)
    if recomputed_digest != normalized_stored_digest:
        raise ValueError("attestation_sha256 mismatch: payload does not match canonical digest")
