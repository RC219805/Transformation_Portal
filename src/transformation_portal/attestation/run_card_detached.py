"""Detached attestation payload builders for Lux run cards."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any

from transformation_portal.ingest.canonical_json import TP_CANONICAL_JSON_PROFILE, canonicalize_json
from transformation_portal.lux_depth_v3.run_card_contract import (
    get_run_card_schema_uri_for_payload,
    infer_run_card_version,
    with_inferred_run_card_version,
)
from transformation_portal.schemas.run_card import RUN_CARD_SCHEMA_URIS

RUN_CARD_ATTESTATION_SCHEMA_VERSION = "tp.run_card.attestation.detached.v1"
RUN_CARD_ATTESTATION_PREIMAGE_SCHEMA_VERSION = "tp.run_card.attestation.detached.v1.preimage"
RUN_CARD_V1_SCHEMA_URI = RUN_CARD_SCHEMA_URIS["v1"]
RUN_CARD_V2_SCHEMA_URI = RUN_CARD_SCHEMA_URIS["v2"]
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


def _validated_artifact_index(run_card_payload: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    artifact_index = run_card_payload.get("artifact_index")
    if not isinstance(artifact_index, Sequence) or isinstance(artifact_index, (str, bytes, bytearray)):
        raise ValueError("run_card.artifact_index must be a list")
    return artifact_index


def build_run_card_commitment(run_card_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Build the versioned artifact commitment block for a run card."""
    payload = with_inferred_run_card_version(run_card_payload)
    version = infer_run_card_version(payload)
    artifact_index = _validated_artifact_index(payload)
    leaf_count = len(artifact_index)

    if version == "v1":
        commitment_sha256 = _validate_sha256(
            payload.get("artifact_merkle_root"),
            field="run_card.artifact_merkle_root",
        )
        return {
            "kind": "artifact_commitment_v1",
            "sha256": commitment_sha256,
            "leaf_count": leaf_count,
        }

    artifact_tree = payload.get("artifact_tree")
    if not isinstance(artifact_tree, Mapping):
        raise ValueError("run_card.artifact_tree must be present for v2 attestation")
    root_sha256 = _validate_sha256(
        artifact_tree.get("root_sha256"),
        field="run_card.artifact_tree.root_sha256",
    )
    algorithm = artifact_tree.get("algorithm")
    leaf_format = artifact_tree.get("leaf_format")
    tree_leaf_count = artifact_tree.get("leaf_count")
    if not isinstance(algorithm, str) or not algorithm:
        raise ValueError("run_card.artifact_tree.algorithm must be a non-empty string")
    if not isinstance(leaf_format, str) or not leaf_format:
        raise ValueError("run_card.artifact_tree.leaf_format must be a non-empty string")
    if not isinstance(tree_leaf_count, int) or tree_leaf_count < 0:
        raise ValueError("run_card.artifact_tree.leaf_count must be a non-negative integer")
    return {
        "kind": "artifact_tree_v2",
        "sha256": root_sha256,
        "leaf_count": tree_leaf_count,
        "algorithm": algorithm,
        "leaf_format": leaf_format,
    }


def _run_card_binding_subject(
    run_card_payload: Mapping[str, Any],
    *,
    run_card_bytes: bytes,
) -> dict[str, Any]:
    payload = with_inferred_run_card_version(run_card_payload)
    batch_id = payload.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("run_card.batch_id must be a non-empty string")
    config_fingerprint = payload.get("config_fingerprint")
    if not isinstance(config_fingerprint, Mapping):
        raise ValueError("run_card.config_fingerprint must be an object")
    config_fingerprint_sha256 = _validate_sha256(
        config_fingerprint.get("sha256"),
        field="run_card.config_fingerprint.sha256",
    )
    git_revision = payload.get("git_revision")
    if not isinstance(git_revision, Mapping):
        raise ValueError("run_card.git_revision must be an object")
    version = infer_run_card_version(payload)
    return {
        "run_card_version": version,
        "run_card_schema": get_run_card_schema_uri_for_payload(payload),
        "batch_id": batch_id,
        "run_card_sha256": compute_run_card_sha256(run_card_bytes),
        "config_fingerprint_sha256": config_fingerprint_sha256,
        "git_revision": dict(git_revision),
        "artifact_commitment": build_run_card_commitment(payload),
    }


def validate_run_card_attestable_surface(run_card_payload: Mapping[str, Any]) -> None:
    """Validate the run-card fields required for trust binding."""
    _run_card_binding_subject(run_card_payload, run_card_bytes=b"")


def validate_run_card_v2_surface(run_card_payload: Mapping[str, Any]) -> None:
    """Backward-compatible v2-only surface validator."""
    payload = with_inferred_run_card_version(run_card_payload)
    if infer_run_card_version(payload) != "v2":
        raise ValueError("run_card detached attestation only supports v2 payloads in this path")
    _run_card_binding_subject(payload, run_card_bytes=b"")


def build_run_card_detached_attestation_preimage(
    run_card_payload: Mapping[str, Any],
    *,
    run_card_bytes: bytes,
) -> dict[str, Any]:
    """Build the signing preimage for a detached run-card attestation."""
    return {
        "schema": RUN_CARD_ATTESTATION_PREIMAGE_SCHEMA_VERSION,
        "subject": _run_card_binding_subject(run_card_payload, run_card_bytes=run_card_bytes),
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
    """Build a detached attestation binding to a run-card payload."""
    attestation = {
        "schema": RUN_CARD_ATTESTATION_SCHEMA_VERSION,
        "canonicalization": TP_CANONICAL_JSON_PROFILE,
        "subject": _run_card_binding_subject(run_card_payload, run_card_bytes=run_card_bytes),
        "signature": _validate_signature(signature),
        "signed_at": signed_at,
        "toolchain": dict(toolchain) if toolchain is not None else None,
        "claims": dict(claims) if claims is not None else None,
        "attestation_sha256": None,
    }
    attestation["attestation_sha256"] = compute_run_card_attestation_sha256(attestation)
    return attestation


def _validate_artifact_commitment(commitment: Mapping[str, Any]) -> dict[str, Any]:
    kind = commitment.get("kind")
    if kind not in {"artifact_commitment_v1", "artifact_tree_v2"}:
        raise ValueError("subject.artifact_commitment.kind must be artifact_commitment_v1 or artifact_tree_v2")
    leaf_count = commitment.get("leaf_count")
    if not isinstance(leaf_count, int) or leaf_count < 0:
        raise ValueError("subject.artifact_commitment.leaf_count must be a non-negative integer")
    normalized: dict[str, Any] = {
        "kind": kind,
        "sha256": _validate_sha256(commitment.get("sha256"), field="subject.artifact_commitment.sha256"),
        "leaf_count": leaf_count,
    }
    if kind == "artifact_tree_v2":
        algorithm = commitment.get("algorithm")
        leaf_format = commitment.get("leaf_format")
        if not isinstance(algorithm, str) or not algorithm:
            raise ValueError("subject.artifact_commitment.algorithm must be a non-empty string for v2")
        if not isinstance(leaf_format, str) or not leaf_format:
            raise ValueError("subject.artifact_commitment.leaf_format must be a non-empty string for v2")
        normalized["algorithm"] = algorithm
        normalized["leaf_format"] = leaf_format
    return normalized


def validate_run_card_detached_attestation_surface(attestation: Mapping[str, Any]) -> None:
    """Validate detached run-card attestation surface fields."""
    if attestation.get("schema") != RUN_CARD_ATTESTATION_SCHEMA_VERSION:
        raise ValueError(f"attestation schema must be {RUN_CARD_ATTESTATION_SCHEMA_VERSION}")
    if attestation.get("canonicalization") != TP_CANONICAL_JSON_PROFILE:
        raise ValueError(f"attestation canonicalization must be {TP_CANONICAL_JSON_PROFILE}")
    subject = attestation.get("subject")
    if not isinstance(subject, Mapping):
        raise ValueError("attestation subject must be an object")
    run_card_version = subject.get("run_card_version")
    if run_card_version not in {"v1", "v2"}:
        raise ValueError("attestation subject run_card_version must be v1 or v2")
    expected_schema = RUN_CARD_SCHEMA_URIS[run_card_version]
    if subject.get("run_card_schema") != expected_schema:
        raise ValueError(f"attestation subject run_card_schema must be {expected_schema}")
    batch_id = subject.get("batch_id")
    if not isinstance(batch_id, str) or not batch_id:
        raise ValueError("attestation subject batch_id must be a non-empty string")
    _validate_sha256(subject.get("run_card_sha256"), field="subject.run_card_sha256")
    _validate_sha256(subject.get("config_fingerprint_sha256"), field="subject.config_fingerprint_sha256")
    git_revision = subject.get("git_revision")
    if not isinstance(git_revision, Mapping):
        raise ValueError("attestation subject git_revision must be an object")
    artifact_commitment = subject.get("artifact_commitment")
    if not isinstance(artifact_commitment, Mapping):
        raise ValueError("attestation subject artifact_commitment must be an object")
    _validate_artifact_commitment(artifact_commitment)
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
    expected_subject = _run_card_binding_subject(run_card_payload, run_card_bytes=run_card_bytes)
    subject = attestation["subject"]
    for field in (
        "run_card_version",
        "run_card_schema",
        "batch_id",
        "run_card_sha256",
        "config_fingerprint_sha256",
        "git_revision",
        "artifact_commitment",
    ):
        if subject.get(field) != expected_subject[field]:
            raise ValueError(f"attestation does not bind to this run card: {field} mismatch")


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
