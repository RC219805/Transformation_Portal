"""Tests for detached attestation builders and evidence hash binding."""

from __future__ import annotations

from typing import Any

import pytest

from transformation_portal.attestation.detached import build_detached_attestation_payload, canonical_attestation_bytes
from transformation_portal.attestation.verify import bind_attestation_to_evidence, validate_detached_attestation_surface
from transformation_portal.ingest.evidence import build_evidence_payload, load_projection_profile


def _machine_extract_payload(*, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "schema": "tp.meta.machine.v1",
        "command": "extract",
        "success": True,
        "exit_code": 0,
        "error": None,
        "data": {
            "input_path": "/tmp/source.cr2",
            "success": True,
            "output_path": "/tmp/source.provenance.json",
            "elapsed_seconds": elapsed_seconds,
            "preset": "luxury",
            "error": None,
        },
    }


def test_detached_attestation_binds_to_evidence_sha256() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}

    attestation = build_detached_attestation_payload(evidence, signature=signature, enforce_recompute_match=True)
    assert attestation["subject"]["evidence_sha256"] == evidence["evidence_sha256"]


def test_attestation_canonical_bytes_are_deterministic() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}

    payload_a = build_detached_attestation_payload(evidence, signature=signature)
    payload_b = dict(payload_a)
    assert canonical_attestation_bytes(payload_a) == canonical_attestation_bytes(payload_b)


def test_attestation_recompute_check_detects_tamper() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}

    evidence["projected_envelope"]["data"]["preset"] = "tampered"

    with pytest.raises(ValueError, match="does not reproduce stored evidence_sha256"):
        build_detached_attestation_payload(evidence, signature=signature, enforce_recompute_match=True)


def test_validate_detached_attestation_surface_rejects_invalid_schema() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}
    attestation = build_detached_attestation_payload(evidence, signature=signature)
    attestation["schema"] = "tp.attestation.detached.v0"

    with pytest.raises(ValueError, match="attestation schema must be"):
        validate_detached_attestation_surface(attestation)


def test_validate_detached_attestation_surface_rejects_invalid_optional_sha_fields() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}
    attestation = build_detached_attestation_payload(evidence, signature=signature)
    attestation["subject"]["file_sha256"] = "invalid"

    with pytest.raises(ValueError, match="subject.file_sha256 must be a 64-character sha256 digest"):
        validate_detached_attestation_surface(attestation)


def test_validate_detached_attestation_surface_rejects_missing_signature_fields() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}
    attestation = build_detached_attestation_payload(evidence, signature=signature)
    attestation["signature"].pop("key_id")

    with pytest.raises(ValueError, match="signature.key_id must be a non-empty string"):
        validate_detached_attestation_surface(attestation)


def test_bind_attestation_to_evidence_rejects_mismatched_hash() -> None:
    evidence_a = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    payload_b = _machine_extract_payload(elapsed_seconds=2.0)
    payload_b["data"]["preset"] = "cinematic"
    evidence_b = build_evidence_payload(
        payload_b,
        projection_profile=load_projection_profile(),
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}
    attestation = build_detached_attestation_payload(evidence_a, signature=signature)

    with pytest.raises(ValueError, match="attestation does not bind to this evidence payload"):
        bind_attestation_to_evidence(attestation, evidence_b)
