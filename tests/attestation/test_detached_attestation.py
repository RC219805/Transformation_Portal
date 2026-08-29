"""Tests for detached attestation builders and evidence hash binding."""

from __future__ import annotations

import hashlib
from typing import Any

import pytest

from transformation_portal.attestation.detached import (
    build_detached_attestation_payload,
    canonical_attestation_bytes,
    canonical_attestation_preimage_bytes,
)
from transformation_portal.attestation.verify import bind_attestation_to_evidence, validate_detached_attestation_surface
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.ingest.evidence import build_evidence_payload, load_projection_profile

pytestmark = pytest.mark.unit


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


def _anchored_evidence() -> dict[str, Any]:
    machine_payload = _machine_extract_payload(elapsed_seconds=1.0)
    machine_payload["data"]["file_integrity"] = {"sha256": "a" * 64}
    return build_evidence_payload(
        machine_payload,
        projection_profile=load_projection_profile(),
        bundle_root_sha256="b" * 64,
    )


def test_detached_attestation_binds_to_evidence_sha256() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}

    attestation = build_detached_attestation_payload(evidence, signature=signature, enforce_recompute_match=True)
    assert attestation["subject"]["evidence_sha256"] == evidence["evidence_sha256"]
    assert bind_attestation_to_evidence(attestation, evidence) is None


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


def test_attestation_preimage_canonical_bytes_are_deterministic_across_field_order() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    reordered_evidence = {
        "projected_envelope": evidence["projected_envelope"],
        "bundle_root_sha256": evidence["bundle_root_sha256"],
        "file_sha256": evidence["file_sha256"],
        "evidence_sha256": evidence["evidence_sha256"],
        "schema": evidence["schema"],
    }

    preimage_a = canonical_attestation_preimage_bytes(evidence)
    preimage_b = canonical_attestation_preimage_bytes(reordered_evidence)

    assert preimage_a == preimage_b
    assert hashlib.sha256(preimage_a).hexdigest() == "3c21ece20c3120fb54ac616eb6ee616a754e37eba0a694274a4e6d729673be39"


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


def test_bind_attestation_to_evidence_rejects_tampered_projected_envelope() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}
    attestation = build_detached_attestation_payload(evidence, signature=signature)

    evidence["projected_envelope"]["data"]["preset"] = "tampered"

    with pytest.raises(ValueError, match="projected_envelope does not reproduce stored evidence_sha256"):
        bind_attestation_to_evidence(attestation, evidence)


def test_bind_attestation_to_evidence_normalizes_valid_digest_case() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}
    attestation = build_detached_attestation_payload(evidence, signature=signature)
    evidence["evidence_sha256"] = evidence["evidence_sha256"].upper()
    attestation["subject"]["evidence_sha256"] = attestation["subject"]["evidence_sha256"].upper()

    assert bind_attestation_to_evidence(attestation, evidence) is None


def test_bind_attestation_to_evidence_normalizes_secondary_anchor_case() -> None:
    evidence = _anchored_evidence()
    attestation = build_detached_attestation_payload(
        evidence,
        signature={"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"},
    )
    evidence["file_sha256"] = evidence["file_sha256"].upper()
    evidence["bundle_root_sha256"] = evidence["bundle_root_sha256"].upper()
    attestation["subject"]["file_sha256"] = attestation["subject"]["file_sha256"].upper()
    attestation["subject"]["bundle_root_sha256"] = attestation["subject"]["bundle_root_sha256"].upper()

    assert bind_attestation_to_evidence(attestation, evidence) is None


@pytest.mark.parametrize("anchor", ["file_sha256", "bundle_root_sha256"])
def test_bind_attestation_to_evidence_rejects_mutated_secondary_anchor(anchor: str) -> None:
    evidence = _anchored_evidence()
    attestation = build_detached_attestation_payload(
        evidence,
        signature={"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"},
    )
    attestation["subject"][anchor] = "c" * 64

    with pytest.raises(ValueError, match=rf"{anchor} mismatch"):
        bind_attestation_to_evidence(attestation, evidence)


@pytest.mark.parametrize("anchor", ["file_sha256", "bundle_root_sha256"])
def test_bind_attestation_to_evidence_rejects_one_sided_missing_secondary_anchor(anchor: str) -> None:
    evidence = _anchored_evidence()
    attestation = build_detached_attestation_payload(
        evidence,
        signature={"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"},
    )
    attestation["subject"].pop(anchor)

    with pytest.raises(ValueError, match=rf"{anchor} mismatch"):
        bind_attestation_to_evidence(attestation, evidence)


def test_bind_attestation_to_evidence_treats_missing_and_null_as_unbound() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0),
        projection_profile=load_projection_profile(),
    )
    attestation = build_detached_attestation_payload(
        evidence,
        signature={"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"},
    )
    attestation["subject"].pop("file_sha256")
    evidence.pop("bundle_root_sha256")

    assert bind_attestation_to_evidence(attestation, evidence) is None


def test_bind_attestation_to_evidence_rejects_file_anchor_divorced_from_projection() -> None:
    evidence = _anchored_evidence()
    attestation = build_detached_attestation_payload(
        evidence,
        signature={"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"},
    )
    evidence["file_sha256"] = "c" * 64
    attestation["subject"]["file_sha256"] = "c" * 64

    with pytest.raises(ValueError, match="file_sha256 does not match projected_envelope file digest"):
        bind_attestation_to_evidence(attestation, evidence)


def test_bind_attestation_to_evidence_rejects_invalid_projected_file_anchor() -> None:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0),
        projection_profile=load_projection_profile(),
    )
    evidence["projected_envelope"]["data"]["file_integrity"] = {"sha256": "invalid"}
    evidence["evidence_sha256"] = hashlib.sha256(canonicalize_json(evidence["projected_envelope"])).hexdigest()
    attestation = build_detached_attestation_payload(
        evidence,
        signature={"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"},
    )

    with pytest.raises(ValueError, match="projected_envelope.data.file_integrity.sha256"):
        bind_attestation_to_evidence(attestation, evidence)
