"""Tests for detached attestation builders and evidence hash binding."""

from __future__ import annotations

from typing import Any

import pytest

from transformation_portal.attestation.detached import build_detached_attestation_payload, canonical_attestation_bytes
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
