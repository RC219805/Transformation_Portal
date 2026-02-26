"""Tests for tp.meta.evidence.v1 projection and hashing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.ingest.evidence import (
    EVIDENCE_SCHEMA_VERSION,
    build_evidence_payload,
    canonical_evidence_bytes,
    load_projection_profile,
    project_machine_envelope,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVIDENCE_SCHEMA_PATH = PROJECT_ROOT / "docs" / "schemas" / "evidence" / "tp.meta.evidence.v1" / "evidence.schema.json"


def _check_system_payload(*, exiftool_version: str, git_version: str) -> dict[str, Any]:
    return {
        "schema": "tp.meta.machine.v1",
        "command": "check-system",
        "success": True,
        "exit_code": 0,
        "error": None,
        "data": {
            "all_required_ok": True,
            "errors": [],
            "exiftool_available": True,
            "exiftool_version": exiftool_version,
            "git_available": True,
            "git_version": git_version,
            "ingest_module_available": True,
            "libraw_version": "0.21.0",
            "pydantic_available": True,
            "pydantic_version": "2.10.0",
            "rawpy_available": True,
            "rawpy_version": "0.24.0",
        },
    }


def _extract_payload(*, elapsed_seconds: float) -> dict[str, Any]:
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


def _extract_batch_payload(*, item_elapsed_seconds: float) -> dict[str, Any]:
    return {
        "schema": "tp.meta.machine.v1",
        "command": "extract-batch",
        "success": True,
        "exit_code": 0,
        "error": None,
        "data": {
            "input_root": "/tmp/input",
            "output_dir": "/tmp/output",
            "fail_fast": False,
            "preserve_structure": True,
            "success": True,
            "items": [
                {
                    "path": "/tmp/input/a.cr2",
                    "success": True,
                    "output_path": "/tmp/output/a.provenance.json",
                    "elapsed_seconds": item_elapsed_seconds,
                    "error": None,
                }
            ],
            "summary_counts": {
                "total": 1,
                "success": 1,
                "failure": 0,
                "by_exit_code": {
                    "SCHEMA_VALIDATION_FAILED": 0,
                    "BIT_DEPTH_VIOLATION": 0,
                    "GAMMA_VIOLATION": 0,
                    "SCHEMA_DRIFT": 0,
                    "OTHER_FAILURE": 0,
                },
            },
            "dominant_error": None,
        },
    }


def test_projection_profile_drops_declared_volatile_fields() -> None:
    profile = load_projection_profile()
    payload = _check_system_payload(exiftool_version="12.70", git_version="git version 2.50.0")

    projected = project_machine_envelope(payload, projection_profile=profile)

    assert projected["schema"] == "tp.meta.machine.v1"
    assert projected["command"] == "check-system"
    assert "exiftool_version" not in projected["data"]
    assert "git_version" not in projected["data"]
    assert "pydantic_version" not in projected["data"]
    assert "rawpy_version" not in projected["data"]
    assert "libraw_version" not in projected["data"]


def test_evidence_hash_is_stable_across_volatile_machine_fields() -> None:
    profile = load_projection_profile()

    check_a = build_evidence_payload(
        _check_system_payload(exiftool_version="12.70", git_version="git version 2.50.0"),
        projection_profile=profile,
    )
    check_b = build_evidence_payload(
        _check_system_payload(exiftool_version="13.10", git_version="git version 2.55.0"),
        projection_profile=profile,
    )
    assert check_a["evidence_sha256"] == check_b["evidence_sha256"]

    extract_a = build_evidence_payload(_extract_payload(elapsed_seconds=0.12), projection_profile=profile)
    extract_b = build_evidence_payload(_extract_payload(elapsed_seconds=9.81), projection_profile=profile)
    assert extract_a["evidence_sha256"] == extract_b["evidence_sha256"]

    batch_a = build_evidence_payload(_extract_batch_payload(item_elapsed_seconds=0.1), projection_profile=profile)
    batch_b = build_evidence_payload(_extract_batch_payload(item_elapsed_seconds=6.5), projection_profile=profile)
    assert batch_a["evidence_sha256"] == batch_b["evidence_sha256"]
    assert batch_a["projected_envelope"]["data"]["items"][0].get("elapsed_seconds") is None


def test_evidence_payload_validates_against_schema() -> None:
    jsonschema = pytest.importorskip("jsonschema")
    Draft202012Validator = jsonschema.Draft202012Validator
    schema = json.loads(EVIDENCE_SCHEMA_PATH.read_text(encoding="utf-8"))
    payload = build_evidence_payload(_extract_payload(elapsed_seconds=1.23), projection_profile=load_projection_profile())

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(payload), key=lambda error: (list(error.path), error.message))
    assert not errors, "\n".join(f"{list(error.path)}: {error.message}" for error in errors)

    assert payload["schema"] == EVIDENCE_SCHEMA_VERSION
    assert payload["canonicalization"] == "tp.canonical.json.v1"


def test_canonical_evidence_bytes_are_deterministic() -> None:
    payload_a = build_evidence_payload(_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile())
    payload_b = {
        "projected_envelope": payload_a["projected_envelope"],
        "schema": payload_a["schema"],
        "evidence_sha256": payload_a["evidence_sha256"],
        "source_schema": payload_a["source_schema"],
        "command": payload_a["command"],
        "success": payload_a["success"],
        "exit_code": payload_a["exit_code"],
        "envelope_projection_profile": payload_a["envelope_projection_profile"],
        "canonicalization": payload_a["canonicalization"],
        "file_sha256": payload_a["file_sha256"],
        "bundle_root_sha256": payload_a["bundle_root_sha256"],
        "signature": payload_a["signature"],
        "timestamp": payload_a["timestamp"],
    }

    assert canonical_evidence_bytes(payload_a) == canonical_evidence_bytes(payload_b)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda payload: payload.update({"command": "not-a-command"}), "machine payload command must be one of"),
        (lambda payload: payload.update({"success": "yes"}), "machine payload success must be a boolean"),
        (lambda payload: payload.update({"exit_code": 999}), r"machine payload exit_code must be an integer in \[0,255\]"),
        (
            lambda payload: payload.update({"success": False, "exit_code": 0}),
            "machine payload exit_code must be non-zero when success is false",
        ),
        (lambda payload: payload.update({"data": []}), "machine payload data must be an object"),
    ],
)
def test_build_evidence_payload_rejects_invalid_machine_contract_surface(
    mutate: Any,
    message: str,
) -> None:
    payload = _extract_payload(elapsed_seconds=1.23)
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        build_evidence_payload(payload, projection_profile=load_projection_profile())
