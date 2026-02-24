"""Tests for deterministic ingest machine-output serializers."""

from __future__ import annotations

from pathlib import Path

from transformation_portal.ingest.errors import (
    BitDepthViolation,
    IngestExitCode,
    OtherIngestFailure,
    SchemaDriftFailure,
    SchemaValidationFailure,
)
from transformation_portal.ingest.machine_output import (
    batch_result_to_dict,
    dump_json,
    error_to_dict,
    exit_code_to_dict,
    extract_result_to_dict,
    validate_result_to_dict,
)
from transformation_portal.ingest.metadata_service import BatchExtractResult, BatchItemResult, ExtractResult, ValidateResult


def test_exit_code_to_dict_contains_name_and_value() -> None:
    payload = exit_code_to_dict(IngestExitCode.SCHEMA_DRIFT)
    assert payload == {"name": "SCHEMA_DRIFT", "value": 4}


def test_error_to_dict_is_typed_and_repr_free() -> None:
    error = SchemaValidationFailure("schema mismatch")
    payload = error_to_dict(error)

    assert payload["type"] == "SchemaValidationFailure"
    assert payload["message"] == "schema mismatch"
    assert payload["exit_code"] == {"name": "SCHEMA_VALIDATION_FAILED", "value": 1}
    serialized = dump_json({"error": payload}, pretty=False)
    assert "SchemaValidationFailure(" not in serialized
    assert "0x" not in serialized


def test_extract_result_to_dict_shape() -> None:
    result = ExtractResult(
        path=Path("/tmp/input.cr2"),
        success=False,
        output_path=None,
        elapsed_seconds=1.25,
        error=OtherIngestFailure("boom"),
    )

    payload = extract_result_to_dict(result, preset="luxury")

    assert payload["input_path"] == "/tmp/input.cr2"
    assert payload["output_path"] is None
    assert payload["preset"] == "luxury"
    assert payload["error"]["type"] == "OtherIngestFailure"


def test_validate_result_to_dict_serializes_typed_errors() -> None:
    errors = [SchemaValidationFailure("schema"), SchemaDriftFailure("drift")]
    result = ValidateResult(
        success=False,
        errors=errors,
        dominant_error=errors[1],
    )

    payload = validate_result_to_dict(
        result,
        sidecar_path=Path("/tmp/sidecar.json"),
        strict=True,
    )

    assert payload["sidecar_path"] == "/tmp/sidecar.json"
    assert payload["strict"] is True
    assert [item["type"] for item in payload["errors"]] == [
        "SchemaValidationFailure",
        "SchemaDriftFailure",
    ]
    assert payload["dominant_error"]["type"] == "SchemaDriftFailure"


def test_batch_result_to_dict_orders_exit_codes_stably() -> None:
    result = BatchExtractResult(
        items=[
            BatchItemResult(
                path=Path("/tmp/z.cr2"),
                success=True,
                output_path=Path("/tmp/out/z.provenance.json"),
                elapsed_seconds=0.1,
            ),
            BatchItemResult(
                path=Path("/tmp/a.cr2"),
                success=False,
                output_path=None,
                elapsed_seconds=0.2,
                error=BitDepthViolation("8-bit"),
            ),
        ],
        total_elapsed=0.3,
        summary_counts={
            "total": 2,
            "success": 1,
            "failure": 1,
            "by_exit_code": {
                "OTHER_FAILURE": 1,
                "BIT_DEPTH_VIOLATION": 3,
            },
        },
        dominant_error=BitDepthViolation("8-bit"),
    )

    payload = batch_result_to_dict(
        result,
        input_root=Path("/tmp"),
        output_dir=Path("/tmp/out"),
        fail_fast=False,
        preserve_structure=True,
    )

    expected_keys = [code.name for code in IngestExitCode if code != IngestExitCode.SUCCESS]
    assert list(payload["summary_counts"]["by_exit_code"].keys()) == expected_keys
    assert payload["items"][0]["path"] == "/tmp/z.cr2"
    assert payload["items"][1]["path"] == "/tmp/a.cr2"


def test_dump_json_is_deterministic() -> None:
    first = dump_json({"b": 1, "a": 2}, pretty=False)
    second = dump_json({"a": 2, "b": 1}, pretty=False)
    assert first == second
