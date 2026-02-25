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
    batch_item_to_dict,
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
    assert payload["success"] is False
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
    assert payload["success"] is False
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
    assert payload["success"] is False
    assert list(payload["summary_counts"]["by_exit_code"].keys()) == expected_keys
    assert payload["items"][0]["path"] == "/tmp/z.cr2"
    assert payload["items"][1]["path"] == "/tmp/a.cr2"


def test_batch_item_to_dict_handles_none_values() -> None:
    item = BatchItemResult(
        path=Path("/tmp/input.cr2"),
        success=False,
        output_path=None,
        elapsed_seconds=0.0,
        error=None,
    )

    payload = batch_item_to_dict(item)

    assert payload == {
        "path": "/tmp/input.cr2",
        "success": False,
        "output_path": None,
        "elapsed_seconds": 0.0,
        "error": None,
    }


def test_batch_result_to_dict_includes_unknown_exit_code_names() -> None:
    result = BatchExtractResult(
        items=[],
        total_elapsed=0.0,
        summary_counts={
            "total": 0,
            "success": 0,
            "failure": 0,
            "by_exit_code": {
                "OTHER_FAILURE": 2,
                "CUSTOM_UNKNOWN_CODE": 7,
            },
        },
        dominant_error=None,
    )

    payload = batch_result_to_dict(
        result,
        input_root=Path("/tmp/input"),
        output_dir=Path("/tmp/output"),
        fail_fast=False,
        preserve_structure=True,
    )

    by_exit = payload["summary_counts"]["by_exit_code"]
    known_keys = [code.name for code in IngestExitCode if code != IngestExitCode.SUCCESS]
    assert list(by_exit.keys())[: len(known_keys)] == known_keys
    assert by_exit["OTHER_FAILURE"] == 2
    assert by_exit["CUSTOM_UNKNOWN_CODE"] == 7
    assert payload["items"] == []
    assert payload["dominant_error"] is None
    assert payload["success"] is True


def test_dump_json_is_deterministic() -> None:
    first = dump_json({"b": 1, "a": 2}, pretty=False)
    second = dump_json({"a": 2, "b": 1}, pretty=False)
    assert first == second


def test_golden_contract_extract_result_canonical_output() -> None:
    """Golden master test: validate exact byte-level output for extract result.

    This test ensures the machine contract remains stable across Python versions,
    platforms, and future refactorings. Any change to this output is a breaking
    change to the tp.meta.machine.v1 contract.
    """
    result = ExtractResult(
        path=Path("/tmp/test.cr2"),
        success=False,
        output_path=None,
        elapsed_seconds=1.5,
        error=SchemaValidationFailure("golden test error"),
    )

    payload = extract_result_to_dict(result, preset="stable")
    canonical_json = dump_json(payload, pretty=False)

    # Golden reference: this exact byte sequence is the contract.
    # If this assertion fails, you are changing the machine contract.
    # You MUST bump the schema version if you need to change this.
    expected = (
        '{"elapsed_seconds":1.5,'
        '"error":{"exit_code":{"name":"SCHEMA_VALIDATION_FAILED","value":1},'
        '"message":"golden test error",'
        '"priority":10,'
        '"type":"SchemaValidationFailure"},'
        '"input_path":"/tmp/test.cr2",'
        '"output_path":null,'
        '"preset":"stable",'
        '"success":false}'
    )
    assert canonical_json == expected, (
        "Machine contract violation: extract_result_to_dict output has changed. "
        "This breaks tp.meta.machine.v1 contract. If intentional, bump MACHINE_SCHEMA_VERSION."
    )


def test_golden_contract_validate_result_canonical_output() -> None:
    """Golden master test: validate exact byte-level output for validate result."""
    result = ValidateResult(
        success=False,
        errors=[SchemaDriftFailure("drift error")],
        dominant_error=SchemaDriftFailure("drift error"),
    )

    payload = validate_result_to_dict(
        result,
        sidecar_path=Path("/tmp/sidecar.json"),
        strict=True,
    )
    canonical_json = dump_json(payload, pretty=False)

    expected = (
        '{"dominant_error":{"exit_code":{"name":"SCHEMA_DRIFT","value":4},'
        '"message":"drift error",'
        '"priority":40,'
        '"type":"SchemaDriftFailure"},'
        '"errors":[{"exit_code":{"name":"SCHEMA_DRIFT","value":4},'
        '"message":"drift error",'
        '"priority":40,'
        '"type":"SchemaDriftFailure"}],'
        '"sidecar_path":"/tmp/sidecar.json",'
        '"strict":true,'
        '"success":false}'
    )
    assert canonical_json == expected, (
        "Machine contract violation: validate_result_to_dict output has changed. "
        "This breaks tp.meta.machine.v1 contract. If intentional, bump MACHINE_SCHEMA_VERSION."
    )
