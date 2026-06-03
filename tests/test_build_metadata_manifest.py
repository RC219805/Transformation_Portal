"""Tests for Phase 4D metadata object hashing and metadata manifest builder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tp.phase4.hash_capture_metadata import (
    METADATA_CONTRACT_VERSION,
    MetadataManifestInputError,
    MetadataSchemaValidationError,
    build_metadata_manifest_payload,
    compute_metadata_sha256,
    serialize_metadata_manifest,
)
from tp.phase4.validation_helpers import validate_records_with_schema

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_TOOL = PROJECT_ROOT / "tools" / "extract_capture_metadata.py"
MANIFEST_TOOL = PROJECT_ROOT / "tools" / "build_metadata_manifest.py"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "phase4"
GOLDEN_CAPTURE = PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_capture_metadata.tp.meta.capture.v1.json"
GOLDEN_MANIFEST = PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_metadata_manifest.tp.meta.capture_manifest.v1.json"

pytestmark = [pytest.mark.regression, pytest.mark.golden]


def _build_fake_exiftool(tmp_path: Path) -> Path:
    fake_exiftool_path = tmp_path / "exiftool"
    script = """#!/usr/bin/env python3
import json
import sys

files = [arg for arg in sys.argv[1:] if not arg.startswith("-")]
records = []

for source_file in files:
    records.append(
        {
            "SourceFile": source_file,
            "Make": " Canon ",
            "Model": "EOS R5",
            "LensModel": "RF24-70mm F2.8 L IS USM",
            "GPSDateStamp": "2024:06:30",
            "GPSTimeStamp": "12:34:56",
            "GPSLatitude": 34.123456789,
            "GPSLongitude": -118.987654321,
            "FocalLength": 24.98765,
            "FNumber": 5.6789,
            "ExposureTime": "1/120",
            "ExposureCompensation": "-0.3333",
            "Orientation": 6,
            "DateTimeOriginal": "2024:06:30 05:34:56",
            "OffsetTimeOriginal": "-07:00"
        }
    )

sys.stdout.write(json.dumps(records))
"""
    fake_exiftool_path.write_text(script, encoding="utf-8")
    fake_exiftool_path.chmod(0o755)
    return fake_exiftool_path


def _run_extract_cli(*, input_root: Path, out_path: Path, fake_exiftool: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PATH"] = f"{fake_exiftool.parent}:{env.get('PATH', '')}"
    command = [
        sys.executable,
        str(EXTRACT_TOOL),
        "--input-root",
        str(input_root),
        "--out",
        str(out_path),
    ]
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _run_manifest_cli(
    *,
    input_path: Path,
    out_path: Path,
    strict_input_order: bool | None = None,
    require_fingerprint_match: bool | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        str(MANIFEST_TOOL),
        "--input",
        str(input_path),
        "--out",
        str(out_path),
    ]
    if strict_input_order is False:
        command.append("--no-strict-input-order")
    if require_fingerprint_match is False:
        command.append("--no-require-fingerprint-match")

    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _load_golden_capture_records() -> list[dict[str, Any]]:
    return json.loads(GOLDEN_CAPTURE.read_text(encoding="utf-8"))


def _deepcopy_record(record: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(record))


def test_phase4d_build_payload_direct_success_with_fingerprint() -> None:
    pytest.importorskip("jsonschema")
    records = _load_golden_capture_records()
    fingerprint = records[0]["extractor"]["config_fingerprint_sha256"]

    payload = build_metadata_manifest_payload(
        records,
        metadata_schema={},
        manifest_schema={},
        required_config_fingerprint_sha256=fingerprint,
    )

    assert payload == json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    assert serialize_metadata_manifest(payload).endswith(b"\n")


def test_phase4d_metadata_hash_is_whitespace_and_key_order_independent() -> None:
    record = _load_golden_capture_records()[0]
    reordered = {key: record[key] for key in reversed(list(record.keys()))}
    pretty = json.dumps(reordered, indent=2, ensure_ascii=False)
    parsed = json.loads(pretty)

    first = compute_metadata_sha256(record)
    second = compute_metadata_sha256(parsed)
    assert first == second


def test_phase4d_metadata_validation_normalizes_validator_runtime_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _ExplodingValidator:
        def iter_errors(self, _record: dict[str, Any]) -> list[Any]:
            raise ValueError("cannot convert float NaN to integer")

    def _fake_build_validator(schema: dict[str, Any], *, error_cls: type[Exception], label: str) -> _ExplodingValidator:
        del schema, error_cls, label
        return _ExplodingValidator()

    import tp.phase4.schema_validation as schema_validation

    monkeypatch.setattr(schema_validation, "build_draft202012_validator", _fake_build_validator)
    with pytest.raises(MetadataSchemaValidationError, match="validator runtime error"):
        validate_records_with_schema([{}], {}, error_cls=MetadataSchemaValidationError, label="metadata")


def test_phase4d_build_payload_reports_missing_relative_path_without_keyerror() -> None:
    pytest.importorskip("jsonschema")
    records = [
        {
            "metadata_contract_version": METADATA_CONTRACT_VERSION,
            "file_sha256": "a" * 64,
        }
    ]

    with pytest.raises(MetadataManifestInputError, match=r"input metadata array record\[0\] missing relative_path"):
        build_metadata_manifest_payload(records, metadata_schema={}, manifest_schema={})


def test_phase4d_build_payload_rejects_contract_version_mismatch() -> None:
    pytest.importorskip("jsonschema")
    records = _load_golden_capture_records()
    records[0]["metadata_contract_version"] = "tp.meta.capture.v999"

    with pytest.raises(MetadataManifestInputError, match="contract mismatch"):
        build_metadata_manifest_payload(records, metadata_schema={}, manifest_schema={})


def test_phase4d_build_payload_rejects_fingerprint_mismatch() -> None:
    pytest.importorskip("jsonschema")
    records = _load_golden_capture_records()

    with pytest.raises(MetadataManifestInputError, match="fingerprint mismatch"):
        build_metadata_manifest_payload(
            records,
            metadata_schema={},
            manifest_schema={},
            required_config_fingerprint_sha256="0" * 64,
        )


def test_phase4d_build_payload_rejects_missing_extractor_when_fingerprint_required() -> None:
    pytest.importorskip("jsonschema")
    records = _load_golden_capture_records()
    del records[0]["extractor"]

    with pytest.raises(MetadataManifestInputError, match="missing extractor object"):
        build_metadata_manifest_payload(
            records,
            metadata_schema={},
            manifest_schema={},
            required_config_fingerprint_sha256="0" * 64,
        )


def test_phase4d_build_payload_wraps_canonical_serialization_errors() -> None:
    pytest.importorskip("jsonschema")
    records = _load_golden_capture_records()
    records[0]["non_jsonable"] = object()

    with pytest.raises(MetadataSchemaValidationError, match="canonical serialization failed"):
        build_metadata_manifest_payload(records, metadata_schema={}, manifest_schema={})


def test_phase4d_golden_manifest_matches_expected(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    out_path = tmp_path / "metadata_manifest.tp.meta.capture_manifest.v1.json"
    result = _run_manifest_cli(input_path=GOLDEN_CAPTURE, out_path=out_path)
    assert result.returncode == 0, result.stderr
    assert out_path.read_bytes() == GOLDEN_MANIFEST.read_bytes()


def test_phase4d_manifest_generation_is_deterministic_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    fake_exiftool = _build_fake_exiftool(tmp_path)

    capture_a = tmp_path / "capture_a.json"
    manifest_a = tmp_path / "manifest_a.json"
    extract_first = _run_extract_cli(input_root=FIXTURE_ROOT, out_path=capture_a, fake_exiftool=fake_exiftool)
    assert extract_first.returncode == 0, extract_first.stderr
    build_first = _run_manifest_cli(input_path=capture_a, out_path=manifest_a)
    assert build_first.returncode == 0, build_first.stderr

    capture_b = tmp_path / "capture_b.json"
    manifest_b = tmp_path / "manifest_b.json"
    extract_second = _run_extract_cli(input_root=FIXTURE_ROOT, out_path=capture_b, fake_exiftool=fake_exiftool)
    assert extract_second.returncode == 0, extract_second.stderr
    build_second = _run_manifest_cli(input_path=capture_b, out_path=manifest_b)
    assert build_second.returncode == 0, build_second.stderr

    assert manifest_a.read_bytes() == manifest_b.read_bytes()


def test_phase4d_cli_fails_on_unsorted_input_when_strict(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    base_record = _load_golden_capture_records()[0]
    record_a = _deepcopy_record(base_record)
    record_a["relative_path"] = "a/sample_01.dng"
    record_b = _deepcopy_record(base_record)
    record_b["relative_path"] = "b/sample_01.dng"
    unsorted_payload = [record_b, record_a]
    input_path = tmp_path / "unsorted_capture.json"
    input_path.write_text(json.dumps(unsorted_payload), encoding="utf-8")

    out_path = tmp_path / "metadata_manifest.json"
    result = _run_manifest_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 3
    assert "input metadata array must be sorted by relative_path" in result.stderr


def test_phase4d_cli_can_relax_input_order_and_sort_output(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    base_record = _load_golden_capture_records()[0]
    record_a = _deepcopy_record(base_record)
    record_a["relative_path"] = "a/sample_01.dng"
    record_b = _deepcopy_record(base_record)
    record_b["relative_path"] = "b/sample_01.dng"
    unsorted_payload = [record_b, record_a]
    input_path = tmp_path / "unsorted_capture.json"
    input_path.write_text(json.dumps(unsorted_payload), encoding="utf-8")

    out_path = tmp_path / "metadata_manifest.json"
    result = _run_manifest_cli(input_path=input_path, out_path=out_path, strict_input_order=False)
    assert result.returncode == 0, result.stderr
    manifest = json.loads(out_path.read_text(encoding="utf-8"))
    assert [entry["relative_path"] for entry in manifest["entries"]] == ["a/sample_01.dng", "b/sample_01.dng"]
    assert out_path.read_bytes().endswith(b"\n")


def test_phase4d_cli_fails_on_duplicate_relative_path(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    base_record = _load_golden_capture_records()[0]
    duplicated_payload = [_deepcopy_record(base_record), _deepcopy_record(base_record)]
    input_path = tmp_path / "duplicate_capture.json"
    input_path.write_text(json.dumps(duplicated_payload), encoding="utf-8")

    out_path = tmp_path / "metadata_manifest.json"
    result = _run_manifest_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 3
    assert "duplicate relative_path" in result.stderr


def test_phase4d_cli_fails_on_fingerprint_mismatch_by_default(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _load_golden_capture_records()
    payload[0]["extractor"]["config_fingerprint_sha256"] = "0" * 64
    input_path = tmp_path / "fingerprint_mismatch.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "metadata_manifest.json"
    result = _run_manifest_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 3
    assert "fingerprint mismatch" in result.stderr


def test_phase4d_cli_can_disable_fingerprint_match(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _load_golden_capture_records()
    payload[0]["extractor"]["config_fingerprint_sha256"] = "0" * 64
    input_path = tmp_path / "fingerprint_mismatch.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "metadata_manifest.json"
    result = _run_manifest_cli(input_path=input_path, out_path=out_path, require_fingerprint_match=False)
    assert result.returncode == 0, result.stderr


def test_phase4d_cli_fails_schema_validation_on_missing_required_field(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _load_golden_capture_records()
    del payload[0]["camera_model"]
    input_path = tmp_path / "missing_required_field.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "metadata_manifest.json"
    result = _run_manifest_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 4
    assert "Schema validation failure:" in result.stderr
    assert "record[0] schema validation failed at <root>" in result.stderr
    assert "camera_model" in result.stderr


def test_phase4d_cli_fails_schema_validation_on_nan_value(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _load_golden_capture_records()
    payload[0]["gps_latitude"] = float("nan")
    input_path = tmp_path / "nan_field.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "metadata_manifest.json"
    result = _run_manifest_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 4
    assert "Schema validation failure:" in result.stderr


def test_phase4d_cli_returns_exit_code_2_for_invalid_input_json(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.json"
    input_path.write_text("{this-is-invalid-json", encoding="utf-8")

    out_path = tmp_path / "metadata_manifest.json"
    result = _run_manifest_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 2
    assert "Input read/parse error:" in result.stderr


def test_phase4d_cli_returns_exit_code_3_for_non_array_input(tmp_path: Path) -> None:
    input_path = tmp_path / "not_array.json"
    input_path.write_text(json.dumps({"not": "an array"}), encoding="utf-8")

    out_path = tmp_path / "metadata_manifest.json"
    result = _run_manifest_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 3
    assert "input metadata payload must be a JSON array" in result.stderr
