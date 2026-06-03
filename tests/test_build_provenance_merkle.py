"""Tests for Phase 4E provenance merkle builder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tp.phase4.hash_capture_metadata import METADATA_CONTRACT_VERSION, compute_metadata_sha256
from tp.phase4.provenance_capture import (
    PROVENANCE_CONTRACT_VERSION,
    PROVENANCE_MERKLE_CONTRACT_VERSION,
    ProvenanceInputError,
    build_provenance_merkle_payload,
    compute_provenance_entry_sha256,
    serialize_provenance_merkle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTRACT_TOOL = PROJECT_ROOT / "tools" / "extract_capture_metadata.py"
METADATA_MANIFEST_TOOL = PROJECT_ROOT / "tools" / "build_metadata_manifest.py"
PROVENANCE_MANIFEST_TOOL = PROJECT_ROOT / "tools" / "build_provenance_manifest.py"
PROVENANCE_MERKLE_TOOL = PROJECT_ROOT / "tools" / "build_provenance_merkle.py"
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "phase4"
GOLDEN_CAPTURE = PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_capture_metadata.tp.meta.capture.v1.json"
GOLDEN_PROVENANCE_MANIFEST = (
    PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_provenance_manifest.tp.meta.provenance.v1.json"
)
GOLDEN_PROVENANCE_MERKLE = (
    PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_provenance_merkle.tp.meta.provenance_merkle.v1.json"
)

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


def _run_command(command: list[str], *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _run_merkle_cli(
    *,
    input_path: Path,
    out_path: Path,
    strict_input_order: bool | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    command = [
        sys.executable,
        str(PROVENANCE_MERKLE_TOOL),
        "--input",
        str(input_path),
        "--out",
        str(out_path),
    ]
    if strict_input_order is False:
        command.append("--no-strict-input-order")

    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _deepcopy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _build_two_entry_provenance_manifest() -> dict[str, Any]:
    base_record = _load_json(GOLDEN_CAPTURE)[0]
    record_a = _deepcopy(base_record)
    record_a["relative_path"] = "a/sample_01.dng"
    record_b = _deepcopy(base_record)
    record_b["relative_path"] = "b/sample_01.dng"

    metadata_sha_a = compute_metadata_sha256(record_a)
    metadata_sha_b = compute_metadata_sha256(record_b)
    return {
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "entries": [
            {
                "relative_path": record_a["relative_path"],
                "file_sha256": record_a["file_sha256"],
                "metadata_sha256": metadata_sha_a,
                "provenance_entry_sha256": compute_provenance_entry_sha256(
                    file_sha256=record_a["file_sha256"],
                    metadata_sha256=metadata_sha_a,
                ),
            },
            {
                "relative_path": record_b["relative_path"],
                "file_sha256": record_b["file_sha256"],
                "metadata_sha256": metadata_sha_b,
                "provenance_entry_sha256": compute_provenance_entry_sha256(
                    file_sha256=record_b["file_sha256"],
                    metadata_sha256=metadata_sha_b,
                ),
            },
        ],
    }


def test_phase4e_build_merkle_payload_direct_success() -> None:
    pytest.importorskip("jsonschema")
    provenance_manifest = _build_two_entry_provenance_manifest()

    payload = build_provenance_merkle_payload(
        provenance_manifest,
        provenance_manifest_schema={},
        provenance_merkle_schema={},
    )

    assert payload["provenance_merkle_contract_version"] == PROVENANCE_MERKLE_CONTRACT_VERSION
    assert payload["provenance_contract_version"] == PROVENANCE_CONTRACT_VERSION
    assert payload["leaf_count"] == 2
    assert serialize_provenance_merkle(payload).endswith(b"\n")


def test_phase4e_build_merkle_payload_rejects_non_object_manifest() -> None:
    pytest.importorskip("jsonschema")

    with pytest.raises(ProvenanceInputError, match="provenance manifest payload must be a JSON object"):
        build_provenance_merkle_payload(
            [],
            provenance_manifest_schema={},
            provenance_merkle_schema={},
        )


def test_phase4e_build_merkle_payload_rejects_contract_mismatch() -> None:
    pytest.importorskip("jsonschema")
    provenance_manifest = _build_two_entry_provenance_manifest()
    provenance_manifest["provenance_contract_version"] = "tp.meta.provenance.v999"

    with pytest.raises(ProvenanceInputError, match="provenance manifest contract mismatch"):
        build_provenance_merkle_payload(
            provenance_manifest,
            provenance_manifest_schema={},
            provenance_merkle_schema={},
        )


def test_phase4e_build_merkle_payload_rejects_entries_that_are_not_array() -> None:
    pytest.importorskip("jsonschema")
    provenance_manifest = _build_two_entry_provenance_manifest()
    provenance_manifest["entries"] = {"not": "an array"}

    with pytest.raises(ProvenanceInputError, match="provenance manifest entries must be an array"):
        build_provenance_merkle_payload(
            provenance_manifest,
            provenance_manifest_schema={},
            provenance_merkle_schema={},
        )


def test_phase4e_build_merkle_payload_rejects_duplicate_relative_path_directly() -> None:
    pytest.importorskip("jsonschema")
    provenance_manifest = _build_two_entry_provenance_manifest()
    provenance_manifest["entries"] = [provenance_manifest["entries"][0], _deepcopy(provenance_manifest["entries"][0])]

    with pytest.raises(ProvenanceInputError, match="duplicate relative_path"):
        build_provenance_merkle_payload(
            provenance_manifest,
            provenance_manifest_schema={},
            provenance_merkle_schema={},
        )


def test_phase4e_build_merkle_payload_rejects_unsorted_entries_directly() -> None:
    pytest.importorskip("jsonschema")
    provenance_manifest = _build_two_entry_provenance_manifest()
    provenance_manifest["entries"] = list(reversed(provenance_manifest["entries"]))

    with pytest.raises(ProvenanceInputError, match="must be sorted by relative_path"):
        build_provenance_merkle_payload(
            provenance_manifest,
            provenance_manifest_schema={},
            provenance_merkle_schema={},
        )


def test_phase4e_build_merkle_payload_can_sort_when_strict_order_disabled() -> None:
    pytest.importorskip("jsonschema")
    provenance_manifest = _build_two_entry_provenance_manifest()
    expected = build_provenance_merkle_payload(
        provenance_manifest,
        provenance_manifest_schema={},
        provenance_merkle_schema={},
    )
    provenance_manifest["entries"] = list(reversed(provenance_manifest["entries"]))

    relaxed = build_provenance_merkle_payload(
        provenance_manifest,
        provenance_manifest_schema={},
        provenance_merkle_schema={},
        strict_input_order=False,
    )

    assert relaxed == expected


def test_phase4e_build_merkle_payload_rejects_invalid_leaf_digest_directly() -> None:
    pytest.importorskip("jsonschema")
    provenance_manifest = _build_two_entry_provenance_manifest()
    provenance_manifest["entries"][0]["provenance_entry_sha256"] = "not-a-sha256"

    with pytest.raises(ProvenanceInputError, match="provenance_entry_sha256"):
        build_provenance_merkle_payload(
            provenance_manifest,
            provenance_manifest_schema={},
            provenance_merkle_schema={},
        )


def test_phase4e_build_merkle_payload_rejects_empty_entries_directly() -> None:
    pytest.importorskip("jsonschema")
    provenance_manifest = _build_two_entry_provenance_manifest()
    provenance_manifest["entries"] = []

    with pytest.raises(ProvenanceInputError, match="entries must be non-empty"):
        build_provenance_merkle_payload(
            provenance_manifest,
            provenance_manifest_schema={},
            provenance_merkle_schema={},
        )


def test_phase4e_golden_provenance_merkle_matches_expected(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    out_path = tmp_path / "provenance_merkle.tp.meta.provenance_merkle.v1.json"
    result = _run_merkle_cli(input_path=GOLDEN_PROVENANCE_MANIFEST, out_path=out_path)
    assert result.returncode == 0, result.stderr
    assert out_path.read_bytes() == GOLDEN_PROVENANCE_MERKLE.read_bytes()


def test_phase4e_provenance_merkle_generation_is_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    out_a = tmp_path / "provenance_merkle_a.json"
    out_b = tmp_path / "provenance_merkle_b.json"

    first = _run_merkle_cli(input_path=GOLDEN_PROVENANCE_MANIFEST, out_path=out_a)
    assert first.returncode == 0, first.stderr

    second = _run_merkle_cli(input_path=GOLDEN_PROVENANCE_MANIFEST, out_path=out_b)
    assert second.returncode == 0, second.stderr

    assert out_a.read_bytes() == out_b.read_bytes()


def test_phase4e_full_pipeline_is_deterministic_end_to_end(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    fake_exiftool = _build_fake_exiftool(tmp_path)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PATH"] = f"{fake_exiftool.parent}:{env.get('PATH', '')}"

    capture_a = tmp_path / "capture_a.json"
    metadata_manifest_a = tmp_path / "metadata_manifest_a.json"
    provenance_manifest_a = tmp_path / "provenance_manifest_a.json"
    provenance_merkle_a = tmp_path / "provenance_merkle_a.json"

    capture_b = tmp_path / "capture_b.json"
    metadata_manifest_b = tmp_path / "metadata_manifest_b.json"
    provenance_manifest_b = tmp_path / "provenance_manifest_b.json"
    provenance_merkle_b = tmp_path / "provenance_merkle_b.json"

    result_extract_a = _run_command(
        [
            sys.executable,
            str(EXTRACT_TOOL),
            "--input-root",
            str(FIXTURE_ROOT),
            "--out",
            str(capture_a),
        ],
        env=env,
    )
    assert result_extract_a.returncode == 0, result_extract_a.stderr

    result_manifest_a = _run_command(
        [
            sys.executable,
            str(METADATA_MANIFEST_TOOL),
            "--input",
            str(capture_a),
            "--out",
            str(metadata_manifest_a),
        ],
        env=env,
    )
    assert result_manifest_a.returncode == 0, result_manifest_a.stderr

    result_provenance_manifest_a = _run_command(
        [
            sys.executable,
            str(PROVENANCE_MANIFEST_TOOL),
            "--capture-metadata",
            str(capture_a),
            "--metadata-manifest",
            str(metadata_manifest_a),
            "--out",
            str(provenance_manifest_a),
        ],
        env=env,
    )
    assert result_provenance_manifest_a.returncode == 0, result_provenance_manifest_a.stderr

    result_provenance_merkle_a = _run_command(
        [
            sys.executable,
            str(PROVENANCE_MERKLE_TOOL),
            "--input",
            str(provenance_manifest_a),
            "--out",
            str(provenance_merkle_a),
        ],
        env=env,
    )
    assert result_provenance_merkle_a.returncode == 0, result_provenance_merkle_a.stderr

    result_extract_b = _run_command(
        [
            sys.executable,
            str(EXTRACT_TOOL),
            "--input-root",
            str(FIXTURE_ROOT),
            "--out",
            str(capture_b),
        ],
        env=env,
    )
    assert result_extract_b.returncode == 0, result_extract_b.stderr

    result_manifest_b = _run_command(
        [
            sys.executable,
            str(METADATA_MANIFEST_TOOL),
            "--input",
            str(capture_b),
            "--out",
            str(metadata_manifest_b),
        ],
        env=env,
    )
    assert result_manifest_b.returncode == 0, result_manifest_b.stderr

    result_provenance_manifest_b = _run_command(
        [
            sys.executable,
            str(PROVENANCE_MANIFEST_TOOL),
            "--capture-metadata",
            str(capture_b),
            "--metadata-manifest",
            str(metadata_manifest_b),
            "--out",
            str(provenance_manifest_b),
        ],
        env=env,
    )
    assert result_provenance_manifest_b.returncode == 0, result_provenance_manifest_b.stderr

    result_provenance_merkle_b = _run_command(
        [
            sys.executable,
            str(PROVENANCE_MERKLE_TOOL),
            "--input",
            str(provenance_manifest_b),
            "--out",
            str(provenance_merkle_b),
        ],
        env=env,
    )
    assert result_provenance_merkle_b.returncode == 0, result_provenance_merkle_b.stderr

    assert provenance_manifest_a.read_bytes() == provenance_manifest_b.read_bytes()
    assert provenance_merkle_a.read_bytes() == provenance_merkle_b.read_bytes()


def test_phase4e_merkle_cli_fails_on_unsorted_input_when_strict(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _build_two_entry_provenance_manifest()
    payload["entries"] = list(reversed(payload["entries"]))
    input_path = tmp_path / "provenance_manifest_unsorted.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "provenance_merkle.json"
    result = _run_merkle_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 3
    assert "provenance manifest entries must be sorted by relative_path" in result.stderr


def test_phase4e_merkle_cli_can_relax_order_and_still_emit_canonical_result(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _build_two_entry_provenance_manifest()
    payload["entries"] = list(reversed(payload["entries"]))
    input_path = tmp_path / "provenance_manifest_unsorted.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "provenance_merkle.json"
    result = _run_merkle_cli(input_path=input_path, out_path=out_path, strict_input_order=False)
    assert result.returncode == 0, result.stderr
    output = _load_json(out_path)
    assert output["leaf_count"] == 2
    assert out_path.read_bytes().endswith(b"\n")


def test_phase4e_merkle_cli_fails_on_duplicate_relative_path(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _build_two_entry_provenance_manifest()
    payload["entries"] = [payload["entries"][0], _deepcopy(payload["entries"][0])]
    input_path = tmp_path / "provenance_manifest_duplicate.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "provenance_merkle.json"
    result = _run_merkle_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 3
    assert "provenance manifest duplicate relative_path" in result.stderr


def test_phase4e_merkle_cli_fails_schema_validation_on_corrupt_leaf_hex(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _build_two_entry_provenance_manifest()
    payload["entries"][0]["provenance_entry_sha256"] = "z" * 64
    input_path = tmp_path / "provenance_manifest_invalid_hex.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "provenance_merkle.json"
    result = _run_merkle_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 4
    assert "Schema validation failure:" in result.stderr


def test_phase4e_merkle_output_changes_when_leaf_is_tampered(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    payload = _load_json(GOLDEN_PROVENANCE_MANIFEST)
    payload["entries"][0]["provenance_entry_sha256"] = "f" * 64
    input_path = tmp_path / "provenance_manifest_tampered_leaf.json"
    input_path.write_text(json.dumps(payload), encoding="utf-8")

    out_path = tmp_path / "provenance_merkle_tampered.json"
    result = _run_merkle_cli(input_path=input_path, out_path=out_path, strict_input_order=False)
    assert result.returncode == 0, result.stderr

    tampered_payload = _load_json(out_path)
    golden_payload = _load_json(GOLDEN_PROVENANCE_MERKLE)
    assert tampered_payload["provenance_merkle_root"] != golden_payload["provenance_merkle_root"]


def test_phase4e_merkle_cli_returns_exit_code_2_for_invalid_input_json(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.json"
    input_path.write_text("{invalid-json", encoding="utf-8")

    out_path = tmp_path / "provenance_merkle.json"
    result = _run_merkle_cli(input_path=input_path, out_path=out_path)
    assert result.returncode == 2
    assert "Input read/parse error:" in result.stderr
