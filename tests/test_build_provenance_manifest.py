"""Tests for Phase 4E provenance manifest builder."""

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
    METADATA_MANIFEST_CONTRACT_VERSION,
    compute_metadata_sha256,
)
from tp.phase4.provenance_capture import (
    PROVENANCE_CONTRACT_VERSION,
    ProvenanceInputError,
    ProvenanceSchemaValidationError,
    build_provenance_manifest_payload,
    compute_provenance_entry_sha256,
    serialize_provenance_manifest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_MANIFEST_TOOL = PROJECT_ROOT / "tools" / "build_provenance_manifest.py"
GOLDEN_CAPTURE = PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_capture_metadata.tp.meta.capture.v1.json"
GOLDEN_METADATA_MANIFEST = (
    PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_metadata_manifest.tp.meta.capture_manifest.v1.json"
)
GOLDEN_PROVENANCE_MANIFEST = (
    PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_provenance_manifest.tp.meta.provenance.v1.json"
)

pytestmark = [pytest.mark.regression, pytest.mark.golden]


def _run_provenance_manifest_cli(
    *,
    capture_metadata_path: Path,
    metadata_manifest_path: Path,
    out_path: Path,
    strict_input_order: bool | None = None,
    require_fingerprint_match: bool | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    command = [
        sys.executable,
        str(PROVENANCE_MANIFEST_TOOL),
        "--capture-metadata",
        str(capture_metadata_path),
        "--metadata-manifest",
        str(metadata_manifest_path),
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _deepcopy(obj: Any) -> Any:
    return json.loads(json.dumps(obj))


def _build_two_record_artifacts() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_record = _load_json(GOLDEN_CAPTURE)[0]
    record_a = _deepcopy(base_record)
    record_a["relative_path"] = "a/sample_01.dng"

    record_b = _deepcopy(base_record)
    record_b["relative_path"] = "b/sample_01.dng"

    manifest_payload = {
        "metadata_manifest_contract_version": METADATA_MANIFEST_CONTRACT_VERSION,
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "entries": [
            {
                "relative_path": record_a["relative_path"],
                "file_sha256": record_a["file_sha256"],
                "metadata_sha256": compute_metadata_sha256(record_a),
            },
            {
                "relative_path": record_b["relative_path"],
                "file_sha256": record_b["file_sha256"],
                "metadata_sha256": compute_metadata_sha256(record_b),
            },
        ],
    }
    return [record_a, record_b], manifest_payload


def test_phase4e_build_manifest_direct_success_with_fingerprint() -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    fingerprint = records[0]["extractor"]["config_fingerprint_sha256"]

    payload = build_provenance_manifest_payload(
        records,
        manifest_payload,
        metadata_schema={},
        metadata_manifest_schema={},
        provenance_manifest_schema={},
        required_config_fingerprint_sha256=fingerprint,
    )

    assert payload["provenance_contract_version"] == PROVENANCE_CONTRACT_VERSION
    assert payload["metadata_contract_version"] == METADATA_CONTRACT_VERSION
    assert [entry["relative_path"] for entry in payload["entries"]] == ["a/sample_01.dng", "b/sample_01.dng"]
    assert serialize_provenance_manifest(payload).endswith(b"\n")


def test_phase4e_golden_provenance_manifest_matches_expected(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    out_path = tmp_path / "provenance_manifest.tp.meta.provenance.v1.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=GOLDEN_CAPTURE,
        metadata_manifest_path=GOLDEN_METADATA_MANIFEST,
        out_path=out_path,
    )
    assert result.returncode == 0, result.stderr
    assert out_path.read_bytes() == GOLDEN_PROVENANCE_MANIFEST.read_bytes()


def test_phase4e_provenance_manifest_generation_is_deterministic(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    out_a = tmp_path / "provenance_a.json"
    out_b = tmp_path / "provenance_b.json"

    first = _run_provenance_manifest_cli(
        capture_metadata_path=GOLDEN_CAPTURE,
        metadata_manifest_path=GOLDEN_METADATA_MANIFEST,
        out_path=out_a,
    )
    assert first.returncode == 0, first.stderr

    second = _run_provenance_manifest_cli(
        capture_metadata_path=GOLDEN_CAPTURE,
        metadata_manifest_path=GOLDEN_METADATA_MANIFEST,
        out_path=out_b,
    )
    assert second.returncode == 0, second.stderr

    assert out_a.read_bytes() == out_b.read_bytes()


def test_phase4e_provenance_entry_hash_binds_contract_versions() -> None:
    manifest_entry = _load_json(GOLDEN_METADATA_MANIFEST)["entries"][0]
    expected_entry = _load_json(GOLDEN_PROVENANCE_MANIFEST)["entries"][0]
    digest = compute_provenance_entry_sha256(
        file_sha256=manifest_entry["file_sha256"],
        metadata_sha256=manifest_entry["metadata_sha256"],
        capture_contract_version=METADATA_CONTRACT_VERSION,
        metadata_contract_version=METADATA_CONTRACT_VERSION,
        provenance_contract_version=PROVENANCE_CONTRACT_VERSION,
    )
    assert digest == expected_entry["provenance_entry_sha256"]


def test_phase4e_provenance_entry_hash_rejects_contract_version_mismatch() -> None:
    manifest_entry = _load_json(GOLDEN_METADATA_MANIFEST)["entries"][0]
    with pytest.raises(ProvenanceInputError, match="provenance_contract_version mismatch"):
        compute_provenance_entry_sha256(
            file_sha256=manifest_entry["file_sha256"],
            metadata_sha256=manifest_entry["metadata_sha256"],
            provenance_contract_version="tp.meta.provenance.v2",
        )


def test_phase4e_provenance_entry_hash_rejects_invalid_sha256() -> None:
    manifest_entry = _load_json(GOLDEN_METADATA_MANIFEST)["entries"][0]
    with pytest.raises(ProvenanceInputError, match="file_sha256"):
        compute_provenance_entry_sha256(
            file_sha256="not-a-sha256",
            metadata_sha256=manifest_entry["metadata_sha256"],
        )


def test_phase4e_build_manifest_reports_missing_relative_path_without_keyerror() -> None:
    pytest.importorskip("jsonschema")
    capture_records = [
        {
            "metadata_contract_version": METADATA_CONTRACT_VERSION,
            "file_sha256": "a" * 64,
        }
    ]
    metadata_manifest_payload = {
        "metadata_manifest_contract_version": METADATA_MANIFEST_CONTRACT_VERSION,
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "entries": [],
    }

    with pytest.raises(ProvenanceInputError, match=r"capture metadata record\[0\] missing relative_path"):
        build_provenance_manifest_payload(
            capture_records,
            metadata_manifest_payload,
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
        )


def test_phase4e_build_manifest_rejects_non_object_metadata_manifest() -> None:
    pytest.importorskip("jsonschema")
    records, _manifest_payload = _build_two_record_artifacts()

    with pytest.raises(ProvenanceInputError, match="metadata manifest payload must be a JSON object"):
        build_provenance_manifest_payload(
            records,
            [],
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
        )


def test_phase4e_build_manifest_rejects_capture_contract_mismatch() -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    records[0]["metadata_contract_version"] = "tp.meta.capture.v999"

    with pytest.raises(ProvenanceInputError, match="capture record\\[0\\] contract mismatch"):
        build_provenance_manifest_payload(
            records,
            manifest_payload,
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
        )


def test_phase4e_build_manifest_rejects_metadata_manifest_contract_mismatch() -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    manifest_payload["metadata_manifest_contract_version"] = "tp.meta.capture_manifest.v999"

    with pytest.raises(ProvenanceInputError, match="metadata manifest contract mismatch"):
        build_provenance_manifest_payload(
            records,
            manifest_payload,
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
        )


def test_phase4e_build_manifest_rejects_metadata_manifest_entries_that_are_not_array() -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    manifest_payload["entries"] = {"not": "an array"}

    with pytest.raises(ProvenanceInputError, match="metadata manifest entries must be an array"):
        build_provenance_manifest_payload(
            records,
            manifest_payload,
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
        )


def test_phase4e_build_manifest_rejects_missing_extractor_when_fingerprint_required() -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    del records[0]["extractor"]

    with pytest.raises(ProvenanceInputError, match="missing extractor object"):
        build_provenance_manifest_payload(
            records,
            manifest_payload,
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
            required_config_fingerprint_sha256="0" * 64,
        )


def test_phase4e_build_manifest_rejects_alignment_mismatch_directly() -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    manifest_payload["entries"] = manifest_payload["entries"][:1]

    with pytest.raises(ProvenanceInputError, match="relative_path alignment mismatch"):
        build_provenance_manifest_payload(
            records,
            manifest_payload,
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
        )


def test_phase4e_build_manifest_rejects_file_sha256_mismatch_directly() -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    manifest_payload["entries"][0]["file_sha256"] = "f" * 64

    with pytest.raises(ProvenanceInputError, match="file_sha256 mismatch"):
        build_provenance_manifest_payload(
            records,
            manifest_payload,
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
        )


def test_phase4e_build_manifest_rejects_metadata_sha256_mismatch_directly() -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    manifest_payload["entries"][0]["metadata_sha256"] = "f" * 64

    with pytest.raises(ProvenanceInputError, match="metadata_sha256 mismatch"):
        build_provenance_manifest_payload(
            records,
            manifest_payload,
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
        )


def test_phase4e_build_manifest_wraps_canonical_serialization_errors() -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    records[0]["non_jsonable"] = object()

    with pytest.raises(ProvenanceSchemaValidationError, match="canonical metadata serialization failed"):
        build_provenance_manifest_payload(
            records,
            manifest_payload,
            metadata_schema={},
            metadata_manifest_schema={},
            provenance_manifest_schema={},
        )


def test_phase4e_cli_fails_on_unsorted_capture_when_strict(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    capture_payload = [records[1], records[0]]

    capture_path = tmp_path / "capture_unsorted.json"
    manifest_path = tmp_path / "metadata_manifest.json"
    capture_path.write_text(json.dumps(capture_payload), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 3
    assert "capture metadata array must be sorted by relative_path" in result.stderr


def test_phase4e_cli_fails_on_unsorted_metadata_manifest_when_strict(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    manifest_payload["entries"] = list(reversed(manifest_payload["entries"]))

    capture_path = tmp_path / "capture.json"
    manifest_path = tmp_path / "metadata_manifest_unsorted.json"
    capture_path.write_text(json.dumps(records), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 3
    assert "metadata manifest entries must be sorted by relative_path" in result.stderr


def test_phase4e_cli_can_relax_order_and_sort_output(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    relaxed_manifest_payload = _deepcopy(manifest_payload)
    relaxed_manifest_payload["entries"] = list(reversed(relaxed_manifest_payload["entries"]))
    capture_path = tmp_path / "capture_unsorted.json"
    manifest_path = tmp_path / "metadata_manifest_unsorted.json"
    capture_path.write_text(json.dumps([records[1], records[0]]), encoding="utf-8")
    manifest_path.write_text(json.dumps(relaxed_manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
        strict_input_order=False,
    )
    assert result.returncode == 0, result.stderr
    payload = _load_json(out_path)
    assert [entry["relative_path"] for entry in payload["entries"]] == ["a/sample_01.dng", "b/sample_01.dng"]
    assert out_path.read_bytes().endswith(b"\n")


def test_phase4e_cli_fails_on_duplicate_relative_path_in_capture(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    duplicate_capture = [records[0], _deepcopy(records[0])]

    capture_path = tmp_path / "capture_duplicate.json"
    manifest_path = tmp_path / "metadata_manifest.json"
    capture_path.write_text(json.dumps(duplicate_capture), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 3
    assert "capture metadata duplicate relative_path" in result.stderr


def test_phase4e_cli_fails_on_duplicate_relative_path_in_metadata_manifest(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    manifest_payload["entries"] = [manifest_payload["entries"][0], _deepcopy(manifest_payload["entries"][0])]

    capture_path = tmp_path / "capture.json"
    manifest_path = tmp_path / "metadata_manifest_duplicate.json"
    capture_path.write_text(json.dumps(records), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 3
    assert "metadata manifest duplicate relative_path" in result.stderr


def test_phase4e_cli_fails_on_relative_path_alignment_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    manifest_payload["entries"] = [manifest_payload["entries"][0]]

    capture_path = tmp_path / "capture.json"
    manifest_path = tmp_path / "metadata_manifest_missing_entry.json"
    capture_path.write_text(json.dumps(records), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 3
    assert "relative_path alignment mismatch" in result.stderr


def test_phase4e_cli_fails_on_file_sha256_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    records[0]["file_sha256"] = "f" * 64

    capture_path = tmp_path / "capture_tampered_file_sha.json"
    manifest_path = tmp_path / "metadata_manifest.json"
    capture_path.write_text(json.dumps(records), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 3
    assert "file_sha256 mismatch" in result.stderr


def test_phase4e_cli_fails_on_metadata_sha256_mismatch(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    records[0]["camera_make"] = "Tampered"

    capture_path = tmp_path / "capture_tampered_metadata.json"
    manifest_path = tmp_path / "metadata_manifest.json"
    capture_path.write_text(json.dumps(records), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 3
    assert "metadata_sha256 mismatch" in result.stderr


def test_phase4e_cli_fails_on_fingerprint_mismatch_by_default(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    records[0]["extractor"]["config_fingerprint_sha256"] = "0" * 64
    manifest_payload["entries"][0]["metadata_sha256"] = compute_metadata_sha256(records[0])

    capture_path = tmp_path / "capture_fingerprint_mismatch.json"
    manifest_path = tmp_path / "metadata_manifest.json"
    capture_path.write_text(json.dumps(records), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 3
    assert "fingerprint mismatch" in result.stderr


def test_phase4e_cli_can_disable_fingerprint_match(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    records[0]["extractor"]["config_fingerprint_sha256"] = "0" * 64
    manifest_payload["entries"][0]["metadata_sha256"] = compute_metadata_sha256(records[0])

    capture_path = tmp_path / "capture_fingerprint_mismatch.json"
    manifest_path = tmp_path / "metadata_manifest.json"
    capture_path.write_text(json.dumps(records), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
        require_fingerprint_match=False,
    )
    assert result.returncode == 0, result.stderr


def test_phase4e_cli_fails_schema_validation_on_corrupt_hex_digest(tmp_path: Path) -> None:
    pytest.importorskip("jsonschema")
    records, manifest_payload = _build_two_record_artifacts()
    manifest_payload["entries"][0]["metadata_sha256"] = "z" * 64

    capture_path = tmp_path / "capture.json"
    manifest_path = tmp_path / "metadata_manifest_invalid_hex.json"
    capture_path.write_text(json.dumps(records), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 4
    assert "Schema validation failure:" in result.stderr


def test_phase4e_cli_returns_exit_code_2_for_invalid_input_json(tmp_path: Path) -> None:
    capture_path = tmp_path / "capture_invalid.json"
    capture_path.write_text("{invalid-json", encoding="utf-8")
    manifest_path = tmp_path / "metadata_manifest.json"
    manifest_path.write_text(GOLDEN_METADATA_MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")

    out_path = tmp_path / "provenance_manifest.json"
    result = _run_provenance_manifest_cli(
        capture_metadata_path=capture_path,
        metadata_manifest_path=manifest_path,
        out_path=out_path,
    )
    assert result.returncode == 2
    assert "Input read/parse error:" in result.stderr
