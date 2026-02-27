"""Tests for Phase 4F external chain verifier and deterministic report emission."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from tp.crypto.merkle import merkle_root_sha256
from tp.phase4.hash_capture_metadata import (
    METADATA_CONTRACT_VERSION,
    METADATA_MANIFEST_CONTRACT_VERSION,
    compute_metadata_sha256,
)
from tp.phase4.provenance_capture import (
    PROVENANCE_CONTRACT_VERSION,
    PROVENANCE_MERKLE_CONTRACT_VERSION,
    compute_provenance_entry_sha256,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERIFY_TOOL = PROJECT_ROOT / "tools" / "verify_phase4_chain.py"
REPORT_SCHEMA_PATH = PROJECT_ROOT / "schemas" / "phase4" / "verification_report.schema.json"

GOLDEN_CAPTURE = PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_capture_metadata.tp.meta.capture.v1.json"
GOLDEN_METADATA_MANIFEST = (
    PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_metadata_manifest.tp.meta.capture_manifest.v1.json"
)
GOLDEN_PROVENANCE_MANIFEST = (
    PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_provenance_manifest.tp.meta.provenance.v1.json"
)
GOLDEN_PROVENANCE_MERKLE = (
    PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_provenance_merkle.tp.meta.provenance_merkle.v1.json"
)
GOLDEN_VERIFICATION_REPORT = (
    PROJECT_ROOT / "tests" / "golden" / "phase4" / "expected_verification_report.tp.meta.verification_report.v1.json"
)

pytestmark = [pytest.mark.regression, pytest.mark.golden]


def _run_verify_cli(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, str(VERIFY_TOOL), *args]
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def _golden_args() -> list[str]:
    return [
        "--capture-metadata",
        str(GOLDEN_CAPTURE),
        "--metadata-manifest",
        str(GOLDEN_METADATA_MANIFEST),
        "--provenance-manifest",
        str(GOLDEN_PROVENANCE_MANIFEST),
        "--provenance-merkle",
        str(GOLDEN_PROVENANCE_MERKLE),
    ]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_two_record_chain_payloads() -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any]]:
    base_record = _load_json(GOLDEN_CAPTURE)[0]
    record_a = json.loads(json.dumps(base_record))
    record_b = json.loads(json.dumps(base_record))
    record_a["relative_path"] = "a/sample_01.dng"
    record_b["relative_path"] = "b/sample_01.dng"

    metadata_sha_a = compute_metadata_sha256(record_a)
    metadata_sha_b = compute_metadata_sha256(record_b)

    metadata_manifest = {
        "metadata_manifest_contract_version": METADATA_MANIFEST_CONTRACT_VERSION,
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "entries": [
            {
                "relative_path": "a/sample_01.dng",
                "file_sha256": record_a["file_sha256"],
                "metadata_sha256": metadata_sha_a,
            },
            {
                "relative_path": "b/sample_01.dng",
                "file_sha256": record_b["file_sha256"],
                "metadata_sha256": metadata_sha_b,
            },
        ],
    }

    provenance_entry_sha_a = compute_provenance_entry_sha256(
        file_sha256=record_a["file_sha256"],
        metadata_sha256=metadata_sha_a,
        capture_contract_version=METADATA_CONTRACT_VERSION,
        metadata_contract_version=METADATA_CONTRACT_VERSION,
        provenance_contract_version=PROVENANCE_CONTRACT_VERSION,
    )
    provenance_entry_sha_b = compute_provenance_entry_sha256(
        file_sha256=record_b["file_sha256"],
        metadata_sha256=metadata_sha_b,
        capture_contract_version=METADATA_CONTRACT_VERSION,
        metadata_contract_version=METADATA_CONTRACT_VERSION,
        provenance_contract_version=PROVENANCE_CONTRACT_VERSION,
    )
    provenance_manifest = {
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "metadata_contract_version": METADATA_CONTRACT_VERSION,
        "entries": [
            {
                "relative_path": "a/sample_01.dng",
                "file_sha256": record_a["file_sha256"],
                "metadata_sha256": metadata_sha_a,
                "provenance_entry_sha256": provenance_entry_sha_a,
            },
            {
                "relative_path": "b/sample_01.dng",
                "file_sha256": record_b["file_sha256"],
                "metadata_sha256": metadata_sha_b,
                "provenance_entry_sha256": provenance_entry_sha_b,
            },
        ],
    }
    merkle_root = merkle_root_sha256([bytes.fromhex(provenance_entry_sha_a), bytes.fromhex(provenance_entry_sha_b)])
    provenance_merkle = {
        "provenance_merkle_contract_version": PROVENANCE_MERKLE_CONTRACT_VERSION,
        "provenance_contract_version": PROVENANCE_CONTRACT_VERSION,
        "leaf_count": 2,
        "provenance_merkle_root": merkle_root,
    }
    return [record_a, record_b], metadata_manifest, provenance_manifest, provenance_merkle


def test_phase4f_cli_help_works_without_pythonpath() -> None:
    result = subprocess.run(
        [sys.executable, str(VERIFY_TOOL), "--help"],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": os.environ.get("PATH", ""), "PYTHONPATH": ""},
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_phase4f_golden_artifacts_verify_successfully() -> None:
    result = _run_verify_cli(*_golden_args())
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""


def test_phase4f_success_report_matches_expected_golden(tmp_path: Path) -> None:
    report_path = tmp_path / "verification_report.tp.meta.verification_report.v1.json"
    result = _run_verify_cli(*_golden_args(), "--out-report", str(report_path))
    assert result.returncode == 0, result.stderr
    assert report_path.read_bytes() == GOLDEN_VERIFICATION_REPORT.read_bytes()


def test_phase4f_report_bytes_are_deterministic_across_runs(tmp_path: Path) -> None:
    report_a = tmp_path / "report_a.json"
    report_b = tmp_path / "report_b.json"

    first = _run_verify_cli(*_golden_args(), "--out-report", str(report_a))
    second = _run_verify_cli(*_golden_args(), "--out-report", str(report_b))

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert report_a.read_bytes() == report_b.read_bytes()


def test_phase4f_report_validates_against_schema(tmp_path: Path) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    report_path = tmp_path / "verification_report.json"
    result = _run_verify_cli(*_golden_args(), "--out-report", str(report_path))
    assert result.returncode == 0, result.stderr

    schema = _load_json(REPORT_SCHEMA_PATH)
    payload = _load_json(report_path)
    jsonschema.Draft202012Validator(schema).validate(payload)


def test_phase4f_report_has_no_nondeterministic_fields(tmp_path: Path) -> None:
    report_path = tmp_path / "verification_report.json"
    result = _run_verify_cli(*_golden_args(), "--out-report", str(report_path))
    assert result.returncode == 0, result.stderr
    payload = _load_json(report_path)

    forbidden_keys = {"timestamp", "generated_at", "host", "hostname", "cwd", "absolute_path"}

    stack = [payload]
    while stack:
        current = stack.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                assert key not in forbidden_keys
                stack.append(value)
        elif isinstance(current, list):
            stack.extend(current)


def test_phase4f_cli_returns_exit_code_31_for_invalid_input_json(tmp_path: Path) -> None:
    bad_capture = tmp_path / "capture_invalid.json"
    bad_capture.write_text("{invalid-json", encoding="utf-8")

    result = _run_verify_cli(
        "--capture-metadata",
        str(bad_capture),
        "--metadata-manifest",
        str(GOLDEN_METADATA_MANIFEST),
        "--provenance-manifest",
        str(GOLDEN_PROVENANCE_MANIFEST),
        "--provenance-merkle",
        str(GOLDEN_PROVENANCE_MERKLE),
    )
    assert result.returncode == 31
    assert "Malformed input:" in result.stderr


def test_phase4f_cli_returns_exit_code_32_for_schema_failure(tmp_path: Path) -> None:
    bad_capture = tmp_path / "capture_schema_invalid.json"
    capture_payload = _load_json(GOLDEN_CAPTURE)
    del capture_payload[0]["camera_model"]
    _write_json(bad_capture, capture_payload)

    result = _run_verify_cli(
        "--capture-metadata",
        str(bad_capture),
        "--metadata-manifest",
        str(GOLDEN_METADATA_MANIFEST),
        "--provenance-manifest",
        str(GOLDEN_PROVENANCE_MANIFEST),
        "--provenance-merkle",
        str(GOLDEN_PROVENANCE_MERKLE),
    )
    assert result.returncode == 32
    assert "Schema validation failure:" in result.stderr


def test_phase4f_cli_returns_exit_code_33_for_alignment_mismatch(tmp_path: Path) -> None:
    bad_manifest = tmp_path / "metadata_manifest_bad_path.json"
    manifest_payload = _load_json(GOLDEN_METADATA_MANIFEST)
    manifest_payload["entries"][0]["relative_path"] = "tampered_path/sample_01.dng"
    _write_json(bad_manifest, manifest_payload)

    result = _run_verify_cli(
        "--capture-metadata",
        str(GOLDEN_CAPTURE),
        "--metadata-manifest",
        str(bad_manifest),
        "--provenance-manifest",
        str(GOLDEN_PROVENANCE_MANIFEST),
        "--provenance-merkle",
        str(GOLDEN_PROVENANCE_MERKLE),
    )
    assert result.returncode == 33
    assert "Alignment failure:" in result.stderr


def test_phase4f_cli_returns_exit_code_33_for_strict_order_violation(tmp_path: Path) -> None:
    capture_payload, metadata_manifest_payload, provenance_manifest_payload, provenance_merkle_payload = (
        _build_two_record_chain_payloads()
    )
    unsorted_capture_payload = [capture_payload[1], capture_payload[0]]

    capture_path = tmp_path / "capture_unsorted.json"
    metadata_manifest_path = tmp_path / "metadata_manifest.json"
    provenance_manifest_path = tmp_path / "provenance_manifest.json"
    provenance_merkle_path = tmp_path / "provenance_merkle.json"
    _write_json(capture_path, unsorted_capture_payload)
    _write_json(metadata_manifest_path, metadata_manifest_payload)
    _write_json(provenance_manifest_path, provenance_manifest_payload)
    _write_json(provenance_merkle_path, provenance_merkle_payload)

    result = _run_verify_cli(
        "--capture-metadata",
        str(capture_path),
        "--metadata-manifest",
        str(metadata_manifest_path),
        "--provenance-manifest",
        str(provenance_manifest_path),
        "--provenance-merkle",
        str(provenance_merkle_path),
    )
    assert result.returncode == 33
    assert "capture metadata array must be sorted by relative_path" in result.stderr


def test_phase4f_cli_returns_exit_code_34_for_metadata_hash_mismatch(tmp_path: Path) -> None:
    bad_manifest = tmp_path / "metadata_manifest_bad_hash.json"
    bad_provenance = tmp_path / "provenance_manifest_bad_hash.json"
    manifest_payload = _load_json(GOLDEN_METADATA_MANIFEST)
    provenance_payload = _load_json(GOLDEN_PROVENANCE_MANIFEST)
    manifest_payload["entries"][0]["metadata_sha256"] = "0" * 64
    provenance_payload["entries"][0]["metadata_sha256"] = "0" * 64
    _write_json(bad_manifest, manifest_payload)
    _write_json(bad_provenance, provenance_payload)

    result = _run_verify_cli(
        "--capture-metadata",
        str(GOLDEN_CAPTURE),
        "--metadata-manifest",
        str(bad_manifest),
        "--provenance-manifest",
        str(bad_provenance),
        "--provenance-merkle",
        str(GOLDEN_PROVENANCE_MERKLE),
    )
    assert result.returncode == 34
    assert "Metadata hash mismatch:" in result.stderr


def test_phase4f_cli_returns_exit_code_35_for_provenance_entry_mismatch(tmp_path: Path) -> None:
    bad_provenance = tmp_path / "provenance_manifest_bad_entry_hash.json"
    provenance_payload = _load_json(GOLDEN_PROVENANCE_MANIFEST)
    provenance_payload["entries"][0]["provenance_entry_sha256"] = "f" * 64
    _write_json(bad_provenance, provenance_payload)

    result = _run_verify_cli(
        "--capture-metadata",
        str(GOLDEN_CAPTURE),
        "--metadata-manifest",
        str(GOLDEN_METADATA_MANIFEST),
        "--provenance-manifest",
        str(bad_provenance),
        "--provenance-merkle",
        str(GOLDEN_PROVENANCE_MERKLE),
    )
    assert result.returncode == 35
    assert "Provenance entry hash mismatch:" in result.stderr


def test_phase4f_cli_returns_exit_code_36_for_merkle_mismatch(tmp_path: Path) -> None:
    bad_merkle = tmp_path / "provenance_merkle_bad_root.json"
    merkle_payload = _load_json(GOLDEN_PROVENANCE_MERKLE)
    merkle_payload["provenance_merkle_root"] = "f" * 64
    _write_json(bad_merkle, merkle_payload)

    result = _run_verify_cli(
        "--capture-metadata",
        str(GOLDEN_CAPTURE),
        "--metadata-manifest",
        str(GOLDEN_METADATA_MANIFEST),
        "--provenance-manifest",
        str(GOLDEN_PROVENANCE_MANIFEST),
        "--provenance-merkle",
        str(bad_merkle),
    )
    assert result.returncode == 36
    assert "Merkle mismatch:" in result.stderr


def test_phase4f_cli_returns_exit_code_37_for_report_write_failure(tmp_path: Path) -> None:
    parent_file = tmp_path / "not_a_directory"
    parent_file.write_text("content", encoding="utf-8")
    bad_report_path = parent_file / "report.json"

    result = _run_verify_cli(*_golden_args(), "--out-report", str(bad_report_path))
    assert result.returncode == 37
    assert "Report write failure:" in result.stderr


def test_phase4f_failure_does_not_emit_report_by_default(tmp_path: Path) -> None:
    bad_merkle = tmp_path / "provenance_merkle_bad_root.json"
    merkle_payload = _load_json(GOLDEN_PROVENANCE_MERKLE)
    merkle_payload["provenance_merkle_root"] = "1" * 64
    _write_json(bad_merkle, merkle_payload)

    report_path = tmp_path / "failure_report.json"
    result = _run_verify_cli(
        "--capture-metadata",
        str(GOLDEN_CAPTURE),
        "--metadata-manifest",
        str(GOLDEN_METADATA_MANIFEST),
        "--provenance-manifest",
        str(GOLDEN_PROVENANCE_MANIFEST),
        "--provenance-merkle",
        str(bad_merkle),
        "--out-report",
        str(report_path),
    )
    assert result.returncode == 36
    assert not report_path.exists()


def test_phase4f_failure_can_emit_report_with_opt_in_flag(tmp_path: Path) -> None:
    bad_merkle = tmp_path / "provenance_merkle_bad_root.json"
    merkle_payload = _load_json(GOLDEN_PROVENANCE_MERKLE)
    merkle_payload["provenance_merkle_root"] = "2" * 64
    _write_json(bad_merkle, merkle_payload)

    report_path = tmp_path / "failure_report.json"
    result = _run_verify_cli(
        "--capture-metadata",
        str(GOLDEN_CAPTURE),
        "--metadata-manifest",
        str(GOLDEN_METADATA_MANIFEST),
        "--provenance-manifest",
        str(GOLDEN_PROVENANCE_MANIFEST),
        "--provenance-merkle",
        str(bad_merkle),
        "--out-report",
        str(report_path),
        "--write-report-on-failure",
    )
    assert result.returncode == 36
    assert report_path.exists()

    report_payload = _load_json(report_path)
    assert report_payload["verification_status"]["passed"] is False
    assert report_payload["verification_status"]["failure_code_label"] == "MERKLE_MISMATCH"


def test_phase4f_verifier_import_and_help_do_not_require_ml_stack() -> None:
    script = f"""
import runpy
import sys

class _Blocker:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in {{"torch", "numpy", "transformers"}}:
            raise ImportError(f"blocked heavy import: {{fullname}}")
        return None

sys.meta_path.insert(0, _Blocker())
import tp.phase4.verify_phase4_chain  # noqa: F401
sys.argv = ["verify_phase4_chain.py", "--help"]
try:
    runpy.run_path({str(VERIFY_TOOL)!r}, run_name="__main__")
except SystemExit as exc:
    raise SystemExit(int(exc.code))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
