"""Integration tests for tp.archive.machine.v1 orchestration CLI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from .schema_utils import validate_archive_machine_payload

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "archive_governance.py"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "archive_small"
FIXTURE_INDEX = FIXTURE_DIR / "archive_index_normalized.csv.gz"
FIXTURE_HASH_MANIFEST = FIXTURE_DIR / "golden" / "hash_manifest.csv.gz"
FIXTURE_ARCHIVE_ROOT = FIXTURE_DIR / "archive_root"
RIGHTS_POLICY = PROJECT_ROOT / "policy" / "archive" / "rights_flags.yml"


def _run_governance_cli(
    *args: str,
    json_output: Path | None = None,
    canonical_profile: str | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(TOOL_PATH), "--json"]
    if canonical_profile is not None:
        command.extend(["--json-canonical-profile", canonical_profile])
    if json_output is not None:
        command.extend(["--json-output", str(json_output)])
    command.extend(args)
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _load_payload(result: subprocess.CompletedProcess[str], *, json_output: Path | None = None) -> dict[str, object]:
    if json_output is not None:
        assert json_output.exists(), "Expected --json-output artifact"
        return json.loads(json_output.read_text(encoding="utf-8"))
    assert result.stdout.strip(), f"Expected JSON payload on stdout; stderr={result.stderr!r}"
    return json.loads(result.stdout)


def _bag_file_digest_map(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    for entry in sorted(path.rglob("*")):
        if entry.is_file():
            files[str(entry.relative_to(path).as_posix())] = entry.read_bytes()
    return files


@pytest.mark.regression
def test_fixity_scan_and_verify_machine_payloads_validate_schema(tmp_path: Path) -> None:
    scan_dir = tmp_path / "scan"
    scan_result = _run_governance_cli(
        "fixity-scan",
        "--archive-index",
        str(FIXTURE_INDEX),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--out-dir",
        str(scan_dir),
        "--workers",
        "1",
        "--no-validate-schemas",
    )
    scan_payload = _load_payload(scan_result)
    assert scan_result.returncode == 0, scan_result.stderr
    validate_archive_machine_payload(scan_payload)
    assert scan_payload["command"] == "fixity-scan"

    verify_report = tmp_path / "verify_report.json"
    verify_result = _run_governance_cli(
        "fixity-verify",
        "--hash-manifest",
        str(scan_dir / "hash_manifest.csv.gz"),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--report-path",
        str(verify_report),
        "--verify-sample",
        "0",
        "--workers",
        "1",
    )
    verify_payload = _load_payload(verify_result)
    assert verify_result.returncode == 0, verify_result.stderr
    validate_archive_machine_payload(verify_payload)
    assert verify_payload["command"] == "fixity-verify"
    assert verify_report.exists()


@pytest.mark.regression
def test_archive_governance_round_trip_commands_emit_valid_contracts(tmp_path: Path) -> None:
    manifest_jsonl = tmp_path / "archive_manifest_v2.jsonl"
    manifest_summary = tmp_path / "archive_manifest_v2.summary.json"
    rights_jsonl = tmp_path / "archive_manifest_v2.rights.jsonl"
    rights_summary = tmp_path / "asset_rights.summary.json"
    bag_dir = tmp_path / "bag"
    bag_report = tmp_path / "bag_build_report.json"
    bag_validate_report = tmp_path / "bag_validate_report.json"
    dedup_ledger = tmp_path / "dedup_ledger.csv"
    dedup_summary = tmp_path / "dedup_summary.json"
    mets_xml = tmp_path / "mets_export.xml"
    mets_summary = tmp_path / "mets_summary.json"
    prov_jsonld = tmp_path / "prov.jsonld"
    prov_summary = tmp_path / "prov_summary.json"
    stac_catalog = tmp_path / "catalog.json"
    stac_items = tmp_path / "stac_items"
    stac_summary = tmp_path / "stac_summary.json"
    premis_jsonl = tmp_path / "premis_events.jsonl"

    manifest_result = _run_governance_cli(
        "manifest-build",
        "--archive-index",
        str(FIXTURE_INDEX),
        "--hash-manifest",
        str(FIXTURE_HASH_MANIFEST),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--out-jsonl",
        str(manifest_jsonl),
        "--out-summary",
        str(manifest_summary),
        "--collection-id",
        "test_collection",
        "--owner",
        "test_owner",
    )
    manifest_payload = _load_payload(manifest_result)
    assert manifest_result.returncode == 0, manifest_result.stderr
    validate_archive_machine_payload(manifest_payload)
    assert manifest_jsonl.exists()
    assert manifest_summary.exists()

    rights_result = _run_governance_cli(
        "rights-apply",
        "--manifest-jsonl",
        str(manifest_jsonl),
        "--policy-yaml",
        str(RIGHTS_POLICY),
        "--out-jsonl",
        str(rights_jsonl),
        "--out-summary",
        str(rights_summary),
    )
    rights_payload = _load_payload(rights_result)
    assert rights_result.returncode == 0, rights_result.stderr
    validate_archive_machine_payload(rights_payload)
    assert rights_jsonl.exists()
    assert rights_summary.exists()

    bag_build_result = _run_governance_cli(
        "bag-build",
        "--manifest-jsonl",
        str(rights_jsonl),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--bag-dir",
        str(bag_dir),
        "--report-json",
        str(bag_report),
    )
    bag_build_payload = _load_payload(bag_build_result)
    assert bag_build_result.returncode == 0, bag_build_result.stderr
    validate_archive_machine_payload(bag_build_payload)
    assert (bag_dir / "bagit.txt").exists()
    assert bag_report.exists()

    bag_validate_result = _run_governance_cli(
        "bag-validate",
        "--bag-dir",
        str(bag_dir),
        "--report-json",
        str(bag_validate_report),
    )
    bag_validate_payload = _load_payload(bag_validate_result)
    assert bag_validate_result.returncode == 0, bag_validate_result.stderr
    validate_archive_machine_payload(bag_validate_payload)
    assert bag_validate_report.exists()

    dedup_result = _run_governance_cli(
        "dedup-plan",
        "--manifest-jsonl",
        str(rights_jsonl),
        "--out-ledger",
        str(dedup_ledger),
        "--out-summary",
        str(dedup_summary),
        "--approver",
        "test_approver",
    )
    dedup_payload = _load_payload(dedup_result)
    assert dedup_result.returncode == 0, dedup_result.stderr
    validate_archive_machine_payload(dedup_payload)
    assert dedup_ledger.exists()
    assert dedup_summary.exists()

    mets_result = _run_governance_cli(
        "mets-export",
        "--manifest-jsonl",
        str(rights_jsonl),
        "--out-xml",
        str(mets_xml),
        "--out-summary",
        str(mets_summary),
    )
    mets_payload = _load_payload(mets_result)
    assert mets_result.returncode == 0, mets_result.stderr
    validate_archive_machine_payload(mets_payload)
    assert mets_xml.exists()
    assert mets_summary.exists()

    prov_result = _run_governance_cli(
        "prov-export",
        "--manifest-jsonl",
        str(rights_jsonl),
        "--out-prov-jsonld",
        str(prov_jsonld),
        "--out-summary",
        str(prov_summary),
    )
    prov_payload = _load_payload(prov_result)
    assert prov_result.returncode == 0, prov_result.stderr
    validate_archive_machine_payload(prov_payload)
    assert prov_jsonld.exists()
    assert prov_summary.exists()

    stac_result = _run_governance_cli(
        "stac-export",
        "--manifest-jsonl",
        str(rights_jsonl),
        "--out-prov-jsonld",
        str(prov_jsonld),
        "--out-stac-catalog",
        str(stac_catalog),
        "--out-stac-items-dir",
        str(stac_items),
        "--out-summary",
        str(stac_summary),
    )
    stac_payload = _load_payload(stac_result)
    assert stac_result.returncode == 0, stac_result.stderr
    validate_archive_machine_payload(stac_payload)
    assert stac_summary.exists()

    premis_output = tmp_path / "premis_envelope.json"
    premis_result = _run_governance_cli(
        "premis-export",
        "--out-jsonl",
        str(premis_jsonl),
        "--event-type",
        "validation",
        "--event-detail",
        "test-event",
        "--event-outcome",
        "success",
        "--agent-id",
        "tp.archive.tests",
        "--object-id",
        str(manifest_jsonl),
        json_output=premis_output,
    )
    premis_payload = _load_payload(premis_result, json_output=premis_output)
    assert premis_result.returncode == 0, premis_result.stderr
    validate_archive_machine_payload(premis_payload)
    assert premis_jsonl.exists()


@pytest.mark.regression
def test_manifest_and_bag_outputs_are_deterministic(tmp_path: Path) -> None:
    manifest_a = tmp_path / "manifest_a.jsonl"
    summary_a = tmp_path / "summary_a.json"
    manifest_b = tmp_path / "manifest_b.jsonl"
    summary_b = tmp_path / "summary_b.json"
    bag_a = tmp_path / "bag_a"
    bag_b = tmp_path / "bag_b"

    result_manifest_a = _run_governance_cli(
        "manifest-build",
        "--archive-index",
        str(FIXTURE_INDEX),
        "--hash-manifest",
        str(FIXTURE_HASH_MANIFEST),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--out-jsonl",
        str(manifest_a),
        "--out-summary",
        str(summary_a),
    )
    assert result_manifest_a.returncode == 0, result_manifest_a.stderr

    result_manifest_b = _run_governance_cli(
        "manifest-build",
        "--archive-index",
        str(FIXTURE_INDEX),
        "--hash-manifest",
        str(FIXTURE_HASH_MANIFEST),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--out-jsonl",
        str(manifest_b),
        "--out-summary",
        str(summary_b),
    )
    assert result_manifest_b.returncode == 0, result_manifest_b.stderr

    assert manifest_a.read_bytes() == manifest_b.read_bytes()
    assert summary_a.read_bytes() == summary_b.read_bytes()

    result_bag_a = _run_governance_cli(
        "bag-build",
        "--manifest-jsonl",
        str(manifest_a),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--bag-dir",
        str(bag_a),
    )
    assert result_bag_a.returncode == 0, result_bag_a.stderr

    result_bag_b = _run_governance_cli(
        "bag-build",
        "--manifest-jsonl",
        str(manifest_a),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--bag-dir",
        str(bag_b),
    )
    assert result_bag_b.returncode == 0, result_bag_b.stderr

    assert _bag_file_digest_map(bag_a) == _bag_file_digest_map(bag_b)


@pytest.mark.regression
def test_failure_path_emits_typed_machine_error(tmp_path: Path) -> None:
    payload_path = tmp_path / "error_envelope.json"
    result = _run_governance_cli(
        "fixity-verify",
        "--hash-manifest",
        str(tmp_path / "missing_hash_manifest.csv.gz"),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--report-path",
        str(tmp_path / "verify_report.json"),
        json_output=payload_path,
    )
    payload = _load_payload(result, json_output=payload_path)
    assert result.returncode != 0
    validate_archive_machine_payload(payload)
    assert payload["success"] is False
    assert payload["error"] is not None
    assert payload["error"]["type"] == "ToolExecutionError"


@pytest.mark.regression
def test_sealed_eval_run_emits_audit_package_and_machine_payload(tmp_path: Path) -> None:
    out_root = tmp_path / "sealed_eval_runs"
    result = _run_governance_cli(
        "sealed-eval-run",
        "--archive-index",
        str(FIXTURE_INDEX),
        "--archive-root",
        str(FIXTURE_ARCHIVE_ROOT),
        "--out-root",
        str(out_root),
        "--eval-command",
        "true",
        "--no-validate-schemas",
        "--allow-writable-subset",
    )
    payload = _load_payload(result)
    assert result.returncode == 0, result.stderr
    validate_archive_machine_payload(payload)

    summary_files = sorted(out_root.rglob("sealed_eval_summary.json"))
    assert summary_files, "Expected sealed eval summary output"
    summary_payload = json.loads(summary_files[-1].read_text(encoding="utf-8"))
    assert summary_payload["sealed_integrity_passed"] is True
    assert summary_payload["evaluation_command_present"] is True
    assert isinstance(summary_payload["evaluation_command_sha256"], str)
    assert len(summary_payload["evaluation_command_sha256"]) == 64
    assert "evaluation_command" not in summary_payload

    audit_manifest = summary_files[-1].parent / "audit_package" / "audit_manifest.json"
    assert audit_manifest.exists()
    audit_payload = json.loads(audit_manifest.read_text(encoding="utf-8"))
    assert audit_payload["schema_version"] == "tp.archive.sealed_eval.audit_package.v1"
    assert int(audit_payload["file_count"]) >= 1


@pytest.mark.regression
def test_jcs_canonical_profile_emits_valid_machine_json(tmp_path: Path) -> None:
    premis_jsonl = tmp_path / "premis_jcs_events.jsonl"
    result = _run_governance_cli(
        "premis-export",
        "--out-jsonl",
        str(premis_jsonl),
        "--event-type",
        "validation",
        "--event-detail",
        "jcs-profile",
        "--event-outcome",
        "success",
        "--agent-id",
        "tp.archive.tests",
        canonical_profile="jcs",
    )
    payload = _load_payload(result)
    assert result.returncode == 0, result.stderr
    validate_archive_machine_payload(payload)
    assert premis_jsonl.exists()
