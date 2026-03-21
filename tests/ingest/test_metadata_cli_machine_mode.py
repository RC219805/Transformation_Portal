"""CLI contract tests for metadata extraction machine-mode JSON output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any
import pytest


pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "test_metadata_extraction.py"


def _run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _normalize_machine_contract_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize known volatile fields before golden comparisons."""
    normalized = json.loads(json.dumps(payload))
    data = normalized.get("data")
    if not isinstance(data, dict):
        return normalized

    if "elapsed_seconds" in data:
        data["elapsed_seconds"] = "__normalized_elapsed_seconds__"

    items = data.get("items")
    if isinstance(items, list):
        for item in items:
            if isinstance(item, dict) and "elapsed_seconds" in item:
                item["elapsed_seconds"] = "__normalized_elapsed_seconds__"

    for key in ("exiftool_version", "pydantic_version", "git_version", "rawpy_version", "libraw_version"):
        if key in data and data[key] is not None:
            data[key] = "__normalized_environment_version__"

    return normalized


def _canonical_machine_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def test_check_system_json_contract_shape_is_stable() -> None:
    result = _run_cli("--json", "check-system")

    payload = json.loads(result.stdout)
    assert payload["command"] == "check-system"
    assert payload["schema"] == "tp.meta.machine.v1"
    assert payload["exit_code"] == result.returncode
    assert payload["success"] is (result.returncode == 0)

    data = payload["data"]
    assert list(data.keys()) == [
        "all_required_ok",
        "errors",
        "exiftool_available",
        "exiftool_version",
        "git_available",
        "git_version",
        "ingest_module_available",
        "libraw_version",
        "pydantic_available",
        "pydantic_version",
        "rawpy_available",
        "rawpy_version",
    ]
    assert isinstance(data["all_required_ok"], bool)
    assert isinstance(data["errors"], list)


def test_check_system_json_contract_stable_after_normalization() -> None:
    first = _run_cli("--json", "check-system")
    second = _run_cli("--json", "check-system")

    assert first.returncode == second.returncode
    first_payload = _normalize_machine_contract_payload(json.loads(first.stdout))
    second_payload = _normalize_machine_contract_payload(json.loads(second.stdout))
    assert _canonical_machine_json(first_payload) == _canonical_machine_json(second_payload)


def test_validate_json_envelope_is_versioned_and_stable(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "missing.provenance.json"

    first = _run_cli("--json", "validate", str(sidecar_path))
    second = _run_cli("--json", "validate", str(sidecar_path))

    assert first.returncode == second.returncode == 5
    assert first.stdout == second.stdout

    payload = json.loads(first.stdout)
    assert list(payload.keys()) == ["command", "data", "error", "exit_code", "schema", "success"]
    assert payload["schema"] == "tp.meta.machine.v1"
    assert payload["command"] == "validate"
    assert payload["success"] is False
    assert payload["exit_code"] == first.returncode
    assert payload["error"] is None

    data = payload["data"]
    assert list(data.keys()) == ["dominant_error", "errors", "sidecar_path", "strict", "success"]
    assert data["sidecar_path"] == str(sidecar_path)
    assert data["strict"] is True
    assert data["success"] is False
    assert isinstance(data["errors"], list)
    assert data["dominant_error"]["type"] == "OtherIngestFailure"
    assert data["dominant_error"]["exit_code"]["name"] == "OTHER_FAILURE"
    assert data["dominant_error"]["exit_code"]["value"] == payload["exit_code"]


def test_extract_batch_setup_failure_uses_data_not_command_error(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing_dir"
    result = _run_cli("--json", "extract-batch", str(missing_dir))

    assert result.returncode == 5
    payload = json.loads(result.stdout)
    assert payload["command"] == "extract-batch"
    assert payload["error"] is None

    data = payload["data"]
    assert list(data.keys()) == [
        "dominant_error",
        "fail_fast",
        "input_root",
        "items",
        "output_dir",
        "preserve_structure",
        "success",
        "summary_counts",
    ]
    assert data["items"] == []
    assert data["summary_counts"]["total"] == 0
    assert data["summary_counts"]["success"] == 0
    assert data["summary_counts"]["failure"] == 0
    assert data["success"] is False
    assert data["dominant_error"]["type"] == "OtherIngestFailure"
    assert data["dominant_error"]["exit_code"]["value"] == payload["exit_code"]


def test_extract_json_contract_is_stable_after_normalizing_volatile_fields(tmp_path: Path) -> None:
    missing_input = tmp_path / "missing.cr2"

    first = _run_cli("--json", "extract", str(missing_input))
    second = _run_cli("--json", "extract", str(missing_input))

    assert first.returncode == second.returncode == 5
    first_payload = _normalize_machine_contract_payload(json.loads(first.stdout))
    second_payload = _normalize_machine_contract_payload(json.loads(second.stdout))

    assert first_payload == second_payload
    assert first_payload["data"]["elapsed_seconds"] == "__normalized_elapsed_seconds__"


def test_json_output_with_pretty_writes_file_and_keeps_stdout_clean(tmp_path: Path) -> None:
    output_path = tmp_path / "machine.json"
    sidecar_path = tmp_path / "missing.provenance.json"

    result = _run_cli(
        "--json",
        "--json-pretty",
        "--json-output",
        str(output_path),
        "validate",
        str(sidecar_path),
    )

    assert result.returncode == 5
    assert result.stdout == ""
    assert output_path.exists()

    written = output_path.read_text(encoding="utf-8")
    assert written.startswith("{\n")
    payload = json.loads(written)
    assert payload["schema"] == "tp.meta.machine.v1"
    assert payload["command"] == "validate"


def test_json_flags_require_json_mode(tmp_path: Path) -> None:
    output_path = tmp_path / "machine.json"
    sidecar_path = tmp_path / "missing.provenance.json"

    output_only = _run_cli("--json-output", str(output_path), "validate", str(sidecar_path))
    pretty_only = _run_cli("--json-pretty", "validate", str(sidecar_path))

    assert output_only.returncode == 2
    assert pretty_only.returncode == 2
    assert "--json-pretty and --json-output require --json" in output_only.stderr
    assert "--json-pretty and --json-output require --json" in pretty_only.stderr
