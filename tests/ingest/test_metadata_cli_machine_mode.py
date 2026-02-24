"""CLI contract tests for metadata extraction machine-mode JSON output."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

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
    assert list(data.keys()) == ["dominant_error", "errors", "sidecar_path", "strict"]
    assert data["sidecar_path"] == str(sidecar_path)
    assert data["strict"] is True
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
    assert data["items"] == []
    assert data["summary_counts"]["total"] == 0
    assert data["summary_counts"]["success"] == 0
    assert data["summary_counts"]["failure"] == 0
    assert data["dominant_error"]["type"] == "OtherIngestFailure"
    assert data["dominant_error"]["exit_code"]["value"] == payload["exit_code"]


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
