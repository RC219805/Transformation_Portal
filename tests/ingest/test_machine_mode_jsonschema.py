"""JSON Schema validation tests for tp.meta.machine.v1 machine-mode outputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from transformation_portal.ingest.schemas import (
    ExifMetadata,
    FileIntegrity,
    HostEnvironment,
    IngestTimestamps,
    PipelineConfig,
    ProvenanceSidecar,
)

from .schema_utils import normalize_machine_payload, validate_machine_payload

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


def _load_payload(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    assert result.stdout.strip(), f"Expected JSON payload on stdout; stderr={result.stderr!r}"
    return json.loads(result.stdout)


def _canonical_json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _write_valid_sidecar(sidecar_path: Path) -> None:
    sidecar = ProvenanceSidecar(
        file_integrity=FileIntegrity(
            sha256="a" * 64,
            size_bytes=1024,
            path="/input/test.cr2",
        ),
        exif=ExifMetadata(all_tags={}),
        toolchain=[],
        host=HostEnvironment(
            hostname="test-host",
            os="Linux",
            os_version="6.0.0",
            python_version="3.11.0",
            arch="x86_64",
        ),
        timestamps=IngestTimestamps(
            ingest_start="2026-02-10T12:00:00+00:00",
            ingest_end="2026-02-10T12:05:00+00:00",
        ),
        pipeline_config=PipelineConfig(config_sha256="b" * 64),
        run_id="test-run",
    )
    sidecar_path.write_text(sidecar.model_dump_json(indent=2), encoding="utf-8")


def test_check_system_machine_payload_matches_json_schema() -> None:
    payload = _load_payload(_run_cli("--json", "check-system"))
    validate_machine_payload(payload)


def test_extract_failure_machine_payload_matches_json_schema(tmp_path: Path) -> None:
    payload = _load_payload(_run_cli("--json", "extract", str(tmp_path / "missing.cr2")))
    validate_machine_payload(payload)


def test_validate_failure_machine_payload_matches_json_schema(tmp_path: Path) -> None:
    payload = _load_payload(_run_cli("--json", "validate", str(tmp_path / "missing.provenance.json")))
    validate_machine_payload(payload)


def test_extract_batch_setup_failure_machine_payload_matches_json_schema(tmp_path: Path) -> None:
    payload = _load_payload(_run_cli("--json", "extract-batch", str(tmp_path / "missing-input-root")))
    validate_machine_payload(payload)


def test_validate_success_machine_payload_matches_json_schema(tmp_path: Path) -> None:
    sidecar_path = tmp_path / "valid.provenance.json"
    _write_valid_sidecar(sidecar_path)

    result = _run_cli("--json", "validate", str(sidecar_path))
    payload = _load_payload(result)

    assert result.returncode == 0
    validate_machine_payload(payload)
    assert payload["success"] is True
    assert payload["error"] is None


def test_extract_batch_empty_directory_success_matches_json_schema(tmp_path: Path) -> None:
    input_root = tmp_path / "empty-inputs"
    input_root.mkdir()

    result = _run_cli("--json", "extract-batch", str(input_root))
    payload = _load_payload(result)

    assert result.returncode == 0
    validate_machine_payload(payload)
    assert payload["success"] is True


def test_summarize_missing_directory_failure_matches_json_schema(tmp_path: Path) -> None:
    payload = _load_payload(_run_cli("--json", "summarize", str(tmp_path / "missing-sidecars")))
    validate_machine_payload(payload)
    assert payload["command"] == "summarize"
    assert payload["success"] is False


def test_summarize_empty_directory_success_matches_json_schema(tmp_path: Path) -> None:
    sidecar_dir = tmp_path / "sidecars"
    sidecar_dir.mkdir()

    result = _run_cli("--json", "summarize", str(sidecar_dir))
    payload = _load_payload(result)

    assert result.returncode == 0
    validate_machine_payload(payload)
    assert payload["success"] is True
    assert payload["data"]["total_sidecars"] == 0


def test_schema_rejects_unknown_top_level_field(tmp_path: Path) -> None:
    payload = _load_payload(_run_cli("--json", "validate", str(tmp_path / "missing.provenance.json")))
    payload["unexpected_contract_field"] = True

    with pytest.raises(AssertionError):
        validate_machine_payload(payload)


def test_schema_rejects_wrong_type_for_exit_code(tmp_path: Path) -> None:
    payload = _load_payload(_run_cli("--json", "validate", str(tmp_path / "missing.provenance.json")))
    payload["exit_code"] = "5"

    with pytest.raises(AssertionError):
        validate_machine_payload(payload)


def test_normalized_check_system_payload_is_stable_and_schema_valid() -> None:
    first_raw = _load_payload(_run_cli("--json", "check-system"))
    second_raw = _load_payload(_run_cli("--json", "check-system"))

    first = normalize_machine_payload(first_raw)
    second = normalize_machine_payload(second_raw)

    validate_machine_payload(first)
    validate_machine_payload(second)
    assert _canonical_json(first) == _canonical_json(second)
