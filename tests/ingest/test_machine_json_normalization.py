"""Contract tests for governed ingest machine JSON normalization."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.ingest.normalize_machine_json import canonical_json_bytes, normalize_machine_payload

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "normalize_machine_json.py"
FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "ingest" / "normalize"


def _load_fixture(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))


def _run_normalizer(input_path: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath_parts = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    return subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--in",
            str(input_path),
            "--out",
            str(output_path),
            "--profile",
            "ingest_v1",
            "--emit-sha256",
        ],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_normalization_contract_strips_volatile_fields_but_preserves_file_derived_values() -> None:
    run_a = _load_fixture("provenance_run_a.json")
    run_b = _load_fixture("provenance_run_b.json")

    normalized_a = normalize_machine_payload(run_a, profile="ingest_v1")
    normalized_b = normalize_machine_payload(run_b, profile="ingest_v1")

    bytes_a = canonical_json_bytes(normalized_a)
    bytes_b = canonical_json_bytes(normalized_b)

    assert bytes_a == bytes_b
    assert normalized_a["file_integrity"]["sha256"] == run_a["file_integrity"]["sha256"]
    assert normalized_a["pipeline_config"]["config_sha256"] == run_a["pipeline_config"]["config_sha256"]
    assert normalized_a["exif"]["all_tags"] == run_a["exif"]["all_tags"]

    for volatile_key in ("run_id", "timestamps", "host", "toolchain", "git_commit"):
        assert volatile_key not in normalized_a
        assert volatile_key not in normalized_b


def test_normalizer_cli_emits_stable_sha256_for_equivalent_normalized_payloads(tmp_path: Path) -> None:
    fixture_a = FIXTURE_DIR / "provenance_run_a.json"
    fixture_b = FIXTURE_DIR / "provenance_run_b.json"
    output_a = tmp_path / "a.normalized.json"
    output_b = tmp_path / "b.normalized.json"

    result_a = _run_normalizer(fixture_a, output_a)
    result_b = _run_normalizer(fixture_b, output_b)

    assert result_a.returncode == 0, result_a.stderr
    assert result_b.returncode == 0, result_b.stderr
    assert output_a.read_bytes() == output_b.read_bytes()

    sha_a = result_a.stderr.strip().splitlines()[-1]
    sha_b = result_b.stderr.strip().splitlines()[-1]
    assert sha_a == sha_b
    assert sha_a == hashlib.sha256(output_a.read_bytes()).hexdigest()
