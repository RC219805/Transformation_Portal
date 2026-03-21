"""CLI tests for tools/verify_evidence_attestation.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.attestation.detached import build_detached_attestation_payload
from transformation_portal.ingest.evidence import build_evidence_payload, load_projection_profile

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "verify_evidence_attestation.py"


def _run_tool(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath_parts = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    return subprocess.run(
        [sys.executable, str(TOOL_PATH), *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _machine_extract_payload(*, elapsed_seconds: float) -> dict[str, Any]:
    return {
        "schema": "tp.meta.machine.v1",
        "command": "extract",
        "success": True,
        "exit_code": 0,
        "error": None,
        "data": {
            "input_path": "/tmp/source.cr2",
            "success": True,
            "output_path": "/tmp/source.provenance.json",
            "elapsed_seconds": elapsed_seconds,
            "preset": "luxury",
            "error": None,
        },
    }


def _build_inputs(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0), projection_profile=load_projection_profile()
    )
    signature = {"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"}
    attestation = build_detached_attestation_payload(evidence, signature=signature)

    evidence_path = tmp_path / "evidence.json"
    attestation_path = tmp_path / "attestation.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    return evidence_path, attestation_path, attestation


def test_verify_cli_succeeds_for_matching_evidence_and_attestation(tmp_path: Path) -> None:
    evidence_path, attestation_path, _ = _build_inputs(tmp_path)

    result = _run_tool("--evidence", str(evidence_path), "--attestation", str(attestation_path))

    assert result.returncode == 0, result.stderr


def test_verify_cli_rejects_attestation_self_hash_mismatch(tmp_path: Path) -> None:
    evidence_path, attestation_path, attestation = _build_inputs(tmp_path)
    attestation["claims"] = {"tampered": True}
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")

    result = _run_tool("--evidence", str(evidence_path), "--attestation", str(attestation_path))

    assert result.returncode == 5
    assert "attestation_sha256 mismatch" in result.stderr


def test_verify_cli_requires_attestation_self_hash_by_default(tmp_path: Path) -> None:
    evidence_path, attestation_path, attestation = _build_inputs(tmp_path)
    attestation["attestation_sha256"] = None
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")

    result = _run_tool("--evidence", str(evidence_path), "--attestation", str(attestation_path))

    assert result.returncode == 5
    assert "attestation_sha256 must be set" in result.stderr


def test_verify_cli_allows_missing_attestation_self_hash_with_flag(tmp_path: Path) -> None:
    evidence_path, attestation_path, attestation = _build_inputs(tmp_path)
    attestation["attestation_sha256"] = None
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")

    result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--attestation",
        str(attestation_path),
        "--allow-missing-attestation-sha",
    )

    assert result.returncode == 0, result.stderr


def test_verify_cli_rejects_non_gpg_algorithm_when_gpg_flag_is_set(tmp_path: Path) -> None:
    evidence_path, attestation_path, _ = _build_inputs(tmp_path)

    result = _run_tool("--evidence", str(evidence_path), "--attestation", str(attestation_path), "--gpg")

    assert result.returncode == 5
    assert "signature.algorithm must be 'openpgp-clearsign'" in result.stderr
