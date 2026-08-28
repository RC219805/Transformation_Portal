"""CLI tests for tools/sign_evidence_attestation.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.attestation.detached import compute_attestation_sha256
from transformation_portal.ingest.evidence import build_evidence_payload, load_projection_profile

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "sign_evidence_attestation.py"
VERIFY_TOOL_PATH = PROJECT_ROOT / "tools" / "verify_evidence_attestation.py"
FAKE_GPG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "attestation" / "fake_gpg.py"


def _run_tool(
    *args: str,
    path_prepend: Path | None = None,
    env_updates: dict[str, str] | None = None,
    tool_path: Path = TOOL_PATH,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath_parts = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if path_prepend is not None:
        env["PATH"] = os.pathsep.join([str(path_prepend), env.get("PATH", "")])
    if env_updates:
        env.update(env_updates)

    return subprocess.run(
        [sys.executable, str(tool_path), *args],
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


def _write_evidence(tmp_path: Path) -> Path:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0),
        projection_profile=load_projection_profile(),
    )
    evidence_path = tmp_path / "evidence.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    return evidence_path


def _write_fake_gpg(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    gpg_path = bin_dir / "gpg"
    gpg_path.write_bytes(FAKE_GPG_PATH.read_bytes())
    gpg_path.chmod(0o755)
    return bin_dir


def test_sign_cli_requires_signing_backend(tmp_path: Path) -> None:
    evidence_path = _write_evidence(tmp_path)
    output_path = tmp_path / "attestation.json"

    result = _run_tool("--evidence", str(evidence_path), "--out", str(output_path), "--key-id", "test-key")

    assert result.returncode == 4
    assert "No signing backend selected" in result.stderr
    assert not output_path.exists()


def test_sign_cli_writes_attestation_with_fake_gpg(tmp_path: Path) -> None:
    evidence_path = _write_evidence(tmp_path)
    output_path = tmp_path / "nested" / "attestation.json"
    fake_path = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--out",
        str(output_path),
        "--gpg",
        "--key-id",
        "test-key",
        path_prepend=fake_path,
    )

    assert result.returncode == 0, result.stderr
    assert output_path.exists()

    attestation = json.loads(output_path.read_text(encoding="utf-8"))
    assert attestation["schema"] == "tp.attestation.detached.v1"
    assert attestation["signature"]["algorithm"] == "openpgp-clearsign"
    assert attestation["signature"]["key_id"] == "test-key"
    assert "BEGIN PGP SIGNED MESSAGE" in attestation["signature"]["signature"]
    assert attestation["attestation_sha256"] == compute_attestation_sha256(attestation)

    verify_result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--attestation",
        str(output_path),
        "--gpg",
        path_prepend=fake_path,
        tool_path=VERIFY_TOOL_PATH,
    )
    assert verify_result.returncode == 0, verify_result.stderr

    tmp_siblings = list(output_path.parent.glob(f".{output_path.name}.*.tmp"))
    assert tmp_siblings == []


def test_sign_cli_fails_closed_when_recorded_key_does_not_match_signature(tmp_path: Path) -> None:
    evidence_path = _write_evidence(tmp_path)
    output_path = tmp_path / "attestation.json"
    fake_path = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--out",
        str(output_path),
        "--gpg",
        "--key-id",
        "logical-label",
        path_prepend=fake_path,
        env_updates={"TP_FAKE_GPG_RESOLVED_FINGERPRINT": "B" * 40},
    )

    assert result.returncode == 4
    assert "primary fingerprint does not match recorded key_id" in result.stderr
    assert not output_path.exists()


def test_sign_cli_marks_no_recompute_check_deprecated() -> None:
    result = _run_tool("--help")

    assert result.returncode == 0
    assert "DEPRECATED" in result.stdout
    assert "forensic migration only" in result.stdout


def test_no_recompute_check_does_not_create_verifier_bypass(tmp_path: Path) -> None:
    evidence_path = _write_evidence(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["projected_envelope"]["data"]["preset"] = "tampered"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    output_path = tmp_path / "attestation.json"
    fake_path = _write_fake_gpg(tmp_path)

    sign_result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--out",
        str(output_path),
        "--gpg",
        "--key-id",
        "test-key",
        "--no-recompute-check",
        path_prepend=fake_path,
    )
    verify_result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--attestation",
        str(output_path),
        tool_path=VERIFY_TOOL_PATH,
    )

    assert sign_result.returncode == 0, sign_result.stderr
    assert "deprecated" in sign_result.stderr
    assert verify_result.returncode == 5
    assert "projected_envelope does not reproduce stored evidence_sha256" in verify_result.stderr
