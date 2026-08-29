"""CLI tests for tools/verify_evidence_attestation.py."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.attestation.detached import (
    build_detached_attestation_payload,
    canonical_attestation_preimage_bytes,
)
from transformation_portal.ingest.evidence import build_evidence_payload, load_projection_profile

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "verify_evidence_attestation.py"
FAKE_GPG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "attestation" / "fake_gpg.py"
PRIMARY_FINGERPRINT = "A" * 40


def _run_tool(
    *args: str,
    path_prepend: Path | None = None,
    env_updates: dict[str, str] | None = None,
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


def _fake_clearsign(payload: bytes) -> str:
    encoded = base64.b64encode(payload).decode("ascii")
    return (
        "-----BEGIN PGP SIGNED MESSAGE-----\n"
        "Hash: SHA256\n"
        f"X-TP-Fake-Payload: {encoded}\n\n"
        "payload\n"
        "-----BEGIN PGP SIGNATURE-----\n"
        "fake-clearsign\n"
        "-----END PGP SIGNATURE-----\n"
    )


def _write_fake_gpg(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    gpg_path = bin_dir / "gpg"
    gpg_path.write_bytes(FAKE_GPG_PATH.read_bytes())
    gpg_path.chmod(0o755)
    return bin_dir


def _build_gpg_inputs(tmp_path: Path, *, signed_payload: bytes | None = None) -> tuple[Path, Path]:
    evidence = build_evidence_payload(
        _machine_extract_payload(elapsed_seconds=1.0),
        projection_profile=load_projection_profile(),
    )
    signature_payload = signed_payload if signed_payload is not None else canonical_attestation_preimage_bytes(evidence)
    attestation = build_detached_attestation_payload(
        evidence,
        signature={
            "algorithm": "openpgp-clearsign",
            "key_id": PRIMARY_FINGERPRINT,
            "signature": _fake_clearsign(signature_payload),
        },
    )
    evidence_path = tmp_path / "evidence.json"
    attestation_path = tmp_path / "attestation.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    return evidence_path, attestation_path


def test_verify_cli_succeeds_for_matching_evidence_and_attestation(tmp_path: Path) -> None:
    evidence_path, attestation_path, _ = _build_inputs(tmp_path)

    result = _run_tool("--evidence", str(evidence_path), "--attestation", str(attestation_path))

    assert result.returncode == 0, result.stderr


def test_verify_cli_rejects_tampered_projected_envelope(tmp_path: Path) -> None:
    evidence_path, attestation_path, _ = _build_inputs(tmp_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["projected_envelope"]["data"]["preset"] = "tampered"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")

    result = _run_tool("--evidence", str(evidence_path), "--attestation", str(attestation_path))

    assert result.returncode == 5
    assert "projected_envelope does not reproduce stored evidence_sha256" in result.stderr


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


def test_verify_cli_rejects_secondary_anchor_mismatch_even_when_self_hash_is_allowed_missing(
    tmp_path: Path,
) -> None:
    machine_payload = _machine_extract_payload(elapsed_seconds=1.0)
    machine_payload["data"]["file_integrity"] = {"sha256": "a" * 64}
    evidence = build_evidence_payload(
        machine_payload,
        projection_profile=load_projection_profile(),
        bundle_root_sha256="b" * 64,
    )
    attestation = build_detached_attestation_payload(
        evidence,
        signature={"algorithm": "unit-test", "key_id": "test", "signature": "deadbeef"},
    )
    attestation["subject"]["bundle_root_sha256"] = "c" * 64
    attestation["attestation_sha256"] = None
    evidence_path = tmp_path / "evidence.json"
    attestation_path = tmp_path / "attestation.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")

    result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--attestation",
        str(attestation_path),
        "--allow-missing-attestation-sha",
    )

    assert result.returncode == 5
    assert "bundle_root_sha256 mismatch" in result.stderr


def test_verify_cli_accepts_exact_gpg_preimage_and_recorded_key(tmp_path: Path) -> None:
    evidence_path, attestation_path = _build_gpg_inputs(tmp_path)
    fake_gpg = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--attestation",
        str(attestation_path),
        "--gpg",
        path_prepend=fake_gpg,
    )

    assert result.returncode == 0, result.stderr


def test_verify_cli_rejects_unrelated_valid_clearsign(tmp_path: Path) -> None:
    evidence_path, attestation_path = _build_gpg_inputs(tmp_path, signed_payload=b'{"schema":"unrelated"}')
    fake_gpg = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--attestation",
        str(attestation_path),
        "--gpg",
        path_prepend=fake_gpg,
    )

    assert result.returncode == 5
    assert "does not match the expected canonical preimage bytes" in result.stderr


@pytest.mark.parametrize("status_mode", ["missing", "ambiguous"])
def test_verify_cli_rejects_missing_or_ambiguous_validsig(tmp_path: Path, status_mode: str) -> None:
    evidence_path, attestation_path = _build_gpg_inputs(tmp_path)
    fake_gpg = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--attestation",
        str(attestation_path),
        "--gpg",
        path_prepend=fake_gpg,
        env_updates={"TP_FAKE_GPG_STATUS_MODE": status_mode},
    )

    assert result.returncode == 5
    assert "exactly one VALIDSIG record" in result.stderr


def test_verify_cli_rejects_recorded_gpg_key_mismatch_with_missing_self_hash_allowed(tmp_path: Path) -> None:
    evidence_path, attestation_path = _build_gpg_inputs(tmp_path)
    attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
    attestation["attestation_sha256"] = None
    attestation_path.write_text(json.dumps(attestation), encoding="utf-8")
    fake_gpg = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--evidence",
        str(evidence_path),
        "--attestation",
        str(attestation_path),
        "--allow-missing-attestation-sha",
        "--gpg",
        path_prepend=fake_gpg,
        env_updates={"TP_FAKE_GPG_RESOLVED_FINGERPRINT": "B" * 40},
    )

    assert result.returncode == 5
    assert "primary fingerprint does not match recorded key_id" in result.stderr
