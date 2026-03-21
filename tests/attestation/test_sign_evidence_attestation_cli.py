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


def _run_tool(*args: str, path_prepend: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    pythonpath_parts = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    if path_prepend is not None:
        env["PATH"] = os.pathsep.join([str(path_prepend), env.get("PATH", "")])

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
    gpg_path.write_text(
        """#!/bin/sh
cat >/dev/null
cat <<'EOF'
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

payload
-----BEGIN PGP SIGNATURE-----
fake-signature
-----END PGP SIGNATURE-----
EOF
""",
        encoding="utf-8",
    )
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

    tmp_siblings = list(output_path.parent.glob(f".{output_path.name}.*.tmp"))
    assert tmp_siblings == []
