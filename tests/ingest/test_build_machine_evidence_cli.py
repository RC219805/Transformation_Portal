"""CLI tests for tools/build_machine_evidence.py."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "build_machine_evidence.py"


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


def test_build_machine_evidence_cli_emits_canonical_evidence_and_sha(tmp_path: Path) -> None:
    machine_payload = {
        "schema": "tp.meta.machine.v1",
        "command": "extract",
        "success": True,
        "exit_code": 0,
        "error": None,
        "data": {
            "input_path": "/tmp/source.cr2",
            "success": True,
            "output_path": "/tmp/source.provenance.json",
            "elapsed_seconds": 0.42,
            "preset": "luxury",
            "error": None,
        },
    }

    input_path = tmp_path / "machine.json"
    output_path = tmp_path / "evidence.json"
    input_path.write_text(json.dumps(machine_payload), encoding="utf-8")

    result = _run_tool("--in", str(input_path), "--out", str(output_path), "--emit-sha256")

    assert result.returncode == 0, result.stderr
    evidence_payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert evidence_payload["schema"] == "tp.meta.evidence.v1"
    assert evidence_payload["envelope_projection_profile"] == "tp.projection.machine_to_evidence.v1"
    assert evidence_payload["canonicalization"] == "tp.canonical.json.v1"
    assert "elapsed_seconds" not in evidence_payload["projected_envelope"]["data"]
    assert evidence_payload["evidence_sha256"] == result.stderr.strip().splitlines()[-1]


def test_build_machine_evidence_cli_rejects_non_object_input(tmp_path: Path) -> None:
    input_path = tmp_path / "bad.json"
    input_path.write_text("[]", encoding="utf-8")

    result = _run_tool("--in", str(input_path), "--out", str(tmp_path / "out.json"))

    assert result.returncode == 4
    assert "Evidence build failed: Input JSON must be an object" in result.stderr
