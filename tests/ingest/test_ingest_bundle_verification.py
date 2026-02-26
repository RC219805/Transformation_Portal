"""Evidence bundle build/verify tests for ingest batch outputs."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from transformation_portal.ingest.batch import BATCH_MANIFEST_FILENAME, run_ingest_batch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_INPUT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "ingest" / "batch_inputs"
TOOL_PATH = PROJECT_ROOT / "tools" / "build_ingest_evidence_bundle.py"


def _fake_ingest_payload_factory(input_path: Path) -> dict[str, Any]:
    file_bytes = input_path.read_bytes()
    source_name = input_path.name
    return {
        "schema_version": "1.0.1",
        "file_integrity": {
            "sha256": hashlib.sha256(file_bytes).hexdigest(),
            "size_bytes": len(file_bytes),
            "path": f"/bundle/{source_name}",
            "mime_type": "application/octet-stream",
        },
        "exif": {"all_tags": {"SourceFile": source_name}},
        "pipeline_config": {
            "config_sha256": hashlib.sha256(f"bundle:{source_name}".encode("utf-8")).hexdigest(),
            "preset": "bundle-validation",
        },
        "toolchain": [{"name": "python", "version": "volatile"}],
        "host": {
            "hostname": "volatile-host",
            "os": "Linux",
            "os_version": "volatile",
            "python_version": "volatile",
            "arch": "x86_64",
        },
        "timestamps": {
            "ingest_start": "volatile-start",
            "ingest_end": "volatile-end",
        },
        "git_commit": "e" * 40,
        "run_id": "volatile-run",
    }


def _run_bundle_tool(*args: str) -> subprocess.CompletedProcess[str]:
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


def test_ingest_evidence_bundle_verification_detects_tampering(tmp_path: Path) -> None:
    output_dir = tmp_path / "batch_run"
    manifest = run_ingest_batch(
        input_dir=FIXTURE_INPUT_DIR,
        output_dir=output_dir,
        profile="ingest_v1",
        ingest_payload_factory=_fake_ingest_payload_factory,
    )
    manifest_path = output_dir / BATCH_MANIFEST_FILENAME
    bundle_path = output_dir / "ingest_evidence_bundle.json"
    proof_target = manifest["items"][0]["relative_path"]

    build = _run_bundle_tool(
        "build",
        "--batch-manifest",
        str(manifest_path),
        "--out",
        str(bundle_path),
        "--proof-target",
        proof_target,
    )
    assert build.returncode == 0, build.stderr

    verify_ok = _run_bundle_tool(
        "verify",
        "--batch-manifest",
        str(manifest_path),
        "--bundle",
        str(bundle_path),
    )
    assert verify_ok.returncode == 0, verify_ok.stderr

    # Tamper one normalized artifact post-build; verification must fail deterministically.
    first_item_relpath = Path(manifest["items"][0]["normalized_json_relpath"])
    tampered_path = output_dir / first_item_relpath
    tampered_path.write_bytes(tampered_path.read_bytes() + b"\n")

    verify_tampered = _run_bundle_tool(
        "verify",
        "--batch-manifest",
        str(manifest_path),
        "--bundle",
        str(bundle_path),
    )
    assert verify_tampered.returncode == 6
    assert "Verification failed: digest mismatch for normalized artifact" in verify_tampered.stderr
