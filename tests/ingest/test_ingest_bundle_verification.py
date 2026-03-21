"""Evidence bundle build/verify tests for ingest batch outputs."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from transformation_portal.ingest.batch import BATCH_MANIFEST_FILENAME, run_ingest_batch

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_INPUT_DIR = PROJECT_ROOT / "tests" / "fixtures" / "ingest" / "batch_inputs"
TOOL_PATH = PROJECT_ROOT / "tools" / "build_ingest_evidence_bundle.py"


def _fake_ingest_payload_factory(input_path: Path) -> dict[str, Any]:
    file_bytes = input_path.read_bytes()
    source_name = input_path.name
    return {
        "schema_version": "1.0.2",
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


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _prepare_bundle(
    tmp_path: Path,
    *,
    include_proof: bool = True,
) -> tuple[dict[str, Any], Path, Path]:
    output_dir = tmp_path / "batch_run"
    manifest = run_ingest_batch(
        input_dir=FIXTURE_INPUT_DIR,
        output_dir=output_dir,
        profile="ingest_v1",
        ingest_payload_factory=_fake_ingest_payload_factory,
    )
    manifest_path = output_dir / BATCH_MANIFEST_FILENAME
    bundle_path = output_dir / "ingest_evidence_bundle.json"
    args = [
        "build",
        "--batch-manifest",
        str(manifest_path),
        "--out",
        str(bundle_path),
    ]
    if include_proof:
        args.extend(["--proof-target", manifest["items"][0]["relative_path"]])

    build = _run_bundle_tool(*args)
    assert build.returncode == 0, build.stderr

    return manifest, manifest_path, bundle_path


def test_ingest_evidence_bundle_verification_detects_tampering(tmp_path: Path) -> None:
    manifest, manifest_path, bundle_path = _prepare_bundle(tmp_path, include_proof=True)
    output_dir = manifest_path.parent

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


def test_build_fails_for_missing_manifest_required_fields(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema": "tp.ingest.batch_manifest.v1",
                "item_count": 0,
                "items": [],
            }
        ),
        encoding="utf-8",
    )
    bundle_path = tmp_path / "bundle.json"

    result = _run_bundle_tool(
        "build",
        "--batch-manifest",
        str(manifest_path),
        "--out",
        str(bundle_path),
    )

    assert result.returncode == 2
    assert "Input error: normalization_profile must be a non-empty string" in result.stderr


def test_build_fails_with_deterministic_exit_on_write_error(tmp_path: Path) -> None:
    _, manifest_path, _ = _prepare_bundle(tmp_path, include_proof=False)
    output_dir_path = tmp_path / "existing_dir"
    output_dir_path.mkdir(parents=True)

    result = _run_bundle_tool(
        "build",
        "--batch-manifest",
        str(manifest_path),
        "--out",
        str(output_dir_path),
    )

    assert result.returncode == 5
    assert "Build failed: unable to write evidence bundle" in result.stderr
    assert "Traceback" not in result.stderr


def test_verify_fails_when_bundle_items_tampered(tmp_path: Path) -> None:
    _, manifest_path, bundle_path = _prepare_bundle(tmp_path, include_proof=False)
    bundle = _load_json(bundle_path)
    bundle["items"][0]["normalized_json_sha256"] = "f" * 64
    _write_json(bundle_path, bundle)

    verify = _run_bundle_tool(
        "verify",
        "--batch-manifest",
        str(manifest_path),
        "--bundle",
        str(bundle_path),
    )

    assert verify.returncode == 6
    assert "Verification failed: bundle items mismatch" in verify.stderr


def test_verify_fails_when_inclusion_leaf_hash_tampered(tmp_path: Path) -> None:
    _, manifest_path, bundle_path = _prepare_bundle(tmp_path, include_proof=True)
    bundle = _load_json(bundle_path)
    inclusion = bundle.get("inclusion_proof")
    assert isinstance(inclusion, dict)
    inclusion["leaf_sha256"] = "f" * 64
    _write_json(bundle_path, bundle)

    verify = _run_bundle_tool(
        "verify",
        "--batch-manifest",
        str(manifest_path),
        "--bundle",
        str(bundle_path),
    )

    assert verify.returncode == 6
    assert "Verification failed: inclusion proof leaf hash mismatch" in verify.stderr


def test_verify_fails_when_bundle_metadata_fields_mismatch(tmp_path: Path) -> None:
    _, manifest_path, bundle_path = _prepare_bundle(tmp_path, include_proof=False)
    bundle = _load_json(bundle_path)
    bundle["normalization_profile"] = "ingest_v999"
    _write_json(bundle_path, bundle)

    verify_profile = _run_bundle_tool(
        "verify",
        "--batch-manifest",
        str(manifest_path),
        "--bundle",
        str(bundle_path),
    )
    assert verify_profile.returncode == 6
    assert "Verification failed: normalization_profile mismatch" in verify_profile.stderr

    bundle = _load_json(bundle_path)
    bundle["normalization_profile"] = "ingest_v1"
    bundle["batch_manifest_schema"] = "tp.ingest.batch_manifest.v999"
    _write_json(bundle_path, bundle)

    verify_schema = _run_bundle_tool(
        "verify",
        "--batch-manifest",
        str(manifest_path),
        "--bundle",
        str(bundle_path),
    )
    assert verify_schema.returncode == 6
    assert "Verification failed: batch_manifest_schema mismatch" in verify_schema.stderr
