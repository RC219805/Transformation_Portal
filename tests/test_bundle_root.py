"""Tests for Phase 3.4 bundle root anchoring and notarization behavior."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = PROJECT_ROOT / "tools"
GENERATE_TOOL = PROJECT_ROOT / "tools" / "generate_evidence_bundle_manifest.py"
VERIFY_TOOL = PROJECT_ROOT / "tools" / "verify_evidence_bundle_manifest.py"
COMPUTE_ROOT_TOOL = PROJECT_ROOT / "tools" / "compute_bundle_root.py"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from bundle_root_fixture import EXPECTED_BUNDLE_ROOT_SHA256, write_bundle_fixture_artifacts  # noqa: E402

pytestmark = [pytest.mark.regression]

EXPECTED_GOLDEN_ROOT = EXPECTED_BUNDLE_ROOT_SHA256


def _run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )


def _write_bundle_artifacts(tmp_path: Path, *, timestamp_target: str = "signature") -> dict[str, Path]:
    return write_bundle_fixture_artifacts(tmp_path, timestamp_target=timestamp_target)


def _generate_manifest(artifacts: dict[str, Path], *, timestamp_target: str = "signature") -> subprocess.CompletedProcess[str]:
    return _run_cli(
        [
            sys.executable,
            str(GENERATE_TOOL),
            "--roots",
            str(artifacts["roots"]),
            "--hash-manifest",
            str(artifacts["hash_manifest"]),
            "--hash-summary",
            str(artifacts["hash_summary"]),
            "--signature",
            str(artifacts["signature"]),
            "--timestamp-target",
            timestamp_target,
            "--timestamp",
            str(artifacts["timestamp"]),
            "--out",
            str(artifacts["out"]),
        ]
    )


def _compute_root(manifest_path: Path, *, write: bool = False) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(COMPUTE_ROOT_TOOL),
        "--bundle-manifest",
        str(manifest_path),
    ]
    if write:
        command.append("--write")
    return _run_cli(command)


def _verify_bundle(manifest_path: Path, bundle_dir: Path) -> subprocess.CompletedProcess[str]:
    return _run_cli(
        [
            sys.executable,
            str(VERIFY_TOOL),
            "--bundle-manifest",
            str(manifest_path),
            "--bundle-dir",
            str(bundle_dir),
        ]
    )


def _load_manifest(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_bundle_root_is_deterministic(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path / "bundle")
    generated = _generate_manifest(artifacts, timestamp_target="signature")
    assert generated.returncode == 0, generated.stderr

    first = _compute_root(artifacts["out"])
    second = _compute_root(artifacts["out"])
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr

    first_root = first.stdout.strip()
    second_root = second.stdout.strip()
    assert first_root == second_root
    assert first_root == EXPECTED_GOLDEN_ROOT


def test_bundle_root_is_invariant_under_bundle_relocation(tmp_path: Path) -> None:
    artifacts_a = _write_bundle_artifacts(tmp_path / "layout_a" / "bundle")
    artifacts_b = _write_bundle_artifacts(tmp_path / "layout_b" / "nested" / "bundle")
    generated_a = _generate_manifest(artifacts_a, timestamp_target="signature")
    generated_b = _generate_manifest(artifacts_b, timestamp_target="signature")
    assert generated_a.returncode == 0, generated_a.stderr
    assert generated_b.returncode == 0, generated_b.stderr

    root_a = _compute_root(artifacts_a["out"])
    root_b = _compute_root(artifacts_b["out"])
    assert root_a.returncode == 0, root_a.stderr
    assert root_b.returncode == 0, root_b.stderr
    assert root_a.stdout.strip() == root_b.stdout.strip()


def test_verify_fails_when_bundle_root_digest_mismatches(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path / "bundle")
    generated = _generate_manifest(artifacts, timestamp_target="signature")
    assert generated.returncode == 0, generated.stderr

    written = _compute_root(artifacts["out"], write=True)
    assert written.returncode == 0, written.stderr

    manifest = _load_manifest(artifacts["out"])
    manifest["bundle_root_sha256"] = "0" * 64
    _write_manifest(artifacts["out"], manifest)

    verify_result = _verify_bundle(artifacts["out"], artifacts["out"].parent)
    assert verify_result.returncode == 11
    assert "bundle_root_sha256 mismatch" in verify_result.stdout


def test_verify_fails_when_notarization_artifact_digest_mismatches(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path / "bundle")
    generated = _generate_manifest(artifacts, timestamp_target="signature")
    assert generated.returncode == 0, generated.stderr

    written = _compute_root(artifacts["out"], write=True)
    assert written.returncode == 0, written.stderr

    notarization_path = artifacts["out"].parent / "bundle_root.tsr"
    notarization_path.write_bytes(b"\x30\x03\x30\x01\x7f")
    notarization_digest = hashlib.sha256(notarization_path.read_bytes()).hexdigest()

    manifest = _load_manifest(artifacts["out"])
    manifest["notarization"] = {
        "rfc3161": {
            "timestamp_path": notarization_path.name,
            "timestamp_sha256": notarization_digest,
        }
    }
    _write_manifest(artifacts["out"], manifest)

    verify_ok = _verify_bundle(artifacts["out"], artifacts["out"].parent)
    assert verify_ok.returncode == 0, verify_ok.stderr

    notarization_path.write_bytes(b"tampered")
    verify_fail = _verify_bundle(artifacts["out"], artifacts["out"].parent)
    assert verify_fail.returncode == 11
    assert "notarization.rfc3161.timestamp_path" in verify_fail.stdout


def test_compute_bundle_root_strict_mode_rejects_unknown_fields(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path / "bundle")
    generated = _generate_manifest(artifacts, timestamp_target="signature")
    assert generated.returncode == 0, generated.stderr

    manifest = _load_manifest(artifacts["out"])
    manifest["extra"] = "not-allowed"
    _write_manifest(artifacts["out"], manifest)

    compute_result = _compute_root(artifacts["out"])
    assert compute_result.returncode == 21
    assert "unexpected field(s): extra" in compute_result.stdout


def test_compute_bundle_root_returns_mismatch_when_existing_root_is_stale(tmp_path: Path) -> None:
    artifacts = _write_bundle_artifacts(tmp_path / "bundle")
    generated = _generate_manifest(artifacts, timestamp_target="signature")
    assert generated.returncode == 0, generated.stderr

    written = _compute_root(artifacts["out"], write=True)
    assert written.returncode == 0, written.stderr

    manifest = _load_manifest(artifacts["out"])
    manifest["phase3_version"] = "2"
    _write_manifest(artifacts["out"], manifest)

    compute_result = _compute_root(artifacts["out"])
    assert compute_result.returncode == 23
    assert "Bundle root mismatch" in compute_result.stdout
