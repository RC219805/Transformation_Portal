"""CLI tests for tools/verify_run_card_attestation.py."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformation_portal.attestation.dsse import DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, pre_auth_encode
from transformation_portal.attestation.run_card_detached import (
    build_run_card_detached_attestation_payload,
    canonical_run_card_attestation_preimage_bytes,
    compute_run_card_attestation_sha256,
)
from transformation_portal.attestation.run_card_intoto import build_run_card_dsse_envelope, canonical_run_card_statement_bytes
from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root
from transformation_portal.lux_depth_v3.artifact_tree import build_artifact_tree

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "verify_run_card_attestation.py"
FAKE_GPG_PATH = PROJECT_ROOT / "tests" / "fixtures" / "attestation" / "fake_gpg.py"


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


def _write_run_card_v2(tmp_path: Path) -> Path:
    artifact_payloads = {
        "depth/image_01_depth.png": (b"depth-preview-fixture", "depth_u16_png"),
        "manifests/batch_01.json": (b'{"batch_id":"2026-04-10_120000"}\n', "batch_manifest"),
    }
    artifact_index = []
    for relative_path, (content, artifact_type) in artifact_payloads.items():
        artifact_path = tmp_path / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(content)
        artifact_index.append(
            {
                "artifact_type": artifact_type,
                "path": relative_path,
                "relative_path": relative_path,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    fingerprint_fields = {
        "model_variant": "METRIC_LARGE",
        "depth_quantization": "u16",
        "depth_device": "cpu",
        "preset": "premium",
        "v2_preset": "premium",
        "v2_device": "cpu",
        "v2_upscaler_backend": "realesrgan",
        "depth_pro_python_executable": None,
        "raw_python_executable": None,
        "da3_python_executable": None,
        "preset_requested": "premium",
        "preset_resolved": "premium",
        "backend_requested": "da3",
        "backend_resolved": "da3",
        "device_requested": "cpu",
        "device_resolved": "cpu",
        "quality_tier": "premium",
        "strict_inputs": False,
        "strict_segmentation": False,
        "apex_strict_mode": False,
    }
    canonical_json = json.dumps(fingerprint_fields, sort_keys=True, separators=(",", ":"))
    payload = {
        "run_card_version": "v2",
        "batch_id": "2026-04-10_120000",
        "start_time": "2026-04-10T12:00:00Z",
        "end_time": "2026-04-10T12:05:00Z",
        "config_fingerprint": {
            **fingerprint_fields,
            "hash_algorithm": "sha256",
            "canonical_json": canonical_json,
            "sha256": hashlib.sha256(canonical_json.encode("utf-8")).hexdigest(),
        },
        "backend_selection": {
            "requested": "da3",
            "resolved": "da3",
            "device": "cpu",
            "model_id": "depth-anything/DA3",
        },
        "backend_summary": {
            "requested_backend": "da3",
            "primary_backend": "da3",
            "final_backends_used": ["da3"],
            "fallback_images": 0,
            "semantic_fallback_images": 0,
            "operational_fallback_images": 0,
        },
        "environment": {
            "python_version": "3.11.9",
            "platform": "macOS-26.3-arm64-arm-64bit",
            "machine": "arm64",
        },
        "git_revision": {
            "v2": "d" * 40,
            "v3": "d" * 40,
        },
        "runtime_stats": {
            "count": 1,
            "total": 1.0,
            "mean": 1.0,
            "min": 1.0,
            "max": 1.0,
            "median": 1.0,
        },
        "outliers": [],
        "total_images": 1,
        "success_count": 1,
        "error_count": 0,
        "artifact_index": artifact_index,
        "artifact_tree": build_artifact_tree(artifact_index, include_proofs=True),
    }
    run_card_path = tmp_path / "run_card_2026-04-10_120000.json"
    run_card_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return run_card_path


def _write_run_card_v1(tmp_path: Path) -> Path:
    run_card_path = _write_run_card_v2(tmp_path)
    payload = json.loads(run_card_path.read_text(encoding="utf-8"))
    payload["run_card_version"] = "v1"
    payload.pop("artifact_tree", None)
    payload["artifact_merkle_root"] = compute_artifact_merkle_root(payload["artifact_index"])
    run_card_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return run_card_path


def _write_fake_gpg(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    gpg_path = bin_dir / "gpg"
    gpg_path.write_bytes(FAKE_GPG_PATH.read_bytes())
    gpg_path.chmod(0o755)
    return bin_dir


def _write_fake_cosign(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    cosign_path = bin_dir / "cosign"
    cosign_path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    cosign_path.chmod(0o755)
    return bin_dir


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


def _build_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    run_card_path = _write_run_card_v2(tmp_path)
    run_card_payload = json.loads(run_card_path.read_text(encoding="utf-8"))
    run_card_bytes = run_card_path.read_bytes()
    native_preimage = canonical_run_card_attestation_preimage_bytes(
        run_card_payload,
        run_card_bytes=run_card_bytes,
    )
    native_attestation = build_run_card_detached_attestation_payload(
        run_card_payload,
        run_card_bytes=run_card_bytes,
        signature={
            "algorithm": "openpgp-clearsign",
            "key_id": "test",
            "signature": _fake_clearsign(native_preimage),
        },
    )
    dsse_statement_bytes = canonical_run_card_statement_bytes(
        run_card_path=run_card_path,
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
    )
    dsse_attestation = build_run_card_dsse_envelope(
        run_card_path=run_card_path,
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
        key_id="test",
        signature_bytes=pre_auth_encode(DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, dsse_statement_bytes),
    )
    native_path = run_card_path.with_suffix(".attestation.native.json")
    dsse_path = run_card_path.with_suffix(".attestation.dsse.json")
    bundle_path = run_card_path.with_suffix(".attestation.dsse.sigstore.bundle.json")
    native_path.write_text(json.dumps(native_attestation), encoding="utf-8")
    dsse_path.write_text(json.dumps(dsse_attestation), encoding="utf-8")
    bundle_path.write_text(json.dumps({"verificationMaterial": {"tlogEntries": [{"logIndex": 1}]}}), encoding="utf-8")
    return run_card_path, native_path, dsse_path, bundle_path


def _build_inputs_v1(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    run_card_path = _write_run_card_v1(tmp_path)
    run_card_payload = json.loads(run_card_path.read_text(encoding="utf-8"))
    run_card_bytes = run_card_path.read_bytes()
    native_preimage = canonical_run_card_attestation_preimage_bytes(
        run_card_payload,
        run_card_bytes=run_card_bytes,
    )
    native_attestation = build_run_card_detached_attestation_payload(
        run_card_payload,
        run_card_bytes=run_card_bytes,
        signature={
            "algorithm": "openpgp-clearsign",
            "key_id": "test",
            "signature": _fake_clearsign(native_preimage),
        },
    )
    dsse_statement_bytes = canonical_run_card_statement_bytes(
        run_card_path=run_card_path,
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
    )
    dsse_attestation = build_run_card_dsse_envelope(
        run_card_path=run_card_path,
        run_card_payload=run_card_payload,
        run_card_bytes=run_card_bytes,
        key_id="test",
        signature_bytes=pre_auth_encode(DSSE_IN_TOTO_JSON_PAYLOAD_TYPE, dsse_statement_bytes),
    )
    native_path = run_card_path.with_suffix(".attestation.native.json")
    dsse_path = run_card_path.with_suffix(".attestation.dsse.json")
    bundle_path = run_card_path.with_suffix(".attestation.dsse.sigstore.bundle.json")
    native_path.write_text(json.dumps(native_attestation), encoding="utf-8")
    dsse_path.write_text(json.dumps(dsse_attestation), encoding="utf-8")
    bundle_path.write_text(json.dumps({"verificationMaterial": {"tlogEntries": [{"logIndex": 1}]}}), encoding="utf-8")
    return run_card_path, native_path, dsse_path, bundle_path


def test_verify_cli_succeeds_for_matching_run_card_attestations(tmp_path: Path) -> None:
    run_card_path, _, _, _ = _build_inputs(tmp_path)

    result = _run_tool("--run-card", str(run_card_path))

    assert result.returncode == 0, result.stderr


def test_verify_cli_succeeds_for_matching_v1_run_card_attestations(tmp_path: Path) -> None:
    run_card_path, _, _, _ = _build_inputs_v1(tmp_path)

    result = _run_tool("--run-card", str(run_card_path))

    assert result.returncode == 0, result.stderr


def test_verify_cli_rejects_native_attestation_self_hash_mismatch(tmp_path: Path) -> None:
    run_card_path, native_path, _, _ = _build_inputs(tmp_path)
    attestation = json.loads(native_path.read_text(encoding="utf-8"))
    attestation["claims"] = {"tampered": True}
    native_path.write_text(json.dumps(attestation), encoding="utf-8")

    result = _run_tool("--run-card", str(run_card_path), "--require-native")

    assert result.returncode == 5
    assert "attestation_sha256 mismatch" in result.stderr


def test_verify_cli_accepts_gpg_and_sigstore_bundle_when_helpers_exist(tmp_path: Path) -> None:
    run_card_path, _, _, _ = _build_inputs(tmp_path)
    fake_gpg = _write_fake_gpg(tmp_path)
    fake_cosign = _write_fake_cosign(tmp_path)
    path_prepend = tmp_path / "path"
    path_prepend.mkdir(parents=True, exist_ok=True)
    for helper in (fake_gpg / "gpg", fake_cosign / "cosign"):
        target = path_prepend / helper.name
        target.write_bytes(helper.read_bytes())
        target.chmod(0o755)

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--require-native",
        "--require-dsse",
        "--require-sigstore-bundle",
        "--gpg",
        path_prepend=path_prepend,
    )

    assert result.returncode == 0, result.stderr


def test_verify_cli_rejects_unrelated_native_clearsign_even_when_self_hash_is_allowed_missing(
    tmp_path: Path,
) -> None:
    run_card_path, native_path, dsse_path, _ = _build_inputs(tmp_path)
    dsse_path.unlink()
    native_attestation = json.loads(native_path.read_text(encoding="utf-8"))
    native_attestation["signature"]["signature"] = _fake_clearsign(b'{"schema":"unrelated"}')
    native_attestation["attestation_sha256"] = None
    native_path.write_text(json.dumps(native_attestation), encoding="utf-8")
    fake_gpg = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--require-native",
        "--allow-missing-attestation-sha",
        "--gpg",
        path_prepend=fake_gpg,
    )

    assert result.returncode == 5
    assert "does not match the expected canonical preimage bytes" in result.stderr


def test_verify_cli_rejects_native_recorded_key_mismatch(tmp_path: Path) -> None:
    run_card_path, _, dsse_path, _ = _build_inputs(tmp_path)
    dsse_path.unlink()
    fake_gpg = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--require-native",
        "--gpg",
        path_prepend=fake_gpg,
        env_updates={"TP_FAKE_GPG_RESOLVED_FINGERPRINT": "B" * 40},
    )

    assert result.returncode == 5
    assert "primary fingerprint does not match recorded key_id" in result.stderr


@pytest.mark.parametrize("status_mode", ["missing", "ambiguous"])
def test_verify_cli_rejects_native_missing_or_ambiguous_validsig(tmp_path: Path, status_mode: str) -> None:
    run_card_path, native_path, dsse_path, _ = _build_inputs(tmp_path)
    dsse_path.unlink()
    native_attestation = json.loads(native_path.read_text(encoding="utf-8"))
    native_attestation["attestation_sha256"] = compute_run_card_attestation_sha256(native_attestation)
    native_path.write_text(json.dumps(native_attestation), encoding="utf-8")
    fake_gpg = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--require-native",
        "--gpg",
        path_prepend=fake_gpg,
        env_updates={"TP_FAKE_GPG_STATUS_MODE": status_mode},
    )

    assert result.returncode == 5
    assert "exactly one VALIDSIG record" in result.stderr


def test_verify_cli_rejects_explicit_sigstore_bundle_without_dsse_attestation(tmp_path: Path) -> None:
    run_card_path, _, dsse_path, bundle_path = _build_inputs(tmp_path)
    dsse_path.unlink()

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--sigstore-bundle",
        str(bundle_path),
    )

    assert result.returncode == 5
    assert "Sigstore bundle" in result.stderr
    assert "DSSE attestation is missing" in result.stderr


def test_verify_cli_rejects_missing_explicit_native_attestation_path(tmp_path: Path) -> None:
    run_card_path, native_path, _, _ = _build_inputs(tmp_path)
    native_path.unlink()

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--native-attestation",
        str(native_path),
    )

    assert result.returncode == 5
    assert f"native detached attestation not found: {native_path}" in result.stderr


def test_verify_cli_rejects_missing_explicit_dsse_attestation_path(tmp_path: Path) -> None:
    run_card_path, _, dsse_path, _ = _build_inputs(tmp_path)
    dsse_path.unlink()

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--dsse-attestation",
        str(dsse_path),
    )

    assert result.returncode == 5
    assert f"DSSE attestation not found: {dsse_path}" in result.stderr


def test_verify_cli_rejects_missing_explicit_sigstore_bundle_path(tmp_path: Path) -> None:
    run_card_path, _, _, bundle_path = _build_inputs(tmp_path)
    bundle_path.unlink()

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--sigstore-bundle",
        str(bundle_path),
    )

    assert result.returncode == 5
    assert f"Sigstore bundle not found: {bundle_path}" in result.stderr


def test_verify_cli_rejects_dsse_release_assessment_missing_status(tmp_path: Path) -> None:
    run_card_path, _, dsse_path, _ = _build_inputs(tmp_path)
    dsse_attestation = json.loads(dsse_path.read_text(encoding="utf-8"))
    statement = json.loads(base64.b64decode(dsse_attestation["payload"]).decode("utf-8"))
    statement["predicate"]["release_assessment"] = {"sha256": "c" * 64}
    dsse_attestation["payload"] = base64.b64encode(
        json.dumps(statement, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).decode("ascii")
    dsse_path.write_text(json.dumps(dsse_attestation), encoding="utf-8")

    result = _run_tool("--run-card", str(run_card_path), "--require-dsse")

    assert result.returncode == 5
    assert "release_assessment.status is required" in result.stderr


def test_verify_cli_runs_from_source_checkout_without_pyproject_install(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    tool_path = repo_root / "tools" / "verify_run_card_attestation.py"
    tool_path.parent.mkdir(parents=True, exist_ok=True)
    tool_path.write_text(TOOL_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    package_roots = [
        repo_root / "src" / "transformation_portal",
        repo_root / "src" / "transformation_portal" / "attestation",
        repo_root / "src" / "transformation_portal" / "lux_depth_v3",
    ]
    for package_dir in package_roots:
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "__init__.py").write_text("", encoding="utf-8")

    (repo_root / "src" / "transformation_portal" / "attestation" / "dsse.py").write_text(
        'DSSE_IN_TOTO_JSON_PAYLOAD_TYPE = "application/test"\n'
        "def decode_dsse_payload(*_args, **_kwargs):\n"
        '    return b""\n'
        "def decode_dsse_signature_bytes(*_args, **_kwargs):\n"
        '    return b""\n'
        "def pre_auth_encode(*_args, **_kwargs):\n"
        '    return b""\n',
        encoding="utf-8",
    )
    (repo_root / "src" / "transformation_portal" / "attestation" / "gpg.py").write_text(
        "def gpg_verify_clearsign(*_args, **_kwargs):\n"
        "    return None\n"
        "def gpg_verify_detached_signature_bytes(*_args, **_kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "transformation_portal" / "attestation" / "run_card_detached.py").write_text(
        "def bind_run_card_detached_attestation(*_args, **_kwargs):\n"
        "    return None\n"
        "def canonical_run_card_attestation_preimage_bytes(*_args, **_kwargs):\n"
        '    return b""\n'
        "def validate_run_card_detached_attestation_surface(*_args, **_kwargs):\n"
        "    return None\n"
        "def verify_run_card_attestation_self_hash(*_args, **_kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "transformation_portal" / "attestation" / "run_card_intoto.py").write_text(
        "def decode_run_card_statement_from_envelope(*_args, **_kwargs):\n"
        "    return {}\n"
        "def validate_run_card_statement_binding(*_args, **_kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "transformation_portal" / "attestation" / "sigstore.py").write_text(
        "def cosign_verify_blob(*_args, **_kwargs):\n    return None\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "transformation_portal" / "lux_depth_v3" / "validators.py").write_text(
        "def verify_run_card_integrity(*_args, **_kwargs):\n    return []\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, str(tool_path), "--help"],
        cwd=repo_root,
        env={"PATH": "/usr/bin:/bin"},
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "--run-card" in result.stdout
