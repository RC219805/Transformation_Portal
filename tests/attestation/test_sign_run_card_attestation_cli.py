"""CLI tests for tools/sign_run_card_attestation.py."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.artifact_manager import compute_artifact_merkle_root
from transformation_portal.lux_depth_v3.artifact_tree import build_artifact_tree

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "sign_run_card_attestation.py"
VERIFY_TOOL_PATH = PROJECT_ROOT / "tools" / "verify_run_card_attestation.py"
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
    cosign_path.write_text(
        """#!/bin/sh
bundle=""
prev=""
for arg in "$@"; do
  if [ "$prev" = "--bundle" ]; then
    bundle="$arg"
    break
  fi
  prev="$arg"
done
if [ -n "$bundle" ]; then
  cat <<'EOF' >"$bundle"
{"verificationMaterial":{"tlogEntries":[{"logIndex":1}]}}
EOF
fi
""",
        encoding="utf-8",
    )
    cosign_path.chmod(0o755)
    return bin_dir


def test_sign_cli_requires_gpg_backend(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)

    result = _run_tool("--run-card", str(run_card_path), "--key-id", "test-key")

    assert result.returncode == 4
    assert "No signing backend selected" in result.stderr


def test_sign_cli_writes_native_and_dsse_sidecars_with_fake_gpg(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)
    fake_path = _write_fake_gpg(tmp_path)

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--gpg",
        "--key-id",
        "test-key",
        path_prepend=fake_path,
    )

    native_path = run_card_path.with_suffix(".attestation.native.json")
    dsse_path = run_card_path.with_suffix(".attestation.dsse.json")
    assert result.returncode == 0, result.stderr
    assert native_path.exists()
    assert dsse_path.exists()

    verify_result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--require-native",
        "--require-dsse",
        "--gpg",
        path_prepend=fake_path,
        tool_path=VERIFY_TOOL_PATH,
    )
    assert verify_result.returncode == 0, verify_result.stderr


def test_sign_cli_rejects_native_signature_key_mismatch_before_writing(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)
    fake_path = _write_fake_gpg(tmp_path)
    native_path = run_card_path.with_suffix(".attestation.native.json")

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--format",
        "native",
        "--gpg",
        "--key-id",
        "logical-label",
        path_prepend=fake_path,
        env_updates={"TP_FAKE_GPG_RESOLVED_FINGERPRINT": "B" * 40},
    )

    assert result.returncode == 4
    assert "primary fingerprint does not match recorded key_id" in result.stderr
    assert not native_path.exists()


def test_sign_cli_supports_v1_run_cards_with_fake_gpg(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v1(tmp_path)
    fake_path = _write_fake_gpg(tmp_path)
    native_path = run_card_path.with_suffix(".attestation.native.json")
    dsse_path = run_card_path.with_suffix(".attestation.dsse.json")

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--gpg",
        "--key-id",
        "test-key",
        path_prepend=fake_path,
    )

    assert result.returncode == 0, result.stderr
    native_payload = json.loads(native_path.read_text(encoding="utf-8"))
    assert native_payload["subject"]["artifact_commitment"]["kind"] == "artifact_commitment_v1"

    native_attestation = json.loads(native_path.read_text(encoding="utf-8"))
    assert native_attestation["schema"] == "tp.run_card.attestation.detached.v1"
    assert native_attestation["signature"]["key_id"] == "test-key"

    dsse_attestation = json.loads(dsse_path.read_text(encoding="utf-8"))
    assert dsse_attestation["payloadType"] == "application/vnd.in-toto+json"
    assert dsse_attestation["signatures"][0]["keyid"] == "test-key"


def test_sign_cli_can_emit_sigstore_bundle_for_dsse_sidecar(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)
    fake_gpg = _write_fake_gpg(tmp_path)
    fake_cosign = _write_fake_cosign(tmp_path)
    path_prepend = tmp_path / "path"
    path_prepend.mkdir(parents=True, exist_ok=True)
    for helper in (fake_gpg / "gpg", fake_cosign / "cosign"):
        target = path_prepend / helper.name
        target.write_bytes(helper.read_bytes())
        target.chmod(0o755)

    bundle_path = run_card_path.with_suffix(".attestation.dsse.sigstore.bundle.json")
    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--gpg",
        "--key-id",
        "test-key",
        "--format",
        "dsse",
        "--sigstore-bundle-out",
        str(bundle_path),
        path_prepend=path_prepend,
    )

    assert result.returncode == 0, result.stderr
    assert bundle_path.exists()


def test_sign_cli_rejects_sigstore_bundle_path_for_native_only_output(tmp_path: Path) -> None:
    run_card_path = _write_run_card_v2(tmp_path)
    bundle_path = run_card_path.with_suffix(".attestation.dsse.sigstore.bundle.json")

    result = _run_tool(
        "--run-card",
        str(run_card_path),
        "--gpg",
        "--key-id",
        "test-key",
        "--format",
        "native",
        "--sigstore-bundle-out",
        str(bundle_path),
    )

    assert result.returncode == 2
    assert "--sigstore-bundle-out requires --format dsse or --format both" in result.stderr
    assert not bundle_path.exists()


def test_sign_cli_runs_from_source_checkout_without_pyproject_install(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    tool_path = repo_root / "tools" / "sign_run_card_attestation.py"
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
        'DSSE_IN_TOTO_JSON_PAYLOAD_TYPE = "application/test"\ndef pre_auth_encode(*_args, **_kwargs):\n    return b""\n',
        encoding="utf-8",
    )
    (repo_root / "src" / "transformation_portal" / "attestation" / "gpg.py").write_text(
        "def gpg_clearsign_bytes(*_args, **_kwargs):\n"
        '    return ""\n'
        "def gpg_detached_sign_bytes(*_args, **_kwargs):\n"
        '    return b""\n'
        "def gpg_verify_clearsign(*_args, **_kwargs):\n"
        "    return None\n",
        encoding="utf-8",
    )
    (repo_root / "src" / "transformation_portal" / "attestation" / "run_card_detached.py").write_text(
        "def build_run_card_detached_attestation_payload(*_args, **_kwargs):\n"
        "    return {}\n"
        "def canonical_run_card_attestation_bytes(*_args, **_kwargs):\n"
        '    return b""\n'
        "def canonical_run_card_attestation_preimage_bytes(*_args, **_kwargs):\n"
        '    return b""\n',
        encoding="utf-8",
    )
    (repo_root / "src" / "transformation_portal" / "attestation" / "run_card_intoto.py").write_text(
        "def build_run_card_dsse_envelope(*_args, **_kwargs):\n"
        "    return {}\n"
        "def canonical_run_card_statement_bytes(*_args, **_kwargs):\n"
        '    return b""\n',
        encoding="utf-8",
    )
    (repo_root / "src" / "transformation_portal" / "attestation" / "sigstore.py").write_text(
        "def cosign_sign_blob(*_args, **_kwargs):\n    return None\n",
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
