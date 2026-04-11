"""CLI tests for tools/sign_run_card_attestation.py."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from transformation_portal.lux_depth_v3.artifact_tree import build_artifact_tree

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TOOL_PATH = PROJECT_ROOT / "tools" / "sign_run_card_attestation.py"


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


def _write_run_card_v2(tmp_path: Path) -> Path:
    artifact_index = [
        {
            "artifact_type": "depth_u16_png",
            "path": "depth/image_01_depth.png",
            "relative_path": "depth/image_01_depth.png",
            "size_bytes": 1024,
            "sha256": "a" * 64,
        },
        {
            "artifact_type": "batch_manifest",
            "path": "manifests/batch_01.json",
            "relative_path": "manifests/batch_01.json",
            "size_bytes": 2048,
            "sha256": "b" * 64,
        },
    ]
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


def _write_fake_gpg(tmp_path: Path) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    gpg_path = bin_dir / "gpg"
    gpg_path.write_text(
        """#!/bin/sh
if printf '%s' "$*" | grep -q -- "--verify"; then
  exit 0
fi
cat >/dev/null
if printf '%s' "$*" | grep -q -- "--clearsign"; then
  cat <<'EOF'
-----BEGIN PGP SIGNED MESSAGE-----
Hash: SHA256

payload
-----BEGIN PGP SIGNATURE-----
fake-clearsign
-----END PGP SIGNATURE-----
EOF
  exit 0
fi
cat <<'EOF'
-----BEGIN PGP SIGNATURE-----
fake-detached
-----END PGP SIGNATURE-----
EOF
""",
        encoding="utf-8",
    )
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
