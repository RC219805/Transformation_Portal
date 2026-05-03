from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit


def _load_script_module(module_name: str, relative_path: str) -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / relative_path
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _manifest(tmp_path: Path, payload: bytes = b"model-config") -> dict:
    return {
        "schema_version": "fastvlm-runtime.v1",
        "runtime_root": ".runtime/fastvlm",
        "python": {"venv_dir": ".venv-fastvlm"},
        "runtime_sources": {
            "ml_fastvlm": {
                "repo_url": "https://github.com/apple/ml-fastvlm.git",
                "revision": "592b4add3c1c8a518e77d95dc6248e76c1dd591f",
                "target_dir": "ml-fastvlm",
            },
            "mlx_vlm": {
                "repo_url": "https://github.com/Blaizzy/mlx-vlm.git",
                "revision": "1884b551bc741f26b2d54d68fa89d4e934b9a3de",
                "target_dir": "mlx-vlm",
            },
        },
        "models": {
            "smoke": {
                "repo_id": "apple/FastVLM-0.5B-fp16",
                "revision": "d241b8ae8acc23e319d79b1022fcba6a967046a3",
                "target_dir": "checkpoints/FastVLM-0.5B-fp16",
                "required_files": [
                    {
                        "path": "config.json",
                        "sha256": _digest(payload),
                        "size_bytes": len(payload),
                    }
                ],
            }
        },
    }


def _write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _write_fixture_runtime(root: Path, payload: bytes = b"model-config") -> None:
    (root / "ml-fastvlm").mkdir(parents=True)
    (root / "mlx-vlm").mkdir(parents=True)
    python_path = root / ".venv-fastvlm" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    python_path.chmod(0o755)
    model_dir = root / "checkpoints" / "FastVLM-0.5B-fp16"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_bytes(payload)


def test_manifest_validator_rejects_unpinned_or_untrusted_manifest_values(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_runtime_manifest_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest = _manifest(tmp_path)
    manifest["runtime_sources"]["mlx_vlm"]["revision"] = "main"
    manifest["models"]["smoke"]["repo_id"] = "attacker/model"
    manifest["models"]["smoke"]["target_dir"] = "../escape"
    manifest["models"]["smoke"]["required_files"][0]["sha256"] = "bad"

    errors = module.validate_manifest(manifest)

    assert any("revision must be a pinned 40-hex revision" in error for error in errors)
    assert any("repo_id is not allowlisted" in error for error in errors)
    assert any("Unsafe FastVLM manifest path" in error for error in errors)
    assert any("sha256 must be a SHA-256 hex digest" in error for error in errors)


def test_runtime_verifier_accepts_fixture_runtime_and_rejects_missing_python(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_runtime_manifest_verify_test", "scripts/validation/fastvlm_runtime_manifest.py")
    root = tmp_path / "fastvlm"
    manifest = _manifest(tmp_path)
    _write_fixture_runtime(root)

    assert module.verify_runtime(manifest, roles=["smoke"], root=root) == []

    os.remove(root / ".venv-fastvlm" / "bin" / "python")
    errors = module.verify_runtime(manifest, roles=["smoke"], root=root)
    assert any("Python executable missing" in error for error in errors)


def test_downloader_dry_run_uses_manifest_without_network(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    module = _load_script_module("download_fastvlm_models_dry_run_test", "scripts/setup/download_fastvlm_models.py")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(tmp_path))

    rc = module.main(
        ["--manifest", str(manifest_path), "--runtime-root", str(tmp_path / "fastvlm"), "--models", "smoke", "--dry-run"]
    )

    captured = capsys.readouterr()
    assert rc == 0
    assert "apple/FastVLM-0.5B-fp16" in captured.out
    assert not (tmp_path / "fastvlm" / "checkpoints").exists()


def test_downloader_rejects_bad_hash_without_promoting_partial_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("download_fastvlm_models_bad_hash_test", "scripts/setup/download_fastvlm_models.py")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(tmp_path, payload=b"expected"))

    def fake_snapshot_download(**kwargs):  # noqa: ANN001
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True)
        (local_dir / "config.json").write_bytes(b"tampered")
        return str(local_dir)

    monkeypatch.setattr(module, "_import_snapshot_download", lambda: fake_snapshot_download)

    rc = module.main(["--manifest", str(manifest_path), "--runtime-root", str(tmp_path / "fastvlm"), "--models", "smoke"])

    assert rc == 1
    assert not (tmp_path / "fastvlm" / "checkpoints" / "FastVLM-0.5B-fp16").exists()
