from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
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

    assert module.verify_runtime(manifest, roles=["smoke"], root=root, include_sources=False) == []

    os.remove(root / ".venv-fastvlm" / "bin" / "python")
    errors = module.verify_runtime(manifest, roles=["smoke"], root=root, include_sources=False)
    assert any("Python executable missing" in error for error in errors)


def test_runtime_verifier_rejects_unverifiable_source_checkout(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_runtime_manifest_source_test", "scripts/validation/fastvlm_runtime_manifest.py")
    root = tmp_path / "fastvlm"
    manifest = _manifest(tmp_path)
    _write_fixture_runtime(root)

    errors = module.verify_runtime_sources(manifest, root=root)

    assert any("not a verifiable git checkout" in error for error in errors)


def test_manifest_validator_rejects_unknown_runtime_sources(tmp_path: Path) -> None:
    module = _load_script_module(
        "fastvlm_runtime_manifest_extra_source_test", "scripts/validation/fastvlm_runtime_manifest.py"
    )
    manifest = _manifest(tmp_path)
    manifest["runtime_sources"]["unexpected"] = {
        "repo_url": "https://github.com/example/untrusted.git",
        "revision": "0" * 40,
        "target_dir": "unexpected",
    }

    errors = module.validate_manifest(manifest)

    assert any("runtime_sources contains unknown key(s): unexpected" in error for error in errors)


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


def test_downloader_import_does_not_mutate_sys_path() -> None:
    before = list(sys.path)

    _load_script_module("download_fastvlm_models_import_path_test", "scripts/setup/download_fastvlm_models.py")

    assert sys.path == before


def test_validate_fastvlm_runtime_verify_only_skips_import_smoke(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    validation_dir = str(repo_root / "scripts" / "validation")
    sys.path.insert(0, validation_dir)
    try:
        module = _load_script_module(
            "validate_fastvlm_runtime_verify_only_test", "scripts/validation/validate_fastvlm_runtime.py"
        )
    finally:
        sys.path.remove(validation_dir)
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(tmp_path))
    runtime_root = tmp_path / "fastvlm"
    _write_fixture_runtime(runtime_root)
    import_smoke_calls: list[bool] = []

    monkeypatch.setattr(module, "verify_python_imports", lambda *_args, **_kwargs: import_smoke_calls.append(True) or [])

    assert (
        module.main(
            [
                "--manifest",
                str(manifest_path),
                "--runtime-root",
                str(runtime_root),
                "--models",
                "smoke",
                "--skip-source-check",
            ]
        )
        == 0
    )
    assert import_smoke_calls == [True]

    import_smoke_calls.clear()
    assert (
        module.main(
            [
                "--manifest",
                str(manifest_path),
                "--runtime-root",
                str(runtime_root),
                "--models",
                "smoke",
                "--skip-source-check",
                "--verify-only",
            ]
        )
        == 0
    )
    assert not import_smoke_calls


def test_fastvlm_import_smoke_imports_modules_and_reports_metal_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module(
        "fastvlm_runtime_manifest_import_smoke_test",
        "scripts/validation/fastvlm_runtime_manifest.py",
    )
    runtime_root = tmp_path / "fastvlm"
    manifest = _manifest(tmp_path)
    _write_fixture_runtime(runtime_root)

    def fake_run(args, **kwargs):  # noqa: ANN001
        assert "importlib.import_module(name)" in args[-1]
        return module.subprocess.CompletedProcess(
            args,
            returncode=1,
            stdout="",
            stderr="mlx_vlm: RuntimeError: [metal::load_device] No Metal device available.",
        )

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    errors = module.verify_python_imports(manifest, root=runtime_root)

    assert len(errors) == 1
    assert "No Metal device available" in errors[0]


def test_fastvlm_import_smoke_reports_missing_python_without_spawning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module(
        "fastvlm_runtime_manifest_import_smoke_missing_python_test",
        "scripts/validation/fastvlm_runtime_manifest.py",
    )
    runtime_root = tmp_path / "fastvlm"
    manifest = _manifest(tmp_path)
    _write_fixture_runtime(runtime_root)
    (runtime_root / ".venv-fastvlm" / "bin" / "python").unlink()

    def fail_run(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("missing Python must be reported before subprocess.run")

    monkeypatch.setattr(module.subprocess, "run", fail_run)

    errors = module.verify_python_imports(manifest, root=runtime_root)

    assert len(errors) == 1
    assert "FastVLM Python executable missing" in errors[0]


def test_validate_fastvlm_runtime_json_reports_static_and_import_checks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    validation_dir = str(repo_root / "scripts" / "validation")
    sys.path.insert(0, validation_dir)
    try:
        module = _load_script_module("validate_fastvlm_runtime_json_test", "scripts/validation/validate_fastvlm_runtime.py")
    finally:
        sys.path.remove(validation_dir)
    manifest_path = tmp_path / "manifest.json"
    runtime_root = tmp_path / "fastvlm"
    _write_manifest(manifest_path, _manifest(tmp_path))
    _write_fixture_runtime(runtime_root)

    rc = module.main(
        [
            "--manifest",
            str(manifest_path),
            "--runtime-root",
            str(runtime_root),
            "--models",
            "smoke",
            "--skip-source-check",
            "--verify-only",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["runtime_status"] == "ready"
    assert payload["advisory_role"] == "advisory"
    assert payload["used_for_quality_gate"] is False
    assert payload["checks"]["manifest"]["status"] == "ready"
    assert payload["checks"]["runtime_sources"]["status"] == "skipped"
    assert payload["checks"]["python_executable"]["status"] == "ready"
    assert payload["checks"]["models"]["smoke"]["status"] == "ready"
    assert payload["checks"]["python_imports"]["status"] == "skipped"


def test_validate_fastvlm_runtime_redacts_human_and_json_error_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    validation_dir = str(repo_root / "scripts" / "validation")
    sys.path.insert(0, validation_dir)
    try:
        module = _load_script_module(
            "validate_fastvlm_runtime_redaction_test",
            "scripts/validation/validate_fastvlm_runtime.py",
        )
    finally:
        sys.path.remove(validation_dir)
    private_detail = "private-runtime-detail-that-must-not-appear"
    monkeypatch.setattr(module, "load_manifest", lambda _path: {})
    monkeypatch.setattr(module, "runtime_root", lambda *_args, **_kwargs: tmp_path / "fastvlm")
    monkeypatch.setattr(module, "selected_model_roles", lambda *_args, **_kwargs: ["smoke"])
    monkeypatch.setattr(
        module,
        "_runtime_evidence",
        lambda **_kwargs: {
            "errors": [private_detail],
            "checks": {},
        },
    )

    rc = module.main(["--manifest", str(tmp_path / "manifest.json")])

    captured = capsys.readouterr()
    assert rc == 1
    assert private_detail not in captured.err
    assert "runtime verification failed" in captured.err
    assert "details redacted" in captured.err.lower()

    def invalid_manifest(_path: Path) -> dict:
        raise module.ManifestError(private_detail)

    monkeypatch.setattr(module, "load_manifest", invalid_manifest)

    rc = module.main(["--manifest", str(tmp_path / "manifest.json")])

    captured = capsys.readouterr()
    assert rc == 2
    assert private_detail not in captured.err
    assert "manifest invalid" in captured.err
    assert "details redacted" in captured.err.lower()

    rc = module.main(["--manifest", str(tmp_path / "manifest.json"), "--json"])

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert rc == 2
    assert private_detail not in captured.out
    assert str(tmp_path) not in captured.out
    assert payload["manifest_path"] == "<redacted>"
    assert payload["runtime_root"] == "<redacted>"
    assert payload["errors"] == ["validation details redacted"]
    assert payload["checks"]["manifest"]["path"] == "<redacted>"
    assert payload["checks"]["manifest"]["errors"] == ["validation details redacted"]


def test_downloader_rejects_bad_hash_without_promoting_partial_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("download_fastvlm_models_bad_hash_test", "scripts/setup/download_fastvlm_models.py")
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(tmp_path, payload=b"expected"))

    def fake_snapshot_download(**kwargs):  # noqa: ANN001
        local_dir = Path(kwargs["local_dir"])
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_bytes(b"tampered")
        return str(local_dir)

    monkeypatch.setattr(module, "_import_snapshot_download", lambda: fake_snapshot_download)

    rc = module.main(["--manifest", str(manifest_path), "--runtime-root", str(tmp_path / "fastvlm"), "--models", "smoke"])

    assert rc == 1
    assert not (tmp_path / "fastvlm" / "checkpoints" / "FastVLM-0.5B-fp16").exists()
