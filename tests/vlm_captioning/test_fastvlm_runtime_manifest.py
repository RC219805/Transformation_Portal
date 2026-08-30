from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
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
        "schema_version": "fastvlm-runtime.v2",
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
                "patch": {
                    "source": "ml_fastvlm",
                    "path": "model_export/fastvlm_mlx-vlm.patch",
                    "sha256": "1904693eb317ef476a2b13eef43c27f28e9e1529ce497b9b01a398332bdfccb8",
                    "patched_tree": "672677cd58a2760d7f8c6cf6b39fbb60940e7c30",
                },
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
            },
            "default": {
                "repo_id": "apple/FastVLM-1.5B-int8",
                "revision": "924716f32f1dbb29e8d2b62aac9010039ebc1ad7",
                "target_dir": "checkpoints/FastVLM-1.5B-int8",
                "required_files": [{"path": "config.json", "sha256": _digest(payload), "size_bytes": len(payload)}],
            },
            "review": {
                "repo_id": "apple/FastVLM-7B-int4",
                "revision": "1aeadbaaba011276f3dcda9582e5e64e2a90873a",
                "target_dir": "checkpoints/FastVLM-7B-int4",
                "required_files": [{"path": "config.json", "sha256": _digest(payload), "size_bytes": len(payload)}],
            },
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


def _git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _init_source_repo(path: Path, origin: str, files: dict[str, str]) -> str:
    path.mkdir(parents=True)
    _git(path, "init", "--quiet")
    _git(path, "remote", "add", "origin", origin)
    for relative_path, content in files.items():
        target = path / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _git(path, "add", "--all")
    _git(
        path,
        "-c",
        "user.name=FastVLM Test",
        "-c",
        "user.email=fastvlm-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        "fixture",
    )
    return _git(path, "rev-parse", "HEAD")


def _write_governed_source_fixture(tmp_path: Path, module: ModuleType) -> tuple[dict, Path, Path]:
    runtime_root = tmp_path / "fastvlm"
    mlx_vlm = runtime_root / "mlx-vlm"
    mlx_revision = _init_source_repo(
        mlx_vlm,
        "https://github.com/Blaizzy/mlx-vlm.git",
        {"mlx_vlm/__init__.py": "", "mlx_vlm/runtime.py": "MODE = 'base'\n"},
    )
    runtime_file = mlx_vlm / "mlx_vlm/runtime.py"
    runtime_file.write_text("MODE = 'patched'\n", encoding="utf-8")
    patch_payload = _git(mlx_vlm, "diff", "--binary", "HEAD") + "\n"
    runtime_file.write_text("MODE = 'base'\n", encoding="utf-8")
    assert _git(mlx_vlm, "status", "--porcelain") == ""

    ml_fastvlm = runtime_root / "ml-fastvlm"
    patch_relative_path = "model_export/fastvlm_mlx-vlm.patch"
    ml_revision = _init_source_repo(
        ml_fastvlm,
        "https://github.com/apple/ml-fastvlm.git",
        {patch_relative_path: patch_payload},
    )
    patch_path = ml_fastvlm / patch_relative_path
    _git(mlx_vlm, "apply", "--index", str(patch_path))

    manifest = _manifest(tmp_path)
    manifest["runtime_sources"]["ml_fastvlm"]["revision"] = ml_revision
    mlx_source = manifest["runtime_sources"]["mlx_vlm"]
    mlx_source["revision"] = mlx_revision
    mlx_source["patch"]["sha256"] = _digest(patch_path.read_bytes())
    mlx_source["patch"]["patched_tree"] = _git(mlx_vlm, "write-tree")
    return manifest, runtime_root, patch_path


def test_manifest_validator_rejects_unpinned_or_untrusted_manifest_values(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_runtime_manifest_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest = _manifest(tmp_path)
    manifest["runtime_sources"]["mlx_vlm"]["revision"] = "main"
    manifest["runtime_sources"]["mlx_vlm"]["target_dir"] = "alternate-mlx-vlm"
    manifest["runtime_sources"]["mlx_vlm"]["patch"]["source"] = "mlx_vlm"
    manifest["runtime_sources"]["mlx_vlm"]["patch"]["path"] = "../runtime.patch"
    manifest["runtime_sources"]["mlx_vlm"]["patch"]["sha256"] = "bad"
    manifest["runtime_sources"]["mlx_vlm"]["patch"]["patched_tree"] = "bad"
    manifest["models"]["smoke"]["repo_id"] = "attacker/model"
    manifest["models"]["smoke"]["target_dir"] = "../escape"
    manifest["models"]["smoke"]["required_files"][0]["sha256"] = "bad"

    errors = module.validate_manifest(manifest)

    assert any("revision must be a pinned 40-hex revision" in error for error in errors)
    assert any("target_dir must be mlx-vlm" in error for error in errors)
    assert any("patch.source must be ml_fastvlm" in error for error in errors)
    assert any("patch.path must be model_export/fastvlm_mlx-vlm.patch" in error for error in errors)
    assert any("patch.sha256 must be a SHA-256 hex digest" in error for error in errors)
    assert any("patch.patched_tree must be a 40-hex Git tree" in error for error in errors)
    assert any("repo_id is not allowlisted" in error for error in errors)
    assert any("Unsafe FastVLM manifest path" in error for error in errors)
    assert any("sha256 must be a SHA-256 hex digest" in error for error in errors)


def test_manifest_v1_remains_validation_compatible_but_cannot_authorize_sources(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_runtime_manifest_v1_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest = _manifest(tmp_path)
    manifest["schema_version"] = "fastvlm-runtime.v1"
    manifest["runtime_sources"]["ml_fastvlm"]["target_dir"] = "legacy-ml-fastvlm"
    manifest["runtime_sources"]["mlx_vlm"]["target_dir"] = "legacy-mlx-vlm"
    manifest["runtime_sources"]["mlx_vlm"].pop("patch")
    runtime_root = tmp_path / "runtime"
    _write_fixture_runtime(runtime_root)

    assert module.validate_manifest(manifest) == []
    assert (
        module.verify_runtime(
            manifest,
            roles=["smoke"],
            root=runtime_root,
            include_sources=False,
        )
        == []
    )
    with pytest.raises(module.ManifestError, match="source integrity requires fastvlm-runtime.v2"):
        module.verify_runtime_sources(manifest, root=runtime_root)


def test_manifest_v2_requires_governed_patch_and_rejects_unknown_schema(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_runtime_manifest_v2_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest = _manifest(tmp_path)
    manifest["runtime_sources"]["mlx_vlm"].pop("patch")

    errors = module.validate_manifest(manifest)

    assert "runtime_sources.mlx_vlm.patch must be an object" in errors
    manifest["schema_version"] = "fastvlm-runtime.v3"
    errors = module.validate_manifest(manifest)
    assert any("schema_version must be fastvlm-runtime.v1 or fastvlm-runtime.v2" in error for error in errors)


@pytest.mark.parametrize(
    ("mutate", "expected_errors"),
    [
        (
            lambda manifest: manifest["python"].update({"venv_dir": "mlx-vlm"}),
            ("python.venv_dir must be .venv-fastvlm", "runtime targets overlap"),
        ),
        (
            lambda manifest: manifest["models"]["smoke"].update({"target_dir": "ml-fastvlm"}),
            ("models.smoke.target_dir must be checkpoints/FastVLM-0.5B-fp16", "runtime targets overlap"),
        ),
        (
            lambda manifest: manifest["models"]["smoke"].update({"target_dir": ".venv-fastvlm/models"}),
            ("models.smoke.target_dir must be checkpoints/FastVLM-0.5B-fp16", "runtime targets overlap"),
        ),
    ],
)
def test_manifest_v2_rejects_noncanonical_or_overlapping_runtime_targets(
    tmp_path: Path,
    mutate,
    expected_errors: tuple[str, ...],
) -> None:
    module = _load_script_module("fastvlm_runtime_target_collision_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest = _manifest(tmp_path)
    mutate(manifest)

    errors = module.validate_manifest(manifest)

    for expected in expected_errors:
        assert any(expected in error for error in errors)


def test_manifest_v2_binds_runtime_root_and_required_model_roles(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_runtime_role_binding_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest = _manifest(tmp_path)
    manifest["runtime_root"] = "alternate/runtime"
    manifest["models"]["default"], manifest["models"]["review"] = (
        manifest["models"]["review"],
        manifest["models"]["default"],
    )

    errors = module.validate_manifest(manifest)

    assert "runtime_root must be .runtime/fastvlm" in errors
    assert "models.default.repo_id must be apple/FastVLM-1.5B-int8" in errors
    assert "models.review.repo_id must be apple/FastVLM-7B-int4" in errors

    manifest = _manifest(tmp_path)
    manifest["models"].pop("default")

    assert "models is missing required role(s): default" in module.validate_manifest(manifest)


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

    assert any("not a standalone Git checkout" in error for error in errors)


def test_runtime_source_verifier_accepts_only_the_governed_patched_tree(tmp_path: Path) -> None:
    module = _load_script_module(
        "fastvlm_runtime_manifest_patched_source_test",
        "scripts/validation/fastvlm_runtime_manifest.py",
    )
    manifest, runtime_root, patch_path = _write_governed_source_fixture(tmp_path, module)
    mlx_vlm = runtime_root / "mlx-vlm"

    assert module.validate_manifest(manifest) == []
    assert module.verify_runtime_sources(manifest, root=runtime_root) == []

    backdoor = mlx_vlm / "mlx_vlm/backdoor.py"
    backdoor.write_text("raise RuntimeError('unexpected source')\n", encoding="utf-8")
    errors = module.verify_runtime_sources(manifest, root=runtime_root)
    assert any("source tree does not match" in error for error in errors)
    backdoor.unlink()

    _git(mlx_vlm, "remote", "set-url", "origin", "https://example.invalid/mlx-vlm.git")
    errors = module.verify_runtime_sources(manifest, root=runtime_root)
    assert any("origin mismatch" in error for error in errors)
    _git(mlx_vlm, "remote", "set-url", "origin", "https://github.com/Blaizzy/mlx-vlm.git")

    patch_path.write_text(patch_path.read_text(encoding="utf-8") + "# tampered\n", encoding="utf-8")
    errors = module.verify_runtime_sources(manifest, root=runtime_root)
    assert any("patch digest mismatch" in error for error in errors)
    assert any("runtime source ml_fastvlm failed verification" in error for error in errors)


def test_fastvlm_installer_applies_and_verifies_manifest_pinned_patch() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    installer = (repo_root / "scripts/setup/install_fastvlm_runtime.sh").read_text(encoding="utf-8")
    source_installer = (repo_root / "scripts/setup/install_fastvlm_sources.py").read_text(encoding="utf-8")

    assert "install_fastvlm_sources.py" in installer
    assert "install_fastvlm_venv.py" in installer
    assert "run_fastvlm_install_locked.py" in installer
    assert "install_sources" in installer
    assert 'if [ "$SKIP_VERIFY" -eq 0 ]' in installer
    assert installer.rstrip().endswith("verify_sources_last")
    assert '["apply", "--index", str(patch_path)]' in source_installer
    assert '["write-tree"]' in source_installer
    assert "_promote_source_set" in source_installer


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
    monkeypatch.setattr(module, "audit_runtime_venv", lambda *_args, **_kwargs: None)

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


def test_fastvlm_import_smoke_includes_network_free_datasets_capability_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module(
        "fastvlm_runtime_manifest_datasets_smoke_test",
        "scripts/validation/fastvlm_runtime_manifest.py",
    )
    runtime_root = tmp_path / "fastvlm"
    manifest = _manifest(tmp_path)
    _write_fixture_runtime(runtime_root)
    smoke_sources: list[str] = []
    monkeypatch.setenv("PYTHONWARNINGS", "error")
    monkeypatch.setenv("VIRTUAL_ENV", str(tmp_path / "attacker-venv"))
    monkeypatch.setenv("__PYVENV_LAUNCHER__", str(tmp_path / "attacker-python"))

    def fake_run(args, **kwargs):  # noqa: ANN001
        smoke_sources.append(args[-1])
        assert kwargs["timeout"] == module.FASTVLM_IMPORT_SMOKE_TIMEOUT_SECONDS == 60
        assert kwargs["env"]["PYTHONDONTWRITEBYTECODE"] == "1"
        assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
        assert kwargs["env"]["PYTHONPATH"] == str(runtime_root / "mlx-vlm")
        assert "PYTHONWARNINGS" not in kwargs["env"]
        assert "VIRTUAL_ENV" not in kwargs["env"]
        assert "__PYVENV_LAUNCHER__" not in kwargs["env"]
        return module.subprocess.CompletedProcess(args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(module.subprocess, "run", fake_run)

    assert module.verify_python_imports(manifest, root=runtime_root) == []
    assert module.FASTVLM_RUNTIME_IMPORTS == ("datasets", "huggingface_hub", "mlx_vlm")
    assert len(smoke_sources) == 1
    smoke_source = smoke_sources[0]
    compile(smoke_source, "<fastvlm-import-smoke>", "exec")
    assert "Dataset.from_dict" in smoke_source
    assert ".map(" in smoke_source
    assert "keep_in_memory=True" in smoke_source
    assert "load_from_cache_file=False" in smoke_source
    assert "datasets API smoke:" in smoke_source
    assert "mlx_vlm import origin:" in smoke_source
    assert "load_dataset" not in smoke_source


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


def test_runtime_validator_starts_with_site_imports_disabled() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(repo_root / "scripts/validation/validate_fastvlm_runtime.py"),
            "--help",
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Validate the governed local FastVLM advisory captioning runtime" in completed.stdout


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
    module.audit_runtime_venv = lambda *_args, **_kwargs: None

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
    assert payload["checks"]["python_environment"]["status"] == "ready"
    assert payload["checks"]["models"]["smoke"]["status"] == "ready"
    assert payload["checks"]["python_imports"]["status"] == "skipped"


def test_runtime_validator_rejects_poisoned_venv_before_import_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validator = _load_script_module(
        "validate_fastvlm_runtime_poison_test",
        "scripts/validation/validate_fastvlm_runtime.py",
    )
    venv_builder = _load_script_module(
        "fastvlm_runtime_poison_venv_builder",
        "scripts/setup/install_fastvlm_venv.py",
    )
    manifest = _manifest(tmp_path)
    runtime_root = tmp_path / "fastvlm"
    _write_fixture_runtime(runtime_root)
    shutil.rmtree(runtime_root / ".venv-fastvlm")
    requirements = tmp_path / "empty-requirements.txt"
    requirements.write_text("", encoding="utf-8")
    venv_builder._build_staged_venv(
        Path(os.path.realpath(sys.executable)),
        runtime_root / ".venv-fastvlm",
        requirements,
    )
    marker = tmp_path / "poison-executed"
    site_packages = next((runtime_root / ".venv-fastvlm").glob("lib/python*/site-packages"))
    (site_packages / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    def fail_import(*_args, **_kwargs):  # noqa: ANN001
        raise AssertionError("poisoned venv must be rejected before import smoke")

    monkeypatch.setattr(validator, "verify_python_imports", fail_import)

    evidence = validator._runtime_evidence(
        manifest_path=tmp_path / "manifest.json",
        root=runtime_root,
        roles=["smoke"],
        manifest=manifest,
        include_sources=False,
        include_python=True,
        include_import_smoke=True,
    )

    assert evidence["runtime_status"] == "invalid"
    assert evidence["checks"]["python_environment"]["status"] == "failed"
    assert evidence["checks"]["python_imports"]["status"] == "skipped"
    assert not marker.exists()

    (site_packages / "sitecustomize.py").unlink()
    metadata = site_packages / "poisoned-1.0.dist-info/direct_url.json"
    metadata.parent.mkdir()
    metadata.write_text("{not-json", encoding="utf-8")

    malformed_evidence = validator._runtime_evidence(
        manifest_path=tmp_path / "manifest.json",
        root=runtime_root,
        roles=["smoke"],
        manifest=manifest,
        include_sources=False,
        include_python=True,
        include_import_smoke=True,
    )

    assert malformed_evidence["runtime_status"] == "invalid"
    assert malformed_evidence["errors"] == ["validation details redacted"]
    assert malformed_evidence["checks"]["python_environment"]["status"] == "failed"
    assert str(tmp_path) not in json.dumps(malformed_evidence)


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
