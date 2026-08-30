from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import py_compile
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from tests.vlm_captioning.test_fastvlm_runtime_manifest import (
    _git,
    _init_source_repo,
    _manifest,
    _write_governed_source_fixture,
)

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


def _find_python_311() -> Path | None:
    candidates = [Path(sys.executable)]
    discovered = shutil.which("python3.11")
    if discovered:
        candidates.append(Path(discovered))
    for candidate in candidates:
        completed = subprocess.run(
            [str(candidate), "-I", "-S", "-c", "import sys; raise SystemExit(sys.version_info[:2] != (3, 11))"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if completed.returncode == 0:
            return candidate
    return None


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(
        repo,
        "-c",
        "user.name=FastVLM Test",
        "-c",
        "user.email=fastvlm-test@example.invalid",
        "commit",
        "--quiet",
        "-m",
        message,
    )
    return _git(repo, "rev-parse", "HEAD")


def _patch_payload(repo: Path, relative_path: str, base: str, patched: str) -> bytes:
    target = repo / relative_path
    target.write_text(patched, encoding="utf-8")
    payload = (_git(repo, "diff", "--binary", "HEAD") + "\n").encode()
    target.write_text(base, encoding="utf-8")
    assert _git(repo, "status", "--porcelain") == ""
    return payload


def _patched_tree(
    tmp_path: Path,
    *,
    mlx_repo: Path,
    revision: str,
    patch_path: Path,
    name: str,
) -> str:
    checkout = tmp_path / name
    subprocess.run(
        ["git", "clone", "--quiet", "--no-checkout", mlx_repo.as_uri(), str(checkout)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(checkout, "checkout", "--quiet", "--detach", revision)
    _git(checkout, "apply", "--index", str(patch_path))
    return _git(checkout, "write-tree")


def _local_source_contract(tmp_path: Path) -> tuple[dict, dict[str, Path | str]]:
    remote_root = tmp_path / "remotes"
    mlx_repo = remote_root / "mlx-vlm"
    base = "MODE = 'base-v1'\n"
    patched = "MODE = 'safe-v1'\n"
    mlx_revision = _init_source_repo(
        mlx_repo,
        "https://example.invalid/original-mlx.git",
        {
            ".gitignore": "__pycache__/\n*.py[cod]\n*.egg-info/\n",
            "mlx_vlm/__init__.py": "",
            "mlx_vlm/runtime.py": base,
        },
    )
    patch_payload = _patch_payload(mlx_repo, "mlx_vlm/runtime.py", base, patched)

    ml_repo = remote_root / "ml-fastvlm"
    patch_relative = "model_export/fastvlm_mlx-vlm.patch"
    ml_revision = _init_source_repo(
        ml_repo,
        "https://example.invalid/original-ml.git",
        {patch_relative: patch_payload.decode()},
    )
    patch_path = ml_repo / patch_relative
    tree = _patched_tree(
        tmp_path,
        mlx_repo=mlx_repo,
        revision=mlx_revision,
        patch_path=patch_path,
        name="tree-v1",
    )

    manifest = _manifest(tmp_path)
    manifest["runtime_sources"]["ml_fastvlm"].update({"repo_url": ml_repo.as_uri(), "revision": ml_revision})
    mlx_source = manifest["runtime_sources"]["mlx_vlm"]
    mlx_source.update({"repo_url": mlx_repo.as_uri(), "revision": mlx_revision})
    mlx_source["patch"].update(
        {
            "sha256": hashlib.sha256(patch_payload).hexdigest(),
            "patched_tree": tree,
        }
    )
    return manifest, {
        "ml_repo": ml_repo,
        "mlx_repo": mlx_repo,
        "base": base,
        "patched": patched,
        "patch_path": patch_path,
    }


def _upgrade_local_source_contract(tmp_path: Path, manifest: dict, state: dict[str, Path | str]) -> dict:
    mlx_repo = Path(state["mlx_repo"])
    ml_repo = Path(state["ml_repo"])
    patch_path = Path(state["patch_path"])
    base = "MODE = 'base-v2'\n"
    patched = "MODE = 'safe-v2'\n"
    runtime_file = mlx_repo / "mlx_vlm/runtime.py"
    runtime_file.write_text(base, encoding="utf-8")
    mlx_revision = _commit(mlx_repo, "upgrade base")
    patch_payload = _patch_payload(mlx_repo, "mlx_vlm/runtime.py", base, patched)
    patch_path.write_bytes(patch_payload)
    ml_revision = _commit(ml_repo, "upgrade governed patch")
    tree = _patched_tree(
        tmp_path,
        mlx_repo=mlx_repo,
        revision=mlx_revision,
        patch_path=patch_path,
        name="tree-v2",
    )
    upgraded = copy.deepcopy(manifest)
    upgraded["runtime_sources"]["ml_fastvlm"]["revision"] = ml_revision
    mlx_source = upgraded["runtime_sources"]["mlx_vlm"]
    mlx_source["revision"] = mlx_revision
    mlx_source["patch"]["sha256"] = hashlib.sha256(patch_payload).hexdigest()
    mlx_source["patch"]["patched_tree"] = tree
    return upgraded


def _allow_local_sources(module: ModuleType, manifest: dict, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("ml_fastvlm", "mlx_vlm"):
        repo_url = manifest["runtime_sources"][name]["repo_url"]
        monkeypatch.setitem(module.TRUSTED_RUNTIME_SOURCES, name, repo_url)
        monkeypatch.setitem(module._manifest_helpers.TRUSTED_RUNTIME_SOURCES, name, repo_url)


def test_verifier_rejects_core_worktree_redirection(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_core_worktree_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest, runtime_root, _patch_path = _write_governed_source_fixture(tmp_path, module)
    mlx_vlm = runtime_root / "mlx-vlm"
    external = tmp_path / "external-worktree"
    external.mkdir()
    _git(mlx_vlm, "config", "core.worktree", str(external))

    errors = module.verify_runtime_sources(manifest, root=runtime_root)

    assert any("unsafe key(s): core.worktree" in error for error in errors)


@pytest.mark.parametrize("environment_key", ["GIT_DIR", "GIT_WORK_TREE", "GIT_CONFIG_GLOBAL"])
def test_verifier_sanitizes_ambient_git_redirections(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    environment_key: str,
) -> None:
    module = _load_script_module(
        f"fastvlm_git_env_{environment_key.lower()}_test", "scripts/validation/fastvlm_runtime_manifest.py"
    )
    manifest, runtime_root, _patch_path = _write_governed_source_fixture(tmp_path, module)
    mlx_vlm = runtime_root / "mlx-vlm"
    redirect = tmp_path / "redirect"
    _init_source_repo(redirect, "https://example.invalid/redirect.git", {"safe.py": "SAFE = True\n"})
    configured_value = redirect / ".git" if environment_key == "GIT_DIR" else redirect
    if environment_key == "GIT_CONFIG_GLOBAL":
        configured_value = tmp_path / "attacker.gitconfig"
        configured_value.write_text(f"[core]\n\tworktree = {redirect}\n", encoding="utf-8")
    monkeypatch.setenv(environment_key, str(configured_value))
    (mlx_vlm / "mlx_vlm/ambient_backdoor.py").write_text("RAISE = True\n", encoding="utf-8")

    errors = module.verify_runtime_sources(manifest, root=runtime_root)

    assert any("ungoverned=mlx_vlm/ambient_backdoor.py" in error for error in errors)


def test_verifier_rejects_info_exclude_hidden_source(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_info_exclude_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest, runtime_root, _patch_path = _write_governed_source_fixture(tmp_path, module)
    mlx_vlm = runtime_root / "mlx-vlm"
    hidden = mlx_vlm / "mlx_vlm/hidden.py"
    hidden.write_text("VALUE = 'executable-untracked'\n", encoding="utf-8")
    (mlx_vlm / ".git/info/exclude").write_text("/mlx_vlm/hidden.py\n", encoding="utf-8")
    assert "hidden.py" not in _git(mlx_vlm, "status", "--porcelain=v1", "--untracked-files=all")

    errors = module.verify_runtime_sources(manifest, root=runtime_root)

    assert any("active local exclude rules" in error for error in errors)


def test_verifier_ignores_global_excludes_and_rejects_hidden_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_global_exclude_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest, runtime_root, _patch_path = _write_governed_source_fixture(tmp_path, module)
    mlx_vlm = runtime_root / "mlx-vlm"
    hidden = mlx_vlm / "mlx_vlm/global_hidden.py"
    hidden.write_text("VALUE = 'executable-untracked'\n", encoding="utf-8")
    exclude_file = tmp_path / "global-excludes"
    exclude_file.write_text("global_hidden.py\n", encoding="utf-8")
    global_config = tmp_path / "global.gitconfig"
    global_config.write_text(f"[core]\n\texcludesFile = {exclude_file}\n", encoding="utf-8")
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    assert "global_hidden.py" not in _git(mlx_vlm, "status", "--porcelain=v1", "--untracked-files=all")

    errors = module.verify_runtime_sources(manifest, root=runtime_root)

    assert any("ungoverned=mlx_vlm/global_hidden.py" in error for error in errors)


def test_verifier_rejects_executable_timestamp_valid_pyc(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_pyc_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest, runtime_root, _patch_path = _write_governed_source_fixture(tmp_path, module)
    mlx_vlm = runtime_root / "mlx-vlm"
    source = mlx_vlm / "mlx_vlm/runtime.py"
    safe_source = source.read_text(encoding="utf-8")
    malicious_source = safe_source.replace("patched", "poison!")
    assert len(malicious_source) == len(safe_source)
    fixed_time_ns = 1_700_000_000_000_000_000
    source.write_text(malicious_source, encoding="utf-8")
    os.utime(source, ns=(fixed_time_ns, fixed_time_ns))
    pyc_path = Path(importlib.util.cache_from_source(str(source)))
    py_compile.compile(str(source), cfile=str(pyc_path), doraise=True)
    source.write_text(safe_source, encoding="utf-8")
    os.utime(source, ns=(fixed_time_ns, fixed_time_ns))
    completed = subprocess.run(
        [sys.executable, "-c", "from mlx_vlm.runtime import MODE; print(MODE)"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(mlx_vlm), "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert completed.stdout.strip() == "poison!"

    errors = module.verify_runtime_sources(manifest, root=runtime_root)

    assert any("__pycache__" in error for error in errors)


@pytest.mark.parametrize(
    "relative_path",
    [
        "sitecustomize.py",
        "attack.pth",
        "mlx_vlm/native.so",
        "mlx_vlm/native.dylib",
        "mlx_vlm/native.pyd",
    ],
)
def test_verifier_rejects_untracked_importable_artifacts(tmp_path: Path, relative_path: str) -> None:
    module = _load_script_module("fastvlm_importable_artifact_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest, runtime_root, _patch_path = _write_governed_source_fixture(tmp_path, module)
    artifact = runtime_root / "mlx-vlm" / relative_path
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"untrusted executable artifact")

    errors = module.verify_runtime_sources(manifest, root=runtime_root)

    assert any(relative_path in error for error in errors)


def test_verifier_rejects_source_and_metadata_symlinks(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_symlink_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest, runtime_root, _patch_path = _write_governed_source_fixture(tmp_path, module)
    mlx_vlm = runtime_root / "mlx-vlm"
    outside = tmp_path / "outside.py"
    outside.write_text("VALUE = 'outside'\n", encoding="utf-8")
    (mlx_vlm / "mlx_vlm/symlinked.py").symlink_to(outside)
    errors = module.verify_runtime_sources(manifest, root=runtime_root)
    assert any("source contains symlink" in error for error in errors)
    (mlx_vlm / "mlx_vlm/symlinked.py").unlink()

    metadata = tmp_path / "external-git-metadata"
    (mlx_vlm / ".git").rename(metadata)
    (mlx_vlm / ".git").symlink_to(metadata, target_is_directory=True)
    errors = module.verify_runtime_sources(manifest, root=runtime_root)
    assert any(".git must be a real directory" in error for error in errors)


def test_verifier_rejects_fifo_git_metadata_without_hanging(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_fifo_metadata_test", "scripts/validation/fastvlm_runtime_manifest.py")
    manifest, runtime_root, _patch_path = _write_governed_source_fixture(tmp_path, module)
    fifo_path = runtime_root / "mlx-vlm/.git/refs/fifo-trap"
    os.mkfifo(fifo_path)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    helper_path = Path(__file__).resolve().parents[2] / "scripts/validation/fastvlm_runtime_manifest.py"
    probe = "\n".join(
        [
            "import importlib.util, json, pathlib, sys",
            "spec = importlib.util.spec_from_file_location('fastvlm_fifo_probe', sys.argv[1])",
            "module = importlib.util.module_from_spec(spec)",
            "spec.loader.exec_module(module)",
            "manifest = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding='utf-8'))",
            "print(json.dumps(module.verify_runtime_sources(manifest, root=pathlib.Path(sys.argv[3]))))",
        ]
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe, str(helper_path), str(manifest_path), str(runtime_root)],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )

    errors = json.loads(completed.stdout)
    assert any("unsupported filesystem entry" in error and "fifo-trap" in error for error in errors)


def test_secure_git_commands_have_a_bounded_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_script_module("fastvlm_git_timeout_test", "scripts/validation/fastvlm_runtime_manifest.py")
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/bin/sh\nsleep 5\n", encoding="utf-8")
    fake_git.chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}:{os.environ['PATH']}")
    monkeypatch.setattr(module, "GIT_SUBPROCESS_TIMEOUT_SECONDS", 0.1)
    started = time.monotonic()

    with pytest.raises(module.RuntimeVerificationError, match="timed out"):
        module.run_secure_git(["status"])

    assert time.monotonic() - started < 2


def test_source_installer_rejects_wrong_origin_before_staging_or_contact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_wrong_origin_install_test", "scripts/setup/install_fastvlm_sources.py")
    manifest, _state = _local_source_contract(tmp_path)
    _allow_local_sources(module, manifest, monkeypatch)
    runtime_root = tmp_path / "runtime"
    assert module.install_runtime_sources(manifest, root=runtime_root) == "installed"
    _git(runtime_root / "mlx-vlm", "remote", "set-url", "origin", "file:///attacker-controlled")
    stage_calls: list[bool] = []
    monkeypatch.setattr(module, "_stage_source_set", lambda *_args, **_kwargs: stage_calls.append(True))

    with pytest.raises(module.RuntimeVerificationError, match="origin mismatch"):
        module.install_runtime_sources(manifest, root=runtime_root)

    assert not stage_calls


def test_source_installer_rejects_legacy_v1_before_filesystem_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_v1_source_install_test", "scripts/setup/install_fastvlm_sources.py")
    manifest, _state = _local_source_contract(tmp_path)
    _allow_local_sources(module, manifest, monkeypatch)
    manifest["schema_version"] = "fastvlm-runtime.v1"
    manifest["runtime_sources"]["mlx_vlm"].pop("patch")
    runtime_root = tmp_path / "missing-runtime"
    git_calls: list[bool] = []
    monkeypatch.setattr(module, "run_secure_git", lambda *_args, **_kwargs: git_calls.append(True))

    with pytest.raises(module.ManifestError, match="source integrity requires fastvlm-runtime.v2"):
        module.install_runtime_sources(manifest, root=runtime_root)

    assert not git_calls
    assert not runtime_root.exists()


def test_source_installer_rejects_target_collisions_before_filesystem_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_colliding_source_install_test", "scripts/setup/install_fastvlm_sources.py")
    manifest, _state = _local_source_contract(tmp_path)
    manifest["models"]["smoke"]["target_dir"] = "ml-fastvlm"
    runtime_root = tmp_path / "missing-runtime"
    stage_calls: list[bool] = []
    monkeypatch.setattr(module, "_stage_source_set", lambda *_args, **_kwargs: stage_calls.append(True))

    with pytest.raises(module.ManifestError, match="runtime targets overlap"):
        module.install_runtime_sources(manifest, root=runtime_root)

    assert not stage_calls
    assert not runtime_root.exists()


def test_source_installer_dry_run_does_not_log_absolute_runtime_targets(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module = _load_script_module("fastvlm_redacted_source_plan_test", "scripts/setup/install_fastvlm_sources.py")
    manifest = _manifest(tmp_path)
    runtime_root = tmp_path / "private-runtime-root"

    assert module.install_runtime_sources(manifest, root=runtime_root, dry_run=True) == "dry-run"

    output = capsys.readouterr().out
    assert str(runtime_root) not in output
    assert output.strip() == "[dry-run] governed source plan validated"


def test_source_installer_rejects_hooks_without_execution_or_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_hook_install_test", "scripts/setup/install_fastvlm_sources.py")
    manifest, _state = _local_source_contract(tmp_path)
    _allow_local_sources(module, manifest, monkeypatch)
    runtime_root = tmp_path / "runtime"
    assert module.install_runtime_sources(manifest, root=runtime_root) == "installed"
    marker = tmp_path / "hook-executed"
    hook = runtime_root / "mlx-vlm/.git/hooks/post-checkout"
    hook.parent.mkdir(parents=True, exist_ok=True)
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n", encoding="utf-8")
    hook.chmod(0o755)
    stage_calls: list[bool] = []
    monkeypatch.setattr(module, "_stage_source_set", lambda *_args, **_kwargs: stage_calls.append(True))

    with pytest.raises(module.RuntimeVerificationError, match="active hooks"):
        module.install_runtime_sources(manifest, root=runtime_root)

    assert not marker.exists()
    assert not stage_calls


def test_source_installer_rejects_symlinked_target_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_symlink_install_test", "scripts/setup/install_fastvlm_sources.py")
    manifest, _state = _local_source_contract(tmp_path)
    _allow_local_sources(module, manifest, monkeypatch)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    outside = tmp_path / "outside-source"
    outside.mkdir()
    (runtime_root / "mlx-vlm").symlink_to(outside, target_is_directory=True)
    stage_calls: list[bool] = []
    monkeypatch.setattr(module, "_stage_source_set", lambda *_args, **_kwargs: stage_calls.append(True))

    with pytest.raises((module.ManifestError, module.RuntimeVerificationError), match="escapes runtime root|real directory"):
        module.install_runtime_sources(manifest, root=runtime_root)

    assert not stage_calls


def test_source_installer_is_idempotent_and_supports_patched_revision_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_upgrade_install_test", "scripts/setup/install_fastvlm_sources.py")
    manifest, state = _local_source_contract(tmp_path)
    _allow_local_sources(module, manifest, monkeypatch)
    runtime_root = tmp_path / "runtime"

    assert module.install_runtime_sources(manifest, root=runtime_root) == "installed"
    original_stage = module._stage_source_set
    monkeypatch.setattr(
        module,
        "_stage_source_set",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("idempotent rerun contacted source")),
    )
    assert module.install_runtime_sources(manifest, root=runtime_root) == "ready"
    monkeypatch.setattr(module, "_stage_source_set", original_stage)

    ignored_artifact = runtime_root / "mlx-vlm/mlx_vlm/__pycache__/runtime.pyc"
    ignored_artifact.parent.mkdir(parents=True)
    ignored_artifact.write_bytes(b"untrusted bytecode")
    assert module.install_runtime_sources(manifest, root=runtime_root) == "installed"
    assert not ignored_artifact.exists()

    upgraded = _upgrade_local_source_contract(tmp_path, manifest, state)
    assert module.install_runtime_sources(upgraded, root=runtime_root) == "installed"
    assert module.verify_runtime_sources(upgraded, root=runtime_root) == []
    assert (runtime_root / "mlx-vlm/mlx_vlm/runtime.py").read_text(encoding="utf-8") == "MODE = 'safe-v2'\n"


def test_source_promotion_rolls_back_both_repositories_on_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_partial_promotion_test", "scripts/setup/install_fastvlm_sources.py")
    manifest, state = _local_source_contract(tmp_path)
    _allow_local_sources(module, manifest, monkeypatch)
    runtime_root = tmp_path / "runtime"
    assert module.install_runtime_sources(manifest, root=runtime_root) == "installed"
    upgraded = _upgrade_local_source_contract(tmp_path, manifest, state)
    stage_root = tmp_path / "staged-upgrade"
    stage_root.mkdir()
    module._stage_source_set(upgraded, stage_root)
    replace_calls = 0
    real_replace = module._replace_path

    def fail_during_second_promotion(source: Path, destination: Path) -> None:
        nonlocal replace_calls
        replace_calls += 1
        if replace_calls == 4:
            raise OSError("injected partial promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(module, "_replace_path", fail_during_second_promotion)

    with pytest.raises(OSError, match="injected partial promotion failure"):
        module._promote_source_set(upgraded, stage_root=stage_root, runtime=runtime_root)

    assert module.verify_runtime_sources(manifest, root=runtime_root) == []
    assert (runtime_root / "mlx-vlm/mlx_vlm/runtime.py").read_text(encoding="utf-8") == "MODE = 'safe-v1'\n"


def test_fresh_venv_rebuild_never_executes_persistent_sitecustomize(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_clean_venv_test", "scripts/setup/install_fastvlm_venv.py")
    manifest = _manifest(tmp_path)
    runtime_root = tmp_path / "runtime"
    old_venv = runtime_root / ".venv-fastvlm"
    runtime_root.mkdir()
    subprocess.run(
        [sys.executable, "-I", "-m", "venv", str(old_venv)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    site_packages = next(old_venv.glob("lib/python*/site-packages"))
    poison_marker = tmp_path / "old-venv-executed"
    (site_packages / "sitecustomize.py").write_text(
        "from pathlib import Path\n" f"Path({str(poison_marker)!r}).write_text('poisoned', encoding='utf-8')\n",
        encoding="utf-8",
    )
    requirements = tmp_path / "empty-requirements.txt"
    requirements.write_text("", encoding="utf-8")

    assert (
        module.prepare_runtime_venv(
            manifest,
            root=runtime_root,
            base_python=Path(sys.executable),
            requirements=requirements,
        )
        == "installed"
    )

    assert not poison_marker.exists()
    assert not list(old_venv.rglob("sitecustomize.py"))
    assert not list(old_venv.rglob("*.pth"))
    module.audit_runtime_venv(old_venv)
    assert module.audit_installed_runtime_venv(manifest, root=runtime_root) == "verified"
    evidence = runtime_root / "fastvlm-pip-freeze.txt"
    assert evidence.is_file() and not evidence.is_symlink()


def test_python_311_staged_venv_removes_bootstrap_setuptools_pth(tmp_path: Path) -> None:
    python_311 = _find_python_311()
    if python_311 is None:
        pytest.skip("Python 3.11 is not available")

    module = _load_script_module("fastvlm_python_311_venv_test", "scripts/setup/install_fastvlm_venv.py")
    requirements = tmp_path / "empty-requirements.txt"
    requirements.write_text("", encoding="utf-8")
    staged_venv = tmp_path / "venv"

    freeze = module._build_staged_venv(python_311, staged_venv, requirements)

    assert b"setuptools==" not in freeze.lower()
    assert not list(staged_venv.rglob("*.pth"))
    module.audit_runtime_venv(staged_venv, expected_base_python=python_311)


def test_venv_install_removes_only_canonical_root_lib64_alias(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_lib64_alias_test", "scripts/setup/install_fastvlm_venv.py")
    staged_venv = tmp_path / "venv"
    (staged_venv / "lib").mkdir(parents=True)
    alias = staged_venv / "lib64"
    alias.symlink_to("lib", target_is_directory=True)

    module._remove_canonical_lib64_alias(staged_venv)

    assert not alias.exists()
    assert not alias.is_symlink()


@pytest.mark.parametrize("target", ["../external", "lib/site-packages"])
def test_venv_install_rejects_noncanonical_root_lib64_alias(tmp_path: Path, target: str) -> None:
    module = _load_script_module("fastvlm_bad_lib64_alias_test", "scripts/setup/install_fastvlm_venv.py")
    staged_venv = tmp_path / "venv"
    (staged_venv / "lib/site-packages").mkdir(parents=True)
    (tmp_path / "external").mkdir()
    alias = staged_venv / "lib64"
    alias.symlink_to(target, target_is_directory=True)

    with pytest.raises(module.RuntimeVerificationError, match="lib64 alias must point exactly"):
        module._remove_canonical_lib64_alias(staged_venv)

    assert alias.is_symlink()


def test_venv_install_removes_allowlisted_setuptools_pth_after_locked_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_locked_setuptools_pth_test", "scripts/setup/install_fastvlm_venv.py")
    requirements = tmp_path / "empty-requirements.txt"
    requirements.write_text("", encoding="utf-8")
    staged_venv = tmp_path / "venv"
    real_run_checked = module._run_checked

    def recreate_setuptools_pth(command, **kwargs):  # noqa: ANN001
        result = real_run_checked(command, **kwargs)
        if command[2:4] == ["-m", "pip"] and "install" in command:
            site_packages = next(staged_venv.glob("lib/python*/site-packages"))
            (site_packages / "distutils-precedence.pth").write_bytes(next(iter(module._ALLOWED_BOOTSTRAP_PTH_PAYLOADS)))
        return result

    monkeypatch.setattr(module, "_run_checked", recreate_setuptools_pth)

    module._build_staged_venv(Path(os.path.realpath(sys.executable)), staged_venv, requirements)

    assert not list(staged_venv.rglob("*.pth"))
    module.audit_runtime_venv(staged_venv)


def test_venv_install_strips_ambient_pip_policy_and_uses_isolated_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_pip_isolation_test", "scripts/setup/install_fastvlm_venv.py")
    monkeypatch.setenv("PIP_INDEX_URL", "https://attacker.invalid/simple")
    monkeypatch.setenv("PIP_EXTRA_INDEX_URL", "https://attacker.invalid/extra")
    monkeypatch.setenv("PIP_TARGET", str(tmp_path / "attacker-target"))
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "attacker-python"))

    environment = module._isolated_python_environment()

    assert environment["PIP_CONFIG_FILE"] == os.devnull
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert "PIP_INDEX_URL" not in environment
    assert "PIP_EXTRA_INDEX_URL" not in environment
    assert "PIP_TARGET" not in environment
    assert environment.get("PYTHONPATH") is None

    requirements = tmp_path / "empty-requirements.txt"
    requirements.write_text("", encoding="utf-8")
    staged_venv = tmp_path / "venv"
    commands: list[list[str]] = []
    real_run_checked = module._run_checked

    def record_run_checked(command, **kwargs):  # noqa: ANN001
        commands.append(list(command))
        return real_run_checked(command, **kwargs)

    monkeypatch.setattr(module, "_run_checked", record_run_checked)
    module._build_staged_venv(Path(os.path.realpath(sys.executable)), staged_venv, requirements)

    pip_commands = [command for command in commands if command[2:4] == ["-m", "pip"]]
    assert pip_commands
    assert all(command[4] == "--isolated" for command in pip_commands)
    install_command = next(command for command in pip_commands if "install" in command)
    assert "--no-deps" in install_command
    assert "--only-binary=:all:" in install_command
    assert any(command[-1] == "check" for command in pip_commands)


def test_staged_venv_stops_before_freeze_when_pip_check_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_pip_check_failure_test", "scripts/setup/install_fastvlm_venv.py")
    requirements = tmp_path / "empty-requirements.txt"
    requirements.write_text("", encoding="utf-8")
    staged_venv = tmp_path / "venv"
    commands: list[list[str]] = []
    real_run_checked = module._run_checked

    def fail_pip_check(command, **kwargs):  # noqa: ANN001
        commands.append(list(command))
        if command[-1] == "check":
            raise module.RuntimeVerificationError("FastVLM dependency consistency check failed")
        return real_run_checked(command, **kwargs)

    monkeypatch.setattr(module, "_run_checked", fail_pip_check)

    with pytest.raises(module.RuntimeVerificationError, match="dependency consistency check failed"):
        module._build_staged_venv(Path(os.path.realpath(sys.executable)), staged_venv, requirements)

    assert any(command[-1] == "check" for command in commands)
    assert not any(command[-2:] == ["freeze", "--all"] for command in commands)


@pytest.mark.parametrize(
    "relative_path",
    [
        "lib/python3.11/site-packages/sitecustomize.pyc",
        "lib/python3.11/site-packages/sitecustomize.cpython-311-darwin.so",
        "lib/python3.11/site-packages/usercustomize.pyd",
        "bin/python._pth",
    ],
)
def test_venv_audit_rejects_import_equivalent_startup_controls(tmp_path: Path, relative_path: str) -> None:
    module = _load_script_module("fastvlm_startup_variant_test", "scripts/setup/install_fastvlm_venv.py")
    venv = tmp_path / "venv"
    artifact = venv / relative_path
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"untrusted")

    with pytest.raises(module.RuntimeVerificationError, match="startup|path-control"):
        module.audit_runtime_venv(venv)


@pytest.mark.parametrize("module_name", ["sitecustomize", "usercustomize"])
def test_venv_audit_rejects_startup_module_packages(tmp_path: Path, module_name: str) -> None:
    module = _load_script_module("fastvlm_startup_package_test", "scripts/setup/install_fastvlm_venv.py")
    package = tmp_path / "venv/lib/python3.11/site-packages" / module_name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("raise RuntimeError('executed')\n", encoding="utf-8")

    with pytest.raises(module.RuntimeVerificationError, match="prohibited startup module"):
        module.audit_runtime_venv(tmp_path / "venv")


@pytest.mark.parametrize("symlink_path", ["lib", "bin/python"])
def test_venv_audit_rejects_ancestor_and_launcher_symlinks(tmp_path: Path, symlink_path: str) -> None:
    module = _load_script_module("fastvlm_venv_symlink_test", "scripts/setup/install_fastvlm_venv.py")
    venv = tmp_path / "venv"
    external = tmp_path / "external"
    external.mkdir()
    link = venv / symlink_path
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(external, target_is_directory=True)

    with pytest.raises(module.RuntimeVerificationError, match="must not contain symlinks"):
        module.audit_runtime_venv(venv)


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("include-system-site-packages", "true", "include-system-site-packages"),
        ("home", "/tmp/untrusted-python", "home does not match"),
        ("executable", "/tmp/untrusted-python", "executable does not match"),
    ],
)
def test_venv_audit_rejects_mutated_pyvenv_controls(
    tmp_path: Path,
    key: str,
    value: str,
    message: str,
) -> None:
    module = _load_script_module("fastvlm_pyvenv_control_test", "scripts/setup/install_fastvlm_venv.py")
    venv = tmp_path / "venv"
    subprocess.run(
        [os.path.realpath(sys.executable), "-I", "-S", "-m", "venv", "--copies", str(venv)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    config = venv / "pyvenv.cfg"
    lines = config.read_text(encoding="utf-8").splitlines()
    config.write_text(
        "\n".join(f"{key} = {value}" if line.lower().startswith(f"{key} =") else line for line in lines) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(module.RuntimeVerificationError, match=message):
        module.audit_runtime_venv(venv, expected_base_python=Path(sys.executable))


@pytest.mark.parametrize(
    ("relative_path", "payload", "symlink"),
    [
        ("attack.pth", "import os\n", False),
        ("distutils-precedence.pth", "import os\n", False),
        ("distutils-precedence.pth", "", True),
        ("sitecustomize.py", "raise RuntimeError('startup hook executed')\n", False),
        (
            "package-1.0.dist-info/direct_url.json",
            '{"url":"file:///tmp/attack","dir_info":{"editable":true}}',
            False,
        ),
    ],
)
def test_staged_venv_rejects_untrusted_startup_before_stage_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    relative_path: str,
    payload: str,
    symlink: bool,
) -> None:
    module = _load_script_module("fastvlm_pre_execution_venv_test", "scripts/setup/install_fastvlm_venv.py")
    requirements = tmp_path / "empty-requirements.txt"
    requirements.write_text("", encoding="utf-8")
    staged_venv = tmp_path / "venv"
    base_python = tmp_path / "base-python"
    commands: list[list[str]] = []

    def fake_run_checked(command, **_kwargs):  # noqa: ANN001
        commands.append(list(command))
        if len(commands) > 1:
            pytest.fail("staged Python executed before the untrusted startup artifact was rejected")
        artifact = staged_venv / "lib/python3.11/site-packages" / relative_path
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if symlink:
            target = tmp_path / "allowlisted-bootstrap-payload.pth"
            target.write_bytes(next(iter(module._ALLOWED_BOOTSTRAP_PTH_PAYLOADS)))
            artifact.symlink_to(target)
        else:
            artifact.write_text(payload, encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "_run_checked", fake_run_checked)

    with pytest.raises(module.RuntimeVerificationError):
        module._build_staged_venv(base_python, staged_venv, requirements)

    assert commands == [[str(base_python), "-I", "-S", "-m", "venv", "--copies", str(staged_venv)]]


def test_staged_venv_rejects_sitecustomize_package_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_pre_execution_package_test", "scripts/setup/install_fastvlm_venv.py")
    requirements = tmp_path / "empty-requirements.txt"
    requirements.write_text("", encoding="utf-8")
    staged_venv = tmp_path / "venv"
    base_python = tmp_path / "base-python"
    marker = tmp_path / "startup-executed"
    commands: list[list[str]] = []

    def fake_run_checked(command, **_kwargs):  # noqa: ANN001
        commands.append(list(command))
        if len(commands) > 1:
            marker.write_text("executed", encoding="utf-8")
            pytest.fail("staged Python executed before the startup package was rejected")
        package = staged_venv / "lib/python3.11/site-packages/sitecustomize"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed')\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(module, "_run_checked", fake_run_checked)

    with pytest.raises(module.RuntimeVerificationError, match="prohibited startup module"):
        module._build_staged_venv(base_python, staged_venv, requirements)

    assert not marker.exists()
    assert commands == [[str(base_python), "-I", "-S", "-m", "venv", "--copies", str(staged_venv)]]


@pytest.mark.parametrize(
    ("relative_path", "payload", "message"),
    [
        ("attack.pth", "import os\n", "startup/editable artifact"),
        ("sitecustomize.py", "VALUE = True\n", "startup/editable artifact"),
        (
            "package-1.0.dist-info/direct_url.json",
            '{"url":"file:///tmp/attack","dir_info":{"editable":true}}',
            "editable install",
        ),
    ],
)
def test_venv_audit_rejects_startup_and_editable_artifacts(
    tmp_path: Path,
    relative_path: str,
    payload: str,
    message: str,
) -> None:
    module = _load_script_module("fastvlm_venv_artifact_test", "scripts/setup/install_fastvlm_venv.py")
    venv = tmp_path / "venv"
    artifact = venv / "lib/python3.11/site-packages" / relative_path
    artifact.parent.mkdir(parents=True)
    artifact.write_text(payload, encoding="utf-8")

    with pytest.raises(module.RuntimeVerificationError, match=message):
        module.audit_runtime_venv(venv)


def test_freeze_evidence_symlink_is_rejected_without_touching_victim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_freeze_symlink_test", "scripts/setup/install_fastvlm_venv.py")
    manifest = _manifest(tmp_path)
    runtime_root = tmp_path / "runtime"
    runtime_root.mkdir()
    victim = tmp_path / "victim.txt"
    victim.write_text("preserve me\n", encoding="utf-8")
    (runtime_root / "fastvlm-pip-freeze.txt").symlink_to(victim)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", encoding="utf-8")
    build_calls: list[bool] = []
    monkeypatch.setattr(module, "_build_staged_venv", lambda *_args: build_calls.append(True))

    with pytest.raises(module.RuntimeVerificationError, match="dependency evidence must be a real regular file"):
        module.prepare_runtime_venv(
            manifest,
            root=runtime_root,
            base_python=Path(sys.executable),
            requirements=requirements,
        )

    assert not build_calls
    assert victim.read_text(encoding="utf-8") == "preserve me\n"
    assert (runtime_root / "fastvlm-pip-freeze.txt").is_symlink()


def test_venv_install_rejects_target_collisions_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_colliding_venv_install_test", "scripts/setup/install_fastvlm_venv.py")
    manifest = _manifest(tmp_path)
    manifest["python"]["venv_dir"] = "mlx-vlm"
    runtime_root = tmp_path / "missing-runtime"
    build_calls: list[bool] = []
    monkeypatch.setattr(module, "_build_staged_venv", lambda *_args: build_calls.append(True))

    with pytest.raises(module.ManifestError, match="runtime targets overlap"):
        module.prepare_runtime_venv(
            manifest,
            root=runtime_root,
            base_python=Path(sys.executable),
            requirements=tmp_path / "requirements.txt",
        )

    assert not build_calls
    assert not runtime_root.exists()


def test_venv_and_evidence_promotion_roll_back_together(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script_module("fastvlm_venv_rollback_test", "scripts/setup/install_fastvlm_venv.py")
    manifest = _manifest(tmp_path)
    runtime_root = tmp_path / "runtime"
    target_venv = runtime_root / ".venv-fastvlm"
    target_venv.mkdir(parents=True)
    (target_venv / "old-runtime").write_text("v1\n", encoding="utf-8")
    evidence = runtime_root / "fastvlm-pip-freeze.txt"
    evidence.write_text("old==1\n", encoding="utf-8")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("", encoding="utf-8")

    def build_staged(_base_python: Path, stage: Path, _requirements: Path) -> bytes:
        (stage / "bin").mkdir()
        python_path = stage / "bin/python"
        python_path.write_text("#!/bin/sh\n", encoding="utf-8")
        python_path.chmod(0o755)
        (stage / "lib/python3.11/site-packages").mkdir(parents=True)
        (stage / "new-runtime").write_text("v2\n", encoding="utf-8")
        return b"new==2\n"

    monkeypatch.setattr(module, "_build_staged_venv", build_staged)
    real_replace = module._replace_path
    injected = False

    def fail_evidence_once(source: Path, destination: Path) -> None:
        nonlocal injected
        if destination == evidence and not injected:
            injected = True
            raise OSError("injected evidence promotion failure")
        real_replace(source, destination)

    monkeypatch.setattr(module, "_replace_path", fail_evidence_once)

    with pytest.raises(OSError, match="injected evidence promotion failure"):
        module.prepare_runtime_venv(
            manifest,
            root=runtime_root,
            base_python=Path(sys.executable),
            requirements=requirements,
        )

    assert (target_venv / "old-runtime").read_text(encoding="utf-8") == "v1\n"
    assert not (target_venv / "new-runtime").exists()
    assert evidence.read_text(encoding="utf-8") == "old==1\n"
    assert not list(runtime_root.glob(".*.backup-*"))
    assert not list(runtime_root.glob(".*.tmp-*"))


def test_install_lock_serializes_concurrent_upgrade_transactions(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    lock_runner = repo_root / "scripts/setup/run_fastvlm_install_locked.py"
    lock_file = tmp_path / "locks/fastvlm.lock"
    log_path = tmp_path / "transactions.log"
    version_path = tmp_path / "version.txt"
    worker = "\n".join(
        [
            "import os, pathlib, sys, time",
            "label, log_name, version_name, delay = sys.argv[1:]",
            "log = pathlib.Path(log_name)",
            "with log.open('a', encoding='utf-8') as stream:",
            "    stream.write(f'{label}:start\\n')",
            "    stream.flush()",
            "    os.fsync(stream.fileno())",
            "time.sleep(float(delay))",
            "pathlib.Path(version_name).write_text(label, encoding='utf-8')",
            "with log.open('a', encoding='utf-8') as stream:",
            "    stream.write(f'{label}:end\\n')",
            "    stream.flush()",
            "    os.fsync(stream.fileno())",
        ]
    )

    def command(label: str, delay: str) -> list[str]:
        return [
            sys.executable,
            str(lock_runner),
            "run",
            "--lock-file",
            str(lock_file),
            "--timeout-seconds",
            "5",
            "--",
            sys.executable,
            "-c",
            worker,
            label,
            str(log_path),
            str(version_path),
            delay,
        ]

    first = subprocess.Popen(  # pylint: disable=consider-using-with
        command("v1", "0.4"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if log_path.exists() and "v1:start" in log_path.read_text(encoding="utf-8"):
            break
        time.sleep(0.02)
    else:
        first.kill()
        raise AssertionError("first transaction did not acquire the install lock")
    second = subprocess.Popen(  # pylint: disable=consider-using-with
        command("v2", "0"), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    first_stdout, first_stderr = first.communicate(timeout=10)
    second_stdout, second_stderr = second.communicate(timeout=10)

    assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 0, second_stdout + second_stderr
    assert log_path.read_text(encoding="utf-8").splitlines() == ["v1:start", "v1:end", "v2:start", "v2:end"]
    assert version_path.read_text(encoding="utf-8") == "v2"


def test_install_lock_assertion_rejects_and_does_not_acquire_forged_fd(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_forged_lock_test", "scripts/setup/run_fastvlm_install_locked.py")
    lock_path = tmp_path / "fastvlm.lock"
    descriptor = module._open_lock(lock_path)  # pylint: disable=consider-using-with
    token = 1234567
    os.lseek(descriptor, token, os.SEEK_SET)
    try:
        with pytest.raises(module.InstallLockError, match="does not hold"):
            module.assert_lock_held(lock_path, descriptor, token)

        probe = module._open_lock(lock_path)  # pylint: disable=consider-using-with
        try:
            module._acquire_lock(probe, timeout_seconds=0.1)
            module.fcntl.flock(probe, module.fcntl.LOCK_UN)
        finally:
            os.close(probe)
    finally:
        os.close(descriptor)


def test_install_lock_assertion_accepts_bound_locked_fd_and_token(tmp_path: Path) -> None:
    module = _load_script_module("fastvlm_valid_lock_test", "scripts/setup/run_fastvlm_install_locked.py")
    lock_path = tmp_path / "fastvlm.lock"
    descriptor = module._open_lock(lock_path)  # pylint: disable=consider-using-with
    token = 7654321
    try:
        module._acquire_lock(descriptor, timeout_seconds=0.1)
        os.lseek(descriptor, token, os.SEEK_SET)

        module.assert_lock_held(lock_path, descriptor, token)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize(
    "terminate_lock_runner",
    [False, True],
    ids=["normal", "lock-runner-parent-terminated"],
)
def test_public_installer_serializes_the_complete_concurrent_transaction(
    tmp_path: Path,
    terminate_lock_runner: bool,
) -> None:
    source_root = Path(__file__).resolve().parents[2]
    repo_root = tmp_path / "repo"
    setup_dir = repo_root / "scripts/setup"
    validation_dir = repo_root / "scripts/validation"
    setup_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)
    installer = setup_dir / "install_fastvlm_runtime.sh"
    installer.write_bytes((source_root / "scripts/setup/install_fastvlm_runtime.sh").read_bytes())
    installer.chmod(0o755)
    lock_runner = setup_dir / "run_fastvlm_install_locked.py"
    lock_runner.write_bytes((source_root / "scripts/setup/run_fastvlm_install_locked.py").read_bytes())
    resolver = setup_dir / "resolve_python_311.sh"
    resolver.write_text(f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(sys.executable)}\n", encoding="utf-8")
    resolver.chmod(0o755)
    log_path = tmp_path / "installer-transactions.log"
    source_installer = setup_dir / "install_fastvlm_sources.py"
    source_installer.write_text(
        "\n".join(
            [
                "import os, pathlib, sys, time",
                f"log = pathlib.Path({str(log_path)!r})",
                "label = os.environ['TP_TEST_TRANSACTION_LABEL']",
                "action = 'final' if '--verify-only' in sys.argv else 'source'",
                "with log.open('a', encoding='utf-8') as stream:",
                "    stream.write(f'{label}:{action}:start\\n')",
                "    stream.flush()",
                "    os.fsync(stream.fileno())",
                "if action == 'source':",
                "    gate_name = os.environ.get('TP_TEST_SOURCE_GATE')",
                "    if gate_name:",
                "        gate = pathlib.Path(gate_name)",
                "        while not gate.exists():",
                "            time.sleep(0.01)",
                "    else:",
                "        time.sleep(0.35)",
                "with log.open('a', encoding='utf-8') as stream:",
                "    stream.write(f'{label}:{action}:end\\n')",
                "    stream.flush()",
                "    os.fsync(stream.fileno())",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    venv_installer = setup_dir / "install_fastvlm_venv.py"
    venv_installer.write_text(
        "\n".join(
            [
                "import os, pathlib, sys",
                f"log = pathlib.Path({str(log_path)!r})",
                "audit_only = '--audit-only' in sys.argv",
                "if not audit_only:",
                "    root = pathlib.Path(sys.argv[sys.argv.index('--runtime-root') + 1])",
                "    venv_bin = root / '.venv-fastvlm/bin'",
                "    venv_bin.mkdir(parents=True, exist_ok=True)",
                "    venv_python = venv_bin / 'python'",
                "    venv_python.unlink(missing_ok=True)",
                f"    venv_python.symlink_to({sys.executable!r})",
                "with log.open('a', encoding='utf-8') as stream:",
                "    action = 'audit' if audit_only else 'venv'",
                "    stream.write(f\"{os.environ['TP_TEST_TRANSACTION_LABEL']}:{action}\\n\")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (validation_dir / "validate_fastvlm_runtime.py").write_text(
        "raise AssertionError('skip-verify must not invoke the Python/model verifier')\n",
        encoding="utf-8",
    )
    (setup_dir / "download_fastvlm_models.py").write_text(
        "\n".join(
            [
                "import os, pathlib",
                f"log = pathlib.Path({str(log_path)!r})",
                "with log.open('a', encoding='utf-8') as stream:",
                "    stream.write(f\"{os.environ['TP_TEST_TRANSACTION_LABEL']}:model\\n\")",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    def launch(label: str, *, source_gate: Path | None = None) -> subprocess.Popen[str]:
        environment = {**os.environ, "TP_TEST_TRANSACTION_LABEL": label}
        if source_gate is not None:
            environment["TP_TEST_SOURCE_GATE"] = str(source_gate)
        return subprocess.Popen(
            [str(installer), "--skip-verify"],
            cwd=repo_root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    source_gate = tmp_path / "release-v1-source" if terminate_lock_runner else None
    first = launch("v1", source_gate=source_gate)
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if log_path.exists() and "v1:source:start" in log_path.read_text(encoding="utf-8"):
            break
        time.sleep(0.02)
    else:
        first.kill()
        raise AssertionError("first public install did not enter its source transaction")

    if terminate_lock_runner:
        first.terminate()
        first.wait(timeout=5)
        lock_module = _load_script_module(
            "fastvlm_orphaned_installer_lock_test",
            "scripts/setup/run_fastvlm_install_locked.py",
        )
        probe = lock_module._open_lock(repo_root / ".runtime/.fastvlm-install.lock")
        try:
            with pytest.raises(lock_module.InstallLockError, match="Timed out"):
                lock_module._acquire_lock(probe, timeout_seconds=0.1)
        finally:
            os.close(probe)
    second = launch("v2")
    second_entered_before_release = False
    if source_gate is not None:
        deadline = time.monotonic() + 1
        while time.monotonic() < deadline:
            if "v2:source:start" in log_path.read_text(encoding="utf-8"):
                second_entered_before_release = True
                break
            time.sleep(0.02)
        source_gate.touch()

    if terminate_lock_runner:
        first_stdout = first.stdout.read() if first.stdout is not None else ""
        first_stderr = first.stderr.read() if first.stderr is not None else ""
    else:
        first_stdout, first_stderr = first.communicate(timeout=10)
    second_stdout, second_stderr = second.communicate(timeout=10)

    if terminate_lock_runner:
        assert first.returncode != 0
        assert not second_entered_before_release, second_stdout + second_stderr
    else:
        assert first.returncode == 0, first_stdout + first_stderr
    assert second.returncode == 0, second_stdout + second_stderr
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "v1:source:start",
        "v1:source:end",
        "v1:venv",
        "v1:model",
        "v1:audit",
        "v1:final:start",
        "v1:final:end",
        "v2:source:start",
        "v2:source:end",
        "v2:venv",
        "v2:model",
        "v2:audit",
        "v2:final:start",
        "v2:final:end",
    ]


def test_skip_verify_dry_run_still_executes_mandatory_source_preflight() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            str(repo_root / "scripts/setup/install_fastvlm_runtime.sh"),
            "--dry-run",
            "--skip-model-download",
            "--skip-verify",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "FastVLM governed sources: dry-run" in completed.stdout
    assert completed.stdout.count("[dry-run] governed source plan validated") == 2
    assert completed.stdout.rstrip().endswith("FastVLM governed sources: dry-run")
    assert completed.stdout.count("FastVLM governed sources: dry-run") == 2


def test_public_installer_scrubs_ambient_python_startup_before_resolver(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    poison_dir = tmp_path / "poison"
    poison_dir.mkdir()
    marker = tmp_path / "sitecustomize-executed"
    (poison_dir / "sitecustomize.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )
    completed = subprocess.run(
        [
            str(repo_root / "scripts/setup/install_fastvlm_runtime.sh"),
            "--dry-run",
            "--skip-model-download",
            "--skip-verify",
        ],
        cwd=repo_root,
        env={
            **os.environ,
            "PYTHONPATH": str(poison_dir),
            "PYTHONSTARTUP": str(poison_dir / "sitecustomize.py"),
        },
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stderr
    assert not marker.exists()


def test_public_installer_ignores_hostile_repo_venv_startup_hooks_before_trust(tmp_path: Path) -> None:
    source_root = Path(__file__).resolve().parents[2]
    repo_root = tmp_path / "repo"
    setup_dir = repo_root / "scripts/setup"
    setup_dir.mkdir(parents=True)
    installer = setup_dir / "install_fastvlm_runtime.sh"
    for name in ("install_fastvlm_runtime.sh", "resolve_python_311.sh", "run_fastvlm_install_locked.py"):
        shutil.copy2(source_root / "scripts/setup" / name, setup_dir / name)
    installer.chmod(0o755)

    helper_log = tmp_path / "helpers.log"
    helper_template = "\n".join(
        [
            "import os",
            "from pathlib import Path",
            "with Path(os.environ['TP_TEST_HELPER_LOG']).open('a', encoding='utf-8') as stream:",
            "    stream.write({label!r} + '\\n')",
        ]
    )
    (setup_dir / "install_fastvlm_sources.py").write_text(helper_template.format(label="source"), encoding="utf-8")
    (setup_dir / "install_fastvlm_venv.py").write_text(helper_template.format(label="venv"), encoding="utf-8")

    repo_venv = repo_root / ".venv"
    subprocess.run(
        [sys.executable, "-I", "-S", "-m", "venv", "--copies", str(repo_venv)],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    site_packages = next(repo_venv.glob("lib/python*/site-packages"))
    startup_marker = tmp_path / "hostile-pth-executed"
    (site_packages / "attack.pth").write_text(
        "import pathlib; " f"pathlib.Path({str(startup_marker)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [str(installer), "--skip-model-download", "--skip-verify"],
        cwd=repo_root,
        env={**os.environ, "TP_TEST_HELPER_LOG": str(helper_log)},
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not startup_marker.exists()
    assert helper_log.read_text(encoding="utf-8").splitlines() == ["source", "venv", "venv", "source"]
