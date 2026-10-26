"""Static installed-version checks for the governed FastVLM runtime."""

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

from tests.vlm_captioning.test_fastvlm_runtime_manifest import _manifest, _write_fixture_runtime

pytestmark = [pytest.mark.unit, pytest.mark.security]
PINS = {"pip": "26.2.2", "Pillow": "13.0.0", "aiohttp": "3.14.4", "certifi": "2026.10.19", "filelock": "4.2.1"}


def _load(relative_path: str):  # noqa: ANN202
    path = Path(__file__).resolve().parents[2] / relative_path
    spec = importlib.util.spec_from_file_location("fastvlm_dependency_version_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _metadata(venv: Path, name: str, version: str, *, directory: str | None = None) -> Path:
    path = venv / "lib/python3.11/site-packages" / (directory or f"{name}-{version}.dist-info") / "METADATA"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\n", encoding="utf-8")
    return path


@pytest.fixture(name="dependency_fixture")
def _dependency_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN201
    module = _load("scripts/setup/install_fastvlm_venv.py")
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("".join(f"{name}=={version}\n" for name, version in PINS.items()), encoding="utf-8")
    monkeypatch.setattr(module, "GOVERNED_REQUIREMENTS_PATH", requirements)
    venv = tmp_path / "runtime/.venv-fastvlm"
    paths = {name: _metadata(venv, name, version) for name, version in PINS.items()}

    def no_execution(*_args, **_kwargs):  # noqa: ANN001
        pytest.fail("static dependency audit must not execute the runtime")

    monkeypatch.setattr(module.subprocess, "run", no_execution)
    return module, venv, paths, requirements


def test_current_pins_are_verified_without_runtime_execution(dependency_fixture) -> None:  # noqa: ANN001
    module, venv, _paths, requirements = dependency_fixture
    requirements.write_text(requirements.read_text() + "typing_extensions==4.15.0\n", encoding="utf-8")
    _metadata(venv, "typing-extensions", "4.15.0")
    module.audit_runtime_requirements(venv)


@pytest.mark.parametrize(
    "name,version",
    [
        ("pip", "26.2.1"),
        ("Pillow", "12.3.0"),
        ("aiohttp", "3.14.3"),
        ("certifi", "2026.6.17"),
        ("filelock", "3.32.5"),
    ],
)
def test_stale_governed_versions_are_rejected(dependency_fixture, name: str, version: str) -> None:  # noqa: ANN001
    module, venv, paths, _requirements = dependency_fixture
    paths[name].write_text(f"Name: {name}\nVersion: {version}\n\n", encoding="utf-8")
    with pytest.raises(module.RuntimeVerificationError, match="does not match the governed pin"):
        module.audit_runtime_requirements(venv)


@pytest.mark.parametrize(
    "problem", ["missing", "duplicate", "ambiguous", "symlink", "oversized", "inventory", "missing-extra"]
)
def test_missing_or_ambiguous_metadata_fails_closed(dependency_fixture, problem: str) -> None:  # noqa: ANN001
    module, venv, paths, _requirements = dependency_fixture
    metadata = paths["pip"]
    if problem == "missing":
        metadata.unlink()
    elif problem == "duplicate":
        _metadata(venv, "PIP", PINS["pip"], directory="duplicate.dist-info")
    elif problem == "ambiguous":
        metadata.write_text("Name: pip\nVersion: 26.2.2\nVersion: 26.2.1\n\n", encoding="utf-8")
    elif problem == "symlink":
        payload = metadata.read_bytes()
        metadata.unlink()
        other = venv / "external-metadata"
        other.write_bytes(payload)
        metadata.symlink_to(other)
    elif problem == "oversized":
        metadata.write_bytes(b"x" * (module.MAX_DEPENDENCY_METADATA_BYTES + 1))
    elif problem == "missing-extra":
        (metadata.parent.parent / "broken.dist-info").mkdir()
    else:
        module.MAX_DEPENDENCY_DISTRIBUTIONS = 1
    with pytest.raises(module.RuntimeVerificationError):
        module.audit_runtime_requirements(venv)


@pytest.mark.parametrize("legacy", [False, True])
def test_governed_runtime_rejects_unexpected_distributions(dependency_fixture, legacy: bool) -> None:  # noqa: ANN001
    module, venv, _paths, _requirements = dependency_fixture
    if legacy:
        path = venv / "lib/python3.11/site-packages/gradio.egg-info"
        path.mkdir()
        (path / "PKG-INFO").write_text("Name: gradio\nVersion: 6.22.0\n\n", encoding="utf-8")
    else:
        _metadata(venv, "gradio", "6.22.0")
    with pytest.raises(module.RuntimeVerificationError, match="outside the governed requirements.*gradio"):
        module.audit_runtime_requirements(venv)


@pytest.mark.parametrize("text", ["", "pip>=26.2.2\n", "pip==26.2.2\nPIP==26.2.2\n"])
def test_governed_requirements_must_be_unambiguous_exact_pins(dependency_fixture, text: str) -> None:  # noqa: ANN001
    module, venv, _paths, requirements = dependency_fixture
    requirements.write_text(text, encoding="utf-8")
    with pytest.raises(module.RuntimeVerificationError):
        module.audit_runtime_requirements(venv)


def test_installer_verify_only_rejects_stale_installed_pins(
    dependency_fixture, monkeypatch: pytest.MonkeyPatch
) -> None:  # noqa: ANN001
    module, venv, paths, _requirements = dependency_fixture
    (venv.parent / module.FREEZE_EVIDENCE_NAME).write_text("pip==26.2.2\n", encoding="utf-8")
    layout_calls = []
    monkeypatch.setattr(module, "audit_runtime_venv", lambda *_args, **_kwargs: layout_calls.append(True))
    paths["pip"].write_text("Name: pip\nVersion: 26.2.1\n\n", encoding="utf-8")
    with pytest.raises(module.RuntimeVerificationError, match="pip==26.2.2"):
        module.audit_installed_runtime_venv(
            _manifest(venv.parent), root=venv.parent, expected_base_python=Path("/trusted/python")
        )
    assert layout_calls == [True]
    paths["pip"].write_text("Name: pip\nVersion: 26.2.2\n\n", encoding="utf-8")
    assert (
        module.audit_installed_runtime_venv(
            _manifest(venv.parent), root=venv.parent, expected_base_python=Path("/trusted/python")
        )
        == "verified"
    )


@pytest.mark.parametrize(
    "name,version", [("pip", "26.2.1"), ("Pillow", "12.3.0"), ("aiohttp", "3.14.3"), ("filelock", "3.32.5")]
)
def test_validator_checks_current_pins_before_import_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str, version: str
) -> None:
    validator = _load("scripts/validation/validate_fastvlm_runtime.py")
    runtime = tmp_path / "runtime"
    _write_fixture_runtime(runtime)
    requirements = tmp_path / "requirements.txt"
    requirements.write_text("".join(f"{key}=={value}\n" for key, value in PINS.items()), encoding="utf-8")
    monkeypatch.setattr(validator._venv_helpers, "GOVERNED_REQUIREMENTS_PATH", requirements)
    monkeypatch.setattr(validator, "audit_runtime_venv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(validator, "verify_runtime_sources", lambda *_args, **_kwargs: [])
    imports = []
    monkeypatch.setattr(validator, "verify_python_imports", lambda *_args, **_kwargs: imports.append(True) or [])
    for key, value in PINS.items():
        _metadata(runtime / ".venv-fastvlm", key, value)
    path = next((runtime / ".venv-fastvlm").glob(f"lib/python*/site-packages/{name}-*.dist-info/METADATA"))
    path.write_text(f"Name: {name}\nVersion: {version}\n\n", encoding="utf-8")
    kwargs = dict(
        manifest_path=tmp_path / "manifest.json",
        root=runtime,
        roles=["smoke"],
        manifest=_manifest(tmp_path),
        include_sources=True,
        include_python=True,
        include_import_smoke=True,
        expected_base_python=Path("/trusted/python"),
    )
    evidence = validator.build_runtime_evidence(**kwargs)
    assert evidence["checks"]["python_environment"]["status"] == "failed"
    assert evidence["checks"]["python_imports"]["status"] == "skipped"
    assert not imports
    path.write_text(f"Name: {name}\nVersion: {PINS[name]}\n\n", encoding="utf-8")
    assert validator.build_runtime_evidence(**kwargs)["runtime_status"] == "ready"
    assert imports == [True]


@pytest.mark.parametrize("version,success", [("26.2.2", True), ("26.2.1", False)])
def test_staged_install_checks_its_requested_pins_before_pip_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str, success: bool
) -> None:
    module = _load("scripts/setup/install_fastvlm_venv.py")
    stage = tmp_path / "stage"
    _metadata(stage, "pip", version)
    _metadata(stage, "custom-extra", "1.0.0")
    requirements = tmp_path / "custom-requirements.txt"
    requirements.write_text("pip==26.2.2\n", encoding="utf-8")
    descriptions = []

    def run(command, *, timeout_seconds, description):  # noqa: ANN001, ANN202
        descriptions.append(description)
        return subprocess.CompletedProcess(command, 0, stdout="pip==26.2.2\n", stderr="")

    monkeypatch.setattr(module, "_run_checked", run)
    monkeypatch.setattr(module, "audit_runtime_venv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_remove_canonical_lib64_alias", lambda *_args: None)
    monkeypatch.setattr(module, "_remove_allowlisted_bootstrap_pth", lambda *_args: None)
    monkeypatch.setattr(module, "_uninstall_bootstrap_setuptools", lambda *_args: None)
    if success:
        assert module._build_staged_venv(Path("/trusted/python"), stage, requirements) == b"pip==26.2.2\n"
    else:
        with pytest.raises(module.RuntimeVerificationError, match="governed pin"):
            module._build_staged_venv(Path("/trusted/python"), stage, requirements)
    assert ("FastVLM dependency consistency check" in descriptions) is success
