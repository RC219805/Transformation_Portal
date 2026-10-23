"""Failure-mode coverage for the native image runtime probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytestmark = pytest.mark.unit
PROBE_PATH = Path(__file__).resolve().parents[2] / "scripts/validation/check_docker_runtime.py"


@pytest.fixture(name="probe")
def _probe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    spec = importlib.util.spec_from_file_location("docker_runtime_probe_test", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    home = tmp_path / "home"
    home.mkdir()
    base = tmp_path / "python"
    wheels = base / "share/python-wheels"
    wheels.mkdir(parents=True)
    for name, version in module.BOOTSTRAP_VERSIONS.items():
        (wheels / f"{name}-{version}-py3-none-any.whl").touch()
    monkeypatch.setattr(module, "EXPECTED_HOME", home)
    monkeypatch.setattr(module, "BASE_PREFIX", base)
    monkeypatch.setattr(module, "LEGACY_PREFIX", tmp_path / "legacy")
    monkeypatch.setattr(
        module,
        "sys",
        SimpleNamespace(
            version_info=SimpleNamespace(major=3, minor=11, releaselevel="final"),
            prefix=str(module.APP_PREFIX),
            base_prefix=str(base),
            executable="/opt/venv/bin/python3.11",
        ),
    )
    monkeypatch.setattr(
        module,
        "os",
        SimpleNamespace(
            getuid=lambda: 10001,
            getgid=lambda: 10001,
            environ={**os.environ, "HOME": str(home)},
        ),
    )
    monkeypatch.setattr(module, "platform", SimpleNamespace(system=lambda: "Linux", machine=lambda: "x86_64"))
    monkeypatch.setattr(module, "sysconfig", SimpleNamespace(get_config_var=lambda _name: str(wheels)))
    monkeypatch.setattr(module.Path, "home", lambda: home)
    return module


def test_current_default_identity_and_clean_prefix_are_accepted(probe) -> None:
    assert probe._check_identity("amd64") == probe.EXPECTED_HOME
    probe._check_prefix()
    probe._check_versions(dict(probe.BOOTSTRAP_VERSIONS))


@pytest.mark.parametrize("problem", ["root", "gid", "home", "python", "prerelease", "architecture", "platform", "prefix"])
def test_runtime_identity_regressions_are_rejected(probe, problem: str) -> None:
    if problem == "root":
        probe.os.getuid = lambda: 0
    elif problem == "gid":
        probe.os.getgid = lambda: 0
    elif problem == "home":
        probe.os.environ["HOME"] = "/root"
    elif problem == "python":
        probe.sys.version_info.minor = 12
    elif problem == "prerelease":
        probe.sys.version_info.releaselevel = "candidate"
    elif problem == "architecture":
        probe.platform.machine = lambda: "aarch64"
    elif problem == "platform":
        probe.platform.system = lambda: "Darwin"
    else:
        probe.sys.base_prefix = "/usr/local"
    with pytest.raises(RuntimeError):
        probe._check_identity("amd64")


def test_native_arm64_validation_is_explicit(probe) -> None:
    probe.platform.machine = lambda: "aarch64"
    assert probe._check_identity("arm64") == probe.EXPECTED_HOME


@pytest.mark.parametrize("problem", ["old-prefix", "historical-bundle", "missing-wheel", "old-wheel", "configuration"])
def test_bootstrap_payload_regressions_are_rejected(probe, problem: str) -> None:
    wheels = probe.BASE_PREFIX / "share/python-wheels"
    if problem == "old-prefix":
        probe.LEGACY_PREFIX.mkdir()
    elif problem == "historical-bundle":
        (probe.BASE_PREFIX / "lib/python3.11/ensurepip/_bundled").mkdir(parents=True)
    elif problem == "missing-wheel":
        next(wheels.iterdir()).unlink()
    elif problem == "old-wheel":
        (wheels / "pip-24.0-py3-none-any.whl").touch()
    else:
        probe.sysconfig.get_config_var = lambda _name: "/usr/local/old-wheels"
    with pytest.raises(RuntimeError):
        probe._check_prefix()


@pytest.mark.parametrize("name,version", [("pip", "24.0"), ("pip", "26.2.1"), ("setuptools", "79.0.1")])
def test_old_installed_bootstrap_versions_are_rejected(probe, name: str, version: str) -> None:
    versions = dict(probe.BOOTSTRAP_VERSIONS, **{name: version})
    with pytest.raises(RuntimeError, match="Bootstrap versions"):
        probe._check_versions(versions)


@pytest.mark.parametrize("stale", [False, True])
def test_fresh_offline_venv_uses_clean_base_and_rejects_old_bootstrap(
    probe, monkeypatch: pytest.MonkeyPatch, stale: bool
) -> None:
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        versions = dict(probe.BOOTSTRAP_VERSIONS)
        if stale:
            versions["pip"] = "24.0"
        return subprocess.CompletedProcess(command, 0, stdout=json.dumps(versions))

    monkeypatch.setattr(probe.subprocess, "run", run)
    if stale:
        with pytest.raises(RuntimeError, match="Bootstrap versions"):
            probe._check_offline_venv(probe.EXPECTED_HOME)
    else:
        assert probe._check_offline_venv(probe.EXPECTED_HOME) == probe.BOOTSTRAP_VERSIONS
    command, options = calls[0]
    assert command[:4] == [str(probe.BASE_PREFIX / "bin/python3.11"), "-I", "-m", "venv"]
    assert options["env"]["PIP_NO_INDEX"] == "1"
    assert options["timeout"] == 90 and options["check"] is True
    assert not Path(command[4]).exists()


def test_cli_failure_returns_nonzero_without_success_evidence(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    spec = importlib.util.spec_from_file_location("docker_runtime_probe_cli_test", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fail(_architecture, **_options):
        raise RuntimeError("stale bootstrap")

    monkeypatch.setattr(module, "check_runtime", fail)
    assert module.main(["--expected-architecture", "amd64"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "stale bootstrap" in output.err


@pytest.fixture(name="gpu_modules")
def _gpu_modules(probe, monkeypatch: pytest.MonkeyPatch):
    versions = dict(probe.GPU_VERSIONS)
    modules = {
        "torch": SimpleNamespace(__version__=versions["torch"], version=SimpleNamespace(cuda="12.6")),
        "torchvision": SimpleNamespace(__version__=versions["torchvision"]),
    }
    monkeypatch.setattr(
        probe,
        "importlib",
        SimpleNamespace(metadata=SimpleNamespace(version=versions.__getitem__), import_module=modules.__getitem__),
    )
    return versions, modules


def test_cuda_packages_are_verified_without_gpu_hardware(probe, gpu_modules) -> None:
    assert probe._check_gpu_packages() == {
        "torch": "2.13.0+cu126",
        "torchvision": "0.28.0+cu126",
        "cuda_build": "12.6",
        "hardware_validation": "not_performed",
    }


@pytest.mark.parametrize("problem", ["torch-metadata", "vision-metadata", "torch-module", "vision-module", "cuda-build"])
def test_wrong_cuda_package_builds_are_rejected(probe, gpu_modules, problem: str) -> None:
    versions, modules = gpu_modules
    if problem == "torch-metadata":
        versions["torch"] = "2.13.0"
    elif problem == "vision-metadata":
        versions["torchvision"] = "0.28.0"
    elif problem == "torch-module":
        modules["torch"].__version__ = "2.12.1+cu126"
    elif problem == "vision-module":
        modules["torchvision"].__version__ = "0.27.0+cu126"
    else:
        modules["torch"].version.cuda = "12.1"
    with pytest.raises(RuntimeError):
        probe._check_gpu_packages()


@pytest.mark.parametrize("missing", ["torch", "torchvision"])
def test_missing_gpu_import_is_rejected(probe, gpu_modules, missing: str) -> None:
    _, modules = gpu_modules

    def import_module(name):
        if name == missing:
            raise ImportError(f"Cannot import {name}")
        return modules[name]

    probe.importlib.import_module = import_module
    with pytest.raises(ImportError, match=missing):
        probe._check_gpu_packages()


@pytest.mark.parametrize("vendored_metadata", [False, True])
def test_bootstrap_inventory_hashes_actual_wheel_and_ca_payload(probe, vendored_metadata: bool) -> None:
    ca = b"task-owned bootstrap CA bytes"
    for name, version in probe.BOOTSTRAP_VERSIONS.items():
        path = probe.BASE_PREFIX / f"share/python-wheels/{name}-{version}-py3-none-any.whl"
        with zipfile.ZipFile(path, "w") as wheel:
            wheel.writestr(f"{name}-{version}.dist-info/METADATA", f"Name: {name}\nVersion: {version}\n")
            if name == "pip":
                wheel.writestr("pip/_vendor/certifi/cacert.pem", ca)
            elif vendored_metadata:
                wheel.writestr("setuptools/_vendor/wheel-0.46.3.dist-info/METADATA", "Name: wheel\nVersion: 0.46.3\n")
    inventory = {package["name"]: package for package in probe._bootstrap_wheel_inventory()}
    pip = inventory["pip"]
    assert pip["pip_ca_sha256"] == hashlib.sha256(ca).hexdigest()
    path = probe.BASE_PREFIX / "share/python-wheels" / pip["filename"]
    assert pip["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert pip["version"] == probe.BOOTSTRAP_VERSIONS["pip"]
    assert "pip_ca_sha256" not in inventory["setuptools"]


def test_installed_inventory_records_metadata_and_actual_ca_without_importing_pip(
    probe, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    ca = tmp_path / "cacert.pem"
    ca.write_bytes(b"task-owned installed CA bytes")
    real_run = subprocess.run

    def run(command, **options):
        assert command[:3] == ["/task/python", "-I", "-c"]
        prelude = (
            "from types import SimpleNamespace as S\n"
            "import importlib.metadata as m\n"
            "m.distributions = lambda: [S(metadata={'Name':'example'}, version='1.2', "
            'read_text=lambda _: \'{"url":"file:///app","dir_info":{"editable":true}}\')]\n'
            f"m.distribution = lambda _: S(locate_file=lambda _: {str(ca)!r})\n"
        )
        return real_run([sys.executable, "-I", "-c", prelude + command[3]], **options)

    monkeypatch.setattr(probe.subprocess, "run", run)
    inventory = probe._installed_inventory(Path("/task/python"))
    assert inventory["pip_ca_sha256"] == hashlib.sha256(ca.read_bytes()).hexdigest()
    assert inventory["packages"] == [
        {"name": "example", "version": "1.2", "direct_url": {"url": "file:///app", "dir_info": {"editable": True}}}
    ]


def test_offline_inventory_is_collected_before_the_temporary_venv_is_removed(probe, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = []
    monkeypatch.setattr(probe.subprocess, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(probe, "_metadata_versions", lambda _python: dict(probe.BOOTSTRAP_VERSIONS))

    def inventory(python):
        paths.append(python)
        assert python.parents[2].is_dir()
        return {"packages": [], "pip_ca_sha256": "task-owned CA digest"}

    monkeypatch.setattr(probe, "_installed_inventory", inventory)
    inventories = {}
    probe._check_offline_venv(probe.EXPECTED_HOME, inventories=inventories)
    assert inventories["offline_venv"]["pip_ca_sha256"] == "task-owned CA digest"
    assert not paths[0].parents[2].exists()


@pytest.mark.parametrize("gpu_enabled", [False, True])
def test_runtime_gpu_checks_and_inventory_are_opt_in(probe, monkeypatch: pytest.MonkeyPatch, gpu_enabled: bool) -> None:
    versions = dict(probe.BOOTSTRAP_VERSIONS)
    monkeypatch.setattr(
        probe,
        "importlib",
        SimpleNamespace(metadata=SimpleNamespace(version=versions.__getitem__), import_module=lambda _: None),
    )
    monkeypatch.setattr(probe.platform, "python_version", lambda: "3.11.17", raising=False)
    monkeypatch.setattr(probe, "_metadata_versions", lambda _: versions)
    monkeypatch.setattr(probe, "_check_module_launcher", lambda: None)

    def offline(_home, *, inventories):
        assert (inventories is not None) == gpu_enabled
        if inventories is not None:
            inventories["offline_venv"] = {"packages": []}
        return versions

    def gpu():
        assert gpu_enabled
        return {"hardware_validation": "not_performed"}

    def inventory(_python):
        assert gpu_enabled
        return {"packages": [], "pip_ca_sha256": "collected CA digest"}

    monkeypatch.setattr(probe, "_check_offline_venv", offline)
    monkeypatch.setattr(probe, "_check_gpu_packages", gpu)
    monkeypatch.setattr(probe, "_installed_inventory", inventory)
    monkeypatch.setattr(probe, "_bootstrap_wheel_inventory", lambda: [])
    evidence = probe.check_runtime("amd64", gpu_packages=gpu_enabled)
    assert ("gpu_packages" in evidence) == gpu_enabled
    if gpu_enabled:
        assert set(evidence["installed_inventory"]) == {"application", "base", "offline_venv"}
        assert evidence["installed_inventory"]["application"]["pip_ca_sha256"] == "collected CA digest"


def test_cli_enables_gpu_package_verification_explicitly(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    spec = importlib.util.spec_from_file_location("docker_runtime_probe_gpu_cli_test", PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def check(architecture, **options):
        calls.append((architecture, options))
        return {"hardware_validation": "not_performed"}

    monkeypatch.setattr(module, "check_runtime", check)
    assert module.main(["--expected-architecture", "amd64", "--gpu-packages"]) == 0
    assert calls == [("amd64", {"gpu_packages": True})]
    assert json.loads(capsys.readouterr().out)["hardware_validation"] == "not_performed"


@pytest.mark.parametrize(
    "returncode,stdout,stderr,expected_error",
    [
        (0, "Usage: python -m transformation_portal.cli [OPTIONS]", "", None),
        (1, "", "No module named transformation_portal.cli.__main__", "No module named"),
        (0, " \n", "", "empty help"),
    ],
)
def test_default_module_launcher_must_execute_and_return_help(
    probe, monkeypatch: pytest.MonkeyPatch, returncode: int, stdout: str, stderr: str, expected_error: str | None
) -> None:
    calls = []

    def run(command, **options):
        calls.append((command, options))
        return subprocess.CompletedProcess(command, returncode, stdout=stdout, stderr=stderr)

    monkeypatch.setattr(probe.subprocess, "run", run)
    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            probe._check_module_launcher()
    else:
        probe._check_module_launcher()
    command, options = calls[0]
    assert command == [probe.sys.executable, "-I", "-m", "transformation_portal.cli", "--help"]
    assert options == {"capture_output": True, "text": True, "timeout": 30, "check": False}


@pytest.mark.parametrize("target", ["cpu", "gpu"])
def test_workflow_builds_both_targets_and_preserves_default_runtime_identity(target: str) -> None:
    workflow = yaml.safe_load((PROBE_PATH.parents[2] / ".github/workflows/build.yml").read_text())
    steps = {step.get("name"): step for step in workflow["jobs"]["docker-cpu-runtime"]["steps"]}
    build_step = "Build CPU runtime image" if target == "cpu" else "Build full GPU runtime image"
    build = shlex.split(steps[build_step]["run"])
    assert build[:2] == ["docker", "build"]
    assert build[build.index("--platform") + 1] == "linux/amd64"
    assert build[build.index("--target") + 1] == target
    assert "--no-cache" in build and "--pull" in build and "--push" not in build
    probe_step = (
        "Validate default CPU runtime offline" if target == "cpu" else "Validate default GPU runtime offline without hardware"
    )
    script = steps[probe_step]["run"].replace("\\\n", " ")
    command = next(line for line in script.splitlines() if line.strip().startswith("timeout "))
    run = shlex.split(command)
    assert run[:4] == ["timeout", "180s", "docker", "run"]
    assert run[run.index("--network") + 1] == "none"
    assert run[run.index("--entrypoint") + 1] == "python3.11"
    assert "/app/scripts/validation/check_docker_runtime.py" in run
    assert run[run.index("--expected-architecture") + 1] == "amd64"
    assert "--user" not in run and not any("HOME=" in part for part in run)
    assert ("--gpu-packages" in run) == (target == "gpu")
    assert "--gpus" not in run and "--device" not in run
