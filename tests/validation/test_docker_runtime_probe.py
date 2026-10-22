"""Failure-mode coverage for the native CPU image runtime probe."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import subprocess
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

    def fail(_architecture):
        raise RuntimeError("stale bootstrap")

    monkeypatch.setattr(module, "check_runtime", fail)
    assert module.main(["--expected-architecture", "amd64"]) == 1
    output = capsys.readouterr()
    assert not output.out
    assert "stale bootstrap" in output.err


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


def test_workflow_builds_the_cpu_target_and_preserves_default_runtime_identity() -> None:
    workflow = yaml.safe_load((PROBE_PATH.parents[2] / ".github/workflows/build.yml").read_text())
    steps = {step.get("name"): step for step in workflow["jobs"]["docker-cpu-runtime"]["steps"]}
    build = shlex.split(steps["Build CPU runtime image"]["run"])
    assert build[:2] == ["docker", "build"]
    assert build[build.index("--platform") + 1] == "linux/amd64"
    assert build[build.index("--target") + 1] == "cpu"
    assert "--no-cache" in build and "--pull" in build and "--push" not in build
    script = steps["Validate default CPU runtime offline"]["run"].replace("\\\n", " ")
    command = next(line for line in script.splitlines() if line.strip().startswith("timeout "))
    run = shlex.split(command)
    assert run[:4] == ["timeout", "180s", "docker", "run"]
    assert run[run.index("--network") + 1] == "none"
    assert run[run.index("--entrypoint") + 1] == "python3.11"
    assert "/app/scripts/validation/check_docker_runtime.py" in run
    assert run[run.index("--expected-architecture") + 1] == "amd64"
    assert "--user" not in run and not any("HOME=" in part for part in run)
