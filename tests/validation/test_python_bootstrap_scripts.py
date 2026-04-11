"""Tests for Python bootstrap and validation shell entrypoints."""

from __future__ import annotations

import os
import shlex
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MAKEFILE_PATH = PROJECT_ROOT / "Makefile"
RESOLVER_PATH = PROJECT_ROOT / "scripts" / "setup" / "resolve_python_311.sh"
DA3_RUNTIME_INSTALLER_PATH = PROJECT_ROOT / "scripts" / "setup" / "install_da3_runtime.sh"
RAW_RUNTIME_INSTALLER_PATH = PROJECT_ROOT / "scripts" / "setup" / "install_raw_runtime.sh"
VALIDATION_SUITE_PATH = PROJECT_ROOT / "scripts" / "validation" / "run_full_validation_suite.sh"
ML_STACK_INSTALLER_PATH = PROJECT_ROOT / "scripts" / "bootstrap" / "install_ml_stack.sh"


def _write_executable(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def _copy_repo_file(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    destination.chmod(source.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return destination


def _write_fake_python(path: Path, *, version: str, real_python: str) -> Path:
    version_parts = version.split(".")
    major = int(version_parts[0])
    minor = int(version_parts[1])
    supported = (major, minor) >= (3, 11)
    exit_code = "0" if supported else "1"
    script = f"""#!/bin/sh
REAL_PYTHON={shlex.quote(real_python)}
FAKE_VERSION={shlex.quote(version)}
if [ "$1" = "-c" ]; then
    exit {exit_code}
fi
if [ "$1" = "-V" ] || [ "$1" = "--version" ]; then
    echo "Python $FAKE_VERSION"
    exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
    echo "pip 24.0 from fake ($FAKE_VERSION)"
    exit 0
fi
export FAKE_PYTHON_LAUNCHER="$0"
exec "$REAL_PYTHON" "$@"
    """
    return _write_executable(path, script)


def _write_fake_ml_core_python(
    path: Path,
    *,
    version: str,
    platform_system: str,
    platform_machine: str,
    pip_log_path: Path,
) -> Path:
    version_parts = version.split(".")
    major = int(version_parts[0])
    minor = int(version_parts[1])
    supported = (major, minor) >= (3, 11)
    exit_code = "0" if supported else "1"
    script = f"""#!/bin/sh
FAKE_VERSION={shlex.quote(version)}
PLATFORM_SYSTEM={shlex.quote(platform_system)}
PLATFORM_MACHINE={shlex.quote(platform_machine)}
PIP_LOG_PATH={shlex.quote(str(pip_log_path))}
if [ "$1" = "-c" ]; then
    case "$2" in
        *"platform.system()"*)
            echo "$PLATFORM_SYSTEM"
            exit 0
            ;;
        *"platform.machine()"*)
            echo "$PLATFORM_MACHINE"
            exit 0
            ;;
        *)
            exit {exit_code}
            ;;
    esac
fi
if [ "$1" = "-V" ] || [ "$1" = "--version" ]; then
    echo "Python $FAKE_VERSION"
    exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "--version" ]; then
    echo "pip 24.0 from fake ($FAKE_VERSION)"
    exit 0
fi
if [ "$1" = "-m" ] && [ "$2" = "pip" ] && [ "$3" = "install" ]; then
    printf '%s\\n' "$*" >> "$PIP_LOG_PATH"
    exit 0
fi
export FAKE_PYTHON_LAUNCHER="$0"
exit 0
"""
    return _write_executable(path, script)


def test_resolver_prefers_repo_venv(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    repo_python = _write_fake_python(
        repo_root / ".venv" / "bin" / "python",
        version="3.11.15",
        real_python=sys.executable,
    )
    fakebin = tmp_path / "fakebin"
    _write_fake_python(fakebin / "python3.11", version="3.12.0", real_python=sys.executable)

    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(["bash", str(script_path)], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(repo_python)


def test_resolver_falls_back_to_python311_when_repo_venv_missing(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    fakebin = tmp_path / "fakebin"
    python311 = _write_fake_python(fakebin / "python3.11", version="3.11.15", real_python=sys.executable)
    # Provide fake python3.12-3.15 that report unsupported versions to isolate from system
    _write_fake_python(fakebin / "python3.15", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.14", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.13", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.12", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3", version="3.9.6", real_python=sys.executable)
    _write_fake_python(fakebin / "python", version="3.9.6", real_python=sys.executable)

    # fakebin first so our fake pythons are found; /usr/bin for bash and other utilities
    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(["bash", str(script_path)], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(python311)


def test_resolver_discovers_newer_versioned_python_commands(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    fakebin = tmp_path / "fakebin"
    python316 = _write_fake_python(fakebin / "python3.16", version="3.16.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.15", version="3.15.2", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.11", version="3.11.15", real_python=sys.executable)
    _write_fake_python(fakebin / "python3", version="3.9.6", real_python=sys.executable)
    _write_fake_python(fakebin / "python", version="3.9.6", real_python=sys.executable)

    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(["bash", str(script_path)], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(python316)


def test_resolver_prefers_windows_repo_venv_layout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    repo_python = _write_fake_python(
        repo_root / ".venv" / "Scripts" / "python.exe",
        version="3.12.4",
        real_python=sys.executable,
    )

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(["bash", str(script_path)], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(repo_python)


def test_resolver_rejects_old_python_candidates(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    fakebin = tmp_path / "fakebin"
    # All candidates report old unsupported versions
    _write_fake_python(fakebin / "python3.15", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.14", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.13", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.12", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.11", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3", version="3.9.6", real_python=sys.executable)
    _write_fake_python(fakebin / "python", version="3.10.14", real_python=sys.executable)

    # fakebin first so our fake pythons are found; /usr/bin for bash and other utilities
    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(["bash", str(script_path)], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 1
    assert "make venv" in result.stderr
    # Updated guidance now includes multiple Python version examples
    assert "python3.13 -m venv .venv" in result.stderr
    assert "python3.12 -m venv .venv" in result.stderr


def test_make_venv_refuses_wrong_version_existing_repo_venv(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _copy_repo_file(MAKEFILE_PATH, repo_root / "Makefile")
    _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    _write_fake_python(repo_root / ".venv" / "bin" / "python", version="3.9.6", real_python=sys.executable)

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(["make", "venv"], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode != 0
    assert "make repair-core-venv" in (result.stdout + result.stderr)


def test_make_venv_accepts_supported_windows_repo_venv_layout(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _copy_repo_file(MAKEFILE_PATH, repo_root / "Makefile")
    _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    _write_fake_python(repo_root / ".venv" / "Scripts" / "python.exe", version="3.12.4", real_python=sys.executable)

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(["make", "venv"], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert ".venv already present" in result.stdout


def test_make_install_ml_core_selects_darwin_arm64_lockfile(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    pip_log_path = repo_root / "pip-install.log"
    _copy_repo_file(MAKEFILE_PATH, repo_root / "Makefile")
    _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    _copy_repo_file(
        PROJECT_ROOT / "requirements" / "ml-core-darwin-arm64.txt",
        repo_root / "requirements" / "ml-core-darwin-arm64.txt",
    )
    fake_python = _write_fake_ml_core_python(
        repo_root / ".venv" / "bin" / "python",
        version="3.11.15",
        platform_system="Darwin",
        platform_machine="arm64",
        pip_log_path=pip_log_path,
    )

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(["make", "install-ml-core"], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert ".venv already present" in result.stdout
    assert "Using requirements/ml-core-darwin-arm64.txt" in result.stdout
    assert "platform-specific ML core lockfile not found" not in (result.stdout + result.stderr)
    pip_commands = pip_log_path.read_text(encoding="utf-8")
    assert "-m pip install -r requirements/ml-core-darwin-arm64.txt" in pip_commands
    assert "-m pip install -e ." in pip_commands
    assert str(fake_python) not in (result.stdout + result.stderr)


def test_make_install_ml_sam2_uses_core_mps_profile_on_darwin_arm64(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    pip_log_path = repo_root / "pip-install.log"
    profile_log_path = repo_root / "sam2-profile.log"
    _copy_repo_file(MAKEFILE_PATH, repo_root / "Makefile")
    _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    _copy_repo_file(ML_STACK_INSTALLER_PATH, repo_root / "scripts" / "bootstrap" / "install_ml_stack.sh")
    fake_python = _write_fake_ml_core_python(
        repo_root / ".venv" / "bin" / "python",
        version="3.11.15",
        platform_system="Darwin",
        platform_machine="arm64",
        pip_log_path=pip_log_path,
    )
    _write_executable(
        repo_root / "scripts" / "bootstrap" / "install_ml_stack.sh",
        ("#!/bin/sh\n" f"printf '%s\\n' \"$*\" >> {shlex.quote(str(profile_log_path))}\n" "exit 0\n"),
    )

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(["make", "install-ml-sam2"], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert ".venv already present" in result.stdout
    assert "Using ML SAM2 profile core-mps,sam2" in result.stdout
    assert profile_log_path.read_text(encoding="utf-8").strip() == "--profile core-mps,sam2"
    pip_commands = pip_log_path.read_text(encoding="utf-8")
    assert "-m pip install -e ." in pip_commands
    assert str(fake_python) not in (result.stdout + result.stderr)


def test_validation_suite_uses_resolved_python_for_preflight(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _copy_repo_file(VALIDATION_SUITE_PATH, repo_root / "scripts" / "validation" / "run_full_validation_suite.sh")
    _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    marker_path = repo_root / "preflight_python.txt"
    _write_executable(
        repo_root / "scripts" / "validation" / "check_local_environment.py",
        (
            "#!/usr/bin/env python3\n"
            "import os\n"
            "from pathlib import Path\n"
            f"Path({str(marker_path)!r}).write_text(os.environ.get('FAKE_PYTHON_LAUNCHER', ''), encoding='utf-8')\n"
        ),
    )

    fakebin = tmp_path / "fakebin"
    python311 = _write_fake_python(fakebin / "python3.11", version="3.11.15", real_python=sys.executable)
    # Provide fake python3.12-3.15 that report unsupported versions to isolate from system
    _write_fake_python(fakebin / "python3.15", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.14", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.13", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.12", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3", version="3.9.6", real_python=sys.executable)
    _write_fake_python(fakebin / "python", version="3.9.6", real_python=sys.executable)
    _write_executable(
        fakebin / "make",
        "#!/bin/sh\n" "exit 0\n",
    )

    # fakebin first so our fake pythons are found; /usr/bin for bash and other utilities
    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "validation" / "run_full_validation_suite.sh"), "--quick", "--skip-frontdoor"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert marker_path.read_text(encoding="utf-8").strip() == str(python311)


def test_validation_suite_help_skips_python_resolution(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    _copy_repo_file(VALIDATION_SUITE_PATH, repo_root / "scripts" / "validation" / "run_full_validation_suite.sh")
    _write_executable(
        repo_root / "scripts" / "setup" / "resolve_python_311.sh",
        "#!/bin/sh\n" "echo missing-python >&2\n" "exit 1\n",
    )

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "validation" / "run_full_validation_suite.sh"), "--help"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Usage" in result.stdout
    assert "missing-python" not in result.stderr


def test_install_da3_runtime_uses_resolved_python_for_venv_creation(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = _copy_repo_file(DA3_RUNTIME_INSTALLER_PATH, repo_root / "scripts" / "setup" / "install_da3_runtime.sh")
    _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")

    fakebin = tmp_path / "fakebin"
    python311 = _write_fake_python(fakebin / "python3.11", version="3.11.15", real_python=sys.executable)
    # Provide fake python3.12-3.15 that report unsupported versions to isolate from system
    _write_fake_python(fakebin / "python3.15", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.14", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.13", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.12", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3", version="3.9.6", real_python=sys.executable)
    _write_fake_python(fakebin / "python", version="3.9.6", real_python=sys.executable)

    # fakebin first so our fake pythons are found; /usr/bin for bash, git, and other utilities
    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--dry-run",
            "--skip-verify",
            "--checkout-dir",
            str(repo_root / ".runtime" / "Depth-Anything-3"),
            "--venv-dir",
            str(repo_root / ".venv-da3"),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Using bootstrap interpreter: {python311}" in result.stdout
    assert f"+ {python311} -m venv {repo_root / '.venv-da3'}" in result.stdout


def test_install_raw_runtime_uses_resolved_python_for_venv_creation(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = _copy_repo_file(RAW_RUNTIME_INSTALLER_PATH, repo_root / "scripts" / "setup" / "install_raw_runtime.sh")
    _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")

    fakebin = tmp_path / "fakebin"
    python311 = _write_fake_python(fakebin / "python3.11", version="3.11.15", real_python=sys.executable)
    # Provide fake python3.12-3.15 that report unsupported versions to isolate from system
    _write_fake_python(fakebin / "python3.15", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.14", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.13", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.12", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3", version="3.9.6", real_python=sys.executable)
    _write_fake_python(fakebin / "python", version="3.9.6", real_python=sys.executable)

    # fakebin first so our fake pythons are found; /usr/bin for bash and other utilities
    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--dry-run",
            "--skip-verify",
            "--venv-dir",
            str(repo_root / ".venv-raw"),
        ],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Using bootstrap interpreter: {python311}" in result.stdout
    assert f"+ {python311} -m venv {repo_root / '.venv-raw'}" in result.stdout


def test_install_ml_stack_sam2_success_does_not_fail_on_return_trap(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    script_path = _copy_repo_file(ML_STACK_INSTALLER_PATH, repo_root / "scripts" / "bootstrap" / "install_ml_stack.sh")
    resolver_path = repo_root / "scripts" / "setup" / "resolve_python_311.sh"
    pip_log_path = repo_root / "pip-install.log"
    fakebin = tmp_path / "fakebin"
    error_log_path = tmp_path / "sam2-error.log"
    fake_python = _write_fake_ml_core_python(
        repo_root / ".venv" / "bin" / "python",
        version="3.11.15",
        platform_system="Darwin",
        platform_machine="arm64",
        pip_log_path=pip_log_path,
    )
    _write_executable(resolver_path, f"#!/bin/sh\nprintf '%s\\n' {shlex.quote(str(fake_python))}\n")
    _write_executable(
        fakebin / "mktemp",
        (
            "#!/bin/sh\n"
            f"tmp={shlex.quote(str(error_log_path))}\n"
            'rm -f "$tmp"\n'
            'touch "$tmp"\n'
            "printf '%s\\n' \"$tmp\"\n"
        ),
    )
    (repo_root / "requirements").mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(
        ["bash", str(script_path), "--profile", "sam2"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "SAM2 installed successfully via standard path." in result.stdout
    assert "unbound variable" not in (result.stdout + result.stderr)
    assert not error_log_path.exists()
    pip_commands = pip_log_path.read_text(encoding="utf-8")
    assert "-m pip install --extra-index-url https://download.pytorch.org/whl/cpu sam2==1.1.0" in pip_commands


def test_resolver_prefers_python314_over_python311(tmp_path: Path) -> None:
    """Test that resolver prefers newer Python versions (e.g., 3.14) over 3.11."""
    repo_root = tmp_path / "repo"
    script_path = _copy_repo_file(RESOLVER_PATH, repo_root / "scripts" / "setup" / "resolve_python_311.sh")
    fakebin = tmp_path / "fakebin"
    # Python 3.14 is available and supported
    python314 = _write_fake_python(fakebin / "python3.14", version="3.14.0", real_python=sys.executable)
    # Python 3.11 is also available but should not be preferred
    _write_fake_python(fakebin / "python3.11", version="3.11.15", real_python=sys.executable)
    # Other candidates report unsupported versions
    _write_fake_python(fakebin / "python3.15", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.13", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3.12", version="3.9.0", real_python=sys.executable)
    _write_fake_python(fakebin / "python3", version="3.9.6", real_python=sys.executable)
    _write_fake_python(fakebin / "python", version="3.9.6", real_python=sys.executable)

    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(["bash", str(script_path)], cwd=repo_root, env=env, capture_output=True, text=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.strip() == str(python314)
