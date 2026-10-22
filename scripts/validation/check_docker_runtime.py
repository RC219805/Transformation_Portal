#!/usr/bin/env python3
"""Check the default CPU image identity and its preserved offline bootstrap."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

BASE_PREFIX = Path("/opt/python3.11")
APP_PREFIX = Path("/opt/venv")
EXPECTED_HOME = Path("/home/tp")
LEGACY_PREFIX = Path("/usr/local/lib/python3.11")
BOOTSTRAP_VERSIONS = {"pip": "26.2.2", "setuptools": "83.0.0"}
MACHINES = {"amd64": {"x86_64"}, "arm64": {"aarch64", "arm64"}}
IMPORTS = (
    "bz2",
    "ctypes",
    "curses",
    "dbm.gnu",
    "dbm.ndbm",
    "lzma",
    "readline",
    "sqlite3",
    "ssl",
    "uuid",
    "zlib",
    "numpy",
    "PIL",
    "scipy",
    "cv2",
    "tifffile",
    "imagecodecs",
    "fastapi",
    "uvicorn",
    "transformation_portal",
    "transformation_portal.cli",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _check_identity(expected_architecture: str) -> Path:
    version = sys.version_info
    _require((version.major, version.minor, version.releaselevel) == (3, 11, "final"), "Python must be a final 3.11 release")
    _require(platform.system() == "Linux", "Runtime must be Linux")
    _require(platform.machine() in MACHINES[expected_architecture], "Runtime architecture does not match the requested target")
    _require(os.getuid() == 10001 and os.getgid() == 10001, "Default runtime UID/GID must be 10001")
    _require(os.environ.get("HOME") == str(EXPECTED_HOME) and Path.home() == EXPECTED_HOME, "Default HOME must be /home/tp")
    _require(
        Path(sys.prefix) == APP_PREFIX and Path(sys.base_prefix) == BASE_PREFIX,
        "Runtime must use the clean /opt application and Python prefixes",
    )
    return EXPECTED_HOME


def _check_prefix() -> None:
    _require(not LEGACY_PREFIX.exists(), "Historical /usr/local Python prefix must be absent")
    _require(not (BASE_PREFIX / "lib/python3.11/ensurepip/_bundled").exists(), "Historical ensurepip bundles must be absent")
    wheels = BASE_PREFIX / "share/python-wheels"
    _require(
        sysconfig.get_config_var("WHEEL_PKG_DIR") == str(wheels), "CPython must preserve the patched offline wheel directory"
    )
    expected = {f"{name}-{version}-py3-none-any.whl" for name, version in BOOTSTRAP_VERSIONS.items()}
    _require(
        {path.name for path in wheels.iterdir()} == expected,
        "Offline wheel directory must contain only current bootstrap wheels",
    )


def _check_versions(versions: dict[str, str]) -> None:
    _require(versions == BOOTSTRAP_VERSIONS, f"Bootstrap versions must match {BOOTSTRAP_VERSIONS}; found {versions}")


def _metadata_versions(python: Path) -> dict[str, str]:
    program = (
        "import importlib.metadata as m, json; " "print(json.dumps({name: m.version(name) for name in ('pip', 'setuptools')}))"
    )
    result = subprocess.run([str(python), "-I", "-c", program], check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def _check_offline_venv(home: Path) -> dict[str, str]:
    with tempfile.TemporaryDirectory(prefix="tp-docker-runtime-", dir=home) as directory:
        target = Path(directory) / "venv"
        subprocess.run(
            [str(BASE_PREFIX / "bin/python3.11"), "-I", "-m", "venv", str(target)],
            check=True,
            timeout=90,
            env={**os.environ, "PIP_NO_INDEX": "1"},
        )
        versions = _metadata_versions(target / "bin/python")
        _check_versions(versions)
    return versions


def _check_module_launcher() -> None:
    result = subprocess.run(
        [sys.executable, "-I", "-m", "transformation_portal.cli", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    _require(result.returncode == 0, f"Default CLI module launcher failed: {result.stderr.strip()}")
    _require(bool(result.stdout.strip()), "Default CLI module launcher returned empty help")


def check_runtime(expected_architecture: str) -> dict:
    home = _check_identity(expected_architecture)
    _check_prefix()
    _check_versions({name: importlib.metadata.version(name) for name in BOOTSTRAP_VERSIONS})
    _check_versions(_metadata_versions(BASE_PREFIX / "bin/python3.11"))
    offline_versions = _check_offline_venv(home)
    for name in IMPORTS:
        importlib.import_module(name)
    _check_module_launcher()
    return {
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "home": str(home),
        "bootstrap_versions": BOOTSTRAP_VERSIONS,
        "offline_venv_versions": offline_versions,
        "imports": list(IMPORTS),
        "module_launcher_help": "passed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", choices=sorted(MACHINES), required=True)
    args = parser.parse_args(argv)
    try:
        evidence = check_runtime(args.expected_architecture)
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.SubprocessError) as exc:
        print(f"Docker runtime check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
