#!/usr/bin/env python3
"""Check default image identity, offline bootstrap, and optional GPU packages."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path

BASE_PREFIX = Path("/opt/python3.11")
APP_PREFIX = Path("/opt/venv")
EXPECTED_HOME = Path("/home/tp")
LEGACY_PREFIX = Path("/usr/local/lib/python3.11")
BOOTSTRAP_VERSIONS = {"pip": "26.2.2", "setuptools": "83.0.0"}
GPU_VERSIONS = {"torch": "2.13.0+cu126", "torchvision": "0.28.0+cu126"}
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


def _installed_inventory(python: Path) -> dict:
    program = """
import hashlib, importlib.metadata as metadata, json
from pathlib import Path
packages = []
for distribution in metadata.distributions():
    package = {"name": distribution.metadata["Name"], "version": distribution.version}
    direct_url = distribution.read_text("direct_url.json")
    if direct_url:
        package["direct_url"] = json.loads(direct_url)
    packages.append(package)
ca = Path(metadata.distribution("pip").locate_file("pip/_vendor/certifi/cacert.pem"))
print(json.dumps({"packages": sorted(packages, key=lambda package: (package["name"].lower(), package["version"])),
                  "pip_ca_sha256": hashlib.sha256(ca.read_bytes()).hexdigest()}))
"""
    result = subprocess.run([str(python), "-I", "-c", program], check=True, capture_output=True, text=True, timeout=30)
    return json.loads(result.stdout)


def _bootstrap_wheel_inventory() -> list[dict]:
    inventory = []
    for path in sorted((BASE_PREFIX / "share/python-wheels").iterdir()):
        with zipfile.ZipFile(path) as archive:
            metadata_paths = [
                name for name in archive.namelist() if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
            ]
            _require(len(metadata_paths) == 1, f"Bootstrap wheel must contain one package metadata record: {path.name}")
            metadata = BytesParser().parsebytes(archive.read(metadata_paths[0]))
            _require(bool(metadata["Name"] and metadata["Version"]), f"Bootstrap wheel metadata is incomplete: {path.name}")
            package = {"filename": path.name, "name": metadata["Name"], "version": metadata["Version"]}
            with path.open("rb") as source:
                package["sha256"] = hashlib.file_digest(source, "sha256").hexdigest()
            ca = "pip/_vendor/certifi/cacert.pem"
            if ca in archive.namelist():
                package["pip_ca_sha256"] = hashlib.sha256(archive.read(ca)).hexdigest()
            inventory.append(package)
    return inventory


def _check_offline_venv(home: Path, *, inventories: dict | None = None) -> dict[str, str]:
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
        if inventories is not None:
            inventories["offline_venv"] = _installed_inventory(target / "bin/python")
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


def _check_gpu_packages() -> dict:
    for name, expected in GPU_VERSIONS.items():
        _require(importlib.metadata.version(name) == expected, f"GPU package {name} must be {expected}")
    torch = importlib.import_module("torch")
    torchvision = importlib.import_module("torchvision")
    _require(torch.__version__ == GPU_VERSIONS["torch"], "Imported torch version must match its CUDA package pin")
    _require(
        torchvision.__version__ == GPU_VERSIONS["torchvision"], "Imported torchvision version must match its CUDA package pin"
    )
    _require(torch.version.cuda == "12.6", "Torch must be built for CUDA 12.6")
    return {**GPU_VERSIONS, "cuda_build": torch.version.cuda, "hardware_validation": "not_performed"}


def check_runtime(expected_architecture: str, *, gpu_packages: bool = False) -> dict:
    home = _check_identity(expected_architecture)
    _check_prefix()
    _check_versions({name: importlib.metadata.version(name) for name in BOOTSTRAP_VERSIONS})
    _check_versions(_metadata_versions(BASE_PREFIX / "bin/python3.11"))
    inventories = {}
    offline_versions = _check_offline_venv(home, inventories=inventories if gpu_packages else None)
    for name in IMPORTS:
        importlib.import_module(name)
    _check_module_launcher()
    evidence = {
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
    if gpu_packages:
        evidence["gpu_packages"] = _check_gpu_packages()
        inventories["application"] = _installed_inventory(Path(sys.executable))
        inventories["base"] = _installed_inventory(BASE_PREFIX / "bin/python3.11")
        evidence["installed_inventory"] = inventories
        evidence["bootstrap_wheels"] = _bootstrap_wheel_inventory()
    return evidence


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-architecture", choices=sorted(MACHINES), required=True)
    parser.add_argument(
        "--gpu-packages", action="store_true", help="Verify CUDA package builds without requiring GPU hardware"
    )
    args = parser.parse_args(argv)
    try:
        evidence = check_runtime(args.expected_architecture, gpu_packages=args.gpu_packages)
    except (OSError, RuntimeError, ValueError, ImportError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        print(f"Docker runtime check failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
