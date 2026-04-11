from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAKEFILE_PATH = PROJECT_ROOT / "requirements" / "Makefile"


def _read_makefile() -> str:
    return MAKEFILE_PATH.read_text(encoding="utf-8")


def _target_body(name: str) -> str:
    text = _read_makefile()
    match = re.search(rf"^{re.escape(name)}:(?:[^\n]*)\n(?P<body>(?:\t.*\n)+)", text, flags=re.MULTILINE)
    assert match is not None, f"Makefile target {name} not found"
    return match.group("body")


def test_generic_targets_do_not_reference_target_owned_ml_locks() -> None:
    text = _read_makefile()
    assert "compile: compile-generic" in text
    assert "update: update-generic" in text
    assert "check: check-generic" in text

    for target in ("compile-generic", "update-generic", "check-generic"):
        body = _target_body(target)
        assert "ml-core-darwin-arm64.txt" not in body
        assert "ml-core-darwin-x86_64.txt" not in body
        assert "ml-core-linux.txt" not in body


def test_compile_ml_layers_refuses_broad_target_owned_regeneration() -> None:
    body = _target_body("compile-ml-layers")

    assert "target-owned ML locks require explicit authoritative-lane commands" in body
    assert "compile-ml-darwin-arm64" in body
    assert "compile-ml-linux-x86_64" in body


def _write_fake_uname(fakebin: Path, *, system: str, machine: str) -> None:
    (fakebin / "uname").write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  -s) echo {system} ;;\n"
        f"  -m) echo {machine} ;;\n"
        "  *) echo unsupported >&2; exit 1 ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    (fakebin / "uname").chmod(0o755)


def test_darwin_arm64_target_fails_closed_off_lane(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    requirements_dir = repo_root / "requirements"
    requirements_dir.mkdir(parents=True)
    (requirements_dir / "Makefile").write_text(MAKEFILE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    _write_fake_uname(fakebin, system="Linux", machine="x86_64")

    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(
        ["make", "compile-ml-darwin-arm64"],
        cwd=requirements_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "authoritative only on native Darwin arm64" in (result.stdout + result.stderr)


def test_linux_x86_64_target_fails_closed_off_lane(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    requirements_dir = repo_root / "requirements"
    requirements_dir.mkdir(parents=True)
    (requirements_dir / "Makefile").write_text(MAKEFILE_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    fakebin = tmp_path / "fakebin"
    fakebin.mkdir()
    _write_fake_uname(fakebin, system="Darwin", machine="arm64")

    env = {**os.environ, "PATH": f"{fakebin}:/usr/bin:/bin"}
    result = subprocess.run(
        ["make", "compile-ml-linux-x86_64"],
        cwd=requirements_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "authoritative only on native Linux x86_64" in (result.stdout + result.stderr)


def test_frozen_darwin_x86_64_target_exits_nonzero_with_frozen_message(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    requirements_dir = repo_root / "requirements"
    requirements_dir.mkdir(parents=True)
    (requirements_dir / "Makefile").write_text(MAKEFILE_PATH.read_text(encoding="utf-8"), encoding="utf-8")

    env = {**os.environ, "PATH": "/usr/bin:/bin"}
    result = subprocess.run(
        ["make", "compile-ml-darwin-x86_64"],
        cwd=requirements_dir,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "ml-core-darwin-x86_64.txt is frozen pending an authoritative Darwin x86_64 lane decision" in (
        result.stdout + result.stderr
    )
