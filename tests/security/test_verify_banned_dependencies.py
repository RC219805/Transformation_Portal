"""Regression coverage for the banned-dependency gate (scripts/security/verify_banned_dependencies.py)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE = REPO_ROOT / "scripts" / "security" / "verify_banned_dependencies.py"


def _sandbox(tmp_path: Path) -> Path:
    (tmp_path / "scripts" / "security").mkdir(parents=True)
    (tmp_path / "requirements" / "locks").mkdir(parents=True)
    shutil.copy(REPO_ROOT / "scripts/security/banned_dependencies.json", tmp_path / "scripts/security/")
    shutil.copy(REPO_ROOT / "requirements/constraints.txt", tmp_path / "requirements/")
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0"\ndependencies = ["numpy"]\n')
    return tmp_path


def _run_gate(cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(GATE), "--strict"],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def test_clean_sandbox_passes_with_constraints_hard_blocks(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    (root / "requirements" / "base.txt").write_text("numpy==2.0.0  # via -r base.in\n")
    result = _run_gate(root)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No banned dependencies found" in result.stdout


def test_root_requirement_files_are_scanned(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    (root / "requirements-ci.txt").write_text("basicsr==1.4.2\n")
    (root / "requirements.txt").write_text("gfpgan\n")
    result = _run_gate(root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "requirements-ci.txt:1 - basicsr" in result.stdout
    assert "requirements.txt:1 - gfpgan" in result.stdout


def test_lock_sources_and_nested_lock_directories_are_scanned(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    (root / "requirements" / "ml.in").write_text("-c constraints.txt\nrealesrgan\n")
    (root / "requirements" / "locks" / "ml-linux.txt").write_text("basicsr==1.4.2\n")
    result = _run_gate(root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "ml.in:2 - realesrgan" in result.stdout
    assert "ml-linux.txt:1 - basicsr" in result.stdout


def test_pep503_name_variants_and_egg_fragments_are_detected(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    (root / "requirements" / "base.txt").write_text(
        "Apache_Iceberg==1.0\n"
        "apache.iceberg==1.0\n"
        "-e git+https://example.invalid/Real-ESRGAN.git#egg=RealESRGAN\n"
        "PyIceberg@https://example.invalid/pyiceberg.tar.gz\n"
    )
    (root / "pyproject.toml").write_text('[project]\nname = "x"\nversion = "0"\ndependencies = ["Apache.Iceberg>=1"]\n')
    result = _run_gate(root)
    assert result.returncode == 1, result.stdout + result.stderr
    for expected in (
        "base.txt:1 - Apache_Iceberg",
        "base.txt:2 - apache.iceberg",
        "base.txt:3 - RealESRGAN",
        "base.txt:4 - PyIceberg",
        "pyproject.toml - Apache.Iceberg",
    ):
        assert expected in result.stdout


def test_constraints_hard_block_pins_remain_exempt_but_other_pins_fail(tmp_path: Path) -> None:
    root = _sandbox(tmp_path)
    constraints = root / "requirements" / "constraints.txt"
    constraints.write_text(constraints.read_text() + "RealESRGAN==0.3.0\n")
    result = _run_gate(root)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "constraints.txt" in result.stdout
    assert "RealESRGAN" in result.stdout
