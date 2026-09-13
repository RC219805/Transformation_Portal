"""Regression tests for the legacy Unicode security-check entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from scripts.security.pre_commit_security_check import check_file

pytestmark = [pytest.mark.unit, pytest.mark.security]
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "security" / "pre_commit_security_check.py"


@pytest.mark.parametrize("contents", [b"# hidden \xe2\x80\xae\n", b"# hidden \xe2\x80\x8b\n", b"# invalid \xff\n"])
def test_legacy_check_blocks_controls_and_invalid_utf8(tmp_path: Path, contents: bytes) -> None:
    filepath = tmp_path / "example.txt"
    filepath.write_bytes(contents)

    passed, error = check_file(str(filepath))

    assert not passed
    assert error is not None


def test_legacy_cli_blocks_missing_paths(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(tmp_path / "missing.py")], capture_output=True, text=True, check=False
    )

    assert result.returncode == 1
    assert "Error reading file:" in result.stdout


def test_legacy_check_preserves_success_tuple(tmp_path: Path) -> None:
    filepath = tmp_path / "with spaces.py"
    filepath.write_text("# caf\u00e9\n", encoding="utf-8")

    assert check_file(str(filepath)) == (True, None)


def test_legacy_cli_preserves_no_argument_behavior() -> None:
    result = subprocess.run([sys.executable, str(SCRIPT_PATH)], capture_output=True, text=True, check=False)

    assert result.returncode == 0
    assert "No files to check" in result.stdout
