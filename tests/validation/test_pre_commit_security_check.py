"""Regression tests for the legacy Unicode security-check entrypoint."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.security]
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "security" / "pre_commit_security_check.py"


def _run_legacy_cli(*paths: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT_PATH), *map(str, paths)], capture_output=True, text=True, check=False)


@pytest.mark.parametrize(
    ("contents", "diagnostic"),
    [
        (b"# hidden \xe2\x80\xae\n", "Bidirectional Unicode U+202E"),
        (b"# hidden \xe2\x80\x8b\n", "Format control character U+200B"),
        (b"# invalid \xff\n", "Error reading file:"),
    ],
)
def test_legacy_cli_blocks_controls_and_invalid_utf8(tmp_path: Path, contents: bytes, diagnostic: str) -> None:
    filepath = tmp_path / "example.txt"
    filepath.write_bytes(contents)

    result = _run_legacy_cli(filepath)

    assert result.returncode == 1
    assert diagnostic in result.stdout


def test_legacy_cli_blocks_missing_paths(tmp_path: Path) -> None:
    result = _run_legacy_cli(tmp_path / "missing.py")

    assert result.returncode == 1
    assert "Error reading file:" in result.stdout


def test_legacy_cli_allows_ordinary_unicode_and_spaced_paths(tmp_path: Path) -> None:
    filepath = tmp_path / "with spaces.py"
    filepath.write_text("# caf\u00e9\n", encoding="utf-8")

    result = _run_legacy_cli(filepath)

    assert result.returncode == 0
    assert "Checked 1 files - no security issues found" in result.stdout


def test_legacy_cli_preserves_no_argument_behavior() -> None:
    result = _run_legacy_cli()

    assert result.returncode == 0
    assert "No files to check" in result.stdout
