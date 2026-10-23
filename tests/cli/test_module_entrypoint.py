"""Exercise the module launcher used by the container runtime command."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.regression]
REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("arguments", "exit_code", "expected_output"),
    [
        (["--help"], 0, "Professional image and video processing toolkit"),
        (["version"], 0, "Transformation Portal v"),
        (["unknown-command"], 2, "No such command"),
    ],
)
def test_module_launcher_preserves_cli_behavior(arguments: list[str], exit_code: int, expected_output: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "transformation_portal.cli", *arguments],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src"), "NO_COLOR": "1"},
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == exit_code, result.stdout + result.stderr
    assert expected_output in result.stdout + result.stderr
