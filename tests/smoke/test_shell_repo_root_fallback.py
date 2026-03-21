from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]


def test_shell_repo_root_fallback_without_git(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "scripts" / "pipelines" / "run_montecito_apex_lean.sh"
    fake_bin = tmp_path / "fake_bin"
    fake_bin.mkdir()
    fake_git = fake_bin / "git"
    fake_git.write_text("#!/usr/bin/env bash\nexit 127\n", encoding="utf-8")
    fake_git.chmod(fake_git.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    result = subprocess.run(
        ["bash", str(script_path)],
        cwd="/tmp",
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    output = f"{result.stdout}\n{result.stderr}"
    expected_input = f"{repo_root}/input_images/Montecito-Shores_press_300dpi_TIFFs"
    assert result.returncode != 0
    assert "Unable to determine repository root" not in output
    assert expected_input in output
