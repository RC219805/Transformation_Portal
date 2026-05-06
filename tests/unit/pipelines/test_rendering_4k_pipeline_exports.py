"""Compatibility tests for the Phase 4D rendering_4k pipeline extraction."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

PACKAGE_NAME = "transformation_portal.pipelines.rendering_4k"
PIPELINE_NAME = f"{PACKAGE_NAME}.pipeline"
LEGACY_NAME = "transformation_portal.pipelines.rendering_4k_pipeline"


def test_legacy_rendering_module_reexports_phase_4d_pipeline() -> None:
    legacy = importlib.import_module(LEGACY_NAME)
    extracted = importlib.import_module(PIPELINE_NAME)

    assert legacy.Rendering4KPipeline is extracted.Rendering4KPipeline
    assert legacy.main is extracted.main
    assert legacy._json_default is extracted._json_default
    assert legacy.logger is extracted.logger


def test_rendering_4k_package_lazily_reexports_phase_4d_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(sys.modules, PIPELINE_NAME, raising=False)
    monkeypatch.delitem(sys.modules, PACKAGE_NAME, raising=False)

    package = importlib.import_module(PACKAGE_NAME)

    assert PIPELINE_NAME not in sys.modules
    assert "Rendering4KPipeline" in package.__all__

    extracted = importlib.import_module(PIPELINE_NAME)
    assert getattr(package, "Rendering4KPipeline") is extracted.Rendering4KPipeline


def test_legacy_rendering_module_cli_dry_run_still_reaches_main(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    input_path = tmp_path / "input.png"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            LEGACY_NAME,
            "-i",
            str(input_path),
            "--preset",
            "preview",
            "--dry-run",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert '"name": "preview"' in f"{result.stdout}\n{result.stderr}"
