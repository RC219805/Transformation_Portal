"""Regression coverage for batch failures at both recipe CLI entry points."""

from __future__ import annotations

import sys
import types

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("entrypoint", ["main", "compatibility"])
@pytest.mark.parametrize("successful_count", [0, 2])
def test_batch_failures_exit_unsuccessfully(entrypoint, successful_count, tmp_path, monkeypatch):
    """Partial and total failures must be visible to shell automation."""
    from transformation_portal.__main__ import app
    from transformation_portal.cli import pipeline_app

    module_name = "transformation_portal.pipeline_unified"
    module = types.ModuleType(module_name)

    class FailingPipeline:
        @classmethod
        def from_recipe(cls, recipe_path):
            return cls()

        def process_batch(self, *args, **kwargs):
            return types.SimpleNamespace(successful_count=successful_count, failed_count=1, total_time=0.01)

    module.UnifiedPipeline = FailingPipeline
    monkeypatch.setitem(sys.modules, module_name, module)
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text("name: batch\nstages: [color_grading]\n", encoding="utf-8")
    result = CliRunner().invoke(
        app if entrypoint == "main" else pipeline_app,
        ["process", "--input", "*.jpg", "--recipe", str(recipe), "--output", str(tmp_path / "output")],
    )

    assert result.exit_code == 1
    assert "1 images failed" in result.output
    assert "Pipeline error:" not in result.output
