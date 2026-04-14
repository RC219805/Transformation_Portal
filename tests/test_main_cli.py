from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pytest
from typer.testing import CliRunner

pytestmark = pytest.mark.unit


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def _write_recipe(path: Path, contents: str) -> Path:
    path.write_text(contents, encoding="utf-8")
    return path


def test_list_recipes_default_discovery_falls_back_to_config_recursively(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.__main__ import app

    recipe_path = tmp_path / "config" / "nested" / "recipe.yaml"
    recipe_path.parent.mkdir(parents=True)
    _write_recipe(
        recipe_path,
        """
name: Recursive Recipe
description: discovered via config fallback
stages:
  - color_grading
output:
  format: png
""".strip(),
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-recipes"])

    assert result.exit_code == 0
    assert "Recursive Recipe" in result.stdout
    assert str(recipe_path.relative_to(tmp_path)) in result.stdout


def test_list_recipes_explicit_dir_lists_only_recipe_candidates(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    from transformation_portal.__main__ import app

    recipes_dir = tmp_path / "recipes"
    recipes_dir.mkdir()
    _write_recipe(
        recipes_dir / "valid.yaml",
        """
name: Explicit Recipe
stages:
  - color_grading
""".strip(),
    )
    _write_recipe(recipes_dir / "ignored.yaml", "description: not a recipe\n")
    _write_recipe(
        recipes_dir / "invalid_shape.yaml",
        """
name: Almost Recipe
stages: color_grading
""".strip(),
    )

    result = runner.invoke(app, ["list-recipes", "--dir", str(recipes_dir)])

    assert result.exit_code == 0
    assert "Explicit Recipe" in result.stdout
    assert "ignored.yaml" not in result.stdout
    assert "Almost Recipe" not in result.stdout


def test_list_recipes_without_candidates_exits_cleanly(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.__main__ import app

    (tmp_path / "config").mkdir()
    _write_recipe(tmp_path / "config" / "manifest.yaml", "description: still not a recipe\n")
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["list-recipes"])

    assert result.exit_code == 0
    assert "No recipe presets found" in result.stdout


def test_validate_recipe_invalid_does_not_emit_duplicate_error(
    runner: CliRunner,
    tmp_path: Path,
) -> None:
    from transformation_portal.__main__ import app

    recipe_path = _write_recipe(tmp_path / "invalid.yaml", "description: missing recipe fields\n")

    result = runner.invoke(app, ["validate-recipe", str(recipe_path)])

    assert result.exit_code == 1
    assert "Recipe validation failed" in result.stdout
    assert "Error validating recipe" not in result.stdout


def test_info_stays_green_when_pipeline_probe_hits_non_import_error(
    runner: CliRunner,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.__main__ import app

    original_import = builtins.__import__

    def _raising_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if name == "transformation_portal.pipeline_unified":
            raise FileNotFoundError("tempdir unavailable")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", _raising_import)

    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert "FileNotFoundError: tempdir unavailable" in result.stdout
    assert "Core Dependencies:" in result.stdout


def test_process_parallel_forwards_flag(
    runner: CliRunner,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from transformation_portal.__main__ import app

    captured: dict[str, object] = {}
    stub_module = types.ModuleType("transformation_portal.pipeline_unified")

    class StubPipeline:
        @classmethod
        def from_recipe(cls, recipe_path):
            captured["recipe"] = str(recipe_path)
            return cls()

        def process_batch(self, input_glob, output_dir, **kwargs):
            captured["input_glob"] = input_glob
            captured["output_dir"] = str(output_dir)
            captured["kwargs"] = kwargs
            return types.SimpleNamespace(successful_count=1, failed_count=0, total_time=0.01)

    stub_module.UnifiedPipeline = StubPipeline
    monkeypatch.setitem(sys.modules, "transformation_portal.pipeline_unified", stub_module)

    recipe_path = _write_recipe(
        tmp_path / "recipe.yaml",
        """
name: Process Recipe
stages:
  - color_grading
""".strip(),
    )

    result = runner.invoke(
        app,
        [
            "process",
            "--input",
            "*.jpg",
            "--output",
            str(tmp_path / "outputs"),
            "--recipe",
            str(recipe_path),
            "--parallel",
        ],
    )

    assert result.exit_code == 0
    assert captured["kwargs"]["parallel"] is True
