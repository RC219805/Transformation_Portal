"""Tests for config_loader security enhancements (SEC-002)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Pytest markers
pytestmark = [
    pytest.mark.unit,
    pytest.mark.security,
]

from transformation_portal.config_loader import _expand_env_vars, load_recipe


def test_expand_env_vars_basic(monkeypatch):
    """Test basic environment variable expansion."""
    monkeypatch.setenv("TEST_VAR", "/test/path")
    result = _expand_env_vars("${TEST_VAR}/config")
    assert result == "/test/path/config"


def test_expand_env_vars_with_traversal(caplog, monkeypatch):
    """Test that path traversal in env vars is logged but not blocked."""
    import logging

    caplog.set_level(logging.WARNING)

    monkeypatch.setenv("TEST_VAR", "../../../etc")
    # Should expand
    result = _expand_env_vars("${TEST_VAR}/passwd")
    assert result == "../../../etc/passwd"

    # Verify no excessive traversal warning (only 3 levels)
    assert "excessive parent traversal" not in caplog.text.lower()


def test_expand_env_vars_excessive_traversal(caplog, monkeypatch):
    """Test that excessive parent directory traversal is detected."""
    import logging

    caplog.set_level(logging.WARNING)

    monkeypatch.setenv("TEST_VAR", "../../../../../../../../../../etc")
    # Should expand but log warning (>5 levels)
    result = _expand_env_vars("${TEST_VAR}/passwd")
    assert "../" in result

    # Verify warning was logged
    assert len(caplog.records) > 0
    assert "excessive parent traversal" in caplog.text.lower()


def test_expand_env_vars_non_path(monkeypatch):
    """Test that non-path env vars are expanded without validation."""
    monkeypatch.setenv("TEST_VAR", "simple_value")
    result = _expand_env_vars("${TEST_VAR}")
    assert result == "simple_value"


def test_load_recipe_basic(temp_workspace):
    """Test basic recipe loading functionality."""
    recipe_path = temp_workspace["root"] / "test_recipe.yaml"
    recipe_content = """
name: Test Recipe
stages:
  - depth_estimation
  - color_grading
"""
    recipe_path.write_text(recipe_content)

    recipe = load_recipe(recipe_path)
    assert recipe["name"] == "Test Recipe"
    assert "depth_estimation" in recipe["stages"]
    assert "_recipe_path" in recipe


def test_load_recipe_with_env_expansion(temp_workspace, monkeypatch):
    """Test recipe loading with environment variable expansion."""
    recipe_path = temp_workspace["root"] / "test_recipe.yaml"
    monkeypatch.setenv("TEST_OUTPUT_DIR", "/tmp/output")
    recipe_content = """
name: Test Recipe with Env
stages:
  - depth_estimation
output_dir: ${TEST_OUTPUT_DIR}
"""
    recipe_path.write_text(recipe_content)

    recipe = load_recipe(recipe_path)
    assert recipe["output_dir"] == "/tmp/output"


def test_load_recipe_file_not_found():
    """Test that FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError, match="Recipe file not found"):
        load_recipe("/nonexistent/recipe.yaml")


def test_load_recipe_empty_file(temp_workspace):
    """Test that ValueError is raised for empty recipe files."""
    recipe_path = temp_workspace["root"] / "empty.yaml"
    recipe_path.write_text("")

    with pytest.raises(ValueError, match="Recipe file is empty"):
        load_recipe(recipe_path)


def test_load_recipe_invalid_type(temp_workspace):
    """Test that ValueError is raised for non-dict recipes."""
    recipe_path = temp_workspace["root"] / "invalid.yaml"
    recipe_path.write_text("- item1\n- item2\n")

    with pytest.raises(ValueError, match="Recipe must be a dictionary"):
        load_recipe(recipe_path)


@pytest.mark.parametrize(
    ("recipe", "expected_error"),
    [
        ({"name": "x", "stages": None}, "Field 'stages' must be a list"),
        ({"name": "x", "stages": 5}, "Field 'stages' must be a list"),
        ({"name": "x", "stages": [{"color_grading": {"contrast": 1.1}}]}, "Invalid stage entry"),
        ({"name": "x", "stages": ["color_grading"], "output": {"format": 16}}, "Output 'format' must be a string"),
        ({"name": "x", "stages": ["color_grading"], "output": "png"}, "Section 'output' must be a mapping"),
        ({"name": "x", "stages": ["color_grading"], "color_grading": True}, "Section 'color_grading' must be a mapping"),
        ({"name": 2024, "stages": ["color_grading"]}, "Field 'name' must be a string"),
        (
            {"name": "x", "stages": ["quality_assessment"], "quality_assessment": 5},
            "Section 'quality_assessment' must be a mapping",
        ),
    ],
)
def test_validate_recipe_reports_malformed_shapes_instead_of_raising(recipe, expected_error):
    """D-08: malformed recipes yield validation errors, never a raw exception."""
    from transformation_portal.config_loader import validate_recipe

    is_valid, errors = validate_recipe(recipe)

    assert is_valid is False
    assert any(expected_error in error for error in errors), errors


def test_validate_recipe_accepts_null_sections_and_unused_non_mapping_sections():
    """A section whose options are all commented out (YAML null) stays valid."""
    from transformation_portal.config_loader import validate_recipe

    recipe = {"name": "x", "stages": ["color_grading"], "color_grading": None, "branding": "unused"}

    assert validate_recipe(recipe) == (True, [])


def test_validate_recipe_reports_each_malformed_section_once():
    """A listed stage that is also always read (output) is reported a single time."""
    from transformation_portal.config_loader import validate_recipe

    recipe = {"name": "x", "stages": ["output", "color_grading", "color_grading"], "output": "png", "color_grading": 1}

    assert validate_recipe(recipe) == (
        False,
        ["Section 'output' must be a mapping, got str", "Section 'color_grading' must be a mapping, got int"],
    )


def test_resolve_relative_paths_matches_path_words_not_substrings(tmp_path: Path):
    """D-09: profile / logo_text / text_direction are not paths; real path keys still resolve."""
    from transformation_portal.config_loader import _resolve_relative_paths

    (tmp_path / "luxury").mkdir()
    obj = {
        "material_response": {"profile": "luxury"},
        "branding": {"logo_text": "Smith/Jones Realty", "text_direction": "left/right", "logo": "brand/logo.png"},
        "color_grading": {"lut": "luts/a.cube", "lut_path": "luts/b.cube"},
        "quality_feedback": {"rag_index_path": "rag/index.jsonl", "filepath": "data/x.jpg", "output_dir": "out/x"},
    }

    result = _resolve_relative_paths(obj, tmp_path)

    assert result["material_response"]["profile"] == "luxury"
    assert result["branding"]["logo_text"] == "Smith/Jones Realty"
    assert result["branding"]["text_direction"] == "left/right"
    assert result["branding"]["logo"] == str(tmp_path / "brand/logo.png")
    assert result["color_grading"]["lut"] == str(tmp_path / "luts/a.cube")
    assert result["color_grading"]["lut_path"] == str(tmp_path / "luts/b.cube")
    assert result["quality_feedback"]["rag_index_path"] == str(tmp_path / "rag/index.jsonl")
    assert result["quality_feedback"]["filepath"] == str(tmp_path / "data/x.jpg")
    assert result["quality_feedback"]["output_dir"] == str(tmp_path / "out/x")
