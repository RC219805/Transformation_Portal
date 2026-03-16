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
