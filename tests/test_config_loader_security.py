"""Tests for config_loader security enhancements (SEC-002)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from transformation_portal.config_loader import _expand_env_vars, load_recipe


def test_expand_env_vars_basic():
    """Test basic environment variable expansion."""
    os.environ["TEST_VAR"] = "/test/path"
    try:
        result = _expand_env_vars("${TEST_VAR}/config")
        assert result == "/test/path/config"
    finally:
        del os.environ["TEST_VAR"]


def test_expand_env_vars_with_traversal(caplog):
    """Test that path traversal in env vars is logged but not blocked."""
    import logging

    caplog.set_level(logging.WARNING)

    os.environ["TEST_VAR"] = "../../../etc"
    try:
        # Should expand
        result = _expand_env_vars("${TEST_VAR}/passwd")
        assert result == "../../../etc/passwd"

        # Verify no excessive traversal warning (only 3 levels)
        assert "excessive parent traversal" not in caplog.text.lower()
    finally:
        del os.environ["TEST_VAR"]


def test_expand_env_vars_excessive_traversal(caplog):
    """Test that excessive parent directory traversal is detected."""
    import logging

    caplog.set_level(logging.WARNING)

    os.environ["TEST_VAR"] = "../../../../../../../../../../etc"
    try:
        # Should expand but log warning (>5 levels)
        result = _expand_env_vars("${TEST_VAR}/passwd")
        assert "../" in result

        # Verify warning was logged
        assert len(caplog.records) > 0
        assert "excessive parent traversal" in caplog.text.lower()
    finally:
        del os.environ["TEST_VAR"]


def test_expand_env_vars_non_path():
    """Test that non-path env vars are expanded without validation."""
    os.environ["TEST_VAR"] = "simple_value"
    try:
        result = _expand_env_vars("${TEST_VAR}")
        assert result == "simple_value"
    finally:
        del os.environ["TEST_VAR"]


def test_load_recipe_basic():
    """Test basic recipe loading functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recipe_path = Path(tmpdir) / "test_recipe.yaml"
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


def test_load_recipe_with_env_expansion():
    """Test recipe loading with environment variable expansion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recipe_path = Path(tmpdir) / "test_recipe.yaml"
        os.environ["TEST_OUTPUT_DIR"] = "/tmp/output"
        try:
            recipe_content = """
name: Test Recipe with Env
stages:
  - depth_estimation
output_dir: ${TEST_OUTPUT_DIR}
"""
            recipe_path.write_text(recipe_content)

            recipe = load_recipe(recipe_path)
            assert recipe["output_dir"] == "/tmp/output"
        finally:
            del os.environ["TEST_OUTPUT_DIR"]


def test_load_recipe_file_not_found():
    """Test that FileNotFoundError is raised for missing files."""
    with pytest.raises(FileNotFoundError, match="Recipe file not found"):
        load_recipe("/nonexistent/recipe.yaml")


def test_load_recipe_empty_file():
    """Test that ValueError is raised for empty recipe files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recipe_path = Path(tmpdir) / "empty.yaml"
        recipe_path.write_text("")

        with pytest.raises(ValueError, match="Recipe file is empty"):
            load_recipe(recipe_path)


def test_load_recipe_invalid_type():
    """Test that ValueError is raised for non-dict recipes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        recipe_path = Path(tmpdir) / "invalid.yaml"
        recipe_path.write_text("- item1\n- item2\n")

        with pytest.raises(ValueError, match="Recipe must be a dictionary"):
            load_recipe(recipe_path)
