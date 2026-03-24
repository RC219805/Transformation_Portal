"""Comprehensive tests for config_loader module.

Provides coverage for recipe loading, validation, and utility functions.

Coverage Target: 80% of config_loader.py (148 statements)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml

pytestmark = [
    pytest.mark.unit,
]


class TestExpandEnvVarsRecursive:
    """Tests for _expand_env_vars_recursive function."""

    def test_expand_string(self, monkeypatch):
        """Test expansion in string values."""
        from transformation_portal.config_loader import _expand_env_vars_recursive

        monkeypatch.setenv("TEST_VAR", "expanded_value")

        result = _expand_env_vars_recursive("prefix_${TEST_VAR}_suffix")

        assert result == "prefix_expanded_value_suffix"

    def test_expand_in_dict(self, monkeypatch):
        """Test expansion in dictionary values."""
        from transformation_portal.config_loader import _expand_env_vars_recursive

        monkeypatch.setenv("PATH_VAR", "/data/output")

        obj = {
            "output_dir": "${PATH_VAR}",
            "name": "static_name",
            "nested": {"path": "${PATH_VAR}/nested"},
        }

        result = _expand_env_vars_recursive(obj)

        assert result["output_dir"] == "/data/output"
        assert result["name"] == "static_name"
        assert result["nested"]["path"] == "/data/output/nested"

    def test_expand_in_list(self, monkeypatch):
        """Test expansion in list values."""
        from transformation_portal.config_loader import _expand_env_vars_recursive

        monkeypatch.setenv("ITEM_VAR", "item_value")

        obj = ["${ITEM_VAR}", "static", {"key": "${ITEM_VAR}"}]

        result = _expand_env_vars_recursive(obj)

        assert result[0] == "item_value"
        assert result[1] == "static"
        assert result[2]["key"] == "item_value"

    def test_non_string_passthrough(self):
        """Test that non-string values are passed through."""
        from transformation_portal.config_loader import _expand_env_vars_recursive

        obj = {"number": 42, "boolean": True, "none": None}

        result = _expand_env_vars_recursive(obj)

        assert result["number"] == 42
        assert result["boolean"] is True
        assert result["none"] is None

    def test_dollar_sign_syntax(self, monkeypatch):
        """Test $VAR syntax (without braces)."""
        from transformation_portal.config_loader import _expand_env_vars_recursive

        monkeypatch.setenv("SIMPLE_VAR", "simple")

        result = _expand_env_vars_recursive("$SIMPLE_VAR/path")

        assert result == "simple/path"

    def test_unexpanded_var_preserved(self):
        """Test that undefined vars are preserved."""
        from transformation_portal.config_loader import _expand_env_vars_recursive

        # Ensure var doesn't exist
        os.environ.pop("UNDEFINED_VAR", None)

        result = _expand_env_vars_recursive("${UNDEFINED_VAR}/path")

        assert result == "${UNDEFINED_VAR}/path"


class TestResolveRelativePaths:
    """Tests for _resolve_relative_paths function."""

    def test_resolve_path_key(self, temp_workspace):
        """Test resolution of keys containing 'path'."""
        from transformation_portal.config_loader import _resolve_relative_paths

        base_dir = temp_workspace["root"]

        obj = {"input_path": "images/test.jpg"}

        result = _resolve_relative_paths(obj, base_dir)

        assert str(base_dir / "images/test.jpg") == result["input_path"]

    def test_resolve_file_key(self, temp_workspace):
        """Test resolution of keys containing 'file'."""
        from transformation_portal.config_loader import _resolve_relative_paths

        base_dir = temp_workspace["root"]

        obj = {"output_file": "results/output.tiff"}

        result = _resolve_relative_paths(obj, base_dir)

        assert str(base_dir / "results/output.tiff") == result["output_file"]

    def test_resolve_dir_key(self, temp_workspace):
        """Test resolution of keys containing 'dir'."""
        from transformation_portal.config_loader import _resolve_relative_paths

        base_dir = temp_workspace["root"]

        obj = {"output_dir": "results"}

        result = _resolve_relative_paths(obj, base_dir)

        # Will be resolved if path separator exists or it's a relative path
        # Since "results" doesn't contain a separator, check behavior
        assert "results" in result["output_dir"]

    def test_resolve_lut_key(self, temp_workspace):
        """Test resolution of keys containing 'lut'."""
        from transformation_portal.config_loader import _resolve_relative_paths

        base_dir = temp_workspace["root"]

        obj = {"lut": "assets/color.cube"}

        result = _resolve_relative_paths(obj, base_dir)

        assert str(base_dir / "assets/color.cube") == result["lut"]

    def test_resolve_logo_key(self, temp_workspace):
        """Test resolution of keys containing 'logo'."""
        from transformation_portal.config_loader import _resolve_relative_paths

        base_dir = temp_workspace["root"]

        obj = {"logo": "branding/logo.png"}

        result = _resolve_relative_paths(obj, base_dir)

        assert str(base_dir / "branding/logo.png") == result["logo"]

    def test_absolute_path_unchanged(self, temp_workspace):
        """Test that absolute paths are not modified."""
        from transformation_portal.config_loader import _resolve_relative_paths

        base_dir = temp_workspace["root"]

        obj = {"input_path": "/absolute/path/to/file.jpg"}

        result = _resolve_relative_paths(obj, base_dir)

        assert result["input_path"] == "/absolute/path/to/file.jpg"

    def test_env_var_preserved(self, temp_workspace):
        """Test that env var references are preserved."""
        from transformation_portal.config_loader import _resolve_relative_paths

        base_dir = temp_workspace["root"]

        obj = {"output_path": "${HOME}/data"}

        result = _resolve_relative_paths(obj, base_dir)

        assert result["output_path"] == "${HOME}/data"

    def test_nested_resolution(self, temp_workspace):
        """Test resolution in nested structures."""
        from transformation_portal.config_loader import _resolve_relative_paths

        base_dir = temp_workspace["root"]

        obj = {
            "color_grading": {"lut_path": "luts/signature.cube"},
            "branding": {"logo_path": "brand/logo.png"},
        }

        result = _resolve_relative_paths(obj, base_dir)

        assert str(base_dir / "luts/signature.cube") == result["color_grading"]["lut_path"]
        assert str(base_dir / "brand/logo.png") == result["branding"]["logo_path"]

    def test_list_resolution(self, temp_workspace):
        """Test resolution in list values."""
        from transformation_portal.config_loader import _resolve_relative_paths

        base_dir = temp_workspace["root"]

        obj = {"items": [{"file_path": "data/item.jpg"}]}

        result = _resolve_relative_paths(obj, base_dir)

        assert str(base_dir / "data/item.jpg") == result["items"][0]["file_path"]


class TestLoadRecipe:
    """Tests for load_recipe function."""

    def test_load_valid_recipe(self, temp_workspace):
        """Test loading a valid recipe file."""
        from transformation_portal.config_loader import load_recipe

        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test Recipe
description: A test recipe
stages:
  - depth_estimation
  - color_grading
"""
        )

        recipe = load_recipe(recipe_path)

        assert recipe["name"] == "Test Recipe"
        assert recipe["description"] == "A test recipe"
        assert "depth_estimation" in recipe["stages"]
        assert "_recipe_path" in recipe
        assert "_recipe_dir" in recipe

    def test_load_recipe_not_found(self):
        """Test loading nonexistent recipe raises FileNotFoundError."""
        from transformation_portal.config_loader import load_recipe

        with pytest.raises(FileNotFoundError, match="Recipe file not found"):
            load_recipe(Path("/nonexistent/recipe.yaml"))

    def test_load_empty_recipe(self, temp_workspace):
        """Test loading empty recipe raises ValueError."""
        from transformation_portal.config_loader import load_recipe

        recipe_path = temp_workspace["root"] / "empty.yaml"
        recipe_path.write_text("")

        with pytest.raises(ValueError, match="Recipe file is empty"):
            load_recipe(recipe_path)

    def test_load_non_dict_recipe(self, temp_workspace):
        """Test loading non-dict recipe raises ValueError."""
        from transformation_portal.config_loader import load_recipe

        recipe_path = temp_workspace["root"] / "list.yaml"
        recipe_path.write_text("- item1\n- item2\n")

        with pytest.raises(ValueError, match="Recipe must be a dictionary"):
            load_recipe(recipe_path)

    def test_load_recipe_with_env_expansion(self, temp_workspace, monkeypatch):
        """Test loading recipe with environment variable expansion."""
        from transformation_portal.config_loader import load_recipe

        monkeypatch.setenv("TEST_OUTPUT", "/custom/output")

        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test
stages:
  - depth_estimation
output_dir: ${TEST_OUTPUT}
"""
        )

        recipe = load_recipe(recipe_path, expand_env=True)

        assert recipe["output_dir"] == "/custom/output"

    def test_load_recipe_no_env_expansion(self, temp_workspace, monkeypatch):
        """Test loading recipe without environment variable expansion."""
        from transformation_portal.config_loader import load_recipe

        monkeypatch.setenv("TEST_OUTPUT", "/custom/output")

        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test
stages:
  - depth_estimation
output_dir: ${TEST_OUTPUT}
"""
        )

        recipe = load_recipe(recipe_path, expand_env=False)

        assert recipe["output_dir"] == "${TEST_OUTPUT}"

    def test_load_recipe_with_path_resolution(self, temp_workspace):
        """Test loading recipe with relative path resolution."""
        from transformation_portal.config_loader import load_recipe

        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test
stages:
  - color_grading
color_grading:
  lut_path: assets/luts/signature.cube
"""
        )

        recipe = load_recipe(recipe_path, resolve_paths=True)

        expected_path = str(temp_workspace["root"] / "assets/luts/signature.cube")
        assert recipe["color_grading"]["lut_path"] == expected_path

    def test_load_recipe_no_path_resolution(self, temp_workspace):
        """Test loading recipe without relative path resolution."""
        from transformation_portal.config_loader import load_recipe

        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test
stages:
  - color_grading
color_grading:
  lut_path: assets/luts/signature.cube
"""
        )

        recipe = load_recipe(recipe_path, resolve_paths=False)

        assert recipe["color_grading"]["lut_path"] == "assets/luts/signature.cube"

    def test_load_recipe_stores_path_info(self, temp_workspace):
        """Test that loaded recipe stores path information."""
        from transformation_portal.config_loader import load_recipe

        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test
stages:
  - depth_estimation
"""
        )

        recipe = load_recipe(recipe_path)

        assert recipe["_recipe_path"] == str(recipe_path.resolve())
        assert recipe["_recipe_dir"] == str(recipe_path.parent.resolve())


class TestValidateRecipe:
    """Tests for validate_recipe function."""

    def test_valid_minimal_recipe(self):
        """Test validation of minimal valid recipe."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["depth_estimation"],
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is True
        assert errors == []

    def test_missing_name(self):
        """Test validation catches missing name."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {"stages": ["depth_estimation"]}

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("name" in e.lower() for e in errors)

    def test_missing_stages(self):
        """Test validation catches missing stages."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {"name": "Test"}

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("stages" in e.lower() for e in errors)

    def test_stages_not_list(self):
        """Test validation catches non-list stages."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": "depth_estimation",
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("list" in e.lower() for e in errors)

    def test_empty_stages(self):
        """Test validation catches empty stages list."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": [],
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("empty" in e.lower() for e in errors)

    def test_invalid_stage_name(self):
        """Test validation catches invalid stage names."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["invalid_stage_xyz"],
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("invalid_stage_xyz" in e for e in errors)

    def test_valid_stage_names(self):
        """Test validation accepts all valid stage names."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": [
                "depth_estimation",
                "ai_enhancement",
                "material_response",
                "color_grading",
                "photo_finishing",
                "branding",
                "output",
                "upscaling_4k",
                "quality_assessment",
            ],
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is True

    def test_material_response_range_validation(self):
        """Test validation of material_response numeric ranges."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["material_response"],
            "material_response": {
                "texture_boost": 1.5,  # Out of range [0, 1]
            },
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("texture_boost" in e for e in errors)

    def test_material_response_enabled_not_bool(self):
        """Test validation catches non-boolean enabled field."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["material_response"],
            "material_response": {
                "enabled": "yes",  # Should be boolean
            },
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("enabled" in e.lower() and "boolean" in e.lower() for e in errors)

    def test_color_grading_lut_strength_validation(self):
        """Test validation of color_grading lut_strength range."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["color_grading"],
            "color_grading": {
                "lut_strength": 2.0,  # Out of range [0, 1]
            },
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("lut_strength" in e for e in errors)

    def test_color_grading_contrast_validation(self):
        """Test validation of color_grading contrast range."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["color_grading"],
            "color_grading": {
                "contrast": 0.1,  # Out of range [0.5, 2.0]
            },
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("contrast" in e for e in errors)

    def test_output_format_validation(self):
        """Test validation of output format."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["output"],
            "output": {
                "format": "gif",  # Invalid format
            },
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("format" in e.lower() for e in errors)

    def test_output_quality_validation(self):
        """Test validation of output quality."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["output"],
            "output": {
                "quality": 150,  # Out of range [1, 100]
            },
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("quality" in e.lower() for e in errors)

    def test_output_quality_not_int(self):
        """Test validation catches non-integer quality."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["output"],
            "output": {
                "quality": "high",  # Should be integer
            },
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is False
        assert any("quality" in e.lower() and "integer" in e.lower() for e in errors)

    def test_output_format_normalized(self):
        """Test that output format is normalized to lowercase."""
        from transformation_portal.config_loader import validate_recipe

        recipe = {
            "name": "Test",
            "stages": ["output"],
            "output": {
                "format": "TIFF",  # Should be normalized
            },
        }

        is_valid, errors = validate_recipe(recipe)

        assert is_valid is True
        assert recipe["output"]["format"] == "tiff"


class TestGetRecipeInfo:
    """Tests for get_recipe_info function."""

    def test_basic_info(self):
        """Test extraction of basic recipe info."""
        from transformation_portal.config_loader import get_recipe_info

        recipe = {
            "name": "Test Recipe",
            "description": "A test description",
            "stages": ["depth_estimation", "color_grading"],
        }

        info = get_recipe_info(recipe)

        assert info["name"] == "Test Recipe"
        assert info["description"] == "A test description"
        assert info["stages"] == ["depth_estimation", "color_grading"]

    def test_has_depth(self):
        """Test has_depth flag."""
        from transformation_portal.config_loader import get_recipe_info

        recipe_with = {"name": "Test", "stages": ["depth_estimation"]}
        recipe_without = {"name": "Test", "stages": ["color_grading"]}

        assert get_recipe_info(recipe_with)["has_depth"] is True
        assert get_recipe_info(recipe_without)["has_depth"] is False

    def test_has_ai(self):
        """Test has_ai flag."""
        from transformation_portal.config_loader import get_recipe_info

        recipe_with = {"name": "Test", "stages": ["ai_enhancement"]}
        recipe_without = {"name": "Test", "stages": ["color_grading"]}

        assert get_recipe_info(recipe_with)["has_ai"] is True
        assert get_recipe_info(recipe_without)["has_ai"] is False

    def test_has_material_response(self):
        """Test has_material_response flag."""
        from transformation_portal.config_loader import get_recipe_info

        recipe_with = {"name": "Test", "stages": ["material_response"]}
        recipe_without = {"name": "Test", "stages": ["color_grading"]}

        assert get_recipe_info(recipe_with)["has_material_response"] is True
        assert get_recipe_info(recipe_without)["has_material_response"] is False

    def test_has_color_grading(self):
        """Test has_color_grading flag."""
        from transformation_portal.config_loader import get_recipe_info

        recipe_with = {"name": "Test", "stages": ["color_grading"]}
        recipe_without = {"name": "Test", "stages": ["depth_estimation"]}

        assert get_recipe_info(recipe_with)["has_color_grading"] is True
        assert get_recipe_info(recipe_without)["has_color_grading"] is False

    def test_has_4k_upscaling(self):
        """Test has_4k_upscaling flag."""
        from transformation_portal.config_loader import get_recipe_info

        recipe_with = {"name": "Test", "stages": ["upscaling_4k"]}
        recipe_without = {"name": "Test", "stages": ["depth_estimation"]}

        assert get_recipe_info(recipe_with)["has_4k_upscaling"] is True
        assert get_recipe_info(recipe_without)["has_4k_upscaling"] is False

    def test_has_quality_feedback_from_config(self):
        """Test has_quality_feedback from config flag."""
        from transformation_portal.config_loader import get_recipe_info

        recipe = {
            "name": "Test",
            "stages": [],
            "quality_feedback": {"enabled": True},
        }

        assert get_recipe_info(recipe)["has_quality_feedback"] is True

    def test_has_quality_feedback_from_stage(self):
        """Test has_quality_feedback from stage."""
        from transformation_portal.config_loader import get_recipe_info

        recipe = {
            "name": "Test",
            "stages": ["quality_assessment"],
        }

        assert get_recipe_info(recipe)["has_quality_feedback"] is True

    def test_has_rag_indexing(self):
        """Test has_rag_indexing flag."""
        from transformation_portal.config_loader import get_recipe_info

        recipe = {
            "name": "Test",
            "stages": [],
            "quality_feedback": {"rag_indexing_enabled": True},
        }

        assert get_recipe_info(recipe)["has_rag_indexing"] is True

    def test_output_format(self):
        """Test output_format extraction."""
        from transformation_portal.config_loader import get_recipe_info

        recipe_with = {
            "name": "Test",
            "stages": [],
            "output": {"format": "png"},
        }
        recipe_without = {"name": "Test", "stages": []}

        assert get_recipe_info(recipe_with)["output_format"] == "png"
        assert get_recipe_info(recipe_without)["output_format"] == "tiff"  # Default

    def test_default_values(self):
        """Test default values for missing fields."""
        from transformation_portal.config_loader import get_recipe_info

        recipe = {}

        info = get_recipe_info(recipe)

        assert info["name"] == "Unnamed"
        assert info["description"] == ""
        assert info["stages"] == []


class TestListRecipes:
    """Tests for list_recipes function."""

    def test_list_empty_directory(self, temp_workspace):
        """Test listing empty recipes directory."""
        from transformation_portal.config_loader import list_recipes

        recipes = list_recipes(temp_workspace["root"])

        assert recipes == []

    def test_list_nonexistent_directory(self):
        """Test listing nonexistent directory returns empty list."""
        from transformation_portal.config_loader import list_recipes

        recipes = list_recipes(Path("/nonexistent/directory"))

        assert recipes == []

    def test_list_valid_recipes(self, temp_workspace):
        """Test listing valid recipe files."""
        from transformation_portal.config_loader import list_recipes

        # Create recipe files
        (temp_workspace["root"] / "recipe1.yaml").write_text(
            """
name: Recipe One
description: First recipe
stages:
  - depth_estimation
"""
        )
        (temp_workspace["root"] / "recipe2.yaml").write_text(
            """
name: Recipe Two
description: Second recipe
stages:
  - color_grading
"""
        )

        recipes = list_recipes(temp_workspace["root"])

        assert len(recipes) == 2
        recipe_names = [r["name"] for r in recipes]
        assert "Recipe One" in recipe_names
        assert "Recipe Two" in recipe_names

    def test_list_recipes_includes_path(self, temp_workspace):
        """Test that listed recipes include file path."""
        from transformation_portal.config_loader import list_recipes

        recipe_path = temp_workspace["root"] / "test.yaml"
        recipe_path.write_text(
            """
name: Test
stages:
  - depth_estimation
"""
        )

        recipes = list_recipes(temp_workspace["root"])

        assert len(recipes) == 1
        assert "path" in recipes[0]
        assert str(recipe_path) == recipes[0]["path"]

    def test_list_recipes_handles_invalid(self, temp_workspace):
        """Test that invalid recipes are handled gracefully."""
        from transformation_portal.config_loader import list_recipes

        # Create valid recipe
        (temp_workspace["root"] / "valid.yaml").write_text(
            """
name: Valid
stages:
  - depth_estimation
"""
        )
        # Create invalid recipe (empty)
        (temp_workspace["root"] / "invalid.yaml").write_text("")

        recipes = list_recipes(temp_workspace["root"])

        assert len(recipes) == 2

        # Find the invalid one
        invalid = next((r for r in recipes if "invalid" in r["path"]), None)
        assert invalid is not None
        assert "error" in invalid

    def test_list_recipes_sorted(self, temp_workspace):
        """Test that recipes are returned in sorted order."""
        from transformation_portal.config_loader import list_recipes

        # Create files in non-alphabetical order
        (temp_workspace["root"] / "z_recipe.yaml").write_text("name: Z\nstages: [depth_estimation]")
        (temp_workspace["root"] / "a_recipe.yaml").write_text("name: A\nstages: [depth_estimation]")
        (temp_workspace["root"] / "m_recipe.yaml").write_text("name: M\nstages: [depth_estimation]")

        recipes = list_recipes(temp_workspace["root"])

        paths = [Path(r["path"]).name for r in recipes]
        assert paths == sorted(paths)


class TestModuleExports:
    """Tests for module exports."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from transformation_portal import config_loader

        assert hasattr(config_loader, "__all__")
        assert "load_recipe" in config_loader.__all__
        assert "validate_recipe" in config_loader.__all__
        assert "get_recipe_info" in config_loader.__all__
        assert "list_recipes" in config_loader.__all__
