"""Tests for RecipeValidator module.

Provides comprehensive coverage of recipe validation functionality
including schema validation, error handling, and file operations.

Coverage Target: 80% of recipe_validator.py (63 statements)
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
]


class TestGetRecipeSchema:
    """Tests for get_recipe_schema function."""

    def test_schema_is_valid_json_schema(self):
        """Test that returned schema is valid JSON schema."""
        from transformation_portal.utils.recipe_validator import get_recipe_schema

        schema = get_recipe_schema()

        assert "$schema" in schema
        assert "http://json-schema.org/draft-07/schema#" in schema["$schema"]
        assert schema["type"] == "object"

    def test_schema_has_required_fields(self):
        """Test that schema defines required fields."""
        from transformation_portal.utils.recipe_validator import get_recipe_schema

        schema = get_recipe_schema()

        assert "required" in schema
        assert "name" in schema["required"]
        assert "stages" in schema["required"]

    def test_schema_has_stages_enum(self):
        """Test that schema defines valid stages."""
        from transformation_portal.utils.recipe_validator import get_recipe_schema

        schema = get_recipe_schema()

        stages_schema = schema["properties"]["stages"]
        assert stages_schema["type"] == "array"
        assert "items" in stages_schema

        valid_stages = stages_schema["items"]["enum"]
        assert "depth_estimation" in valid_stages
        assert "color_grading" in valid_stages
        assert "material_response" in valid_stages
        assert "photo_finishing" in valid_stages

    def test_schema_has_stage_configs(self):
        """Test that schema defines stage-specific configurations."""
        from transformation_portal.utils.recipe_validator import get_recipe_schema

        schema = get_recipe_schema()
        props = schema["properties"]

        # Check that major stage configs are defined
        assert "depth_estimation" in props
        assert "ai_enhancement" in props
        assert "material_response" in props
        assert "color_grading" in props
        assert "photo_finishing" in props
        assert "branding" in props
        assert "output" in props

    def test_depth_estimation_config(self):
        """Test depth estimation schema properties."""
        from transformation_portal.utils.recipe_validator import get_recipe_schema

        schema = get_recipe_schema()
        depth_props = schema["properties"]["depth_estimation"]["properties"]

        assert "enabled" in depth_props
        assert "model" in depth_props
        assert "device" in depth_props
        assert depth_props["device"]["enum"] == ["auto", "cpu", "cuda", "mps"]

    def test_output_config(self):
        """Test output schema properties."""
        from transformation_portal.utils.recipe_validator import get_recipe_schema

        schema = get_recipe_schema()
        output_props = schema["properties"]["output"]["properties"]

        assert "format" in output_props
        assert output_props["format"]["enum"] == ["jpeg", "png", "tiff", "exr"]
        assert output_props["bit_depth"]["enum"] == [8, 16, 32]


class TestRecipeValidatorInit:
    """Tests for RecipeValidator initialization."""

    def test_default_init(self):
        """Test default initialization with embedded schema."""
        from transformation_portal.utils.recipe_validator import RecipeValidator

        validator = RecipeValidator()

        assert validator.schema is not None
        assert "name" in validator.schema["required"]

    def test_custom_schema_path(self, temp_workspace):
        """Test initialization with custom schema file."""
        import json

        from transformation_portal.utils.recipe_validator import RecipeValidator

        # Create custom schema file
        custom_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "required": ["custom_field"],
            "properties": {"custom_field": {"type": "string"}},
        }
        schema_path = temp_workspace["root"] / "custom_schema.json"
        with open(schema_path, "w") as f:
            json.dump(custom_schema, f)

        validator = RecipeValidator(schema_path=schema_path)

        assert "custom_field" in validator.schema["required"]

    def test_nonexistent_schema_path_uses_default(self, temp_workspace):
        """Test that nonexistent schema path falls back to default."""
        from transformation_portal.utils.recipe_validator import RecipeValidator

        nonexistent = temp_workspace["root"] / "nonexistent.json"
        validator = RecipeValidator(schema_path=nonexistent)

        # Should fall back to embedded schema
        assert "name" in validator.schema["required"]


class TestRecipeValidatorValidate:
    """Tests for RecipeValidator.validate method."""

    @pytest.fixture
    def validator(self):
        """Create RecipeValidator instance."""
        from transformation_portal.utils.recipe_validator import RecipeValidator

        return RecipeValidator()

    def test_valid_minimal_recipe(self, validator):
        """Test validation of minimal valid recipe."""
        recipe = {
            "name": "Test Recipe",
            "stages": ["depth_estimation"],
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is True
        assert errors == []

    def test_valid_full_recipe(self, validator):
        """Test validation of full recipe with all configurations."""
        recipe = {
            "name": "Complete Recipe",
            "description": "A complete test recipe",
            "stages": ["depth_estimation", "color_grading", "photo_finishing"],
            "depth_estimation": {
                "enabled": True,
                "model": "depth-anything-v2-small",
                "device": "auto",
            },
            "color_grading": {
                "enabled": True,
                "lut_strength": 0.7,
                "contrast": 1.0,
            },
            "output": {
                "format": "tiff",
                "quality": 95,
                "bit_depth": 16,
            },
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is True
        assert errors == []

    def test_missing_name_field(self, validator):
        """Test that missing name field is detected."""
        recipe = {
            "stages": ["depth_estimation"],
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is False
        assert any("name" in error for error in errors)

    def test_missing_stages_field(self, validator):
        """Test that missing stages field is detected."""
        recipe = {
            "name": "Test Recipe",
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is False
        assert any("stages" in error for error in errors)

    def test_empty_stages_array(self, validator):
        """Test that empty stages array is detected."""
        recipe = {
            "name": "Test Recipe",
            "stages": [],
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is False
        assert any("stages" in error.lower() for error in errors)

    def test_invalid_stage_name(self, validator):
        """Test that invalid stage names are detected."""
        recipe = {
            "name": "Test Recipe",
            "stages": ["invalid_stage_name"],
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is False
        assert any("invalid_stage_name" in error for error in errors)

    def test_stages_not_array(self, validator):
        """Test that non-array stages field is detected."""
        recipe = {
            "name": "Test Recipe",
            "stages": "depth_estimation",  # Should be array
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is False
        assert any("array" in error.lower() or "stages" in error.lower() for error in errors)


class TestRecipeValidatorFallback:
    """Tests for fallback validation without jsonschema."""

    def test_fallback_validation_name_not_string(self):
        """Test fallback validation catches non-string name."""
        from transformation_portal.utils.recipe_validator import RecipeValidator

        validator = RecipeValidator()
        # Force fallback by temporarily disabling validator
        validator._validator = None

        recipe = {
            "name": 123,  # Should be string
            "stages": ["depth_estimation"],
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is False
        assert any("name" in error.lower() and "string" in error.lower() for error in errors)

    def test_fallback_validation_missing_required(self):
        """Test fallback validation catches missing required fields."""
        from transformation_portal.utils.recipe_validator import RecipeValidator

        validator = RecipeValidator()
        validator._validator = None

        recipe = {}

        is_valid, errors = validator.validate(recipe)

        assert is_valid is False
        assert any("name" in error.lower() for error in errors)
        assert any("stages" in error.lower() for error in errors)

    def test_fallback_validation_invalid_stage(self):
        """Test fallback validation catches invalid stage."""
        from transformation_portal.utils.recipe_validator import RecipeValidator

        validator = RecipeValidator()
        validator._validator = None

        recipe = {
            "name": "Test",
            "stages": ["unknown_stage"],
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is False
        assert any("unknown_stage" in error for error in errors)

    def test_fallback_validation_valid_recipe(self):
        """Test fallback validation passes valid recipe."""
        from transformation_portal.utils.recipe_validator import RecipeValidator

        validator = RecipeValidator()
        validator._validator = None

        recipe = {
            "name": "Test",
            "stages": ["depth_estimation", "color_grading"],
        }

        is_valid, errors = validator.validate(recipe)

        assert is_valid is True
        assert errors == []


class TestRecipeValidatorValidateFile:
    """Tests for RecipeValidator.validate_file method."""

    @pytest.fixture
    def validator(self):
        """Create RecipeValidator instance."""
        from transformation_portal.utils.recipe_validator import RecipeValidator

        return RecipeValidator()

    def test_validate_valid_file(self, validator, temp_workspace):
        """Test validation of valid YAML file."""
        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test Recipe
stages:
  - depth_estimation
  - color_grading
"""
        )

        is_valid, errors = validator.validate_file(recipe_path)

        assert is_valid is True
        assert errors == []

    def test_validate_nonexistent_file(self, validator):
        """Test validation of nonexistent file."""
        is_valid, errors = validator.validate_file(Path("/nonexistent/recipe.yaml"))

        assert is_valid is False
        assert any("not found" in error.lower() for error in errors)

    def test_validate_invalid_yaml(self, validator, temp_workspace):
        """Test validation of malformed YAML."""
        recipe_path = temp_workspace["root"] / "invalid.yaml"
        recipe_path.write_text("invalid: yaml: content:\n  - broken")

        is_valid, errors = validator.validate_file(recipe_path)

        assert is_valid is False
        assert any("yaml" in error.lower() or "invalid" in error.lower() for error in errors)

    def test_validate_empty_file(self, validator, temp_workspace):
        """Test validation of empty file."""
        recipe_path = temp_workspace["root"] / "empty.yaml"
        recipe_path.write_text("")

        is_valid, errors = validator.validate_file(recipe_path)

        assert is_valid is False
        assert any("empty" in error.lower() for error in errors)


class TestValidateRecipeFileFunction:
    """Tests for validate_recipe_file convenience function."""

    def test_validate_recipe_file_valid(self, temp_workspace):
        """Test convenience function with valid file."""
        from transformation_portal.utils.recipe_validator import validate_recipe_file

        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
name: Test
stages:
  - depth_estimation
"""
        )

        is_valid, errors = validate_recipe_file(recipe_path)

        assert is_valid is True
        assert errors == []

    def test_validate_recipe_file_invalid(self, temp_workspace):
        """Test convenience function with invalid file."""
        from transformation_portal.utils.recipe_validator import validate_recipe_file

        recipe_path = temp_workspace["root"] / "recipe.yaml"
        recipe_path.write_text(
            """
stages:
  - depth_estimation
"""
        )

        is_valid, errors = validate_recipe_file(recipe_path)

        assert is_valid is False


class TestModuleExports:
    """Tests for module exports and __all__."""

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        from transformation_portal.utils import recipe_validator

        assert hasattr(recipe_validator, "__all__")
        assert "RecipeValidator" in recipe_validator.__all__
        assert "get_recipe_schema" in recipe_validator.__all__
        assert "validate_recipe_file" in recipe_validator.__all__

    def test_has_jsonschema_flag(self):
        """Test that HAS_JSONSCHEMA flag is defined."""
        from transformation_portal.utils.recipe_validator import HAS_JSONSCHEMA

        # Should be boolean (True if jsonschema installed)
        assert isinstance(HAS_JSONSCHEMA, bool)
