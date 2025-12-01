#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recipe Validator for Transformation Portal.

Provides JSON Schema validation for YAML recipe files using jsonschema.

Example:
    from transformation_portal.utils.recipe_validator import RecipeValidator

    validator = RecipeValidator()
    is_valid, errors = validator.validate(recipe_dict)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from jsonschema import Draft7Validator
    HAS_JSONSCHEMA = True
except ImportError:
    HAS_JSONSCHEMA = False
    Draft7Validator = None  # type: ignore


# Default schema path relative to repo root
DEFAULT_SCHEMA_PATH = Path(__file__).parent.parent.parent.parent / "config" / "schemas" / "recipe_schema.json"


def get_recipe_schema() -> Dict[str, Any]:
    """Get the JSON schema for recipe validation.

    Returns:
        The recipe JSON schema as a dictionary.
    """
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Transformation Portal Recipe",
        "description": "Schema for unified pipeline recipe configuration",
        "type": "object",
        "required": ["name", "stages"],
        "properties": {
            "name": {
                "type": "string",
                "description": "Human-readable recipe name",
                "minLength": 1
            },
            "description": {
                "type": "string",
                "description": "Recipe description"
            },
            "stages": {
                "type": "array",
                "description": "Processing stages to execute",
                "items": {
                    "type": "string",
                    "enum": [
                        "depth_estimation",
                        "ai_enhancement",
                        "material_response",
                        "color_grading",
                        "photo_finishing",
                        "branding",
                        "upscaling_4k",
                        "quality_assessment"
                    ]
                },
                "minItems": 1,
                "uniqueItems": True
            },
            "depth_estimation": {
                "type": "object",
                "description": "Depth estimation stage configuration",
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "model": {
                        "type": "string",
                        "description": "Depth model name",
                        "default": "depth-anything-v2-small"
                    },
                    "device": {
                        "type": "string",
                        "enum": ["auto", "cpu", "cuda", "mps"],
                        "default": "auto"
                    }
                }
            },
            "ai_enhancement": {
                "type": "object",
                "description": "AI enhancement stage configuration",
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "model": {"type": "string"},
                    "strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.5
                    },
                    "steps": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 30
                    }
                }
            },
            "material_response": {
                "type": "object",
                "description": "Material Response stage configuration",
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "profile": {
                        "type": "string",
                        "description": "Material profile name",
                        "default": "luxury_interior"
                    },
                    "texture_boost": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.25
                    },
                    "ambient_occlusion": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.12
                    },
                    "highlight_warmth": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.08
                    },
                    "window_light_wrap": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.14
                    }
                }
            },
            "color_grading": {
                "type": "object",
                "description": "Color grading stage configuration",
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "lut": {
                        "type": "string",
                        "description": "Path to LUT file"
                    },
                    "lut_strength": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "default": 0.7
                    },
                    "contrast": {
                        "type": "number",
                        "minimum": 0.5,
                        "maximum": 2.0,
                        "default": 1.0
                    },
                    "saturation": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 2.0,
                        "default": 1.0
                    },
                    "warmth": {
                        "type": "number",
                        "minimum": -0.5,
                        "maximum": 0.5,
                        "default": 0.0
                    },
                    "exposure": {
                        "type": "number",
                        "minimum": -2.0,
                        "maximum": 2.0,
                        "default": 0.0
                    }
                }
            },
            "photo_finishing": {
                "type": "object",
                "description": "Photo finishing stage configuration",
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "aces": {
                        "type": "boolean",
                        "description": "Apply ACES tone mapping",
                        "default": True
                    },
                    "bloom": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": True},
                            "threshold": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 0.8
                            },
                            "intensity": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 0.25
                            }
                        }
                    },
                    "vignette": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": True},
                            "strength": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 1.0,
                                "default": 0.18
                            }
                        }
                    },
                    "grain": {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": True},
                            "amount": {
                                "type": "number",
                                "minimum": 0.0,
                                "maximum": 0.1,
                                "default": 0.012
                            }
                        }
                    }
                }
            },
            "branding": {
                "type": "object",
                "description": "Branding overlay configuration",
                "properties": {
                    "enabled": {"type": "boolean", "default": False},
                    "logo": {"type": "string", "description": "Path to logo file"},
                    "text": {"type": "string", "description": "Brand text overlay"},
                    "watermark": {"type": "boolean", "default": False}
                }
            },
            "output": {
                "type": "object",
                "description": "Output configuration",
                "properties": {
                    "format": {
                        "type": "string",
                        "enum": ["jpeg", "png", "tiff", "exr"],
                        "default": "tiff"
                    },
                    "quality": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 95
                    },
                    "bit_depth": {
                        "type": "integer",
                        "enum": [8, 16, 32],
                        "default": 16
                    }
                }
            }
        },
        "additionalProperties": True
    }


class RecipeValidator:
    """Validator for recipe configurations using JSON Schema."""

    def __init__(self, schema_path: Optional[Path] = None):
        """Initialize the validator.

        Args:
            schema_path: Optional path to a custom schema file.
        """
        if schema_path and schema_path.exists():
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.schema = json.load(f)
        else:
            self.schema = get_recipe_schema()

        if HAS_JSONSCHEMA:
            self._validator = Draft7Validator(self.schema)
        else:
            self._validator = None

    def validate(self, recipe_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate a recipe against the schema.

        Args:
            recipe_dict: Recipe dictionary to validate.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: List[str] = []

        if self._validator is None:
            # Fallback validation without jsonschema
            return self._validate_fallback(recipe_dict)

        for error in self._validator.iter_errors(recipe_dict):
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            errors.append(f"{path}: {error.message}")

        return (len(errors) == 0, errors)

    def _validate_fallback(self, recipe_dict: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Fallback validation without jsonschema library.

        Args:
            recipe_dict: Recipe dictionary to validate.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: List[str] = []

        # Required fields
        if 'name' not in recipe_dict:
            errors.append("root: 'name' is a required property")
        elif not isinstance(recipe_dict['name'], str):
            errors.append("name: must be a string")

        if 'stages' not in recipe_dict:
            errors.append("root: 'stages' is a required property")
        elif not isinstance(recipe_dict['stages'], list):
            errors.append("stages: must be an array")
        elif len(recipe_dict['stages']) == 0:
            errors.append("stages: must have at least 1 item")

        # Validate stage names
        valid_stages = {
            'depth_estimation', 'ai_enhancement', 'material_response',
            'color_grading', 'photo_finishing', 'branding', 'output'
        }
        for stage in recipe_dict.get('stages', []):
            if stage not in valid_stages:
                errors.append(f"stages: '{stage}' is not a valid stage")

        return (len(errors) == 0, errors)

    def validate_file(self, recipe_path: Path) -> Tuple[bool, List[str]]:
        """Validate a recipe file.

        Args:
            recipe_path: Path to the recipe YAML file.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        import yaml

        errors: List[str] = []

        if not recipe_path.exists():
            return (False, [f"File not found: {recipe_path}"])

        try:
            with open(recipe_path, 'r', encoding='utf-8') as f:
                recipe_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            return (False, [f"Invalid YAML: {e}"])

        if recipe_dict is None:
            return (False, ["Recipe file is empty"])

        return self.validate(recipe_dict)


def validate_recipe_file(recipe_path: Path) -> Tuple[bool, List[str]]:
    """Convenience function to validate a recipe file.

    Args:
        recipe_path: Path to the recipe YAML file.

    Returns:
        Tuple of (is_valid, list of error messages).
    """
    validator = RecipeValidator()
    return validator.validate_file(recipe_path)


__all__ = [
    'RecipeValidator',
    'get_recipe_schema',
    'validate_recipe_file',
]
