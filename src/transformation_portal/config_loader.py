#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration Loader for Transformation Portal.

Provides YAML recipe loading with validation, environment variable expansion,
and relative path resolution for the unified pipeline architecture.

Example:
    from transformation_portal.config_loader import load_recipe, validate_recipe

    # Load a recipe file
    recipe = load_recipe("config/recipes/signature_estate.yaml")

    # Validate a recipe dict
    is_valid, errors = validate_recipe({"name": "Test", "stages": ["color_grading"]})
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import yaml


# Environment variable pattern: ${VAR_NAME} or $VAR_NAME
_ENV_VAR_PATTERN = re.compile(r'\$\{([^}]+)\}|\$([A-Za-z_][A-Za-z0-9_]*)')


def _expand_env_vars(value: str) -> str:
    """Expand environment variables in a string.

    Supports both ${VAR_NAME} and $VAR_NAME syntax.

    Args:
        value: String potentially containing environment variable references.

    Returns:
        String with environment variables expanded.

    Example:
        >>> os.environ['HOME'] = '/home/user'
        >>> _expand_env_vars('${HOME}/config')
        '/home/user/config'
    """
    def replace_var(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))

    return _ENV_VAR_PATTERN.sub(replace_var, value)


def _expand_env_vars_recursive(obj: Any) -> Any:
    """Recursively expand environment variables in a data structure.

    Args:
        obj: Any Python object (dict, list, str, etc.).

    Returns:
        Object with all string values having environment variables expanded.
    """
    if isinstance(obj, str):
        return _expand_env_vars(obj)
    elif isinstance(obj, dict):
        return {k: _expand_env_vars_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_expand_env_vars_recursive(item) for item in obj]
    return obj


def _resolve_relative_paths(obj: Any, base_dir: Path) -> Any:
    """Resolve relative paths in a data structure relative to a base directory.

    Only resolves paths for keys containing 'path', 'lut', 'file', or 'dir'.

    Args:
        obj: Any Python object.
        base_dir: Base directory for relative path resolution.

    Returns:
        Object with relative paths resolved.
    """
    path_keys = {'path', 'lut', 'file', 'dir', 'logo', 'texture'}

    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if isinstance(v, str) and any(pk in k.lower() for pk in path_keys):
                # Check if it's a relative path (cross-platform check)
                if v and not os.path.isabs(v) and not v.startswith('$'):
                    resolved = base_dir / v
                    # Only use resolved path if it exists or looks like a path
                    if resolved.exists() or os.sep in v or '/' in v:
                        result[k] = str(resolved)
                    else:
                        result[k] = v
                else:
                    result[k] = v
            else:
                result[k] = _resolve_relative_paths(v, base_dir)
        return result
    elif isinstance(obj, list):
        return [_resolve_relative_paths(item, base_dir) for item in obj]
    return obj


def load_recipe(
    path: Union[str, Path],
    expand_env: bool = True,
    resolve_paths: bool = True
) -> Dict[str, Any]:
    """Load and parse a YAML recipe file.

    Args:
        path: Path to the YAML recipe file.
        expand_env: Whether to expand environment variables.
        resolve_paths: Whether to resolve relative paths.

    Returns:
        Parsed recipe dictionary.

    Raises:
        FileNotFoundError: If the recipe file doesn't exist.
        yaml.YAMLError: If the YAML is malformed.
        ValueError: If the recipe is empty or invalid.

    Example:
        >>> recipe = load_recipe("config/recipes/signature_estate.yaml")
        >>> print(recipe['name'])
        'Signature Estate'
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Recipe file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        recipe = yaml.safe_load(f)

    if recipe is None:
        raise ValueError(f"Recipe file is empty: {path}")

    if not isinstance(recipe, dict):
        raise ValueError(f"Recipe must be a dictionary, got {type(recipe).__name__}")

    # Expand environment variables
    if expand_env:
        recipe = _expand_env_vars_recursive(recipe)

    # Resolve relative paths (relative to recipe file location)
    if resolve_paths:
        base_dir = path.parent.resolve()
        recipe = _resolve_relative_paths(recipe, base_dir)

    # Store the recipe file path for reference
    recipe['_recipe_path'] = str(path.resolve())
    recipe['_recipe_dir'] = str(path.parent.resolve())

    return recipe


def validate_recipe(
    recipe_dict: Dict[str, Any],
    strict: bool = False
) -> Tuple[bool, List[str]]:
    """Validate a recipe dictionary against the schema.

    Args:
        recipe_dict: Recipe dictionary to validate.
        strict: If True, require all optional fields.

    Returns:
        Tuple of (is_valid, list of error messages).

    Example:
        >>> is_valid, errors = validate_recipe({"name": "Test", "stages": ["depth"]})
        >>> print(is_valid)
        True
    """
    errors: List[str] = []

    # Required fields
    if 'name' not in recipe_dict:
        errors.append("Missing required field: 'name'")

    if 'stages' not in recipe_dict:
        errors.append("Missing required field: 'stages'")
    elif not isinstance(recipe_dict.get('stages'), list):
        errors.append("Field 'stages' must be a list")
    elif len(recipe_dict.get('stages', [])) == 0:
        errors.append("Field 'stages' must not be empty")

    # Validate stages
    valid_stages = {
        'depth_estimation', 'ai_enhancement', 'material_response',
        'color_grading', 'photo_finishing', 'branding', 'output',
        'upscaling_4k', 'quality_assessment'
    }
    stages = recipe_dict.get('stages', [])
    for stage in stages:
        if stage not in valid_stages:
            errors.append(f"Invalid stage: '{stage}'. Valid stages: {sorted(valid_stages)}")

    # Validate stage configurations
    stage_configs = {
        'depth_estimation': ['enabled', 'model', 'device'],
        'ai_enhancement': ['enabled', 'model', 'strength', 'steps'],
        'material_response': ['enabled', 'profile', 'texture_boost', 'ambient_occlusion'],
        'color_grading': ['enabled', 'lut', 'lut_strength', 'contrast', 'saturation', 'warmth'],
        'photo_finishing': ['enabled', 'aces', 'bloom', 'vignette', 'grain'],
        'branding': ['enabled', 'logo', 'text', 'watermark'],
        'output': ['format', 'quality', 'bit_depth'],
        'upscaling_4k': ['enabled', 'target_width', 'target_height', 'method', 'preserve_sharpness'],
        'quality_feedback': ['enabled', 'hybrid_mode', 'use_lpips', 'lpips_network', 'rag_indexing_enabled'],
    }

    for stage_name, valid_keys in stage_configs.items():
        if stage_name in recipe_dict and isinstance(recipe_dict[stage_name], dict):
            stage_config = recipe_dict[stage_name]

            # Check enabled field type
            if 'enabled' in stage_config:
                if not isinstance(stage_config['enabled'], bool):
                    errors.append(f"Stage '{stage_name}': 'enabled' must be a boolean")

            # Validate numeric ranges
            if stage_name == 'material_response':
                for field in ['texture_boost', 'ambient_occlusion']:
                    if field in stage_config:
                        val = stage_config[field]
                        if not isinstance(val, (int, float)):
                            errors.append(f"Stage '{stage_name}': '{field}' must be a number")
                        elif not 0.0 <= val <= 1.0:
                            errors.append(
                                f"Stage '{stage_name}': '{field}' must be between 0.0 and 1.0"
                            )

            if stage_name == 'color_grading':
                if 'lut_strength' in stage_config:
                    val = stage_config['lut_strength']
                    if not isinstance(val, (int, float)):
                        errors.append(f"Stage '{stage_name}': 'lut_strength' must be a number")
                    elif not 0.0 <= val <= 1.0:
                        errors.append(
                            f"Stage '{stage_name}': 'lut_strength' must be between 0.0 and 1.0"
                        )

                if 'contrast' in stage_config:
                    val = stage_config['contrast']
                    if not isinstance(val, (int, float)):
                        errors.append(f"Stage '{stage_name}': 'contrast' must be a number")
                    elif not 0.5 <= val <= 2.0:
                        errors.append(
                            f"Stage '{stage_name}': 'contrast' must be between 0.5 and 2.0"
                        )

    # Validate output configuration
    if 'output' in recipe_dict and isinstance(recipe_dict['output'], dict):
        output_config = recipe_dict['output']

        if 'format' in output_config:
            valid_formats = {'jpeg', 'png', 'tiff', 'exr'}
            fmt = output_config['format'].lower()
            if fmt not in valid_formats:
                errors.append(f"Invalid output format: '{fmt}'. Valid formats: {sorted(valid_formats)}")

        if 'quality' in output_config:
            quality = output_config['quality']
            if not isinstance(quality, int):
                errors.append("Output 'quality' must be an integer")
            elif not 1 <= quality <= 100:
                errors.append("Output 'quality' must be between 1 and 100")

    return (len(errors) == 0, errors)


def get_recipe_info(recipe_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Extract summary information from a recipe.

    Args:
        recipe_dict: Parsed recipe dictionary.

    Returns:
        Dictionary with recipe summary information.
    """
    stages = recipe_dict.get('stages', [])
    quality_feedback = recipe_dict.get('quality_feedback', {})

    return {
        'name': recipe_dict.get('name', 'Unnamed'),
        'description': recipe_dict.get('description', ''),
        'stages': stages,
        'has_depth': 'depth_estimation' in stages,
        'has_ai': 'ai_enhancement' in stages,
        'has_material_response': 'material_response' in stages,
        'has_color_grading': 'color_grading' in stages,
        'has_4k_upscaling': 'upscaling_4k' in stages,
        'has_quality_feedback': quality_feedback.get('enabled', False) or 'quality_assessment' in stages,
        'has_rag_indexing': quality_feedback.get('rag_indexing_enabled', False),
        'output_format': recipe_dict.get('output', {}).get('format', 'tiff'),
    }


def list_recipes(recipes_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """List all available recipes in a directory.

    Args:
        recipes_dir: Path to the recipes directory.

    Returns:
        List of recipe info dictionaries.
    """
    recipes_dir = Path(recipes_dir)
    recipes = []

    if not recipes_dir.exists():
        return recipes

    for recipe_file in sorted(recipes_dir.glob("*.yaml")):
        try:
            recipe = load_recipe(recipe_file, expand_env=False, resolve_paths=False)
            info = get_recipe_info(recipe)
            info['path'] = str(recipe_file)
            recipes.append(info)
        except Exception as e:
            # Skip invalid recipes but log the error
            recipes.append({
                'path': str(recipe_file),
                'name': recipe_file.stem,
                'error': str(e),
            })

    return recipes


__all__ = [
    'load_recipe',
    'validate_recipe',
    'get_recipe_info',
    'list_recipes',
]
