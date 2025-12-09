"""
Configuration validation utilities.

Validates configuration schemas and ensures consistency across pipelines.
"""

from typing import Any, Dict, List
from pathlib import Path


class ConfigValidationError(Exception):
    """Configuration validation error."""
    
    def __init__(self, message: str, errors: List[str] = None):
        super().__init__(message)
        self.errors = errors or []


def validate_config(config: Dict[str, Any]) -> List[str]:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Validate device config
    if "device" in config:
        device = config["device"]
        if isinstance(device, dict):
            # Check memory fraction
            if "memory_fraction" in device:
                mem_frac = device["memory_fraction"]
                if not (0.1 <= mem_frac <= 0.95):
                    errors.append(f"device.memory_fraction must be between 0.1 and 0.95, got {mem_frac}")
    
    # Validate paths config
    if "paths" in config:
        paths = config["paths"]
        if isinstance(paths, dict):
            # Check that output_dir is set if write_outputs is enabled
            output = config.get("output", {})
            if isinstance(output, dict) and output.get("write_outputs", True):
                if not paths.get("output_dir"):
                    errors.append("paths.output_dir must be set when output.write_outputs is True")
    
    # Validate performance config
    if "performance" in config:
        perf = config["performance"]
        if isinstance(perf, dict):
            # Check tile overlap vs tile size
            tile_size = perf.get("tile_size", 512)
            tile_overlap = perf.get("tile_overlap", 64)
            if tile_overlap >= tile_size:
                errors.append(f"performance.tile_overlap ({tile_overlap}) must be less than tile_size ({tile_size})")
            
            # Check batch size is reasonable
            batch_size = perf.get("batch_size", 1)
            if batch_size < 1:
                errors.append(f"performance.batch_size must be >= 1, got {batch_size}")
            if batch_size > 64:
                errors.append(f"performance.batch_size seems unreasonably large: {batch_size}")
    
    # Validate output config
    if "output" in config:
        output = config["output"]
        if isinstance(output, dict):
            # Check preview scale
            preview_scale = output.get("preview_scale", 0.25)
            if not (0.01 <= preview_scale <= 1.0):
                errors.append(f"output.preview_scale must be between 0.01 and 1.0, got {preview_scale}")
            
            # Check conflicting flags
            if output.get("skip_existing") and output.get("overwrite"):
                errors.append("output.skip_existing and output.overwrite cannot both be True")
    
    # Validate validation config
    if "validation" in config:
        val = config["validation"]
        if isinstance(val, dict):
            # Check max input size
            max_size = val.get("max_input_size_mb", 500.0)
            if max_size < 1.0:
                errors.append(f"validation.max_input_size_mb must be >= 1.0, got {max_size}")
    
    return errors


def validate_config_strict(config: Dict[str, Any]) -> None:
    """
    Validate configuration and raise exception if invalid.
    
    Args:
        config: Configuration dictionary to validate
        
    Raises:
        ConfigValidationError: If configuration is invalid
    """
    errors = validate_config(config)
    if errors:
        raise ConfigValidationError(
            f"Configuration validation failed with {len(errors)} error(s)",
            errors=errors
        )


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Override values take precedence over base values.
    Nested dictionaries are merged recursively.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursive merge for nested dicts
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result
