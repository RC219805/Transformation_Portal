"""
Configuration Validation Logic.

Extends Pydantic schema validation with runtime checks (filesystem, permissions).
"""

import os
from pathlib import Path
from typing import Dict, Any, Union

from .schemas import ConfigSchema

class ConfigValidationError(ValueError):
    """Raised when configuration fails validation rules."""
    pass


def validate_config(config_data: Union[Dict[str, Any], ConfigSchema], create_dirs: bool = True) -> ConfigSchema:
    """
    Validate and finalize a configuration object.
    
    Args:
        config_data: Raw dictionary or partial ConfigSchema
        create_dirs: If True, attempts to create missing output directories
        
    Returns:
        Validated ConfigSchema object
    
    Raises:
        ConfigValidationError: If rules are violated
    """
    # 1. Type Validation (Pydantic)
    if isinstance(config_data, dict):
        try:
            config = ConfigSchema(**config_data)
        except Exception as e:
            raise ConfigValidationError(f"Schema violation: {str(e)}") from e
    else:
        config = config_data

    # 2. Runtime Environment Checks
    
    # Input Directory must exist
    if not config.paths.input_dir.exists():
        raise ConfigValidationError(f"Input directory does not exist: {config.paths.input_dir}")
    
    # Output Directory logic
    if not config.paths.output_dir.exists():
        if create_dirs:
            try:
                config.paths.output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                raise ConfigValidationError(f"Could not create output directory: {e}")
        else:
             raise ConfigValidationError(f"Output directory does not exist: {config.paths.output_dir}")

    # Write permission check
    if not os.access(config.paths.output_dir, os.W_OK):
        raise ConfigValidationError(f"Output directory is not writable: {config.paths.output_dir}")

    # 3. Business Logic / Cross-Field Validation
    
    # Tiled processing requires overlapping
    if config.performance.tile_size > 0 and config.performance.tile_size < 256:
         raise ConfigValidationError("Tile size must be at least 256 pixels.")
         
    # Precision conflicts
    if config.device.device == "cpu" and config.device.precision == "fp16":
        # Warn or coerce? For now, we raise, but in prod we might log warning and coerce to fp32
        # raise ConfigValidationError("FP16 is not typically supported on CPU.")
        pass # PyTorch CPU supports Half in newer versions, passing for flexibility

    return config
