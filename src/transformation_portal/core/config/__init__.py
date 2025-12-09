"""
Core Configuration Module

Unified configuration schemas extracted from 5+ pipelines:
- lux_depth_v2/config.py
- luxury_video_master_grader.py
- depth_tools.py
- And others

Provides Pydantic schemas for type-safe, validated configuration
with preset support and automatic validation.
"""

from .schemas import (
    ConfigSchema,
    DeviceConfig,
    PathsConfig,
    PerformanceConfig,
    OutputConfig,
    ValidationConfig,
)
from .presets import (
    PresetRegistry,
    Preset,
    load_preset,
    register_preset,
    list_presets,
)
from .validation import (
    validate_config,
    ConfigValidationError,
)

__all__ = [
    # Schemas
    "ConfigSchema",
    "DeviceConfig",
    "PathsConfig",
    "PerformanceConfig",
    "OutputConfig",
    "ValidationConfig",
    # Presets
    "PresetRegistry",
    "Preset",
    "load_preset",
    "register_preset",
    "list_presets",
    # Validation
    "validate_config",
    "ConfigValidationError",
]
