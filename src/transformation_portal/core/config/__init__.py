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

from .preset_health import PresetHealthReport, validate_preset
from .presets import Preset, PresetRegistry, list_presets, load_preset, register_preset
from .schemas import ConfigSchema, DeviceConfig, OutputConfig, PathsConfig, PerformanceConfig, ValidationConfig
from .validation import ConfigValidationError, validate_config

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
    # Preset Health
    "PresetHealthReport",
    "validate_preset",
]
