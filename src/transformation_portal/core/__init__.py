"""
Platform Core Module

Unified infrastructure for all Transformation Portal pipelines.
Eliminates duplication across lux_depth_v2, luxury_video_master_grader,
and other pipelines while providing a clean, maintainable foundation.

This module provides:
- Config schemas and preset management (config/)
- Device detection and optimization (device/)
- Artifact and cache management (storage/)
- Security validation and sanitization (security/)
- Observability integration (observability/)

Architecture Goals:
- Zero breaking changes during migration
- Performance neutral or improved
- Clean, intuitive APIs
- Comprehensive test coverage
- Foundation for future stage graph

Version: 1.0.0 (Platform Core Extraction - PR-2)
"""

from .config import ConfigSchema, DeviceConfig, PathsConfig, PerformanceConfig, PresetRegistry, load_preset, validate_config
from .device import DeviceCapabilities, DeviceDetector, DeviceType, MemoryManager, PerformanceProfiler
from .security import InputValidator, PathValidator, SanitizationPolicy, safe_resolve_path, validate_input_file

__all__ = [
    # Config
    "ConfigSchema",
    "DeviceConfig",
    "PathsConfig",
    "PerformanceConfig",
    "PresetRegistry",
    "load_preset",
    "validate_config",
    # Device
    "DeviceDetector",
    "DeviceCapabilities",
    "DeviceType",
    "PerformanceProfiler",
    "MemoryManager",
    # Security
    "InputValidator",
    "PathValidator",
    "SanitizationPolicy",
    "validate_input_file",
    "safe_resolve_path",
]

__version__ = "1.0.0"
