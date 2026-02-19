"""
Platform Core Module

Unified infrastructure for all Transformation Portal pipelines.
Eliminates duplication across lux_depth_v2, luxury_video_master_grader,
and other pipelines while providing a clean, maintainable foundation.

This module provides:
- Config schemas and preset management (config/)
- Device detection and optimization (device/)
- Artifact and cache management (artifacts/)
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
from .security import InputValidator, PathValidator, SanitizationPolicy, safe_resolve_path, validate_input_file

try:
    from .artifacts import ArtifactStorage, CacheManager, ContentAddressedCache
except (ImportError, ModuleNotFoundError):
    ArtifactStorage = None  # type: ignore[assignment, misc]
    CacheManager = None  # type: ignore[assignment, misc]
    ContentAddressedCache = None  # type: ignore[assignment, misc]

try:
    from .device import DeviceCapabilities, DeviceDetector, DeviceType, MemoryManager, PerformanceProfiler
except (ImportError, ModuleNotFoundError):
    DeviceCapabilities = None  # type: ignore[assignment, misc]
    DeviceDetector = None  # type: ignore[assignment, misc]
    DeviceType = None  # type: ignore[assignment, misc]
    MemoryManager = None  # type: ignore[assignment, misc]
    PerformanceProfiler = None  # type: ignore[assignment, misc]

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
    # Artifacts
    "CacheManager",
    "ArtifactStorage",
    "ContentAddressedCache",
    # Security
    "InputValidator",
    "PathValidator",
    "SanitizationPolicy",
    "validate_input_file",
    "safe_resolve_path",
]

__version__ = "1.0.0"
