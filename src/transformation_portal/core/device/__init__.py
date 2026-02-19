"""
Core Device Management Module

Unified device detection, profiling, and memory management.
Consolidates patterns from foundation.device_manager and multiple pipelines.

Note: This module requires torch at runtime (lazy-imported in submodules).
The __init__ import is guarded so that ``import transformation_portal.core``
succeeds even when torch is absent (e.g., core-only CI).
"""

try:
    from .detector import DeviceCapabilities, DeviceDetector, DeviceInfo, DeviceType
    from .memory import MemoryManager, MemoryStats, calculate_safe_batch_size, estimate_memory_usage
    from .profiler import PerformanceProfiler, ProfileResult
except ImportError:
    # torch (or psutil) not installed — stubs will be None at package level.
    # Callers must guard usage behind availability checks.
    DeviceCapabilities = None  # type: ignore[assignment, misc]
    DeviceDetector = None  # type: ignore[assignment, misc]
    DeviceInfo = None  # type: ignore[assignment, misc]
    DeviceType = None  # type: ignore[assignment, misc]
    MemoryManager = None  # type: ignore[assignment, misc]
    MemoryStats = None  # type: ignore[assignment, misc]
    calculate_safe_batch_size = None  # type: ignore[assignment, misc]
    estimate_memory_usage = None  # type: ignore[assignment, misc]
    PerformanceProfiler = None  # type: ignore[assignment, misc]
    ProfileResult = None  # type: ignore[assignment, misc]

__all__ = [
    # Detector
    "DeviceDetector",
    "DeviceCapabilities",
    "DeviceType",
    "DeviceInfo",
    # Profiler
    "PerformanceProfiler",
    "ProfileResult",
    # Memory
    "MemoryManager",
    "MemoryStats",
    "estimate_memory_usage",
    "calculate_safe_batch_size",
]
