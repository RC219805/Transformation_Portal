"""
Core Device Management Module

Unified device detection, profiling, and memory management.
Consolidates patterns from foundation.device_manager and multiple pipelines.
Compatibility note: retained as an internal/shared helper surface with
direct smoke coverage, but it currently has no production imports.

Note:
- ``torch`` and ``psutil`` are optional runtime dependencies.
- When either dependency is unavailable, exports in this module are set to
  ``None`` so ``import transformation_portal.core`` still succeeds in lightweight
  environments (for example, core-only CI).
"""

from importlib.util import find_spec


def _device_deps_available() -> bool:
    """Return True when all device module optional deps are importable."""
    try:
        return find_spec("torch") is not None and find_spec("psutil") is not None
    except (ImportError, ValueError):
        return False


if _device_deps_available():
    from .detector import DeviceCapabilities, DeviceDetector, DeviceInfo, DeviceType
    from .memory import MemoryManager, MemoryStats, calculate_safe_batch_size, estimate_memory_usage
    from .profiler import PerformanceProfiler, ProfileResult
else:
    # Optional dependency set incomplete — publish stubs at package level.
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
