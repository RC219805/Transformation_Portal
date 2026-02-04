"""
Core Device Management Module

Unified device detection, profiling, and memory management.
Consolidates patterns from foundation.device_manager and multiple pipelines.
"""

from .detector import (
    DeviceCapabilities,
    DeviceDetector,
    DeviceInfo,
    DeviceType,
)
from .memory import (
    MemoryManager,
    MemoryStats,
    calculate_safe_batch_size,
    estimate_memory_usage,
)
from .profiler import (
    PerformanceProfiler,
    ProfileResult,
)

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
