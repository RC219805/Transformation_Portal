"""
Core Device Management Module

Unified device detection, profiling, and memory management.
Consolidates patterns from foundation.device_manager and multiple pipelines.
"""

from .detector import (
    DeviceDetector,
    DeviceCapabilities,
    DeviceType,
    DeviceInfo,
)
from .profiler import (
    PerformanceProfiler,
    ProfileResult,
)
from .memory import (
    MemoryManager,
    MemoryStats,
    estimate_memory_usage,
    calculate_safe_batch_size,
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
