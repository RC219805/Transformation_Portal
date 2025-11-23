"""
Foundation Architecture for Transformation Portal

Phase 1: Computational Substrate for Apple Silicon M4 Max

This module provides the core tensor processing framework and memory allocation
patterns optimized for Apple Silicon M4 Max architecture. All components are
designed to leverage Metal Performance Shaders (MPS), Neural Engine (ANE),
and unified memory architecture for optimal performance.

Key Components:
- Device Manager: M4 Max-optimized device detection and configuration
- Tensor Processor: Advanced tensor operations with hardware acceleration
- Memory Manager: Intelligent allocation patterns for unified memory
- Hardware Abstraction: Platform-agnostic interface for tensor operations
- Performance Monitor: Real-time profiling and optimization feedback

Usage:
    from transformation_portal.foundation import ComputationalSubstrate

    # Initialize foundation layer
    substrate = ComputationalSubstrate()

    # Get optimized device
    device = substrate.get_device()

    # Allocate tensors with optimal memory patterns
    tensor = substrate.allocate_tensor(shape=(1024, 1024, 3))
"""

from .substrate import ComputationalSubstrate, SubstrateConfig
from .device_manager import DeviceManager, DeviceInfo, DeviceCapabilities
from .tensor_processor import TensorProcessor, TensorConfig, PrecisionMode
from .memory_manager import MemoryManager, AllocationStrategy
from .hardware_abstraction import HardwareAbstraction, BackendType
from .performance_monitor import PerformanceMonitor, MetricsCollector

__all__ = [
    "ComputationalSubstrate",
    "SubstrateConfig",
    "DeviceManager",
    "DeviceInfo",
    "DeviceCapabilities",
    "TensorProcessor",
    "TensorConfig",
    "PrecisionMode",
    "MemoryManager",
    "AllocationStrategy",
    "HardwareAbstraction",
    "BackendType",
    "PerformanceMonitor",
    "MetricsCollector",
]

__version__ = "1.0.0"
