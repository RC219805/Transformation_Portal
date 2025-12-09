"""Tests for core device module."""

import pytest

from transformation_portal.core.device import (
    DeviceDetector,
    DeviceType,
    PerformanceProfiler,
    MemoryManager,
    estimate_memory_usage,
    calculate_safe_batch_size,
)


def test_device_detector():
    """Test device detector."""
    detector = DeviceDetector()
    device_info = detector.detect()
    
    assert device_info is not None
    assert device_info.capabilities is not None
    assert device_info.capabilities.device_type in [
        DeviceType.CPU, DeviceType.CUDA, DeviceType.MPS
    ]


def test_device_detector_caching():
    """Test device detector caching."""
    detector = DeviceDetector()
    
    info1 = detector.detect()
    info2 = detector.detect()
    
    # Should return cached result
    assert info1 is info2


def test_device_detector_force_refresh():
    """Test device detector force refresh."""
    detector = DeviceDetector()
    
    info1 = detector.detect()
    info2 = detector.detect(force_refresh=True)
    
    # Should create new detection
    assert info1 is not info2


def test_device_capabilities():
    """Test device capabilities detection."""
    detector = DeviceDetector()
    device_info = detector.detect()
    
    cap = device_info.capabilities
    
    assert cap.total_memory_gb > 0
    assert cap.available_memory_gb > 0
    assert cap.available_memory_gb <= cap.total_memory_gb
    assert cap.recommended_batch_size >= 1


def test_performance_profiler():
    """Test performance profiler."""
    profiler = PerformanceProfiler(enable_memory_tracking=False)
    
    with profiler.profile("test_operation"):
        x = sum(range(1000000))
    
    results = profiler.get_results()
    assert len(results) == 1
    assert results[0].name == "test_operation"
    assert results[0].duration_ms > 0


def test_performance_profiler_multiple():
    """Test profiler with multiple operations."""
    profiler = PerformanceProfiler(enable_memory_tracking=False)
    
    with profiler.profile("op1"):
        x = sum(range(100000))
    
    with profiler.profile("op2"):
        y = sum(range(200000))
    
    results = profiler.get_results()
    assert len(results) == 2
    
    # Get specific result
    op1_result = profiler.get_result("op1")
    assert op1_result is not None
    assert op1_result.name == "op1"


def test_memory_manager():
    """Test memory manager."""
    manager = MemoryManager()
    
    # Get stats (may be None if psutil not available)
    stats = manager.get_stats()
    if stats:
        assert stats.total_mb > 0
        assert stats.available_mb > 0
        assert 0 <= stats.percent <= 100


def test_memory_manager_process():
    """Test process memory tracking."""
    manager = MemoryManager()
    
    process_mem = manager.get_process_memory_mb()
    if process_mem:
        assert process_mem > 0


def test_estimate_memory_usage():
    """Test memory usage estimation."""
    # 4K image (3840x2160)
    memory_mb = estimate_memory_usage(3840, 2160)
    
    assert memory_mb > 0
    # Should be reasonable (around 285MB for float32 RGB with 3x overhead)
    assert 50 < memory_mb < 400


def test_calculate_safe_batch_size():
    """Test batch size calculation."""
    # 16GB available, 4K images
    batch_size = calculate_safe_batch_size(
        image_width=3840,
        image_height=2160,
        available_memory_gb=16.0,
        memory_reserve_gb=2.0
    )
    
    assert batch_size >= 1
    # Should handle at least a few 4K images
    assert batch_size >= 1


def test_calculate_safe_batch_size_low_memory():
    """Test batch size with low memory."""
    # Only 2GB available
    batch_size = calculate_safe_batch_size(
        image_width=7680,
        image_height=4320,
        available_memory_gb=2.0,
        memory_reserve_gb=0.5
    )
    
    # Should still return at least 1
    assert batch_size >= 1
