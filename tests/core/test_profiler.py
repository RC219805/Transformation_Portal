"""Tests for GPU profiler."""

import pytest
import time
from src.transformation_portal.core.device.profiler import (
    PerformanceProfiler,
    GPUProfiler,
    ProfileResult
)


def test_performance_profiler_basic():
    """Test basic profiling."""
    profiler = PerformanceProfiler(enable_memory_tracking=False)
    
    with profiler.profile("test_operation"):
        time.sleep(0.01)  # 10ms
    
    results = profiler.get_results()
    assert len(results) == 1
    assert results[0].name == "test_operation"
    assert results[0].duration_ms >= 10.0


def test_performance_profiler_multiple_operations():
    """Test profiling multiple operations."""
    profiler = PerformanceProfiler(enable_memory_tracking=False)
    
    with profiler.profile("op1"):
        time.sleep(0.005)
    
    with profiler.profile("op2"):
        time.sleep(0.01)
    
    results = profiler.get_results()
    assert len(results) == 2
    assert results[0].name == "op1"
    assert results[1].name == "op2"


def test_performance_profiler_get_result_by_name():
    """Test getting specific result by name."""
    profiler = PerformanceProfiler(enable_memory_tracking=False)
    
    with profiler.profile("target_op"):
        time.sleep(0.005)
    
    result = profiler.get_result("target_op")
    assert result is not None
    assert result.name == "target_op"
    
    missing = profiler.get_result("nonexistent")
    assert missing is None


def test_performance_profiler_clear():
    """Test clearing results."""
    profiler = PerformanceProfiler(enable_memory_tracking=False)
    
    with profiler.profile("op1"):
        pass
    
    assert len(profiler.get_results()) == 1
    
    profiler.clear()
    assert len(profiler.get_results()) == 0


def test_performance_profiler_metadata():
    """Test metadata storage."""
    profiler = PerformanceProfiler(enable_memory_tracking=False)
    
    with profiler.profile("op1", key="value", num=42):
        pass
    
    result = profiler.get_result("op1")
    assert result.metadata["key"] == "value"
    assert result.metadata["num"] == 42


@pytest.mark.skipif(not pytest.importorskip("psutil", reason="psutil not available"),
                    reason="psutil not available")
def test_performance_profiler_memory_tracking():
    """Test memory tracking."""
    profiler = PerformanceProfiler(enable_memory_tracking=True)
    
    with profiler.profile("memory_op"):
        # Allocate some memory
        data = [0] * 1000000
        _ = data  # Use the variable
    
    result = profiler.get_result("memory_op")
    assert result.memory_start_mb is not None
    assert result.memory_end_mb is not None


def test_gpu_profiler_disabled():
    """Test GPU profiler when disabled."""
    profiler = GPUProfiler(enabled=False)
    
    with profiler.profile("op1"):
        time.sleep(0.005)
    
    report = profiler.report()
    assert report["total_ms"] == 0.0
    assert len(report["stages"]) == 0


def test_gpu_profiler_enabled_cpu_only():
    """Test GPU profiler on CPU (no GPU timing)."""
    profiler = GPUProfiler(enabled=True)
    
    with profiler.profile("cpu_op"):
        time.sleep(0.01)
    
    report = profiler.report()
    assert report["total_ms"] >= 10.0
    assert len(report["stages"]) == 1
    assert "cpu_ms" in report["stages"][0]


def test_gpu_profiler_clear():
    """Test clearing GPU profiler."""
    profiler = GPUProfiler(enabled=True)
    
    with profiler.profile("op1"):
        pass
    
    assert len(profiler.report()["stages"]) == 1
    
    profiler.clear()
    assert len(profiler.report()["stages"]) == 0


@pytest.mark.skipif(not pytest.importorskip("torch", reason="torch not available"),
                    reason="torch not available")
def test_performance_profiler_with_torch():
    """Test profiler with torch tensors."""
    import torch
    
    profiler = PerformanceProfiler(enable_gpu_profiling=True)
    
    with profiler.profile("tensor_op"):
        x = torch.randn(100, 100)
        y = torch.mm(x, x)
        _ = y  # Use result
    
    result = profiler.get_result("tensor_op")
    assert result is not None
    assert result.duration_ms > 0


def test_profile_result_properties():
    """Test ProfileResult properties."""
    result = ProfileResult(
        name="test",
        duration_ms=100.0,
        memory_start_mb=50.0,
        memory_end_mb=75.0,
        gpu_memory_start_mb=100.0,
        gpu_memory_end_mb=150.0
    )
    
    assert result.memory_delta_mb == 25.0
    assert result.gpu_memory_delta_mb == 50.0


def test_profile_result_str():
    """Test ProfileResult string representation."""
    result = ProfileResult(
        name="test_op",
        duration_ms=123.4,
        memory_start_mb=50.0,
        memory_end_mb=75.0
    )
    
    string = str(result)
    assert "test_op" in string
    assert "123.4ms" in string
    assert "25.0MB" in string
