"""Tests for StageProfiler."""

import pytest
import time
from src.transformation_portal.core.device.profiler import StageProfiler


def test_stage_profiler_basic():
    """Test basic stage timing."""
    profiler = StageProfiler(enabled=True)
    
    with profiler.stage("test_stage"):
        time.sleep(0.01)  # 10ms
    
    timings = profiler.summary_s()
    assert "test_stage" in timings
    assert timings["test_stage"] >= 0.01


def test_stage_profiler_multiple_stages():
    """Test multiple stages."""
    profiler = StageProfiler(enabled=True)
    
    with profiler.stage("stage1"):
        time.sleep(0.005)
    
    with profiler.stage("stage2"):
        time.sleep(0.01)
    
    timings = profiler.summary_s()
    assert len(timings) == 2
    assert "stage1" in timings
    assert "stage2" in timings
    assert timings["stage1"] >= 0.005
    assert timings["stage2"] >= 0.01


def test_stage_profiler_accumulation():
    """Test stage accumulation for repeated stages."""
    profiler = StageProfiler(enabled=True)
    
    # Same stage entered twice
    with profiler.stage("repeated"):
        time.sleep(0.005)
    
    with profiler.stage("repeated"):
        time.sleep(0.005)
    
    timings = profiler.summary_s()
    assert "repeated" in timings
    # Should accumulate both sleeps
    assert timings["repeated"] >= 0.01


def test_stage_profiler_disabled():
    """Test profiler when disabled."""
    profiler = StageProfiler(enabled=False)
    
    with profiler.stage("test_stage"):
        time.sleep(0.01)
    
    timings = profiler.summary_s()
    assert len(timings) == 0


def test_stage_profiler_clear():
    """Test clearing stage timings."""
    profiler = StageProfiler(enabled=True)
    
    with profiler.stage("stage1"):
        time.sleep(0.005)
    
    assert len(profiler.summary_s()) == 1
    
    profiler.clear()
    assert len(profiler.summary_s()) == 0


def test_stage_profiler_overhead():
    """Test that profiler overhead is minimal (<5% on real work)."""
    profiler = StageProfiler(enabled=True)
    iterations = 100
    work_duration = 0.001  # 1ms of actual work
    
    # Measure without profiling
    start = time.perf_counter()
    for _ in range(iterations):
        time.sleep(work_duration)
    baseline = time.perf_counter() - start
    
    # Measure with profiling
    start = time.perf_counter()
    for _ in range(iterations):
        with profiler.stage("overhead_test"):
            time.sleep(work_duration)
    profiled = time.perf_counter() - start
    
    # Calculate overhead percentage
    overhead_pct = ((profiled - baseline) / baseline) * 100
    
    # Should be < 5% overhead on real work (generous for test stability)
    assert overhead_pct < 5.0


def test_stage_profiler_summary_returns_dict():
    """Test that summary_s returns a valid dictionary."""
    profiler = StageProfiler(enabled=True)
    
    with profiler.stage("test"):
        time.sleep(0.001)
    
    summary = profiler.summary_s()
    assert isinstance(summary, dict)
    assert all(isinstance(k, str) for k in summary.keys())
    assert all(isinstance(v, float) for v in summary.values())


def test_stage_profiler_snake_case_keys():
    """Test that stage names use snake_case."""
    profiler = StageProfiler(enabled=True)
    
    # Test standard stage names from pipeline
    stage_names = ["load", "depth", "material", "grade", "upscale_infer", 
                   "export_master", "export_upscaled"]
    
    for name in stage_names:
        with profiler.stage(name):
            time.sleep(0.001)
    
    summary = profiler.summary_s()
    for name in stage_names:
        assert name in summary
        assert summary[name] > 0


def test_stage_profiler_nested_stages():
    """Test nested stage contexts (inner stages accumulate)."""
    profiler = StageProfiler(enabled=True)
    
    with profiler.stage("outer"):
        time.sleep(0.005)
        with profiler.stage("inner"):
            time.sleep(0.005)
        time.sleep(0.005)
    
    summary = profiler.summary_s()
    # Both stages should exist
    assert "outer" in summary
    assert "inner" in summary
    # Outer should include all sleeps
    assert summary["outer"] >= 0.015
    # Inner should only have its sleep
    assert summary["inner"] >= 0.005
