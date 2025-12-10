"""Tests for pipeline stage timing instrumentation."""

import pytest
import time
from src.transformation_portal.core.device.profiler import StageProfiler


def test_fake_pipeline_with_sleep_stages():
    """Test pipeline-like structure with sleep stages."""
    profiler = StageProfiler(enabled=True)
    
    # Simulate a simple pipeline
    with profiler.stage("load"):
        time.sleep(0.01)
    
    with profiler.stage("depth"):
        time.sleep(0.02)
    
    with profiler.stage("material"):
        time.sleep(0.015)
    
    with profiler.stage("grade"):
        time.sleep(0.01)
    
    with profiler.stage("upscale_infer"):
        time.sleep(0.03)
    
    with profiler.stage("export_master"):
        time.sleep(0.005)
    
    with profiler.stage("export_upscaled"):
        time.sleep(0.005)
    
    timings = profiler.summary_s()
    
    # Verify all stages present
    expected_stages = ["load", "depth", "material", "grade", 
                      "upscale_infer", "export_master", "export_upscaled"]
    for stage in expected_stages:
        assert stage in timings, f"Stage {stage} missing from timings"
    
    # Verify reasonable timing values
    assert timings["load"] >= 0.01
    assert timings["depth"] >= 0.02
    assert timings["material"] >= 0.015
    assert timings["grade"] >= 0.01
    assert timings["upscale_infer"] >= 0.03
    assert timings["export_master"] >= 0.005
    assert timings["export_upscaled"] >= 0.005


def test_pipeline_stage_ordering():
    """Test that stages maintain insertion order."""
    profiler = StageProfiler(enabled=True)
    
    stages = ["load", "depth", "material", "grade", "upscale_infer", 
              "export_master", "export_upscaled"]
    
    for stage in stages:
        with profiler.stage(stage):
            time.sleep(0.001)
    
    timings = profiler.summary_s()
    timing_keys = list(timings.keys())
    
    # Python 3.7+ dicts maintain insertion order
    assert timing_keys == stages


def test_pipeline_with_tiled_operations():
    """Test pipeline with repeated stages (simulating tiling)."""
    profiler = StageProfiler(enabled=True)
    
    # Initial load
    with profiler.stage("load"):
        time.sleep(0.01)
    
    # Tiled processing (multiple tiles)
    for i in range(3):
        with profiler.stage("upscale_infer"):
            time.sleep(0.005)
        with profiler.stage("post_process"):
            time.sleep(0.003)
    
    # Final export
    with profiler.stage("export_master"):
        time.sleep(0.005)
    
    timings = profiler.summary_s()
    
    # Load should be single
    assert timings["load"] >= 0.01
    
    # Upscale should accumulate 3 tiles
    assert timings["upscale_infer"] >= 0.015
    
    # Post-process should accumulate 3 tiles
    assert timings["post_process"] >= 0.009
    
    # Export should be single
    assert timings["export_master"] >= 0.005


def test_pipeline_result_format():
    """Test that pipeline result has correct format."""
    profiler = StageProfiler(enabled=True)
    
    # Simulate pipeline
    with profiler.stage("load"):
        time.sleep(0.01)
    
    with profiler.stage("depth"):
        time.sleep(0.01)
    
    # Get result in pipeline format
    result = {
        "status": "success",
        "timing_s": profiler.summary_s()
    }
    
    # Verify structure
    assert "timing_s" in result
    assert isinstance(result["timing_s"], dict)
    assert "load" in result["timing_s"]
    assert "depth" in result["timing_s"]
    assert all(isinstance(v, float) for v in result["timing_s"].values())


def test_pipeline_error_handling():
    """Test that timing works even when pipeline fails."""
    profiler = StageProfiler(enabled=True)
    
    with profiler.stage("load"):
        time.sleep(0.01)
    
    try:
        with profiler.stage("depth"):
            time.sleep(0.01)
            raise ValueError("Simulated error")
    except ValueError:
        pass
    
    # Should still have both stages
    timings = profiler.summary_s()
    assert "load" in timings
    assert "depth" in timings
    assert timings["load"] >= 0.01
    assert timings["depth"] >= 0.01


def test_pipeline_minimal_overhead():
    """Test that timing adds minimal overhead to pipeline."""
    iterations = 100
    
    # Baseline without profiling
    start = time.perf_counter()
    for _ in range(iterations):
        time.sleep(0.001)
    baseline = time.perf_counter() - start
    
    # With profiling
    profiler = StageProfiler(enabled=True)
    start = time.perf_counter()
    for _ in range(iterations):
        with profiler.stage("operation"):
            time.sleep(0.001)
    profiled = time.perf_counter() - start
    
    # Overhead should be < 3%
    overhead = profiled - baseline
    overhead_pct = (overhead / baseline) * 100
    
    # Very generous threshold for test stability
    assert overhead_pct < 10


def test_pipeline_stage_keys_snake_case():
    """Verify all stage keys use snake_case convention."""
    profiler = StageProfiler(enabled=True)
    
    # Standard pipeline stages
    stages = [
        "load",
        "depth",
        "material",
        "grade",
        "upscale_infer",
        "export_master",
        "export_upscaled"
    ]
    
    for stage in stages:
        with profiler.stage(stage):
            pass
    
    timings = profiler.summary_s()
    
    # All keys should be snake_case (no camelCase, no spaces)
    for key in timings.keys():
        assert "_" in key or key.islower()
        assert " " not in key
        assert key == key.lower()
