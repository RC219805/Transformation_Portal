"""Tests for edge cases and uncommon scenarios."""

import pytest
from pathlib import Path
import numpy as np


@pytest.mark.skipif(not pytest.importorskip("torch", reason="torch not available"),
                    reason="torch not available")
def test_multi_gpu_device_selection():
    """Test explicit GPU selection when multiple GPUs available."""
    import torch
    
    if torch.cuda.device_count() < 2:
        pytest.skip("Multi-GPU test requires 2+ GPUs")
    
    # Test GPU selection via torch directly
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:1")
    
    assert device0.index == 0
    assert device1.index == 1
    
    # Verify can allocate tensors on different GPUs
    x0 = torch.zeros(10, device=device0)
    x1 = torch.zeros(10, device=device1)
    
    assert x0.device.index == 0
    assert x1.device.index == 1


def test_hdr_image_processing():
    """Test processing of HDR images."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    # Simulate HDR image with extended dynamic range
    hdr_ref = np.random.rand(100, 100, 3).astype(np.float32) * 2.0  # Values > 1.0
    hdr_proc = hdr_ref * 1.1
    
    # Should handle gracefully (clipping internally)
    metrics = computer.compute(hdr_ref, hdr_proc, metrics=["mae", "mse"])
    
    assert metrics.mae is not None
    assert metrics.mse is not None


def test_zero_sized_image():
    """Test handling of zero-sized images."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    # Zero-sized array
    empty = np.array([]).reshape(0, 0, 3).astype(np.float32)
    
    # Should handle gracefully or raise appropriate error
    try:
        metrics = computer.compute(empty, empty)
        # If it succeeds, metrics should be defined
        assert metrics is not None
    except (ValueError, ZeroDivisionError):
        # Expected for edge case
        pass


def test_single_pixel_image():
    """Test processing of 1x1 images."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    # 1x1 image
    ref = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
    proc = np.array([[[0.6, 0.6, 0.6]]], dtype=np.float32)
    
    metrics = computer.compute(ref, proc, metrics=["mae", "psnr"])
    
    assert metrics.mae is not None
    assert metrics.psnr is not None


def test_extremely_high_resolution_tiling():
    """Test tiling with extreme resolutions."""
    pytest.importorskip("torch", reason="torch required")
    
    from src.transformation_portal.core.processing.tiling import TiledProcessor
    
    processor = TiledProcessor(tile_size=256, overlap=32)
    
    # Estimate tiles for 32K image
    num_tiles = processor.estimate_tiles(30720, 30720)
    
    # Should handle large images
    assert num_tiles > 100
    assert num_tiles < 100000  # Reasonable upper bound


def test_mismatched_image_dimensions():
    """Test handling of mismatched image dimensions."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    proc = np.random.rand(150, 150, 3).astype(np.float32)
    
    # Should handle or raise appropriate error
    try:
        metrics = computer.compute(ref, proc)
    except (ValueError, AssertionError):
        # Expected - dimensions don't match
        pass


def test_grayscale_vs_color_comparison():
    """Test comparing grayscale and color images."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    grayscale = np.random.rand(100, 100).astype(np.float32)
    color = np.random.rand(100, 100, 3).astype(np.float32)
    
    # Should handle dimension mismatch gracefully
    try:
        metrics = computer.compute(grayscale, color)
    except (ValueError, AssertionError):
        # Expected - different formats
        pass


def test_special_float_values():
    """Test handling of NaN and Inf in images."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    
    # Image with NaN
    proc_nan = ref.copy()
    proc_nan[50, 50, 0] = np.nan
    
    try:
        metrics = computer.compute(ref, proc_nan, metrics=["mae"])
        # If it succeeds, result should be valid or NaN
        assert metrics.mae is None or np.isnan(metrics.mae) or np.isfinite(metrics.mae)
    except ValueError:
        # Expected - NaN not supported
        pass
    
    # Image with Inf
    proc_inf = ref.copy()
    proc_inf[50, 50, 0] = np.inf
    
    try:
        metrics = computer.compute(ref, proc_inf, metrics=["mae"])
    except ValueError:
        # Expected - Inf not supported
        pass


def test_very_dark_images():
    """Test processing very dark images (near black)."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    # Very dark images
    ref = np.random.rand(100, 100, 3).astype(np.float32) * 0.01  # Max 0.01
    proc = ref * 1.1
    
    metrics = computer.compute(ref, proc, metrics=["psnr", "mae"])
    
    # Should handle low dynamic range
    assert metrics.psnr is not None
    assert metrics.mae is not None


def test_very_bright_images():
    """Test processing very bright images (near white)."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    # Very bright images
    ref = 0.99 + np.random.rand(100, 100, 3).astype(np.float32) * 0.01
    proc = ref * 0.99
    
    metrics = computer.compute(ref, proc, metrics=["psnr", "mae"])
    
    assert metrics.psnr is not None
    assert metrics.mae is not None


def test_checkpoint_with_special_characters(tmp_path):
    """Test checkpoint paths with special characters."""
    from src.transformation_portal.core.batch.job import BatchJob, JobItem
    
    # Filename with special characters (that are valid)
    checkpoint_path = tmp_path / "job_with_special_chars_#123.json"
    
    items = [JobItem("input.jpg", "output.jpg")]
    
    job = BatchJob(
        job_id="special_test",
        items=items,
        checkpoint_path=checkpoint_path,
        created_at="2025-01-01T00:00:00Z"
    )
    
    job.save_checkpoint()
    assert checkpoint_path.exists()
    
    loaded = BatchJob.load_checkpoint(checkpoint_path)
    assert loaded.job_id == "special_test"


def test_batch_with_duplicate_outputs(tmp_path):
    """Test batch processing with duplicate output paths."""
    from src.transformation_portal.core.batch.job import BatchProcessor, JobItem
    
    input1 = tmp_path / "input1.txt"
    input2 = tmp_path / "input2.txt"
    input1.write_text("test1")
    input2.write_text("test2")
    
    def processor(path):
        class Result:
            def save(self, output_path):
                # All outputs go to same file (simulating collision)
                Path(output_path).write_text(f"processed: {path.name}")
        return Result()
    
    batch_processor = BatchProcessor(
        processor_fn=processor,
        checkpoint_dir=tmp_path / "checkpoints"
    )
    
    # Both inputs map to same output
    output_dir = tmp_path / "output"
    
    # Process - should handle duplicate outputs
    job = batch_processor.process_batch([input1, input2], output_dir)
    
    # Both should process (last one wins for file content)
    assert len(job.get_completed_items()) >= 1


def test_profiler_with_nested_operations():
    """Test profiler with nested profiled blocks."""
    from src.transformation_portal.core.device.profiler import PerformanceProfiler
    import time
    
    profiler = PerformanceProfiler(enable_memory_tracking=False)
    
    with profiler.profile("outer"):
        time.sleep(0.01)
        
        with profiler.profile("inner1"):
            time.sleep(0.01)
        
        with profiler.profile("inner2"):
            time.sleep(0.01)
    
    results = profiler.get_results()
    
    # Should have all 3 operations (order: inner1, inner2, outer)
    assert len(results) == 3
    
    # Find each operation
    names = [r.name for r in results]
    assert "outer" in names
    assert "inner1" in names
    assert "inner2" in names


def test_metrics_with_different_dtypes():
    """Test metrics computation with different data types."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    # uint8
    ref_uint8 = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    proc_uint8 = ref_uint8.copy()
    
    metrics_uint8 = computer.compute(ref_uint8, proc_uint8, metrics=["ssim"])
    assert metrics_uint8.ssim is not None
    
    # float16
    ref_float16 = np.random.rand(50, 50, 3).astype(np.float16)
    proc_float16 = ref_float16.copy()
    
    metrics_float16 = computer.compute(ref_float16, proc_float16, metrics=["ssim"])
    assert metrics_float16.ssim is not None
    
    # float64
    ref_float64 = np.random.rand(50, 50, 3).astype(np.float64)
    proc_float64 = ref_float64.copy()
    
    metrics_float64 = computer.compute(ref_float64, proc_float64, metrics=["ssim"])
    assert metrics_float64.ssim is not None


def test_report_with_very_long_paths(tmp_path):
    """Test reports with very long file paths."""
    from src.transformation_portal.core.validation.report import ProcessingReport
    
    # Very long path
    long_path = tmp_path / ("a" * 200 + ".jpg")
    
    config = {"preset": "test"}
    
    report = ProcessingReport.create(
        config=config,
        input_path=long_path,
        output_path=long_path,
        duration_ms=100.0,
        metrics={"ssim": 0.95}
    )
    
    # Should handle long paths
    assert len(report.input_path) > 200
    
    # Should save successfully
    report_path = tmp_path / "report.json"
    report.save(report_path)
    
    loaded = ProcessingReport.load(report_path)
    assert loaded.input_path == str(long_path)


def test_baseline_with_missing_metrics(tmp_path):
    """Test baseline comparison with missing metrics."""
    from src.transformation_portal.core.validation.comparison import BaselineComparator
    import json
    
    # Baseline with some metrics
    baseline_data = {
        "test_preset": {
            "ssim": 0.95,
            "psnr": 35.0
        }
    }
    
    baseline_path = tmp_path / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)
    
    comparator = BaselineComparator(tmp_path)
    
    # Current metrics with different set
    metrics = {
        "ssim": 0.96,  # Present in baseline
        "mae": 0.05    # Not in baseline
    }
    
    result = comparator.compare("test_preset", metrics)
    
    # Should only compare ssim (common metric)
    assert "ssim" in result.delta
    assert "psnr" not in result.delta  # Not in current
    assert "mae" not in result.delta   # Not in baseline
