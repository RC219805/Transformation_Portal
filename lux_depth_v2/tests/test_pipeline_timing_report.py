"""Integration test for pipeline timing instrumentation."""

import pytest
import tempfile
import json
from pathlib import Path
import numpy as np
from PIL import Image

try:
    from lux_depth_v2.pipeline import LuxPipelineV2
    from lux_depth_v2.config import PipelineConfig, Preset
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
def test_pipeline_includes_timing_s_in_report():
    """Test that pipeline report includes timing_s field."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy test image
        img_path = tmpdir / "test_image.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(img_path)
        
        # Create dummy depth map
        depth_dir = tmpdir / "depth"
        depth_dir.mkdir()
        depth_path = depth_dir / "test_image.tif"
        depth = Image.fromarray(np.random.randint(0, 65535, (256, 256), dtype=np.uint16))
        depth.save(depth_path)
        
        # Configure pipeline with minimal processing
        cfg = PipelineConfig(
            preset=Preset.PHOTO_REALISTIC,
            input_dir=tmpdir,
            output_dir=tmpdir / "output",
            depth_dir=depth_dir,
            upscale=1,  # No upscaling for speed
            upscaler_backend="none",  # No AI upscaler
            enable_material=False,  # No material processing
            write_outputs=True,
            save_master=True,
            save_upscaled=False,
            save_preview_jpg=False,
        )
        
        # Initialize pipeline
        pipeline = LuxPipelineV2(cfg)
        
        # Process image
        result = pipeline.process_one(img_path, depth_path=depth_path)
        
        # Verify report structure
        assert "timing_s" in result, "Report missing timing_s field"
        assert isinstance(result["timing_s"], dict), "timing_s should be a dict"
        
        # Verify timing keys exist
        timing_s = result["timing_s"]
        assert len(timing_s) > 0, "timing_s should not be empty"
        
        # Check for expected stages
        expected_stages = ["io/read_input", "io/read_depth", "grade/master"]
        for stage in expected_stages:
            assert stage in timing_s, f"Expected stage '{stage}' missing from timing_s"
        
        # Verify all values are floats (seconds)
        for stage, time_val in timing_s.items():
            assert isinstance(time_val, float), f"Stage '{stage}' timing should be float"
            assert time_val >= 0, f"Stage '{stage}' timing should be non-negative"


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
def test_pipeline_timing_reasonable_values():
    """Test that timing values are reasonable (not zero, not absurd)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test image
        img_path = tmpdir / "test_image.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(img_path)
        
        # Create depth
        depth_dir = tmpdir / "depth"
        depth_dir.mkdir()
        depth_path = depth_dir / "test_image.tif"
        depth = Image.fromarray(np.random.randint(0, 65535, (256, 256), dtype=np.uint16))
        depth.save(depth_path)
        
        cfg = PipelineConfig(
            preset=Preset.PHOTO_REALISTIC,
            input_dir=tmpdir,
            output_dir=tmpdir / "output",
            depth_dir=depth_dir,
            upscale=1,
            upscaler_backend="none",
            enable_material=False,
            write_outputs=True,
            save_master=True,
            save_upscaled=False,
        )
        
        pipeline = LuxPipelineV2(cfg)
        result = pipeline.process_one(img_path, depth_path=depth_path)
        
        timing_s = result["timing_s"]
        
        # All timings should be positive
        for stage, time_val in timing_s.items():
            assert time_val > 0, f"Stage '{stage}' has zero timing"
            assert time_val < 60, f"Stage '{stage}' took too long: {time_val}s"


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
def test_pipeline_report_json_contains_timing():
    """Test that JSON report file contains timing_s."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test image
        img_path = tmpdir / "test_image.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(img_path)
        
        # Create depth
        depth_dir = tmpdir / "depth"
        depth_dir.mkdir()
        depth_path = depth_dir / "test_image.tif"
        depth = Image.fromarray(np.random.randint(0, 65535, (256, 256), dtype=np.uint16))
        depth.save(depth_path)
        
        output_dir = tmpdir / "output"
        cfg = PipelineConfig(
            preset=Preset.PHOTO_REALISTIC,
            input_dir=tmpdir,
            output_dir=output_dir,
            depth_dir=depth_dir,
            upscale=1,
            upscaler_backend="none",
            enable_material=False,
            write_outputs=True,
            save_master=True,
            save_upscaled=False,
        )
        
        pipeline = LuxPipelineV2(cfg)
        result = pipeline.process_one(img_path, depth_path=depth_path)
        
        # Load JSON report
        report_path = output_dir / "test_image_report.json"
        assert report_path.exists(), "Report JSON not written"
        
        with open(report_path) as f:
            report_data = json.load(f)
        
        # Verify timing_s in JSON
        assert "timing_s" in report_data
        assert isinstance(report_data["timing_s"], dict)
        assert len(report_data["timing_s"]) > 0


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
def test_pipeline_backward_compatibility():
    """Test that pipeline maintains backward compatibility with stage_times_sec."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        img_path = tmpdir / "test_image.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(img_path)
        
        depth_dir = tmpdir / "depth"
        depth_dir.mkdir()
        depth_path = depth_dir / "test_image.tif"
        depth = Image.fromarray(np.random.randint(0, 65535, (256, 256), dtype=np.uint16))
        depth.save(depth_path)
        
        cfg = PipelineConfig(
            preset=Preset.PHOTO_REALISTIC,
            input_dir=tmpdir,
            output_dir=tmpdir / "output",
            depth_dir=depth_dir,
            upscale=1,
            upscaler_backend="none",
            enable_material=False,
            write_outputs=True,
        )
        
        pipeline = LuxPipelineV2(cfg)
        result = pipeline.process_one(img_path, depth_path=depth_path)
        
        # Both stage_times_sec and timing_s should exist
        assert "stage_times_sec" in result
        assert "timing_s" in result
        
        # They should have the same content
        assert result["stage_times_sec"] == result["timing_s"]


@pytest.mark.skipif(not PIPELINE_AVAILABLE, reason="Pipeline not available")
def test_pipeline_timing_stage_names_stable():
    """Test that stage names are stable and follow snake_case convention."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        img_path = tmpdir / "test_image.jpg"
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        img.save(img_path)
        
        depth_dir = tmpdir / "depth"
        depth_dir.mkdir()
        depth_path = depth_dir / "test_image.tif"
        depth = Image.fromarray(np.random.randint(0, 65535, (256, 256), dtype=np.uint16))
        depth.save(depth_path)
        
        cfg = PipelineConfig(
            preset=Preset.PHOTO_REALISTIC,
            input_dir=tmpdir,
            output_dir=tmpdir / "output",
            depth_dir=depth_dir,
            upscale=1,
            upscaler_backend="none",
            enable_material=False,
            write_outputs=True,
        )
        
        pipeline = LuxPipelineV2(cfg)
        result = pipeline.process_one(img_path, depth_path=depth_path)
        
        timing_s = result["timing_s"]
        
        # All stage names should use slash separator or snake_case
        for stage_name in timing_s.keys():
            # Should not have spaces or camelCase
            assert " " not in stage_name
            # Should use / for hierarchy or _ for words
            assert "/" in stage_name or "_" in stage_name or stage_name.islower()
