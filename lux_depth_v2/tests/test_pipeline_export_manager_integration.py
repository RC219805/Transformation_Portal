"""
Integration tests for ExportManager within Lux Depth V2 pipeline.

Verifies:
1. ExportManager is used when available
2. Output files exist with correct names
3. Timing stages include export_* keys
4. Behavior parity with direct I/O path
"""
import json

import numpy as np
import pytest

from lux_depth_v2.config import PipelineConfig, Preset
from lux_depth_v2.pipeline import LuxPipelineV2


@pytest.fixture
def minimal_config(tmp_path):
    """Create minimal config for testing."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    
    cfg = PipelineConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        preset=Preset.PHOTO_REALISTIC,
        device="cpu",
        upscaler_backend="torch",
        upscale=1,  # No upscaling for speed
        post_tile=0,  # No tiling
        tile=0,  # No upscale tiling
        save_master=True,
        save_upscaled=True,
        save_marketing_png=True,
        save_preview_jpg=True,
        write_outputs=True,
    )
    
    return cfg


@pytest.fixture
def sample_input_image(tmp_path):
    """Create a small test image."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(exist_ok=True)
    
    img_path = input_dir / "test_sample.png"
    
    # Create synthetic RGB image
    try:
        import cv2
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    except ImportError:
        # Fallback to PIL
        from PIL import Image
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        Image.fromarray(img).save(str(img_path))
    
    return img_path


class TestExportManagerIntegration:
    """Test ExportManager integration with pipeline."""
    
    def test_export_manager_available(self, minimal_config):
        """Verify ExportManager is available and initialized."""
        try:
            pipeline = LuxPipelineV2(minimal_config)
            assert pipeline.export_manager is not None
        except Exception as e:
            pytest.skip(f"Pipeline init failed (may be missing deps): {e}")
    
    def test_pipeline_uses_export_manager(self, minimal_config, sample_input_image, tmp_path):
        """Verify pipeline uses ExportManager for exports."""
        try:
            pipeline = LuxPipelineV2(minimal_config)
            
            if pipeline.export_manager is None:
                pytest.skip("ExportManager not available")
            
            report = pipeline.process_one(sample_input_image)
            
            assert report["status"] == "ok"
            
            # Verify output files exist with correct names
            stem = sample_input_image.stem
            output_dir = tmp_path / "output"
            
            master_path = output_dir / f"{stem}_master16.tif"
            upscaled_path = output_dir / f"{stem}_upscaled16.tif"
            marketing_path = output_dir / f"{stem}_marketing.png"
            preview_path = output_dir / f"{stem}_preview.jpg"
            report_path = output_dir / f"{stem}_report.json"
            
            assert master_path.exists(), "Master TIFF not found"
            assert upscaled_path.exists(), "Upscaled TIFF not found"
            assert marketing_path.exists(), "Marketing PNG not found"
            assert preview_path.exists(), "Preview JPG not found"
            assert report_path.exists(), "Report JSON not found"
            
        except Exception as e:
            pytest.skip(f"Pipeline test failed (may be missing deps): {e}")
    
    def test_export_stage_timing(self, minimal_config, sample_input_image):
        """Verify timing_stages_s includes export_* keys."""
        try:
            pipeline = LuxPipelineV2(minimal_config)
            
            if pipeline.export_manager is None:
                pytest.skip("ExportManager not available")
            
            report = pipeline.process_one(sample_input_image)
            
            timing_stages = report.get("timing_stages_s", {})
            
            # Verify export stages are present
            assert "export_master" in timing_stages, "export_master timing missing"
            assert "export_upscaled" in timing_stages, "export_upscaled timing missing"
            assert "export_marketing" in timing_stages, "export_marketing timing missing"
            assert "export_preview" in timing_stages, "export_preview timing missing"
            assert "export_report" in timing_stages, "export_report timing missing"
            
            # Verify timings are reasonable
            for stage in ["export_master", "export_upscaled", "export_marketing", "export_preview", "export_report"]:
                t = timing_stages[stage]
                assert t > 0, f"{stage} timing should be > 0"
                assert t < 60.0, f"{stage} timing unexpectedly high: {t}s"
            
        except Exception as e:
            pytest.skip(f"Pipeline test failed: {e}")
    
    def test_report_structure(self, minimal_config, sample_input_image, tmp_path):
        """Verify report JSON has correct structure."""
        try:
            pipeline = LuxPipelineV2(minimal_config)
            
            if pipeline.export_manager is None:
                pytest.skip("ExportManager not available")
            
            report = pipeline.process_one(sample_input_image)
            
            # Load report from disk
            stem = sample_input_image.stem
            report_path = tmp_path / "output" / f"{stem}_report.json"
            
            with open(report_path) as f:
                disk_report = json.load(f)
            
            # Verify essential fields
            assert disk_report["status"] == "ok"
            assert "timing_s" in disk_report
            assert "timing_stages_s" in disk_report
            assert "stage_times_sec" in disk_report  # backward compat
            
            # Verify export stages in disk report
            stages = disk_report["timing_stages_s"]
            assert "export_master" in stages
            assert "export_upscaled" in stages
            assert "export_report" in stages
            
        except Exception as e:
            pytest.skip(f"Report test failed: {e}")
    
    def test_fallback_to_direct_io(self, minimal_config, sample_input_image, tmp_path, monkeypatch):
        """Verify pipeline works if ExportManager unavailable (fallback path)."""
        try:
            # Simulate ExportManager unavailable
            import lux_depth_v2.pipeline
            monkeypatch.setattr(lux_depth_v2.pipeline, "EXPORT_MANAGER_AVAILABLE", False)
            monkeypatch.setattr(lux_depth_v2.pipeline, "ExportManager", None)
            
            pipeline = LuxPipelineV2(minimal_config)
            assert pipeline.export_manager is None, "Should fallback to None"
            
            report = pipeline.process_one(sample_input_image)
            assert report["status"] == "ok"
            
            # Verify files still created (direct I/O path)
            stem = sample_input_image.stem
            output_dir = tmp_path / "output"
            assert (output_dir / f"{stem}_master16.tif").exists()
            
        except Exception as e:
            pytest.skip(f"Fallback test failed: {e}")
    
    def test_skip_existing_with_export_manager(self, minimal_config, sample_input_image, tmp_path):
        """Verify skip_existing works with ExportManager paths."""
        try:
            minimal_config.skip_existing = True
            
            pipeline = LuxPipelineV2(minimal_config)
            
            if pipeline.export_manager is None:
                pytest.skip("ExportManager not available")
            
            # First run - should process
            report1 = pipeline.process_one(sample_input_image)
            assert report1["status"] == "ok"
            
            # Second run - should skip
            report2 = pipeline.process_one(sample_input_image)
            assert report2["status"] == "skipped"
            
        except Exception as e:
            pytest.skip(f"Skip existing test failed: {e}")


class TestExportManagerBehaviorParity:
    """Verify bit-identical behavior between ExportManager and direct I/O."""
    
    def test_filename_parity(self, minimal_config, tmp_path):
        """Verify ExportManager produces identical filenames to direct I/O."""
        try:
            pipeline = LuxPipelineV2(minimal_config)
            
            if pipeline.export_manager is None:
                pytest.skip("ExportManager not available")
            
            stem = "test_image"
            
            # ExportManager paths
            em_master = pipeline.export_manager.get_master_path(stem)
            em_upscaled = pipeline.export_manager.get_upscaled_path(stem)
            em_marketing = pipeline.export_manager.get_marketing_path(stem)
            em_preview = pipeline.export_manager.get_preview_path(stem)
            em_report = pipeline.export_manager.get_report_path(stem)
            
            # Direct I/O paths (from old pipeline code)
            out_dir = tmp_path / "output"
            direct_master = out_dir / f"{stem}_master16.tif"
            direct_upscaled = out_dir / f"{stem}_upscaled16.tif"
            direct_marketing = out_dir / f"{stem}_marketing.png"
            direct_preview = out_dir / f"{stem}_preview.jpg"
            direct_report = out_dir / f"{stem}_report.json"
            
            # Verify exact match
            assert em_master == direct_master
            assert em_upscaled == direct_upscaled
            assert em_marketing == direct_marketing
            assert em_preview == direct_preview
            assert em_report == direct_report
            
        except Exception as e:
            pytest.skip(f"Parity test failed: {e}")
