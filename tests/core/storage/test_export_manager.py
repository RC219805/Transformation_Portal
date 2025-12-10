"""
Unit tests for ExportManager (Phase 2 Slice 2).

Tests verify:
1. Path naming matches existing behavior exactly
2. Delegation to I/O functions is correct
3. ExportConfig variations work as expected
4. Error handling is appropriate
"""
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from transformation_portal.core.storage import ExportConfig, ExportManager


class TestExportConfig:
    """Test ExportConfig dataclass."""
    
    def test_default_values(self, tmp_path):
        """Verify default naming conventions match existing behavior."""
        cfg = ExportConfig(output_dir=tmp_path)
        
        assert cfg.output_dir == tmp_path
        assert cfg.master_prefix == ""
        assert cfg.upscaled_prefix == ""
        assert cfg.preview_prefix == ""
        assert cfg.report_suffix == "_report.json"
        assert cfg.master_suffix == "_master16"
        assert cfg.upscaled_suffix == "_upscaled16"
        assert cfg.marketing_suffix == "_marketing"
        assert cfg.preview_jpg_suffix == "_preview"
    
    def test_custom_prefixes(self, tmp_path):
        """Test custom prefix configuration."""
        cfg = ExportConfig(
            output_dir=tmp_path,
            master_prefix="gold_",
            upscaled_prefix="hires_"
        )
        
        assert cfg.master_prefix == "gold_"
        assert cfg.upscaled_prefix == "hires_"
    
    def test_frozen_dataclass(self, tmp_path):
        """Verify immutability for thread safety."""
        cfg = ExportConfig(output_dir=tmp_path)
        
        with pytest.raises(Exception):  # FrozenInstanceError
            cfg.output_dir = Path("/new/path")


class TestExportManager:
    """Test ExportManager behavior-identical I/O delegation."""
    
    @pytest.fixture
    def mock_io_utils(self):
        """Mock I/O module for testing."""
        mock = MagicMock()
        mock.atomic_write_rgb16_tiff = MagicMock()
        mock.atomic_write_png8 = MagicMock()
        mock.atomic_write_jpg8 = MagicMock()
        return mock
    
    @pytest.fixture
    def export_manager(self, tmp_path, mock_io_utils):
        """Create ExportManager with mock I/O."""
        cfg = ExportConfig(output_dir=tmp_path)
        return ExportManager(cfg, mock_io_utils)
    
    @pytest.fixture
    def sample_image(self):
        """Create sample RGB float32 array."""
        return np.random.rand(100, 100, 3).astype(np.float32)
    
    def test_init_creates_output_dir(self, tmp_path, mock_io_utils):
        """Verify output directory is created on init."""
        out_dir = tmp_path / "outputs"
        assert not out_dir.exists()
        
        cfg = ExportConfig(output_dir=out_dir)
        manager = ExportManager(cfg, mock_io_utils)
        
        assert out_dir.exists()
        assert manager.config.output_dir == out_dir
    
    def test_write_master_path_naming(self, export_manager, sample_image, mock_io_utils):
        """Verify master TIFF path matches existing naming: stem_master16.tif."""
        result_path = export_manager.write_master("test_image", sample_image)
        
        assert result_path.name == "test_image_master16.tif"
        assert result_path.parent == export_manager.config.output_dir
        
        # Verify delegation
        mock_io_utils.atomic_write_rgb16_tiff.assert_called_once()
        call_args = mock_io_utils.atomic_write_rgb16_tiff.call_args
        assert call_args[0][0] == result_path
        assert np.array_equal(call_args[0][1], sample_image)
        assert call_args[1]["compression"] == "deflate"
    
    def test_write_upscaled_path_naming(self, export_manager, sample_image, mock_io_utils):
        """Verify upscaled TIFF path matches existing naming: stem_upscaled16.tif."""
        result_path = export_manager.write_upscaled("test_image", sample_image)
        
        assert result_path.name == "test_image_upscaled16.tif"
        mock_io_utils.atomic_write_rgb16_tiff.assert_called_once()
    
    def test_write_preview_path_naming(self, export_manager, sample_image, mock_io_utils):
        """Verify preview JPG path matches existing naming: stem_preview.jpg."""
        result_path = export_manager.write_preview("test_image", sample_image, quality=92)
        
        assert result_path.name == "test_image_preview.jpg"
        mock_io_utils.atomic_write_jpg8.assert_called_once()
        
        call_args = mock_io_utils.atomic_write_jpg8.call_args
        assert call_args[1]["quality"] == 92
    
    def test_write_marketing_png_path_naming(self, export_manager, sample_image, mock_io_utils):
        """Verify marketing PNG path matches existing naming: stem_marketing.png."""
        result_path = export_manager.write_marketing_png("test_image", sample_image)
        
        assert result_path.name == "test_image_marketing.png"
        mock_io_utils.atomic_write_png8.assert_called_once()
    
    def test_write_report_path_naming(self, export_manager, mock_io_utils, tmp_path):
        """Verify report JSON path matches existing naming: stem_report.json."""
        report_dict = {"status": "ok", "timing_s": 1.234}
        
        result_path = export_manager.write_report("test_image", report_dict)
        
        assert result_path.name == "test_image_report.json"
        assert result_path.exists()
        
        # Verify JSON content matches
        with open(result_path) as f:
            loaded = json.load(f)
        assert loaded == report_dict
    
    def test_write_report_atomic(self, export_manager, tmp_path):
        """Verify report uses atomic write pattern."""
        report_dict = {"test": "data"}
        result_path = export_manager.write_report("test", report_dict)
        
        # Verify no .tmp file remains
        tmp_files = list(tmp_path.glob("*.tmp"))
        assert len(tmp_files) == 0
        
        assert result_path.exists()
    
    def test_export_manager_custom_prefixes(self, tmp_path, mock_io_utils, sample_image):
        """Test ExportManager with custom prefix configuration."""
        cfg = ExportConfig(
            output_dir=tmp_path,
            master_prefix="gold_",
            upscaled_prefix="hires_"
        )
        manager = ExportManager(cfg, mock_io_utils)
        
        master_path = manager.write_master("test", sample_image)
        assert master_path.name == "gold_test_master16.tif"
        
        upscaled_path = manager.write_upscaled("test", sample_image)
        assert upscaled_path.name == "hires_test_upscaled16.tif"
    
    def test_get_path_methods(self, export_manager):
        """Verify path getter methods match write methods."""
        stem = "test_image"
        
        assert export_manager.get_master_path(stem).name == "test_image_master16.tif"
        assert export_manager.get_upscaled_path(stem).name == "test_image_upscaled16.tif"
        assert export_manager.get_marketing_path(stem).name == "test_image_marketing.png"
        assert export_manager.get_preview_path(stem).name == "test_image_preview.jpg"
        assert export_manager.get_report_path(stem).name == "test_image_report.json"
    
    def test_compression_parameter(self, export_manager, sample_image, mock_io_utils):
        """Verify compression parameter is passed through."""
        export_manager.write_master("test", sample_image, compression="lzw")
        
        call_args = mock_io_utils.atomic_write_rgb16_tiff.call_args
        assert call_args[1]["compression"] == "lzw"
    
    def test_write_report_indentation(self, export_manager, tmp_path):
        """Verify report JSON uses indent=2 (matching existing behavior)."""
        report = {"key": "value", "nested": {"data": [1, 2, 3]}}
        result_path = export_manager.write_report("test", report)
        
        content = result_path.read_text()
        # Check for indentation
        assert "  " in content
        assert "{\n  \"key\"" in content or "{\n  \"nested\"" in content


class TestExportManagerIntegration:
    """Integration tests with real I/O (requires lux_depth_v2 dependencies)."""
    
    @pytest.fixture
    def real_export_manager(self, tmp_path):
        """Create ExportManager with real io_utils."""
        try:
            # Import real io_utils from lux_depth_v2
            import sys
            from pathlib import Path as _Path
            lux_path = _Path(__file__).parent.parent.parent.parent / "lux_depth_v2"
            if str(lux_path) not in sys.path:
                sys.path.insert(0, str(lux_path))
            
            import io_utils
            
            # Check if dependencies are available
            try:
                io_utils.ensure_deps()
            except RuntimeError as e:
                pytest.skip(f"Missing dependencies: {e}")
            
            cfg = ExportConfig(output_dir=tmp_path)
            return ExportManager(cfg, io_utils)
        except ImportError as e:
            pytest.skip(f"lux_depth_v2.io_utils not available: {e}")
    
    def test_real_tiff_write(self, real_export_manager, tmp_path):
        """Test real TIFF write produces valid file."""
        image = np.random.rand(50, 50, 3).astype(np.float32)
        
        result_path = real_export_manager.write_master("real_test", image)
        
        assert result_path.exists()
        assert result_path.suffix == ".tif"
        
        # Verify file is readable
        try:
            import tifffile
            loaded = tifffile.imread(str(result_path))
            assert loaded.dtype == np.uint16
            assert loaded.shape == (50, 50, 3)
        except ImportError:
            pytest.skip("tifffile not available for verification")
    
    def test_real_png_write(self, real_export_manager, tmp_path):
        """Test real PNG write produces valid file."""
        image = np.random.rand(50, 50, 3).astype(np.float32)
        
        result_path = real_export_manager.write_marketing_png("real_test", image)
        
        assert result_path.exists()
        assert result_path.suffix == ".png"
    
    def test_real_jpg_write(self, real_export_manager, tmp_path):
        """Test real JPG write produces valid file."""
        image = np.random.rand(50, 50, 3).astype(np.float32)
        
        result_path = real_export_manager.write_preview("real_test", image, quality=85)
        
        assert result_path.exists()
        assert result_path.suffix == ".jpg"
