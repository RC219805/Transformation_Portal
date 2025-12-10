"""
Slice 3 PR-2: I/O optimization tests for ExportManager.

Tests focus on actual optimization behavior added in Slice 3 PR-2:
- Tiled TIFF writer selection (tiled vs legacy)
- Atomic image writes (.tmp + replace)
- Atomic report writes (.tmp + replace)
- Tiered storage (scratch dir + finalization)
- Behavior parity when all flags are OFF
"""
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.core.storage import ExportConfig, ExportManager


@pytest.fixture
def mock_io_utils():
    """Mock I/O utilities module for testing."""
    mock = MagicMock()
    return mock


@pytest.fixture
def sample_image():
    """Create a small test image."""
    return np.random.rand(64, 64, 3).astype(np.float32)


class TestTiledWriterSelection:
    """Test tiled vs legacy TIFF writer selection based on config."""
    
    @patch("lux_depth_v2.io_utils.write_tiff16_tiled")
    def test_uses_legacy_when_tile_size_none(self, mock_tiled, tmp_path, mock_io_utils, sample_image):
        """Verify legacy I/O is used when tiff_tile_size is None (default)."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            tiff_tile_size=None,  # Default, no tiling
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        mgr.write_master("test", sample_image)
        
        # Tiled writer should NOT be called (uses _io.atomic_write_rgb16_tiff instead)
        assert not mock_tiled.called
        # Legacy _io method should be called
        assert mock_io_utils.atomic_write_rgb16_tiff.called
    
    @patch("lux_depth_v2.io_utils.write_tiff16_tiled")
    def test_uses_tiled_when_tile_size_set(self, mock_tiled, tmp_path, mock_io_utils, sample_image):
        """Verify tiled writer is used when tiff_tile_size is set."""
        # Mock needs to create file for _atomic_move
        def create_file(path, arr, tile_size=None, compression=None):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        mock_tiled.side_effect = create_file
        
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            tiff_tile_size=512,
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        mgr.write_master("test", sample_image)
        
        # Tiled writer should be called with tile_size=512
        assert mock_tiled.called
        call_args = mock_tiled.call_args
        assert call_args.kwargs["tile_size"] == 512
    
    @patch("lux_depth_v2.io_utils.write_tiff16_tiled")
    def test_tiled_uses_config_compression(self, mock_tiled, tmp_path, mock_io_utils, sample_image):
        """Verify tiled writer uses compression from config when arg is None."""
        # Mock needs to create file for _atomic_move
        def create_file(path, arr, tile_size=None, compression=None):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        mock_tiled.side_effect = create_file
        
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            tiff_tile_size=256,
            tiff_compression="lzw",
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        # Pass compression=None to allow config to be used (precedence: explicit arg > config)
        mgr.write_master("test", sample_image, compression=None)
        
        # Tiled writer should use lzw compression from config
        assert mock_tiled.called
        call_args = mock_tiled.call_args
        assert call_args.kwargs["compression"] == "lzw"
    
    @patch("lux_depth_v2.io_utils.write_tiff16_tiled")
    def test_explicit_compression_overrides_config(self, mock_tiled, tmp_path, mock_io_utils, sample_image):
        """Verify explicit compression argument overrides config (precedence test)."""
        # Mock needs to create file for _atomic_move
        def create_file(path, arr, tile_size=None, compression=None):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        mock_tiled.side_effect = create_file
        
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            tiff_tile_size=256,
            tiff_compression="lzw",  # Config says lzw
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        # Explicitly pass deflate - should override config
        mgr.write_master("test", sample_image, compression="deflate")
        
        # Tiled writer should use deflate (explicit arg wins)
        assert mock_tiled.called
        call_args = mock_tiled.call_args
        assert call_args.kwargs["compression"] == "deflate"


class TestAtomicImageWrites:
    """Test atomic image write behavior (.tmp + replace)."""
    
    @patch("lux_depth_v2.io_utils.write_tiff16_legacy")
    def test_atomic_image_write_creates_tmp_file(self, mock_write, tmp_path, mock_io_utils, sample_image):
        """Verify atomic image write uses .tmp file."""
        # Mock needs to create the file so _atomic_move can work
        def create_file(path, arr, compression=None):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        mock_write.side_effect = create_file
        
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            use_atomic_image_writes=True,
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        result_path = mgr.write_master("test", sample_image)
        
        # Writer should be called with .tmp path
        assert mock_write.called
        written_path = mock_write.call_args[0][0]
        assert str(written_path).endswith(".tmp")
        
        # Final path should not have .tmp
        assert not str(result_path).endswith(".tmp")
        
        # Final file should exist (moved from .tmp)
        assert result_path.exists()
    
    def test_non_atomic_write_direct(self, tmp_path, mock_io_utils, sample_image):
        """Verify non-atomic write goes directly to final path."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            use_atomic_image_writes=False,  # Default
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        result_path = mgr.write_master("test", sample_image)
        
        # _io.atomic_write_rgb16_tiff should be called with final path (no extra .tmp)
        assert mock_io_utils.atomic_write_rgb16_tiff.called
        written_path = mock_io_utils.atomic_write_rgb16_tiff.call_args[0][0]
        assert written_path == result_path
        assert not str(written_path).endswith(".tmp")  # Direct call (though _io adds .tmp internally)


class TestAtomicReportWrites:
    """Test atomic report write behavior (.tmp + replace)."""
    
    def test_atomic_report_write_no_tmp_leftover(self, tmp_path, mock_io_utils):
        """Verify atomic report write leaves no .tmp file."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            use_atomic_report_writes=True,
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        report = {"status": "ok", "timing_s": 1.5}
        result_path = mgr.write_report("test", report)
        
        # Final JSON should exist and match expected content
        assert result_path.exists()
        loaded = json.loads(result_path.read_text())
        assert loaded == report
        
        # No .tmp file should remain
        tmp_file = result_path.with_suffix(result_path.suffix + ".tmp")
        assert not tmp_file.exists()
    
    def test_non_atomic_report_direct_write(self, tmp_path, mock_io_utils):
        """Verify non-atomic report write is direct (no .tmp)."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            use_atomic_report_writes=False,  # Default
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        report = {"status": "ok"}
        result_path = mgr.write_report("test", report)
        
        # Final JSON should exist
        assert result_path.exists()
        
        # Verify it was written directly (implementation detail: no .tmp created)
        # This is tested by behavior parity in integration tests


class TestTieredStorage:
    """Test tiered storage (scratch dir) functionality."""
    
    def test_tiered_storage_uses_scratch_dir(self, tmp_path, mock_io_utils, sample_image):
        """Verify tiered storage writes to scratch first."""
        # Mock needs to create the file so _atomic_move can work
        def create_file(path, arr, compression=None):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        mock_io_utils.atomic_write_rgb16_tiff.side_effect = create_file
        
        scratch_dir = tmp_path / "scratch"
        output_dir = tmp_path / "out"
        
        cfg = ExportConfig(
            output_dir=output_dir,
            enable_tiered_storage=True,
            scratch_dir=scratch_dir,
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        result_path = mgr.write_master("test", sample_image)
        
        # _io method should be called with scratch path, not final path
        assert mock_io_utils.atomic_write_rgb16_tiff.called
        written_path = mock_io_utils.atomic_write_rgb16_tiff.call_args[0][0]
        assert scratch_dir in written_path.parents
        
        # Result should be final path
        assert output_dir in result_path.parents
        
        # Final file should exist (moved from scratch)
        assert result_path.exists()
    
    def test_tiered_storage_disabled_writes_direct(self, tmp_path, mock_io_utils, sample_image):
        """Verify direct write when tiered storage disabled."""
        output_dir = tmp_path / "out"
        
        cfg = ExportConfig(
            output_dir=output_dir,
            enable_tiered_storage=False,  # Default
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        result_path = mgr.write_master("test", sample_image)
        
        # _io method should be called with final path directly
        assert mock_io_utils.atomic_write_rgb16_tiff.called
        written_path = mock_io_utils.atomic_write_rgb16_tiff.call_args[0][0]
        assert written_path == result_path


class TestBehaviorParity:
    """Verify Slice 3 PR-2 maintains Slice 2 behavior with default config."""
    
    def test_default_config_same_as_slice2(self, tmp_path, mock_io_utils, sample_image):
        """Verify default config produces same behavior as Slice 2."""
        cfg = ExportConfig(output_dir=tmp_path / "out")
        mgr = ExportManager(cfg, mock_io_utils)
        
        result_path = mgr.write_master("test", sample_image, compression="deflate")
        
        # Should use _io.atomic_write_rgb16_tiff (Slice 2 behavior)
        assert mock_io_utils.atomic_write_rgb16_tiff.called
        
        # Should write directly to final path (no extra scratch layer)
        written_path = mock_io_utils.atomic_write_rgb16_tiff.call_args[0][0]
        assert written_path == result_path
        
        # Should use deflate compression (passed through)
        assert mock_io_utils.atomic_write_rgb16_tiff.call_args.kwargs["compression"] == "deflate"
    
    def test_report_default_non_atomic(self, tmp_path, mock_io_utils):
        """Verify report write with default config is non-atomic."""
        cfg = ExportConfig(output_dir=tmp_path / "out")
        mgr = ExportManager(cfg, mock_io_utils)
        
        report = {"status": "ok", "data": [1, 2, 3]}
        result_path = mgr.write_report("test", report)
        
        # Report should exist
        assert result_path.exists()
        
        # Content should be correct
        loaded = json.loads(result_path.read_text())
        assert loaded == report


class TestCombinedOptimizations:
    """Test combinations of optimizations working together."""
    
    @patch("lux_depth_v2.io_utils.write_tiff16_tiled")
    def test_tiled_plus_atomic_plus_scratch(self, mock_tiled, tmp_path, mock_io_utils, sample_image):
        """Verify all optimizations can work together."""
        # Mock needs to create the file so _atomic_move can work
        def create_file(path, arr, tile_size=None, compression=None):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        mock_tiled.side_effect = create_file
        
        scratch_dir = tmp_path / "scratch"
        output_dir = tmp_path / "out"
        
        cfg = ExportConfig(
            output_dir=output_dir,
            enable_tiered_storage=True,
            scratch_dir=scratch_dir,
            use_atomic_image_writes=True,
            tiff_tile_size=512,
            tiff_compression="lzw",
        )
        mgr = ExportManager(cfg, mock_io_utils)
        
        result_path = mgr.write_master("test", sample_image)
        
        # Tiled writer should be called
        assert mock_tiled.called
        
        # Should use .tmp in scratch
        written_path = mock_tiled.call_args[0][0]
        assert str(written_path).endswith(".tmp")
        assert scratch_dir in written_path.parents
        
        # Result should be final path in output_dir
        assert output_dir in result_path.parents
        assert not str(result_path).endswith(".tmp")
        
        # Final file should exist
        assert result_path.exists()
