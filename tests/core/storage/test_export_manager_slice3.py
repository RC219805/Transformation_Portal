"""
Slice 3 PR-1: Config validation tests for ExportManager.

Tests focus on configuration validation added in Slice 3 PR-1:
- Scratch directory requirements
- TIFF tile size bounds
- Async worker validation

No behavior changes - only testing that config validation works correctly.
"""
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from transformation_portal.core.storage import ExportConfig, ExportManager


@pytest.fixture
def mock_io_utils():
    """Mock I/O utilities module for testing."""
    mock = MagicMock()
    return mock


class TestExportConfigValidation:
    """Test Slice 3 PR-1 config validation logic."""
    
    def test_tiered_storage_requires_scratch_when_flagged(self, tmp_path, mock_io_utils):
        """Verify config validation when scratch is required but missing."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            enable_tiered_storage=True,
            scratch_dir=None,
            require_scratch_on_enable=True,
        )
        
        with pytest.raises(ValueError, match="requires scratch_dir"):
            ExportManager(cfg, mock_io_utils)
    
    def test_tiered_storage_allows_missing_scratch_when_not_required(self, tmp_path, mock_io_utils):
        """Verify tiered storage works when scratch is optional."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            enable_tiered_storage=True,
            scratch_dir=None,
            require_scratch_on_enable=False,
        )
        
        # Should not raise
        mgr = ExportManager(cfg, mock_io_utils)
        assert mgr.config.enable_tiered_storage is True
        assert mgr.config.scratch_dir is None
    
    @pytest.mark.parametrize("tile_size", [64, 127, 1025, 2048])
    def test_tiff_tile_size_out_of_bounds_raises(self, tmp_path, mock_io_utils, tile_size):
        """Verify tile size validation rejects values outside bounds."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            tiff_tile_size=tile_size,
            tiff_tile_size_min=128,
            tiff_tile_size_max=1024,
        )
        
        with pytest.raises(ValueError, match="must be between 128 and 1024"):
            ExportManager(cfg, mock_io_utils)
    
    @pytest.mark.parametrize("tile_size", [128, 256, 512, 1024])
    def test_tiff_tile_size_within_bounds_ok(self, tmp_path, mock_io_utils, tile_size):
        """Verify tile size validation accepts valid values."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            tiff_tile_size=tile_size,
            tiff_tile_size_min=128,
            tiff_tile_size_max=1024,
        )
        
        # Should not raise
        mgr = ExportManager(cfg, mock_io_utils)
        assert mgr.config.tiff_tile_size == tile_size
    
    def test_tiff_tile_size_none_bypasses_validation(self, tmp_path, mock_io_utils):
        """Verify None tile size (default) skips validation."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            tiff_tile_size=None,  # Default, no tiling
        )
        
        # Should not raise
        mgr = ExportManager(cfg, mock_io_utils)
        assert mgr.config.tiff_tile_size is None
    
    @pytest.mark.parametrize("workers", [0, -1, -10])
    def test_max_async_workers_must_be_positive(self, tmp_path, mock_io_utils, workers):
        """Verify async worker count validation."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            max_async_workers=workers,
        )
        
        with pytest.raises(ValueError, match="must be >= 1"):
            ExportManager(cfg, mock_io_utils)
    
    @pytest.mark.parametrize("workers", [1, 2, 4, 8])
    def test_max_async_workers_valid_values(self, tmp_path, mock_io_utils, workers):
        """Verify valid async worker counts are accepted."""
        cfg = ExportConfig(
            output_dir=tmp_path / "out",
            max_async_workers=workers,
        )
        
        # Should not raise
        mgr = ExportManager(cfg, mock_io_utils)
        assert mgr.config.max_async_workers == workers


class TestExportManagerSkeleton:
    """Test Slice 3 PR-1 skeleton helper methods."""
    
    def test_resolve_scratch_path_returns_final_path_in_pr1(self, tmp_path, mock_io_utils):
        """Verify _resolve_scratch_path returns final path (no optimization yet)."""
        cfg = ExportConfig(output_dir=tmp_path / "out")
        mgr = ExportManager(cfg, mock_io_utils)
        
        final_path = tmp_path / "out" / "test_master16.tif"
        resolved = mgr._resolve_scratch_path(final_path)
        
        # PR-1: Should return final path unchanged
        assert resolved == final_path
    
    def test_atomic_move_creates_parent_dirs(self, tmp_path, mock_io_utils):
        """Verify _atomic_move creates parent directories."""
        cfg = ExportConfig(output_dir=tmp_path / "out")
        mgr = ExportManager(cfg, mock_io_utils)
        
        src = tmp_path / "scratch" / "test.tif"
        src.parent.mkdir(parents=True, exist_ok=True)
        src.write_text("test content")
        
        dst = tmp_path / "out" / "subdir" / "test.tif"
        
        mgr._atomic_move(src, dst)
        
        assert dst.exists()
        assert dst.read_text() == "test content"
        assert not src.exists()
    
    def test_cleanup_scratch_is_noop_in_pr1(self, tmp_path, mock_io_utils):
        """Verify cleanup_scratch is a no-op in PR-1."""
        cfg = ExportConfig(output_dir=tmp_path / "out")
        mgr = ExportManager(cfg, mock_io_utils)
        
        # Should not raise
        mgr.cleanup_scratch()
    
    def test_close_is_safe_without_executor(self, tmp_path, mock_io_utils):
        """Verify close() is safe when executor not initialized."""
        cfg = ExportConfig(output_dir=tmp_path / "out")
        mgr = ExportManager(cfg, mock_io_utils)
        
        # Should not raise
        mgr.close()


class TestBackwardCompatibility:
    """Verify Slice 3 PR-1 maintains Slice 2 behavior with default config."""
    
    def test_default_config_creates_manager_without_errors(self, tmp_path, mock_io_utils):
        """Verify default ExportConfig (all optimizations OFF) works."""
        cfg = ExportConfig(output_dir=tmp_path / "out")
        
        # Should not raise with all defaults
        mgr = ExportManager(cfg, mock_io_utils)
        
        # Verify all optimization flags are OFF
        assert mgr.config.enable_tiered_storage is False
        assert mgr.config.scratch_dir is None
        assert mgr.config.tiff_tile_size is None
        assert mgr.config.tiff_compression is None
        assert mgr.config.use_atomic_image_writes is False
        assert mgr.config.use_atomic_report_writes is False
        assert mgr.config.async_flush is False
        assert mgr.config.max_async_workers == 2
    
    def test_output_dir_created_on_init(self, tmp_path, mock_io_utils):
        """Verify output_dir is created during initialization."""
        out_dir = tmp_path / "out" / "subdir"
        cfg = ExportConfig(output_dir=out_dir)
        
        assert not out_dir.exists()
        
        ExportManager(cfg, mock_io_utils)
        
        assert out_dir.exists()
        assert out_dir.is_dir()
