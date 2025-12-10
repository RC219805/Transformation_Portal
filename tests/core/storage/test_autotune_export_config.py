"""
Tests for autotune_export_config() adaptive configuration.

Validates benchmark-based heuristics for enabling/disabling Slice 3 optimizations.
"""
from pathlib import Path

import pytest

from transformation_portal.core.storage.export_manager import (
    ExportConfig,
    autotune_export_config,
)


class TestAutotuneExportConfig:
    """Test adaptive export configuration based on image characteristics."""
    
    def test_baseline_when_adaptive_disabled(self, tmp_path: Path):
        """When enable_adaptive=False, always return baseline config."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=6000,
            image_height=3600,
            scene_complexity=0.2,  # Low complexity (would enable optimizations)
            enable_adaptive=False,
        )
        
        assert cfg.output_dir == tmp_path
        assert cfg.tiff_tile_size is None
        assert cfg.tiff_compression is None
        assert cfg.use_atomic_image_writes is False
        assert cfg.use_atomic_report_writes is False
    
    def test_aerial_like_scene_enables_optimizations(self, tmp_path: Path):
        """Aerial-like scene (large, low complexity) enables tiled_atomic."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=6000,
            image_height=3600,
            scene_complexity=0.3,  # Low complexity (sky/terrain)
        )
        
        # Should enable tiled_atomic mode
        assert cfg.tiff_tile_size == 512
        assert cfg.tiff_compression is None  # LZW disabled (zero benefit)
        assert cfg.use_atomic_image_writes is True
        assert cfg.use_atomic_report_writes is True
    
    def test_interior_scene_disables_optimizations(self, tmp_path: Path):
        """Interior scene (high complexity) uses baseline config."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=4000,
            image_height=3000,
            scene_complexity=0.8,  # High complexity (textures/details)
        )
        
        # Should use baseline mode
        assert cfg.tiff_tile_size is None
        assert cfg.use_atomic_image_writes is False
        assert cfg.use_atomic_report_writes is False
    
    def test_large_image_unknown_complexity_enables_conservatively(self, tmp_path: Path):
        """Very large image with unknown complexity enables optimizations."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=8000,
            image_height=6000,  # 48 MP > 40 MP threshold
            scene_complexity=None,  # Unknown complexity
        )
        
        # Should enable conservatively for very large images
        assert cfg.tiff_tile_size == 512
        assert cfg.use_atomic_image_writes is True
    
    def test_small_image_unknown_complexity_uses_baseline(self, tmp_path: Path):
        """Small image with unknown complexity uses baseline."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=2000,
            image_height=1500,  # 3 MP < 20 MP threshold
            scene_complexity=None,  # Unknown complexity
        )
        
        # Should use baseline (too small)
        assert cfg.tiff_tile_size is None
        assert cfg.use_atomic_image_writes is False
    
    def test_medium_complexity_large_image_uses_baseline(self, tmp_path: Path):
        """Medium complexity + large image still uses baseline (conservative)."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=6000,
            image_height=4000,  # 24 MP > 20 MP
            scene_complexity=0.5,  # Exactly at threshold (not below)
        )
        
        # Should use baseline (complexity not low enough)
        assert cfg.tiff_tile_size is None
        assert cfg.use_atomic_image_writes is False
    
    def test_boundary_case_just_below_complexity_threshold(self, tmp_path: Path):
        """Scene just below complexity threshold enables optimizations."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=6000,
            image_height=4000,  # 24 MP > 20 MP
            scene_complexity=0.49,  # Just below 0.5 threshold
        )
        
        # Should enable (barely below threshold)
        assert cfg.tiff_tile_size == 512
        assert cfg.use_atomic_image_writes is True
    
    def test_boundary_case_just_above_megapixel_threshold(self, tmp_path: Path):
        """Image just above megapixel threshold enables optimizations."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=4500,
            image_height=4500,  # 20.25 MP > 20 MP
            scene_complexity=0.3,
        )
        
        # Should enable (just above megapixel threshold)
        assert cfg.tiff_tile_size == 512
        assert cfg.use_atomic_image_writes is True
    
    def test_zero_dimensions_uses_baseline(self, tmp_path: Path):
        """Unknown dimensions (0x0) uses baseline config."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=0,
            image_height=0,
            scene_complexity=0.1,  # Even with low complexity
        )
        
        # Should use baseline (can't determine megapixels)
        assert cfg.tiff_tile_size is None
        assert cfg.use_atomic_image_writes is False
    
    def test_lzw_compression_always_disabled(self, tmp_path: Path):
        """LZW compression should always be disabled (benchmark finding)."""
        # Test with optimizations enabled
        cfg1 = autotune_export_config(
            output_dir=tmp_path,
            image_width=6000,
            image_height=3600,
            scene_complexity=0.2,
        )
        assert cfg1.tiff_compression is None  # Not "lzw"
        
        # Test with optimizations disabled
        cfg2 = autotune_export_config(
            output_dir=tmp_path,
            image_width=2000,
            image_height=1500,
            scene_complexity=0.9,
        )
        assert cfg2.tiff_compression is None
    
    def test_tiered_storage_disabled_by_default(self, tmp_path: Path):
        """Tiered storage should not be enabled by autotune (requires explicit scratch_dir)."""
        cfg = autotune_export_config(
            output_dir=tmp_path,
            image_width=6000,
            image_height=3600,
            scene_complexity=0.2,
        )
        
        assert cfg.enable_tiered_storage is False
        assert cfg.scratch_dir is None
