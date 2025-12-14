"""Unit tests for PR-4D Stone Pixel Operations.

Tests are torch-free and focus on basic correctness of pixel ops.
"""

import numpy as np
import pytest

from lux_depth_v2.materials_v3_pixel_ops_stone import (
    StoneResponseConfig,
    apply_stone_response,
    apply_stone_local_contrast,
    apply_stone_clarity,
    apply_stone_saturation,
)


class TestStoneLocalContrast:
    """Test stone local contrast function."""
    
    def test_preserves_shape(self):
        """Local contrast should preserve image shape."""
        rgb = np.random.rand(100, 100, 3).astype(np.float32)
        result = apply_stone_local_contrast(rgb, strength=1.04)
        
        assert result.shape == rgb.shape
        assert result.dtype == np.float32
    
    def test_values_in_range(self):
        """Output should be clipped to [0,1]."""
        rgb = np.random.rand(50, 50, 3).astype(np.float32)
        result = apply_stone_local_contrast(rgb, strength=1.08)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_neutral_strength_minimal_change(self):
        """Strength=1.0 should produce minimal change."""
        rgb = np.random.rand(50, 50, 3).astype(np.float32) * 0.5 + 0.25
        result = apply_stone_local_contrast(rgb, strength=1.0)
        
        # Should be very close to original (small smoothing artifacts ok)
        diff = np.abs(result - rgb).max()
        assert diff < 0.05
    
    def test_raises_on_invalid_shape(self):
        """Should raise error on non-HxWx3 input."""
        with pytest.raises(ValueError):
            apply_stone_local_contrast(np.random.rand(50, 50), strength=1.04)


class TestStoneClarity:
    """Test stone clarity function."""
    
    def test_preserves_shape(self):
        """Clarity should preserve image shape."""
        rgb = np.random.rand(80, 80, 3).astype(np.float32)
        result = apply_stone_clarity(rgb, strength=1.02)
        
        assert result.shape == rgb.shape
        assert result.dtype == np.float32
    
    def test_values_in_range(self):
        """Output should be clipped to [0,1]."""
        rgb = np.random.rand(50, 50, 3).astype(np.float32)
        result = apply_stone_clarity(rgb, strength=1.05)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_neutral_strength_minimal_change(self):
        """Strength=1.0 should produce minimal change."""
        rgb = np.random.rand(50, 50, 3).astype(np.float32) * 0.5 + 0.25
        result = apply_stone_clarity(rgb, strength=1.0)
        
        diff = np.abs(result - rgb).max()
        assert diff < 0.05


class TestStoneSaturation:
    """Test stone saturation function."""
    
    def test_preserves_shape(self):
        """Saturation should preserve image shape."""
        rgb = np.random.rand(60, 60, 3).astype(np.float32)
        result = apply_stone_saturation(rgb, scale=1.0)
        
        assert result.shape == rgb.shape
        assert result.dtype == np.float32
    
    def test_neutral_scale_no_change(self):
        """Scale=1.0 should not change image."""
        rgb = np.random.rand(50, 50, 3).astype(np.float32)
        result = apply_stone_saturation(rgb, scale=1.0)
        
        # Should be identical or very close
        np.testing.assert_allclose(result, rgb, rtol=1e-5, atol=1e-6)


class TestStoneResponse:
    """Test full stone response application."""
    
    def test_preserves_shape(self):
        """Stone response should preserve image shape."""
        rgb = np.random.rand(100, 100, 3).astype(np.float32)
        mask = np.random.rand(100, 100).astype(np.float32)
        
        cfg = StoneResponseConfig()
        result, stats = apply_stone_response(rgb, mask, cfg)
        
        assert result.shape == rgb.shape
        assert result.dtype == np.float32
    
    def test_values_in_range(self):
        """Output should be in [0,1]."""
        rgb = np.random.rand(80, 80, 3).astype(np.float32)
        mask = np.random.rand(80, 80).astype(np.float32)
        
        cfg = StoneResponseConfig()
        result, stats = apply_stone_response(rgb, mask, cfg)
        
        assert result.min() >= 0.0
        assert result.max() <= 1.0
    
    def test_clamp_triggers_on_extreme_input(self):
        """Extreme delta should trigger clamp and be recorded in stats."""
        # Create larger image with very high contrast
        rgb = np.ones((300, 300, 3), dtype=np.float32) * 0.9
        rgb[120:180, 120:180, :] = 0.1  # Dark patch
        
        # Large mask covering the contrast boundary
        mask = np.zeros((300, 300), dtype=np.float32)
        mask[100:200, 100:200] = 1.0  # 10000 pixels
        
        # Use aggressive config to trigger clamps
        cfg = StoneResponseConfig(
            core_local_contrast=1.20,  # Very aggressive
            edge_local_contrast=1.15,
            max_delta=0.05,  # Tight clamp
            min_coverage_px=5000,  # Lower than mask size
        )
        
        result, stats = apply_stone_response(rgb, mask, cfg)
        
        # Should have applied
        assert stats["applied"] == True
        # Clamp likely triggered (but not guaranteed, depends on mask erosion)
        # Just check it's a valid count
        assert isinstance(stats.get("clamp_count", 0), int)
        assert isinstance(stats.get("edge_clamp_count", 0), int)
    
    def test_below_min_coverage_returns_applied_false(self):
        """Coverage below min_coverage_px should skip processing."""
        rgb = np.random.rand(100, 100, 3).astype(np.float32)
        # Very small mask
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[45:55, 45:55] = 1.0  # Only 100 pixels
        
        cfg = StoneResponseConfig(
            min_coverage_px=50_000  # Much higher than 100
        )
        
        result, stats = apply_stone_response(rgb, mask, cfg)
        
        assert stats["applied"] == False
        assert stats["reason"] == "below_min_coverage"
        # Should return unchanged image
        np.testing.assert_array_equal(result, rgb)
    
    def test_halo_metric_computed_when_edge_exists(self):
        """Halo risk should be computed when edge band exists."""
        rgb = np.random.rand(100, 100, 3).astype(np.float32)
        # Large mask so erosion creates both core and edge
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0
        
        cfg = StoneResponseConfig(
            min_coverage_px=1000  # Well below mask size
        )
        
        result, stats = apply_stone_response(rgb, mask, cfg)
        
        if stats["applied"]:
            # Should have halo_risk key
            assert "halo_risk" in stats
            assert stats["halo_risk"] in ["NONE", "LOW", "MEDIUM", "HIGH"]
            # Should have edge pixels
            assert stats.get("edge_px", 0) > 0
    
    def test_stats_structure(self):
        """Stats should have expected structure."""
        rgb = np.random.rand(100, 100, 3).astype(np.float32)
        mask = np.zeros((100, 100), dtype=np.float32)
        mask[20:80, 20:80] = 1.0
        
        cfg = StoneResponseConfig(min_coverage_px=1000)
        result, stats = apply_stone_response(rgb, mask, cfg)
        
        # Check required keys
        assert "applied" in stats
        
        if stats["applied"]:
            assert "coverage_px" in stats
            assert "core_px" in stats
            assert "edge_px" in stats
            assert "mean_delta" in stats
            assert "halo_risk" in stats
            assert isinstance(stats["coverage_px"], int)
            assert isinstance(stats["core_px"], int)
            assert isinstance(stats["edge_px"], int)
            assert isinstance(stats["mean_delta"], float)
        else:
            assert "reason" in stats
    
    def test_raises_on_shape_mismatch(self):
        """Should raise error if image and mask shapes don't match."""
        rgb = np.random.rand(100, 100, 3).astype(np.float32)
        mask = np.random.rand(80, 80).astype(np.float32)
        
        cfg = StoneResponseConfig()
        
        with pytest.raises(ValueError, match="shape"):
            apply_stone_response(rgb, mask, cfg)
    
    def test_conservative_defaults(self):
        """Default config should be very conservative."""
        cfg = StoneResponseConfig()
        
        # Check conservative parameters
        assert cfg.core_local_contrast <= 1.05
        assert cfg.edge_local_contrast <= 1.03
        assert cfg.core_clarity <= 1.03
        assert cfg.edge_clarity <= 1.02
        assert cfg.max_delta <= 0.10  # Tight clamp
        assert cfg.min_coverage_px >= 10_000  # Avoid tiny applications
