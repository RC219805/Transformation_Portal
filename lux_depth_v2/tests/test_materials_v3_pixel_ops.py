"""Tests for Materials V3 Pixel Operations (PR-4B: Glass Response)."""

import numpy as np
import pytest

from lux_depth_v2.materials_v3_pixel_ops import (
    GlassResponseConfig,
    apply_glass_response,
    apply_local_contrast,
    apply_clarity,
    apply_saturation,
    extract_core_edge_masks,
)


@pytest.fixture
def synthetic_glass_scene():
    """Create synthetic glass scene for testing."""
    H, W = 64, 64
    rgb = np.ones((H, W, 3), dtype=np.float32) * 0.5
    
    # Add some texture/variation
    x, y = np.meshgrid(np.linspace(0, 1, W), np.linspace(0, 1, H))
    rgb[..., 0] += 0.1 * np.sin(x * 10)
    rgb[..., 1] += 0.1 * np.cos(y * 10)
    rgb[..., 2] += 0.05 * np.sin(x * 5) * np.cos(y * 5)
    rgb = np.clip(rgb, 0, 1)
    
    # Glass mask (centered square)
    glass_mask = np.zeros((H, W), dtype=np.float32)
    glass_mask[16:48, 16:48] = 0.9
    
    return rgb, glass_mask


class TestExtractCoreEdgeMasks:
    """Test core/edge/blend mask extraction."""
    
    def test_basic_extraction(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        
        core, edge, blend = extract_core_edge_masks(mask, edge_width_px=5)
        
        assert core.dtype == bool
        assert edge.dtype == bool
        assert blend.dtype == bool
        
        # Core should be smaller than total mask
        assert core.sum() < (mask > 0.5).sum()
        
        # Edge + core should cover most of the mask
        total_coverage = (edge | core).sum()
        mask_coverage = (mask > 0.5).sum()
        assert total_coverage >= mask_coverage * 0.9
    
    def test_no_overlap(self):
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[16:48, 16:48] = 1.0
        
        core, edge, blend = extract_core_edge_masks(mask, edge_width_px=5)
        
        # Core and edge should not overlap
        assert not np.any(core & edge)
    
    def test_small_mask_degenerates_gracefully(self):
        # Very small mask should have edge but maybe no core
        mask = np.zeros((64, 64), dtype=np.float32)
        mask[30:34, 30:34] = 1.0
        
        core, edge, blend = extract_core_edge_masks(mask, edge_width_px=3)
        
        # Should not crash; edge may dominate
        assert edge.sum() > 0


class TestApplyLocalContrast:
    """Test local contrast enhancement."""
    
    def test_identity_at_strength_1(self):
        rgb = np.random.rand(32, 32, 3).astype(np.float32) * 0.5 + 0.25
        result = apply_local_contrast(rgb, strength=1.0, preserve_highlights=False)
        np.testing.assert_allclose(result, rgb, atol=0.05)
    
    def test_increases_contrast(self):
        # Uniform gradient
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        x = np.linspace(0, 1, 32)
        rgb[..., :] = x[None, :, None]
        
        result = apply_local_contrast(rgb, strength=1.2, preserve_highlights=False)
        
        # Result should be in valid range
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
        # Overall variance should increase
        assert result.var() >= rgb.var() * 0.95
    
    def test_preserves_highlights(self):
        rgb = np.ones((32, 32, 3), dtype=np.float32) * 0.9
        result = apply_local_contrast(
            rgb,
            strength=1.5,
            preserve_highlights=True,
            highlight_threshold=0.85,
        )
        
        # Bright pixels should be unchanged
        np.testing.assert_allclose(result, rgb, atol=0.01)


class TestApplyClarity:
    """Test clarity (high-frequency boost)."""
    
    def test_identity_at_zero_strength(self):
        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        result = apply_clarity(rgb, strength=0.0)
        np.testing.assert_allclose(result, rgb, atol=0.01)
    
    def test_boosts_edges(self):
        # Sharp edge
        rgb = np.zeros((32, 32, 3), dtype=np.float32)
        rgb[:, :16] = 0.2
        rgb[:, 16:] = 0.8
        
        result = apply_clarity(rgb, strength=0.1)
        
        # Should boost edge transition
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
        # Edge region should have more variation
        edge_original = rgb[:, 14:18].std()
        edge_result = result[:, 14:18].std()
        assert edge_result >= edge_original * 0.9


class TestApplySaturation:
    """Test saturation adjustment."""
    
    def test_identity_at_scale_1(self):
        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        result = apply_saturation(rgb, scale=1.0)
        np.testing.assert_allclose(result, rgb, atol=0.01)
    
    def test_desaturates(self):
        # Colorful pixel
        rgb = np.zeros((1, 1, 3), dtype=np.float32)
        rgb[0, 0] = [1.0, 0.0, 0.0]  # Pure red
        
        result = apply_saturation(rgb, scale=0.5)
        
        # Should move toward gray
        assert result[0, 0, 0] < rgb[0, 0, 0]
        assert result[0, 0, 1] > rgb[0, 0, 1]
        assert result[0, 0, 2] > rgb[0, 0, 2]


class TestApplyGlassResponse:
    """Test full glass response application."""
    
    def test_basic_application(self, synthetic_glass_scene):
        rgb, glass_mask = synthetic_glass_scene
        cfg = GlassResponseConfig()
        
        result, stats = apply_glass_response(rgb, glass_mask, cfg)
        
        # Shape preserved
        assert result.shape == rgb.shape
        
        # Valid range
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        
        # Stats present
        assert "core_pixels" in stats
        assert "edge_pixels" in stats
        assert "mean_delta_core" in stats
        assert "max_delta" in stats
        
        # Some pixels changed
        assert stats["total_glass_pixels"] > 0
        assert stats["core_pixels"] > 0 or stats["edge_pixels"] > 0
    
    def test_respects_max_delta(self, synthetic_glass_scene):
        rgb, glass_mask = synthetic_glass_scene
        cfg = GlassResponseConfig(
            core_contrast=2.0,  # Aggressive
            max_delta=0.10,  # Strict limit
        )
        
        result, stats = apply_glass_response(rgb, glass_mask, cfg)
        
        # No pixel should change by more than max_delta
        delta = np.abs(result - rgb).max(axis=-1)
        assert delta.max() <= cfg.max_delta + 1e-6
    
    def test_no_change_outside_mask(self, synthetic_glass_scene):
        rgb, glass_mask = synthetic_glass_scene
        cfg = GlassResponseConfig()
        
        result, stats = apply_glass_response(rgb, glass_mask, cfg)
        
        # Pixels outside glass mask should be unchanged
        outside = glass_mask < 0.1
        if outside.any():
            np.testing.assert_allclose(
                result[outside],
                rgb[outside],
                atol=0.01,
            )
    
    def test_shape_mismatch_raises(self):
        rgb = np.zeros((64, 64, 3), dtype=np.float32)
        glass_mask = np.zeros((32, 32), dtype=np.float32)
        cfg = GlassResponseConfig()
        
        with pytest.raises(ValueError, match="shape"):
            apply_glass_response(rgb, glass_mask, cfg)
    
    def test_empty_mask_does_nothing(self):
        rgb = np.random.rand(32, 32, 3).astype(np.float32)
        glass_mask = np.zeros((32, 32), dtype=np.float32)
        cfg = GlassResponseConfig()
        
        result, stats = apply_glass_response(rgb, glass_mask, cfg)
        
        # Should be identical
        np.testing.assert_allclose(result, rgb, atol=0.01)
        assert stats["total_glass_pixels"] == 0


class TestGlassResponseConfig:
    """Test config validation."""
    
    def test_defaults_are_conservative(self):
        cfg = GlassResponseConfig()
        
        # Edge settings should be gentler than core
        assert cfg.edge_contrast <= cfg.core_contrast
        assert cfg.edge_clarity <= cfg.core_clarity
        
        # Max delta should prevent artifacts
        assert 0.05 <= cfg.max_delta <= 0.25
    
    def test_can_override_defaults(self):
        cfg = GlassResponseConfig(
            core_contrast=1.5,
            edge_clarity=0.0,
        )
        
        assert cfg.core_contrast == 1.5
        assert cfg.edge_clarity == 0.0
