"""Unit tests for material_profiles module."""
from __future__ import annotations

import pytest

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not available")

from lux_depth_v2 import material_profiles
from lux_depth_v2.config import PipelineConfig


class TestSurfaceProfile:
    """Test SurfaceProfile dataclass."""

    def test_default_profile(self):
        """Test default surface profile values."""
        prof = material_profiles.SurfaceProfile()
        assert prof.temp_offset == 0.0
        assert prof.sat_mult == 1.0
        assert prof.exp_mult == 1.0
        assert prof.con_mult == 1.0
        assert prof.detail_mult == 1.0
        assert prof.clarity_mult == 1.0
        assert prof.sharpen_mult == 1.0
        assert prof.highlight_compress == 0.0

    def test_custom_profile(self):
        """Test custom surface profile."""
        prof = material_profiles.SurfaceProfile(
            temp_offset=0.01,
            sat_mult=1.1,
            detail_mult=1.2,
        )
        assert prof.temp_offset == 0.01
        assert prof.sat_mult == 1.1
        assert prof.detail_mult == 1.2


class TestSurfaceProfiles:
    """Test predefined surface profiles."""

    def test_all_surfaces_exist(self):
        """Test all expected surface profiles are defined."""
        expected = ["wood", "metal", "glass", "stone", "sky", "foliage"]
        for surf in expected:
            assert surf in material_profiles.SURFACE_PROFILES

    def test_wood_profile(self):
        """Test wood surface profile characteristics."""
        prof = material_profiles.SURFACE_PROFILES["wood"]
        assert prof.temp_offset > 0  # Warmer
        assert prof.sat_mult > 1.0  # More saturated
        assert prof.detail_mult > 1.0  # More detail

    def test_metal_profile(self):
        """Test metal surface profile characteristics."""
        prof = material_profiles.SURFACE_PROFILES["metal"]
        assert prof.temp_offset < 0  # Cooler
        assert prof.sat_mult < 1.0  # Less saturated
        assert prof.sharpen_mult > 1.0  # More sharpening

    def test_glass_profile(self):
        """Test glass surface profile characteristics."""
        prof = material_profiles.SURFACE_PROFILES["glass"]
        assert prof.sat_mult < 1.0  # Less saturated
        assert prof.detail_mult < 1.0  # Less detail (smooth)
        assert prof.clarity_mult < 1.0  # Less clarity
        assert prof.highlight_compress > 0  # Compress highlights

    def test_sky_profile(self):
        """Test sky surface profile characteristics."""
        prof = material_profiles.SURFACE_PROFILES["sky"]
        assert prof.temp_offset < 0  # Cooler (blue)
        assert prof.sat_mult > 1.0  # More saturated
        assert prof.detail_mult < 1.0  # Less detail (smooth)
        assert prof.highlight_compress > 0  # Compress highlights

    def test_profile_reasonableness(self):
        """Test all profiles have reasonable value ranges."""
        for name, prof in material_profiles.SURFACE_PROFILES.items():
            assert -0.02 <= prof.temp_offset <= 0.02
            assert 0.9 <= prof.sat_mult <= 1.1
            assert 0.9 <= prof.exp_mult <= 1.1
            assert 0.9 <= prof.con_mult <= 1.1
            assert 0.7 <= prof.detail_mult <= 1.2
            assert 0.6 <= prof.clarity_mult <= 1.2
            assert 0.6 <= prof.sharpen_mult <= 1.2
            assert 0.0 <= prof.highlight_compress <= 1.0


class TestMaterialMods:
    """Test MaterialMods dataclass."""

    def test_material_mods_creation(self, torch_device):
        """Test MaterialMods creation."""
        mods = material_profiles.MaterialMods(
            temp_offset=torch.zeros((1, 1, 32, 32), device=torch_device),
            sat_mult=torch.ones((1, 1, 32, 32), device=torch_device),
            exp_mult=torch.ones((1, 1, 32, 32), device=torch_device),
            con_mult=torch.ones((1, 1, 32, 32), device=torch_device),
            detail_mult=torch.ones((1, 1, 32, 32), device=torch_device),
            clarity_mult=torch.ones((1, 1, 32, 32), device=torch_device),
            sharpen_mult=torch.ones((1, 1, 32, 32), device=torch_device),
            highlight_compress=torch.zeros((1, 1, 32, 32), device=torch_device),
            source="test",
        )
        assert mods.source == "test"
        assert mods.temp_offset.shape == (1, 1, 32, 32)


class TestBuildMaterialMods:
    """Test material mods building from masks."""

    def test_build_with_no_masks(self, torch_device):
        """Test returns None when no masks provided."""
        cfg = PipelineConfig(enable_material=True)
        mods = material_profiles.build_material_mods({}, cfg)
        assert mods is None

    def test_build_with_material_disabled(self, torch_device):
        """Test returns None when material processing disabled."""
        cfg = PipelineConfig(enable_material=False)
        mask = torch.ones((1, 1, 32, 32), device=torch_device)
        mods = material_profiles.build_material_mods({"wood": mask}, cfg)
        assert mods is None

    def test_build_with_single_surface(self, torch_device):
        """Test building mods with single surface mask."""
        cfg = PipelineConfig(enable_material=True, material_strength=0.8)
        mask = torch.ones((1, 1, 32, 32), device=torch_device) * 0.5

        mods = material_profiles.build_material_mods({"wood": mask}, cfg)

        assert mods is not None
        assert mods.source == "material_segmentation"
        assert mods.temp_offset.shape == (1, 1, 32, 32)

        # Check wood profile is applied
        wood_prof = material_profiles.SURFACE_PROFILES["wood"]
        # Temperature should be influenced by wood profile
        temp_mean = mods.temp_offset.mean().item()
        assert temp_mean > 0  # Wood has positive temp offset

    def test_build_with_multiple_surfaces(self, torch_device):
        """Test building mods with multiple surface masks."""
        cfg = PipelineConfig(enable_material=True, material_strength=1.0)

        masks = {
            "wood": torch.ones((1, 1, 32, 32), device=torch_device) * 0.5,
            "metal": torch.ones((1, 1, 32, 32), device=torch_device) * 0.3,
            "glass": torch.ones((1, 1, 32, 32), device=torch_device) * 0.2,
        }

        mods = material_profiles.build_material_mods(masks, cfg)

        assert mods is not None
        # Mods should be combination of all surface profiles
        assert mods.temp_offset.shape == (1, 1, 32, 32)

    def test_material_strength_scaling(self, torch_device):
        """Test material strength scales effect."""
        mask = torch.ones((1, 1, 32, 32), device=torch_device)

        cfg_low = PipelineConfig(enable_material=True, material_strength=0.3)
        mods_low = material_profiles.build_material_mods({"wood": mask}, cfg_low)

        cfg_high = PipelineConfig(enable_material=True, material_strength=0.9)
        mods_high = material_profiles.build_material_mods({"wood": mask}, cfg_high)

        # Higher strength should produce stronger effect
        temp_low = abs(mods_low.temp_offset.mean().item())
        temp_high = abs(mods_high.temp_offset.mean().item())
        assert temp_high > temp_low

    def test_unknown_surface_ignored(self, torch_device):
        """Test unknown surface names are ignored."""
        cfg = PipelineConfig(enable_material=True)
        masks = {
            "unknown_material": torch.ones((1, 1, 32, 32), device=torch_device),
        }

        mods = material_profiles.build_material_mods(masks, cfg)

        # Should still create mods but with neutral values
        assert mods is not None
        # Since no known surfaces, should be close to identity
        assert torch.allclose(mods.sat_mult, torch.ones_like(mods.sat_mult), atol=0.1)

    def test_safety_clamping(self, torch_device):
        """Test safety clamping of mod values."""
        cfg = PipelineConfig(enable_material=True, material_strength=5.0)  # Extreme
        mask = torch.ones((1, 1, 32, 32), device=torch_device)

        mods = material_profiles.build_material_mods({"wood": mask}, cfg)

        # Check all values are within safe ranges
        assert torch.all(mods.sat_mult >= 0.80)
        assert torch.all(mods.sat_mult <= 1.35)
        assert torch.all(mods.exp_mult >= 0.90)
        assert torch.all(mods.exp_mult <= 1.10)
        assert torch.all(mods.con_mult >= 0.90)
        assert torch.all(mods.con_mult <= 1.15)
        assert torch.all(mods.temp_offset >= -0.05)
        assert torch.all(mods.temp_offset <= 0.05)
        assert torch.all(mods.highlight_compress >= 0.0)
        assert torch.all(mods.highlight_compress <= 1.0)

    def test_zero_mask_has_no_effect(self, torch_device):
        """Test zero mask produces no modification."""
        cfg = PipelineConfig(enable_material=True, material_strength=1.0)
        mask = torch.zeros((1, 1, 32, 32), device=torch_device)

        mods = material_profiles.build_material_mods({"wood": mask}, cfg)

        # Should still create mods but with identity values
        assert mods is not None
        # No mask means no modification
        assert torch.allclose(mods.temp_offset, torch.zeros_like(mods.temp_offset), atol=1e-3)
