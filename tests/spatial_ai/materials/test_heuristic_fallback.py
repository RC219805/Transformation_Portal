"""Unit tests for heuristic fallback (Phase 2.2)."""

import numpy as np
import pytest

from transformation_portal.spatial_ai.materials.heuristic_fallback import HeuristicFallback


class TestHeuristicFallback:
    """Test heuristic PBR texture generation."""

    @pytest.fixture
    def generator(self):
        """Create heuristic generator."""
        return HeuristicFallback()

    @pytest.fixture
    def sample_rgb(self):
        """Create sample RGB image."""
        return np.random.rand(256, 256, 3).astype(np.float32)

    @pytest.fixture
    def sample_mask(self):
        """Create sample mask."""
        mask = np.zeros((256, 256), dtype=bool)
        mask[64:192, 64:192] = True  # Central region
        return mask

    @pytest.fixture
    def sample_depth(self):
        """Create sample depth map."""
        return np.random.rand(256, 256).astype(np.float32) * 10.0  # 0-10 meters

    def test_generate_basic(self, generator, sample_rgb):
        """Test basic PBR generation without mask or depth."""
        albedo, normal, roughness, metallic, ao, height = generator.generate_pbr_textures(
            rgb=sample_rgb,
        )

        # Check shapes
        assert albedo.shape == sample_rgb.shape
        assert normal.shape == sample_rgb.shape
        assert roughness.shape == sample_rgb.shape[:2]
        assert metallic.shape == sample_rgb.shape[:2]
        assert ao.shape == sample_rgb.shape[:2]
        assert height.shape == sample_rgb.shape[:2]

        # Check dtypes
        assert albedo.dtype == np.float32
        assert normal.dtype == np.float32
        assert roughness.dtype == np.float32
        assert metallic.dtype == np.float32
        assert ao.dtype == np.float32
        assert height.dtype == np.float32

    def test_generate_with_mask(self, generator, sample_rgb, sample_mask):
        """Test PBR generation with segmentation mask."""
        albedo, normal, roughness, metallic, ao, height = generator.generate_pbr_textures(
            rgb=sample_rgb,
            mask=sample_mask,
        )

        # Check that masked regions are zeroed (albedo)
        assert np.all(albedo[~sample_mask] == 0.0)

        # Normal should be [0, 0, 1] outside mask
        assert np.allclose(normal[~sample_mask], [0, 0, 1])

    def test_generate_with_depth(self, generator, sample_rgb, sample_depth):
        """Test PBR generation with depth map."""
        albedo, normal, roughness, metallic, ao, height = generator.generate_pbr_textures(
            rgb=sample_rgb,
            depth=sample_depth,
        )

        # Depth should influence normal map and AO
        assert normal.shape == sample_rgb.shape
        assert ao.shape == sample_rgb.shape[:2]

    def test_albedo_generation(self, generator, sample_rgb, sample_mask):
        """Test albedo map generation."""
        albedo = generator._generate_albedo(sample_rgb, sample_mask)

        # Albedo should be in [0, 1]
        assert np.all(albedo >= 0.0)
        assert np.all(albedo <= 1.0)

        # Should preserve color information
        assert albedo.shape == sample_rgb.shape

    def test_normal_generation(self, generator, sample_rgb, sample_mask):
        """Test normal map generation."""
        normal = generator._generate_normal(
            rgb=sample_rgb,
            depth=None,
            strength=1.0,
            mask=sample_mask,
        )

        # Normals should be in [-1, 1]
        assert np.all(normal >= -1.0)
        assert np.all(normal <= 1.0)

        # Normals should be normalized (length ≈ 1)
        norms = np.linalg.norm(normal, axis=2)
        assert np.allclose(norms, 1.0, atol=1e-5)

    def test_normal_strength_parameter(self, generator, sample_rgb, sample_mask):
        """Test normal strength parameter effect."""
        normal_weak = generator._generate_normal(
            rgb=sample_rgb,
            depth=None,
            strength=0.5,
            mask=sample_mask,
        )

        normal_strong = generator._generate_normal(
            rgb=sample_rgb,
            depth=None,
            strength=2.0,
            mask=sample_mask,
        )

        # Stronger normals should have more XY variation
        # (Z should be smaller relative to XY)
        assert np.mean(np.abs(normal_strong[:, :, :2])) > np.mean(np.abs(normal_weak[:, :, :2]))

    def test_roughness_generation(self, generator, sample_rgb, sample_mask):
        """Test roughness map generation."""
        roughness = generator._generate_roughness(
            rgb=sample_rgb,
            material_hint=None,
            mask=sample_mask,
        )

        # Roughness should be in [0, 1]
        assert np.all(roughness >= 0.0)
        assert np.all(roughness <= 1.0)

    def test_roughness_material_hints(self, generator, sample_rgb, sample_mask):
        """Test material hints affect roughness."""
        roughness_metal = generator._generate_roughness(
            rgb=sample_rgb,
            material_hint="metal",
            mask=sample_mask,
        )

        roughness_wood = generator._generate_roughness(
            rgb=sample_rgb,
            material_hint="wood",
            mask=sample_mask,
        )

        # Wood should be rougher than metal
        assert np.mean(roughness_wood[sample_mask]) > np.mean(roughness_metal[sample_mask])

    def test_metallic_generation(self, generator, sample_rgb, sample_mask):
        """Test metallic map generation."""
        metallic = generator._generate_metallic(
            rgb=sample_rgb,
            material_hint=None,
            mask=sample_mask,
        )

        # Metallic should be in [0, 1]
        assert np.all(metallic >= 0.0)
        assert np.all(metallic <= 1.0)

    def test_metallic_material_hints(self, generator, sample_rgb, sample_mask):
        """Test material hints affect metallic."""
        metallic_metal = generator._generate_metallic(
            rgb=sample_rgb,
            material_hint="metal",
            mask=sample_mask,
        )

        metallic_wood = generator._generate_metallic(
            rgb=sample_rgb,
            material_hint="wood",
            mask=sample_mask,
        )

        # Metal should have high metallic, wood should be zero
        assert np.mean(metallic_metal[sample_mask]) > 0.8
        assert np.mean(metallic_wood[sample_mask]) < 0.1

    def test_ao_generation(self, generator, sample_rgb, sample_mask):
        """Test ambient occlusion generation."""
        ao = generator._generate_ao(
            rgb=sample_rgb,
            depth=None,
            intensity=0.7,
            mask=sample_mask,
        )

        # AO should be in [0, 1]
        assert np.all(ao >= 0.0)
        assert np.all(ao <= 1.0)

    def test_ao_intensity_parameter(self, generator, sample_rgb, sample_depth, sample_mask):
        """Test AO intensity parameter effect."""
        ao_weak = generator._generate_ao(
            rgb=sample_rgb,
            depth=sample_depth,
            intensity=0.3,
            mask=sample_mask,
        )

        ao_strong = generator._generate_ao(
            rgb=sample_rgb,
            depth=sample_depth,
            intensity=1.0,
            mask=sample_mask,
        )

        # Stronger intensity should create darker occlusion
        assert np.mean(ao_weak) > np.mean(ao_strong)

    def test_height_generation(self, generator, sample_rgb, sample_mask):
        """Test height map generation."""
        height = generator._generate_height(
            rgb=sample_rgb,
            depth=None,
            mask=sample_mask,
        )

        # Height should be in [0, 1]
        assert np.all(height >= 0.0)
        assert np.all(height <= 1.0)

    def test_height_from_depth(self, generator, sample_rgb, sample_depth, sample_mask):
        """Test height generation from depth map."""
        height = generator._generate_height(
            rgb=sample_rgb,
            depth=sample_depth,
            mask=sample_mask,
        )

        # Height should reflect depth structure
        assert height.shape == sample_depth.shape
        assert np.all(height >= 0.0)
        assert np.all(height <= 1.0)

    def test_all_material_hints(self, generator, sample_rgb):
        """Test all material hints work without errors."""
        materials = ["wood", "stone", "metal", "glass", "fabric", "concrete", "leather", "ceramic"]

        for material in materials:
            albedo, normal, roughness, metallic, ao, height = generator.generate_pbr_textures(
                rgb=sample_rgb,
                material_hint=material,
            )

            # All outputs should be valid
            assert albedo.shape == sample_rgb.shape
            assert normal.shape == sample_rgb.shape
            assert roughness.shape == sample_rgb.shape[:2]
            assert metallic.shape == sample_rgb.shape[:2]
