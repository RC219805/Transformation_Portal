"""Unit tests for PBR generator orchestrator (Phase 2.2)."""

import numpy as np
import pytest

from transformation_portal.spatial_ai.materials.contracts import MaterialGenerationConfig
from transformation_portal.spatial_ai.materials.pbr_generator import PBRGenerator

pytestmark = pytest.mark.unit


class TestPBRGenerator:
    """Test PBR generator orchestrator."""

    @pytest.fixture
    def generator(self):
        """Create PBR generator."""
        return PBRGenerator(backend="heuristic", device="cpu")

    @pytest.fixture
    def sample_rgb(self):
        """Create sample RGB image."""
        return np.random.rand(256, 256, 3).astype(np.float32)

    @pytest.fixture
    def sample_masks(self):
        """Create sample segmentation masks."""
        masks = []

        # Mask 1: top-left region
        mask1 = np.zeros((256, 256), dtype=bool)
        mask1[0:128, 0:128] = True
        masks.append(mask1)

        # Mask 2: bottom-right region
        mask2 = np.zeros((256, 256), dtype=bool)
        mask2[128:256, 128:256] = True
        masks.append(mask2)

        return masks

    def test_basic_generation(self, generator, sample_rgb):
        """Test basic PBR texture generation."""
        result = generator.generate(
            image=sample_rgb,
            gamma=1.0,
        )

        # Check contract validation passed
        assert result.albedo.shape == sample_rgb.shape
        assert result.normal.shape == sample_rgb.shape
        assert result.roughness.shape == sample_rgb.shape[:2]
        assert result.metallic.shape == sample_rgb.shape[:2]
        assert result.ambient_occlusion.shape == sample_rgb.shape[:2]

    def test_gamma_enforcement(self, generator, sample_rgb):
        """Test gamma=1.0 enforcement."""
        with pytest.raises(ValueError, match="gamma=1.0"):
            generator.generate(
                image=sample_rgb,
                gamma=2.2,  # Invalid
            )

    def test_generation_with_mask(self, generator, sample_rgb, sample_masks):
        """Test generation with segmentation mask."""
        result = generator.generate(
            image=sample_rgb,
            gamma=1.0,
            mask=sample_masks[0],
        )

        # Masked regions should be handled correctly
        assert result.albedo.shape == sample_rgb.shape
        assert np.all(result.albedo[~sample_masks[0]] == 0.0)

    def test_generation_with_depth(self, generator, sample_rgb):
        """Test generation with depth map."""
        depth = np.random.rand(256, 256).astype(np.float32) * 10.0

        result = generator.generate(
            image=sample_rgb,
            gamma=1.0,
            depth=depth,
        )

        assert result.albedo.shape == sample_rgb.shape

    def test_generation_with_material_hint(self, generator, sample_rgb):
        """Test generation with material hint."""
        result = generator.generate(
            image=sample_rgb,
            gamma=1.0,
            material_hint="metal",
        )

        # Metal should have high metallic value
        assert result.properties.metallic_mean > 0.8

        result_wood = generator.generate(
            image=sample_rgb,
            gamma=1.0,
            material_hint="wood",
        )

        # Wood should have low metallic value
        assert result_wood.properties.metallic_mean < 0.1

    def test_generation_with_config(self, generator, sample_rgb):
        """Test generation with custom configuration."""
        config = MaterialGenerationConfig(
            backend="heuristic",
            resolution=512,
            normal_strength=1.5,
            ao_intensity=0.8,
        )

        result = generator.generate(
            image=sample_rgb,
            gamma=1.0,
            config=config,
        )

        assert result.albedo.shape == sample_rgb.shape

    def test_batch_generation(self, generator, sample_rgb, sample_masks):
        """Test batch generation for multiple segments."""
        results = generator.generate_batch(
            image=sample_rgb,
            gamma=1.0,
            masks=sample_masks,
        )

        # Should return one result per mask
        assert len(results) == len(sample_masks)

        # Each result should be valid
        for i, result in enumerate(results):
            assert result.albedo.shape == sample_rgb.shape
            assert np.all(result.albedo[~sample_masks[i]] == 0.0)

    def test_batch_with_material_hints(self, generator, sample_rgb, sample_masks):
        """Test batch generation with material hints."""
        material_hints = ["metal", "wood"]

        results = generator.generate_batch(
            image=sample_rgb,
            gamma=1.0,
            masks=sample_masks,
            material_hints=material_hints,
        )

        # Metal segment should have high metallic
        assert results[0].properties.metallic_mean > 0.8

        # Wood segment should have low metallic
        assert results[1].properties.metallic_mean < 0.1

    def test_batch_with_depth(self, generator, sample_rgb, sample_masks):
        """Test batch generation with shared depth map."""
        depth = np.random.rand(256, 256).astype(np.float32) * 10.0

        results = generator.generate_batch(
            image=sample_rgb,
            gamma=1.0,
            masks=sample_masks,
            depth=depth,
        )

        assert len(results) == len(sample_masks)

    def test_batch_heuristic_shared_matches_per_segment_outputs(self, generator):
        """Shared heuristic intermediates should be equivalent to per-mask generation."""
        rng = np.random.default_rng(1234)
        image = rng.random((32, 32, 3), dtype=np.float32)
        depth = rng.random((32, 32), dtype=np.float32)
        masks = []
        mask_a = np.zeros((32, 32), dtype=bool)
        mask_a[2:20, 3:18] = True
        masks.append(mask_a)
        mask_b = np.zeros((32, 32), dtype=bool)
        mask_b[10:30, 12:31] = True
        masks.append(mask_b)
        hints = ["metal", "wood"]
        config = MaterialGenerationConfig(
            backend="heuristic",
            device="cpu",
            normal_strength=1.3,
            ao_intensity=0.6,
        )

        expected = [
            generator.generate(
                image=image,
                gamma=1.0,
                mask=mask,
                depth=depth,
                material_hint=hint,
                config=config,
            )
            for mask, hint in zip(masks, hints)
        ]
        actual = generator.generate_batch(
            image=image,
            gamma=1.0,
            masks=masks,
            depth=depth,
            material_hints=hints,
            config=config,
        )

        for expected_pbr, actual_pbr in zip(expected, actual):
            np.testing.assert_allclose(actual_pbr.albedo, expected_pbr.albedo, atol=1e-6)
            np.testing.assert_allclose(actual_pbr.normal, expected_pbr.normal, atol=1e-6)
            np.testing.assert_allclose(actual_pbr.roughness, expected_pbr.roughness, atol=1e-6)
            np.testing.assert_allclose(actual_pbr.metallic, expected_pbr.metallic, atol=1e-6)
            np.testing.assert_allclose(
                actual_pbr.ambient_occlusion,
                expected_pbr.ambient_occlusion,
                atol=1e-6,
            )
            np.testing.assert_allclose(actual_pbr.height, expected_pbr.height, atol=1e-6)
            assert actual_pbr.properties.roughness_mean == pytest.approx(expected_pbr.properties.roughness_mean)
            assert actual_pbr.properties.metallic_mean == pytest.approx(expected_pbr.properties.metallic_mean)
            assert actual_pbr.properties.ao_strength == pytest.approx(expected_pbr.properties.ao_strength)

    def test_contract_validation(self, generator):
        """Test input contract validation."""
        # Invalid image dtype
        image = np.random.rand(256, 256, 3).astype(np.float64)
        with pytest.raises(ValueError, match="float32"):
            generator.generate(image=image, gamma=1.0)

        # Invalid mask dtype
        image = np.random.rand(256, 256, 3).astype(np.float32)
        mask = np.ones((256, 256), dtype=np.uint8)
        with pytest.raises(ValueError, match="bool"):
            generator.generate(image=image, gamma=1.0, mask=mask)

    def test_output_contract_validation(self, generator, sample_rgb):
        """Test output contract validation."""
        # Generate should return valid PBRTextures
        result = generator.generate(
            image=sample_rgb,
            gamma=1.0,
        )

        # All values should be in valid ranges
        assert np.all(result.albedo >= 0.0) and np.all(result.albedo <= 1.0)
        assert np.all(result.normal >= -1.0) and np.all(result.normal <= 1.0)
        assert np.all(result.roughness >= 0.0) and np.all(result.roughness <= 1.0)
        assert np.all(result.metallic >= 0.0) and np.all(result.metallic <= 1.0)
        assert np.all(result.ambient_occlusion >= 0.0) and np.all(result.ambient_occlusion <= 1.0)

    def test_unload_model(self, generator):
        """Test model unloading."""
        generator.unload_model()

        # Should still work after unload
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        result = generator.generate(image=rgb, gamma=1.0)
        assert result.albedo.shape == rgb.shape

    def test_integration_with_sam2_masks(self, generator, sample_rgb):
        """Test integration with Phase 2.1 SAM2 segmentation masks."""
        # Simulate SAM2 output: boolean masks
        sam2_mask = np.zeros((256, 256), dtype=bool)
        sam2_mask[50:200, 50:200] = True

        result = generator.generate(
            image=sample_rgb,
            gamma=1.0,
            mask=sam2_mask,
        )

        # Should handle SAM2 masks correctly
        assert result.albedo.shape == sample_rgb.shape
        assert np.all(result.albedo[~sam2_mask] == 0.0)
