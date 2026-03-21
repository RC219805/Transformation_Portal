"""Unit tests for materials contracts (Phase 2.2)."""

import numpy as np
import pytest

from transformation_portal.spatial_ai.materials.contracts import (

pytestmark = pytest.mark.unit

    MaterialGenerationConfig,
    MaterialInput,
    MaterialProperties,
    PBRTextures,
)


class TestMaterialInput:
    """Test MaterialInput contract validation."""

    def test_valid_input(self):
        """Test valid material input."""
        image = np.random.rand(512, 512, 3).astype(np.float32)
        mat_input = MaterialInput(
            image=image,
            gamma=1.0,
        )
        assert mat_input.gamma == 1.0
        assert mat_input.mask is None

    def test_gamma_enforcement(self):
        """Test gamma=1.0 enforcement (SpatialCaptureV1 contract)."""
        image = np.random.rand(512, 512, 3).astype(np.float32)

        with pytest.raises(ValueError, match="gamma=1.0"):
            MaterialInput(
                image=image,
                gamma=2.2,  # Invalid
            )

    def test_dtype_enforcement(self):
        """Test float32 dtype enforcement."""
        image = np.random.rand(512, 512, 3).astype(np.float64)  # Wrong dtype

        with pytest.raises(ValueError, match="float32"):
            MaterialInput(
                image=image,
                gamma=1.0,
            )

    def test_shape_validation(self):
        """Test image shape validation."""
        # Wrong dimensions
        image = np.random.rand(512, 512).astype(np.float32)

        with pytest.raises(ValueError, match="\\(H, W, 3\\)"):
            MaterialInput(
                image=image,
                gamma=1.0,
            )

    def test_mask_validation(self):
        """Test mask validation."""
        image = np.random.rand(512, 512, 3).astype(np.float32)

        # Wrong dtype
        mask = np.ones((512, 512), dtype=np.uint8)
        with pytest.raises(ValueError, match="bool"):
            MaterialInput(
                image=image,
                gamma=1.0,
                mask=mask,
            )

        # Wrong shape
        mask = np.ones((256, 256), dtype=bool)
        with pytest.raises(ValueError, match="must match"):
            MaterialInput(
                image=image,
                gamma=1.0,
                mask=mask,
            )

    def test_depth_validation(self):
        """Test depth validation."""
        image = np.random.rand(512, 512, 3).astype(np.float32)

        # Wrong dtype
        depth = np.random.rand(512, 512).astype(np.float64)
        with pytest.raises(ValueError, match="float32"):
            MaterialInput(
                image=image,
                gamma=1.0,
                depth=depth,
            )

        # Negative values
        depth = np.random.rand(512, 512).astype(np.float32) - 0.5
        with pytest.raises(ValueError, match="non-negative"):
            MaterialInput(
                image=image,
                gamma=1.0,
                depth=depth,
            )

    def test_material_hint_validation(self):
        """Test material hint validation."""
        image = np.random.rand(512, 512, 3).astype(np.float32)

        # Invalid hint
        with pytest.raises(ValueError, match="Material hint"):
            MaterialInput(
                image=image,
                gamma=1.0,
                material_hint="unknown_material",
            )

        # Valid hints
        for hint in ["wood", "stone", "metal", "glass", "fabric", "concrete", "leather", "ceramic"]:
            mat_input = MaterialInput(
                image=image,
                gamma=1.0,
                material_hint=hint,
            )
            assert mat_input.material_hint == hint


class TestMaterialProperties:
    """Test MaterialProperties validation."""

    def test_valid_properties(self):
        """Test valid material properties."""
        props = MaterialProperties(
            roughness_mean=0.5,
            metallic_mean=0.0,
            ao_strength=0.3,
        )
        assert props.roughness_mean == 0.5
        assert props.metallic_mean == 0.0

    def test_roughness_range(self):
        """Test roughness must be in [0, 1]."""
        with pytest.raises(ValueError, match="Roughness"):
            MaterialProperties(
                roughness_mean=1.5,  # Out of range
                metallic_mean=0.0,
                ao_strength=0.3,
            )

    def test_metallic_range(self):
        """Test metallic must be in [0, 1]."""
        with pytest.raises(ValueError, match="Metallic"):
            MaterialProperties(
                roughness_mean=0.5,
                metallic_mean=-0.1,  # Out of range
                ao_strength=0.3,
            )

    def test_ao_strength_range(self):
        """Test AO strength must be in [0, 1]."""
        with pytest.raises(ValueError, match="AO strength"):
            MaterialProperties(
                roughness_mean=0.5,
                metallic_mean=0.0,
                ao_strength=1.5,  # Out of range
            )

    def test_normal_strength_range(self):
        """Test normal strength must be in [0, 2]."""
        with pytest.raises(ValueError, match="Normal strength"):
            MaterialProperties(
                roughness_mean=0.5,
                metallic_mean=0.0,
                ao_strength=0.3,
                normal_strength=3.0,  # Out of range
            )


class TestPBRTextures:
    """Test PBRTextures contract validation."""

    def test_valid_pbr_textures(self):
        """Test valid PBR textures."""
        H, W = 512, 512
        pbr = PBRTextures(
            albedo=np.random.rand(H, W, 3).astype(np.float32),
            normal=np.random.rand(H, W, 3).astype(np.float32) * 2 - 1,  # [-1, 1]
            roughness=np.random.rand(H, W).astype(np.float32),
            metallic=np.random.rand(H, W).astype(np.float32),
            ambient_occlusion=np.random.rand(H, W).astype(np.float32),
        )
        assert pbr.albedo.shape == (H, W, 3)
        assert pbr.normal.shape == (H, W, 3)

    def test_albedo_validation(self):
        """Test albedo validation."""
        H, W = 512, 512

        # Wrong dtype
        with pytest.raises(ValueError, match="Albedo must be float32"):
            PBRTextures(
                albedo=np.random.rand(H, W, 3).astype(np.float64),
                normal=np.random.rand(H, W, 3).astype(np.float32) * 2 - 1,
                roughness=np.random.rand(H, W).astype(np.float32),
                metallic=np.random.rand(H, W).astype(np.float32),
                ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            )

        # Out of range
        with pytest.raises(ValueError, match="Albedo must be in"):
            PBRTextures(
                albedo=np.random.rand(H, W, 3).astype(np.float32) * 2,  # > 1
                normal=np.random.rand(H, W, 3).astype(np.float32) * 2 - 1,
                roughness=np.random.rand(H, W).astype(np.float32),
                metallic=np.random.rand(H, W).astype(np.float32),
                ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            )

    def test_normal_validation(self):
        """Test normal map validation."""
        H, W = 512, 512

        # Out of range
        with pytest.raises(ValueError, match="Normal must be in"):
            PBRTextures(
                albedo=np.random.rand(H, W, 3).astype(np.float32),
                normal=np.random.rand(H, W, 3).astype(np.float32) * 3,  # > 1
                roughness=np.random.rand(H, W).astype(np.float32),
                metallic=np.random.rand(H, W).astype(np.float32),
                ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            )

    def test_dimension_consistency(self):
        """Test all textures have consistent dimensions."""
        H, W = 512, 512

        # Mismatched roughness dimensions
        with pytest.raises(ValueError, match="Roughness shape"):
            PBRTextures(
                albedo=np.random.rand(H, W, 3).astype(np.float32),
                normal=np.random.rand(H, W, 3).astype(np.float32) * 2 - 1,
                roughness=np.random.rand(256, 256).astype(np.float32),  # Wrong size
                metallic=np.random.rand(H, W).astype(np.float32),
                ambient_occlusion=np.random.rand(H, W).astype(np.float32),
            )


class TestMaterialGenerationConfig:
    """Test MaterialGenerationConfig validation."""

    def test_valid_config(self):
        """Test valid configuration."""
        config = MaterialGenerationConfig(
            backend="nvdiffrec",
            resolution=1024,
            optimize_iterations=100,
        )
        assert config.backend == "nvdiffrec"
        assert config.resolution == 1024

    def test_resolution_validation(self):
        """Test resolution must be power of 2."""
        with pytest.raises(ValueError, match="Resolution must be"):
            MaterialGenerationConfig(
                backend="nvdiffrec",
                resolution=1000,  # Not power of 2
            )

    def test_iterations_validation(self):
        """Test iterations must be positive."""
        with pytest.raises(ValueError, match="Iterations must be positive"):
            MaterialGenerationConfig(
                backend="nvdiffrec",
                optimize_iterations=0,
            )

    def test_normal_strength_range(self):
        """Test normal strength must be in [0, 2]."""
        with pytest.raises(ValueError, match="Normal strength"):
            MaterialGenerationConfig(
                backend="nvdiffrec",
                normal_strength=3.0,  # Out of range
            )

    def test_ao_intensity_range(self):
        """Test AO intensity must be in [0, 1]."""
        with pytest.raises(ValueError, match="AO intensity"):
            MaterialGenerationConfig(
                backend="nvdiffrec",
                ao_intensity=1.5,  # Out of range
            )
