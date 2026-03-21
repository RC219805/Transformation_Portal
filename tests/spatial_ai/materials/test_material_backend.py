"""Unit tests for material backend (Phase 2.2)."""

import warnings

import numpy as np
import pytest

from transformation_portal.spatial_ai.materials.contracts import MaterialGenerationConfig
from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend



pytestmark = pytest.mark.unit

class TestMaterialBackend:
    """Test material backend wrapper."""

    @pytest.fixture
    def sample_rgb(self):
        """Create sample RGB image."""
        return np.random.rand(256, 256, 3).astype(np.float32)

    @pytest.fixture
    def sample_mask(self):
        """Create sample mask."""
        mask = np.zeros((256, 256), dtype=bool)
        mask[64:192, 64:192] = True
        return mask

    def test_heuristic_backend(self, sample_rgb):
        """Test heuristic backend initialization."""
        backend = MaterialBackend(backend="heuristic", device="cpu")

        result = backend.generate_pbr_textures(
            rgb=sample_rgb,
        )

        # Check outputs
        assert result.albedo.shape == sample_rgb.shape
        assert result.normal.shape == sample_rgb.shape
        assert result.roughness.shape == sample_rgb.shape[:2]
        assert result.metallic.shape == sample_rgb.shape[:2]
        assert result.ambient_occlusion.shape == sample_rgb.shape[:2]
        assert result.height.shape == sample_rgb.shape[:2]
        assert result.properties is not None

    def test_nvdiffrec_fallback(self, sample_rgb):
        """Test NVDIFFREC falls back to heuristic (not yet implemented)."""
        backend = MaterialBackend(backend="nvdiffrec", device="cpu")

        # Should warn and fall back to heuristic
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backend.generate_pbr_textures(
                rgb=sample_rgb,
            )

            # Check warning was raised
            assert len(w) == 1
            assert "not yet implemented" in str(w[0].message).lower()
            assert "falling back" in str(w[0].message).lower()

        # Should still return valid outputs
        assert result.albedo.shape == sample_rgb.shape

    def test_material_gan_fallback(self, sample_rgb):
        """Test MaterialGAN falls back to heuristic (not yet implemented)."""
        backend = MaterialBackend(backend="material_gan", device="cpu")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backend.generate_pbr_textures(
                rgb=sample_rgb,
            )

            # Check warning was raised
            assert len(w) == 1
            assert "not yet implemented" in str(w[0].message).lower()

    def test_with_config(self, sample_rgb):
        """Test backend with custom configuration."""
        backend = MaterialBackend(backend="heuristic", device="cpu")

        config = MaterialGenerationConfig(
            backend="heuristic",
            resolution=512,
            optimize_iterations=50,
            normal_strength=1.5,
            ao_intensity=0.8,
        )

        result = backend.generate_pbr_textures(
            rgb=sample_rgb,
            config=config,
        )

        assert result.albedo.shape == sample_rgb.shape

    def test_with_mask_and_depth(self, sample_rgb, sample_mask):
        """Test backend with mask and depth."""
        backend = MaterialBackend(backend="heuristic", device="cpu")

        depth = np.random.rand(256, 256).astype(np.float32) * 10.0

        result = backend.generate_pbr_textures(
            rgb=sample_rgb,
            mask=sample_mask,
            depth=depth,
        )

        # Masked regions should be zeroed in albedo
        assert np.all(result.albedo[~sample_mask] == 0.0)

    def test_material_properties_output(self, sample_rgb, sample_mask):
        """Test that material properties are computed correctly."""
        backend = MaterialBackend(backend="heuristic", device="cpu")

        result = backend.generate_pbr_textures(
            rgb=sample_rgb,
            mask=sample_mask,
        )

        # Properties should reflect masked region statistics
        assert result.properties.roughness_mean == pytest.approx(np.mean(result.roughness[sample_mask]), abs=0.01)
        assert result.properties.metallic_mean == pytest.approx(np.mean(result.metallic[sample_mask]), abs=0.01)

    def test_unload_model(self):
        """Test model unloading."""
        backend = MaterialBackend(backend="heuristic", device="cpu")

        # Should not raise error
        backend.unload_model()

        # Should still be functional after unload
        rgb = np.random.rand(64, 64, 3).astype(np.float32)
        result = backend.generate_pbr_textures(rgb=rgb)
        assert result.albedo.shape == rgb.shape

    def test_pbr_fusion_fallback(self, sample_rgb):
        """Test PBRFusion falls back to heuristic when not installed (Phase 5B)."""
        backend = MaterialBackend(backend="pbr_fusion", device="cpu")

        # Should warn and fall back to heuristic
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backend.generate_pbr_textures(
                rgb=sample_rgb,
            )

            # Check warning was issued
            assert len(w) == 1
            assert "PBRFusion not installed" in str(w[0].message) or "not yet implemented" in str(w[0].message)

        # Should still produce valid output (via heuristic fallback)
        assert result.albedo.shape == sample_rgb.shape
        assert result.normal.shape == sample_rgb.shape
        assert result.roughness.shape == sample_rgb.shape[:2]
        assert result.metallic.shape == sample_rgb.shape[:2]
        assert result.ambient_occlusion.shape == sample_rgb.shape[:2]
        assert result.height.shape == sample_rgb.shape[:2]
        assert result.properties is not None

    def test_metadata_bilateral_flag_reflects_available_backend(self, sample_rgb):
        """Metadata should record the actual albedo filtering capability."""
        backend = MaterialBackend(backend="heuristic", device="cpu")
        backend._bilateral_filter_available = False
        result = backend.generate_pbr_textures(rgb=sample_rgb)
        assert result.metadata.bilateral_enabled is False

        backend._bilateral_filter_available = True
        result = backend.generate_pbr_textures(rgb=sample_rgb)
        assert result.metadata.bilateral_enabled is True
