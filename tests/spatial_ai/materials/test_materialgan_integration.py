"""Tests for MaterialGAN integration (ADR-026 §3.2 - Phase 2.2 roadmap).

These tests validate MaterialGAN-specific functionality when integrated.
Currently, MaterialGAN is a placeholder that falls back to heuristic.

Coverage:
- MaterialGAN initialization
- PBR texture quality (when available)
- Material classification accuracy
- Fallback behavior
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend

# Module-level availability check for MaterialGAN
try:
    import materialgan  # noqa: F401

    HAS_MATERIALGAN = True
except ImportError:
    HAS_MATERIALGAN = False

pytestmark = [
    pytest.mark.ml,
]


def _materialgan_available() -> bool:
    """Check if MaterialGAN package is available."""
    return HAS_MATERIALGAN


class TestMaterialGANFallback:
    """Test MaterialGAN fallback behavior (current implementation).

    MaterialGAN is intentionally a placeholder that falls back to heuristic.
    This is documented in TODO_INVENTORY.md §2.0.4-5.
    """

    @pytest.fixture
    def sample_rgb(self):
        """Create sample RGB image."""
        return np.random.rand(256, 256, 3).astype(np.float32)

    def test_materialgan_fallback_warning(self, sample_rgb):
        """MaterialGAN should emit input-contract warning and fall back to heuristic."""
        from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend

        backend = MaterialBackend(backend="material_gan", device="cpu")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = backend.generate_pbr_textures(rgb=sample_rgb)

            assert len(w) == 1
            assert "single-image input" in str(w[0].message).lower()

        assert result.albedo.shape == sample_rgb.shape
        assert result.normal.shape == sample_rgb.shape
        assert result.roughness.shape == sample_rgb.shape[:2]
        assert result.metadata is not None
        assert result.metadata.backend_decision is not None
        assert result.metadata.backend_decision.availability_state.value == "input_contract_mismatch"

    def test_materialgan_produces_valid_pbr_output(self, sample_rgb):
        """Even in fallback mode, output should be valid PBR textures."""
        from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend

        backend = MaterialBackend(backend="material_gan", device="cpu")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = backend.generate_pbr_textures(rgb=sample_rgb)

        # Validate PBR ranges
        assert result.albedo.min() >= 0.0
        assert result.albedo.max() <= 1.0
        assert result.roughness.min() >= 0.0
        assert result.roughness.max() <= 1.0
        assert result.metallic.min() >= 0.0
        assert result.metallic.max() <= 1.0

    def test_materialgan_with_material_hint(self, sample_rgb):
        """Material hints should affect fallback behavior."""
        from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend

        backend = MaterialBackend(backend="material_gan", device="cpu")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result_metal = backend.generate_pbr_textures(rgb=sample_rgb, material_hint="metal")
            result_wood = backend.generate_pbr_textures(rgb=sample_rgb, material_hint="wood")

        # Metal should have higher metallic mean
        assert result_metal.properties.metallic_mean > result_wood.properties.metallic_mean


@pytest.mark.skipif(not HAS_MATERIALGAN, reason="MaterialGAN package not installed (optional dependency)")
class TestMaterialGANIntegration:
    """Test MaterialGAN when package is available (future implementation).

    These tests are skipped unless the materialgan package is installed.
    They document expected behavior for Phase 2.2 completion.
    """

    @pytest.fixture
    def sample_rgb(self):
        """Create sample RGB image."""
        return np.random.rand(256, 256, 3).astype(np.float32)

    @pytest.fixture
    def sample_depth(self):
        """Create sample depth map."""
        return np.random.rand(256, 256).astype(np.float32)

    @pytest.mark.skip(reason="MaterialGAN integration not yet implemented (Phase 2.2 roadmap)")
    def test_materialgan_checkpoint_loading(self):
        """MaterialGAN should load checkpoint from HuggingFace.

        Expected: checkpoints/materialgan_v2.pth
        License: CC BY-NC 4.0 (non-commercial research only)

        Note: CC BY-NC 4.0 prohibits commercial use. When implemented,
        the backend should enforce this via config gating (similar to
        Depth Pro's non_commercial_ok flag requirement).
        """
        # Phase 2.2 implementation placeholder
        pass

    @pytest.mark.skip(reason="MaterialGAN integration not yet implemented (Phase 2.2 roadmap)")
    def test_materialgan_pbr_quality(self, sample_rgb, sample_depth):
        """MaterialGAN should produce higher-quality PBR than heuristic.

        Expected quality improvement:
        - Normal map detail: +38% (artist preference)
        - Material consistency: higher across scene
        """
        # Phase 2.2 implementation placeholder
        pass

    @pytest.mark.skip(reason="MaterialGAN integration not yet implemented (Phase 2.2 roadmap)")
    def test_materialgan_depth_integration(self, sample_rgb, sample_depth):
        """MaterialGAN should use depth for geometry-aware PBR.

        Expected behavior:
        - Depth input improves normal estimation
        - Geometric normals fused with predicted normals
        """
        # Phase 2.2 implementation placeholder
        pass


class TestMaterialGANLicenseCompliance:
    """Test MaterialGAN license restrictions (CC BY-NC 4.0)."""

    def test_materialgan_license_documented(self):
        """MaterialGAN license should be documented as research-only."""
        from transformation_portal.spatial_ai.materials.material_backend import MaterialBackend

        backend = MaterialBackend(backend="material_gan", device="cpu")

        docstring = backend._generate_material_gan.__doc__ or ""
        assert "CC BY-NC" in docstring or "research" in docstring.lower() or "placeholder" in docstring.lower()


class TestMaterialGANClassification:
    """Test material classification accuracy with MaterialGAN (Phase 2.2).

    These tests document expected classification behavior when
    MaterialGAN is integrated with the material classification pipeline.
    """

    @pytest.mark.skip(reason="MaterialGAN classification not yet implemented (Phase 2.2 roadmap)")
    def test_material_classification_accuracy(self):
        """MaterialGAN should improve material classification.

        Expected: +10% IoU improvement over heuristic baseline.
        Target materials: wood, stone, metal, glass, fabric, concrete
        """
        # Phase 2.2 implementation placeholder
        pass

    @pytest.mark.skip(reason="MaterialGAN classification not yet implemented (Phase 2.2 roadmap)")
    def test_multi_material_scene_classification(self):
        """MaterialGAN should handle scenes with multiple materials.

        Expected: Per-segment material classification with SAM2 masks.
        """
        # Phase 2.2 implementation placeholder
        pass
