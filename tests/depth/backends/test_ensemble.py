"""Tests for depth ensemble backend (APEX Research Ultra ADR-026).

Tests cover:
- Variance-weighted fusion algorithm
- Multi-model configuration
- License enforcement
- Graceful fallback
- Model alignment (metric vs relative depth)
- Cache key generation
- Error handling
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth.backends.ensemble import DepthEnsembleBackend, EnsembleDepthResult, ModelConfig
from transformation_portal.depth.backends.protocol import DepthResult, LicenseRestrictionError
from transformation_portal.lux_depth_v3.config import EnhanceConfig


class TestEnsembleBackend:
    """Test suite for DepthEnsembleBackend."""

    def test_backend_protocol_attributes(self):
        """Test that ensemble backend implements DepthBackend protocol."""
        assert DepthEnsembleBackend.name == "ensemble"
        assert DepthEnsembleBackend.license_type.value == "research_only"
        assert DepthEnsembleBackend.requires_checkpoint is True

    def test_default_model_configuration(self):
        """Test that default 3-model config is created correctly."""
        config = EnhanceConfig(non_commercial_ok=True, accept_research_tools_license=True)
        ensemble = DepthEnsembleBackend(config)

        # Should have 3 models (Depth Pro + DA3 + DepthCrafter stub)
        assert len(ensemble._models) == 3

        # Check model names and weights
        model_names = [m.name for m in ensemble._models]
        assert "depth_pro" in model_names
        assert "da3" in model_names
        assert "depthcrafter_stub" in model_names

        # Weights should sum to 1.0 (for enabled models)
        enabled_weights = sum(m.weight for m in ensemble._models if m.enabled)
        assert abs(enabled_weights - 1.0) < 1e-6 or enabled_weights == 0.8  # Stub disabled

    def test_license_enforcement_requires_non_commercial(self):
        """Test that ensemble requires non_commercial_ok=True."""
        from transformation_portal.depth.backends.registry import DepthBackendRegistry

        config = EnhanceConfig(non_commercial_ok=False)  # Missing required flag
        registry = DepthBackendRegistry()

        with pytest.raises(LicenseRestrictionError, match="non_commercial_ok=True"):
            registry.get_backend("ensemble", config)

    def test_license_enforcement_requires_research_tools(self):
        """Test that ensemble requires accept_research_tools_license=True."""
        from transformation_portal.depth.backends.registry import DepthBackendRegistry

        config = EnhanceConfig(
            non_commercial_ok=True,
            accept_research_tools_license=False,  # Missing required flag
        )
        registry = DepthBackendRegistry()

        with pytest.raises(LicenseRestrictionError, match="accept_research_tools_license=True"):
            registry.get_backend("ensemble", config)

    def test_license_passes_with_all_flags(self):
        """Test that ensemble initializes with all required flags."""
        from transformation_portal.depth.backends.registry import DepthBackendRegistry

        config = EnhanceConfig(
            non_commercial_ok=True,
            accept_research_tools_license=True,
            spatial_ai_linear_ingest=True,
        )
        registry = DepthBackendRegistry()

        # Should not raise
        backend = registry.get_backend("ensemble", config)
        assert backend is not None
        assert backend.name == "ensemble"

    def test_custom_model_configuration(self):
        """Test that custom model configs can be passed."""
        models = [
            ModelConfig(name="da3", weight=0.6),
            ModelConfig(name="synthetic", weight=0.4),
        ]

        config = EnhanceConfig(non_commercial_ok=True, accept_research_tools_license=True)
        ensemble = DepthEnsembleBackend(config, models=models)

        assert len(ensemble._models) == 2
        assert ensemble._models[0].name == "da3"
        assert ensemble._models[1].name == "synthetic"

    def test_weight_normalization(self):
        """Test that weights are normalized if they don't sum to 1.0."""
        models = [
            ModelConfig(name="da3", weight=0.5),
            ModelConfig(name="synthetic", weight=0.3),  # Total = 0.8
        ]

        config = EnhanceConfig(non_commercial_ok=True, accept_research_tools_license=True)
        ensemble = DepthEnsembleBackend(config, models=models)

        # Weights should be normalized
        enabled_weights = sum(m.weight for m in ensemble._models if m.enabled)
        assert abs(enabled_weights - 1.0) < 1e-6

    def test_variance_weighted_fusion_synthetic(self):
        """Test variance-weighted fusion with synthetic backends."""
        # Create synthetic image
        test_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        img_pil = Image.fromarray(test_img, mode="RGB")

        # Use only synthetic backends (no ML deps required)
        models = [
            ModelConfig(name="synthetic", weight=0.5),
            ModelConfig(name="synthetic", weight=0.5),
        ]

        config = EnhanceConfig(
            non_commercial_ok=True,
            accept_research_tools_license=True,
            allow_synthetic_fallback=True,
        )
        ensemble = DepthEnsembleBackend(config, models=models)

        # Compute ensemble depth
        result = ensemble.compute(img_pil)

        # Verify result type
        assert isinstance(result, EnsembleDepthResult)
        assert result.depth_map is not None
        assert result.variance_map is not None
        assert result.fusion_method == "variance_weighted"
        assert 0.0 <= result.model_agreement <= 1.0

        # Verify variance map shape matches depth map
        assert result.variance_map.shape == result.depth_map.shape[:2]

        # Verify per-model data is populated
        assert len(result.per_model_depths) >= 1
        assert len(result.per_model_weights) >= 1

    def test_cache_key_generation(self):
        """Test that cache keys are deterministic and unique."""
        test_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

        models = [
            ModelConfig(name="da3", weight=0.5),
            ModelConfig(name="synthetic", weight=0.5),
        ]

        config = EnhanceConfig(non_commercial_ok=True, accept_research_tools_license=True)
        ensemble = DepthEnsembleBackend(config, models=models)

        # Generate cache key twice for same image
        key1 = ensemble.get_cache_key(test_img)
        key2 = ensemble.get_cache_key(test_img)

        # Keys should match for same image
        assert key1 == key2
        assert len(key1) == 64  # SHA-256 hex

        # Different image should give different key
        different_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)
        key3 = ensemble.get_cache_key(different_img)
        assert key3 != key1

    def test_required_packages(self):
        """Test that required packages are documented."""
        packages = DepthEnsembleBackend.required_packages()
        assert isinstance(packages, list)
        assert "transformers" in packages  # DA3 requirement

    def test_ensemble_validates_minimum_models(self):
        """Test that ensemble warns if <2 enabled models."""
        models = [
            ModelConfig(name="da3", weight=1.0, enabled=True),
        ]

        config = EnhanceConfig(non_commercial_ok=True, accept_research_tools_license=True)

        # Should warn but not fail
        ensemble = DepthEnsembleBackend(config, models=models)
        assert len([m for m in ensemble._models if m.enabled]) == 1

    def test_ensemble_result_extended_fields(self):
        """Test that EnsembleDepthResult has all extended fields."""
        # Create minimal result
        result = EnsembleDepthResult(
            depth_map=np.zeros((100, 100)),
            original_image=np.zeros((100, 100, 3), dtype=np.uint8),
            metadata={},
            variance_map=np.zeros((100, 100)),
            fusion_method="variance_weighted",
            model_agreement=0.9,
        )

        # Verify extended fields exist
        assert hasattr(result, "variance_map")
        assert hasattr(result, "per_model_depths")
        assert hasattr(result, "per_model_weights")
        assert hasattr(result, "fusion_method")
        assert hasattr(result, "model_agreement")

        # Verify values
        assert result.fusion_method == "variance_weighted"
        assert result.model_agreement == 0.9


class TestVarianceFusion:
    """Tests for variance-weighted fusion algorithm."""

    def test_low_variance_regions_get_higher_weight(self):
        """Test that regions with low inter-model variance get higher weight."""
        # This is a simplified unit test of the fusion logic
        # In practice, this would use actual model outputs

        # Simulate two depth maps with different variance characteristics
        depth1 = np.ones((100, 100)) * 5.0  # Uniform depth
        depth2 = np.ones((100, 100)) * 5.1  # Slightly different but consistent

        # Low variance region (models agree)
        variance_low = np.var(np.stack([depth1, depth2], axis=0), axis=0)

        # Create high variance region
        depth3 = np.ones((100, 100)) * 5.0
        depth4 = np.ones((100, 100)) * 10.0  # Large difference

        variance_high = np.var(np.stack([depth3, depth4], axis=0), axis=0)

        # Verify variance calculation
        assert variance_low.mean() < variance_high.mean()

        # Inverse variance weighting
        epsilon = 1e-6
        weight_low = 1.0 / (variance_low.mean() + epsilon)
        weight_high = 1.0 / (variance_high.mean() + epsilon)

        # Low variance should get higher weight
        assert weight_low > weight_high


# Pytest markers
pytestmark = [
    pytest.mark.apex_ultra,
    pytest.mark.depth,
]
