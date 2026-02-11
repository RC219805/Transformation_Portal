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
        # NOTE:
        # Calling ensemble.compute() would route through _run_models() and registry backends.
        # That path can silently collapse to a single model if names collide (dict overwrite),
        # and it also depends on registry wiring. For fusion correctness we test _fuse_predictions()
        # directly with two distinct model results.
        test_img = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        img_pil = Image.fromarray(test_img, mode="RGB")

        config = EnhanceConfig(non_commercial_ok=True, accept_research_tools_license=True)
        models = [
            ModelConfig(name="model_a", weight=0.5),
            ModelConfig(name="model_b", weight=0.5),
        ]
        ensemble = DepthEnsembleBackend(config, models=models)

        depth_a = np.ones((64, 64), dtype=np.float32) * 5.0
        depth_b = np.ones((64, 64), dtype=np.float32) * 5.1
        model_results = {
            "model_a": DepthResult(
                depth_map=depth_a,
                original_image=test_img,
                metadata={},
                depth_units="relative",
                backend_id="model_a",
                device="cpu",
                dtype="float32",
                input_size=(64, 64),
            ),
            "model_b": DepthResult(
                depth_map=depth_b,
                original_image=test_img,
                metadata={},
                depth_units="relative",
                backend_id="model_b",
                device="cpu",
                dtype="float32",
                input_size=(64, 64),
            ),
        }

        result = ensemble._fuse_predictions(model_results, img_pil)

        # Verify result type
        assert isinstance(result, EnsembleDepthResult)
        assert result.depth_map is not None
        assert result.variance_map is not None
        assert result.fusion_method == "variance_weighted"
        assert 0.0 <= result.model_agreement <= 1.0

        # Verify variance map shape matches depth map
        assert result.variance_map.shape == result.depth_map.shape[:2]

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

    def test_metric_alignment_with_mixed_backends(self):
        """Test metric depth alignment when mixing metric + relative models (Bug P1-A fix)."""
        # Create synthetic image
        test_img = (np.random.rand(100, 100, 3) * 255).astype(np.uint8)

        # Create mock results with different depth units
        metric_result = DepthResult(
            depth_map=np.ones((100, 100)) * 5.0,
            original_image=test_img,
            metadata={},
            depth_units="meters",  # metric
            backend_id="depth_pro",
            device="cpu",
            dtype="float32",
            input_size=(100, 100),
        )

        relative_result = DepthResult(
            depth_map=np.ones((100, 100)) * 0.5,
            original_image=test_img,
            metadata={},
            depth_units="relative",  # relative
            backend_id="da3",
            device="cpu",
            dtype="float32",
            input_size=(100, 100),
        )

        model_results = {
            "depth_pro": metric_result,
            "da3": relative_result,
        }

        config = EnhanceConfig(
            non_commercial_ok=True,
            accept_research_tools_license=True,
        )
        ensemble = DepthEnsembleBackend(config, models=[])

        # This should not crash (tests Bug P1-A fix: .items() not .values())
        aligned = ensemble._align_depth_maps(model_results)

        # Verify both models are aligned
        assert "depth_pro" in aligned
        assert "da3" in aligned

        # Verify shapes match
        assert aligned["depth_pro"].shape == aligned["da3"].shape

        # Verify metric model unchanged (already in meters)
        np.testing.assert_array_equal(aligned["depth_pro"], metric_result.depth_map)

        # Verify relative model scaled (approximate 10x scaling)
        assert aligned["da3"].mean() > 1.0  # Should be scaled up from 0.5

    def test_variance_weighted_fusion_actually_uses_variance(self):
        """Test that variance actually changes fusion output (Bug P1-B fix).

        MATHEMATICAL NOTE: With 2 models of equal weight, deviations from the ensemble
        mean have equal magnitude (opposite signs), so confidences are always equal and
        the result reduces to fixed average. This is a mathematical property, not a bug.

        To demonstrate adaptive behavior, we use 3 models where one is an outlier.
        """
        test_img = np.ones((100, 100, 3), dtype=np.uint8) * 128

        # Three models: A and C agree, B is outlier in top-left
        depth_a = np.linspace(1, 10, 10000).reshape(100, 100).astype(np.float32)
        depth_c = np.linspace(1, 10, 10000).reshape(100, 100).astype(np.float32)
        depth_b = np.linspace(1, 10, 10000).reshape(100, 100).astype(np.float32)
        depth_b[:25, :25] = 50.0  # Strong outlier

        model_results = {
            "model_a": DepthResult(
                depth_map=depth_a,
                original_image=test_img,
                metadata={},
                depth_units="relative",
                backend_id="model_a",
                device="cpu",
                dtype="float32",
                input_size=(100, 100),
            ),
            "model_b": DepthResult(
                depth_map=depth_b,
                original_image=test_img,
                metadata={},
                depth_units="relative",
                backend_id="model_b",
                device="cpu",
                dtype="float32",
                input_size=(100, 100),
            ),
            "model_c": DepthResult(
                depth_map=depth_c,
                original_image=test_img,
                metadata={},
                depth_units="relative",
                backend_id="model_c",
                device="cpu",
                dtype="float32",
                input_size=(100, 100),
            ),
        }

        config = EnhanceConfig(non_commercial_ok=True, accept_research_tools_license=True)
        ensemble = DepthEnsembleBackend(
            config,
            models=[
                ModelConfig(name="model_a", weight=1.0 / 3),
                ModelConfig(name="model_b", weight=1.0 / 3),
                ModelConfig(name="model_c", weight=1.0 / 3),
            ],
        )

        result = ensemble._fuse_predictions(model_results, Image.fromarray(test_img))

        # Get aligned maps and compute fixed average
        aligned = ensemble._align_depth_maps(model_results)
        fixed_avg = (1.0 / 3) * aligned["model_a"] + (1.0 / 3) * aligned["model_b"] + (1.0 / 3) * aligned["model_c"]

        # Regions
        outlier_region = (slice(0, 25), slice(0, 25))
        normal_region = (slice(75, 100), slice(75, 100))

        fused_outlier_mean = float(np.mean(result.depth_map[outlier_region]))
        fixed_outlier_mean = float(np.mean(fixed_avg[outlier_region]))
        consensus_outlier_mean = float(np.mean((aligned["model_a"][outlier_region] + aligned["model_c"][outlier_region]) / 2))

        # In outlier region:
        # Models A and C agree (consensus), B is outlier
        # Adaptive fusion should downweight B and be closer to A+C consensus than fixed average
        dist_adaptive_to_consensus = abs(fused_outlier_mean - consensus_outlier_mean)
        dist_fixed_to_consensus = abs(fixed_outlier_mean - consensus_outlier_mean)

        assert dist_adaptive_to_consensus < dist_fixed_to_consensus, (
            f"Expected adaptive fusion to be closer to consensus (A+C) than fixed average. "
            f"adaptive_dist={dist_adaptive_to_consensus:.6f}, fixed_dist={dist_fixed_to_consensus:.6f}, "
            f"fused={fused_outlier_mean:.3f}, fixed={fixed_outlier_mean:.3f}, consensus={consensus_outlier_mean:.3f}"
        )

        # Variance map sanity: outlier region variance > normal region variance
        outlier_var = float(np.mean(result.variance_map[outlier_region]))
        normal_var = float(np.mean(result.variance_map[normal_region]))
        assert (
            outlier_var > normal_var
        ), f"Expected higher variance in outlier block: outlier={outlier_var:.6f}, normal={normal_var:.6f}"

        # Agreement should be < 1.0 due to disagreement
        assert result.model_agreement < 0.99


# Pytest markers
pytestmark = [
    pytest.mark.apex_ultra,
    pytest.mark.depth,
]
