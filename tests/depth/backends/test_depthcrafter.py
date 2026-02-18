"""Tests for DepthCrafter temporal depth backend (ADR-026).

Tests cover:
- Backend protocol compliance
- Synthetic fallback when checkpoint unavailable
- Temporal EMA filtering across video frames
- Temporal state reset between sequences
- Cache key generation
- Registry integration
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth.backends.depthcrafter import DepthCrafterBackend
from transformation_portal.depth.backends.protocol import DepthResult, LicenseType


class TestDepthCrafterProtocol:
    """Test that DepthCrafterBackend implements DepthBackend protocol."""

    def test_backend_name(self):
        assert DepthCrafterBackend.name == "depthcrafter"

    def test_license_type_commercial(self):
        """Apache 2.0 = commercial use allowed."""
        assert DepthCrafterBackend.license_type == LicenseType.COMMERCIAL

    def test_requires_checkpoint(self):
        assert DepthCrafterBackend.requires_checkpoint is True

    def test_required_packages_empty_for_fallback(self):
        """Fallback mode has no extra deps beyond numpy/PIL."""
        assert DepthCrafterBackend.required_packages() == []


class TestDepthCrafterCompute:
    """Test compute() method and synthetic fallback."""

    def test_compute_returns_depth_result(self):
        """compute() returns a valid DepthResult."""
        backend = DepthCrafterBackend()
        img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))

        result = backend.compute(img)

        assert isinstance(result, DepthResult)
        assert result.depth_map.shape == (64, 64)
        assert result.depth_units == "relative"
        assert result.backend_id == "depthcrafter"

    def test_compute_accepts_numpy_array(self):
        """compute() accepts numpy array input."""
        backend = DepthCrafterBackend()
        img_array = np.random.randint(0, 255, (48, 48, 3), dtype=np.uint8)

        result = backend.compute(img_array)

        assert result.depth_map.shape == (48, 48)

    def test_compute_depth_values_in_unit_range(self):
        """Synthetic fallback depth values should be in [0, 1]."""
        backend = DepthCrafterBackend()
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))

        result = backend.compute(img)

        assert result.depth_map.min() >= 0.0
        assert result.depth_map.max() <= 1.0

    def test_compute_metadata_indicates_fallback(self):
        """When checkpoint unavailable, metadata should indicate fallback mode."""
        backend = DepthCrafterBackend()
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

        result = backend.compute(img)

        assert result.metadata["fallback_mode"] is True
        assert result.metadata["checkpoint_used"] is False
        assert result.metadata["backend"] == "depthcrafter"

    def test_compute_input_size_correct(self):
        """input_size should match image dimensions."""
        backend = DepthCrafterBackend()
        img = Image.fromarray(np.zeros((80, 120, 3), dtype=np.uint8))

        result = backend.compute(img)

        assert result.input_size == (80, 120)


class TestTemporalFilter:
    """Test temporal EMA filtering across video frames."""

    def test_first_frame_unsmoothed(self):
        """First frame should equal raw depth (no prior state)."""
        backend = DepthCrafterBackend(temporal_alpha=0.3)
        img = Image.fromarray(np.full((32, 32, 3), 128, dtype=np.uint8))

        result = backend.compute(img)

        # First frame: EMA state is initialized, depth should equal raw
        expected_raw = backend._compute_synthetic_depth(img)
        np.testing.assert_array_almost_equal(result.depth_map, expected_raw, decimal=5)

    def test_temporal_smoothing_across_frames(self):
        """Subsequent frames should be smoothed by EMA."""
        alpha = 0.5
        backend = DepthCrafterBackend(temporal_alpha=alpha)

        # Frame 1: uniform brightness
        frame1 = Image.fromarray(np.full((32, 32, 3), 100, dtype=np.uint8))
        result1 = backend.compute(frame1)

        # Frame 2: different brightness (simulates scene change)
        frame2 = Image.fromarray(np.full((32, 32, 3), 200, dtype=np.uint8))
        result2 = backend.compute(frame2)

        # Frame 2 should be smoothed: alpha*raw + (1-alpha)*prev
        raw_depth2 = backend._compute_synthetic_depth(frame2)

        # result2 should NOT equal raw_depth2 (it's smoothed with frame1)
        # The smoothed result should be between frame1 and raw frame2 values
        assert not np.allclose(result2.depth_map, raw_depth2, atol=1e-3)

        # EMA: smoothed = alpha * raw + (1-alpha) * prev
        expected = alpha * raw_depth2 + (1.0 - alpha) * result1.depth_map
        np.testing.assert_array_almost_equal(result2.depth_map, expected, decimal=5)

    def test_temporal_buffer_tracks_frames(self):
        """Temporal buffer should grow as frames are processed."""
        backend = DepthCrafterBackend()

        assert backend.temporal_buffer_length == 0

        for i in range(5):
            img = Image.fromarray(np.full((16, 16, 3), i * 50, dtype=np.uint8))
            backend.compute(img)

        assert backend.temporal_buffer_length == 5

    def test_temporal_buffer_respects_max_size(self):
        """Buffer should not exceed max_buffer_size."""
        backend = DepthCrafterBackend(max_buffer_size=3)

        for i in range(10):
            img = Image.fromarray(np.full((16, 16, 3), i * 25, dtype=np.uint8))
            backend.compute(img)

        assert backend.temporal_buffer_length == 3

    def test_alpha_zero_full_smoothing(self):
        """alpha=0 means full smoothing: always returns first frame."""
        backend = DepthCrafterBackend(temporal_alpha=0.0)

        frame1 = Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8))
        result1 = backend.compute(frame1)
        first_depth = result1.depth_map.copy()

        frame2 = Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8))
        result2 = backend.compute(frame2)

        # alpha=0 → smoothed = 0*raw + 1*prev = prev
        np.testing.assert_array_almost_equal(result2.depth_map, first_depth, decimal=5)

    def test_alpha_one_no_smoothing(self):
        """alpha=1 means no smoothing: always returns raw frame."""
        backend = DepthCrafterBackend(temporal_alpha=1.0)

        frame1 = Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8))
        backend.compute(frame1)

        frame2 = Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8))
        result2 = backend.compute(frame2)

        raw2 = backend._compute_synthetic_depth(frame2)
        np.testing.assert_array_almost_equal(result2.depth_map, raw2, decimal=5)


class TestTemporalReset:
    """Test temporal state reset between sequences."""

    def test_reset_clears_ema_state(self):
        """reset_temporal_state() clears EMA and buffer."""
        backend = DepthCrafterBackend()

        # Process some frames
        for i in range(3):
            img = Image.fromarray(np.full((16, 16, 3), i * 80, dtype=np.uint8))
            backend.compute(img)

        assert backend.temporal_buffer_length == 3

        # Reset
        backend.reset_temporal_state()

        assert backend.temporal_buffer_length == 0
        assert backend._ema_state is None

    def test_first_frame_after_reset_is_unsmoothed(self):
        """After reset, the next frame should be treated as first frame."""
        backend = DepthCrafterBackend(temporal_alpha=0.3)

        # Process a frame
        frame1 = Image.fromarray(np.full((16, 16, 3), 100, dtype=np.uint8))
        backend.compute(frame1)

        # Reset
        backend.reset_temporal_state()

        # Next frame should be unsmoothed (like first frame)
        frame2 = Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8))
        result = backend.compute(frame2)

        raw2 = backend._compute_synthetic_depth(frame2)
        np.testing.assert_array_almost_equal(result.depth_map, raw2, decimal=5)


class TestCacheKey:
    """Test cache key generation."""

    def test_cache_key_deterministic(self):
        """Same image should produce same cache key."""
        backend = DepthCrafterBackend()
        img = np.random.RandomState(42).randint(0, 255, (32, 32, 3)).astype(np.uint8)

        key1 = backend.get_cache_key(img)
        key2 = backend.get_cache_key(img)

        assert key1 == key2
        assert key1.startswith("depthcrafter-v")

    def test_cache_key_different_for_different_images(self):
        """Different images should produce different cache keys."""
        backend = DepthCrafterBackend()
        img1 = np.zeros((32, 32, 3), dtype=np.uint8)
        img2 = np.ones((32, 32, 3), dtype=np.uint8) * 255

        assert backend.get_cache_key(img1) != backend.get_cache_key(img2)

    def test_cache_key_works_with_pil_image(self):
        """Cache key should work with PIL Image input."""
        backend = DepthCrafterBackend()
        img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))

        key = backend.get_cache_key(img)
        assert isinstance(key, str)
        assert len(key) > 0


class TestRegistryIntegration:
    """Test DepthCrafter registration in backend registry."""

    def test_depthcrafter_registered(self):
        """DepthCrafter should be registered in the backend registry."""
        from transformation_portal.depth.backends.registry import DepthBackendRegistry

        registry = DepthBackendRegistry()
        assert registry.has_backend("depthcrafter")

    def test_depthcrafter_instantiation_via_registry(self):
        """DepthCrafter should be instantiable via registry (no license restrictions)."""
        from transformation_portal.depth.backends.registry import DepthBackendRegistry

        registry = DepthBackendRegistry()
        backend = registry.get_backend("depthcrafter")

        assert backend is not None
        assert backend.name == "depthcrafter"

    def test_depthcrafter_in_list_backends(self):
        """DepthCrafter should appear in list_backends() output."""
        from transformation_portal.depth.backends.registry import DepthBackendRegistry

        registry = DepthBackendRegistry()
        backends = registry.list_backends()

        assert "depthcrafter" in backends
        assert backends["depthcrafter"]["license_type"] == "commercial"


class TestEnsembleDefaultModels:
    """Test that ensemble default models reference depthcrafter (not stub)."""

    def test_default_models_reference_depthcrafter(self):
        """Default ensemble config should reference 'depthcrafter', not 'depthcrafter_stub'."""
        from transformation_portal.depth.backends.ensemble import DepthEnsembleBackend
        from transformation_portal.lux_depth_v3.config import EnhanceConfig

        config = EnhanceConfig(non_commercial_ok=True, accept_research_tools_license=True)
        ensemble = DepthEnsembleBackend(config)

        model_names = [m.name for m in ensemble._models]
        assert "depthcrafter" in model_names
        assert "depthcrafter_stub" not in model_names


# Pytest markers
pytestmark = [
    pytest.mark.unit,
]
