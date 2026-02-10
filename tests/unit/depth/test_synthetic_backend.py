"""Tests for synthetic depth backend (CI-friendly, no ML deps).

The synthetic backend provides deterministic depth estimation without requiring
ML frameworks, making it ideal for CI environments and fast tests.
"""

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth.backends.protocol import LicenseType
from transformation_portal.depth.backends.registry import DepthBackendRegistry
from transformation_portal.depth.backends.synthetic import SyntheticDepthBackend
from transformation_portal.lux_depth_v3.config import EnhanceConfig


class TestSyntheticBackend:
    """Test suite for SyntheticDepthBackend."""

    def test_basic_properties(self):
        """Test basic backend instantiation and properties."""
        backend = SyntheticDepthBackend()
        assert backend.name == "synthetic"
        assert backend.license_type == LicenseType.COMMERCIAL
        assert backend.requires_checkpoint is False

    def test_ensure_available(self):
        """Test that synthetic backend is always available."""
        backend = SyntheticDepthBackend()
        backend.ensure_available()  # Should not raise

    def test_required_packages(self):
        """Test that synthetic backend has no additional package requirements."""
        packages = SyntheticDepthBackend.required_packages()
        assert packages == []

    def test_compute_with_pil_image(self):
        """Test synthetic depth computation with PIL Image."""
        backend = SyntheticDepthBackend()
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))

        result = backend.compute(img)

        assert result.depth_map.shape == (100, 100)
        assert result.depth_map.dtype == np.float32
        assert result.backend_id == "synthetic"
        assert result.metadata["synthetic"] is True
        assert result.metadata["method"] == "luminance"
        assert 0.0 <= result.depth_map.min() <= 1.0
        assert 0.0 <= result.depth_map.max() <= 1.0

    def test_compute_with_numpy_array(self):
        """Test synthetic depth computation with numpy array."""
        backend = SyntheticDepthBackend()
        img_array = np.full((100, 100, 3), 128, dtype=np.uint8)

        result = backend.compute(img_array)

        assert result.depth_map.shape == (100, 100)
        assert result.backend_id == "synthetic"

    def test_deterministic_output(self):
        """Test that synthetic backend produces deterministic output."""
        backend = SyntheticDepthBackend()
        img = Image.new("RGB", (50, 50), color=(100, 150, 200))

        result1 = backend.compute(img)
        result2 = backend.compute(img)

        np.testing.assert_array_equal(result1.depth_map, result2.depth_map)

    def test_luminance_based_depth(self):
        """Test that depth is based on luminance (brighter = closer)."""
        backend = SyntheticDepthBackend()

        # Create images with different brightness
        dark_img = Image.new("RGB", (50, 50), color=(50, 50, 50))
        bright_img = Image.new("RGB", (50, 50), color=(200, 200, 200))

        dark_result = backend.compute(dark_img)
        bright_result = backend.compute(bright_img)

        # Darker image should have larger depth values (farther away)
        # Brighter image should have smaller depth values (closer)
        assert dark_result.depth_map.mean() > bright_result.depth_map.mean()

    def test_registry_integration(self):
        """Test that synthetic backend is available in registry."""
        registry = DepthBackendRegistry()

        assert registry.has_backend("synthetic")
        backend = registry.get_backend("synthetic")
        assert backend.name == "synthetic"

    def test_cache_key_deterministic(self):
        """Test cache key generation is deterministic."""
        backend = SyntheticDepthBackend()
        img = Image.new("RGB", (50, 50), color=(100, 150, 200))

        key1 = backend.get_cache_key(img)
        key2 = backend.get_cache_key(img)

        assert key1 == key2  # Deterministic
        assert "synthetic" in key1
        assert len(key1) > 20  # Should include hash

    def test_cache_key_changes_with_image(self):
        """Test cache key changes when image content changes."""
        backend = SyntheticDepthBackend()
        img1 = Image.new("RGB", (50, 50), color=(100, 100, 100))
        img2 = Image.new("RGB", (50, 50), color=(200, 200, 200))

        key1 = backend.get_cache_key(img1)
        key2 = backend.get_cache_key(img2)

        assert key1 != key2

    def test_device_parameter_ignored(self):
        """Test that device parameter is accepted but ignored."""
        backend = SyntheticDepthBackend()
        img = Image.new("RGB", (50, 50), color=(128, 128, 128))

        # Should work with any device string (synthetic is device-agnostic)
        result_cpu = backend.compute(img, device="cpu")
        result_mps = backend.compute(img, device="mps")
        result_cuda = backend.compute(img, device="cuda")

        # All should produce the same deterministic output
        np.testing.assert_array_equal(result_cpu.depth_map, result_mps.depth_map)
        np.testing.assert_array_equal(result_cpu.depth_map, result_cuda.depth_map)

    def test_repr(self):
        """Test string representation."""
        backend = SyntheticDepthBackend()
        repr_str = repr(backend)
        assert "SyntheticDepthBackend" in repr_str
        assert "synthetic" in repr_str


class TestSyntheticBackendFallbackPolicy:
    """Test that synthetic backend is fallback-only, never default."""

    def test_default_config_is_not_synthetic(self):
        """Test that default config does not select synthetic backend."""
        config = EnhanceConfig()
        # Config default is None, which orchestrator converts to "da3"
        effective_default = config.depth_backend or "da3"
        assert effective_default == "da3", "Default should be DA3, not synthetic"
        assert effective_default != "synthetic", "Synthetic must not be default"

    def test_synthetic_backend_metadata_is_traceable(self):
        """Test that synthetic backend usage is traceable in metadata."""
        backend = SyntheticDepthBackend()
        img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        result = backend.compute(img)

        # Metadata must clearly flag synthetic usage
        assert result.backend_id == "synthetic"
        assert result.metadata["synthetic"] is True
        assert result.metadata["backend"] == "synthetic"
        assert "method" in result.metadata  # Should document method used
