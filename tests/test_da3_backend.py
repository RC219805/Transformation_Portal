"""Tests for DA3Backend adapter.

Tests that DA3Backend implements the DepthBackend protocol correctly
and integrates with the registry.
"""

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth.backends.da3 import DA3Backend
from transformation_portal.depth.backends.protocol import DepthResult, LicenseType
from transformation_portal.depth.backends.registry import DepthBackendRegistry

# Import availability helpers from conftest
from tests.conftest import can_run_da3_compute

# Mark all tests in this module as ML tier (require torch + transformers)
pytestmark = pytest.mark.ml


def test_da3_backend_implements_protocol():
    """DA3Backend implements DepthBackend protocol."""
    backend = DA3Backend()
    assert backend.name == "da3"
    assert backend.license_type == LicenseType.COMMERCIAL
    assert backend.requires_checkpoint is False


def test_da3_backend_availability():
    """DA3Backend.ensure_available() checks dependencies.

    This test verifies the availability check mechanism works,
    but doesn't require actual transformers/torch to be installed.
    It only checks that the method exists and is callable.
    """
    backend = DA3Backend()
    # Just verify the method exists - don't call it (requires transformers)
    assert hasattr(backend, "ensure_available")
    assert callable(backend.ensure_available)


@pytest.mark.skipif(
    not can_run_da3_compute(),
    reason="DA3 compute requires depth_anything_3 + transformers + online mode",
)
def test_da3_backend_compute():
    """DA3Backend.compute() returns DepthResult."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    config = EnhanceConfig(depth_device="cpu")
    backend = DA3Backend(config)

    # Create test image
    image = Image.new("RGB", (64, 64), color="white")

    # Run inference
    result = backend.compute(image)

    assert isinstance(result, DepthResult)
    # DA3 may resize the input, so check that we got a depth map
    assert len(result.depth_map.shape) == 2  # 2D depth map
    assert result.depth_map.dtype == np.float32
    assert result.depth_units == "relative"
    assert result.focal_length_px is None  # DA3 doesn't provide focal length
    assert result.backend_id == "da3"


@pytest.mark.skipif(
    not can_run_da3_compute(),
    reason="DA3 compute requires depth_anything_3 + transformers + online mode",
)
def test_da3_backend_compute_numpy():
    """DA3Backend.compute() accepts numpy arrays."""
    backend = DA3Backend()

    # Create test image as numpy array
    image_array = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

    # Run inference
    result = backend.compute(image_array)

    assert isinstance(result, DepthResult)
    # DA3 may resize the input
    assert len(result.depth_map.shape) == 2  # 2D depth map


def test_da3_backend_cache_key():
    """DA3Backend generates consistent cache keys."""
    backend = DA3Backend()

    image = Image.new("RGB", (64, 64))

    key1 = backend.get_cache_key(image)
    key2 = backend.get_cache_key(image)

    assert key1 == key2
    assert key1.startswith("da3_")


def test_da3_backend_registry_integration():
    """DA3Backend is registered in DepthBackendRegistry."""
    registry = DepthBackendRegistry()

    backends = registry.list_backends()
    assert "da3" in backends
    assert backends["da3"]["license_type"] == "commercial"
    assert backends["da3"]["requires_checkpoint"] is False


@pytest.mark.skipif(
    not can_run_da3_compute(),
    reason="DA3 compute requires depth_anything_3 + transformers + online mode",
)
def test_da3_backend_via_registry():
    """DA3Backend can be instantiated via registry."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    config = EnhanceConfig(depth_device="cpu")
    registry = DepthBackendRegistry()

    backend = registry.get_backend("da3", config)

    assert isinstance(backend, DA3Backend)
    assert backend.name == "da3"


@pytest.mark.skipif(
    not can_run_da3_compute(),
    reason="DA3 compute requires depth_anything_3 + transformers + online mode",
)
def test_da3_backend_device_override():
    """DA3Backend respects device parameter in compute()."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    config = EnhanceConfig(depth_device="cpu")
    backend = DA3Backend(config)

    image = Image.new("RGB", (64, 64))

    # Should not raise even if device override is specified
    result = backend.compute(image, device="cpu")
    assert result.device == "cpu"
