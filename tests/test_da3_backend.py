"""Tests for DA3Backend adapter.

Tests that DA3Backend implements the DepthBackend protocol correctly
and integrates with the registry.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

# Import availability helpers from conftest
from tests.conftest import can_run_da3_compute
from transformation_portal.depth.backends.da3 import DA3Backend
from transformation_portal.depth.backends.protocol import DepthResult, LicenseType
from transformation_portal.depth.backends.registry import DepthBackendRegistry

# Mark all tests in this module as ML tier (require torch + transformers)
pytestmark = pytest.mark.ml


def _install_fake_depth_anything3(monkeypatch):
    """Install a lightweight fake depth_anything_3 module for device smoke tests."""
    import types

    class FakeDepthAnything3:
        def __init__(self):
            self.loaded_device = None

        @classmethod
        def from_pretrained(cls, model_id):
            del model_id
            return cls()

        def to(self, device):
            dev = str(device)
            if "cuda" in dev:
                raise RuntimeError("Unexpected CUDA path in DA3 smoke test")
            self.loaded_device = dev
            return self

        def eval(self):
            return self

        def inference(self, images):
            del images
            if self.loaded_device is None:
                raise RuntimeError("Model device not set before inference")
            if "cuda" in str(self.loaded_device):
                raise RuntimeError("Torch not compiled with CUDA enabled")
            depth = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape((1, 64, 64))
            return SimpleNamespace(depth=depth)

    fake_pkg = types.ModuleType("depth_anything_3")
    fake_api = types.ModuleType("depth_anything_3.api")
    fake_pkg.DepthAnything3 = FakeDepthAnything3
    fake_api.DepthAnything3 = FakeDepthAnything3

    monkeypatch.setitem(sys.modules, "depth_anything_3", fake_pkg)
    monkeypatch.setitem(sys.modules, "depth_anything_3.api", fake_api)


def test_da3_backend_implements_protocol():
    """DA3Backend implements DepthBackend protocol."""
    backend = DA3Backend()
    assert backend.name == "da3"
    assert backend.license_type == LicenseType.COMMERCIAL
    assert backend.requires_checkpoint is False


def test_da3_backend_availability():
    """DA3Backend has ensure_available() method.

    Verifies the method exists and is callable.
    Actual error handling is tested in test_da3_backend_availability_missing_transformers.
    """
    backend = DA3Backend()

    # Verify method exists
    assert hasattr(backend, "ensure_available")
    assert callable(backend.ensure_available)


def test_da3_backend_availability_missing_transformers(monkeypatch):
    """DA3Backend.ensure_available() detects missing transformers dependency.

    Uses monkeypatch to manage sys.modules, simulating missing dependency.
    """
    # Use monkeypatch to safely modify sys.modules
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    monkeypatch.setitem(sys.modules, "transformers", None)

    backend = DA3Backend()

    # This should raise ImportError when ensure_available tries to import transformers
    with pytest.raises(ImportError, match="transformers"):
        backend.ensure_available()


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


def test_da3_backend_unit_contract_metadata(monkeypatch):
    """DA3 adapter should expose source/output unit semantics in metadata."""
    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    config = EnhanceConfig(depth_device="cpu")
    backend = DA3Backend(config)

    # Avoid dependency checks and heavy model loading.
    monkeypatch.setattr(backend, "ensure_available", lambda: None)
    backend._engine = SimpleNamespace(
        predict=lambda _image: SimpleNamespace(
            depth_map=np.ones((32, 32), dtype=np.float32),
            metadata={"resolved_model_id": "depth-anything/DA3NESTED-GIANT-LARGE-1.1"},
        )
    )

    result = backend.compute(Image.new("RGB", (32, 32), color="white"))

    assert result.depth_units == "relative"
    assert result.metadata["source_depth_units"] == "meters"
    assert result.metadata["output_depth_units"] == "relative"
    assert result.metadata["output_normalization"] == "minmax_0_1_per_image"
    assert any("normalized to relative" in warning for warning in result.warnings)


def test_da3_backend_smoke_cpu_no_hidden_cuda(monkeypatch):
    """CPU DA3 inference path should not invoke CUDA implicitly."""
    pytest.importorskip("torch")
    import transformation_portal.lux_depth_v3.inference as inference_module
    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    _install_fake_depth_anything3(monkeypatch)
    monkeypatch.setattr(DA3Backend, "ensure_available", lambda self: None)
    monkeypatch.setattr(inference_module, "TRANSFORMERS_AVAILABLE", True)

    backend = DA3Backend(EnhanceConfig(depth_device="cpu"))
    result = backend.compute(Image.new("RGB", (64, 64), color="white"))

    assert result.device == "cpu"
    assert result.depth_map.shape == (64, 64)


def test_da3_backend_smoke_mps_no_hidden_cuda(monkeypatch):
    """MPS DA3 inference path should not invoke CUDA implicitly."""
    torch = pytest.importorskip("torch")
    import transformation_portal.lux_depth_v3.inference as inference_module
    from transformation_portal.lux_depth_v3.config import EnhanceConfig

    _install_fake_depth_anything3(monkeypatch)
    monkeypatch.setattr(DA3Backend, "ensure_available", lambda self: None)
    monkeypatch.setattr(inference_module, "TRANSFORMERS_AVAILABLE", True)

    if hasattr(torch.backends, "mps"):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    else:
        monkeypatch.setattr(torch.backends, "mps", SimpleNamespace(is_available=lambda: True), raising=False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    backend = DA3Backend(EnhanceConfig(depth_device="mps"))
    result = backend.compute(Image.new("RGB", (64, 64), color="white"))

    assert result.device == "mps"
    assert result.depth_map.shape == (64, 64)
