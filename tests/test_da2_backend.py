"""Tests for DA2Backend adapter."""

import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image

from transformation_portal.depth.backends.da2 import DA2Backend
from transformation_portal.depth.backends.protocol import DepthResult, LicenseType
from transformation_portal.depth.backends.registry import DepthBackendRegistry



pytestmark = pytest.mark.unit

def test_da2_backend_implements_protocol():
    """DA2 backend exposes expected protocol attributes."""
    backend = DA2Backend()
    assert backend.name == "da2"
    assert backend.license_type == LicenseType.COMMERCIAL
    assert backend.requires_checkpoint is False


def test_da2_backend_registry_integration():
    """DA2 backend should be discoverable via registry."""
    registry = DepthBackendRegistry()
    backends = registry.list_backends()
    assert "da2" in backends
    assert backends["da2"]["license_type"] == "commercial"
    assert backends["da2"]["requires_checkpoint"] is False


def test_da2_backend_compute_contract(monkeypatch):
    """DA2 backend compute returns unified DepthResult contract."""
    backend = DA2Backend()
    monkeypatch.setattr(backend, "ensure_available", lambda: None)
    backend._model = SimpleNamespace(
        estimate_depth=lambda _image: {
            "depth": np.ones((32, 32), dtype=np.float32) * 0.5,
            "metadata": {"variant": "SMALL"},
        }
    )

    result = backend.compute(Image.new("RGB", (32, 32), color="white"))

    assert isinstance(result, DepthResult)
    assert result.depth_map.shape == (32, 32)
    assert result.depth_units == "relative"
    assert result.backend_id == "da2"
    assert result.metadata["source_depth_units"] == "relative"
    assert result.metadata["output_depth_units"] == "relative"


def test_da2_backend_cuda_request_without_cuda_falls_back_to_cpu(monkeypatch):
    """Requested CUDA should gracefully normalize to CPU when unavailable."""

    class _Unavailable:
        @staticmethod
        def is_available():
            return False

    fake_torch = SimpleNamespace(
        cuda=_Unavailable(),
        backends=SimpleNamespace(mps=_Unavailable()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = DA2Backend(SimpleNamespace(depth_device="cuda"))
    assert backend._device == "cpu"


def test_da2_backend_cuda_request_is_normalized_to_cpu_model(monkeypatch):
    """DA2 should normalize CUDA requests to CPU to keep backend/device semantics coherent."""
    from transformation_portal.depth.models import depth_anything_v2 as da2_model_module
import pytest

    captured = {}

    class _CudaAvailable:
        @staticmethod
        def is_available():
            return True

    class _MpsUnavailable:
        @staticmethod
        def is_available():
            return False

    fake_torch = SimpleNamespace(
        cuda=_CudaAvailable(),
        backends=SimpleNamespace(mps=_MpsUnavailable()),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class _FakeDepthAnythingV2Model:
        def __init__(self, variant, backend, device):
            captured["variant"] = variant
            captured["backend"] = backend
            captured["device"] = device

    monkeypatch.setattr(da2_model_module, "DepthAnythingV2Model", _FakeDepthAnythingV2Model)

    backend = DA2Backend(SimpleNamespace(depth_device="cuda"))
    monkeypatch.setattr(backend, "ensure_available", lambda: None)
    backend._model = None
    backend._load_model()

    assert backend._device == "cpu"
    assert captured["backend"] == da2_model_module.ModelBackend.PYTORCH_CPU
    assert captured["device"] == "cpu"
