"""Tests for DA2Backend adapter."""

from types import SimpleNamespace

import numpy as np
from PIL import Image

from transformation_portal.depth.backends.da2 import DA2Backend
from transformation_portal.depth.backends.protocol import DepthResult, LicenseType
from transformation_portal.depth.backends.registry import DepthBackendRegistry


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
