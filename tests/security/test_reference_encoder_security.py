"""Security tests for style feature cache deserialization."""

from __future__ import annotations

import importlib
import pickle
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", reason="torch required for reference encoder security tests")

from transformation_portal.style_transfer.reference_encoder import ReferenceImageEncoder


class _MaliciousPayload:
    """Payload used to verify restricted pickle loading."""

    def __init__(self, marker_path: Path):
        self.marker_path = marker_path

    def __reduce__(self):
        expr = f"__import__('pathlib').Path({str(self.marker_path)!r}).write_text('owned')"
        return (eval, (expr,))


def _make_encoder() -> ReferenceImageEncoder:
    """Construct encoder without loading heavy model dependencies."""
    encoder = ReferenceImageEncoder.__new__(ReferenceImageEncoder)
    encoder.device = "cpu"
    return encoder


def test_load_features_roundtrip(tmp_path: Path):
    """Valid feature cache should round-trip correctly."""
    encoder = _make_encoder()
    features = torch.randn(2, 8, dtype=torch.float32)
    path = tmp_path / "features.pkl"

    encoder.save_features(features, path, metadata={"source": "test"})
    loaded_features, metadata = encoder.load_features(path)

    assert loaded_features.shape == features.shape
    assert torch.allclose(loaded_features.cpu(), features.cpu(), atol=1e-6)
    assert metadata == {"source": "test"}


def test_load_features_blocks_malicious_pickle(tmp_path: Path):
    """Malicious pickle payloads should be rejected without side effects."""
    encoder = _make_encoder()
    marker = tmp_path / "owned.txt"
    path = tmp_path / "malicious.pkl"

    payload = {
        "features": _MaliciousPayload(marker),
        "padding": "x" * 1024,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)

    with pytest.raises(ValueError, match="Unsafe or invalid feature cache file"):
        encoder.load_features(path)

    assert not marker.exists()


def test_load_features_requires_numpy_array(tmp_path: Path):
    """Cache files with non-array features should be rejected."""
    encoder = _make_encoder()
    path = tmp_path / "invalid.pkl"

    with open(path, "wb") as f:
        pickle.dump({"features": "not-an-array"}, f)

    with pytest.raises(ValueError, match="missing ndarray 'features'"):
        encoder.load_features(path)


def test_reference_encoder_import_does_not_eagerly_load_ip_adapter(monkeypatch):
    """Reference encoder imports should not require optional FLUX adapter deps."""
    module_names = [
        "transformation_portal.style_transfer",
        "transformation_portal.style_transfer.ip_adapter",
        "transformation_portal.style_transfer.reference_encoder",
    ]
    for module_name in module_names:
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    style_transfer_package = importlib.import_module("transformation_portal.style_transfer")
    assert "transformation_portal.style_transfer.ip_adapter" not in sys.modules

    reference_encoder_module = importlib.import_module("transformation_portal.style_transfer.reference_encoder")
    assert "transformation_portal.style_transfer.ip_adapter" not in sys.modules
    assert style_transfer_package.ReferenceImageEncoder is reference_encoder_module.ReferenceImageEncoder


pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
    pytest.mark.security,
    pytest.mark.ml,
]
