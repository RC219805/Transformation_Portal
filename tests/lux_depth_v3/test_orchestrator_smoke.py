"""Smoke coverage for the Lux Depth V3 orchestrator public path."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit


class TestShape2dHelper:
    """Unit tests for the _shape_2d helper function."""

    def test_shape_2d_with_2d_array(self) -> None:
        """_shape_2d extracts correct shape from 2D array."""
        from transformation_portal.lux_depth_v3.orchestrator import _shape_2d

        arr = np.zeros((100, 200), dtype=np.float32)
        result = _shape_2d(arr)
        assert result == (100, 200)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_shape_2d_with_3d_array(self) -> None:
        """_shape_2d extracts first two dimensions from 3D array."""
        from transformation_portal.lux_depth_v3.orchestrator import _shape_2d

        arr = np.zeros((100, 200, 3), dtype=np.float32)
        result = _shape_2d(arr)
        assert result == (100, 200)

    def test_shape_2d_returns_int_types(self) -> None:
        """_shape_2d returns Python ints, not numpy int types."""
        from transformation_portal.lux_depth_v3.orchestrator import _shape_2d

        arr = np.zeros((100, 200), dtype=np.float32)
        height, width = _shape_2d(arr)
        assert type(height) is int
        assert type(width) is int

    def test_shape_2d_rejects_1d_array(self) -> None:
        """_shape_2d raises IndexError for 1D array."""
        from transformation_portal.lux_depth_v3.orchestrator import _shape_2d

        arr = np.zeros((100,), dtype=np.float32)
        with pytest.raises(IndexError, match="at least 2 dimensions"):
            _shape_2d(arr)

    def test_shape_2d_rejects_0d_array(self) -> None:
        """_shape_2d raises IndexError for 0D (scalar) array."""
        from transformation_portal.lux_depth_v3.orchestrator import _shape_2d

        arr = np.array(5.0)
        with pytest.raises(IndexError, match="at least 2 dimensions"):
            _shape_2d(arr)


PIPELINE_PACKAGE = "transformation_portal.lux_depth_v3"
ORCHESTRATOR_MODULE = "transformation_portal.lux_depth_v3.orchestrator"


def _make_depth_result(width: int = 64, height: int = 64):
    """Create a deterministic synthetic depth result for smoke testing."""

    from transformation_portal.depth.backends.protocol import DepthResult

    original_image = np.array(Image.new("RGB", (width, height), color="white"))
    depth = np.linspace(0.0, 1.0, width * height, dtype=np.float32).reshape(height, width)
    return DepthResult(
        depth_map=depth,
        original_image=original_image,
        metadata={},
        depth_units="relative",
        backend_id="da3",
        device="cpu",
    )


def test_orchestrator_smoke_small_happy_path(tmp_path: Path) -> None:
    """Lazy public API can initialize and process one tiny image with stubbed depth."""

    lux_depth_v3 = importlib.import_module(PIPELINE_PACKAGE)
    orchestrator_module = importlib.import_module(ORCHESTRATOR_MODULE)
    image_input_module = importlib.import_module("transformation_portal.lux_depth_v3.input_manager")

    test_image = tmp_path / "orchestrator_smoke.png"
    Image.new("RGB", (64, 64), color="white").save(test_image)

    config = lux_depth_v3.EnhanceConfig(
        depth_backend="da3",
        depth_device="cpu",
        enable_v2=False,
        enable_materials_v3=False,
    )

    backend = Mock()
    backend.name = "da3"
    backend.license_type = Mock(value="commercial")
    backend.ensure_available.return_value = None
    backend.compute.return_value = _make_depth_result()

    registry = Mock()
    registry.get_backend.return_value = backend

    with patch.object(orchestrator_module, "DepthBackendRegistry", return_value=registry):
        orchestrator = lux_depth_v3.EnhanceOrchestrator(config, tmp_path)
        orchestrator.postprocessor = Mock(process=lambda result: result)
        result = orchestrator.enhance_image(
            image_input_module.ImageInput(path=test_image),
            input_root=tmp_path,
        )

    assert lux_depth_v3.EnhanceOrchestrator is orchestrator_module.EnhanceOrchestrator
    assert result["status"] == "ok"
    assert result["backend"] == "da3"
    assert result["image"] == str(test_image)
    assert result["manifest"] is not None
    assert result["depth_path"] is not None
    assert Path(result["manifest"]).exists()
    assert Path(result["depth_path"]).exists()
