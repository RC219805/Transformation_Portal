"""Regression tests for the OOM fallback of ``safe_depth_estimation`` (D-26)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from transformation_portal.depth.models import depth_anything_v2 as da2_model

pytestmark = pytest.mark.unit


class _OOMOnceModel:
    """Raises an OOM error on the first call, then returns a depth map at the input size."""

    def __init__(self) -> None:
        self.calls = 0
        self.retry_sizes: list[tuple[int, int]] = []

    def estimate_depth(self, image):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("MPS backend out of memory")
        assert isinstance(image, Image.Image)
        self.retry_sizes.append(image.size)
        width, height = image.size
        return {"depth": np.zeros((height, width), np.float32), "metadata": {}}


@pytest.fixture
def fake_resize(monkeypatch: pytest.MonkeyPatch) -> list[tuple[int, int]]:
    """Stand in for scikit-image's resize, recording the requested output shapes."""
    requested: list[tuple[int, int]] = []

    def _resize(depth, output_shape, **_kwargs):
        requested.append(tuple(output_shape))
        return np.zeros(tuple(output_shape), np.float32)

    monkeypatch.setattr(da2_model, "SKIMAGE_AVAILABLE", True)
    monkeypatch.setattr(da2_model, "resize", _resize, raising=False)
    return requested


def test_pil_fallback_leaves_the_callers_image_untouched(fake_resize: list[tuple[int, int]]) -> None:
    image = Image.new("RGB", (2000, 1000))
    model = _OOMOnceModel()

    result = da2_model.safe_depth_estimation(image, model, fallback_size=512)

    assert image.size == (2000, 1000)
    assert model.retry_sizes == [(512, 256)]
    assert fake_resize == [(1000, 2000)]
    assert result["depth"].shape == (1000, 2000)
    assert result["metadata"]["fallback_resolution"] == 512


def test_float_array_fallback_is_downscaled_like_estimate_depth(fake_resize: list[tuple[int, int]]) -> None:
    image = np.full((64, 96, 3), 0.5, np.float32)
    model = _OOMOnceModel()

    result = da2_model.safe_depth_estimation(image, model, fallback_size=32)

    assert model.retry_sizes == [(32, 21)]
    assert fake_resize == [(64, 96)]
    assert result["depth"].shape == (64, 96)
    np.testing.assert_array_equal(image, np.full((64, 96, 3), 0.5, np.float32))


def test_non_oom_runtime_errors_are_not_retried() -> None:
    class _Broken:
        def estimate_depth(self, image):
            raise RuntimeError("weights corrupt")

    with pytest.raises(RuntimeError, match="weights corrupt"):
        da2_model.safe_depth_estimation(Image.new("RGB", (8, 8)), _Broken())
