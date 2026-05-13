"""Unit coverage for depth tools post-processing effects."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.depth import tools

pytestmark = pytest.mark.unit


def _image(size: int = 8) -> np.ndarray:
    x = np.linspace(0.1, 0.9, size, dtype=np.float32)
    base = np.tile(x[None, :], (size, 1))
    return np.dstack([base, np.flipud(base), np.full_like(base, 0.35)]).astype(np.float32)


def _depth(size: int = 8) -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :], (size, 1))


def test_apply_depth_haze_returns_hwc_float_in_range() -> None:
    out = tools.apply_depth_haze(_image(), _depth(), strength=0.25)

    assert out.shape == (8, 8, 3)
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_apply_depth_haze_building_mask_suppresses_far_haze() -> None:
    image = _image()
    depth = _depth()
    building = np.zeros((8, 8), dtype=np.float32)
    building[:, 4:] = 1.0

    unprotected = tools.apply_depth_haze(image, depth, haze_color=(1.0, 1.0, 1.0), strength=0.4)
    protected = tools.apply_depth_haze(
        image,
        depth,
        haze_color=(1.0, 1.0, 1.0),
        strength=0.4,
        building_mask=building,
    )

    assert protected[:, 4:].mean() < unprotected[:, 4:].mean()


def test_apply_depth_clarity_shape_range_and_mask_strength(monkeypatch: pytest.MonkeyPatch) -> None:
    image = _image()
    depth = _depth()
    sky = np.ones((8, 8), dtype=np.float32)
    building = np.ones((8, 8), dtype=np.float32)
    monkeypatch.setattr(tools, "gaussian_blur_float", lambda img, sigma, backend=None: np.zeros_like(img))

    sky_protected = tools.apply_depth_clarity(image, depth, amount=0.5, sky_mask=sky)
    building_boosted = tools.apply_depth_clarity(image, depth, amount=0.5, building_mask=building)

    assert sky_protected.shape == image.shape
    assert 0.0 <= float(sky_protected.min()) <= float(sky_protected.max()) <= 1.0
    assert np.allclose(sky_protected, image)
    assert np.mean(np.abs(building_boosted - image)) > 0.0


def test_apply_depth_dof_fast_mode_returns_hwc_float_in_range() -> None:
    out = tools.apply_depth_dof(_image(), _depth(), edge_preserving=False, quality="fast", clarity=0.0)

    assert out.shape == (8, 8, 3)
    assert out.dtype == np.float32
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_apply_depth_dof_building_mask_reduces_blur(monkeypatch: pytest.MonkeyPatch) -> None:
    image = np.ones((8, 8, 3), dtype=np.float32)
    depth = _depth()
    building = np.ones((8, 8), dtype=np.float32)
    monkeypatch.setattr(tools, "gaussian_blur_float", lambda img, sigma, backend=None: np.zeros_like(img))

    unprotected = tools.apply_depth_dof(
        image,
        depth,
        edge_preserving=False,
        quality="fast",
        clarity=0.0,
        building_mask=None,
    )
    protected = tools.apply_depth_dof(
        image,
        depth,
        edge_preserving=False,
        quality="fast",
        clarity=0.0,
        building_mask=building,
    )

    assert protected.mean() > unprotected.mean()


def test_apply_depth_dof_high_quality_uses_bilateral_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[float] = []

    def fake_bilateral(
        img: np.ndarray,
        depth: np.ndarray,
        sigma_spatial: float,
        sigma_depth: float = 0.08,
        diameter: int | None = None,
    ) -> np.ndarray:
        calls.append(sigma_spatial)
        return np.full_like(img, 0.25)

    monkeypatch.setattr(tools, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(tools, "bilateral_blur_float", fake_bilateral)

    out = tools.apply_depth_dof(_image(), _depth(), quality="high", clarity=0.0)

    assert out.shape == (8, 8, 3)
    assert len(calls) == 2


def test_apply_depth_dof_balanced_quality_skips_bilateral_for_low_complexity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gaussian_calls: list[float] = []

    monkeypatch.setattr(tools, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(tools, "_estimate_image_complexity", lambda img, depth: 0.1)
    monkeypatch.setattr(
        tools,
        "bilateral_blur_float",
        lambda *args, **kwargs: pytest.fail("balanced low-complexity path should skip bilateral"),
    )

    def fake_gaussian(img: np.ndarray, sigma: float, backend=None) -> np.ndarray:
        gaussian_calls.append(sigma)
        return np.full_like(img, 0.2)

    monkeypatch.setattr(tools, "gaussian_blur_float", fake_gaussian)

    out = tools.apply_depth_dof(_image(), _depth(), quality="balanced", clarity=0.0)

    assert out.shape == (8, 8, 3)
    assert len(gaussian_calls) == 2
