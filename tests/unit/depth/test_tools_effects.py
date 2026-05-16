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


def test_apply_depth_dof_balanced_quality_boosts_bilateral_for_high_complexity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Covers the high-complexity adaptive branch where bilateral_sigma_depth
    # is scaled rather than skipped.
    seen_sigmas: list[float] = []

    monkeypatch.setattr(tools, "_CV2_AVAILABLE", True)
    monkeypatch.setattr(tools, "_estimate_image_complexity", lambda img, depth: 0.8)

    def fake_bilateral(img, depth, sigma_spatial, sigma_depth=0.08, diameter=None):
        seen_sigmas.append(sigma_depth)
        return np.full_like(img, 0.3)

    monkeypatch.setattr(tools, "bilateral_blur_float", fake_bilateral)

    out = tools.apply_depth_dof(
        _image(),
        _depth(),
        quality="balanced",
        clarity=0.0,
        bilateral_sigma_depth=0.10,
    )

    assert out.shape == (8, 8, 3)
    assert len(seen_sigmas) == 2
    # 0.10 * (0.5 + 0.8 * 0.5) = 0.09; second call uses *0.75 of that
    assert seen_sigmas[0] == pytest.approx(0.09)
    assert seen_sigmas[1] == pytest.approx(0.09 * 0.75)


def test_apply_depth_dof_clarity_runs_unsharp_mask_branch() -> None:
    # clarity > 1e-6 takes the unsharp-mask branch and should alter output.
    with_clarity = tools.apply_depth_dof(
        _image(),
        _depth(),
        edge_preserving=False,
        quality="fast",
        clarity=0.5,
    )
    without_clarity = tools.apply_depth_dof(
        _image(),
        _depth(),
        edge_preserving=False,
        quality="fast",
        clarity=0.0,
    )

    assert with_clarity.shape == without_clarity.shape
    assert with_clarity.dtype == np.float32
    assert np.isfinite(with_clarity).all()
    assert not np.array_equal(with_clarity, without_clarity)


def test_estimate_image_complexity_returns_bounded_float() -> None:
    # Direct exercise — existing tests only mock this helper.
    img = _image()
    flat = np.zeros_like(_depth())
    sharp = _depth()

    flat_score = tools._estimate_image_complexity(img, flat)
    sharp_score = tools._estimate_image_complexity(img, sharp)

    assert 0.0 <= flat_score <= 1.0
    assert 0.0 <= sharp_score <= 1.0
    assert sharp_score >= flat_score


def test_gaussian_blur_float_short_circuits_below_threshold() -> None:
    # sigma <= 0.5 returns the input untouched.
    img = _image()
    out = tools.gaussian_blur_float(img, sigma=0.3)
    assert out is img


def test_gaussian_blur_float_scipy_handles_2d_input(monkeypatch: pytest.MonkeyPatch) -> None:
    # Force the SciPy 2D code path by passing a 2D depth-like array.
    pytest.importorskip("scipy.ndimage")
    assert tools.gaussian_filter is not None
    monkeypatch.setattr(tools, "_SCIPY_AVAILABLE", True)
    arr = _depth()

    out = tools.gaussian_blur_float(arr, sigma=1.0, backend="scipy")

    assert out.shape == arr.shape
    assert out.dtype == np.float32


def test_gaussian_blur_float_cv2_backend_returns_normalized_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise the cv2 backend branch via explicit selection.
    pytest.importorskip("cv2")
    monkeypatch.setattr(tools, "_CV2_AVAILABLE", True)
    img = _image()

    out = tools.gaussian_blur_float(img, sigma=1.5, backend="cv2")

    assert out.shape == img.shape
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_gaussian_blur_float_pil_fallback_handles_2d(monkeypatch: pytest.MonkeyPatch) -> None:
    # PIL fallback with 2D input.
    monkeypatch.setattr(tools, "_SCIPY_AVAILABLE", False)
    monkeypatch.setattr(tools, "_CV2_AVAILABLE", False)
    arr = _depth()

    out = tools.gaussian_blur_float(arr, sigma=1.0)

    assert out.shape == arr.shape
    assert out.dtype == np.float32


def test_gaussian_blur_float_pil_fallback_handles_rgb(monkeypatch: pytest.MonkeyPatch) -> None:
    # PIL fallback with 3D input.
    monkeypatch.setattr(tools, "_SCIPY_AVAILABLE", False)
    monkeypatch.setattr(tools, "_CV2_AVAILABLE", False)

    out = tools.gaussian_blur_float(_image(), sigma=1.0)

    assert out.shape == (8, 8, 3)
    assert out.dtype == np.float32


def test_bilateral_blur_float_falls_back_to_gaussian_without_cv2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without cv2, bilateral degrades to gaussian.
    monkeypatch.setattr(tools, "_CV2_AVAILABLE", False)
    called: dict[str, float] = {}

    def fake_gaussian(img, sigma, backend=None):
        called["sigma"] = sigma
        return np.full_like(img, 0.4)

    monkeypatch.setattr(tools, "gaussian_blur_float", fake_gaussian)

    out = tools.bilateral_blur_float(_image(), _depth(), sigma_spatial=2.0)

    assert called["sigma"] == 2.0
    assert out.shape == (8, 8, 3)


def test_bilateral_blur_float_cv2_path_handles_rgb_and_2d(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Exercise the per-channel loop and the 2D branch. cv2 must be available
    # for both.
    pytest.importorskip("cv2")
    monkeypatch.setattr(tools, "_CV2_AVAILABLE", True)

    rgb_out = tools.bilateral_blur_float(_image(), _depth(), sigma_spatial=1.5)
    gray_out = tools.bilateral_blur_float(_depth(), _depth(), sigma_spatial=1.5)

    assert rgb_out.shape == (8, 8, 3)
    assert gray_out.shape == (8, 8)
    assert 0.0 <= float(rgb_out.min()) <= float(rgb_out.max()) <= 1.0


def test_apply_depth_haze_handles_none_masks() -> None:
    # sky_mask=None and building_mask=None take the zero-mask shortcuts
    # before the structurally distinct masked path.
    out = tools.apply_depth_haze(_image(), _depth(), sky_mask=None, building_mask=None)
    assert out.shape == (8, 8, 3)
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0


def test_apply_depth_haze_handles_empty_masks() -> None:
    # The `.size == 0` arm of the same mask-normalization conditional.
    empty = np.zeros((0,), dtype=np.float32)
    out = tools.apply_depth_haze(_image(), _depth(), sky_mask=empty, building_mask=empty)
    assert out.shape == (8, 8, 3)


def test_apply_depth_clarity_handles_flat_depth() -> None:
    # depth_range collapses to the 1e-6 floor; the function must still produce
    # a valid float image rather than dividing by zero.
    flat = np.full((8, 8), 0.5, dtype=np.float32)
    out = tools.apply_depth_clarity(_image(), flat, amount=0.3)

    assert out.shape == (8, 8, 3)
    assert 0.0 <= float(out.min()) <= float(out.max()) <= 1.0
