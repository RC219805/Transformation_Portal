"""Regression coverage for image processors when optional OpenCV is unavailable."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.depth.processors import depth_aware_denoise, depth_guided_filters

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.fixture(autouse=True)
def without_opencv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(depth_aware_denoise, "CV2_AVAILABLE", False)
    monkeypatch.setattr(depth_guided_filters, "CV2_AVAILABLE", False)


@pytest.mark.parametrize("processor_type", [depth_aware_denoise.DepthAwareDenoise, depth_aware_denoise.FastDepthDenoise])
@pytest.mark.parametrize("dtype", [np.float32, np.uint8])
def test_denoising_fallback_preserves_constant_rgb_channels(processor_type, dtype) -> None:
    image = np.zeros((17, 19, 3), dtype=dtype)
    image[..., 0] = 1.0 if dtype == np.float32 else 255
    original = image.copy()
    depth = np.zeros(image.shape[:2], dtype=np.float32)

    result = processor_type().process(image, depth)

    np.testing.assert_array_equal(result, image)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("processor_type", [depth_aware_denoise.DepthAwareDenoise, depth_aware_denoise.FastDepthDenoise])
def test_denoising_fallback_smooths_spatial_noise_without_color_bleed(processor_type) -> None:
    image = np.zeros((17, 19, 3), dtype=np.float32)
    image[8, 9, 0] = 1.0
    depth = np.zeros(image.shape[:2], dtype=np.float32)

    result = processor_type().process(image, depth)

    assert 0.0 < result[8, 9, 0] < 1.0
    assert result[8, 8, 0] > 0.0
    np.testing.assert_array_equal(result[..., 1:], 0.0)
    assert result.dtype == np.float32


def test_clarity_fallback_preserves_constant_rgb_channels() -> None:
    image = np.full((17, 19, 3), (0.2, 0.5, 0.8), dtype=np.float32)
    depth = np.zeros(image.shape[:2], dtype=np.float32)

    result = depth_guided_filters.DepthGuidedFilters().process(image, depth)

    np.testing.assert_array_equal(result, image)


def test_clarity_fallback_uses_configured_strength() -> None:
    image = np.full((17, 19, 3), 0.4, dtype=np.float32)
    image[8, 9, :] = 0.6
    depth = np.zeros(image.shape[:2], dtype=np.float32)

    unchanged = depth_guided_filters.DepthGuidedFilters(clarity_strength=0.0).process(image, depth)
    weaker = depth_guided_filters.DepthGuidedFilters(clarity_strength=0.25).process(image, depth)
    stronger = depth_guided_filters.DepthGuidedFilters(clarity_strength=0.5).process(image, depth)

    np.testing.assert_array_equal(unchanged, image)
    assert image[8, 9, 0] < weaker[8, 9, 0] < stronger[8, 9, 0]
    assert stronger.dtype == np.float32


def test_clarity_fallback_keeps_normalized_range_and_respects_mask() -> None:
    image = np.zeros((17, 19, 3), dtype=np.float32)
    image[:, 9:, 0] = 1.0
    depth = np.zeros(image.shape[:2], dtype=np.float32)
    mask = np.zeros(image.shape[:2], dtype=bool)
    mask[4:13, 4:15] = True

    result = depth_guided_filters.DepthGuidedFilters(clarity_strength=1.0).process(image, depth, mask=mask)

    assert np.min(result) >= 0.0
    assert np.max(result) <= 1.0
    np.testing.assert_array_equal(result[~mask], image[~mask])
    np.testing.assert_array_equal(result[..., 1:], 0.0)


@pytest.mark.parametrize("use_opencv", [False, True])
@pytest.mark.parametrize("processor_type", [depth_aware_denoise.DepthAwareDenoise, depth_aware_denoise.FastDepthDenoise])
def test_denoising_keeps_float_frames_with_out_of_range_samples(
    processor_type, use_opencv: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """D-10: one float sample above 1.0 must not cast the whole frame to uint8 unscaled (all-black output)."""
    if use_opencv:
        pytest.importorskip("cv2")
        monkeypatch.setattr(depth_aware_denoise, "CV2_AVAILABLE", True)
    image = np.full((24, 32, 3), 0.5, dtype=np.float32)
    image[0, 0, 0] = 1.0001
    hdr = np.full((24, 32, 3), 3.0, dtype=np.float32)
    depth = np.full(image.shape[:2], 0.5, dtype=np.float32)

    result = processor_type().process(image, depth)
    hdr_result = processor_type().process(hdr, depth)

    assert result.dtype == np.float32
    assert abs(float(result[1:, 1:].mean()) - 0.5) < 0.01
    assert float(np.abs(hdr_result - 3.0).max()) < 3.0 / 255 * 2


@pytest.mark.parametrize("use_opencv", [False, True])
@pytest.mark.parametrize("processor_type", [depth_aware_denoise.DepthAwareDenoise, depth_aware_denoise.FastDepthDenoise])
def test_denoising_in_range_float_frames_keep_their_quantisation(
    processor_type, use_opencv: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-range [0, 1] frames keep today's 8-bit quantisation exactly (no output drift for ordinary inputs)."""
    if use_opencv:
        pytest.importorskip("cv2")
        monkeypatch.setattr(depth_aware_denoise, "CV2_AVAILABLE", True)
    image = np.random.default_rng(3).random((24, 32, 3)).astype(np.float32)
    depth = np.full(image.shape[:2], 0.5, dtype=np.float32)

    image_8bit, value_range = depth_aware_denoise._to_8bit(image)

    np.testing.assert_array_equal(image_8bit, (image * 255).astype(np.uint8))
    np.testing.assert_array_equal(
        depth_aware_denoise._from_8bit(image_8bit, value_range), image_8bit.astype(np.float32) / 255.0
    )
    assert processor_type().process(image, depth).dtype == np.float32
