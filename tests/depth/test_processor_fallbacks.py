"""Regression coverage for image processors when optional OpenCV is unavailable."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.depth.processors import (
    atmospheric_effects,
    depth_aware_denoise,
    depth_guided_filters,
    zone_tone_mapping,
)

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


def _depth_gradient(height: int = 24, width: int = 32) -> np.ndarray:
    return np.tile(np.linspace(0.0, 1.0, width, dtype=np.float32), (height, 1))


def test_config_override_restores_every_overridden_attribute() -> None:
    """D-11: an override of an attribute outside _get_config_params() must not leak into later calls."""
    processor = atmospheric_effects.AtmosphericEffects(depth_scale=100.0)
    depth = _depth_gradient()
    image = np.random.default_rng(0).random((*depth.shape, 3)).astype(np.float32)

    processor(image, depth, config={"depth_scale": 5.0, "haze_density": 0.5})

    assert processor.depth_scale == 100.0
    assert processor.haze_density == 0.015


def test_config_override_is_restored_when_process_raises() -> None:
    """D-11: a failing call must not leave its config override behind."""
    processor = atmospheric_effects.AtmosphericEffects()
    depth = _depth_gradient()
    image = np.zeros((*depth.shape, 3), dtype=np.float32)
    haze_color = processor.haze_color.copy()

    with pytest.raises(ValueError):
        processor(image, depth[:10], config={"haze_density": 0.9, "haze_color": [0.1, 0.2, 0.3], "depth_scale": 7.0})

    assert processor.haze_density == 0.015
    assert processor.depth_scale == 100.0
    np.testing.assert_array_equal(processor.haze_color, haze_color)


def test_zone_tone_mapping_config_override_is_restored_when_process_raises() -> None:
    processor = zone_tone_mapping.ZoneToneMapping(num_zones=3)
    original_params = processor.zone_params
    depth = _depth_gradient()
    image = np.zeros((*depth.shape, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        processor(image, depth[:10], config={"zone_params": [{"contrast": 2.0}] * 3})

    assert processor.zone_params is original_params


def test_depth_guided_filters_config_override_is_restored_when_process_raises() -> None:
    processor = depth_guided_filters.DepthGuidedFilters(clarity_strength=0.5)
    depth = _depth_gradient()
    image = np.zeros((*depth.shape, 3), dtype=np.float32)

    with pytest.raises(ValueError):
        processor(image, depth[:10], config={"clarity_strength": 0.9})

    assert processor.clarity_strength == 0.5


@pytest.mark.parametrize("via_config", [False, True])
def test_zone_tone_mapping_rejects_fewer_zone_params_than_zones(via_config: bool) -> None:
    """D-11: zip() truncation used to leave the zones without parameters black."""
    two = [{"contrast": 1.0, "saturation": 1.0, "exposure": 0.0, "gamma": 1.0}] * 2
    depth = _depth_gradient()
    image = np.full((*depth.shape, 3), 0.5, dtype=np.float32)

    if via_config:
        processor = zone_tone_mapping.ZoneToneMapping(num_zones=3, method="filmic")
        with pytest.raises(ValueError, match="zone_params has 2 entries but num_zones is 3"):
            processor(image, depth, config={"zone_params": two})
    else:
        processor = zone_tone_mapping.ZoneToneMapping(num_zones=3, zone_params=two, method="filmic")
        with pytest.raises(ValueError, match="zone_params has 2 entries but num_zones is 3"):
            processor(image, depth)


def test_zone_tone_mapping_with_matching_zone_params_covers_every_zone() -> None:
    three = [{"contrast": 1.0, "saturation": 1.0, "exposure": 0.0, "gamma": 1.0}] * 3
    depth = _depth_gradient()
    image = np.full((*depth.shape, 3), 0.5, dtype=np.float32)

    result = zone_tone_mapping.ZoneToneMapping(num_zones=3, zone_params=three, method="filmic")(image, depth)

    # The far third of the frame is tone-mapped like the near third instead of being left black.
    np.testing.assert_allclose(result[:, -8:].mean(), result[:, :8].mean(), rtol=1e-3)


_DENOISE_HAS_OPENCV = depth_aware_denoise.CV2_AVAILABLE


@pytest.mark.parametrize("opencv", [False, True])
@pytest.mark.parametrize(
    "processor_type",
    [atmospheric_effects.AtmosphericEffects, depth_aware_denoise.DepthAwareDenoise, zone_tone_mapping.ZoneToneMapping],
)
def test_rgba_input_processes_rgb_and_passes_alpha_through(
    monkeypatch: pytest.MonkeyPatch, processor_type, opencv: bool
) -> None:
    """D-26: RGBA crashed AtmosphericEffects/DepthAwareDenoise and ZoneToneMapping tone-mapped alpha."""
    if opencv:
        if not _DENOISE_HAS_OPENCV:
            pytest.skip("OpenCV not installed")
        monkeypatch.setattr(depth_aware_denoise, "CV2_AVAILABLE", True)
    depth = _depth_gradient()
    rgb = np.random.default_rng(3).random((*depth.shape, 3)).astype(np.float32)
    alpha = np.linspace(0.1, 0.9, rgb.shape[0] * rgb.shape[1], dtype=np.float32).reshape(*depth.shape, 1)
    rgba = np.concatenate([rgb, alpha], axis=-1)
    original = rgba.copy()

    result = processor_type()(rgba, depth)

    assert result.shape == rgba.shape
    np.testing.assert_array_equal(result[..., 3:], alpha)
    np.testing.assert_array_equal(result[..., :3], processor_type()(rgb, depth))
    np.testing.assert_array_equal(rgba, original)
