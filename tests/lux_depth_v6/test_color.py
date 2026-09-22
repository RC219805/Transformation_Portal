"""Numerical contracts for explicit grading and independent SDR rendering."""

from dataclasses import replace

import numpy as np
import pytest

from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v6.color import (
    GradeRecipe,
    RenderRecipe,
    grade_master,
    linear_srgb_to_oklab,
    oklab_to_linear_srgb,
    render_master,
)

pytestmark = pytest.mark.unit


def master(pixels, alpha=None):
    return ImageMaster(np.asarray(pixels, np.float32), "a" * 64, 32, alpha)


def test_identity_grade_is_bitwise_for_signed_hdr_and_alpha():
    values = np.array([[[-0.0, -2, 18], [0.02, 0.18, 0.97]], [[4, 0, 2], [1, 1, 1]]], np.float32)
    source = master(values, np.array([[0, 0.5], [1, 1]], np.float32))
    graded, receipt = grade_master(source, GradeRecipe())
    assert graded.pixels.tobytes() == source.pixels.tobytes()
    assert graded.alpha.tobytes() == source.alpha.tobytes()
    assert receipt["changed_pixels"] == 0
    assert receipt["protected_pixels"] == 2
    assert receipt["input_master_sha256"] == source.content_hash()
    assert receipt["output_master_sha256"] == graded.content_hash()


def test_exposure_and_explicit_white_balance_have_no_hidden_normalization():
    source = master([[[0.1, 0.2, 0.3]]])
    result, _ = grade_master(source, GradeRecipe(exposure_stops=1, white_balance=(2, 3, 4)))
    np.testing.assert_allclose(result.pixels, [[[0.4, 1.2, 2.4]]], rtol=1e-6)


def test_contrast_fixes_pivot_and_zero_preserves_chromaticity_and_signed_symmetry():
    source = master(np.repeat(np.array([[-0.36, -0.18, 0, 0.18, 0.36]], np.float32)[..., None], 3, axis=-1))
    result, _ = grade_master(source, GradeRecipe(contrast=2))
    np.testing.assert_allclose(result.pixels[..., 0], [[-0.72, -0.18, 0, 0.18, 0.72]], rtol=2e-6, atol=1e-8)
    chromatic = master([[[0.15, 0.3, 0.6]]])
    result, _ = grade_master(chromatic, GradeRecipe(contrast=1.5))
    np.testing.assert_allclose(result.pixels / chromatic.pixels, np.full((1, 1, 3), result.pixels[0, 0, 0] / 0.15), rtol=2e-6)


def test_oklab_reference_vectors_and_inverse():
    # Independent known linear-sRGB primary coordinates from Ottosson's 2021
    # reference transform, not computed from the implementation under test.
    rgb = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]], np.float64)
    expected = np.array(
        [
            [0.6279553606, 0.2248630611, 0.1258462985],
            [0.8664396115, -0.2338875742, 0.1794984799],
            [0.4520137184, -0.0324569842, -0.3115281477],
            [1, 0, 0],
        ]
    )
    np.testing.assert_allclose(linear_srgb_to_oklab(rgb), expected, atol=4e-8)
    np.testing.assert_allclose(oklab_to_linear_srgb(expected), rgb, atol=2e-7)


def test_chroma_scaling_preserves_oklab_lightness_and_hue():
    source = master([[[0.2, 0.4, 0.7], [0.8, 0.2, 0.1]]])
    original_lab = linear_srgb_to_oklab(source.pixels)
    result, _ = grade_master(source, GradeRecipe(saturation=1.6))
    actual_lab = linear_srgb_to_oklab(result.pixels)
    np.testing.assert_allclose(actual_lab[..., 0], original_lab[..., 0], atol=1e-7)
    np.testing.assert_allclose(actual_lab[..., 1:], original_lab[..., 1:] * 1.6, atol=1e-7)


def test_saturation_zero_is_neutral_and_signed_hdr_is_finite():
    source = master([[[-0.2, 0.1, 2], [4, -2, 0.01]]])
    result, _ = grade_master(source, GradeRecipe(saturation=0))
    assert np.isfinite(result.pixels).all()
    np.testing.assert_allclose(result.pixels[..., 0], result.pixels[..., 1], atol=1e-7)
    np.testing.assert_allclose(result.pixels[..., 1], result.pixels[..., 2], atol=1e-7)


def test_nonopaque_grade_samples_remain_bitwise_protected():
    source = master([[[-0.0, 2, -1], [0.4, 0.5, 0.6], [0.1, 0.2, 0.3]]], np.array([[0, 0.5, 1]], np.float32))
    result, receipt = grade_master(source, GradeRecipe(exposure_stops=2, contrast=1.3, saturation=1.2))
    assert result.pixels[:, :2].tobytes() == source.pixels[:, :2].tobytes()
    assert receipt["protected_pixels"] == 2
    assert receipt["changed_pixels"] == 1


def test_float32_overflow_fails_without_modifying_source():
    source = master([[[np.finfo(np.float32).max] * 3]])
    before = source.pixels.tobytes()
    with pytest.raises(ValueError, match="finite float32"):
        grade_master(source, GradeRecipe(exposure_stops=1))
    assert source.pixels.tobytes() == before


def test_soft_render_preserves_in_gamut_below_shoulder_and_alpha():
    source = master([[[0.2, 0.3, 0.4], [-0.0, 0.01, 0.03]]], np.array([[0.5, 0]], np.float32))
    result, receipt = render_master(source, RenderRecipe(mode="soft_srgb"))
    assert result.pixels.tobytes() == source.pixels.tobytes()
    assert result.alpha.tobytes() == source.alpha.tobytes()
    assert receipt["changed_pixels"] == 0
    assert result.metadata["color_domain"] == "display_linear_srgb"


def test_highlight_shoulder_is_monotonic_preserves_detail_and_neutrality():
    levels = np.linspace(0.8, 2.0, 2000, dtype=np.float32)[None, :, None]
    source = master(np.repeat(levels, 3, axis=-1))
    result, receipt = render_master(source, RenderRecipe(mode="soft_srgb"))
    assert np.all(np.diff(result.pixels[0, :, 0]) > 0)
    np.testing.assert_array_equal(result.pixels[..., 0], result.pixels[..., 1])
    np.testing.assert_array_equal(result.pixels[..., 1], result.pixels[..., 2])
    assert result.pixels.max() < 1
    assert receipt["highlight_mapped_pixels"] == 2000
    assert source.pixels.max() == 2
    legacy, legacy_receipt = render_master(source, RenderRecipe(mode="clip_srgb"))
    assert legacy_receipt["gamut_method"] == "independent_channel_clip"
    assert np.count_nonzero(np.diff(legacy.pixels[0, :, 0]) == 0) > 1000


def test_soft_gamut_contraction_is_bounded_and_preserves_linear_chroma_direction():
    source = master([[[1.5, -0.1, 0.4], [3, 1, 0.5], [-1, -1, -1]]])
    result, receipt = render_master(source, RenderRecipe(mode="soft_srgb"))
    assert np.min(result.pixels) >= 0 and np.max(result.pixels) <= 1
    luminance_weights = np.array([0.2126, 0.7152, 0.0722])
    for index in (0, 1):
        before = source.pixels[0, index].astype(np.float64)
        after = result.pixels[0, index].astype(np.float64)
        original_chroma = before - before @ luminance_weights
        rendered_chroma = after - after @ luminance_weights
        np.testing.assert_allclose(np.cross(original_chroma, rendered_chroma), 0, atol=1e-7)
    np.testing.assert_array_equal(result.pixels[0, 2], 0)
    assert receipt["gamut_compressed_pixels"] >= 1


def test_extreme_hdr_render_is_finite_and_double_render_is_rejected():
    maximum = np.finfo(np.float32).max
    source = master([[[maximum, maximum, maximum], [-maximum, maximum, -maximum]]])
    result, _ = render_master(source, RenderRecipe())
    assert np.isfinite(result.pixels).all()
    assert np.min(result.pixels) >= 0 and np.max(result.pixels) <= 1
    with pytest.raises(ValueError, match="twice"):
        render_master(result, RenderRecipe())
    with pytest.raises(ValueError, match="scene grading"):
        grade_master(result, GradeRecipe())


def test_tiling_does_not_change_pointwise_results():
    pixels = np.random.default_rng(345).uniform(-0.05, 2, (3, 33000, 3)).astype(np.float32)
    source = master(pixels)
    recipe = GradeRecipe(exposure_stops=0.2, contrast=1.1, saturation=0.9)
    large, _ = grade_master(source, recipe)
    small, _ = grade_master(master(pixels[:, 1000:1050]), recipe)
    np.testing.assert_array_equal(large.pixels[:, 1000:1050], small.pixels)
    rendered_large, _ = render_master(large, RenderRecipe())
    rendered_small, _ = render_master(small, RenderRecipe())
    np.testing.assert_array_equal(rendered_large.pixels[:, 1000:1050], rendered_small.pixels)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"exposure_stops": True},
        {"exposure_stops": 10**1000},
        {"exposure_stops": float("nan")},
        {"exposure_stops": 9},
        {"white_balance": (1, False, 1)},
        {"white_balance": (1, 1)},
        {"white_balance": "111"},
        {"white_balance": (1, 0, 1)},
        {"contrast": "1"},
        {"contrast": 0},
        {"saturation": float("inf")},
        {"pivot": 0},
    ],
)
def test_invalid_grade_controls_fail_closed(kwargs):
    with pytest.raises(ValueError):
        GradeRecipe(**kwargs)


@pytest.mark.parametrize(
    "kwargs", [{"mode": []}, {"mode": "aces"}, {"shoulder": True}, {"shoulder": 1}, {"shoulder": float("nan")}]
)
def test_invalid_render_controls_fail_closed(kwargs):
    with pytest.raises(ValueError):
        RenderRecipe(**kwargs)


@pytest.mark.parametrize("recipe", [GradeRecipe(), RenderRecipe()])
def test_recipe_payloads_are_closed_and_round_trip(recipe):
    payload = recipe.to_payload()
    assert type(recipe).from_payload(payload) == recipe
    for bad in (
        dict(payload, extra=True),
        {key: value for key, value in payload.items() if key != "schema"},
        dict(payload, schema="future"),
        None,
    ):
        with pytest.raises(ValueError):
            type(recipe).from_payload(bad)


def test_frozen_recipe_copies_mutable_white_balance():
    gains = [1, 2, 3]
    recipe = GradeRecipe(white_balance=gains)
    gains[0] = 4
    assert recipe.white_balance == (1, 2, 3)
    assert replace(recipe, exposure_stops=2).exposure_stops == 2


def test_log_distributed_signed_hdr_render_challenge_is_finite_and_bounded():
    rng = np.random.default_rng(883)
    pixels = (np.exp2(rng.uniform(-140, 126, (50, 80, 3))) * rng.choice([-1, 1], (50, 80, 3))).astype(np.float32)
    source = master(pixels)
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        for mode in ("perceptual_srgb", "soft_srgb"):
            for shoulder in (0.1, 0.8, 0.95):
                result, _ = render_master(source, RenderRecipe(mode=mode, shoulder=shoulder))
                assert np.isfinite(result.pixels).all()
                assert np.all((result.pixels >= 0) & (result.pixels <= 1))


@pytest.mark.parametrize("exposure", [1, 2, 4, 16])
def test_perceptual_render_saturated_highlights_preserve_named_hue(exposure):
    rng = np.random.default_rng(991)
    rgb = rng.uniform(0.01, 1, (40, 60, 3)).astype(np.float32)
    rgb /= np.max(rgb, axis=-1, keepdims=True)
    rgb *= exposure
    source = master(rgb)
    result, receipt = render_master(source, RenderRecipe())
    before, after = linear_srgb_to_oklab(source.pixels), linear_srgb_to_oklab(result.pixels)
    valid = (np.linalg.norm(before[..., 1:], axis=-1) > 0.01) & (np.linalg.norm(after[..., 1:], axis=-1) > 0.001)
    # Very bright near-neutrals necessarily lose a defined hue as their SDR
    # lightness approaches white. Require measurable chroma on most samples,
    # and report only that population instead of assigning neutral hues.
    assert np.count_nonzero(valid) > rgb.shape[0] * rgb.shape[1] * 0.7
    angle = np.arctan2(before[..., 2], before[..., 1]) - np.arctan2(after[..., 2], after[..., 1])
    degrees = np.rad2deg(np.abs(np.arctan2(np.sin(angle), np.cos(angle))))
    assert degrees[valid].max() < 0.02
    assert receipt["gamut_method"] == "oklab_constant_lightness_hue_chroma_contraction_24"
    assert np.all((result.pixels >= 0) & (result.pixels <= 1))
    if exposure == 16:
        comparison, _ = render_master(source, RenderRecipe(mode="soft_srgb"))
        comparison_lab = linear_srgb_to_oklab(comparison.pixels)
        assert np.count_nonzero(valid) > np.count_nonzero(np.linalg.norm(comparison_lab[..., 1:], axis=-1) > 0.001)


def test_perceptual_render_preserves_neutral_hdr_detail_and_low_range_identity():
    values = np.linspace(0.01, 16, 8000, dtype=np.float32)[None, :, None]
    source = master(np.repeat(values, 3, axis=-1))
    result, _ = render_master(source, RenderRecipe())
    low = values[..., 0] <= 0.8
    assert result.pixels[low].tobytes() == source.pixels[low].tobytes()
    assert np.all(np.diff(result.pixels[0, :, 0]) > 0)
    np.testing.assert_array_equal(result.pixels[..., 0], result.pixels[..., 1])
    np.testing.assert_array_equal(result.pixels[..., 1], result.pixels[..., 2])
    assert result.pixels.max() < 1


@pytest.mark.parametrize("saturation", [0.25, 0.5, 1.5, 2.0])
def test_positive_hdr_chroma_sweep_preserves_oklab_hue_and_lightness(saturation):
    # Signed RGB is accepted for finite arithmetic, but perceptual fidelity is
    # assessed on positive RGB where cone-response cancellation is absent.
    pixels = np.random.default_rng(884).uniform(0, 8, (50, 80, 3)).astype(np.float32)
    source = master(pixels)
    reference_lab = linear_srgb_to_oklab(pixels)
    result, _ = grade_master(source, GradeRecipe(saturation=saturation))
    actual_lab = linear_srgb_to_oklab(result.pixels)
    np.testing.assert_allclose(actual_lab[..., 0], reference_lab[..., 0], atol=1e-7, rtol=0)
    np.testing.assert_allclose(actual_lab[..., 1:], reference_lab[..., 1:] * saturation, atol=2e-7, rtol=0)
