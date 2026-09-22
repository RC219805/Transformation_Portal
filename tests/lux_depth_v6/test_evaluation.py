"""Analytic paired color measurements do not promote photographic quality."""

import numpy as np
import pytest

from transformation_portal.lux_depth_v6.evaluation import evaluate_pairs, numeric_summary

pytestmark = pytest.mark.unit


def test_identical_grayscale_has_zero_error_and_no_defined_hue():
    reference = np.repeat(np.linspace(0, 1, 32, dtype=np.float32)[None, :, None], 3, axis=2)
    report = evaluate_pairs(reference, reference.copy())
    assert report["linear_rgb"] == {"mae": 0, "rmse": 0, "max_error": 0}
    assert report["linear_luminance"] == {"mae": 0, "rmse": 0, "max_error": 0}
    assert report["oklab"]["mean_distance"] == 0
    assert report["oklab"]["hue"]["status"] == "unavailable"
    assert report["style_quality"] == "not_measured"
    assert report["production_acceptance"] == "not_established"


def test_known_linear_offset_and_neutral_lightness_distance():
    reference = np.full((3, 5, 3), 0.125, np.float32)
    prediction = np.full_like(reference, 1.0)
    report = evaluate_pairs(reference, prediction)
    for family in ("linear_rgb", "linear_luminance"):
        assert report[family]["mae"] == pytest.approx(0.875)
        assert report[family]["rmse"] == pytest.approx(0.875)
    assert report["oklab"]["mean_distance"] == pytest.approx(0.5, abs=1e-7)
    assert report["oklab"]["distance"] == "unscaled_euclidean_not_ciede2000"


def test_headroom_is_retained_in_linear_metrics_and_excluded_from_sdr_perception():
    reference = np.array([[[0.5, 0.5, 0.5], [2.0, -0.1, 0.5]]], np.float32)
    prediction = np.clip(reference, 0, 1)
    report = evaluate_pairs(reference, prediction)
    assert report["linear_rgb"]["mae"] == pytest.approx(1.1 / 6)
    assert report["range"]["reference"]["above_one_samples"] == 1
    assert report["range"]["reference"]["below_zero_samples"] == 1
    assert report["oklab"]["coverage"] == 0.5
    assert report["oklab"]["mean_distance"] == 0
    assert "do_not_prove_clipping" in report["range"]["prediction"]["clipping_provenance"]


def test_no_sdr_overlap_returns_unavailable_not_false_perceptual_success():
    report = evaluate_pairs(np.full((2, 3, 3), 2, np.float32), np.ones((2, 3, 3), np.float32))
    assert report["oklab"]["status"] == "unavailable"
    assert report["oklab"]["reason"] == "no_mutually_in_gamut_sdr_pairs"
    assert report["oklab"]["coverage"] == 0
    assert report["linear_rgb"]["mae"] == 1


def test_mask_limits_both_metrics_and_range_accounting():
    reference = np.zeros((2, 3, 3), np.float32)
    prediction = reference.copy()
    prediction[1] = 2
    mask = np.array([[True, True, True], [False, False, False]])
    report = evaluate_pairs(reference, prediction, mask)
    assert report["selected_pixels"] == 3
    assert report["linear_rgb"]["max_error"] == 0
    assert report["range"]["prediction"]["outside_sdr_pixels"] == 0


def test_strided_input_and_small_tiles_preserve_measurement(monkeypatch):
    from transformation_portal.lux_depth_v6 import evaluation

    reference = np.random.default_rng(77).uniform(0, 1, (13, 19, 3)).astype(np.float32)[:, ::2]
    prediction = reference * np.float32(0.8)
    expected = evaluate_pairs(reference, prediction)
    monkeypatch.setattr(evaluation, "MAX_TILE_PIXELS", 7)
    result = evaluate_pairs(reference, prediction)
    for family in ("linear_rgb", "linear_luminance"):
        assert result[family] == pytest.approx(expected[family], abs=1e-12)
    assert result["oklab"]["mean_distance"] == pytest.approx(expected["oklab"]["mean_distance"], abs=1e-12)
    assert result["range"] == expected["range"]


@pytest.mark.parametrize("kind", ["float64", "shape", "nan", "mask_dtype", "empty_mask", "different_grid"])
def test_invalid_color_evidence_fails_closed(kind):
    reference = np.zeros((2, 3, 3), np.float32)
    prediction = reference.copy()
    mask = None
    if kind == "float64":
        prediction = prediction.astype(np.float64)
    elif kind == "shape":
        prediction = prediction[..., 0]
    elif kind == "nan":
        prediction[0, 0, 0] = np.nan
        mask = np.zeros((2, 3), bool)
        mask[1] = True
    elif kind == "mask_dtype":
        mask = np.ones((2, 3), np.float32)
    elif kind == "empty_mask":
        mask = np.zeros((2, 3), bool)
    else:
        prediction = np.zeros((3, 2, 3), np.float32)
    with pytest.raises(ValueError):
        evaluate_pairs(reference, prediction, mask)


def test_pixel_budget_is_checked_before_measurements(monkeypatch):
    from transformation_portal.lux_depth_v6 import evaluation

    monkeypatch.setattr(evaluation, "MAX_PIXELS", 5)
    with pytest.raises(ValueError, match="pixel budget"):
        numeric_summary(np.zeros((2, 3, 3), np.float32))
