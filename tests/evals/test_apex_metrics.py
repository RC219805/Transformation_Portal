"""Tests for deterministic APEX metrics."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.evals.apex_metrics import compute_apex_metrics

pytestmark = pytest.mark.unit


def _base_image() -> np.ndarray:
    arr = np.zeros((32, 32, 3), dtype=np.uint16)
    arr[..., 0] = 12000
    arr[..., 1] = 16000
    arr[..., 2] = 20000
    return arr


def test_noop_candidate_has_near_zero_deltas():
    reference = _base_image()
    metrics = compute_apex_metrics(
        reference,
        reference.copy(),
        working_color_space="ProPhoto RGB",
        working_transfer_function="linear",
    )

    assert metrics["visible_delta"]["status"] == "ok"
    assert metrics["visible_delta"]["value"] == pytest.approx(0.0)
    assert metrics["p95_delta"]["value"] == pytest.approx(0.0)
    assert metrics["visible_delta"]["comparison"]["working_color_space"] == "ProPhoto RGB"


def test_outside_mask_edit_raises_outside_mask_metric():
    reference = _base_image()
    candidate = reference.copy()
    candidate[:8, :8] = 60000
    mask = np.zeros((32, 32), dtype=bool)
    mask[16:24, 16:24] = True

    metrics = compute_apex_metrics(reference, candidate, mask=mask)

    assert metrics["outside_mask_delta"]["status"] == "ok"
    assert metrics["outside_mask_delta"]["value"] > 0.0


def test_highlight_clipping_raises_clipping_metric():
    reference = _base_image()
    candidate = reference.copy()
    candidate[8:16, 8:16] = np.iinfo(np.uint16).max

    metrics = compute_apex_metrics(reference, candidate)

    assert metrics["highlight_clipping"]["status"] == "ok"
    assert metrics["highlight_clipping"]["value"] > 0.0


def test_edge_expansion_raises_halo_score():
    reference = _base_image()
    candidate = reference.copy()
    candidate[11:21, 11:21] = 50000
    mask = np.zeros((32, 32), dtype=bool)
    mask[12:20, 12:20] = True

    metrics = compute_apex_metrics(reference, candidate, mask=mask)

    assert metrics["seam_halo_score"]["status"] == "ok"
    assert metrics["seam_halo_score"]["value"] > 0.0


def test_posterized_gradient_raises_banding_risk():
    gradient = np.tile(np.linspace(0, 65535, 32, dtype=np.uint16), (32, 1))
    reference = np.stack([gradient, gradient, gradient], axis=2)
    candidate = ((reference // 8192) * 8192).astype(np.uint16)

    metrics = compute_apex_metrics(reference, candidate)

    assert metrics["banding_risk"]["status"] == "ok"
    assert metrics["banding_risk"]["value"] > 0.0


def test_missing_mask_returns_status_not_zero_for_mask_metrics():
    reference = _base_image()
    metrics = compute_apex_metrics(reference, reference.copy(), mask=None)

    assert metrics["outside_mask_delta"]["status"] == "mask_missing"
    assert metrics["outside_mask_delta"]["value"] is None
    assert metrics["seam_halo_score"]["status"] == "not_applicable"


def test_working_space_mismatch_returns_invalid_input():
    reference = _base_image()
    metrics = compute_apex_metrics(
        reference,
        reference.copy(),
        working_color_space="ProPhoto RGB",
        candidate_working_color_space="sRGB",
    )

    assert metrics["visible_delta"]["status"] == "invalid_input"
    assert metrics["visible_delta"]["reason"] == "working_space_mismatch"


def test_dimension_mismatch_fails_closed():
    metrics = compute_apex_metrics(np.zeros((8, 8), dtype=np.uint16), np.zeros((4, 4), dtype=np.uint16))

    assert metrics["visible_delta"]["status"] == "dimension_mismatch"
    assert metrics["visible_delta"]["reason"] == "dimension_mismatch"


def test_delivery_compression_does_not_change_master_metrics():
    reference = _base_image()
    candidate = reference.copy()
    delivery = np.zeros_like(candidate, dtype=np.uint8)

    metrics = compute_apex_metrics(reference, candidate, delivery=delivery)

    assert metrics["visible_delta"]["value"] == pytest.approx(0.0)
    assert metrics["delivery_conversion_delta"]["status"] == "ok"
    assert metrics["delivery_conversion_delta"]["value"] > 0.0


@pytest.mark.parametrize("axis", [0, 1])
def test_4096_to_256_level_quantization_is_detected_before_8bit_reduction(axis):
    ramp = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
    gray = np.tile(ramp, (8, 1))
    if axis == 0:
        gray = gray.T
    reference = np.repeat(gray[..., None], 3, axis=2)
    candidate = np.rint(reference * 255.0) / 255.0

    unchanged = compute_apex_metrics(reference, reference)
    metrics = compute_apex_metrics(reference, candidate)

    assert unchanged["banding_risk"]["value"] == 0.0
    assert metrics["banding_risk"]["value"] > 0.9
    assert metrics["banding_risk"]["comparison"]["eligible_reference_steps"] > 0
    assert metrics["banding_risk"]["comparison"]["algorithm_revision"] == "spatial_v2"


def test_swapped_color_regions_cannot_cancel_in_color_drift():
    reference = np.empty((32, 64, 3), dtype=np.float32)
    reference[:, :32] = [0.8, 0.2, 0.1]
    reference[:, 32:] = [0.1, 0.2, 0.8]
    candidate = reference[:, ::-1].copy()
    np.testing.assert_allclose(reference.mean(axis=(0, 1)), candidate.mean(axis=(0, 1)), atol=1e-5)

    metrics = compute_apex_metrics(reference, candidate)

    assert metrics["color_drift"]["value"] > 0.4
    assert metrics["color_drift"]["comparison"]["method"] == "spatial_centered_rgb_mae"


def test_color_drift_separates_neutral_exposure_delta_from_chroma():
    reference = np.full((32, 32, 3), 0.4, dtype=np.float32)
    metrics = compute_apex_metrics(reference, reference + 0.1)

    assert metrics["visible_delta"]["value"] == pytest.approx(0.1)
    assert metrics["color_drift"]["value"] == pytest.approx(0.0)


def test_clipping_in_one_region_is_not_hidden_by_unclipping_another():
    reference = np.full((128, 128, 3), 0.5, dtype=np.float32)
    reference[:64, :64] = 1.0
    candidate = reference.copy()
    candidate[:64, :64] = 0.5
    candidate[64:, 64:] = 1.0

    metrics = compute_apex_metrics(reference, candidate)

    assert metrics["highlight_clipping"]["value"] == pytest.approx(0.25)
    assert metrics["highlight_clipping"]["comparison"]["peak_native_64px_tile_fraction"] == 1.0


def test_edge_halo_signal_detects_ringing_without_a_material_mask():
    reference = np.full((32, 64, 3), 0.2, dtype=np.float32)
    reference[:, 32:] = 0.7
    candidate = reference.copy()
    candidate[:, 30:32] = 0.05
    candidate[:, 32:34] = 0.9

    unchanged = compute_apex_metrics(reference, reference)
    ringing = compute_apex_metrics(reference, candidate)

    assert unchanged["edge_halo_score"]["value"] == 0.0
    assert ringing["edge_halo_score"]["value"] > 0.1
    assert ringing["seam_halo_score"]["status"] == "not_applicable"


def test_edge_halo_strip_boundaries_do_not_change_the_signal():
    reference = np.full((256, 32, 3), 0.2, dtype=np.float32)
    reference[128:] = 0.7
    candidate = reference.copy()
    candidate[126:128] = 0.05
    candidate[128:130] = 0.9

    strip_boundary = compute_apex_metrics(reference, candidate)["edge_halo_score"]["value"]
    rotated = compute_apex_metrics(reference.transpose(1, 0, 2), candidate.transpose(1, 0, 2))

    assert strip_boundary == pytest.approx(rotated["edge_halo_score"]["value"])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_samples_fail_closed(value):
    reference = np.zeros((8, 8, 3), dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 0, 0] = value
    metrics = compute_apex_metrics(reference, candidate)
    assert all(metric["status"] == "invalid_input" and metric["value"] is None for metric in metrics.values())


@pytest.mark.parametrize("shape", [(0, 8, 3), (8,), (8, 8, 2)])
def test_invalid_image_shapes_return_explicit_status(shape):
    image = np.zeros(shape, dtype=np.float32)
    assert compute_apex_metrics(image, image)["visible_delta"]["status"] == "invalid_input"


def test_flat_and_one_pixel_images_have_defined_zero_banding_and_halo_signals():
    for shape in ((32, 32, 3), (1, 1, 3)):
        image = np.full(shape, 0.5, dtype=np.float32)
        metrics = compute_apex_metrics(image, image)
        assert metrics["banding_risk"]["value"] == 0.0
        assert metrics["edge_halo_score"]["value"] == 0.0
