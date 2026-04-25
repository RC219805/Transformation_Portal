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
