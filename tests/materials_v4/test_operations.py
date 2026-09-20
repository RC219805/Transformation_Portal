"""Numerical contracts for Materials V4 baseline-derived delta kernels."""

import numpy as np
import pytest

from transformation_portal.materials_v4.operations import OPERATION_SPECS, operation_contract_hash, operation_delta

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("operation", tuple(OPERATION_SPECS))
def test_operations_are_neutral_and_leave_baseline_unchanged(operation):
    ramp = np.linspace(0.1, 0.8, 63, dtype=np.float32).reshape(7, 9)
    pixels = np.repeat(ramp[..., None], 3, axis=2)
    before = pixels.copy()
    delta = operation_delta(pixels, operation, 0.02)
    assert delta.dtype == np.float64
    np.testing.assert_array_equal(delta[..., 0], delta[..., 1])
    np.testing.assert_array_equal(delta[..., 1], delta[..., 2])
    np.testing.assert_array_equal(pixels, before)


@pytest.mark.parametrize("operation", tuple(OPERATION_SPECS))
def test_zero_strength_is_exact_noop(operation):
    pixels = np.random.default_rng(10).random((7, 9, 3), dtype=np.float32)
    np.testing.assert_array_equal(operation_delta(pixels, operation, 0.0), np.zeros_like(pixels))


@pytest.mark.parametrize("operation", tuple(OPERATION_SPECS))
@pytest.mark.parametrize("shape", [(1, 1), (1, 9), (9, 1), (13, 17)])
def test_one_pixel_halo_reproduces_full_frame_delta(operation, shape):
    pixels = np.random.default_rng(4).random((*shape, 3), dtype=np.float32)
    full = operation_delta(pixels, operation, 0.03)
    tiled = np.empty_like(full)
    for y in range(0, shape[0], 4):
        for x in range(0, shape[1], 4):
            end_y, end_x = min(shape[0], y + 4), min(shape[1], x + 4)
            y0, x0 = max(0, y - 1), max(0, x - 1)
            y1, x1 = min(shape[0], end_y + 1), min(shape[1], end_x + 1)
            patch = operation_delta(pixels[y0:y1, x0:x1], operation, 0.03)
            tiled[y:end_y, x:end_x] = patch[y - y0 : end_y - y0, x - x0 : end_x - x0]
    np.testing.assert_array_equal(tiled, full)


def test_spatial_kernels_have_opposite_detail_direction():
    pixels = np.full((5, 5, 3), 0.3, dtype=np.float32)
    pixels[2, 2] = 0.6
    gain = operation_delta(pixels, "luminance_detail_gain_v1", 0.05)
    smooth = operation_delta(pixels, "luminance_detail_attenuation_v1", 0.05)
    assert np.all(gain[2, 2] > 0)
    assert np.all(smooth[2, 2] < 0)
    np.testing.assert_array_equal(gain, -smooth)


@pytest.mark.parametrize("strength", [True, -0.1, 0.5, float("nan"), float("inf"), "0.1"])
def test_operation_rejects_unbounded_or_malformed_strength(strength):
    with pytest.raises(ValueError):
        operation_delta(np.zeros((2, 2, 3), dtype=np.float32), "linear_gain_v1", strength)


def test_registry_is_immutable_and_contract_hash_is_stable():
    assert operation_contract_hash() == operation_contract_hash()
    with pytest.raises(TypeError):
        OPERATION_SPECS["unexpected"] = OPERATION_SPECS["linear_gain_v1"]
