"""Pixel-center coordinates without changing the historical proxy identity."""

from __future__ import annotations

import math

import numpy as np
import pytest

from transformation_portal.core.image_artifact import ImageMaster, ProxyTransform
from transformation_portal.lux_depth_v4.photography import create_proxy

pytestmark = pytest.mark.unit


def test_asymmetric_rounded_geometry_has_independent_pixel_center_scales():
    transform = ProxyTransform((73, 119), (26, 42), (28, 42))
    assert transform.native_bounds == (-0.5, -0.5, 118.5, 72.5)
    assert transform.content_bounds == (-0.5, -0.5, 41.5, 25.5)
    assert transform.padded_bounds == (-0.5, -0.5, 41.5, 27.5)
    # Independently evaluated landmarks, including the first sampling center
    # and the geometric center. Rounding means sx and sy are not identical.
    assert transform.native_to_proxy(0, 0) == pytest.approx((-77 / 238, -47 / 146))
    assert transform.native_to_proxy(59, 36) == (20.5, 12.5)
    assert transform.native_to_proxy(0.25, 0.75) == pytest.approx((-28 / 119, -4 / 73))
    assert transform.proxy_to_native(0, 0) == pytest.approx((77 / 84, 47 / 52))
    assert transform.proxy_to_native(20.5, 12.5) == (59, 36)
    assert transform.native_to_proxy(-0.5, -0.5) == (-0.5, -0.5)
    assert transform.proxy_to_native(-0.5, -0.5) == (-0.5, -0.5)


@pytest.mark.parametrize(
    "original,resized,padded",
    [
        ((73, 119), (26, 42), (28, 42)),
        ((119, 73), (42, 26), (42, 28)),
        ((5, 9), (5, 9), (14, 14)),
        ((1, 1), (1, 1), (14, 14)),
    ],
)
def test_coordinates_round_trip_without_padding_offset(original, resized, padded):
    transform = ProxyTransform(original, resized, padded)
    height, width = original
    for x, y in [(0, 0), (width - 1, height - 1), (0.125, 0.375), ((width - 1) / 2, (height - 1) / 2)]:
        mapped = transform.native_to_proxy(x, y)
        assert transform.proxy_to_native(*mapped) == pytest.approx((x, y), abs=1e-12)
    # Additional model-only edge context must never shift a source coordinate.
    larger_padding = ProxyTransform(original, resized, (padded[0] + 14, padded[1] + 14))
    assert larger_padding.native_to_proxy(0, 0) == transform.native_to_proxy(0, 0)


@pytest.mark.parametrize("x,y", [(0, 26), (41, 26), (0, 27), (0, 25.5), (41.5, 0), (-0.5001, 0)])
def test_inverse_rejects_padding_and_outside_content(x, y):
    transform = ProxyTransform((73, 119), (26, 42), (28, 42))
    with pytest.raises(ValueError, match="outside"):
        transform.proxy_to_native(x, y)


@pytest.mark.parametrize("x,y", [(118.5, 0), (0, 72.5), (-0.5001, 0), (0, -0.5001), (119, 73)])
def test_forward_rejects_outside_native_domain(x, y):
    transform = ProxyTransform((73, 119), (26, 42), (28, 42))
    with pytest.raises(ValueError, match="outside"):
        transform.native_to_proxy(x, y)


@pytest.mark.parametrize("method", ["native_to_proxy", "proxy_to_native"])
@pytest.mark.parametrize(
    "value", [True, False, np.bool_(True), None, "1", [1], (1, 2), 1j, math.nan, math.inf, -math.inf, 10**1000]
)
def test_coordinates_reject_nonfinite_or_malformed_points(method, value):
    transform = ProxyTransform((73, 119), (26, 42), (28, 42))
    for point in [(value, 0), (0, value)]:
        with pytest.raises(ValueError, match="finite real"):
            getattr(transform, method)(*point)


def test_mapping_preserves_historical_payload_and_master_proxy_pixels():
    pixels = np.arange(73 * 119 * 3, dtype=np.float32).reshape(73, 119, 3) / (73 * 119 * 3)
    master = ImageMaster(pixels, "a" * 64, 16, metadata={"source_orientation": 6, "orientation_normalized": True})
    proxy = create_proxy(master, target_size=42)
    transform = proxy.transform
    expected_payload = {
        "schema": "tp.image.proxy_transform.v1",
        "original_shape": [73, 119],
        "resized_shape": [26, 42],
        "padded_shape": [28, 42],
        "padding": "bottom_right_edge",
        "resize": "bilinear_pixel_centers",
    }
    master_hash = master.content_hash()
    proxy_bytes = proxy.pixels.tobytes()
    for x, y in [(0, 0), (118, 72), (59, 36)]:
        transform.proxy_to_native(*transform.native_to_proxy(x, y))
    assert transform.to_payload() == expected_payload
    assert ProxyTransform.from_payload(expected_payload) == transform
    assert master.content_hash() == master_hash == proxy.master_content_hash
    assert proxy.pixels.tobytes() == proxy_bytes
    assert not proxy.pixels.flags.writeable


def test_existing_noncanonical_payload_is_not_reinterpreted_or_tightened():
    # The v1 constructor historically permits these dimensions. The audit
    # helpers describe that exact transform; they do not relabel it as output
    # of create_proxy or silently tighten historical artifact verification.
    transform = ProxyTransform((9, 5), (8, 20), (28, 42))
    assert ProxyTransform.from_payload(transform.to_payload()) == transform
    assert transform.native_to_proxy(2, 4) == (9.5, 3.5)
    assert transform.proxy_to_native(9.5, 3.5) == (2, 4)


@pytest.mark.parametrize("original,resized", [(1, 1), (149, 1), (119, 42), (73, 26)])
def test_representable_points_below_excluded_edges_stay_inside_the_other_frame(original, resized):
    transform = ProxyTransform((original, original), (resized, resized), (((resized + 13) // 14) * 14,) * 2)
    native = math.nextafter(original - 0.5, -math.inf)
    proxy = math.nextafter(resized - 0.5, -math.inf)
    for point in [transform.native_to_proxy(native, native), transform.native_to_proxy(-0.5, -0.5)]:
        assert all(-0.5 <= value < resized - 0.5 for value in point)
        transform.proxy_to_native(*point)
    for point in [transform.proxy_to_native(proxy, proxy), transform.proxy_to_native(-0.5, -0.5)]:
        assert all(-0.5 <= value < original - 0.5 for value in point)
        transform.native_to_proxy(*point)
