"""Depth evidence v3 support, calibration, preservation, and failure contracts."""

from dataclasses import replace

import numpy as np
import pytest

from transformation_portal.core.depth_evidence import build_depth_evidence
from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v4.photography import create_proxy

pytestmark = pytest.mark.unit


def fixture(shape=(40, 60), target=28):
    master = ImageMaster(np.full((*shape, 3), 0.2, np.float32), "a" * 64, 16)
    proxy = create_proxy(master, target)
    native = np.tile(np.linspace(1, 3, proxy.pixels.shape[1], dtype=np.float32), (proxy.pixels.shape[0], 1))
    return master, proxy, native, np.zeros(native.shape, bool)


def companion(master):
    return {
        "path": "photo.tif",
        "source_sha256": master.source_sha256,
        "calibration": {
            "height": master.shape[0],
            "width": master.shape[1],
            "fx": 600.0,
            "fy": 500.0,
            "cx": 29.5,
            "cy": 19.5,
            "source": "measured camera fixture",
            "coordinate_space": "canonical_master",
        },
    }


def test_padding_changes_cannot_change_in_frame_normalization():
    master, proxy, native, sky = fixture((345, 518), 518)
    first = build_depth_evidence(native, sky, proxy, master.source_sha256)
    changed = native.copy()
    changed[345:] = 10000
    second = build_depth_evidence(changed, sky, proxy, master.source_sha256)
    np.testing.assert_array_equal(first.relative_depth()[:345], second.relative_depth()[:345])
    assert first.to_payload()["relative_derivative"] == second.to_payload()["relative_derivative"]
    assert not first.support_mask[345:].any()
    assert first.numeric_valid[345:].all()
    assert not first.valid_mask[345:].any()
    assert first.content_hash() != second.content_hash()


def test_raw_api_samples_including_invalid_bit_patterns_are_retained_immutably():
    master, proxy, native, sky = fixture()
    native.view(np.uint32)[0, :4] = [0x7FC00017, 0x7F800000, 0x80000000, 0xBF800000]
    expected = native.tobytes()
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256)
    assert evidence.native_depth.tobytes() == expected
    assert not evidence.numeric_valid[0, :4].any()
    native[:] = 100
    sky[:] = True
    assert evidence.native_depth.tobytes() == expected
    for value in (
        evidence.native_depth,
        evidence.numeric_valid,
        evidence.support_mask,
        evidence.sky_mask,
        evidence.valid_mask,
    ):
        with pytest.raises(ValueError):
            value.setflags(write=True)


def test_sky_is_retained_separately_and_excluded_from_relative_statistics():
    master, proxy, native, sky = fixture()
    sky[:3] = True
    native[:3] = 9000
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256)
    assert evidence.numeric_valid[:3].all()
    assert not evidence.valid_mask[:3].any()
    assert evidence.to_payload()["relative_derivative"]["high"] < 4
    assert not evidence.relative_depth()[:3].any()
    assert evidence.confidence is None
    assert evidence.to_payload()["confidence_status"] == "unavailable"


def test_unavailable_sky_retains_evidence_and_abstains_from_surface_use():
    master, proxy, native, _ = fixture()
    evidence = build_depth_evidence(native, None, proxy, master.source_sha256)
    assert evidence.numeric_valid.any() and evidence.support_mask.any()
    assert not evidence.valid_mask.any()
    assert not evidence.relative_depth().any()
    assert evidence.to_payload()["sky_status"] == "unavailable"
    assert evidence.to_payload()["relative_derivative"]["status"] == "unavailable"


def test_calibration_never_changes_native_numeric_or_surface_validity():
    master, proxy, native, sky = fixture()
    native[0, :3] = [0, -1, np.nan]
    sky[1, :3] = True
    plain = build_depth_evidence(native, sky, proxy, master.source_sha256)
    calibrated = build_depth_evidence(native, sky, proxy, master.source_sha256, companion=companion(master))
    np.testing.assert_array_equal(plain.numeric_valid, calibrated.numeric_valid)
    np.testing.assert_array_equal(plain.valid_mask, calibrated.valid_mask)
    np.testing.assert_array_equal(plain.relative_depth(), calibrated.relative_depth())
    assert plain.native_depth.tobytes() == calibrated.native_depth.tobytes()
    factor = (600 * 28 / 60 + 500 * 19 / 40) / 600
    np.testing.assert_array_equal(calibrated.metric_map_m[calibrated.numeric_valid], native[calibrated.numeric_valid] * factor)
    np.testing.assert_array_equal(calibrated.metric_map_m[~calibrated.numeric_valid], 0)
    assert calibrated.intrinsics.cx == (29.5 + 0.5) * 28 / 60 - 0.5
    assert calibrated.intrinsics.cy == (19.5 + 0.5) * 19 / 40 - 0.5
    assert calibrated.to_payload()["metric_status"] == "inferred_with_supplied_camera_calibration"


def test_all_invalid_or_all_sky_outputs_have_explicit_empty_surface_support():
    master, proxy, native, sky = fixture()
    for values, mask in ((np.zeros_like(native), sky), (native, np.ones_like(sky))):
        evidence = build_depth_evidence(values, mask, proxy, master.source_sha256, companion=companion(master))
        assert not evidence.valid_mask.any()
        assert not evidence.relative_depth().any()
        assert evidence.to_payload()["relative_derivative"]["low"] is None


@pytest.mark.parametrize("precision", ["fp32", "fp16"])
def test_precision_participates_in_evidence_identity(precision):
    master, proxy, native, sky = fixture()
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256, precision=precision)
    assert evidence.to_payload()["precision"] == {
        "compute": precision,
        "weights": "float32",
        "depth_head": "float32",
        "storage": "float32",
        "autocast": {"enabled": precision == "fp16", "dtype": "float16" if precision == "fp16" else None},
    }
    other = replace(evidence, precision="fp16" if precision == "fp32" else "fp32")
    assert evidence.content_hash() != other.content_hash()


@pytest.mark.parametrize("sky_change", [lambda sky: sky.astype(np.float32), lambda sky: sky[:-1]])
def test_sky_masks_cannot_be_silently_coerced_or_resized(sky_change):
    master, proxy, native, sky = fixture()
    with pytest.raises(ValueError, match="sky_mask"):
        build_depth_evidence(native, sky_change(sky), proxy, master.source_sha256)


@pytest.mark.parametrize("field", ["numeric_valid", "support_mask", "valid_mask"])
def test_forged_authorizing_masks_are_rejected(field):
    master, proxy, native, sky = fixture()
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256)
    mask = getattr(evidence, field).copy()
    mask[0, 0] = ~mask[0, 0]
    with pytest.raises(ValueError):
        replace(evidence, **{field: mask})


def test_unsupported_native_dtype_precision_and_wrong_source_fail_closed():
    master, proxy, native, sky = fixture()
    with pytest.raises(ValueError, match="float32"):
        build_depth_evidence(native.astype(np.float64), sky, proxy, master.source_sha256)
    with pytest.raises(ValueError, match="precision"):
        build_depth_evidence(native, sky, proxy, master.source_sha256, precision="auto")
    with pytest.raises(ValueError, match="source"):
        build_depth_evidence(native, sky, proxy, "b" * 64, companion=companion(master))


def test_metric_receipt_and_arrays_are_independently_checked():
    master, proxy, native, sky = fixture()
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256, companion=companion(master))
    with pytest.raises(ValueError, match="calibration"):
        replace(evidence, metric_map_m=evidence.metric_map_m * 2)
    calibration = evidence.to_payload()["calibration"]
    calibration["meters_per_native_unit"] *= 2
    with pytest.raises(ValueError, match="receipt"):
        replace(evidence, calibration=calibration)
    with pytest.raises(ValueError, match="grid"):
        replace(evidence, intrinsics=replace(evidence.intrinsics, width=42))
    with pytest.raises(ValueError, match="accompany"):
        replace(evidence, metric_map_m=None)
    with pytest.raises(TypeError):
        evidence.calibration["input_intrinsics"]["fx"] = 1


def test_constant_depth_is_not_artificially_stretched():
    master, proxy, native, sky = fixture()
    evidence = build_depth_evidence(np.ones_like(native), sky, proxy, master.source_sha256)
    assert evidence.valid_mask.any()
    assert not evidence.relative_depth().any()
    assert evidence.to_payload()["relative_derivative"]["status"] == "constant"


@pytest.mark.parametrize("base", [np.nextafter(np.float32(0), np.float32(1)), np.finfo(np.float32).tiny])
def test_subnormal_percentile_span_keeps_finite_positive_depth_usable(base):
    master, proxy, native, sky = fixture((10, 10), 14)
    native[:] = base
    native[9, 9] = np.nextafter(np.float32(base), np.float32(np.inf))
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256)
    before = evidence.native_depth.tobytes()
    with np.errstate(divide="raise", invalid="raise"):
        relative = evidence.relative_depth()
    expected = np.zeros_like(native)
    expected[9, 9] = 1
    np.testing.assert_array_equal(relative, expected)
    assert relative.dtype == np.float32
    assert evidence.valid_mask[:10, :10].all()
    assert evidence.native_depth.tobytes() == before
    assert not relative.flags.writeable


@pytest.mark.parametrize("scale", [np.float32(1), np.nextafter(np.float32(0), np.float32(1))])
def test_representable_percentile_span_retains_existing_float32_bytes(scale):
    master, proxy, native, sky = fixture()
    native = np.arange(1, native.size + 1, dtype=np.float32).reshape(native.shape) * scale
    native[0, :3] = [0, -1, np.nan]
    sky[1, :3] = True
    evidence = build_depth_evidence(native, sky, proxy, master.source_sha256)
    low, high = (float(value) for value in np.percentile(native[evidence.valid_mask], [1, 99]))
    assert np.float32(high - low) > 0
    expected = np.zeros_like(native)
    expected[evidence.valid_mask] = np.clip((native[evidence.valid_mask] - low) / (high - low), 0, 1)
    assert evidence.relative_depth().tobytes() == expected.tobytes()


def test_tampered_calibration_geometry_is_rejected():
    master, proxy, native, sky = fixture()
    record = companion(master)
    record["calibration"]["height"] += 1
    with pytest.raises(ValueError, match="geometry"):
        build_depth_evidence(native, sky, proxy, master.source_sha256, companion=record)
