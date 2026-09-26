"""Depth Pro V6 products retain estimates without inventing surface authority."""

from __future__ import annotations

import hashlib
import io
import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v4.photography import decode_master
from transformation_portal.lux_depth_v6.color import GradeRecipe, RenderRecipe
from transformation_portal.lux_depth_v6.depth_pro_products import model_input_bytes, native_image_products

pytestmark = pytest.mark.unit


def source(*, alpha=False, orientation=1):
    samples = np.arange(24, dtype=np.uint8).reshape(2, 4, 3) * 10
    if alpha:
        opacity = np.full((2, 4, 1), 255, np.uint8)
        opacity[0, 0] = 0
        opacity[0, 1] = 128
        samples = np.concatenate([samples, opacity], axis=2)
    buffer = io.BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    Image.fromarray(samples).save(buffer, format="PNG", exif=exif)
    raw = buffer.getvalue()
    return raw, decode_master(raw, source_name="source.png", input_color="srgb")


def worker(shape):
    return {
        "depth_units": "meters",
        "dtype": "float32",
        "input_size": list(shape),
        "provenance": {"engine": "apple_depth_pro", "camera": {"focal_length_source": "model_estimated"}},
    }


def products(raw, native, *, metadata=None, grade=None, checkpoint=lambda: None, input_id="input-0000"):
    return {
        name.removeprefix(f"{input_id}/"): data
        for name, data in native_image_products(
            raw,
            "source.png",
            "srgb",
            native,
            worker(native.shape) if metadata is None else metadata,
            input_id,
            grade or GradeRecipe(),
            RenderRecipe(),
            max_pixels=10000,
            checkpoint=checkpoint,
        )
    }


def array(data):
    return np.load(io.BytesIO(data), allow_pickle=False)


def test_original_grid_input_has_explicit_quantization_and_no_orientation_metadata():
    raw, master = source(orientation=6)
    assert master.shape == (4, 2)
    before = master.content_hash()
    encoded = model_input_bytes(master)
    assert encoded == model_input_bytes(master)
    with Image.open(io.BytesIO(encoded)) as image:
        assert image.mode == "RGB"
        assert image.size == (2, 4)
        assert len(image.getexif()) == 0
        with Image.open(io.BytesIO(raw)) as original:
            expected = np.rot90(np.asarray(original), -1)
        np.testing.assert_array_equal(np.asarray(image), expected)
    assert master.content_hash() == before


def test_model_input_clips_extended_samples_without_clipping_master():
    values = np.array([[[-3e38, 0, 3e38], [0.18, 0.5, 1]]], np.float32)
    master = ImageMaster(values, "a" * 64, 32)
    with Image.open(io.BytesIO(model_input_bytes(master))) as image:
        np.testing.assert_array_equal(np.asarray(image)[0, 0], [0, 0, 255])
    np.testing.assert_array_equal(master.pixels, values)


def test_depth_preview_bounds_and_sampling_preserve_numeric_mask():
    shape = (1603, 3)
    raw = io.BytesIO()
    Image.fromarray(np.full((*shape, 3), 128, np.uint8)).save(raw, format="PNG")
    native = np.arange(1, 4810, dtype=np.float32).reshape(shape)
    native[::2] = np.nan
    outputs = products(raw.getvalue(), native)
    descriptor = json.loads(outputs["depth.json"])["preview"]
    assert descriptor["shape"] == [1600, 3]
    rows = ((2 * np.arange(1600) + 1) * shape[0]) // (2 * 1600)
    with Image.open(io.BytesIO(outputs["depth-preview-numeric-valid.png"])) as image:
        np.testing.assert_array_equal(np.asarray(image), np.isfinite(native[rows]).astype(np.uint8) * 255)


def test_products_are_deterministic_and_depth_never_changes_baseline():
    raw, master = source()
    native = np.arange(1, 9, dtype=np.float32).reshape(master.shape)
    outputs = products(raw, native)
    assert outputs == products(raw, native)
    assert set(outputs) == {
        "estimated-depth-meters.npy",
        "estimated-depth-meters.tif",
        "numeric-valid.npy",
        "depth-preview.png",
        "depth-preview-numeric-valid.png",
        "depth.json",
        "baseline.npy",
        "master.npy",
        "display.npy",
        "delivery.tif",
        "preview.png",
        "photograph.json",
    }
    np.testing.assert_array_equal(array(outputs["baseline.npy"]), master.pixels)
    np.testing.assert_array_equal(array(outputs["master.npy"]), master.pixels)
    changed_depth = products(raw, native * 100)
    for name in ("baseline.npy", "master.npy", "display.npy", "delivery.tif", "preview.png", "photograph.json"):
        assert outputs[name] == changed_depth[name]
    descriptor = json.loads(outputs["photograph.json"])
    assert descriptor["schema"] == "tp.lux.depth_pro.photograph.v1"
    assert descriptor["reconstruction"]["depth_edits"] == "abstained"
    assert descriptor["reconstruction"]["reason"] == "sky_evidence_unavailable"
    assert descriptor["reconstruction"]["changed_pixels"] == 0


def test_native_invalid_values_are_retained_and_mask_is_numeric_only():
    raw, master = source()
    native = np.array([[np.nan, np.inf, -np.inf, -1], [0, 1, 2, 3]], np.float32)
    outputs = products(raw, native)
    retained = array(outputs["estimated-depth-meters.npy"])
    assert retained.dtype == np.float32
    assert retained.tobytes() == native.tobytes()
    assert tifffile.imread(io.BytesIO(outputs["estimated-depth-meters.tif"])).tobytes() == native.tobytes()
    expected = np.isfinite(native) & (native > 0)
    np.testing.assert_array_equal(array(outputs["numeric-valid.npy"]), expected)
    with Image.open(io.BytesIO(outputs["depth-preview-numeric-valid.png"])) as image:
        np.testing.assert_array_equal(np.asarray(image), expected.astype(np.uint8) * 255)
    descriptor = json.loads(outputs["depth.json"])
    assert descriptor["native_semantics"] == "model_estimated_meters"
    assert descriptor["numeric_valid_pixels"] == 3
    assert descriptor["sky_status"] == descriptor["surface_status"] == "unavailable"
    assert descriptor["calibration"] is None
    assert descriptor["measured_scene_accuracy"] == "not_established"
    assert descriptor["shape"] == list(master.shape)
    assert descriptor["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert descriptor["model_input"]["sha256"] == hashlib.sha256(model_input_bytes(master)).hexdigest()
    for name, record in descriptor["artifacts"].items():
        assert record["sha256"] == hashlib.sha256(outputs[name]).hexdigest()
        assert record["size_bytes"] == len(outputs[name])


@pytest.mark.parametrize("sample", [np.nan, 0, 1, np.nextafter(np.float32(0), np.float32(1))])
def test_empty_and_constant_depths_have_zero_preview_without_false_invalidity(sample):
    raw, master = source()
    native = np.full(master.shape, sample, np.float32)
    outputs = products(raw, native)
    with Image.open(io.BytesIO(outputs["depth-preview.png"])) as image:
        assert not np.asarray(image).any()
    expected_valid = np.isfinite(sample) and sample > 0
    assert bool(array(outputs["numeric-valid.npy"]).all()) == expected_valid
    descriptor = json.loads(outputs["depth.json"])
    assert descriptor["preview"]["normalization_limits_m"] == ([float(sample)] * 2 if expected_valid else None)


def test_subnormal_depth_span_does_not_overflow_or_disappear():
    raw, master = source()
    smallest = np.nextafter(np.float32(0), np.float32(1))
    native = (np.arange(1, 9, dtype=np.float32) * smallest).reshape(master.shape)
    outputs = products(raw, native)
    with Image.open(io.BytesIO(outputs["depth-preview.png"])) as image:
        values = np.asarray(image)
        assert values.min() == 0 and values.max() == 65535
        assert len(np.unique(values)) == 8


def test_alpha_is_preserved_and_nonopaque_pixels_are_protected_from_grading():
    raw, master = source(alpha=True, orientation=6)
    outputs = products(raw, np.ones(master.shape, np.float32), grade=GradeRecipe(exposure_stops=1))
    np.testing.assert_array_equal(array(outputs["alpha.npy"]), master.alpha)
    graded = array(outputs["master.npy"])
    protected = master.alpha != 1
    np.testing.assert_array_equal(graded[protected], master.pixels[protected])
    np.testing.assert_array_equal(graded[~protected], master.pixels[~protected] * 2)
    assert tifffile.imread(io.BytesIO(outputs["delivery.tif"])).shape == (*master.shape, 4)
    with Image.open(io.BytesIO(outputs["preview.png"])) as image:
        assert image.mode == "RGBA"
    with Image.open(io.BytesIO(model_input_bytes(master))) as image:
        assert image.mode == "RGB"


@pytest.mark.parametrize("native", [np.ones((2, 4), np.float64), np.ones((4, 2), np.float32), np.ones((2, 4, 1), np.float32)])
def test_invalid_native_dtype_or_geometry_fails_before_products(native):
    raw, _ = source()
    with pytest.raises(ValueError, match="float32.*original grid"):
        products(raw, native)


@pytest.mark.parametrize(
    "field,value",
    [("depth_units", "relative"), ("dtype", "float64"), ("input_size", [True, 4]), ("provenance", {"engine": "da3"})],
)
def test_worker_semantic_mismatches_are_rejected(field, value):
    raw, master = source()
    metadata = worker(master.shape)
    metadata[field] = value
    with pytest.raises(ValueError, match="worker units, dtype, engine, or original-grid geometry"):
        products(raw, np.ones(master.shape, np.float32), metadata=metadata)


def test_nonfinite_worker_metadata_and_unsafe_identifier_fail_closed():
    raw, master = source()
    native = np.ones(master.shape, np.float32)
    metadata = worker(master.shape)
    metadata["focal_length_px"] = np.inf
    with pytest.raises(ValueError, match="finite JSON"):
        products(raw, native, metadata=metadata)
    with pytest.raises(ValueError, match="portable input identifier"):
        products(raw, native, input_id="../escape")


def test_product_stream_honors_cancellation():
    raw, master = source()

    def cancel():
        raise TimeoutError("cancelled")

    with pytest.raises(TimeoutError, match="cancelled"):
        products(raw, np.ones(master.shape, np.float32), checkpoint=cancel)
