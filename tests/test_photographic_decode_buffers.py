"""Owned TIFF streams and decoded rasters end before subsequent native phases."""

from __future__ import annotations

import gc
import hashlib
import inspect
import io
import weakref
from dataclasses import replace

import numpy as np
import pytest
import tifffile

from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v4 import photography
from transformation_portal.materials_v4 import engine
from transformation_portal.materials_v4.contracts import MaterialEvidence, RegionEvidence

pytestmark = pytest.mark.unit


def _source(*, bits=16, alpha=False, orientation=1):
    channels = 4 if alpha else 3
    values = np.arange(5 * 7 * channels, dtype=np.uint16).reshape(5, 7, channels)
    if bits == 8:
        values = values.astype(np.uint8)
    elif bits == 32:
        values = values.astype(np.float32) / 30 - 0.1
    if alpha:
        values[..., 3] = np.linspace(0, 1 if bits == 32 else np.iinfo(values.dtype).max, 35).reshape(5, 7)
    stream = io.BytesIO()
    tifffile.imwrite(
        stream,
        values,
        photometric="rgb",
        extrasamples="unassalpha" if alpha else None,
        iccprofile=photography.output_srgb_icc(),
        extratags=[(274, "H", 1, orientation, False)],
    )
    return stream.getvalue()


def _historical_decode():
    source = inspect.getsource(photography.decode_master)
    source = source.replace(
        "with io.BytesIO(source_bytes) as source, tifffile.TiffFile(source) as tif:",
        "with tifffile.TiffFile(io.BytesIO(source_bytes)) as tif:",
    ).replace("    del pixels\n", "")
    source = source.replace(
        "values = pixels.astype(np.float32)\n    values /= divisor",
        "values = pixels.astype(np.float32) / divisor",
    )
    namespace = dict(vars(photography))
    # The oracle reconstructs only the checked-in function, never external code.
    # pylint: disable-next=exec-used
    exec(compile(source, "<historical TIFF decoder oracle>", "exec"), namespace)  # noqa: S102
    return namespace["decode_master"]


def _track_streams(monkeypatch, raw):
    original = io.BytesIO
    streams = []

    class TrackedBytesIO(original):
        def __init__(self, initial=b""):
            super().__init__(initial)
            if initial is raw:
                streams.append(self)

    monkeypatch.setattr(photography.io, "BytesIO", TrackedBytesIO)
    return streams


def test_historical_caller_owned_tiff_stream_remains_open_without_gc(monkeypatch):
    raw = _source()
    historical = _historical_decode()
    streams = _track_streams(monkeypatch, raw)
    enabled = gc.isenabled()
    gc.disable()
    try:
        historical(raw, source_name="source.tif")
        assert len(streams) == 1 and not streams[0].closed
    finally:
        for stream in streams:
            stream.close()
        if enabled:
            gc.enable()


def test_tiff_stream_and_decoded_raster_are_released_before_color_preparation(monkeypatch):
    raw = _source(alpha=True, orientation=6)
    streams = _track_streams(monkeypatch, raw)
    arrays = []
    asarray, prepare = tifffile.TiffPage.asarray, photography.apply_input_color

    def tracked_array(page, *args, **kwargs):
        result = asarray(page, *args, **kwargs)
        arrays.append(weakref.ref(result))
        return result

    def prepared(samples, policy):
        assert len(streams) == 1 and streams[0].closed
        assert arrays and all(reference() is None for reference in arrays)
        return prepare(samples, policy)

    monkeypatch.setattr(tifffile.TiffPage, "asarray", tracked_array)
    monkeypatch.setattr(photography, "apply_input_color", prepared)
    enabled = gc.isenabled()
    gc.disable()
    try:
        master = photography.decode_master(raw, source_name="source.tif")
        assert master.shape == (7, 5) and master.alpha is not None
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("alpha", [False, True])
def test_normalization_reuses_its_independent_float_copy_without_mutating_decoded_samples(bits, alpha, monkeypatch):
    raw = _source(bits=bits, alpha=alpha)
    asarray, prepare = tifffile.TiffPage.asarray, photography.apply_input_color
    converted, decoded = [], []

    class TrackedRaster(np.ndarray):
        def astype(self, dtype, order="K", casting="unsafe", subok=True, copy=True):
            result = super().astype(dtype, order=order, casting=casting, subok=subok, copy=copy)
            converted.append(weakref.ref(result))
            return result

    def tracked_array(page, *args, **kwargs):
        result = asarray(page, *args, **kwargs).view(TrackedRaster)
        decoded.append((result, result.tobytes()))
        return result

    def prepared(samples, policy):
        assert len(converted) == 1 and converted[0]() is not None
        assert samples.dtype == np.float32 and np.shares_memory(samples, converted[0]())
        assert all(not np.shares_memory(samples, array) and array.tobytes() == original for array, original in decoded)
        return prepare(samples, policy)

    monkeypatch.setattr(tifffile.TiffPage, "asarray", tracked_array)
    monkeypatch.setattr(photography, "apply_input_color", prepared)
    photography.decode_master(raw, source_name="source.tif")


@pytest.mark.parametrize("phase", ["constructor", "geometry", "raster", "color"])
def test_tiff_stream_closes_on_decode_failure_without_gc(monkeypatch, phase):
    raw = _source()
    streams = _track_streams(monkeypatch, raw)

    def failed(*args, **kwargs):
        raise ValueError("controlled decode failure")

    if phase == "constructor":
        monkeypatch.setattr(tifffile, "TiffFile", failed)
    elif phase == "raster":
        monkeypatch.setattr(tifffile.TiffPage, "asarray", failed)
    elif phase == "color":
        monkeypatch.setattr(photography, "apply_input_color", failed)
    enabled = gc.isenabled()
    gc.disable()
    try:
        with pytest.raises(ValueError):
            photography.decode_master(raw, source_name="source.tif", max_pixels=1 if phase == "geometry" else 1000)
        assert len(streams) == 1 and streams[0].closed
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("bits", [8, 16, 32])
@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("input_color", ["auto", "linear_srgb"])
def test_tiff_buffer_lifetimes_preserve_exact_historical_master_identity(bits, alpha, orientation, input_color):
    raw = _source(bits=bits, alpha=alpha, orientation=orientation)
    expected = _historical_decode()(raw, source_name="source.tif", input_color=input_color)
    actual = photography.decode_master(raw, source_name="source.tif", input_color=input_color)
    assert actual.pixels.tobytes() == expected.pixels.tobytes()
    assert actual.to_payload() == expected.to_payload()
    assert actual.content_hash() == expected.content_hash()
    assert actual.source_sha256 == hashlib.sha256(raw).hexdigest()
    if alpha:
        assert actual.alpha.tobytes() == expected.alpha.tobytes()


@pytest.mark.parametrize("alpha", [False, True])
@pytest.mark.parametrize("strength", [0, 0.02])
def test_materials_releases_mutable_output_before_authoritative_result_hash(alpha, strength, monkeypatch):
    pixels = np.full((7, 9, 3), 0.4, np.float32)
    plane = np.ones((7, 9), np.float32) if alpha else None
    master = ImageMaster(pixels, "a" * 64, 16, alpha=plane)
    evidence = MaterialEvidence(
        master.source_sha256,
        master.shape,
        (RegionEvidence("glass", "glass", np.ones(master.shape, np.float32), semantic_confidence=1.0),),
    )
    policy = engine.ResponsePolicy(min_coverage_px=1)
    policy = replace(policy, operations=tuple(replace(item, strength=strength) for item in policy.operations))
    plan = engine.plan_response(master, evidence, policy)
    source = inspect.getsource(engine.apply_response).replace("    del output\n", "")
    namespace = dict(vars(engine))
    # pylint: disable-next=exec-used
    exec(compile(source, "<historical Materials output oracle>", "exec"), namespace)  # noqa: S102
    expected, expected_receipt = namespace["apply_response"](master, evidence, plan)
    mutable = []
    initialize, content_hash = ImageMaster.__post_init__, ImageMaster.content_hash

    def initialized(value):
        if value.pixels.flags.writeable:
            mutable.append(weakref.ref(value.pixels))
        initialize(value)

    def hashed(value):
        if mutable:
            assert all(reference() is None for reference in mutable)
        return content_hash(value)

    monkeypatch.setattr(ImageMaster, "__post_init__", initialized)
    monkeypatch.setattr(ImageMaster, "content_hash", hashed)
    actual, receipt = engine.apply_response(master, evidence, plan)
    assert receipt == expected_receipt and actual.content_hash() == expected.content_hash()
    assert actual.pixels.tobytes() == expected.pixels.tobytes()
    assert (actual is master) == (strength == 0)
