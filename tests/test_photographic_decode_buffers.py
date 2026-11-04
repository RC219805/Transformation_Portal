"""Owned TIFF streams and decoded rasters end before subsequent native phases."""

from __future__ import annotations

import gc
import hashlib
import inspect
import io
import weakref
from dataclasses import replace

import imagecodecs
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
    ownership = """    # Independent color output no longer needs the interleaved normalized raster.
    # Detach its alpha view before immutable master copies; an aliasing linear
    # result retains its original storage for as long as it is needed.
    if alpha is not None and not np.may_share_memory(linear, values):
        alpha = alpha.copy(order="C")
    del samples, values
"""
    assert source.count(ownership) == 1
    source = source.replace(ownership, "")
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


def _rgba_source(bits, shape, color_case, *, orientation=1, alpha=True):
    dtype = np.uint8 if bits == 8 else np.uint16
    maximum = np.iinfo(dtype).max
    channels = 4 if alpha else 3
    samples = (np.arange(np.prod(shape) * channels, dtype=np.uint64) * 137 % (maximum + 1)).astype(dtype)
    samples = samples.reshape(*shape, channels)
    if alpha:
        samples[..., 3] = np.resize(np.array([0, maximum // 2, maximum], dtype), shape)
    profile = imagecodecs.cms_profile("adobergb") if color_case == "icc" else photography.output_srgb_icc()
    with io.BytesIO() as stream:
        tifffile.imwrite(
            stream,
            samples,
            photometric="rgb",
            extrasamples="unassalpha" if alpha else None,
            iccprofile=profile,
            extratags=[(274, "H", 1, orientation, False)],
        )
        raw = stream.getvalue()
    return raw, "linear_srgb" if color_case == "linear" else "auto"


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


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("shape", [(5, 7), (1, 1), (1, 11), (13, 1)])
@pytest.mark.parametrize("orientation", [1, 3, 6, 8])
@pytest.mark.parametrize("color_case", ["srgb", "icc", "linear"])
def test_detached_normalized_alpha_preserves_historical_pixels_metadata_and_identity(bits, shape, orientation, color_case):
    raw, input_color = _rgba_source(bits, shape, color_case, orientation=orientation)
    source_sha = hashlib.sha256(raw).hexdigest()
    expected = _historical_decode()(raw, source_name="source.tif", input_color=input_color)
    actual = photography.decode_master(raw, source_name="source.tif", input_color=input_color)
    assert actual.pixels.tobytes() == expected.pixels.tobytes()
    assert actual.alpha.tobytes() == expected.alpha.tobytes()
    assert actual.to_payload() == expected.to_payload()
    assert actual.content_hash() == expected.content_hash()
    assert actual.source_sha256 == hashlib.sha256(raw).hexdigest() == source_sha
    assert not actual.pixels.flags.writeable and not actual.alpha.flags.writeable


@pytest.mark.parametrize("bits", [8, 16])
@pytest.mark.parametrize("shape", [(5, 7), (1, 1), (1, 11), (13, 1)])
@pytest.mark.parametrize("color_case", ["srgb", "icc", "linear"])
@pytest.mark.parametrize("alpha", [False, True])
def test_normalized_raster_ownership_ends_before_master_freezing_unless_linear_aliases(
    bits, shape, color_case, alpha, monkeypatch
):
    raw, input_color = _rgba_source(bits, shape, color_case, alpha=alpha)
    prepare, construct = photography.apply_input_color, photography.ImageMaster
    normalized = []
    constructors = []

    def prepared(samples, policy):
        owner = samples
        while isinstance(owner.base, np.ndarray):
            owner = owner.base
        normalized.append(weakref.ref(owner))
        return prepare(samples, policy)

    def constructed(linear, *args, **kwargs):
        assert len(normalized) == 1
        parent = normalized[0]()
        if color_case == "linear":
            assert parent is not None and np.shares_memory(linear, parent)
            if alpha:
                assert np.shares_memory(kwargs["alpha"], parent)
        else:
            # The barrier is before either immutable RGB or alpha is allocated.
            # It fails the historical decoder even with cyclic GC disabled.
            assert parent is None
            if alpha:
                assert kwargs["alpha"].flags.c_contiguous
                assert kwargs["alpha"].base is None
                assert not np.may_share_memory(linear, kwargs["alpha"])
        constructors.append(True)
        return construct(linear, *args, **kwargs)

    monkeypatch.setattr(photography, "apply_input_color", prepared)
    monkeypatch.setattr(photography, "ImageMaster", constructed)
    enabled = gc.isenabled()
    gc.disable()
    try:
        result = photography.decode_master(raw, source_name="source.tif", input_color=input_color)
        assert constructors == [True] and normalized[0]() is None
        assert (result.alpha is not None) is alpha
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize("bad_alpha", [float("nan"), float("inf"), -0.1, 1.1])
@pytest.mark.parametrize("input_color", ["srgb", "linear_srgb"])
def test_normalized_alpha_lifetime_change_preserves_invalid_alpha_rejection(bad_alpha, input_color):
    samples = np.full((3, 5, 4), 0.5, np.float32)
    samples[0, 0, 3] = bad_alpha
    with io.BytesIO() as stream:
        tifffile.imwrite(stream, samples, photometric="rgb", extrasamples="unassalpha")
        raw = stream.getvalue()
    original = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError) as historical:
        _historical_decode()(raw, source_name="source.tif", input_color=input_color)
    with pytest.raises(ValueError) as current:
        photography.decode_master(raw, source_name="source.tif", input_color=input_color)
    assert str(current.value) == str(historical.value)
    assert hashlib.sha256(raw).hexdigest() == original


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
