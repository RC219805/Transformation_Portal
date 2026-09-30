"""Numerical, metadata and immutable-source contracts for color preparation."""

from __future__ import annotations

import hashlib
import io
import struct
import zlib

import imagecodecs
import numpy as np
import pytest
import tifffile
from PIL import Image

from transformation_portal.core.image_artifact import metadata_payload
from transformation_portal.lux_depth_v4 import color_preparation as color
from transformation_portal.lux_depth_v4.photography import decode_master, validate_input_color_metadata

pytestmark = pytest.mark.unit


def _prepare(profile=None, **kwargs):
    options = dict(input_color="auto", image_format="TIFF", profile=profile, declared_color=None, exif_color=None)
    options.update(kwargs)
    return color.prepare_input_color(**options)


def _tiff(samples, *, profile=None, metadata=None):
    stream = io.BytesIO()
    channels = samples.shape[-1] if samples.ndim == 3 else 1
    tifffile.imwrite(
        stream,
        samples,
        photometric="rgb" if channels in {3, 4} else "minisblack",
        metadata=metadata,
        iccprofile=profile,
        extrasamples="unassalpha" if channels in {2, 4} else None,
    )
    return stream.getvalue()


def _png(samples, *chunks):
    encoded = imagecodecs.png_encode(samples)
    extra = b"".join(
        struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data)) for kind, data in chunks
    )
    return encoded[:33] + extra + encoded[33:]


def _inspect(source, name="photo.tif", mode="auto"):
    return validate_input_color_metadata(io.BytesIO(source), source_name=name, input_color=mode)


def _icc_chunk(profile):
    return b"iCCP", b"source\0\0" + zlib.compress(profile)


def test_canonical_srgb_preserves_historical_numeric_result_exactly():
    samples = np.array([[[-0.1, 0.5, 1.2], [0.04045, 0.0031308, 1.0]]], np.float32)
    expected = np.where(samples <= 0.04045, samples / 12.92, ((np.maximum(samples, 0) + 0.055) / 1.055) ** 2.4).astype(
        np.float32
    )
    prepared = _prepare(color.output_srgb_icc())
    np.testing.assert_array_equal(color.apply_input_color(samples, prepared), expected)
    assert prepared.evidence["engine"] == "numpy"
    assert prepared.resolution == "recognized_srgb_icc"


@pytest.mark.parametrize("profile_name", ["srgb", "adobergb", "display_p3"])
def test_icc_float_conversion_preserves_all_uint16_levels_and_alpha(profile_name):
    profile = (
        color._rgb_profile(color._P3_CHROMATICITY) if profile_name == "display_p3" else imagecodecs.cms_profile(profile_name)
    )
    if profile_name == "srgb":
        # A valid source profile need not be byte-identical to the legacy fast path.
        changed = bytearray(profile)
        changed[80:84] = b"TEST"
        profile = bytes(changed)
    levels = np.arange(65536, dtype=np.uint16).reshape(256, 256)
    samples = np.stack((levels, levels, levels, 65535 - levels), axis=-1)
    source = _tiff(samples, profile=profile)
    original = hashlib.sha256(source).hexdigest()
    master = decode_master(source, source_name="precise.tif")
    assert master.source_bit_depth == 16
    assert master.pixels.dtype == np.float32
    assert np.unique(master.pixels[..., 0]).size == 65536
    np.testing.assert_array_equal(master.alpha, (65535 - levels).astype(np.float32) / 65535)
    assert master.source_icc == profile
    assert hashlib.sha256(source).hexdigest() == original == master.source_sha256
    receipt = master.metadata["color_preparation"]
    assert receipt["source_icc_sha256"] == hashlib.sha256(profile).hexdigest()
    assert receipt["schema"] == "tp.color.preparation.v1"
    assert receipt["intent"] == "relative_colorimetric"
    assert receipt["requested_input_color"] == "auto"
    assert receipt["output_precision"] == "float32"
    assert imagecodecs.cms_version() in receipt["engine_version"]
    assert receipt == _inspect(source)


@pytest.mark.parametrize("space", ["adobergb", "display_p3"])
def test_wide_gamut_primary_conversion_retains_negative_and_overrange(space):
    profile = imagecodecs.cms_profile("adobergb") if space == "adobergb" else color._rgb_profile(color._P3_CHROMATICITY)
    samples = np.eye(3, dtype=np.float32)[None, ...]
    output = color.apply_input_color(samples, _prepare(profile))
    # Independent matrix expectations for D65 Adobe RGB / Display P3 primaries
    # mapped to linear sRGB; ICC fixed-point matrix quantization needs 1e-3.
    expected = (
        [[1.3984, 0, 0], [-0.3984, 1, -0.04294], [0, 0, 1.04294]]
        if space == "adobergb"
        else [[1.22494, -0.04206, -0.01964], [-0.22494, 1.04206, -0.07864], [0, 0, 1.09827]]
    )
    np.testing.assert_allclose(output[0], expected, atol=0.001, rtol=0)
    assert output.min() < -0.1 and output.max() > 1.2


def test_float_icc_source_is_not_clipped_before_or_after_transform():
    profile = imagecodecs.cms_profile("srgb")
    samples = np.array([[[-0.1, 0.5, 1.2]]], np.float32)
    master = decode_master(_tiff(samples, profile=profile), source_name="float.tif")
    assert master.source_bit_depth == 32
    assert master.pixels.min() < 0 and master.pixels.max() > 1
    np.testing.assert_allclose(master.pixels, color.srgb_to_linear(samples), atol=1e-5)


def test_gray_icc_to_rgb_preserves_gray_precision_and_straight_alpha():
    profile = imagecodecs.cms_profile("gray", gamma=2.2)
    levels = np.arange(2048, dtype=np.uint16).reshape(32, 64) * 31
    samples = np.stack((levels, 65535 - levels), axis=-1)
    source = _tiff(samples, profile=profile)
    master = decode_master(source, source_name="gray.tif")
    assert master.shape == levels.shape
    assert master.metadata["color_preparation"]["source_color"] == "icc_gray"
    assert np.unique(master.pixels[..., 0]).size == 2048
    np.testing.assert_array_equal(master.alpha, (65535 - levels).astype(np.float32) / 65535)
    assert _inspect(source) == metadata_payload(master.metadata["color_preparation"])


@pytest.mark.parametrize(
    "mutation", ["short", "oversize", "size", "signature", "cmyk", "tag_count", "tag_offset", "duplicate", "overlap"]
)
def test_untrusted_icc_ranges_fail_before_native_profile_parsing(monkeypatch, mutation):
    profile = bytearray(imagecodecs.cms_profile("srgb"))
    if mutation == "short":
        profile = bytearray(b"invalid")
    elif mutation == "oversize":
        profile = bytearray(color.MAX_ICC_BYTES + 1)
    elif mutation == "size":
        struct.pack_into(">I", profile, 0, len(profile) + 1)
    elif mutation == "signature":
        profile[36:40] = b"fake"
    elif mutation == "cmyk":
        profile[16:20] = b"CMYK"
    elif mutation == "tag_count":
        struct.pack_into(">I", profile, 128, 1025)
    elif mutation == "tag_offset":
        struct.pack_into(">I", profile, 136, len(profile) - 4)
    elif mutation == "duplicate":
        profile[144:148] = profile[132:136]
    elif mutation == "overlap":
        first_offset = struct.unpack_from(">I", profile, 136)[0]
        struct.pack_into(">I", profile, 148, first_offset + 4)
    monkeypatch.setattr(imagecodecs, "cms_profile_validate", lambda *_: pytest.fail("Malformed ICC entered native parser"))
    with pytest.raises(color.InputColorError) as error:
        _prepare(bytes(profile), input_color="auto_assume_srgb")
    assert error.value.code == "input_color_unsupported_icc"


def test_profile_description_is_not_color_authority():
    profile = bytearray(imagecodecs.cms_profile("adobergb"))
    # The profile's actual matrix/TRC bytes remain Adobe RGB regardless of label.
    profile[80:84] = b"sRGB"
    prepared = _prepare(bytes(profile))
    assert color.apply_input_color(np.eye(3, dtype=np.float32)[None, ...], prepared).max() > 1.3
    with pytest.raises(color.InputColorError, match="disagree"):
        _prepare(bytes(profile), declared_color="srgb")


def test_target_profiles_and_transform_results_are_deterministic():
    profile = imagecodecs.cms_profile("adobergb")
    first, second = _prepare(profile), _prepare(profile)
    assert first.target_profile == second.target_profile
    assert first.evidence == second.evidence
    samples = np.random.default_rng(7).random((19, 23, 3), dtype=np.float32)
    np.testing.assert_array_equal(color.apply_input_color(samples, first), color.apply_input_color(samples, second))


def test_tiling_does_not_change_numeric_result(monkeypatch):
    prepared = _prepare(imagecodecs.cms_profile("adobergb"))
    samples = np.random.default_rng(17).random((9, 11, 3), dtype=np.float32)
    expected = color.apply_input_color(samples, prepared)
    monkeypatch.setattr(color, "_TILE_PIXELS", 12)
    np.testing.assert_array_equal(color.apply_input_color(samples, prepared), expected)


def test_clipping_cms_engine_is_rejected(monkeypatch):
    transform = color._cms_transform
    monkeypatch.setattr(color, "_cms_transform", lambda *args: np.clip(transform(*args), 0, 1))
    with pytest.raises(color.InputColorError, match="unclipped float32"):
        _prepare(imagecodecs.cms_profile("adobergb"))


@pytest.mark.parametrize("extension", ["png", "tif", "jpg"])
def test_auto_assumption_is_explicit_recorded_and_source_immutable(extension):
    pixels = np.full((3, 4, 3), 128, np.uint8)
    if extension == "tif":
        source = _tiff(pixels)
    else:
        stream = io.BytesIO()
        Image.fromarray(pixels).save(stream, format="JPEG" if extension == "jpg" else "PNG")
        source = stream.getvalue()
    original = source[:]
    if extension != "jpg":
        with pytest.raises(color.InputColorError, match="Ambiguous"):
            decode_master(source, source_name=f"photo.{extension}")
    master = decode_master(source, source_name=f"photo.{extension}", input_color="auto_assume_srgb")
    receipt = master.metadata["color_preparation"]
    assert receipt["action"] == "assume" and receipt["assumed_srgb"]
    assert receipt["requested_input_color"] == "auto_assume_srgb"
    assert receipt == _inspect(source, f"photo.{extension}", "auto_assume_srgb")
    assert source == original


def test_new_auto_policy_never_assumes_raw_encoding():
    source = b"governed raw"
    samples = np.full((2, 2, 3), 1.5, np.float32)
    master = decode_master(
        source,
        source_name="photo.dng",
        input_color="auto_assume_srgb",
        raw_decoder=lambda *_: (samples, {"color_space": "linear_srgb"}),
    )
    receipt = master.metadata["color_preparation"]
    assert receipt["action"] == "identity" and not receipt["assumed_srgb"]
    assert receipt["resolution"] == "governed_raw_decoder"
    assert receipt["requested_input_color"] == "auto_assume_srgb"
    assert receipt == _inspect(source, "photo.dng", "auto_assume_srgb")
    np.testing.assert_array_equal(master.pixels, samples)


@pytest.mark.parametrize("exif_value", [65535, 2, "1", True])
def test_unknown_exif_is_never_untagged_fallback(exif_value):
    with pytest.raises(color.InputColorError):
        _prepare(input_color="auto_assume_srgb", exif_color=exif_value)


def test_tiff_nested_exif_color_is_shared_by_preview_and_execution(monkeypatch):
    exif = Image.Exif()
    exif[34665] = 1
    exif._ifds[34665] = {40961: 1}
    stream = io.BytesIO()
    Image.new("RGB", (2, 2), (128, 128, 128)).save(stream, format="TIFF", exif=exif)
    source = stream.getvalue()
    master = decode_master(source, source_name="exif.tif")
    assert master.metadata["color_resolution"] == "exif_srgb"
    monkeypatch.setattr(tifffile.TiffPage, "asarray", lambda *_: pytest.fail("Preflight decoded TIFF"))
    assert _inspect(source) == metadata_payload(master.metadata["color_preparation"])


@pytest.mark.parametrize("gray", [False, True])
def test_icc_raster_channel_mismatch_fails_identically_in_preview_and_decode(gray):
    profile = imagecodecs.cms_profile("srgb") if gray else imagecodecs.cms_profile("gray", gamma=2.2)
    pixels = np.zeros((2, 2) if gray else (2, 2, 3), np.uint16)
    source = _tiff(pixels, profile=profile)
    for operation in (lambda: _inspect(source), lambda: decode_master(source, source_name="photo.tif")):
        with pytest.raises(color.InputColorError, match="channel layout"):
            operation()


def test_cmyk_jpeg_never_passes_header_preflight():
    stream = io.BytesIO()
    Image.new("CMYK", (2, 2)).save(stream, format="JPEG")
    with pytest.raises(ValueError, match="RGB/gray"):
        _inspect(stream.getvalue(), "photo.jpg", "auto_assume_srgb")


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_png16_precision_alpha_and_header_preflight(channels, monkeypatch):
    ramp = np.arange(2048, dtype=np.uint16).reshape(32, 64) * 31
    samples = ramp if channels == 1 else np.repeat(ramp[..., None], channels, axis=-1)
    source = _png(samples, (b"sRGB", b"\0"))
    master = decode_master(source, source_name="photo.png")
    assert master.source_bit_depth == 16
    assert np.unique(master.pixels[..., 0]).size == 2048
    np.testing.assert_array_equal(master.pixels[..., 0], color.srgb_to_linear(ramp.astype(np.float32) / 65535))
    if channels in {2, 4}:
        np.testing.assert_array_equal(master.alpha, ramp.astype(np.float32) / 65535)
    monkeypatch.setattr(Image.Image, "load", lambda *_: pytest.fail("Preflight loaded PNG"))
    monkeypatch.setattr(imagecodecs, "png_decode", lambda *_: pytest.fail("Preflight decoded PNG"))
    assert _inspect(source, "photo.png") == metadata_payload(master.metadata["color_preparation"])


def test_png16_color_key_alpha_does_not_quantize():
    samples = np.array([[[12345, 23456, 34567], [12346, 23457, 34568]]], np.uint16)
    source = _png(samples, (b"sRGB", b"\0"), (b"tRNS", struct.pack(">HHH", *samples[0, 0])))
    master = decode_master(source, source_name="photo.png")
    np.testing.assert_array_equal(master.alpha, [[0, 1]])
    np.testing.assert_array_equal(master.pixels, color.srgb_to_linear(samples.astype(np.float32) / 65535))


@pytest.mark.parametrize("cicp", [bytes((1, 13, 0, 1)), bytes((12, 13, 0, 1))])
def test_png_cicp_srgb_and_p3_are_resolved_without_assumption(cicp):
    source = _png(np.eye(3, dtype=np.uint16)[None, ...] * 65535, (b"cICP", cicp))
    master = decode_master(source, source_name="photo.png", input_color="auto_assume_srgb")
    assert master.metadata["color_preparation"]["action"] == "convert"
    assert not master.metadata["color_preparation"]["assumed_srgb"]
    if cicp[0] == 12:
        assert master.pixels.min() < -0.1 and master.pixels.max() > 1.2
    else:
        np.testing.assert_array_equal(master.pixels, np.eye(3, dtype=np.float32)[None, ...])


@pytest.mark.parametrize(
    "chunks",
    [
        [(b"gAMA", struct.pack(">I", 45455))],
        [(b"cHRM", struct.pack(">8I", *(round(v * 100000) for v in color._SRGB_CHROMATICITY)))],
        [(b"cICP", bytes((9, 16, 0, 1)))],
        [(b"cICP", bytes((12, 13, 0, 1))), (b"sRGB", b"\0")],
        [(b"sRGB", b"\0"), (b"gAMA", struct.pack(">I", 100000))],
    ],
)
def test_png_unknown_partial_or_conflicting_metadata_never_assumes(chunks):
    source = _png(np.zeros((2, 2, 3), np.uint8), *chunks)
    for operation in (
        lambda: _inspect(source, "photo.png", "auto_assume_srgb"),
        lambda: decode_master(source, source_name="photo.png", input_color="auto_assume_srgb"),
    ):
        with pytest.raises(color.InputColorError):
            operation()


def test_png_complete_calibration_uses_declared_gamma_without_srgb_relabeling():
    chunks = [
        (b"gAMA", struct.pack(">I", 100000)),
        (b"cHRM", struct.pack(">8I", *(round(v * 100000) for v in color._SRGB_CHROMATICITY))),
    ]
    source = _png(np.full((2, 2, 3), 32768, np.uint16), *chunks)
    master = decode_master(source, source_name="photo.png")
    assert master.metadata["color_resolution"] == "png_calibrated_rgb"
    np.testing.assert_allclose(master.pixels, 32768 / 65535, atol=0.0001)


def test_png_icc_precedes_fallback_gamma_but_explicit_cicp_must_agree():
    profile = color._rgb_profile(color._P3_CHROMATICITY)
    # A fallback gamma is not an additional transfer over the embedded profile.
    source = _png(np.eye(3, dtype=np.uint16)[None, ...] * 65535, _icc_chunk(profile), (b"gAMA", struct.pack(">I", 45455)))
    master = decode_master(source, source_name="photo.png")
    assert master.pixels.min() < -0.1
    conflict = _png(np.zeros((2, 2, 3), np.uint8), _icc_chunk(profile), (b"cICP", bytes((1, 13, 0, 1))))
    with pytest.raises(color.InputColorError, match="disagree"):
        _inspect(conflict, "photo.png", "auto_assume_srgb")
    matching = _png(np.zeros((2, 2, 3), np.uint8), _icc_chunk(profile), (b"cICP", bytes((12, 13, 0, 1))))
    assert _inspect(matching, "photo.png")["action"] == "convert"


@pytest.mark.parametrize("kind", ["cicp", "calibrated"])
def test_png_gray_calibration_produces_rgb_without_quantizing(kind):
    ramp = np.arange(2048, dtype=np.uint16).reshape(32, 64) * 31
    chunks = (
        [(b"cICP", bytes((12, 13, 0, 1)))]
        if kind == "cicp"
        else [
            (b"gAMA", struct.pack(">I", 100000)),
            (b"cHRM", struct.pack(">8I", *(round(v * 100000) for v in color._SRGB_CHROMATICITY))),
        ]
    )
    source = _png(ramp, *chunks)
    master = decode_master(source, source_name="photo.png")
    assert master.pixels.shape == (*ramp.shape, 3)
    assert np.unique(master.pixels[..., 0]).size == 2048
    assert _inspect(source, "photo.png") == metadata_payload(master.metadata["color_preparation"])


def test_cms_unavailable_fails_closed_but_analytical_srgb_still_works(monkeypatch):
    from types import SimpleNamespace

    profile = imagecodecs.cms_profile("adobergb")
    monkeypatch.setattr(imagecodecs, "CMS", SimpleNamespace(available=False))
    with pytest.raises(color.InputColorError, match="governed imagecodecs CMS runtime"):
        _prepare(profile, input_color="auto_assume_srgb")
    assert _prepare(color.output_srgb_icc()).evidence["engine"] == "numpy"


def test_png16_missing_decoder_is_explicit_in_preview_and_execution(monkeypatch):
    from types import SimpleNamespace

    source = _png(np.zeros((2, 2, 3), np.uint16), (b"sRGB", b"\0"))
    monkeypatch.setattr(imagecodecs, "PNG", SimpleNamespace(available=False))
    for operation in (lambda: _inspect(source, "photo.png"), lambda: decode_master(source, source_name="photo.png")):
        with pytest.raises(ValueError, match="16-bit PNG ingest requires"):
            operation()


def test_png_duplicate_cicp_cannot_authorize_an_assumption():
    source = _png(np.zeros((2, 2, 3), np.uint8), (b"cICP", bytes((1, 13, 0, 1))), (b"cICP", bytes((1, 13, 0, 1))))
    with pytest.raises(color.InputColorError, match="Duplicate"):
        _inspect(source, "photo.png", "auto_assume_srgb")


def test_png_late_exif_bad_checksum_cannot_authorize_an_assumption():
    source = _png(np.zeros((2, 2, 3), np.uint8))
    exif = Image.Exif()
    exif[34665] = {40961: 1}
    data = exif.tobytes()[6:]
    chunk = struct.pack(">I", len(data)) + b"eXIf" + data + b"\0\0\0\0"
    source = source[:-12] + chunk + source[-12:]
    with pytest.raises(color.InputColorError, match="checksum"):
        _inspect(source, "photo.png", "auto_assume_srgb")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"profile": color.output_srgb_icc()},
        {"profile": None, "input_color": "auto_assume_srgb"},
        {"profile": None, "png_metadata": {"cicp": [12, 13, 0, 1]}, "image_format": "PNG"},
        {
            "profile": None,
            "png_metadata": {"gamma": 1.0, "chromaticity": list(color._SRGB_CHROMATICITY)},
            "image_format": "PNG",
        },
    ],
)
def test_receipt_metadata_reproduces_the_exact_color_recipe(kwargs):
    prepared = _prepare(**kwargs)
    receipt = prepared.evidence
    metadata = receipt["source_metadata"]
    reproduced = color.prepare_input_color(
        input_color=receipt["requested_input_color"],
        profile=prepared.source_profile,
        image_format=metadata["image_format"],
        declared_color=metadata["declared_color"],
        exif_color=metadata["exif_color"],
        png_metadata=metadata["png"],
    )
    assert reproduced.evidence == receipt
    # Evidence is detached; modifying it cannot mutate the prepared recipe.
    metadata["png"]["srgb"] = 99
    assert prepared.evidence == reproduced.evidence


def test_shared_icc_tag_ranges_do_not_amplify_metadata_allocation():
    import tracemalloc

    count, payload_size = 128, 64 * 1024
    offset = 132 + 12 * count
    profile = bytearray(imagecodecs.cms_profile("srgb")[:128])
    struct.pack_into(">I", profile, 0, offset + payload_size)
    profile.extend(struct.pack(">I", count))
    for index in range(count):
        profile.extend(struct.pack(">III", index, offset, payload_size))
    profile.extend(b"data" + bytes(payload_size - 4))
    snapshot = bytes(profile)
    tracemalloc.start()
    try:
        parsed = color._profile_tags(snapshot)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(parsed) == count
    # Shared ICC tag payloads must not allocate count * payload_size copies.
    assert peak < 1024 * 1024


_TIFF_CALIBRATION_TAGS = [
    (301, "H", 768, tuple(range(256)) * 3, False),
    (318, "2I", 2, (3127, 10000, 3290, 10000), False),
    (319, "2I", 6, (64, 100, 33, 100, 21, 100, 71, 100, 15, 100, 6, 100), False),
    (532, "2I", 6, (0, 1, 255, 1, 0, 1, 255, 1, 0, 1, 255, 1), False),
]


def _calibrated_tiff(tag, *, icc=None):
    stream = io.BytesIO()
    tifffile.imwrite(
        stream,
        np.full((2, 2, 3), 128, np.uint8),
        photometric="rgb",
        metadata=None,
        iccprofile=icc,
        extratags=[tag],
    )
    return stream.getvalue()


@pytest.mark.parametrize(
    "tag", _TIFF_CALIBRATION_TAGS, ids=["transfer_function", "whitepoint", "primaries", "reference_black_white"]
)
@pytest.mark.parametrize("mode", ["auto", "auto_assume_srgb"])
def test_tiff_calibration_without_icc_never_becomes_an_untagged_assumption(tag, mode):
    source = _calibrated_tiff(tag)
    for operation in (
        lambda: _inspect(source, "calibrated.tif", mode),
        lambda: decode_master(source, source_name="calibrated.tif", input_color=mode),
    ):
        with pytest.raises(color.InputColorError, match="TIFF calibration metadata") as error:
            operation()
        assert error.value.code == "input_color_unsupported_metadata"


@pytest.mark.parametrize(
    "tag", _TIFF_CALIBRATION_TAGS, ids=["transfer_function", "whitepoint", "primaries", "reference_black_white"]
)
@pytest.mark.parametrize("mode", ["srgb", "linear_srgb"])
def test_tiff_calibration_keeps_intentional_color_overrides(tag, mode):
    source = _calibrated_tiff(tag)
    master = decode_master(source, source_name="calibrated.tif", input_color=mode)
    receipt = metadata_payload(master.metadata["color_preparation"])
    assert receipt["action"] == "explicit" and not receipt["assumed_srgb"]
    assert _inspect(source, "calibrated.tif", mode) == receipt


def test_tiff_icc_remains_authority_over_fallback_calibration():
    source = _calibrated_tiff(_TIFF_CALIBRATION_TAGS[2], icc=imagecodecs.cms_profile("adobergb"))
    master = decode_master(source, source_name="calibrated.tif", input_color="auto_assume_srgb")
    receipt = metadata_payload(master.metadata["color_preparation"])
    assert receipt["action"] == "convert" and not receipt["assumed_srgb"]
    assert _inspect(source, "calibrated.tif", "auto_assume_srgb") == receipt


@pytest.mark.parametrize(
    "chunk",
    [
        (b"mDCV", struct.pack(">8HII", 34000, 16000, 13250, 34500, 7500, 3000, 15635, 16450, 10000000, 50)),
        (b"cLLI", struct.pack(">II", 10000000, 1000000)),
    ],
)
@pytest.mark.parametrize("mode", ["auto", "auto_assume_srgb"])
def test_standard_png_hdr_metadata_never_becomes_an_untagged_assumption(chunk, mode):
    source = _png(np.full((2, 2, 3), 128, np.uint8), chunk)
    for operation in (
        lambda: _inspect(source, "hdr.png", mode),
        lambda: decode_master(source, source_name="hdr.png", input_color=mode),
    ):
        with pytest.raises(color.InputColorError, match="Unsupported PNG HDR color metadata") as error:
            operation()
        assert error.value.code == "input_color_unsupported_metadata"


@pytest.mark.parametrize("kind,size", [(b"mDCV", 24), (b"cLLI", 8)])
@pytest.mark.parametrize("mode", ["srgb", "linear_srgb"])
def test_png_hdr_metadata_keeps_intentional_color_overrides(kind, size, mode):
    source = _png(np.full((2, 2, 3), 128, np.uint8), (kind, bytes(size)))
    master = decode_master(source, source_name="hdr.png", input_color=mode)
    receipt = metadata_payload(master.metadata["color_preparation"])
    assert receipt["action"] == "explicit" and not receipt["assumed_srgb"]
    assert _inspect(source, "hdr.png", mode) == receipt


@pytest.mark.parametrize("kind", [b"mDCV", b"cLLI"])
def test_malformed_png_hdr_chunks_fail_structure_checks_even_with_override(kind):
    source = _png(np.full((2, 2, 3), 128, np.uint8), (kind, b"\0"))
    for operation in (
        lambda: _inspect(source, "hdr.png", "srgb"),
        lambda: decode_master(source, source_name="hdr.png", input_color="srgb"),
    ):
        with pytest.raises(color.InputColorError, match="supported bounds"):
            operation()


def _jpeg_with_icc_fragments(fragments, *, after_sof=False):
    stream = io.BytesIO()
    Image.new("RGB", (3, 4), (128, 64, 192)).save(stream, format="JPEG")
    source = stream.getvalue()
    offset = 2
    if after_sof:
        marker = source.index(b"\xff\xc0")
        offset = marker + 2 + struct.unpack_from(">H", source, marker + 2)[0]
    markers = b"".join(b"\xff\xe2" + struct.pack(">H", len(payload) + 2) + payload for payload in fragments)
    return source[:offset] + markers + source[offset:]


def _jpeg_icc_fragments(profile):
    middle = len(profile) // 2
    return [b"ICC_PROFILE\0\x01\x02" + profile[:middle], b"ICC_PROFILE\0\x02\x02" + profile[middle:]]


@pytest.mark.parametrize(
    "case", ["incomplete", "duplicate", "sequence_zero", "sequence_gap", "inconsistent_count", "zero_count", "bad_separator"]
)
@pytest.mark.parametrize("mode", ["auto", "auto_assume_srgb"])
def test_malformed_jpeg_icc_fragments_never_become_an_untagged_assumption(case, mode):
    fragments = _jpeg_icc_fragments(color.output_srgb_icc())
    if case == "incomplete":
        fragments.pop()
    elif case == "duplicate":
        fragments[1] = fragments[0]
    elif case == "sequence_zero":
        fragments[0] = fragments[0][:12] + b"\0" + fragments[0][13:]
    elif case == "sequence_gap":
        fragments[1] = fragments[1][:12] + b"\x03" + fragments[1][13:]
    elif case == "inconsistent_count":
        fragments[1] = fragments[1][:13] + b"\x03" + fragments[1][14:]
    elif case == "zero_count":
        fragments = [b"ICC_PROFILE\0\x01\0" + color.output_srgb_icc()]
    else:
        fragments = [b"ICC_PROFILEX\x01\x01" + color.output_srgb_icc()]
    source = _jpeg_with_icc_fragments(fragments)
    for operation in (
        lambda: _inspect(source, "broken.jpg", mode),
        lambda: decode_master(source, source_name="broken.jpg", input_color=mode),
    ):
        with pytest.raises(color.InputColorError, match="Invalid JPEG ICC fragments") as error:
            operation()
        assert error.value.code == "input_color_unsupported_icc"


@pytest.mark.parametrize("after_sof", [False, True])
@pytest.mark.parametrize("reverse_order", [False, True])
def test_valid_jpeg_icc_fragments_are_preserved_and_assembled_in_sequence(after_sof, reverse_order, monkeypatch):
    profile = imagecodecs.cms_profile("adobergb")
    fragments = _jpeg_icc_fragments(profile)
    if reverse_order:
        fragments.reverse()
    source = _jpeg_with_icc_fragments(fragments, after_sof=after_sof)
    master = decode_master(source, source_name="profiled.jpg")
    assert master.source_icc == profile
    receipt = metadata_payload(master.metadata["color_preparation"])
    assert receipt["action"] == "convert" and not receipt["assumed_srgb"]
    assert receipt["source_icc_sha256"] == hashlib.sha256(profile).hexdigest()
    monkeypatch.setattr(Image.Image, "load", lambda *_: pytest.fail("JPEG preflight decoded pixels"))
    assert _inspect(source, "profiled.jpg") == receipt


@pytest.mark.parametrize("mode", ["srgb", "linear_srgb"])
def test_incomplete_jpeg_icc_keeps_intentional_explicit_override(mode):
    source = _jpeg_with_icc_fragments(_jpeg_icc_fragments(color.output_srgb_icc())[:1])
    master = decode_master(source, source_name="broken.jpg", input_color=mode)
    receipt = metadata_payload(master.metadata["color_preparation"])
    assert receipt["action"] == "explicit" and not receipt["assumed_srgb"]
    assert master.source_icc is None
    assert _inspect(source, "broken.jpg", mode) == receipt


@pytest.mark.parametrize("mode", ["auto_assume_srgb", "srgb"])
def test_jpeg_icc_fragment_byte_budget_applies_before_assembly(monkeypatch, mode):
    source = _jpeg_with_icc_fragments(_jpeg_icc_fragments(color.output_srgb_icc()))
    from transformation_portal.lux_depth_v4 import photography

    monkeypatch.setattr(photography, "MAX_ICC_BYTES", 32)
    for operation in (
        lambda: _inspect(source, "profiled.jpg", mode),
        lambda: decode_master(source, source_name="profiled.jpg", input_color=mode),
    ):
        with pytest.raises(color.InputColorError, match="fragment bounds"):
            operation()


@pytest.mark.parametrize("mode", ["auto_assume_srgb", "srgb"])
def test_jpeg_icc_fragment_count_is_bounded_even_for_explicit_override(mode):
    source = _jpeg_with_icc_fragments([b"ICC_PROFILE\0\x01\x01"] * 256)
    for operation in (
        lambda: _inspect(source, "fragments.jpg", mode),
        lambda: decode_master(source, source_name="fragments.jpg", input_color=mode),
    ):
        with pytest.raises(color.InputColorError, match="fragment bounds"):
            operation()
