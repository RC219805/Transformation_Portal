"""Cross-stage pixel fidelity and geometry contracts for Lux Depth V4."""

from __future__ import annotations

import io

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageOps

from transformation_portal.core.depth_artifact import DepthArtifact
from transformation_portal.core.image_artifact import ImageMaster, ProxyTransform
from transformation_portal.lux_depth_v4.photography import (
    apply_materials,
    create_proxy,
    decode_master,
    enhance_master,
    generate_preview_maps,
    linear_to_srgb,
    output_srgb_icc,
    restore_depth,
    restore_depth_with_validity,
    restore_mask,
    srgb_to_linear,
    write_delivery,
)

pytestmark = pytest.mark.unit


def _tiff(array, *, orientation=1, color_space="linear_srgb", icc=None):
    stream = io.BytesIO()
    tifffile.imwrite(
        stream,
        array,
        photometric="rgb",
        metadata={"color_space": color_space} if color_space is not None else None,
        iccprofile=icc,
        extratags=[(274, "H", 1, orientation, False)],
        extrasamples="unassalpha" if array.shape[-1] == 4 else None,
    )
    return stream.getvalue()


def _master(array, alpha=None):
    return ImageMaster(array, "a" * 64, 16, alpha=alpha)


@pytest.mark.parametrize("orientation", range(1, 9))
@pytest.mark.parametrize("image_format", ["JPEG", "TIFF"])
def test_orientation_agrees_with_pillow_for_master_and_proxy(orientation, image_format):
    array = np.arange(28 * 42 * 3, dtype=np.uint16).reshape(28, 42, 3)
    if image_format == "TIFF":
        source = _tiff(array, orientation=orientation)
        # Apply Pillow's orientation operation to distinct float channel data
        # as a reference, without Pillow's 16-bit RGB decoding path.
        reference = Image.fromarray(array[..., 0].astype(np.float32))
        reference.getexif()[274] = orientation
        expected = np.asarray(ImageOps.exif_transpose(reference)) / 65535
    else:
        stream = io.BytesIO()
        exif = Image.Exif()
        exif[274] = orientation
        Image.fromarray((array % 256).astype(np.uint8)).save(stream, format="JPEG", exif=exif)
        source = stream.getvalue()
        with Image.open(io.BytesIO(source)) as reference:
            expected = srgb_to_linear(np.asarray(ImageOps.exif_transpose(reference)).astype(np.float32) / 255)[..., 0]
    master = decode_master(source, source_name=f"source.{image_format.lower()}")
    np.testing.assert_allclose(master.pixels[..., 0], expected, rtol=1e-6)
    proxy = create_proxy(master, target_size=518)
    assert proxy.transform.original_shape == expected.shape
    assert master.metadata["orientation_normalized"] is True


def test_uint16_levels_survive_decode_finish_and_delivery(tmp_path):
    source = np.repeat((np.arange(1176, dtype=np.uint16).reshape(28, 42) + 10000)[..., None], 3, axis=2)
    master = decode_master(_tiff(source, color_space="srgb"), source_name="source.tif")
    finished = enhance_master(master, strength=0, clarity=0)
    assert np.unique(finished.pixels).size == 1176
    report = write_delivery(finished, tmp_path / "delivery.tif")
    output = tifffile.imread(report["path"])
    np.testing.assert_array_equal(output, source)
    assert report["source_bit_depth"] == report["output_bit_depth"] == 16
    with tifffile.TiffFile(report["path"]) as tif:
        assert tif.pages[0].tags[34675].value == output_srgb_icc()
        assert tif.pages[0].tags[274].value == 1


def test_alpha_is_separate_and_not_transfer_encoded(tmp_path):
    rgb = np.full((28, 28, 3), 32768, np.uint16)
    alpha = np.arange(784, dtype=np.uint16).reshape(28, 28) * 80
    source = np.concatenate([rgb, alpha[..., None]], axis=2)
    master = decode_master(_tiff(source, color_space="srgb"), source_name="alpha.tif")
    np.testing.assert_array_equal(master.alpha, alpha.astype(np.float32) / 65535)
    report = write_delivery(enhance_master(master, strength=0), tmp_path / "alpha.tif")
    np.testing.assert_array_equal(tifffile.imread(report["path"])[..., 3], alpha)
    assert report["alpha_preserved"]


@pytest.mark.parametrize("mode,key", [("RGB", (255, 0, 0)), ("L", 128)])
def test_png_color_key_transparency_survives_ingest_and_delivery(mode, key, tmp_path):
    shape = (14, 14, 3) if mode == "RGB" else (14, 14)
    pixels = np.zeros(shape, np.uint8)
    pixels[:7] = key
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG", transparency=key)
    with Image.open(io.BytesIO(stream.getvalue())) as reference:
        expected = np.asarray(reference.convert("RGBA"))
    master = decode_master(stream.getvalue(), source_name="transparent.png", input_color="srgb")
    np.testing.assert_array_equal(master.alpha, expected[..., 3].astype(np.float32) / 255)
    np.testing.assert_allclose(master.pixels, srgb_to_linear(expected[..., :3].astype(np.float32) / 255))
    report = write_delivery(master, tmp_path / "delivery.tif")
    np.testing.assert_array_equal(tifffile.imread(report["path"])[..., 3], expected[..., 3].astype(np.uint16) * 257)


def test_float_master_retains_out_of_range_until_delivery(tmp_path):
    array = np.array([[[-0.1, 0.5, 2.0]]], dtype=np.float32)
    master = decode_master(_tiff(array), source_name="hdr.tif")
    np.testing.assert_array_equal(master.pixels, array)
    report = write_delivery(master, tmp_path / "bounded.tif")
    assert report["clipped_low_sample_fraction"] == pytest.approx(1 / 3)
    assert report["clipped_high_sample_fraction"] == pytest.approx(1 / 3)
    np.testing.assert_array_equal(master.pixels, array)


def test_transfer_function_has_known_midgray_and_roundtrip():
    np.testing.assert_allclose(srgb_to_linear(np.array([0.5], np.float32)), [0.21404114], atol=1e-7)
    values = np.array([-0.1, 0, 0.0031308, 0.2, 1, 2], np.float32)
    np.testing.assert_allclose(srgb_to_linear(linear_to_srgb(values)), values, atol=3e-7)


def test_unknown_tiff_color_and_icc_fail_unless_corrected():
    array = np.full((14, 14, 3), 32768, np.uint16)
    untagged = _tiff(array, color_space=None)
    with pytest.raises(ValueError, match="Ambiguous input color"):
        decode_master(untagged, source_name="unknown.tif")
    explicit = decode_master(untagged, source_name="unknown.tif", input_color="srgb")
    assert explicit.metadata["color_resolution"] == "explicit_input_color"
    unsupported = _tiff(array, color_space=None, icc=b"unsupported-profile")
    with pytest.raises(ValueError, match="Unsupported ICC"):
        decode_master(unsupported, source_name="unknown.tif")
    corrected = decode_master(unsupported, source_name="unknown.tif", input_color="linear_srgb")
    assert corrected.source_icc == b"unsupported-profile"


def test_recognized_srgb_profile_and_conflicting_metadata():
    array = np.full((14, 14, 3), 128, np.uint8)
    recognized = decode_master(_tiff(array, color_space=None, icc=output_srgb_icc()), source_name="srgb.tif")
    assert recognized.metadata["color_resolution"] == "recognized_srgb_icc"
    with pytest.raises(ValueError, match="disagree"):
        decode_master(_tiff(array, icc=output_srgb_icc()), source_name="contradictory.tif")
    assert output_srgb_icc() == output_srgb_icc()


def test_untagged_jpeg_records_color_assumption():
    stream = io.BytesIO()
    Image.new("RGB", (28, 28), (128, 128, 128)).save(stream, format="JPEG")
    master = decode_master(stream.getvalue(), source_name="source.jpg")
    assert master.source_bit_depth == 8
    assert master.metadata["color_resolution"] == "untagged_jpeg_srgb_assumption"


def test_raw_requires_governed_callback_and_linear_declaration():
    with pytest.raises(ValueError, match="governed decoder"):
        decode_master(b"raw-source", source_name="source.cr2")
    array = np.full((14, 14, 3), 1.2, np.float32)
    master = decode_master(
        b"raw-source",
        source_name="source.cr2",
        raw_decoder=lambda snapshot, name: (array, {"color_space": "linear_srgb", "ingest_fingerprint": "test"}),
    )
    np.testing.assert_array_equal(master.pixels, array)
    with pytest.raises(ValueError, match="declare linear_srgb"):
        decode_master(b"raw-source", source_name="source.cr2", raw_decoder=lambda *_: (array, {"color_space": "srgb"}))


def test_proxy_padding_restore_does_not_crop_or_stretch():
    pixels = np.zeros((28, 41, 3), np.float32)
    pixels[:, 0] = 1
    pixels[:, -1] = 0.5
    proxy = create_proxy(_master(pixels))
    assert proxy.transform.resized_shape == (28, 41)
    assert proxy.transform.padded_shape == (28, 42)
    depth = np.tile(np.arange(42, dtype=np.float32), (28, 1))
    restored = restore_depth(depth, proxy.transform)
    np.testing.assert_array_equal(restored, depth[:, :41])
    assert proxy.pixels[0, 0, 0] == 255
    assert proxy.pixels[0, 40, 0] == proxy.pixels[0, 41, 0]
    assert ProxyTransform.from_payload(proxy.transform.to_payload()) == proxy.transform
    mask = depth == 0
    np.testing.assert_array_equal(restore_mask(mask, proxy.transform), mask[:, :41])


def test_resized_proxy_restoration_preserves_full_extent():
    proxy = create_proxy(_master(np.full((100, 200, 3), 0.25, np.float32)), target_size=56)
    assert proxy.transform.resized_shape == (28, 56)
    restored = restore_depth(np.ones((14, 28), np.float32), proxy.transform)
    assert restored.shape == (100, 200)
    np.testing.assert_array_equal(restored, np.ones((100, 200), np.float32))


def test_finishing_never_quantizes_or_mutates_master():
    pixels = np.repeat(np.linspace(0.1, 0.11, 1176, dtype=np.float32).reshape(28, 42)[..., None], 3, axis=2)
    master = _master(pixels)
    original_hash = master.content_hash()
    depth = np.tile(np.linspace(0, 1, 42, dtype=np.float32), (28, 1))
    finished = enhance_master(master, depth, strength=0.25, clarity=0.2)
    assert finished.pixels.dtype == np.float32
    assert np.unique(finished.pixels).size > 1000
    assert master.content_hash() == original_hash
    assert finished.content_hash() != original_hash
    with pytest.raises(ValueError, match="geometry"):
        enhance_master(master, np.ones((42, 28), np.float32))


def test_associated_tiff_alpha_is_rejected():
    stream = io.BytesIO()
    tifffile.imwrite(stream, np.zeros((14, 14, 4), np.uint16), photometric="rgb", extrasamples="assocalpha")
    with pytest.raises(ValueError, match="straight"):
        decode_master(stream.getvalue(), source_name="associated.tif", input_color="srgb")


def test_packed_tiff_precision_is_rejected_before_pixel_decode(monkeypatch):
    # Patch the TIFF precision tag in a fixture rather than requiring an
    # optional packed-integer codec to construct this header regression.
    stream = io.BytesIO(_tiff(np.full((14, 14, 3), 4095, np.uint16)))
    with tifffile.TiffFile(stream, mode="r+b") as tif:
        tif.pages[0].tags["BitsPerSample"].overwrite((12, 12, 12))

    def forbidden(*_args, **_kwargs):
        pytest.fail("Packed TIFF was decoded before its precision was rejected")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden)
    with pytest.raises(ValueError, match="BitsPerSample"):
        decode_master(stream.getvalue(), source_name="packed.tif")


def test_depth_remapping_excludes_invalid_samples_and_preserves_valid_behavior():
    transform = ProxyTransform((56, 84), (28, 42), (28, 42))
    values = np.full((28, 42), 3.0, np.float32)
    valid = np.ones(values.shape, bool)
    valid[8:20, 15:27] = False
    values[~valid] = -1000
    aligned, aligned_valid = restore_depth_with_validity(values, valid, transform)
    assert aligned_valid.dtype == np.bool_ and aligned_valid.shape == transform.original_shape
    assert not aligned_valid[16:40, 30:54].any()
    np.testing.assert_array_equal(aligned[~aligned_valid], 0)
    np.testing.assert_allclose(aligned[aligned_valid], 3, rtol=1e-7)

    finite = np.arange(28 * 42, dtype=np.float32).reshape(28, 42)
    unchanged, all_valid = restore_depth_with_validity(finite, np.ones_like(valid), transform)
    np.testing.assert_array_equal(unchanged, restore_depth(finite, transform))
    assert all_valid.all()


def test_depth_finishing_abstains_at_invalid_aligned_samples():
    master = _master(np.full((28, 42, 3), 0.25, np.float32))
    depth = np.linspace(0, 1, 28 * 42, dtype=np.float32).reshape(28, 42)
    valid = np.ones(master.shape, bool)
    valid[8:20, 15:27] = False
    finished = enhance_master(master, depth, strength=1, valid_mask=valid)
    np.testing.assert_array_equal(finished.pixels[~valid], master.pixels[~valid])
    assert np.any(finished.pixels[valid] != master.pixels[valid])
    empty = enhance_master(master, depth, strength=1, valid_mask=np.zeros(master.shape, bool))
    np.testing.assert_array_equal(empty.pixels, master.pixels)
    assert not empty.metadata["finishing"]["depth_applied"]
    with pytest.raises(ValueError, match="boolean mask"):
        enhance_master(master, depth, valid_mask=valid.astype(np.float32))


def test_tiff_limit_is_checked_before_pixel_decode(monkeypatch):
    source = _tiff(np.ones((28, 42, 3), np.uint16))

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("oversized pixel allocation occurred")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden_decode)
    with pytest.raises(ValueError, match="max_pixels"):
        decode_master(source, source_name="large.tif", max_pixels=100)


def test_jpeg_limit_is_checked_before_pixel_decode(monkeypatch):
    stream = io.BytesIO()
    Image.new("RGB", (28, 42)).save(stream, format="JPEG")
    from PIL import JpegImagePlugin

    def forbidden_decode(*args, **kwargs):
        raise AssertionError("oversized pixel allocation occurred")

    monkeypatch.setattr(JpegImagePlugin.JpegImageFile, "load", forbidden_decode)
    with pytest.raises(ValueError, match="max_pixels"):
        decode_master(stream.getvalue(), source_name="large.jpg", max_pixels=100)


def test_materials_default_and_uncertain_evidence_abstain():
    master = _master(np.full((28, 42, 3), 0.25, np.float32))
    unchanged, report = apply_materials(master)
    assert unchanged is master
    assert report["status"] == "abstained"
    assert report["reason"] == "no_supplied_material_masks"
    mask = np.ones(master.shape, np.float32)
    unchanged, report = apply_materials(
        master,
        {"glass": mask, "water": mask, "fictional": mask},
        {"water": 0.2, "fictional": 1.0},
    )
    assert unchanged is master
    assert report["materials"]["glass"]["reason"] == "missing_confidence"
    assert report["materials"]["water"]["reason"] == "below_confidence_threshold"
    assert report["materials"]["fictional"]["reason"] == "unsupported_material"


def test_supplied_material_operations_are_float_and_deterministic():
    pixels = np.repeat(np.linspace(0.2, 0.21, 1176, dtype=np.float32).reshape(28, 42)[..., None], 3, axis=2)
    master = _master(pixels)
    mask = np.ones(master.shape, np.float32)
    first, report = apply_materials(master, {"water": mask}, {"water": 0.9})
    second, repeated = apply_materials(master, {"water": mask}, {"water": 0.9})
    assert first.pixels.dtype == np.float32
    assert np.unique(first.pixels).size > 1000
    assert report["status"] == "applied"
    assert report["segmentation_inferred"] is False
    assert report == repeated
    assert first.content_hash() == second.content_hash()
    assert first.content_hash() != master.content_hash()
    np.testing.assert_array_equal(master.pixels, pixels)


def test_material_adapter_preserves_unbounded_highlights():
    pixels = np.full((28, 42, 3), 2.0, np.float32)
    master = _master(pixels)
    finished, _ = apply_materials(master, {"glass": np.ones(master.shape, np.float32)}, {"glass": 0.9})
    np.testing.assert_array_equal(finished.pixels, pixels)


def test_material_evidence_shape_confidence_and_range_fail_closed():
    master = _master(np.full((28, 42, 3), 0.25, np.float32))
    with pytest.raises(ValueError, match="matching the master"):
        apply_materials(master, {"water": np.ones((42, 28), np.float32)}, {"water": 0.9})
    with pytest.raises(ValueError, match="finite"):
        apply_materials(master, {"water": np.full(master.shape, np.nan)}, {"water": 0.9})
    with pytest.raises(ValueError, match="confidence"):
        apply_materials(master, {"water": np.ones(master.shape)}, {"water": float("nan")})
    with pytest.raises(ValueError, match="absent mask"):
        apply_materials(master, {}, {"water": 0.9})


def test_preview_maps_are_labelled_proxies_and_bind_source_depth():
    valid = np.ones((28, 42), bool)
    valid[0, 0] = False
    depth = DepthArtifact(np.arange(1176, dtype=np.float32).reshape(28, 42), "relative_distance", valid, "a" * 64)
    maps, report = generate_preview_maps(depth)
    assert report["kind"] == "depth_derived_preview"
    assert report["physical_material_estimate"] is False
    assert report["depth_content_hash"] == depth.content_hash()
    assert set(maps) == {"normal", "roughness", "ao"}
    np.testing.assert_array_equal(maps["normal"][0, 0], [128, 128, 255])
    assert maps["roughness"][0, 0] == 0
    assert maps["ao"][0, 0] == 255


@pytest.mark.parametrize("volumetric", [True, False])
def test_unsupported_tiff_allocation_is_rejected_from_header(monkeypatch, volumetric):
    stream = io.BytesIO()
    values = np.zeros((3, 2, 2, 3) if volumetric else (2, 2, 3), dtype=np.uint16 if volumetric else np.float64)
    tifffile.imwrite(stream, values, photometric="rgb", volumetric=volumetric)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unsupported TIFF was allocated before header validation")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden)
    with pytest.raises(ValueError, match="geometry|samples"):
        decode_master(stream.getvalue(), source_name="source.tif", input_color="srgb", max_pixels=4)
