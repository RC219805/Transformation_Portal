"""Header preflight preserves color policy without decoding or escaping inputs."""

from __future__ import annotations

import io
import os
import struct
import zlib

import numpy as np
import pytest
import tifffile
from PIL import Image, ImageCms

from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
from transformation_portal.lux_depth_v4 import input_preflight
from transformation_portal.lux_depth_v4.photography import InputColorError, output_srgb_icc, validate_input_color_metadata

pytestmark = pytest.mark.unit


def _inspect(root, **kwargs):
    options = {"input_color": "auto", "max_input_bytes": 1024**2, "max_pixels": 10_000}
    options.update(kwargs)
    input_preflight.validate_input_directory_colors(root, **options)


@pytest.mark.parametrize("extension", ["jpg", "tif"])
def test_unsupported_profile_is_rejected_without_decoding_pixels(tmp_path, monkeypatch, extension):
    path = tmp_path / f"photo.{extension}"
    if extension == "jpg":
        Image.new("RGB", (14, 14)).save(path, icc_profile=b"Adobe RGB profile bytes")
    else:
        tifffile.imwrite(path, np.zeros((14, 14, 3), np.uint16), photometric="rgb", iccprofile=b"Adobe RGB profile bytes")
    monkeypatch.setattr(Image.Image, "load", lambda *_: pytest.fail("Preflight decoded pixels"))
    monkeypatch.setattr(tifffile.TiffPage, "asarray", lambda *_: pytest.fail("Preflight decoded TIFF pixels"))
    with pytest.raises(InputColorError, match="Convert the photographs to sRGB") as error:
        _inspect(tmp_path)
    assert error.value.code == "input_color_unsupported_icc"
    # An intentional correction remains the existing explicit ingest contract.
    _inspect(tmp_path, input_color="srgb")


@pytest.mark.parametrize("color", ["srgb", "linear_srgb", None])
def test_tiff_declaration_and_ambiguous_metadata_match_ingest(tmp_path, color):
    tifffile.imwrite(
        tmp_path / "photo.tif",
        np.zeros((14, 14, 3), np.uint16),
        photometric="rgb",
        metadata={"color_space": color} if color else None,
    )
    if color:
        _inspect(tmp_path)
    else:
        with pytest.raises(InputColorError, match="Ambiguous") as error:
            _inspect(tmp_path)
        assert error.value.code == "input_color_ambiguous"


def test_canonical_srgb_and_untagged_jpeg_remain_accepted(tmp_path):
    Image.new("RGB", (14, 14)).save(tmp_path / "tagged.png", icc_profile=output_srgb_icc())
    Image.new("RGB", (14, 14)).save(tmp_path / "untagged.jpg")
    _inspect(tmp_path)


def test_valid_noncanonical_srgb_profile_is_color_managed(tmp_path):
    # Creator metadata changes neither primaries nor curves. Native profile
    # interpretation replaces the historical exact-byte rejection.
    profile = bytearray(output_srgb_icc())
    profile[80:84] = b"TEST"
    assert ImageCms.getProfileName(ImageCms.ImageCmsProfile(io.BytesIO(profile))).strip() == "sRGB built-in"
    Image.new("RGB", (14, 14)).save(tmp_path / "converted.jpg", icc_profile=bytes(profile))
    _inspect(tmp_path)
    with (tmp_path / "converted.jpg").open("rb") as source:
        evidence = validate_input_color_metadata(source, source_name="converted.jpg")
    assert evidence["engine"] == "imagecodecs_lcms"
    assert evidence["action"] == "convert"


@pytest.mark.parametrize("metadata", ["canonical_icc", "late_exif", "untagged"])
def test_png_color_headers_never_decode_pixels(tmp_path, monkeypatch, metadata):
    path = tmp_path / "photo.png"
    options = {"icc_profile": output_srgb_icc()} if metadata == "canonical_icc" else {}
    Image.new("RGB", (14, 14)).save(path, **options)
    if metadata == "late_exif":
        exif = Image.Exif()
        exif[34665] = {40961: 1}
        payload = exif.tobytes()[6:]
        chunk = b"eXIf" + payload
        data = path.read_bytes()
        path.write_bytes(
            data[:-12] + struct.pack(">I", len(payload)) + chunk + struct.pack(">I", zlib.crc32(chunk)) + data[-12:]
        )
    monkeypatch.setattr(Image.Image, "load", lambda *_: pytest.fail("PNG preflight decoded pixels"))
    if metadata == "untagged":
        with pytest.raises(InputColorError, match="Ambiguous"):
            _inspect(tmp_path)
    else:
        _inspect(tmp_path)


@pytest.mark.parametrize("input_color", ["auto", "linear_srgb", "srgb"])
def test_raw_keeps_governed_decoder_color_rule(input_color):
    source = io.BytesIO(b"raw-source")
    if input_color == "srgb":
        with pytest.raises(InputColorError, match="RAW decoder output is linear_srgb"):
            validate_input_color_metadata(source, source_name="photo.dng", input_color=input_color)
    else:
        validate_input_color_metadata(source, source_name="photo.dng", input_color=input_color)


@pytest.mark.parametrize("budget", ["file_bytes", "pixels", "metadata_file", "metadata_batch", "inventory"])
def test_preflight_work_is_bounded(tmp_path, monkeypatch, budget):
    Image.new("RGB", (14, 14)).save(tmp_path / "photo.jpg")
    options = {}
    if budget == "file_bytes":
        options["max_input_bytes"] = 1
    elif budget == "pixels":
        options["max_pixels"] = 1
    elif budget == "metadata_file":
        monkeypatch.setattr(input_preflight, "_MAX_FILE_METADATA_BYTES", 1)
    elif budget == "metadata_batch":
        monkeypatch.setattr(input_preflight, "_MAX_BATCH_METADATA_BYTES", 1)
    else:
        monkeypatch.setattr(input_preflight, "_MAX_ENTRIES", 0)
    with pytest.raises(ValueError):
        _inspect(tmp_path, **options)


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "hardlink"])
def test_preflight_rejects_linked_inputs_before_parsing(tmp_path, monkeypatch, kind):
    source = tmp_path / "source"
    source.mkdir()
    outside = tmp_path / "photo.jpg"
    Image.new("RGB", (14, 14)).save(outside)
    if kind == "file_symlink":
        (source / "photo.jpg").symlink_to(outside)
    elif kind == "directory_symlink":
        (source / "linked").symlink_to(tmp_path, target_is_directory=True)
    else:
        os.link(outside, source / "photo.jpg")
    monkeypatch.setattr(input_preflight, "validate_input_color_metadata", lambda *_args, **_kwargs: pytest.fail("Read link"))
    with pytest.raises((ValueError, ArtifactEvidenceError)):
        _inspect(source)


def test_preflight_detects_replaced_file_after_header_read(tmp_path, monkeypatch):
    source = tmp_path / "photo.jpg"
    Image.new("RGB", (14, 14)).save(source)
    parse = input_preflight.validate_input_color_metadata

    def replace_after_parse(*args, **kwargs):
        parse(*args, **kwargs)
        replacement = tmp_path / "replacement"
        replacement.write_bytes(source.read_bytes())
        replacement.replace(source)

    monkeypatch.setattr(input_preflight, "validate_input_color_metadata", replace_after_parse)
    with pytest.raises(ArtifactEvidenceError, match="captured inode"):
        _inspect(tmp_path)
