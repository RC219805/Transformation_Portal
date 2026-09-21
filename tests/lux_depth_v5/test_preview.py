"""Browser derivatives retain color, transparency, aspect ratio, and bounded storage."""

import hashlib
import io

import numpy as np
import pytest
from PIL import Image

from transformation_portal.core.image_artifact import ImageMaster
from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.lux_depth_v5.preview import (
    MAX_PREVIEW_BYTES,
    MAX_PREVIEW_EDGE,
    encode_preview,
    preview_samples,
    preview_shape,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("shape,expected", [((29, 43), (29, 43)), ((2400, 3200), (1200, 1600)), ((4000, 1), (1600, 1))])
def test_preview_geometry_is_bounded_and_never_upscaled(shape, expected):
    assert preview_shape(shape) == expected
    assert max(expected) <= MAX_PREVIEW_EDGE


def test_browser_preview_encodes_linear_master_as_srgb_and_preserves_it():
    pixels = np.full((3, 7, 3), 0.5, np.float32)
    master = ImageMaster(pixels, "a" * 64, 16)
    before = master.content_hash()
    raw, descriptor = encode_preview(master, relative_path="input-0000/preview.png")
    assert len(raw) < MAX_PREVIEW_BYTES
    assert descriptor["shape"] == [3, 7]
    assert descriptor["master_content_hash"] == before == master.content_hash()
    with Image.open(io.BytesIO(raw)) as image:
        assert image.mode == "RGB" and image.format == "PNG"
        assert image.info["icc_profile"] == output_srgb_icc()
        np.testing.assert_array_equal(np.asarray(image), np.full((3, 7, 3), 188, np.uint8))


def test_area_reduction_uses_premultiplied_alpha_without_color_bleed():
    pixels = np.zeros((2, 3200, 3), np.float32)
    pixels[:, ::2, 0] = 1  # Invisible red must not bleed into opaque blue.
    pixels[:, 1::2, 2] = 1
    alpha = np.zeros((2, 3200), np.float32)
    alpha[:, 1::2] = 1
    samples = preview_samples(ImageMaster(pixels, "a" * 64, 16, alpha))
    assert samples.shape == (1, 1600, 4)
    np.testing.assert_array_equal(samples[..., :3], np.broadcast_to([0, 0, 255], (1, 1600, 3)))
    np.testing.assert_array_equal(samples[..., 3], np.full((1, 1600), 128, np.uint8))


def test_area_reduction_averages_linear_light_before_encoding():
    pixels = np.zeros((2, 3200, 3), np.float32)
    pixels[:, 1::2] = 1
    samples = preview_samples(ImageMaster(pixels, "a" * 64, 16))
    np.testing.assert_array_equal(samples, np.full((1, 1600, 3), 188, np.uint8))


@pytest.mark.parametrize("mutation", ["geometry", "encoded_size"])
def test_preview_bounds_are_enforced_before_decoding_pixels(tmp_path, monkeypatch, mutation):
    from PIL import PngImagePlugin

    from transformation_portal.lux_depth_v5.evidence import _verify_browser_preview

    master = ImageMaster(np.full((3, 7, 3), 0.5, np.float32), "a" * 64, 16)
    relative = "input-0000/preview.png"
    raw, descriptor = encode_preview(master, relative_path=relative)
    if mutation == "geometry":
        buffer = io.BytesIO()
        with Image.new("RGB", (MAX_PREVIEW_EDGE + 1, 1)) as image:
            image.save(buffer, format="PNG", icc_profile=output_srgb_icc())
        raw = buffer.getvalue()
    else:
        raw += b"\0" * (MAX_PREVIEW_BYTES + 1 - len(raw))
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(raw)
    record = {"kind": "image", "path": relative, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", lambda *_a, **_kw: pytest.fail("Unbounded PNG reached decode"))
    with pytest.raises(ValueError, match="geometry|bounded regular file"):
        _verify_browser_preview(tmp_path, relative, master, descriptor, {relative: record})


def test_palette_png_cannot_substitute_the_declared_rgb_encoding(tmp_path):
    from transformation_portal.lux_depth_v5.evidence import _verify_browser_preview

    master = ImageMaster(np.full((3, 7, 3), 0.5, np.float32), "a" * 64, 16)
    relative = "input-0000/preview.png"
    _, descriptor = encode_preview(master, relative_path=relative)
    expected = preview_samples(master)
    buffer = io.BytesIO()
    with Image.fromarray(expected) as image:
        with image.convert("P", palette=Image.Palette.ADAPTIVE) as indexed:
            indexed.save(buffer, format="PNG", icc_profile=output_srgb_icc())
    raw = buffer.getvalue()
    with Image.open(io.BytesIO(raw)) as image:
        assert image.mode == "P"
        np.testing.assert_array_equal(np.asarray(image.convert("RGB")), expected)
    path = tmp_path / relative
    path.parent.mkdir()
    path.write_bytes(raw)
    record = {"kind": "image", "path": relative, "size_bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError, match="browser preview.*encoding"):
        _verify_browser_preview(tmp_path, relative, master, descriptor, {relative: record})
