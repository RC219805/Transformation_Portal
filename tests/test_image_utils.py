"""Tests for common image utilities module.

Note: These tests assume the package is installed in development mode.
Run `pip install -e .` from the repository root before running tests.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.utils.image_utils import load_image, load_image_rgb, np_to_pil, pil_to_np, save_image


def test_load_image():
    """Test loading an image converts to RGB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        # Create a test image
        test_img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        test_img.save(img_path)

        # Load it back
        loaded = load_image(img_path)
        assert isinstance(loaded, Image.Image)
        assert loaded.mode == "RGB"
        assert loaded.size == (100, 100)


def test_save_image():
    """Test saving an image creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "subdir" / "test.png"
        test_img = Image.new("RGB", (50, 50), color=(0, 255, 0))

        save_image(test_img, output_path)

        assert output_path.exists()
        loaded = Image.open(output_path)
        assert loaded.mode == "RGB"
        assert loaded.size == (50, 50)


def test_pil_to_np_float():
    """Test converting PIL to NumPy with float normalization."""
    img = Image.new("RGB", (10, 10), color=(255, 128, 0))
    arr = pil_to_np(img, to_float=True)

    assert arr.dtype == np.float32
    assert arr.shape == (10, 10, 3)
    assert np.allclose(arr[0, 0], [1.0, 128 / 255.0, 0.0], atol=0.01)


def test_pil_to_np_uint8():
    """Test converting PIL to NumPy without normalization."""
    img = Image.new("RGB", (10, 10), color=(255, 128, 0))
    arr = pil_to_np(img, to_float=False)

    assert arr.dtype == np.uint8
    assert arr.shape == (10, 10, 3)
    assert np.array_equal(arr[0, 0], [255, 128, 0])


def test_np_to_pil():
    """Test converting NumPy float array to PIL Image."""
    arr = np.ones((20, 20, 3), dtype=np.float32) * 0.5
    img = np_to_pil(arr)

    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (20, 20)
    # 0.5 should convert to approximately 128
    pixel = img.getpixel((0, 0))
    assert all(abs(p - 128) <= 1 for p in pixel)


def test_np_to_pil_clipping():
    """Test that np_to_pil clips values to [0, 1]."""
    arr = np.array([[[2.0, -0.5, 0.5]]], dtype=np.float32)
    img = np_to_pil(arr)

    pixel = img.getpixel((0, 0))
    assert pixel[0] == 255  # 2.0 clipped to 1.0 -> 255
    assert pixel[1] == 0  # -0.5 clipped to 0.0 -> 0
    assert abs(pixel[2] - 128) <= 1  # 0.5 -> ~128


def test_load_image_rgb():
    """Test load_image_rgb convenience function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_path = Path(tmpdir) / "test.png"
        test_img = Image.new("RGB", (30, 40), color=(200, 100, 50))
        test_img.save(img_path)

        arr = load_image_rgb(img_path)

        assert arr.dtype == np.float32
        assert arr.shape == (40, 30, 3)  # Note: height, width order
        # Check approximate color values
        assert np.allclose(arr[0, 0], [200 / 255.0, 100 / 255.0, 50 / 255.0], atol=0.01)


def test_load_image_rgb_file_not_found():
    """Test load_image_rgb raises FileNotFoundError for missing file."""
    with pytest.raises(FileNotFoundError, match="Image not found"):
        load_image_rgb("/nonexistent/path/image.png")


def test_roundtrip_conversion():
    """Test that PIL -> NumPy -> PIL roundtrip preserves data."""
    original = Image.new("RGB", (15, 15), color=(100, 150, 200))

    # Convert to numpy and back
    arr = pil_to_np(original, to_float=True)
    reconstructed = np_to_pil(arr)

    # Should be very close (allowing for rounding)
    orig_px = original.getpixel((0, 0))
    recon_px = reconstructed.getpixel((0, 0))
    assert all(abs(o - r) <= 1 for o, r in zip(orig_px, recon_px))


def _cube_text(size, transfer, header=""):
    """Spec-conformant .cube text: red index fastest, then green, then blue."""
    lines = [f"LUT_3D_SIZE {size}"] + ([header] if header else [])
    for b in range(size):
        for g in range(size):
            for r in range(size):
                lines.append(" ".join(f"{v:.6f}" for v in transfer(r / (size - 1), g / (size - 1), b / (size - 1))))
    return "\n".join(lines) + "\n"


def test_load_cube_lut_indexes_table_in_channel_order(tmp_path):
    """lut[r, g, b] holds the entry for input (r, g, b) even though .cube lists red fastest."""
    from transformation_portal.utils.image_utils import load_cube_lut

    cube = tmp_path / "swap.cube"
    cube.write_text(_cube_text(3, lambda r, g, b: (b, g, r)), encoding="utf-8")

    lut = load_cube_lut(cube)

    assert lut.shape == (3, 3, 3, 3)
    assert lut.dtype == np.float32
    np.testing.assert_allclose(lut[2, 0, 0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(lut[0, 1, 2], [1.0, 0.5, 0.0])


def test_load_cube_lut_caches_until_file_changes(tmp_path):
    from transformation_portal.utils.image_utils import load_cube_lut

    cube = tmp_path / "look.cube"
    cube.write_text(_cube_text(2, lambda r, g, b: (r, g, b)), encoding="utf-8")

    first = load_cube_lut(cube)
    assert load_cube_lut(cube) is first
    assert not first.flags.writeable

    cube.write_text(_cube_text(3, lambda r, g, b: (b, g, r)), encoding="utf-8")
    reloaded = load_cube_lut(cube)
    assert reloaded.shape == (3, 3, 3, 3)
    np.testing.assert_allclose(reloaded[2, 0, 0], [0.0, 0.0, 1.0])


@pytest.mark.parametrize(
    "text, message",
    [
        ("LUT_3D_SIZE 1\n0.5 0.5 0.5\n", "LUT_3D_SIZE"),
        ("0 0 0\n1 1 1\n", "LUT_3D_SIZE"),
        ("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n", "expected 8 entries"),
        (_cube_text(2, lambda r, g, b: (r, g, b)).replace("1.000000 1.000000 1.000000", "nan 1 1"), "non-finite"),
    ],
)
def test_load_cube_lut_rejects_malformed_tables(tmp_path, text, message):
    from transformation_portal.utils.image_utils import load_cube_lut

    cube = tmp_path / "bad.cube"
    cube.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cube_lut(cube)


def test_apply_cube_lut_maps_unit_range_endpoints_exactly(tmp_path):
    from transformation_portal.utils.image_utils import apply_cube_lut, load_cube_lut

    cube = tmp_path / "identity.cube"
    cube.write_text(_cube_text(4, lambda r, g, b: (r, g, b), header="DOMAIN_MIN 0 0 0"), encoding="utf-8")
    image = np.array([[[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.5, -0.5, 0.5]]], dtype=np.float32)

    result = apply_cube_lut(image, load_cube_lut(cube))

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, np.clip(image, 0.0, 1.0), atol=1e-6)


def test_apply_cube_lut_rejects_non_rgb_images():
    from transformation_portal.utils.image_utils import apply_cube_lut

    lut = np.zeros((2, 2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="3 colour channels"):
        apply_cube_lut(np.zeros((4, 4, 4), dtype=np.float32), lut)


def _icc_profile(colour_space: bytes) -> bytes:
    """Minimal ICC header bytes whose data colour space signature is ``colour_space``."""
    header = bytearray(128)
    header[16:20] = colour_space
    return bytes(header)


def test_extract_output_metadata_keeps_rgb_icc_dpi_and_exif():
    """M-07: the source ICC profile, DPI and EXIF are collected for re-attachment on save."""
    from transformation_portal.utils.image_utils import extract_output_metadata

    img = Image.new("RGB", (4, 4))
    exif = img.getexif()
    exif[0x0110] = "Camera"
    img.info.update({"icc_profile": _icc_profile(b"RGB "), "dpi": (300, 240), "exif": exif.tobytes()})

    metadata = extract_output_metadata(img)

    assert metadata["icc_profile"] == _icc_profile(b"RGB ")
    assert metadata["dpi"] == (300, 240)
    assert metadata["exif"][0x0110] == "Camera"


def test_extract_output_metadata_drops_non_rgb_icc_and_malformed_values():
    """M-07: a CMYK profile would mis-describe RGB output; junk DPI/EXIF are dropped."""
    from transformation_portal.utils.image_utils import extract_output_metadata

    img = Image.new("RGB", (4, 4))
    img.info.update({"icc_profile": _icc_profile(b"CMYK"), "dpi": (0, 0), "exif": b"not exif"})

    assert extract_output_metadata(img) == {}
    assert extract_output_metadata(Image.new("RGB", (4, 4))) == {}


def test_output_stem_allocator_keeps_unique_stems_and_moves_only_colliders(tmp_path):
    """M-02: same-stem batch inputs get distinct output stems; other names are unchanged."""
    from transformation_portal.utils.image_utils import OutputStemAllocator

    allocator = OutputStemAllocator()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()

    assert allocator.claim(tmp_path / "lounge.png") == "lounge"
    assert allocator.claim(tmp_path / "kitchen.png") == "kitchen"
    assert allocator.claim(tmp_path / "kitchen.tif") == "kitchen_tif"
    # Case-only differences collide on case-insensitive filesystems (macOS default).
    assert allocator.claim(tmp_path / "Kitchen.jpg") == "Kitchen_jpg"
    # Same name in several directories: after the extension, a path digest separates them.
    assert allocator.claim(tmp_path / "bath.png") == "bath"
    assert allocator.claim(tmp_path / "a" / "bath.png") == "bath_png"
    third = allocator.claim(tmp_path / "b" / "bath.png")
    assert third.startswith("bath_png_") and len(third) == len("bath_png_") + 8
    # Claiming the same input again is stable, and a real input named like a
    # disambiguated stem does not take over that stem.
    assert allocator.claim(tmp_path / "kitchen.tif") == "kitchen_tif"
    assert allocator.claim(tmp_path / "a" / "kitchen_tif.png") != "kitchen_tif"


def test_save_image_failure_keeps_existing_file(tmp_path):
    """M-02: save_image writes through a temp file, so a failed save leaves the old file."""
    target = tmp_path / "out.png"
    target.write_bytes(b"previous")

    class _Broken:
        def save(self, path, **kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        save_image(_Broken(), target)

    assert target.read_bytes() == b"previous"
    assert [p.name for p in tmp_path.iterdir()] == ["out.png"]
