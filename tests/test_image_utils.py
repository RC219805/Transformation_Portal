"""Tests for common image utilities module."""
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from image_utils import (
    load_image,
    save_image,
    pil_to_np,
    np_to_pil,
    load_image_rgb,
)


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
    assert pixel[1] == 0    # -0.5 clipped to 0.0 -> 0
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
