"""Tests for EXIF orientation pre-normalization.

This module tests that EXIF orientation is properly handled to ensure
alignment between DA3 (PIL-based) and V2 (OpenCV-based) pipelines.
"""

import pytest
from PIL import Image, ImageOps
import cv2
import numpy as np
from pathlib import Path

from lux_depth_v3.enhance.preprocessing import (
    normalize_exif_orientation,
    get_exif_orientation,
    has_exif_orientation,
)


class TestEXIFOrientation:
    """Test EXIF orientation normalization."""

    def test_orientation_1_normal(self, tmp_path):
        """Orientation 1 (normal) should be pass-through."""
        # Create 200x100 image (landscape) with orientation 1
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        exif = img.getexif()
        exif[0x0112] = 1  # Normal orientation

        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.png"
        img.save(input_path, exif=exif)

        was_normalized = normalize_exif_orientation(input_path, output_path)

        # Should return True (tag existed)
        assert was_normalized == True

        # Dimensions should remain unchanged
        img_norm = Image.open(output_path)
        assert img_norm.size == (200, 100)

        # EXIF tag should be removed
        exif_norm = img_norm.getexif()
        assert 0x0112 not in exif_norm

    def test_orientation_3_rotate_180(self, tmp_path):
        """Orientation 3 (180° rotation) should rotate dimensions."""
        # Create 200x100 image with orientation 3
        img = Image.new("RGB", (200, 100), color=(255, 0, 0))
        exif = img.getexif()
        exif[0x0112] = 3  # Rotate 180°

        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.png"
        img.save(input_path, exif=exif)

        was_normalized = normalize_exif_orientation(input_path, output_path)

        assert was_normalized == True

        # Dimensions stay same for 180° rotation
        img_norm = Image.open(output_path)
        assert img_norm.size == (200, 100)

        # EXIF tag removed
        exif_norm = img_norm.getexif()
        assert 0x0112 not in exif_norm

    def test_orientation_6_rotate_90cw(self, tmp_path):
        """Orientation 6 (90° CW rotation) should swap dimensions."""
        # Create 100x200 image (portrait) with orientation 6
        # After rotation, should become 200x100 (landscape)
        img = Image.new("RGB", (100, 200), color=(255, 0, 0))
        exif = img.getexif()
        exif[0x0112] = 6  # Rotate 90° CW

        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.png"
        img.save(input_path, exif=exif)

        was_normalized = normalize_exif_orientation(input_path, output_path)

        assert was_normalized == True

        # Verify dimensions swapped
        img_norm = Image.open(output_path)
        assert img_norm.size == (200, 100)  # Width/height swapped

        # EXIF tag removed
        exif_norm = img_norm.getexif()
        assert 0x0112 not in exif_norm

    def test_orientation_8_rotate_90ccw(self, tmp_path):
        """Orientation 8 (90° CCW rotation) should swap dimensions."""
        # Create 100x200 image with orientation 8
        # After rotation, should become 200x100
        img = Image.new("RGB", (100, 200), color=(0, 255, 0))
        exif = img.getexif()
        exif[0x0112] = 8  # Rotate 90° CCW

        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.png"
        img.save(input_path, exif=exif)

        was_normalized = normalize_exif_orientation(input_path, output_path)

        assert was_normalized == True

        # Verify dimensions swapped
        img_norm = Image.open(output_path)
        assert img_norm.size == (200, 100)  # Width/height swapped

        # EXIF tag removed
        exif_norm = img_norm.getexif()
        assert 0x0112 not in exif_norm

    def test_no_exif_orientation(self, tmp_path):
        """Image without EXIF orientation should be pass-through."""
        # Create image WITHOUT orientation tag
        img = Image.new("RGB", (200, 100), color=(0, 0, 255))

        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.png"
        img.save(input_path)

        was_normalized = normalize_exif_orientation(input_path, output_path)

        # Should return False (no tag)
        assert was_normalized == False

        # Dimensions unchanged
        img_norm = Image.open(output_path)
        assert img_norm.size == (200, 100)

    def test_pil_opencv_consistency(self, tmp_path):
        """PIL and OpenCV should see same dimensions after normalization."""
        # Create image with orientation 6 (90° CW)
        # Original: 100x200 (portrait, but stored rotated)
        # After normalization: 200x100 (landscape)
        img = Image.new("RGB", (100, 200), color=(255, 0, 0))
        exif = img.getexif()
        exif[0x0112] = 6

        input_path = tmp_path / "input.jpg"
        normalized_path = tmp_path / "normalized.png"
        img.save(input_path, exif=exif)

        normalize_exif_orientation(input_path, normalized_path)

        # Read with PIL (DA3 simulation)
        img_pil = Image.open(normalized_path)
        img_pil = ImageOps.exif_transpose(img_pil)  # Should be no-op (tag removed)

        # Read with OpenCV (V2 simulation)
        img_cv = cv2.imread(str(normalized_path))

        # Compare dimensions
        assert img_pil.size[0] == img_cv.shape[1]  # Width
        assert img_pil.size[1] == img_cv.shape[0]  # Height

        # Both should see landscape orientation (200x100)
        assert img_pil.size == (200, 100)
        assert img_cv.shape[:2] == (100, 200)  # H, W in OpenCV


class TestEXIFHelpers:
    """Test EXIF helper functions."""

    def test_get_exif_orientation_with_tag(self, tmp_path):
        """get_exif_orientation() should return correct value."""
        img = Image.new("RGB", (100, 100), color="red")
        exif = img.getexif()
        exif[0x0112] = 6  # Rotate 90° CW

        path = tmp_path / "test.jpg"
        img.save(path, exif=exif)

        orientation = get_exif_orientation(path)
        assert orientation == 6

    def test_get_exif_orientation_without_tag(self, tmp_path):
        """get_exif_orientation() should return 1 (normal) if no tag."""
        img = Image.new("RGB", (100, 100), color="blue")

        path = tmp_path / "test.jpg"
        img.save(path)

        orientation = get_exif_orientation(path)
        assert orientation == 1  # Default: normal

    def test_has_exif_orientation_true(self, tmp_path):
        """has_exif_orientation() should detect tag."""
        img = Image.new("RGB", (100, 100))
        exif = img.getexif()
        exif[0x0112] = 3

        path = tmp_path / "test.jpg"
        img.save(path, exif=exif)

        assert has_exif_orientation(path) == True

    def test_has_exif_orientation_false(self, tmp_path):
        """has_exif_orientation() should return False if no tag."""
        img = Image.new("RGB", (100, 100))

        path = tmp_path / "test.jpg"
        img.save(path)

        assert has_exif_orientation(path) == False


class TestEXIFEdgeCases:
    """Test edge cases and error handling."""

    def test_fallback_on_error(self, tmp_path):
        """Normalization should fallback to copy on error."""
        # Create a valid image
        img = Image.new("RGB", (100, 100))
        input_path = tmp_path / "input.jpg"
        img.save(input_path)

        # Mock exif_transpose to fail
        from unittest.mock import patch

        output_path = tmp_path / "output.png"

        with patch("PIL.ImageOps.exif_transpose", side_effect=Exception("Bad EXIF")):
            # Should not raise, should fallback to copy
            was_normalized = normalize_exif_orientation(input_path, output_path)

            # Should return False (fallback mode)
            assert was_normalized == False

            # Output should exist (copied)
            assert output_path.exists()

    def test_parent_dir_created(self, tmp_path):
        """Parent directories should be created automatically."""
        img = Image.new("RGB", (100, 100))
        input_path = tmp_path / "input.jpg"
        img.save(input_path)

        output_path = tmp_path / "a" / "b" / "c" / "output.png"
        normalize_exif_orientation(input_path, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()

    def test_preserves_pixel_data(self, tmp_path):
        """Normalization should preserve pixel values (not re-compress)."""
        # Create image with specific pixel pattern
        img = Image.new("RGB", (10, 10))
        pixels = img.load()
        for x in range(10):
            for y in range(10):
                pixels[x, y] = (x * 25, y * 25, 128)

        input_path = tmp_path / "input.png"  # Use PNG to avoid JPEG compression
        output_path = tmp_path / "output.png"
        img.save(input_path)

        normalize_exif_orientation(input_path, output_path)

        # Load and verify
        img_loaded = Image.open(output_path)
        pixels_loaded = img_loaded.load()

        # Check exact pixels (PNG is lossless)
        assert pixels_loaded[0, 0] == (0, 0, 128)
        assert pixels_loaded[5, 5] == (125, 125, 128)
        assert pixels_loaded[9, 9] == (225, 225, 128)

    def test_handles_various_formats(self, tmp_path):
        """Should handle JPEG and PNG formats."""
        for ext, use_exif in [(".jpg", True), (".png", False)]:
            img = Image.new("RGB", (100, 100))

            input_path = tmp_path / f"input{ext}"
            output_path = tmp_path / f"output{ext}.png"

            if use_exif:
                exif = img.getexif()
                exif[0x0112] = 6
                img.save(input_path, exif=exif)
            else:
                # PNG doesn't support EXIF in the same way
                img.save(input_path)

            # Should not raise
            normalize_exif_orientation(input_path, output_path)
            assert output_path.exists()


# Fixtures
@pytest.fixture
def sample_image_with_exif(tmp_path):
    """Create sample image with EXIF orientation."""
    img = Image.new("RGB", (100, 200), color=(255, 0, 0))
    exif = img.getexif()
    exif[0x0112] = 6  # Rotate 90° CW

    path = tmp_path / "sample_exif.jpg"
    img.save(path, exif=exif)
    return path


@pytest.fixture
def sample_image_without_exif(tmp_path):
    """Create sample image without EXIF orientation."""
    img = Image.new("RGB", (200, 100), color=(0, 255, 0))

    path = tmp_path / "sample_no_exif.jpg"
    img.save(path)
    return path
