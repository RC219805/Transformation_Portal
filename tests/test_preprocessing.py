"""Tests for preprocessing module.

Tests image validation, format conversion, dimension enforcement,
and normalization for depth inference.
"""

import hashlib
import os
import stat
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Pytest markers
pytestmark = [
    pytest.mark.unit,
]

from transformation_portal.lux_depth_v3.preprocessing import (
    DIMENSION_MULTIPLE,
    SUPPORTED_EXTENSIONS,
    preprocess_image,
    preprocess_image_snapshot,
    validate_image_format,
)


class TestValidateImageFormat:
    """Test image format validation."""

    def test_valid_image_passes(self, tmp_path):
        """Test that valid image passes validation."""
        # Create valid PNG
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (64, 64), color="red")
        img.save(img_path)

        result = validate_image_format(img_path)

        assert result == img_path

    def test_nonexistent_file_raises_filenotfounderror(self, tmp_path):
        """Test that missing file raises FileNotFoundError."""
        fake_path = tmp_path / "nonexistent.jpg"

        with pytest.raises(FileNotFoundError, match="not found"):
            validate_image_format(fake_path)

    def test_unsupported_extension_raises_valueerror(self, tmp_path):
        """Test that unsupported format raises ValueError."""
        # Use .gif which is actually unsupported (we now support .bmp and .webp)
        bad_path = tmp_path / "test.gif"
        bad_path.write_text("fake gif")

        with pytest.raises(ValueError, match="Unsupported image format"):
            validate_image_format(bad_path)

    def test_corrupt_image_raises_valueerror(self, tmp_path):
        """Test that corrupt image raises ValueError."""
        corrupt_path = tmp_path / "corrupt.jpg"
        corrupt_path.write_bytes(b"not a valid image")

        with pytest.raises(ValueError, match="corrupt or invalid"):
            validate_image_format(corrupt_path)

    @pytest.mark.parametrize("ext", [".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp", ".bmp"])
    def test_all_supported_extensions(self, tmp_path, ext):
        """Test that all supported extensions are accepted."""
        img_path = tmp_path / f"test{ext}"
        img = Image.new("RGB", (32, 32))
        # Map extensions to PIL format names
        fmt_map = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".tiff": "TIFF",
            ".tif": "TIFF",
            ".webp": "WEBP",
            ".bmp": "BMP",
        }
        img.save(img_path, fmt_map[ext])

        result = validate_image_format(img_path)

        assert result == img_path


class TestPreprocessImage:
    """Test image preprocessing and normalization."""

    def test_uint8_rgb_to_float32(self, tmp_path):
        """Test conversion from uint8 RGB to float32 [0, 1]."""
        # Create uint8 RGB image
        img_path = tmp_path / "rgb.png"
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))
        img.save(img_path)

        result, original_shape = preprocess_image(img_path)

        # Check dtype and range
        assert result.dtype == np.float32
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.min() >= 0.0
        assert result.max() <= 1.0

        # Check approximate value (128/255 ≈ 0.5)
        assert np.abs(result.mean() - 0.5) < 0.01

    def test_snapshot_digest_covers_exact_bytes_used_for_decode(self, tmp_path):
        image_path = tmp_path / "snapshot.png"
        Image.new("RGB", (64, 42), color=(10, 20, 30)).save(image_path)
        source_bytes = image_path.read_bytes()

        result, original_shape, digest = preprocess_image_snapshot(image_path, verify_snapshot=True)
        expected, expected_shape = preprocess_image(image_path)

        assert digest == hashlib.sha256(source_bytes).hexdigest()
        assert original_shape == expected_shape == (42, 64)
        assert np.array_equal(result, expected)

    def test_snapshot_path_replacement_after_open_fails_closed(self, tmp_path, monkeypatch):
        image_path = tmp_path / "snapshot.png"
        replacement_path = tmp_path / "replacement.png"
        Image.new("RGB", (32, 24), color=(10, 20, 30)).save(image_path)
        Image.new("RGB", (32, 24), color=(200, 210, 220)).save(replacement_path)
        source_bytes = image_path.read_bytes()
        replacement_bytes = replacement_path.read_bytes()
        original_fdopen = os.fdopen
        swapped = False
        observed_source_bytes = bytearray()

        class _SwapAfterFirstRead:
            def __init__(self, handle):
                self._handle = handle

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return self._handle.__exit__(exc_type, exc, traceback)

            def fileno(self):
                return self._handle.fileno()

            def read(self, size=-1):
                nonlocal swapped
                data = self._handle.read(size)
                observed_source_bytes.extend(data)
                if not swapped:
                    replacement_path.replace(image_path)
                    swapped = True
                return data

        def replacing_fdopen(descriptor, *args, **kwargs):
            handle = original_fdopen(descriptor, *args, **kwargs)
            if not swapped:
                return _SwapAfterFirstRead(handle)
            return handle

        monkeypatch.setattr(os, "fdopen", replacing_fdopen)
        with pytest.raises(ValueError, match="changed while"):
            preprocess_image_snapshot(image_path)

        assert image_path.read_bytes() == replacement_bytes
        assert bytes(observed_source_bytes) == source_bytes

    def test_snapshot_rejects_unsupported_extension_before_decode(self, tmp_path):
        image_path = tmp_path / "snapshot.gif"
        image_path.write_bytes(b"not-used")

        with pytest.raises(ValueError, match="Unsupported image format"):
            preprocess_image_snapshot(image_path)

    def test_snapshot_normalizes_corrupt_standard_image_failure(self, tmp_path):
        image_path = tmp_path / "snapshot.png"
        image_path.write_bytes(b"not-a-png")

        with pytest.raises(ValueError, match="corrupt or invalid"):
            preprocess_image_snapshot(image_path, verify_snapshot=True)

    def test_snapshot_rejects_final_path_symlink(self, tmp_path):
        target_path = tmp_path / "target.png"
        symlink_path = tmp_path / "snapshot.png"
        Image.new("RGB", (24, 18), color=(10, 20, 30)).save(target_path)
        try:
            symlink_path.symlink_to(target_path)
        except OSError as exc:  # pragma: no cover - platform capability
            pytest.skip(f"symlinks unavailable: {exc}")

        with pytest.raises(ValueError, match="regular non-symlink"):
            preprocess_image_snapshot(symlink_path)

    def test_snapshot_rejects_symlink_swap_between_lstat_and_open(self, tmp_path, monkeypatch):
        image_path = tmp_path / "snapshot.png"
        outside_path = tmp_path / "outside.png"
        original_path = tmp_path / "original.png"
        Image.new("RGB", (24, 18), color=(10, 20, 30)).save(image_path)
        Image.new("RGB", (24, 18), color=(200, 210, 220)).save(outside_path)
        original_open = os.open
        swapped = False

        def swapping_open(path, flags, *args, **kwargs):
            nonlocal swapped
            if Path(path) == image_path and not swapped:
                swapped = True
                image_path.replace(original_path)
                image_path.symlink_to(outside_path)
                try:
                    return original_open(path, flags, *args, **kwargs)
                finally:
                    image_path.unlink()
                    original_path.replace(image_path)
            return original_open(path, flags, *args, **kwargs)

        monkeypatch.setattr(os, "open", swapping_open)

        with pytest.raises(ValueError, match="regular non-symlink|changed before"):
            preprocess_image_snapshot(image_path)

    def test_snapshot_exposes_opened_regular_file_stat_to_validator(self, tmp_path):
        image_path = tmp_path / "snapshot.png"
        Image.new("RGB", (28, 20), color=(10, 20, 30)).save(image_path)
        expected_stat = image_path.stat()
        observed = []

        def validate_opened_file(opened_stat):
            assert stat.S_ISREG(opened_stat.st_mode)
            observed.append(
                (
                    opened_stat.st_dev,
                    opened_stat.st_ino,
                    opened_stat.st_size,
                )
            )

        preprocess_image_snapshot(
            image_path,
            opened_file_stat_validator=validate_opened_file,
        )

        assert observed == [
            (
                expected_stat.st_dev,
                expected_stat.st_ino,
                expected_stat.st_size,
            )
        ]

    def test_snapshot_opened_file_validator_can_fail_closed(self, tmp_path):
        image_path = tmp_path / "snapshot.png"
        Image.new("RGB", (28, 20), color=(10, 20, 30)).save(image_path)

        def reject_opened_file(_opened_stat):
            raise RuntimeError("opened input is not plan-bound")

        with pytest.raises(RuntimeError, match="not plan-bound"):
            preprocess_image_snapshot(
                image_path,
                opened_file_stat_validator=reject_opened_file,
            )

    def test_raw_snapshot_is_explicitly_non_cache_authorizing(self, tmp_path, monkeypatch):
        raw_path = tmp_path / "image.dng"
        raw_path.write_bytes(b"raw-fixture")
        expected = np.zeros((14, 14, 3), dtype=np.float32)
        monkeypatch.setattr(
            "transformation_portal.lux_depth_v3.preprocessing.preprocess_image",
            lambda *_args, **_kwargs: (expected, (14, 14)),
        )

        result, shape, digest = preprocess_image_snapshot(
            raw_path,
            opened_file_stat_validator=lambda _opened_stat: pytest.fail(
                "RAW inputs must not enter cache-authorizing opened-file validation"
            ),
        )

        assert result is expected
        assert shape == (14, 14)
        assert digest is None

    def test_grayscale_converted_to_rgb(self, tmp_path):
        """Test that grayscale images are converted to 3-channel RGB."""
        img_path = tmp_path / "gray.png"
        img = Image.new("L", (50, 50), color=100)
        img.save(img_path)

        result, original_shape = preprocess_image(img_path)

        # Should be 3-channel
        assert result.shape[2] == 3

        # All channels should be identical (grayscale)
        assert np.allclose(result[:, :, 0], result[:, :, 1])
        assert np.allclose(result[:, :, 1], result[:, :, 2])

    def test_rgba_converted_to_rgb(self, tmp_path):
        """Test that RGBA images drop alpha channel."""
        img_path = tmp_path / "rgba.png"
        img = Image.new("RGBA", (50, 50), color=(255, 0, 0, 128))
        img.save(img_path)

        result, original_shape = preprocess_image(img_path)

        # Should be 3-channel (alpha dropped)
        assert result.shape[2] == 3

    def test_dimensions_enforced_to_multiple_of_14(self):
        """Test that dimensions are enforced to multiples of 14."""
        # Create image with non-compliant dimensions
        test_cases = [
            ((100, 100), (98, 98)),  # 100 → 98 (7×14)
            ((50, 70), (42, 70)),  # 50 → 42 (3×14), 70 → 70 (5×14)
            ((15, 15), (14, 14)),  # 15 → 14 (1×14)
            ((7, 7), (14, 14)),  # 7 → 14 (minimum)
        ]

        for input_size, expected_size in test_cases:
            img_array = np.random.rand(*input_size, 3).astype(np.float32)

            result, original_shape = preprocess_image(img_array)

            # Check dimensions are multiples of 14
            h, w = result.shape[:2]
            assert h % DIMENSION_MULTIPLE == 0, f"Height {h} not multiple of {DIMENSION_MULTIPLE}"
            assert w % DIMENSION_MULTIPLE == 0, f"Width {w} not multiple of {DIMENSION_MULTIPLE}"

            # Check expected size
            assert result.shape[:2] == expected_size

    def test_original_shape_preserved(self, tmp_path):
        """Test that original shape is returned correctly."""
        # Create image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (120, 80))  # W=120, H=80
        img.save(img_path)

        result, original_shape = preprocess_image(img_path)

        # Original shape should be (H, W)
        assert original_shape == (80, 120)

    def test_target_size_resizes_long_edge(self, tmp_path):
        """Test that target_size resizes long edge while maintaining aspect."""
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (200, 100))  # W=200 (long), H=100
        img.save(img_path)

        result, original_shape = preprocess_image(img_path, target_size=112)

        # Long edge should be close to 112 (after 14-alignment)
        # 112 is already multiple of 14 (8×14)
        h, w = result.shape[:2]
        assert max(h, w) <= 112

        # Aspect ratio should be approximately preserved
        aspect_original = 200 / 100
        aspect_result = w / h
        assert abs(aspect_original - aspect_result) < 0.2

    def test_numpy_array_input_uint8(self):
        """Test preprocessing from numpy uint8 array."""
        img_array = np.random.randint(0, 256, (56, 56, 3), dtype=np.uint8)

        result, original_shape = preprocess_image(img_array)

        assert result.dtype == np.float32
        assert result.shape[:2] == (56, 56)
        assert original_shape == (56, 56)

    def test_numpy_array_input_float32(self):
        """Test preprocessing from numpy float32 array."""
        img_array = np.random.rand(70, 70, 3).astype(np.float32)

        result, original_shape = preprocess_image(img_array)

        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_numpy_grayscale_to_rgb(self):
        """Test preprocessing from numpy grayscale (H, W)."""
        img_array = np.random.rand(56, 56).astype(np.float32)

        result, original_shape = preprocess_image(img_array)

        # Should be 3-channel
        assert result.shape[2] == 3

        # All channels should be identical
        assert np.allclose(result[:, :, 0], result[:, :, 1])

    def test_invalid_array_shape_raises_error(self):
        """Test that invalid array shapes raise ValueError."""
        # 4D array (invalid)
        bad_array = np.random.rand(10, 10, 3, 1).astype(np.float32)

        with pytest.raises(ValueError, match="Unsupported array shape"):
            preprocess_image(bad_array)

    def test_invalid_type_raises_error(self):
        """Test that invalid input types raise TypeError."""
        with pytest.raises(TypeError, match="must be np.ndarray, Path, or str"):
            preprocess_image(123)

    def test_minimum_dimension_enforced(self):
        """Test that minimum dimension is enforced (14)."""
        # Very small image
        img_array = np.random.rand(5, 5, 3).astype(np.float32)

        result, original_shape = preprocess_image(img_array)

        # Should be at least 14×14
        assert result.shape[0] >= DIMENSION_MULTIPLE
        assert result.shape[1] >= DIMENSION_MULTIPLE


class TestDimensionEnforcement:
    """Test dimension enforcement edge cases."""

    @pytest.mark.parametrize(
        "input_dim,expected_dim",
        [
            (14, 14),  # Already compliant
            (28, 28),  # Already compliant
            (42, 42),  # Already compliant
            (15, 14),  # Round down
            (27, 14),  # Round down
            (29, 28),  # Round down
            (100, 98),  # Round down
            (1, 14),  # Clamp to minimum
            (7, 14),  # Clamp to minimum
        ],
    )
    def test_dimension_rounding(self, input_dim, expected_dim):
        """Test dimension rounding behavior."""
        img_array = np.random.rand(input_dim, input_dim, 3).astype(np.float32)

        result, _ = preprocess_image(img_array)

        h, w = result.shape[:2]
        assert h == expected_dim
        assert w == expected_dim
        assert h % DIMENSION_MULTIPLE == 0
        assert w % DIMENSION_MULTIPLE == 0


class TestEndToEnd:
    """End-to-end preprocessing tests."""

    def test_full_pipeline_from_file(self, tmp_path):
        """Test complete preprocessing pipeline from file."""
        # Create test image
        img_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (100, 75), color=(200, 100, 50))
        img.save(img_path)

        # Preprocess
        result, original_shape = preprocess_image(img_path, target_size=None)

        # Validate all requirements
        assert result.dtype == np.float32
        assert result.ndim == 3
        assert result.shape[2] == 3
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.shape[0] % DIMENSION_MULTIPLE == 0
        assert result.shape[1] % DIMENSION_MULTIPLE == 0
        assert original_shape == (75, 100)  # H, W


class TestNormalizeExifOrientation:
    """Tests for normalize_exif_orientation function."""

    def test_normalize_unrotated_image(self, tmp_path):
        """Test normalization of image without EXIF orientation."""
        from transformation_portal.lux_depth_v3.preprocessing import normalize_exif_orientation

        # Create simple image without EXIF
        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.jpg"

        img = Image.new("RGB", (100, 50), color="red")
        img.save(input_path)

        # Should complete without error
        normalize_exif_orientation(input_path, output_path)

        # Output should exist and have same dimensions
        assert output_path.exists()
        with Image.open(output_path) as out_img:
            assert out_img.size == (100, 50)

    def test_normalize_creates_output_directory(self, tmp_path):
        """Test that output directory is created if needed."""
        from transformation_portal.lux_depth_v3.preprocessing import normalize_exif_orientation

        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "subdir" / "nested" / "output.jpg"

        img = Image.new("RGB", (64, 64), color="blue")
        img.save(input_path)

        normalize_exif_orientation(input_path, output_path)

        assert output_path.exists()

    def test_normalize_input_not_found_raises(self, tmp_path):
        """Test FileNotFoundError for missing input."""
        from transformation_portal.lux_depth_v3.preprocessing import normalize_exif_orientation

        fake_input = tmp_path / "nonexistent.jpg"
        output_path = tmp_path / "output.jpg"

        with pytest.raises(FileNotFoundError, match="Input image not found"):
            normalize_exif_orientation(fake_input, output_path)

    def test_normalize_in_place(self, tmp_path):
        """Test in-place normalization (same input/output path)."""
        from transformation_portal.lux_depth_v3.preprocessing import normalize_exif_orientation

        image_path = tmp_path / "test.jpg"
        img = Image.new("RGB", (80, 60), color="green")
        img.save(image_path)

        # Should work with same input and output
        normalize_exif_orientation(image_path, image_path)

        assert image_path.exists()
        with Image.open(image_path) as out_img:
            assert out_img.size == (80, 60)

    def test_normalize_invalid_image_raises_valueerror(self, tmp_path):
        """Test ValueError for corrupt/invalid image file."""
        from transformation_portal.lux_depth_v3.preprocessing import normalize_exif_orientation

        corrupt_path = tmp_path / "corrupt.jpg"
        output_path = tmp_path / "output.jpg"

        # Write invalid data
        corrupt_path.write_text("not a valid image")

        with pytest.raises(ValueError, match="Invalid image file"):
            normalize_exif_orientation(corrupt_path, output_path)


class TestValidateDepthImageAlignment:
    """Tests for validate_depth_image_alignment function."""

    def test_exact_match_returns_true(self, tmp_path):
        """Test exact dimension match."""
        from transformation_portal.lux_depth_v3.preprocessing import validate_depth_image_alignment

        # Create matching images
        img_path = tmp_path / "image.png"
        depth_path = tmp_path / "depth.png"

        Image.new("RGB", (100, 80)).save(img_path)
        Image.new("L", (100, 80)).save(depth_path)

        assert validate_depth_image_alignment(img_path, depth_path) is True

    def test_mismatch_returns_false(self, tmp_path):
        """Test dimension mismatch."""
        from transformation_portal.lux_depth_v3.preprocessing import validate_depth_image_alignment

        img_path = tmp_path / "image.png"
        depth_path = tmp_path / "depth.png"

        Image.new("RGB", (100, 80)).save(img_path)
        Image.new("L", (200, 160)).save(depth_path)

        assert validate_depth_image_alignment(img_path, depth_path) is False

    def test_padded_depth_returns_true(self, tmp_path):
        """Test that padded depth maps (multiples of 14) match."""
        from transformation_portal.lux_depth_v3.preprocessing import (
            DIMENSION_MULTIPLE,
            validate_depth_image_alignment,
        )

        img_path = tmp_path / "image.png"
        depth_path = tmp_path / "depth.png"

        # Image at 100x80, depth padded to 112x84 (next multiple of 14)
        Image.new("RGB", (100, 80)).save(img_path)

        def next_multiple(val):
            return ((val + DIMENSION_MULTIPLE - 1) // DIMENSION_MULTIPLE) * DIMENSION_MULTIPLE

        padded_w = next_multiple(100)  # 112
        padded_h = next_multiple(80)  # 84
        Image.new("L", (padded_w, padded_h)).save(depth_path)

        assert validate_depth_image_alignment(img_path, depth_path) is True

    def test_npy_depth_format(self, tmp_path):
        """Test .npy depth map format."""
        from transformation_portal.lux_depth_v3.preprocessing import validate_depth_image_alignment

        img_path = tmp_path / "image.png"
        depth_path = tmp_path / "depth.npy"

        Image.new("RGB", (64, 48)).save(img_path)
        np.save(depth_path, np.zeros((48, 64), dtype=np.float32))

        assert validate_depth_image_alignment(img_path, depth_path) is True

    def test_missing_image_raises(self, tmp_path):
        """Test FileNotFoundError for missing image."""
        from transformation_portal.lux_depth_v3.preprocessing import validate_depth_image_alignment

        fake_img = tmp_path / "nonexistent.png"
        depth_path = tmp_path / "depth.png"
        Image.new("L", (64, 64)).save(depth_path)

        with pytest.raises(FileNotFoundError, match="Image not found"):
            validate_depth_image_alignment(fake_img, depth_path)

    def test_missing_depth_raises(self, tmp_path):
        """Test FileNotFoundError for missing depth."""
        from transformation_portal.lux_depth_v3.preprocessing import validate_depth_image_alignment

        img_path = tmp_path / "image.png"
        Image.new("RGB", (64, 64)).save(img_path)
        fake_depth = tmp_path / "nonexistent.npy"

        with pytest.raises(FileNotFoundError, match="Depth map not found"):
            validate_depth_image_alignment(img_path, fake_depth)

    def test_corrupt_image_raises(self, tmp_path):
        """Test ValueError for corrupt image."""
        from transformation_portal.lux_depth_v3.preprocessing import validate_depth_image_alignment

        img_path = tmp_path / "corrupt.png"
        depth_path = tmp_path / "depth.png"

        img_path.write_text("not an image")
        Image.new("L", (64, 64)).save(depth_path)

        with pytest.raises(ValueError, match="Cannot read image"):
            validate_depth_image_alignment(img_path, depth_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
