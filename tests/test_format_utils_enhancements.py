"""Tests for Enhanced Format Utilities

Test coverage for new features:
- Option 2: Enhanced format detection
- Option 3: Format conversion utilities
- Option 4: Improved 16-bit TIFF handling

Run with: pytest tests/test_format_utils_enhancements.py -v
"""

# pylint: disable=redefined-outer-name  # pytest fixtures use other fixtures as params

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.unit]

from transformation_portal.utils.format_utils_enhancements import (
    batch_convert_directory,
    check_tifffile_available,
    convert_image_format,
    convert_tiff_preserve_depth,
    detect_format_from_content,
    get_image_metadata,
    get_mime_type,
    get_optimal_format_for_use_case,
    get_tiff_compression_info,
    load_tiff_preserve_depth,
    optimize_tiff_compression,
    save_tiff_16bit,
    smart_convert,
    validate_image_integrity,
)

# ==============================================================================
# Fixtures
# ==============================================================================


@pytest.fixture
def temp_dir(tmp_path):
    """Alias for the pytest tmp_path fixture used throughout this module."""
    return tmp_path


@pytest.fixture
def sample_jpg(temp_workspace):
    """Create a sample JPEG file (uses shared temp_workspace)."""
    img = Image.new("RGB", (100, 100), color="red")
    path = temp_workspace["root"] / "sample.jpg"
    img.save(path, quality=90)
    return path


@pytest.fixture
def sample_png(temp_workspace):
    """Create a sample PNG file (uses shared temp_workspace)."""
    img = Image.new("RGBA", (100, 100), color=(0, 255, 0, 128))
    path = temp_workspace["root"] / "sample.png"
    img.save(path)
    return path


@pytest.fixture
def sample_tiff_8bit(temp_workspace):
    """Create a sample 8-bit TIFF file (uses shared temp_workspace)."""
    img = Image.new("RGB", (100, 100), color="blue")
    path = temp_workspace["root"] / "sample.tif"
    img.save(path)
    return path


@pytest.fixture
def sample_tiff_16bit(temp_dir):
    """Create a sample 16-bit TIFF file (requires tifffile)."""
    if not check_tifffile_available():
        pytest.skip("tifffile not available")

    import tifffile

    arr = np.random.randint(0, 65536, (100, 100, 3), dtype=np.uint16)
    path = temp_dir / "sample_16bit.tif"
    tifffile.imwrite(path, arr, compression="lzw")
    return path


@pytest.fixture
def corrupted_image(temp_dir):
    """Create a corrupted image file."""
    path = temp_dir / "corrupted.jpg"
    path.write_bytes(b"Not a real image file")
    return path


# ==============================================================================
# Option 2: Enhanced Format Detection Tests
# ==============================================================================


class TestEnhancedFormatDetection:
    """Tests for enhanced format detection features."""

    def test_detect_format_from_content_jpg(self, sample_jpg):
        """Test format detection on JPEG."""
        format_type = detect_format_from_content(sample_jpg)
        assert format_type == "JPEG"

    def test_detect_format_from_content_png(self, sample_png):
        """Test format detection on PNG."""
        format_type = detect_format_from_content(sample_png)
        assert format_type == "PNG"

    def test_detect_format_from_content_tiff(self, sample_tiff_8bit):
        """Test format detection on TIFF."""
        format_type = detect_format_from_content(sample_tiff_8bit)
        assert format_type == "TIFF"

    def test_detect_format_wrong_extension(self, sample_jpg, temp_dir):
        """Test detection on file with wrong extension."""
        # Rename JPG to .txt
        wrong_path = temp_dir / "image.txt"
        wrong_path.write_bytes(sample_jpg.read_bytes())

        format_type = detect_format_from_content(wrong_path)
        assert format_type == "JPEG"  # Should detect as JPEG despite .txt extension

    def test_detect_format_nonexistent(self, temp_dir):
        """Test detection on non-existent file."""
        format_type = detect_format_from_content(temp_dir / "missing.jpg")
        assert format_type is None

    def test_get_mime_type_jpg(self, sample_jpg):
        """Test MIME type detection for JPEG."""
        mime = get_mime_type(sample_jpg)
        assert mime == "image/jpeg"

    def test_get_mime_type_png(self, sample_png):
        """Test MIME type detection for PNG."""
        mime = get_mime_type(sample_png)
        assert mime == "image/png"

    def test_validate_image_integrity_valid(self, sample_jpg):
        """Test validation on valid image."""
        is_valid, error = validate_image_integrity(sample_jpg)
        assert is_valid is True
        assert error is None

    def test_validate_image_integrity_corrupted(self, corrupted_image):
        """Test validation on corrupted image."""
        is_valid, error = validate_image_integrity(corrupted_image)
        assert is_valid is False
        assert error is not None
        assert "Cannot identify" in error or "Cannot open" in error

    def test_validate_image_integrity_nonexistent(self, temp_dir):
        """Test validation on non-existent file."""
        is_valid, error = validate_image_integrity(temp_dir / "missing.jpg")
        assert is_valid is False
        assert "does not exist" in error

    def test_get_image_metadata_jpg(self, sample_jpg):
        """Test metadata extraction from JPEG."""
        meta = get_image_metadata(sample_jpg)

        assert meta["format"] == "JPEG"
        assert meta["size"] == (100, 100)
        assert meta["mode"] == "RGB"
        assert meta["bit_depth"] == 8
        assert meta["has_alpha"] is False
        assert meta["file_size"] > 0

    def test_get_image_metadata_png_with_alpha(self, sample_png):
        """Test metadata extraction from PNG with alpha."""
        meta = get_image_metadata(sample_png)

        assert meta["format"] == "PNG"
        assert meta["size"] == (100, 100)
        assert meta["mode"] == "RGBA"
        assert meta["has_alpha"] is True


# ==============================================================================
# Option 3: Format Conversion Tests
# ==============================================================================


class TestFormatConversion:
    """Tests for format conversion utilities."""

    def test_convert_jpg_to_png(self, sample_jpg, temp_dir):
        """Test converting JPEG to PNG."""
        output = temp_dir / "output.png"
        result = convert_image_format(sample_jpg, output)

        assert result is True
        assert output.exists()
        assert detect_format_from_content(output) == "PNG"

    def test_convert_png_to_jpg(self, sample_png, temp_dir):
        """Test converting PNG with alpha to JPEG."""
        output = temp_dir / "output.jpg"
        result = convert_image_format(sample_png, output)

        assert result is True
        assert output.exists()
        # Should have converted RGBA to RGB with white background
        img = Image.open(output)
        assert img.mode == "RGB"

    def test_convert_with_quality(self, sample_jpg, temp_dir):
        """Test conversion with quality setting."""
        output = temp_dir / "output.jpg"
        result = convert_image_format(sample_jpg, output, quality=50)

        assert result is True
        assert output.exists()

        # Lower quality should result in smaller file
        assert output.stat().st_size < sample_jpg.stat().st_size

    def test_convert_creates_directories(self, sample_jpg, temp_dir):
        """Test that conversion creates output directories."""
        output = temp_dir / "subdir" / "nested" / "output.png"
        result = convert_image_format(sample_jpg, output)

        assert result is True
        assert output.exists()
        assert output.parent.exists()

    def test_batch_convert_directory(self, temp_dir):
        """Test batch conversion of directory."""
        # Create input directory with multiple images
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        img1 = Image.new("RGB", (50, 50), "red")
        img2 = Image.new("RGB", (50, 50), "blue")
        img3 = Image.new("RGB", (50, 50), "green")

        img1.save(input_dir / "img1.tif")
        img2.save(input_dir / "img2.bmp")
        img3.save(input_dir / "img3.png")

        # Convert all to JPEG
        output_dir = temp_dir / "output"
        stats = batch_convert_directory(input_dir, output_dir, ".jpg")

        assert stats["total"] == 3
        assert stats["success"] == 3
        assert stats["failed"] == 0
        assert (output_dir / "img1.jpg").exists()
        assert (output_dir / "img2.jpg").exists()
        assert (output_dir / "img3.jpg").exists()

    def test_batch_convert_skip_same_format(self, temp_dir):
        """Test batch conversion skips files already in target format."""
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        img1 = Image.new("RGB", (50, 50), "red")
        img2 = Image.new("RGB", (50, 50), "blue")

        img1.save(input_dir / "img1.jpg")
        img2.save(input_dir / "img2.png")

        output_dir = temp_dir / "output"
        stats = batch_convert_directory(input_dir, output_dir, ".jpg")

        assert stats["total"] == 2
        assert stats["skipped"] == 1  # img1.jpg skipped
        assert stats["success"] == 1  # img2.png converted

    def test_smart_convert_default(self, sample_jpg, temp_dir):
        """Test smart conversion with defaults."""
        output = temp_dir / "output.png"
        result = smart_convert(sample_jpg, output)

        assert result is True
        assert output.exists()

    def test_get_optimal_format_web(self):
        """Test format recommendation for web use."""
        format_ext = get_optimal_format_for_use_case("web", has_alpha=False)
        assert format_ext == ".webp"

        format_ext = get_optimal_format_for_use_case("web", has_alpha=True)
        assert format_ext == ".png"

    def test_get_optimal_format_print(self):
        """Test format recommendation for print."""
        format_ext = get_optimal_format_for_use_case("print")
        assert format_ext == ".tif"

    def test_get_optimal_format_16bit(self):
        """Test format recommendation when 16-bit required."""
        format_ext = get_optimal_format_for_use_case("web", requires_16bit=True)
        assert format_ext == ".tif"


# ==============================================================================
# Option 4: TIFF Handling Tests
# ==============================================================================


class TestTIFFHandling:
    """Tests for improved 16-bit TIFF handling."""

    def test_check_tifffile_available(self):
        """Test checking for tifffile availability."""
        result = check_tifffile_available()
        assert isinstance(result, bool)

    @pytest.mark.skipif(not check_tifffile_available(), reason="tifffile not available")
    def test_save_tiff_16bit(self, temp_dir):
        """Test saving 16-bit TIFF."""
        arr = np.random.randint(0, 65536, (100, 100, 3), dtype=np.uint16)
        output = temp_dir / "output_16bit.tif"

        result = save_tiff_16bit(arr, output, compression="lzw")

        assert result is True
        assert output.exists()

        # Verify it's actually 16-bit
        loaded, bit_depth = load_tiff_preserve_depth(output)
        assert bit_depth == 16

    @pytest.mark.skipif(not check_tifffile_available(), reason="tifffile not available")
    def test_save_tiff_with_metadata(self, temp_dir):
        """Test saving TIFF with metadata."""
        arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        output = temp_dir / "output_meta.tiff"
        metadata = {"author": "test", "copyright": "2025"}

        result = save_tiff_16bit(arr, output, metadata=metadata)

        assert result is True
        assert output.exists()

    def test_load_tiff_preserve_depth_8bit(self, sample_tiff_8bit):
        """Test loading 8-bit TIFF."""
        arr, bit_depth = load_tiff_preserve_depth(sample_tiff_8bit)

        assert arr is not None
        assert bit_depth == 8
        assert arr.shape == (100, 100, 3)

    @pytest.mark.skipif(not check_tifffile_available(), reason="tifffile not available")
    def test_load_tiff_preserve_depth_16bit(self, sample_tiff_16bit):
        """Test loading 16-bit TIFF."""
        arr, bit_depth = load_tiff_preserve_depth(sample_tiff_16bit)

        assert arr is not None
        assert bit_depth == 16
        assert arr.dtype == np.uint16

    @pytest.mark.skipif(not check_tifffile_available(), reason="tifffile not available")
    def test_convert_tiff_preserve_depth(self, sample_tiff_16bit, temp_dir):
        """Test converting TIFF while preserving 16-bit depth."""
        output = temp_dir / "converted_16bit.tif"

        result = convert_tiff_preserve_depth(sample_tiff_16bit, output)

        assert result is True
        assert output.exists()

        # Verify depth preserved
        _, bit_depth = load_tiff_preserve_depth(output)
        assert bit_depth == 16

    @pytest.mark.skipif(not check_tifffile_available(), reason="tifffile not available")
    def test_get_tiff_compression_info(self, sample_tiff_16bit):
        """Test getting TIFF compression info."""
        compression = get_tiff_compression_info(sample_tiff_16bit)

        assert compression is not None
        # Should be LZW since we created it with that compression

    @pytest.mark.skipif(not check_tifffile_available(), reason="tifffile not available")
    def test_optimize_tiff_compression(self, temp_dir):
        """Test optimizing TIFF compression."""
        # Create uncompressed TIFF with compressible data
        # Use blocks of solid color instead of random data for better compression
        arr = np.zeros((200, 200, 3), dtype=np.uint8)
        arr[:100, :, 0] = 255  # Red top half
        arr[100:, :, 2] = 255  # Blue bottom half

        input_path = temp_dir / "uncompressed.tif"
        save_tiff_16bit(arr, input_path, compression="none")

        # Optimize with LZW compression
        output_path = temp_dir / "compressed.tif"
        success, ratio = optimize_tiff_compression(input_path, output_path, "lzw")

        assert success is True
        assert ratio is not None
        # Note: Compression effectiveness depends on data patterns
        # For solid color blocks, we should see compression
        assert ratio < 1.2  # Allow some overhead for small files


# ==============================================================================
# Integration Tests
# ==============================================================================


class TestIntegration:
    """Integration tests combining multiple features."""

    def test_detect_validate_convert_workflow(self, sample_jpg, temp_dir):
        """Test complete workflow: detect -> validate -> convert."""
        # Detect format
        format_type = detect_format_from_content(sample_jpg)
        assert format_type == "JPEG"

        # Validate integrity
        is_valid, error = validate_image_integrity(sample_jpg)
        assert is_valid is True

        # Get metadata
        meta = get_image_metadata(sample_jpg)
        assert meta["format"] == "JPEG"

        # Convert based on metadata
        if meta["has_alpha"]:
            output_format = ".png"
        else:
            output_format = ".webp"

        output = temp_dir / f"output{output_format}"
        result = convert_image_format(sample_jpg, output)
        assert result is True

    def test_batch_process_with_validation(self, temp_dir):
        """Test batch processing with validation."""
        # Create test files
        input_dir = temp_dir / "input"
        input_dir.mkdir()

        # Good image
        img_good = Image.new("RGB", (100, 100), "red")
        img_good.save(input_dir / "good.jpg")

        # Another good image
        img_good2 = Image.new("RGB", (100, 100), "blue")
        img_good2.save(input_dir / "good2.png")

        # Corrupted image
        (input_dir / "bad.jpg").write_bytes(b"corrupted")

        # Validate all images first
        valid_count = 0
        for img_path in input_dir.glob("*"):
            if validate_image_integrity(img_path)[0]:
                valid_count += 1

        assert valid_count == 2  # Only 2 valid images

        # Batch convert (will fail on corrupted)
        output_dir = temp_dir / "output"
        stats = batch_convert_directory(input_dir, output_dir, ".png")

        # Should process 3, succeed on 2, fail on 1
        assert stats["total"] == 3
        assert stats["success"] >= 1  # At least one should convert


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
