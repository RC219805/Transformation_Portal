"""Tests for depth_writer module.

Validates atomic write, precision, and readback logic.
"""

from pathlib import Path

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.depth_writer import (
    HAS_CV2,
    MAX_DEPTH_PNG_DECODED_PIXELS,
    DepthWriteStats,
    atomic_write_depth_u16_png_with_stats,
    read_depth_u16_png,
    read_depth_u16_png_bytes,
)

# Skip all tests if OpenCV is not installed
pytestmark = [pytest.mark.unit, pytest.mark.skipif(not HAS_CV2, reason="OpenCV not installed")]


class TestDepthWriter:
    """Test atomic depth map writing."""

    def test_atomic_write_and_read(self, tmp_path):
        """Verify write -> read cycle preserves data within u16 precision."""
        # Create a random float depth map [0, 1]
        original = np.random.rand(100, 100).astype(np.float32)
        output_path = tmp_path / "depth.png"

        # Write
        path, verification_path, stats = atomic_write_depth_u16_png_with_stats(output_path, original, debug_verify=True)

        assert path.exists()
        assert stats.shape == (100, 100)
        assert stats.dtype == "float32"
        assert 0.0 <= stats.min <= 1.0
        assert 0.0 <= stats.max <= 1.0

        # Read back
        loaded = read_depth_u16_png(path)

        # Check error is minimal (quantization error for 16-bit is ~1.5e-5)
        # We verify mean absolute error is small
        mae = np.mean(np.abs(original - loaded))
        assert mae < 1e-4, f"MAE {mae} exceeds threshold"

    def test_atomic_overwrite(self, tmp_path):
        """Verify atomic overwrite works correctly."""
        output_path = tmp_path / "overwrite.png"

        # Write 1: all zeros
        data1 = np.zeros((10, 10), dtype=np.float32)
        path1, _, stats1 = atomic_write_depth_u16_png_with_stats(output_path, data1)
        assert path1.exists()

        # Write 2: all ones (different data)
        data2 = np.ones((10, 10), dtype=np.float32)
        path2, _, stats2 = atomic_write_depth_u16_png_with_stats(output_path, data2)

        # Should be the same file
        assert path2 == path1

        # Read back should be ones (second write)
        loaded = read_depth_u16_png(output_path)
        assert np.mean(loaded) > 0.99  # Should be close to 1.0

    def test_statistics_calculation(self, tmp_path):
        """Verify statistics are calculated correctly."""
        # Create known distribution
        depth_map = np.linspace(0.0, 1.0, 10000).reshape(100, 100).astype(np.float32)
        output_path = tmp_path / "stats_test.png"

        _, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        # Check statistics
        assert abs(stats.min - 0.0) < 1e-6
        assert abs(stats.max - 1.0) < 1e-6
        assert abs(stats.mean - 0.5) < 0.01  # Should be close to 0.5
        assert stats.shape == (100, 100)

    def test_clipping_out_of_range(self, tmp_path):
        """Verify values outside [0, 1] are clipped."""
        # Create data with values outside valid range
        depth_map = np.array([[-0.5, 0.5], [1.5, 0.75]], dtype=np.float32)
        output_path = tmp_path / "clipped.png"

        path, _, _ = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        # Read back - should be clipped to [0, 1]
        loaded = read_depth_u16_png(path)
        assert np.min(loaded) >= 0.0
        assert np.max(loaded) <= 1.0
        assert loaded[0, 0] == 0.0  # -0.5 clipped to 0
        assert loaded[1, 0] == 1.0  # 1.5 clipped to 1

    def test_metric_depth_is_percentile_normalized_for_png(self, tmp_path):
        """Metric meter values should not collapse into a saturated PNG."""
        depth_map = np.linspace(2.0, 40.0, 10000, dtype=np.float32).reshape(100, 100)
        output_path = tmp_path / "metric_depth.png"

        path, _, stats = atomic_write_depth_u16_png_with_stats(
            output_path,
            depth_map,
            compute_encoded_unique_values=True,
        )
        loaded = read_depth_u16_png(path)

        assert stats.normalization is not None
        assert stats.normalization["mode"] == "percentile_1_99"
        assert stats.encoded_min == 0
        assert stats.encoded_max == 65535
        assert stats.encoded_unique_values is not None
        assert stats.encoded_unique_values > 100
        assert float(np.mean(loaded > 0.999)) < 0.05

    def test_encoded_unique_values_are_opt_in(self, tmp_path):
        """Avoid exact image cardinality scans outside audit paths."""
        depth_map = np.linspace(0.0, 1.0, 10000, dtype=np.float32).reshape(100, 100)
        output_path = tmp_path / "default_stats.png"

        _, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        assert stats.encoded_min == 0
        assert stats.encoded_max == 65535
        assert stats.encoded_unique_values is None

    def test_directory_creation(self, tmp_path):
        """Verify parent directories are created if needed."""
        # Use nested path that doesn't exist yet
        output_path = tmp_path / "subdir1" / "subdir2" / "depth.png"
        depth_map = np.random.rand(50, 50).astype(np.float32)

        path, _, _ = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        assert path.exists()
        assert path.parent.exists()

    def test_read_nonexistent_file(self, tmp_path):
        """Verify reading nonexistent file raises FileNotFoundError."""
        nonexistent = tmp_path / "does_not_exist.png"

        with pytest.raises(FileNotFoundError, match="Depth file not found"):
            read_depth_u16_png(nonexistent)

    def test_precision_preservation(self, tmp_path):
        """Verify 16-bit precision is preserved."""
        # Create depth map with specific values that should be representable in u16
        # Use values like 0.5, 0.25, 0.75 which have exact binary representations
        depth_map = np.array([[0.0, 0.25, 0.5, 0.75, 1.0], [0.1, 0.2, 0.3, 0.4, 0.6]], dtype=np.float32)

        output_path = tmp_path / "precision.png"
        path, _, _ = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        loaded = read_depth_u16_png(path)

        # Check each value is preserved within quantization error
        max_error = np.max(np.abs(depth_map - loaded))
        # 16-bit quantization step is 1/65535 ≈ 1.5e-5
        assert max_error < 2e-5, f"Max error {max_error} exceeds quantization threshold"

    def test_verification_mode(self, tmp_path):
        """Verify debug_verify mode doesn't crash."""
        depth_map = np.random.rand(20, 20).astype(np.float32)
        output_path = tmp_path / "verified.png"

        # Should complete without error when verification enabled
        path, verification_path, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map, debug_verify=True)

        assert path.exists()
        # verification_path is currently always None (reserved for future use)
        assert verification_path is None


class TestDepthWriterEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_pixel(self, tmp_path):
        """Verify single-pixel depth map works."""
        depth_map = np.array([[0.5]], dtype=np.float32)
        output_path = tmp_path / "single_pixel.png"

        path, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        assert path.exists()
        assert stats.shape == (1, 1)

        loaded = read_depth_u16_png(path)
        assert abs(loaded[0, 0] - 0.5) < 1e-4

    def test_large_depth_map(self, tmp_path):
        """Verify large depth maps work (stress test)."""
        # 4K resolution
        depth_map = np.random.rand(2160, 3840).astype(np.float32)
        output_path = tmp_path / "large.png"

        path, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        assert path.exists()
        assert stats.shape == (2160, 3840)

        # Read back (verify it doesn't crash)
        loaded = read_depth_u16_png(path)
        assert loaded.shape == depth_map.shape

    def test_all_zeros(self, tmp_path):
        """Verify all-zero depth map works."""
        depth_map = np.zeros((50, 50), dtype=np.float32)
        output_path = tmp_path / "zeros.png"

        path, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        assert stats.min == 0.0
        assert stats.max == 0.0
        assert stats.mean == 0.0

        loaded = read_depth_u16_png(path)
        assert np.all(loaded == 0.0)

    def test_all_ones(self, tmp_path):
        """Verify all-one depth map works."""
        depth_map = np.ones((50, 50), dtype=np.float32)
        output_path = tmp_path / "ones.png"

        path, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        assert stats.min == 1.0
        assert stats.max == 1.0
        assert stats.mean == 1.0

        loaded = read_depth_u16_png(path)
        assert np.all(loaded > 0.99)  # Account for quantization


class TestDepthWriteStats:
    """Test DepthWriteStats dataclass and compatibility."""

    def test_stats_has_asdict_method(self, tmp_path):
        """Verify DepthWriteStats has _asdict() for orchestrator compatibility."""
        depth_map = np.random.rand(10, 10).astype(np.float32)
        output_path = tmp_path / "stats_test.png"

        _, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        # Should have _asdict method
        assert hasattr(stats, "_asdict")
        assert callable(stats._asdict)

        # Should return dict
        stats_dict = stats._asdict()
        assert isinstance(stats_dict, dict)
        assert "min" in stats_dict
        assert "max" in stats_dict
        assert "mean" in stats_dict
        assert "std" in stats_dict
        assert "shape" in stats_dict
        assert "dtype" in stats_dict
        assert "method" in stats_dict

    def test_stats_includes_method(self, tmp_path):
        """Verify stats includes method field."""
        depth_map = np.random.rand(10, 10).astype(np.float32)
        output_path = tmp_path / "method_test.png"

        _, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map, method="u16")

        assert stats.method == "u16"
        assert stats._asdict()["method"] == "u16"

    def test_method_normalization(self, tmp_path):
        """Verify legacy/config method values are normalized to u16."""
        depth_map = np.random.rand(10, 10).astype(np.float32)

        # Test each legacy value that should normalize to "u16"
        for legacy_method in ["none", "", None]:
            output_path = tmp_path / f"norm_{legacy_method or 'None'}.png"

            _, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map, method=legacy_method)

            # Should normalize to "u16"
            assert stats.method == "u16"
            assert output_path.exists()

    def test_invalid_method_raises_error(self, tmp_path):
        """Verify unsupported quantization methods raise ValueError."""
        depth_map = np.random.rand(10, 10).astype(np.float32)
        output_path = tmp_path / "invalid_method.png"

        # "none" is now normalized, so test with truly invalid methods
        with pytest.raises(ValueError, match="Unsupported depth quantization method"):
            atomic_write_depth_u16_png_with_stats(output_path, depth_map, method="u8")

        with pytest.raises(ValueError, match="Unsupported depth quantization method"):
            atomic_write_depth_u16_png_with_stats(output_path, depth_map, method="float32")

    def test_stats_is_frozen(self, tmp_path):
        """Verify DepthWriteStats is immutable (frozen dataclass)."""
        depth_map = np.random.rand(10, 10).astype(np.float32)
        output_path = tmp_path / "frozen_test.png"

        _, _, stats = atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        # Should not be able to modify
        with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
            stats.min = 999.0

    def test_no_temp_files_after_write(self, tmp_path):
        """Verify no temp files remain after successful write."""
        depth_map = np.random.rand(10, 10).astype(np.float32)
        output_path = tmp_path / "clean_test.png"

        atomic_write_depth_u16_png_with_stats(output_path, depth_map)

        # Verify output exists
        assert output_path.exists()

        # Verify no temp files remain (new atomic write helper should clean up)
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0, f"Temp files leaked: {temp_files}"

    def test_cleanup_on_write_failure(self, tmp_path):
        """Verify temp files are cleaned up even when write fails."""
        # Create an invalid depth map that will cause cv2.imwrite to fail
        # (cv2 doesn't support certain data types or shapes)
        invalid_map = np.zeros((10, 10, 10), dtype=np.float32)  # 3D instead of 2D
        output_path = tmp_path / "fail_test.png"

        try:
            # This should fail during the write
            atomic_write_depth_u16_png_with_stats(output_path, invalid_map)
        except Exception:
            # Failure expected
            pass

        # Verify no temp files remain even after failure
        temp_files = list(tmp_path.glob(".tmp_*"))
        assert len(temp_files) == 0, f"Temp files leaked after failure: {temp_files}"

        # Verify output file was not created
        assert not output_path.exists()


def test_exact_byte_depth_decoder_accepts_u16_grayscale(tmp_path: Path) -> None:
    depth_map = np.linspace(0.0, 1.0, 64, dtype=np.float32).reshape(8, 8)
    path, _, _ = atomic_write_depth_u16_png_with_stats(tmp_path / "depth.png", depth_map)

    decoded = read_depth_u16_png_bytes(path.read_bytes())

    assert decoded.shape == (8, 8)
    assert decoded.dtype == np.float32


def test_exact_byte_depth_decoder_rejects_uint8_png(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "uint8.png"
    assert cv2.imwrite(str(path), np.zeros((8, 8), dtype=np.uint8))

    with pytest.raises(IOError, match="16-bit grayscale"):
        read_depth_u16_png_bytes(path.read_bytes())


def test_exact_byte_depth_decoder_bounds_declared_pixels_before_decode(tmp_path: Path) -> None:
    depth_map = np.zeros((2, 2), dtype=np.float32)
    path, _, _ = atomic_write_depth_u16_png_with_stats(tmp_path / "depth.png", depth_map)
    payload = bytearray(path.read_bytes())
    oversized_width = MAX_DEPTH_PNG_DECODED_PIXELS + 1
    payload[16:20] = oversized_width.to_bytes(4, "big")

    with pytest.raises(IOError, match="decoded pixels exceed the bounded limit"):
        read_depth_u16_png_bytes(bytes(payload))
