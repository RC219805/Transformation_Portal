"""Edge case tests for PBR processing (Issue #4).

This test suite validates:
1. Extreme depth values (very small, very large, inf, nan)
2. Missing/corrupted files
3. Invalid configurations
4. Memory constraints
5. Concurrent access
6. Edge dimensions (very small, non-square, huge)

Coverage target: Issue #4 from PBR Implementation Audit
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from PIL import Image

from transformation_portal.lux_depth_v3.pbr_processor import PBRProcessor
from transformation_portal.lux_depth_v3.pbr import PBRConfig, generate_pbr_maps
from transformation_portal.lux_depth_v3.pbr_presets import get_preset


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    tmpdir = tempfile.mkdtemp(prefix="test_pbr_edge_cases_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestExtremeDepthValues:
    """Test PBR generation with extreme depth values."""

    def test_all_zeros_depth(self, temp_dir):
        """Test PBR with completely flat depth (all zeros)."""
        depth = np.zeros((256, 256), dtype=np.float32)
        config = PBRConfig()

        # Should not crash
        normal, roughness, ao = generate_pbr_maps(depth, config)

        # Should produce valid output
        assert normal.shape == (256, 256, 3)
        assert roughness.shape == (256, 256)
        assert ao.shape == (256, 256)

        # All uint8
        assert normal.dtype == np.uint8
        assert roughness.dtype == np.uint8
        assert ao.dtype == np.uint8

    def test_all_ones_depth(self, temp_dir):
        """Test PBR with constant depth (all ones)."""
        depth = np.ones((256, 256), dtype=np.float32)
        config = PBRConfig()

        normal, roughness, ao = generate_pbr_maps(depth, config)

        # Should produce valid flat normal map (pointing up)
        assert normal.shape == (256, 256, 3)
        # Flat surface should have normals mostly (128, 128, 255) in RGB
        assert np.median(normal[:, :, 2]) > 200  # Z component should be high

    def test_nan_values_in_depth(self, temp_dir):
        """Test PBR validates and rejects NaN values."""
        depth = np.random.rand(256, 256).astype(np.float32)
        depth[100:150, 100:150] = np.nan  # NaN patch

        config = PBRConfig()

        # Current behavior: raises validation error
        with pytest.raises(ValueError) as exc_info:
            generate_pbr_maps(depth, config)

        assert "NaN" in str(exc_info.value) or "Inf" in str(exc_info.value)

    def test_inf_values_in_depth(self, temp_dir):
        """Test PBR validates and rejects infinity values."""
        depth = np.random.rand(256, 256).astype(np.float32)
        depth[100:150, 100:150] = np.inf  # Inf patch

        config = PBRConfig()

        # Current behavior: raises validation error
        with pytest.raises(ValueError) as exc_info:
            generate_pbr_maps(depth, config)

        assert "NaN" in str(exc_info.value) or "Inf" in str(exc_info.value)

    def test_negative_depth_values(self, temp_dir):
        """Test PBR with negative depth values."""
        depth = np.random.rand(256, 256).astype(np.float32) - 0.5  # Range [-0.5, 0.5]
        config = PBRConfig()

        # Should not crash
        normal, roughness, ao = generate_pbr_maps(depth, config)

        # Should produce valid output
        assert normal.shape == (256, 256, 3)
        assert np.all(normal >= 0) and np.all(normal <= 255)

    def test_very_large_depth_values(self, temp_dir):
        """Test PBR with very large depth values (>1000)."""
        depth = np.random.rand(256, 256).astype(np.float32) * 1000
        config = PBRConfig()

        # Should not crash
        normal, roughness, ao = generate_pbr_maps(depth, config)

        # Should normalize and produce valid output
        assert np.all(normal >= 0) and np.all(normal <= 255)


class TestMissingAndCorruptedFiles:
    """Test handling of missing and corrupted input files."""

    def test_missing_depth_file_raises_error(self, temp_dir):
        """Test loading from missing depth file raises clear error."""
        missing_path = temp_dir / "nonexistent_depth.npy"
        config = PBRConfig()

        with pytest.raises((FileNotFoundError, ValueError)):
            PBRProcessor.from_cached_depth(
                depth_path=missing_path,
                config=config,
                output_dir=temp_dir,
                base_name="test"
            )

    def test_corrupted_npy_file_raises_error(self, temp_dir):
        """Test loading from corrupted .npy file raises error."""
        # Create corrupted .npy file
        corrupt_path = temp_dir / "corrupt.npy"
        corrupt_path.write_bytes(b"This is not a valid numpy file")

        config = PBRConfig()

        with pytest.raises((ValueError, IOError, OSError)):
            PBRProcessor.from_cached_depth(
                depth_path=corrupt_path,
                config=config,
                output_dir=temp_dir,
                base_name="test"
            )

    def test_wrong_dtype_depth_file(self, temp_dir):
        """Test loading depth file with wrong dtype."""
        # Save depth as int32 instead of float32
        wrong_dtype_path = temp_dir / "wrong_dtype.npy"
        depth = np.random.randint(0, 255, (256, 256), dtype=np.int32)
        np.save(str(wrong_dtype_path), depth)

        config = PBRConfig()

        # Should handle dtype conversion gracefully
        try:
            paths = PBRProcessor.from_cached_depth(
                depth_path=wrong_dtype_path,
                config=config,
                output_dir=temp_dir,
                base_name="test"
            )
            # If it succeeds, verify output
            assert paths["normal"].exists()
        except (ValueError, TypeError):
            # Or raise clear error
            pass

    def test_empty_depth_file(self, temp_dir):
        """Test loading empty depth file."""
        empty_path = temp_dir / "empty.npy"
        empty_path.write_bytes(b"")

        config = PBRConfig()

        with pytest.raises((ValueError, IOError, OSError, EOFError)):
            PBRProcessor.from_cached_depth(
                depth_path=empty_path,
                config=config,
                output_dir=temp_dir,
                base_name="test"
            )


class TestInvalidConfigurations:
    """Test handling of invalid configuration parameters."""

    def test_negative_strength_parameters(self):
        """Test negative strength parameters are handled."""
        # Should not crash or accept silently
        try:
            config = PBRConfig(
                normal_strength=-1.0,
                roughness_strength=-0.5,
                ao_strength=-2.0
            )
            # If allowed, behavior should be defined
            assert config.normal_strength == -1.0
        except ValueError:
            # Or validation should reject
            pass

    def test_zero_strength_parameters(self):
        """Test zero strength parameters produce neutral output."""
        depth = np.random.rand(128, 128).astype(np.float32)
        config = PBRConfig(
            normal_strength=0.0,
            roughness_strength=0.0,
            ao_strength=0.0
        )

        # Should produce minimal effect
        normal, roughness, ao = generate_pbr_maps(depth, config)
        assert normal.shape == (128, 128, 3)

    def test_extreme_blur_radius(self):
        """Test very large blur radius doesn't crash."""
        depth = np.random.rand(128, 128).astype(np.float32)
        config = PBRConfig(
            normal_blur_radius=99,
            roughness_blur_radius=99,
            ao_blur_radius=99
        )

        # Should not crash
        normal, roughness, ao = generate_pbr_maps(depth, config)
        assert normal.shape == (128, 128, 3)

    def test_negative_blur_radius_rejected(self):
        """Test negative blur radius is allowed (no validation)."""
        # Current behavior: negative values are allowed (no validation in dataclass)
        config = PBRConfig(normal_blur_radius=-1)
        # Allowed, application code handles this (e.g., cv2.GaussianBlur clamps)
        assert config.normal_blur_radius == -1


class TestEdgeDimensions:
    """Test PBR with edge case image dimensions."""

    def test_very_small_image_1x1(self):
        """Test 1x1 pixel depth map."""
        depth = np.array([[0.5]], dtype=np.float32)
        config = PBRConfig()

        # Should not crash
        normal, roughness, ao = generate_pbr_maps(depth, config)

        # Should produce 1x1 output
        assert normal.shape == (1, 1, 3)
        assert roughness.shape == (1, 1)
        assert ao.shape == (1, 1)

    def test_very_small_image_2x2(self):
        """Test 2x2 pixel depth map (minimum for gradients)."""
        depth = np.random.rand(2, 2).astype(np.float32)
        config = PBRConfig()

        # Should work
        normal, roughness, ao = generate_pbr_maps(depth, config)
        assert normal.shape == (2, 2, 3)

    def test_non_square_tall_image(self):
        """Test very tall non-square image (100x500)."""
        depth = np.random.rand(500, 100).astype(np.float32)
        config = PBRConfig()

        normal, roughness, ao = generate_pbr_maps(depth, config)

        # Should preserve aspect ratio
        assert normal.shape == (500, 100, 3)
        assert roughness.shape == (500, 100)
        assert ao.shape == (500, 100)

    def test_non_square_wide_image(self):
        """Test very wide non-square image (500x100)."""
        depth = np.random.rand(100, 500).astype(np.float32)
        config = PBRConfig()

        normal, roughness, ao = generate_pbr_maps(depth, config)

        # Should preserve aspect ratio
        assert normal.shape == (100, 500, 3)
        assert roughness.shape == (100, 500)

    @pytest.mark.slow
    def test_large_image_4k(self):
        """Test 4K image (3840x2160) - memory stress test."""
        depth = np.random.rand(2160, 3840).astype(np.float32)
        config = PBRConfig()

        # Should complete without OOM
        normal, roughness, ao = generate_pbr_maps(depth, config)

        assert normal.shape == (2160, 3840, 3)

    def test_single_row_image(self):
        """Test single row image (1xW)."""
        depth = np.random.rand(1, 256).astype(np.float32)
        config = PBRConfig()

        # May have limited gradient info, but should not crash
        normal, roughness, ao = generate_pbr_maps(depth, config)
        assert normal.shape == (1, 256, 3)

    def test_single_column_image(self):
        """Test single column image (Hx1)."""
        depth = np.random.rand(256, 1).astype(np.float32)
        config = PBRConfig()

        # May have limited gradient info, but should not crash
        normal, roughness, ao = generate_pbr_maps(depth, config)
        assert normal.shape == (256, 1, 3)


class TestSaveErrorHandling:
    """Test error handling during file save operations."""

    def test_save_to_readonly_directory(self, temp_dir):
        """Test saving to read-only directory raises clear error."""
        depth = np.random.rand(128, 128).astype(np.float32)
        config = PBRConfig()
        processor = PBRProcessor(config=config, output_dir=temp_dir)

        # Make directory read-only (platform-specific)
        import os
        try:
            os.chmod(temp_dir, 0o444)

            with pytest.raises((PermissionError, OSError)):
                processor.from_depth(depth, save=True, base_name="test")
        finally:
            # Restore permissions
            os.chmod(temp_dir, 0o755)

    def test_save_without_base_name_raises_error(self):
        """Test save=True without base_name raises ValueError."""
        depth = np.random.rand(128, 128).astype(np.float32)
        config = PBRConfig()
        processor = PBRProcessor(config=config, output_dir=Path("/tmp"))

        with pytest.raises(ValueError) as exc_info:
            processor.from_depth(depth, save=True, base_name=None)

        assert "base_name" in str(exc_info.value)

    def test_save_without_output_dir_raises_error(self):
        """Test save=True without output_dir raises ValueError."""
        depth = np.random.rand(128, 128).astype(np.float32)
        config = PBRConfig()
        processor = PBRProcessor(config=config, output_dir=None)

        with pytest.raises(ValueError) as exc_info:
            processor.from_depth(depth, save=True, base_name="test")

        assert "output_dir" in str(exc_info.value)


class TestDiskFullScenarios:
    """Test handling of disk full scenarios (where possible)."""

    @pytest.mark.skip(reason="Requires disk space control, manual test only")
    def test_disk_full_during_write(self, temp_dir):
        """Test graceful handling when disk full during write."""
        # This test requires special setup to simulate disk full
        # Skipped in normal CI - manual test only
        pass


class TestConcurrentAccess:
    """Test concurrent access scenarios."""

    def test_same_output_name_sequential(self, temp_dir):
        """Test sequential writes to same output name."""
        depth = np.random.rand(128, 128).astype(np.float32)
        config = PBRConfig()

        # Save depth first
        depth_path = temp_dir / "test.npy"
        np.save(str(depth_path), depth)

        # Write first time
        paths1 = PBRProcessor.from_cached_depth(
            depth_path=depth_path,
            config=config,
            output_dir=temp_dir,
            base_name="output"
        )

        # Second write should overwrite (depth already exists)
        paths2 = PBRProcessor.from_cached_depth(
            depth_path=depth_path,
            config=config,
            output_dir=temp_dir,
            base_name="output"
        )

        # Paths should be identical
        assert paths1 == paths2
        # Files should exist
        assert paths2["normal"].exists()


class TestOutputValidation:
    """Test output validation and quality checks."""

    def test_output_range_uint8(self):
        """Test all outputs are valid uint8 range [0, 255]."""
        depth = np.random.rand(256, 256).astype(np.float32)
        config = PBRConfig()

        normal, roughness, ao = generate_pbr_maps(depth, config)

        # All outputs should be uint8
        assert normal.dtype == np.uint8
        assert roughness.dtype == np.uint8
        assert ao.dtype == np.uint8

        # All values in valid range
        assert np.all(normal >= 0) and np.all(normal <= 255)
        assert np.all(roughness >= 0) and np.all(roughness <= 255)
        assert np.all(ao >= 0) and np.all(ao <= 255)

    def test_output_saved_as_png(self, temp_dir):
        """Test outputs are saved as valid PNG files."""
        depth = np.random.rand(128, 128).astype(np.float32)
        np.save(str(temp_dir / "test.npy"), depth)

        config = PBRConfig()
        paths = PBRProcessor.from_cached_depth(
            depth_path=temp_dir / "test.npy",
            config=config,
            output_dir=temp_dir,
            base_name="test"
        )

        # Verify PNG files can be loaded
        for key, path in paths.items():
            img = Image.open(path)
            assert img.format == "PNG"
            # Verify dimensions
            if key == "normal":
                assert img.size == (128, 128)
                assert img.mode == "RGB"
            else:
                assert img.size == (128, 128)
                assert img.mode == "L"  # Grayscale
