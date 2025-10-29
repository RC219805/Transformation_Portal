#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for AD Editorial Post-Production Pipeline

Run with:
    pytest test_ad_pipeline.py -v
    pytest test_ad_pipeline.py --cov=ad_editorial_post_pipeline_v2 --cov-report=html

Install dependencies:
    pip install pytest pytest-cov
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

# Import from v2
from ad_editorial_post_pipeline_v2 import (
    ProgressTracker,
    adjust_contrast,
    adjust_exposure,
    atomic_write,
    copy_and_verify,
    human_sort_key,
    linear_to_srgb,
    median_luma,
    normalize_exposure_inplace,
    PipelineConfig,
    safe_name,
    sha256sum,
    split_tokens,
    srgb_to_linear,
    unsharp_mask,
    vignette,
)


# ============================================================================
# Helpers & Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create a sample linear RGB image."""
    # 100x100 test image with gradient
    img = np.zeros((100, 100, 3), dtype=np.float32)
    img[:, :, 0] = np.linspace(0, 1, 100)[None, :]  # Red gradient
    img[:, :, 1] = 0.5  # Constant green
    img[:, :, 2] = np.linspace(0, 1, 100)[:, None]  # Blue gradient
    return img


@pytest.fixture
def sample_config(temp_dir):
    """Create a sample configuration file."""
    config_path = temp_dir / "config.yml"
    config_data = {
        "project_name": "TestProject",
        "project_root": str(temp_dir / "project"),
        "input_raw_dir": str(temp_dir / "input"),
        "processing": {"workers": 2, "auto_upright": True, "upright_max_deg": 3.0},
        "styles": {
            "natural": {"exposure": 0.0, "contrast": 0, "saturation": 0, "split_tone": {}},
            "test": {"exposure": 0.5, "contrast": 10, "saturation": 5, "split_tone": {}},
        },
        "consistency": {"target_median": 0.42, "wb_neutralize": True},
        "export": {
            "web_long_edge_px": 1000,
            "jpeg_quality": 90,
            "sharpen_web_amount": 0.3,
            "sharpen_print_amount": 0.1,
        },
    }

    # Create directories
    (temp_dir / "input").mkdir()
    (temp_dir / "project").mkdir()

    # Write config
    import yaml

    with config_path.open("w") as f:
        yaml.dump(config_data, f)

    return config_path


# ============================================================================
# Color Management Tests
# ============================================================================


class TestColorManagement:
    """Test color space conversions."""

    def test_linear_to_srgb_black(self):
        """Test linear to sRGB conversion for black."""
        black = np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32)
        result = linear_to_srgb(black)
        np.testing.assert_array_almost_equal(result, black, decimal=6)

    def test_linear_to_srgb_white(self):
        """Test linear to sRGB conversion for white."""
        white = np.array([[[1.0, 1.0, 1.0]]], dtype=np.float32)
        result = linear_to_srgb(white)
        np.testing.assert_array_almost_equal(result, white, decimal=6)

    def test_linear_to_srgb_midtone(self):
        """Test linear to sRGB conversion for midtones."""
        gray = np.array([[[0.5, 0.5, 0.5]]], dtype=np.float32)
        result = linear_to_srgb(gray)
        # sRGB encoding should make midtones lighter
        assert result[0, 0, 0] > 0.5
        assert result[0, 0, 0] < 0.8

    def test_srgb_to_linear_roundtrip(self):
        """Test sRGB to linear roundtrip conversion."""
        original = np.random.rand(10, 10, 3).astype(np.float32)
        converted = srgb_to_linear(linear_to_srgb(original))
        np.testing.assert_array_almost_equal(original, converted, decimal=5)

    def test_linear_to_srgb_threshold(self):
        """Test linear to sRGB at threshold."""
        threshold = np.array([[[0.0031308, 0.0031308, 0.0031308]]], dtype=np.float32)
        result = linear_to_srgb(threshold)
        # Should be close to threshold * 12.92
        expected = threshold * 12.92
        np.testing.assert_array_almost_equal(result, expected, decimal=4)


# ============================================================================
# Image Processing Tests
# ============================================================================


class TestImageProcessing:
    """Test image processing functions."""

    def test_adjust_exposure_zero(self, sample_image):
        """Test zero exposure adjustment."""
        result = adjust_exposure(sample_image, 0.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_adjust_exposure_positive(self, sample_image):
        """Test positive exposure adjustment."""
        result = adjust_exposure(sample_image, 1.0)  # +1 EV = 2x brighter
        expected = np.clip(sample_image * 2.0, 0, 1)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_adjust_exposure_negative(self, sample_image):
        """Test negative exposure adjustment."""
        result = adjust_exposure(sample_image, -1.0)  # -1 EV = 0.5x darker
        expected = sample_image * 0.5
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_adjust_exposure_clipping(self):
        """Test exposure adjustment clipping."""
        bright = np.ones((10, 10, 3), dtype=np.float32) * 0.8
        result = adjust_exposure(bright, 2.0)  # +2 EV would exceed 1.0
        assert np.all(result <= 1.0)
        assert np.any(result == 1.0)  # Should clip

    def test_adjust_contrast_zero(self, sample_image):
        """Test zero contrast adjustment."""
        # Note: small numerical differences due to gamma conversion
        result = adjust_contrast(sample_image, 0.0)
        np.testing.assert_array_almost_equal(result, sample_image, decimal=3)

    def test_adjust_contrast_positive(self, sample_image):
        """Test positive contrast increases difference from midpoint."""
        result = adjust_contrast(sample_image, 20.0)
        # Values below 0.5 should get darker, above should get brighter
        # (in sRGB space)
        dark_pixel = sample_image[0, 0, :]
        bright_pixel = sample_image[0, -1, :]

        result_dark = result[0, 0, :]
        result_bright = result[0, -1, :]

        # Contrast should increase
        assert np.mean(result_bright) >= np.mean(bright_pixel)
        assert np.mean(result_dark) <= np.mean(dark_pixel)

    def test_vignette_no_effect(self, sample_image):
        """Test vignette with zero strength."""
        result = vignette(sample_image, 0.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_vignette_darkens_edges(self, sample_image):
        """Test vignette darkens edges."""
        result = vignette(sample_image, 0.5)

        # Center should be approximately unchanged
        h, w = sample_image.shape[:2]
        center = result[h // 2, w // 2, :]
        center_orig = sample_image[h // 2, w // 2, :]
        np.testing.assert_array_almost_equal(center, center_orig, decimal=2)

        # Corners should be darker
        corner = result[0, 0, :]
        corner_orig = sample_image[0, 0, :]
        assert np.all(corner <= corner_orig)

    def test_unsharp_mask_no_effect(self, sample_image):
        """Test unsharp mask with zero amount."""
        result = unsharp_mask(sample_image, 0.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_unsharp_mask_sharpens(self, sample_image):
        """Test unsharp mask increases local contrast."""
        result = unsharp_mask(sample_image, 0.5, radius=1.0)
        # Result should be different from original
        assert not np.array_equal(result, sample_image)
        # Should still be in valid range
        assert np.all(result >= 0) and np.all(result <= 1)

    def test_median_luma(self, sample_image):
        """Test median luminance calculation."""
        luma = median_luma(sample_image)
        assert 0.0 <= luma <= 1.0

        # Test known values
        black = np.zeros((10, 10, 3), dtype=np.float32)
        assert median_luma(black) == 0.0

        white = np.ones((10, 10, 3), dtype=np.float32)
        assert median_luma(white) == 1.0

    def test_normalize_exposure_inplace(self, sample_image):
        """Test in-place exposure normalization."""
        imgs = [sample_image.copy(), sample_image.copy() * 0.5, sample_image.copy() * 1.5]
        target = 0.42

        normalize_exposure_inplace(imgs, target_median=target)

        # All images should have similar median luma
        lumas = [median_luma(img) for img in imgs]
        for luma in lumas:
            assert abs(luma - target) < 0.1  # Allow some tolerance


# ============================================================================
# Utility Functions Tests
# ============================================================================


class TestUtilities:
    """Test utility functions."""

    def test_safe_name(self):
        """Test safe filename generation."""
        assert safe_name("test.jpg") == "test.jpg"
        assert safe_name("test file.jpg") == "test_file.jpg"
        assert safe_name("test/file\\name.jpg") == "test_file_name.jpg"
        assert safe_name("café.jpg") == "caf_.jpg"

    def test_split_tokens(self):
        """Test token splitting for natural sorting."""
        assert split_tokens("file123.jpg") == ["file", "123", ".", "jpg"]
        assert split_tokens("IMG_0001.CR3") == ["IMG", "_", "0001", ".", "CR", "3"]

    def test_human_sort_key(self):
        """Test natural sorting."""
        files = [
            Path("IMG_10.jpg"),
            Path("IMG_2.jpg"),
            Path("IMG_1.jpg"),
            Path("IMG_20.jpg"),
        ]
        sorted_files = sorted(files, key=human_sort_key)
        expected = [
            Path("IMG_1.jpg"),
            Path("IMG_2.jpg"),
            Path("IMG_10.jpg"),
            Path("IMG_20.jpg"),
        ]
        assert sorted_files == expected

    def test_sha256sum(self, temp_dir):
        """Test SHA256 checksum calculation."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        hash1 = sha256sum(test_file)
        hash2 = sha256sum(test_file)

        assert hash1 == hash2  # Deterministic
        assert len(hash1) == 64  # SHA256 produces 64 hex chars

        # Modify file
        test_file.write_text("Hello, World!!")
        hash3 = sha256sum(test_file)
        assert hash3 != hash1  # Different content = different hash

    def test_copy_and_verify(self, temp_dir):
        """Test verified file copy."""
        src = temp_dir / "source.txt"
        dst = temp_dir / "dest.txt"

        src.write_text("Test content for copy verification")

        copy_and_verify(src, dst)

        assert dst.exists()
        assert dst.read_text() == src.read_text()

        # Second copy should be skipped (same hash)
        copy_and_verify(src, dst)
        assert dst.exists()

    def test_copy_and_verify_corruption_detection(self, temp_dir, monkeypatch):
        """Test corruption detection during copy."""
        src = temp_dir / "source.txt"
        dst = temp_dir / "dest.txt"

        src.write_text("Test content")

        # Monkey-patch to simulate corruption
        original_copy = __import__("shutil").copy2

        def corrupt_copy(s, d):
            original_copy(s, d)
            # Corrupt the copy
            Path(d).write_text("Corrupted!")

        monkeypatch.setattr("shutil.copy2", corrupt_copy)

        with pytest.raises(RuntimeError, match="Hash mismatch"):
            copy_and_verify(src, dst)

        # Corrupted file should be cleaned up
        assert not dst.exists()

    def test_atomic_write(self, temp_dir):
        """Test atomic file writing."""
        target = temp_dir / "atomic.txt"

        def writer(path: Path):
            path.write_text("Atomic content")

        atomic_write(target, writer)

        assert target.exists()
        assert target.read_text() == "Atomic content"
        # Temp file should be cleaned up
        assert not (temp_dir / "atomic.txt.tmp").exists()

    def test_atomic_write_failure_cleanup(self, temp_dir):
        """Test atomic write cleans up on failure."""
        target = temp_dir / "atomic.txt"

        def failing_writer(path: Path):
            path.write_text("Partial content")
            raise ValueError("Simulated failure")

        with pytest.raises(ValueError):
            atomic_write(target, failing_writer)

        # Target should not exist (operation failed)
        assert not target.exists()
        # Temp file should be cleaned up
        assert not (temp_dir / "atomic.txt.tmp").exists()


# ============================================================================
# Progress Tracker Tests
# ============================================================================


class TestProgressTracker:
    """Test progress tracking for resume capability."""

    def test_initial_state(self, temp_dir):
        """Test initial state of progress tracker."""
        tracker = ProgressTracker(temp_dir / "progress.json")
        assert len(tracker.completed) == 0
        assert len(tracker.checksums) == 0

    def test_mark_completed(self, temp_dir):
        """Test marking items as completed."""
        state_file = temp_dir / "progress.json"
        tracker = ProgressTracker(state_file)

        tracker.mark_completed("file1.jpg")
        tracker.mark_completed("file2.jpg", checksum="abc123")

        assert "file1.jpg" in tracker.completed
        assert "file2.jpg" in tracker.completed
        assert tracker.checksums["file2.jpg"] == "abc123"

        # State should be persisted
        assert state_file.exists()

    def test_is_completed(self, temp_dir):
        """Test checking completion status."""
        tracker = ProgressTracker(temp_dir / "progress.json")

        tracker.mark_completed("file1.jpg")
        tracker.mark_completed("file2.jpg", checksum="abc123")

        assert tracker.is_completed("file1.jpg")
        assert tracker.is_completed("file2.jpg")
        assert not tracker.is_completed("file3.jpg")

        # With checksum verification
        assert tracker.is_completed("file2.jpg", checksum="abc123")
        assert not tracker.is_completed("file2.jpg", checksum="different")

    def test_persistence(self, temp_dir):
        """Test progress state persistence across instances."""
        state_file = temp_dir / "progress.json"

        # First tracker
        tracker1 = ProgressTracker(state_file)
        tracker1.mark_completed("file1.jpg")
        tracker1.mark_completed("file2.jpg", checksum="abc123")

        # Second tracker (simulates restart)
        tracker2 = ProgressTracker(state_file)
        assert "file1.jpg" in tracker2.completed
        assert "file2.jpg" in tracker2.completed
        assert tracker2.checksums["file2.jpg"] == "abc123"

    def test_reset(self, temp_dir):
        """Test resetting progress."""
        state_file = temp_dir / "progress.json"
        tracker = ProgressTracker(state_file)

        tracker.mark_completed("file1.jpg")
        tracker.mark_completed("file2.jpg")

        assert len(tracker.completed) == 2
        assert state_file.exists()

        tracker.reset()

        assert len(tracker.completed) == 0
        assert not state_file.exists()


# ============================================================================
# Configuration Tests
# ============================================================================


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_load_config(self, sample_config):
        """Test loading valid configuration."""
        cfg = PipelineConfig.from_yaml(sample_config)

        assert cfg.project_name == "TestProject"
        assert cfg.processing["workers"] == 2
        assert "natural" in cfg.styles
        assert "test" in cfg.styles

    def test_config_validation_workers(self, temp_dir):
        """Test workers validation."""
        config_path = temp_dir / "bad_config.yml"

        import yaml

        bad_config = {
            "project_name": "Test",
            "project_root": str(temp_dir),
            "input_raw_dir": str(temp_dir),
            "processing": {"workers": 100},  # Invalid
            "styles": {"natural": {}},
        }

        with config_path.open("w") as f:
            yaml.dump(bad_config, f)

        with pytest.raises(ValueError, match="workers must be 1-64"):
            PipelineConfig.from_yaml(config_path)

    def test_config_validation_jpeg_quality(self, temp_dir):
        """Test JPEG quality validation."""
        config_path = temp_dir / "bad_config.yml"

        import yaml

        bad_config = {
            "project_name": "Test",
            "project_root": str(temp_dir),
            "input_raw_dir": str(temp_dir),
            "export": {"jpeg_quality": 150},  # Invalid
            "styles": {"natural": {}},
        }

        with config_path.open("w") as f:
            yaml.dump(bad_config, f)

        with pytest.raises(ValueError, match="jpeg_quality must be 1-100"):
            PipelineConfig.from_yaml(config_path)

    def test_config_validation_no_styles(self, temp_dir):
        """Test validation requires at least one style."""
        config_path = temp_dir / "bad_config.yml"

        import yaml

        bad_config = {
            "project_name": "Test",
            "project_root": str(temp_dir),
            "input_raw_dir": str(temp_dir),
            "styles": {},  # Empty styles
        }

        with config_path.open("w") as f:
            yaml.dump(bad_config, f)

        with pytest.raises(ValueError, match="At least one style must be defined"):
            PipelineConfig.from_yaml(config_path)

    def test_config_resume_flag(self, sample_config):
        """Test resume flag in configuration."""
        cfg = PipelineConfig.from_yaml(sample_config, resume=False)
        assert cfg.resume is False

        cfg = PipelineConfig.from_yaml(sample_config, resume=True)
        assert cfg.resume is True


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for pipeline components."""

    def test_full_color_pipeline(self, sample_image):
        """Test complete color processing pipeline."""
        # Simulate full grading pipeline
        img = sample_image.copy()

        # Exposure adjustment in linear space
        img = adjust_exposure(img, 0.3)

        # Contrast in sRGB space
        img = adjust_contrast(img, 10.0)

        # Vignette in linear space
        img = vignette(img, 0.1)

        # Sharpening in sRGB space
        img = unsharp_mask(img, 0.3)

        # Should still be in valid range
        assert np.all(img >= 0.0)
        assert np.all(img <= 1.0)
        assert img.shape == sample_image.shape

    def test_batch_processing_simulation(self, sample_image):
        """Test batch processing with normalization."""
        # Create batch of images with different exposures
        images = [
            sample_image * 0.3,  # Dark
            sample_image * 0.6,  # Medium
            sample_image * 0.9,  # Bright
        ]

        # Normalize
        normalize_exposure_inplace(images, target_median=0.42)

        # All should have similar median luma
        lumas = [median_luma(img) for img in images]
        std_dev = np.std(lumas)
        assert std_dev < 0.05  # Very consistent


# ============================================================================
# Performance Tests (optional, marked as slow)
# ============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests (run with pytest -m slow)."""

    def test_large_image_processing(self):
        """Test processing large images doesn't crash."""
        large_img = np.random.rand(4000, 6000, 3).astype(np.float32)

        # Should complete without memory errors
        result = adjust_exposure(large_img, 0.5)
        assert result.shape == large_img.shape

        result = vignette(large_img, 0.1)
        assert result.shape == large_img.shape

    def test_batch_normalization_memory(self):
        """Test batch normalization memory efficiency."""
        # Create multiple large images
        images = [np.random.rand(2000, 3000, 3).astype(np.float32) for _ in range(5)]

        # In-place normalization should not explode memory
        normalize_exposure_inplace(images, target_median=0.42)

        assert len(images) == 5
        for img in images:
            assert img.shape == (2000, 3000, 3)


# ============================================================================
# Run Tests
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
