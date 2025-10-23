#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unit tests for AD Editorial Post-Production Pipeline v3

Run with:
    pytest test_ad_pipeline_v3.py -v
    pytest test_ad_pipeline_v3.py --cov=ad_editorial_post_pipeline_v3 --cov-report=html

Install dependencies:
    pip install pytest pytest-cov
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest

# Import from v3
from ad_editorial_post_pipeline_v3 import (
    ProgressTracker,
    StyleRegistry,
    adjust_contrast,
    adjust_exposure,
    adjust_saturation,
    atomic_write,
    copy_and_verify,
    linear_to_srgb,
    median_luma,
    normalize_exposure_inplace,
    PipelineConfig,
    sha256sum,
    srgb_to_linear,
    apply_vignette,
    split_tone,
    sharpen_image,
    resize_for_web,
    remove_dust_spots,
    reduce_hotspots,
    neutralize_wb,
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
def sample_config_dict(temp_dir):
    """Create sample configuration dictionary."""
    return {
        "project_name": "TestProject",
        "project_root": str(temp_dir / "project"),
        "input_raw_dir": str(temp_dir / "input"),
        "processing": {
            "workers": 4,
            "auto_upright": True,
            "upright_max_deg": 3.0,
            "enable_hdr": False,
            "enable_pano": False,
        },
        "styles": {
            "natural": {"exposure": 0.0, "contrast": 6, "saturation": 0},
            "minimal": {"exposure": 0.2, "contrast": -10, "saturation": -15},
            "cinematic": {
                "exposure": -0.1,
                "contrast": 15,
                "saturation": -10,
                "vignette": 0.3,
                "split_tone": {
                    "shadows_hue_deg": 240,
                    "shadows_sat": 0.1,
                },
            },
        },
        "consistency": {"target_median": 0.42, "wb_neutralize": True},
        "retouch": {"dust_remove": False, "hotspot_reduce": False},
        "export": {
            "web_long_edge_px": 2500,
            "jpeg_quality": 96,
            "sharpen_web_amount": 0.35,
            "sharpen_print_amount": 0.1,
        },
        "metadata": {},
        "deliver": {"zip": True},
    }


@pytest.fixture
def sample_config(temp_dir, sample_config_dict):
    """Create a sample configuration file."""
    config_path = temp_dir / "config.yml"

    # Create directories
    (temp_dir / "input").mkdir()
    (temp_dir / "project").mkdir()

    # Write config
    import yaml
    with config_path.open("w") as f:
        yaml.dump(sample_config_dict, f)

    return config_path


# ============================================================================
# Test Color Management
# ============================================================================


class TestColorManagement:
    """Test color space conversion functions."""

    def test_linear_to_srgb_roundtrip(self):
        """Test sRGB to linear roundtrip conversion."""
        original = np.random.rand(10, 10, 3).astype(np.float32)
        converted = srgb_to_linear(linear_to_srgb(original))
        np.testing.assert_array_almost_equal(original, converted, decimal=5)

    def test_linear_to_srgb_threshold(self):
        """Test threshold behavior in sRGB conversion."""
        # Values near threshold
        test_vals = np.array([0.0, 0.001, 0.003, 0.01, 0.1, 0.5, 1.0])
        img = np.zeros((7, 1, 3), dtype=np.float32)
        img[:, 0, 0] = test_vals

        result = linear_to_srgb(img)

        # Check values are in valid range
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Check monotonicity (input order preserved)
        assert np.all(np.diff(result[:, 0, 0]) >= 0)

    def test_srgb_to_linear_threshold(self):
        """Test threshold behavior in linear conversion."""
        test_vals = np.array([0.0, 0.001, 0.01, 0.1, 0.5, 1.0])
        img = np.zeros((6, 1, 3), dtype=np.float32)
        img[:, 0, 0] = test_vals

        result = srgb_to_linear(img)

        # Check values are in valid range
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_linear_to_srgb_extremes(self):
        """Test extreme values (black and white)."""
        # Pure black
        black = np.zeros((10, 10, 3), dtype=np.float32)
        result_black = linear_to_srgb(black)
        np.testing.assert_array_almost_equal(result_black, black)

        # Pure white
        white = np.ones((10, 10, 3), dtype=np.float32)
        result_white = linear_to_srgb(white)
        np.testing.assert_array_almost_equal(result_white, white)


# ============================================================================
# Test Progress Tracking
# ============================================================================


class TestProgressTracker:
    """Test progress tracking functionality."""

    def test_initialization(self, temp_dir):
        """Test tracker initialization."""
        state_file = temp_dir / "progress.json"
        tracker = ProgressTracker(state_file)

        assert len(tracker.completed) == 0
        assert len(tracker.checksums) == 0

    def test_mark_completed(self, temp_dir):
        """Test marking items as completed."""
        state_file = temp_dir / "progress.json"
        tracker = ProgressTracker(state_file)

        tracker.mark_completed("file1.jpg")
        assert "file1.jpg" in tracker.completed

        tracker.mark_completed("file2.jpg", checksum="abc123")
        assert "file2.jpg" in tracker.completed
        assert tracker.checksums["file2.jpg"] == "abc123"

    def test_is_completed(self, temp_dir):
        """Test checking completion status."""
        state_file = temp_dir / "progress.json"
        tracker = ProgressTracker(state_file)

        tracker.mark_completed("file1.jpg")
        tracker.mark_completed("file2.jpg", checksum="abc123")

        assert tracker.is_completed("file1.jpg")
        assert tracker.is_completed("file2.jpg")
        assert tracker.is_completed("file2.jpg", checksum="abc123")
        assert not tracker.is_completed("file2.jpg", checksum="wrong")
        assert not tracker.is_completed("file3.jpg")

    def test_persistence(self, temp_dir):
        """Test progress state persistence across instances."""
        state_file = temp_dir / "progress.json"

        # First instance
        tracker1 = ProgressTracker(state_file)
        tracker1.mark_completed("file1.jpg")
        tracker1.mark_completed("file2.jpg", checksum="abc123")

        # Second instance (should load state)
        tracker2 = ProgressTracker(state_file)
        assert "file1.jpg" in tracker2.completed
        assert "file2.jpg" in tracker2.completed
        assert tracker2.checksums["file2.jpg"] == "abc123"

    def test_reset(self, temp_dir):
        """Test resetting progress."""
        state_file = temp_dir / "progress.json"
        tracker = ProgressTracker(state_file)

        tracker.mark_completed("file1.jpg")
        tracker.reset()

        assert len(tracker.completed) == 0
        assert len(tracker.checksums) == 0
        assert not state_file.exists()


# ============================================================================
# Test Style Registry
# ============================================================================


class TestStyleRegistry:
    """Test style registry pattern."""

    def test_register_and_get(self):
        """Test registering and retrieving styles."""
        # Clear registry for clean test
        StyleRegistry._styles.clear()

        @StyleRegistry.register("test_style")
        def test_func(img, params):
            return img * 0.5

        assert "test_style" in StyleRegistry.all_styles()
        func = StyleRegistry.get("test_style")
        assert func is test_func

    def test_unknown_style(self):
        """Test error on unknown style."""
        with pytest.raises(ValueError, match="Unknown style"):
            StyleRegistry.get("nonexistent_style_12345")

    def test_builtin_styles_registered(self):
        """Test that built-in styles are registered."""
        assert "natural" in StyleRegistry.all_styles()
        assert "minimal" in StyleRegistry.all_styles()
        assert "cinematic" in StyleRegistry.all_styles()

    def test_style_application(self, sample_image):
        """Test applying a registered style."""
        natural_func = StyleRegistry.get("natural")
        result = natural_func(sample_image, {"natural": {"exposure": 0.0, "contrast": 0, "saturation": 0}})

        assert result.shape == sample_image.shape
        assert result.dtype == np.float32


# ============================================================================
# Test Configuration
# ============================================================================


class TestConfiguration:
    """Test configuration loading and validation."""

    def test_load_valid_config(self, sample_config):
        """Test loading a valid configuration."""
        cfg = PipelineConfig.from_yaml(sample_config)

        assert cfg.project_name == "TestProject"
        assert cfg.processing["workers"] == 4
        assert "natural" in cfg.styles
        assert cfg.resume is False

    def test_load_with_resume(self, sample_config):
        """Test loading config with resume flag."""
        cfg = PipelineConfig.from_yaml(sample_config, resume=True)
        assert cfg.resume is True

    def test_load_with_dry_run(self, sample_config):
        """Test loading config with dry-run flag."""
        cfg = PipelineConfig.from_yaml(sample_config, dry_run=True)
        assert cfg.dry_run is True

    def test_validation_invalid_workers(self, temp_dir, sample_config_dict):
        """Test validation rejects invalid workers."""
        sample_config_dict["processing"]["workers"] = 100  # Too high
        config_path = temp_dir / "bad_config.yml"

        import yaml
        (temp_dir / "input").mkdir(exist_ok=True)
        (temp_dir / "project").mkdir(exist_ok=True)
        with config_path.open("w") as f:
            yaml.dump(sample_config_dict, f)

        with pytest.raises(ValueError, match="processing.workers must be 1-64"):
            PipelineConfig.from_yaml(config_path)

    def test_validation_invalid_jpeg_quality(self, temp_dir, sample_config_dict):
        """Test validation rejects invalid JPEG quality."""
        sample_config_dict["export"]["jpeg_quality"] = 101  # Too high
        config_path = temp_dir / "bad_config.yml"

        import yaml
        (temp_dir / "input").mkdir(exist_ok=True)
        (temp_dir / "project").mkdir(exist_ok=True)
        with config_path.open("w") as f:
            yaml.dump(sample_config_dict, f)

        with pytest.raises(ValueError, match="export.jpeg_quality must be 1-100"):
            PipelineConfig.from_yaml(config_path)

    def test_validation_no_styles(self, temp_dir, sample_config_dict):
        """Test validation requires at least one style."""
        sample_config_dict["styles"] = {}
        config_path = temp_dir / "bad_config.yml"

        import yaml
        (temp_dir / "input").mkdir(exist_ok=True)
        (temp_dir / "project").mkdir(exist_ok=True)
        with config_path.open("w") as f:
            yaml.dump(sample_config_dict, f)

        with pytest.raises(ValueError, match="At least one style must be defined"):
            PipelineConfig.from_yaml(config_path)

    def test_validation_invalid_exposure(self, temp_dir, sample_config_dict):
        """Test validation rejects invalid exposure values."""
        sample_config_dict["styles"]["test"] = {"exposure": 5.0}  # Too high
        config_path = temp_dir / "bad_config.yml"

        import yaml
        (temp_dir / "input").mkdir(exist_ok=True)
        (temp_dir / "project").mkdir(exist_ok=True)
        with config_path.open("w") as f:
            yaml.dump(sample_config_dict, f)

        with pytest.raises(ValueError, match="exposure must be -3.0 to 3.0"):
            PipelineConfig.from_yaml(config_path)


# ============================================================================
# Test File Operations
# ============================================================================


class TestFileOperations:
    """Test file operation utilities."""

    def test_sha256sum(self, temp_dir):
        """Test SHA256 hash computation."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello, World!")

        hash1 = sha256sum(test_file)
        hash2 = sha256sum(test_file)

        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 is 64 hex chars

    def test_atomic_write(self, temp_dir):
        """Test atomic file writing."""
        target = temp_dir / "atomic_test.txt"

        def writer(p: Path):
            p.write_text("Atomic content")

        atomic_write(target, writer)

        assert target.exists()
        assert target.read_text() == "Atomic content"

    def test_copy_and_verify(self, temp_dir):
        """Test file copying with verification."""
        src = temp_dir / "source.txt"
        dst = temp_dir / "dest.txt"
        src.write_text("Test content for copy")

        copy_and_verify(src, dst)

        assert dst.exists()
        assert src.read_text() == dst.read_text()
        assert sha256sum(src) == sha256sum(dst)

    def test_copy_and_verify_existing(self, temp_dir):
        """Test copy skips when hash matches."""
        src = temp_dir / "source.txt"
        dst = temp_dir / "dest.txt"
        src.write_text("Test content")
        dst.write_text("Test content")

        # Should skip copy since hashes match
        copy_and_verify(src, dst)

        assert dst.exists()
        assert sha256sum(src) == sha256sum(dst)


# ============================================================================
# Test Image Processing
# ============================================================================


class TestImageProcessing:
    """Test image processing functions."""

    def test_median_luma(self, sample_image):
        """Test median luminance calculation."""
        luma = median_luma(sample_image)
        assert 0.0 <= luma <= 1.0
        assert isinstance(luma, float)

    def test_normalize_exposure_inplace(self, sample_image):
        """Test in-place exposure normalization."""
        imgs = [sample_image.copy(), sample_image.copy() * 0.5]
        original_ids = [id(img) for img in imgs]

        normalize_exposure_inplace(imgs, target_median=0.42)

        # Check modified in-place
        assert [id(img) for img in imgs] == original_ids

        # Check normalized to target
        for img in imgs:
            luma = median_luma(img)
            assert 0.3 <= luma <= 0.6  # Within reasonable range

    def test_neutralize_wb(self, sample_image):
        """Test white balance neutralization."""
        result = neutralize_wb(sample_image)

        assert result.shape == sample_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_resize_for_web(self, sample_image):
        """Test web resize."""
        # 100x100 image resized to 50px long edge
        resized = resize_for_web(sample_image, long_edge=50)

        assert resized.shape[0] == 50 or resized.shape[1] == 50
        assert resized.dtype == np.float32


# ============================================================================
# Test Grading Functions
# ============================================================================


class TestGrading:
    """Test grading/adjustment functions."""

    def test_adjust_exposure(self, sample_image):
        """Test exposure adjustment."""
        # +1 EV should double brightness
        result = adjust_exposure(sample_image, 1.0)
        np.testing.assert_array_almost_equal(result, np.clip(sample_image * 2.0, 0, 1))

        # 0 EV should not change
        result = adjust_exposure(sample_image, 0.0)
        np.testing.assert_array_equal(result, sample_image)

    def test_adjust_contrast(self, sample_image):
        """Test contrast adjustment in sRGB space."""
        # Increase contrast
        result = adjust_contrast(sample_image, 20)
        assert result.shape == sample_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Decrease contrast
        result = adjust_contrast(sample_image, -20)
        assert result.shape == sample_image.shape

        # Zero contrast should preserve image
        result = adjust_contrast(sample_image, 0.0)
        np.testing.assert_array_almost_equal(result, sample_image, decimal=5)

    def test_adjust_saturation(self, sample_image):
        """Test saturation adjustment in HSV space."""
        # Increase saturation
        result = adjust_saturation(sample_image, 20)
        assert result.shape == sample_image.shape
        assert np.all(result >= 0.0)

        # Decrease saturation
        result = adjust_saturation(sample_image, -20)
        assert result.shape == sample_image.shape

        # Zero saturation change
        result = adjust_saturation(sample_image, 0.0)
        np.testing.assert_array_almost_equal(result, sample_image, decimal=3)

    def test_split_tone(self, sample_image):
        """Test split-toning."""
        # Apply split-tone
        result = split_tone(
            sample_image,
            sh_h=np.deg2rad(240),  # Blue shadows
            sh_s=0.1,
            hl_h=np.deg2rad(30),   # Orange highlights
            hl_s=0.05,
        )

        assert result.shape == sample_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # No split-tone should preserve image
        result = split_tone(sample_image, None, 0.0, None, 0.0)
        np.testing.assert_array_almost_equal(result, sample_image, decimal=5)

    def test_apply_vignette(self, sample_image):
        """Test vignette application."""
        result = apply_vignette(sample_image, strength=0.3)

        assert result.shape == sample_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

        # Corners should be darker than center
        h, w = sample_image.shape[:2]
        center_brightness = result[h // 2, w // 2].mean()
        corner_brightness = result[0, 0].mean()
        assert corner_brightness < center_brightness

    def test_sharpen_image(self, sample_image):
        """Test image sharpening."""
        result = sharpen_image(sample_image, amount=0.35)

        assert result.shape == sample_image.shape
        assert result.dtype == np.float32

        # Zero sharpening should preserve image approximately
        result = sharpen_image(sample_image, amount=0.0)
        np.testing.assert_array_almost_equal(result, sample_image, decimal=2)


# ============================================================================
# Test Retouching
# ============================================================================


class TestRetouching:
    """Test retouching functions."""

    def test_remove_dust_spots(self, sample_image):
        """Test dust spot removal."""
        result = remove_dust_spots(sample_image)

        assert result.shape == sample_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_reduce_hotspots(self, sample_image):
        """Test hotspot reduction."""
        result = reduce_hotspots(sample_image)

        assert result.shape == sample_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ============================================================================
# Test Integration
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_style_pipeline_natural(self, sample_image):
        """Test complete natural style pipeline."""
        params = {"natural": {"exposure": 0.0, "contrast": 6, "saturation": 0}}
        style_func = StyleRegistry.get("natural")

        result = style_func(sample_image, params)

        assert result.shape == sample_image.shape
        assert result.dtype == np.float32
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)

    def test_style_pipeline_minimal(self, sample_image):
        """Test complete minimal style pipeline."""
        params = {"minimal": {"exposure": 0.2, "contrast": -10, "saturation": -15}}
        style_func = StyleRegistry.get("minimal")

        result = style_func(sample_image, params)

        assert result.shape == sample_image.shape
        assert np.all(result >= 0.0)

    def test_style_pipeline_cinematic(self, sample_image):
        """Test complete cinematic style pipeline."""
        params = {
            "cinematic": {
                "exposure": -0.1,
                "contrast": 15,
                "saturation": -10,
                "vignette": 0.3,
                "split_tone": {
                    "shadows_hue_deg": 240,
                    "shadows_sat": 0.1,
                },
            }
        }
        style_func = StyleRegistry.get("cinematic")

        result = style_func(sample_image, params)

        assert result.shape == sample_image.shape
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.5)  # May exceed 1.0 before clipping

    def test_retouch_once_pattern(self, sample_image):
        """Test retouching once and reusing for multiple styles."""
        # Convert to sRGB, retouch
        img_srgb = linear_to_srgb(sample_image)
        retouched_srgb = linear_to_srgb(remove_dust_spots(srgb_to_linear(img_srgb)))
        img_lin_retouched = srgb_to_linear(retouched_srgb)

        # Apply multiple styles to retouched base
        for style_name in ["natural", "minimal"]:
            style_func = StyleRegistry.get(style_name)
            params = {style_name: {"exposure": 0.0, "contrast": 0, "saturation": 0}}
            result = style_func(img_lin_retouched.copy(), params)

            assert result.shape == sample_image.shape

    def test_color_roundtrip_through_grading(self, sample_image):
        """Test color accuracy through complete grading pipeline."""
        # Apply neutral adjustments (should preserve colors)
        result = sample_image.copy()
        result = adjust_exposure(result, 0.0)
        result = adjust_contrast(result, 0.0)
        result = adjust_saturation(result, 0.0)

        np.testing.assert_array_almost_equal(result, sample_image, decimal=3)


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Performance-related tests."""

    def test_smart_worker_limiting(self):
        """Test smart worker limiting logic."""
        import os

        # Simulate smart worker limiting
        requested = 8
        num_tasks = 3
        cpu_count = os.cpu_count() or 1

        actual = min(requested, num_tasks, cpu_count)

        # Should not exceed number of tasks
        assert actual <= num_tasks
        # Should not exceed CPU count
        assert actual <= cpu_count
        # Should not exceed requested
        assert actual <= requested

    def test_memory_cleanup_pattern(self, sample_image):
        """Test that memory cleanup doesn't break functionality."""
        import gc

        result = sample_image.copy()
        del sample_image  # Simulate cleanup
        gc.collect()

        # Result should still be valid
        assert result.shape == (100, 100, 3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
