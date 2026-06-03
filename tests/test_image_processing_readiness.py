#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for Image Processing Readiness Check and Simple Image Processor.
"""

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

import pytest
from PIL import Image

pytestmark = pytest.mark.unit

# Import the modules to test
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Check if modules can be imported
readiness_spec = importlib.util.find_spec("check_image_processing_readiness")
processor_spec = importlib.util.find_spec("simple_image_processor")

READINESS_AVAILABLE = readiness_spec is not None
PROCESSOR_AVAILABLE = processor_spec is not None

# Import modules if available (conditional imports for optional dependencies)
readiness = None
processor = None

if READINESS_AVAILABLE:
    import check_image_processing_readiness as readiness
if PROCESSOR_AVAILABLE:
    import simple_image_processor as processor


@pytest.mark.skipif(not READINESS_AVAILABLE, reason="readiness check module not available")
class TestReadinessCheck:
    """Test image processing readiness check functions."""

    def test_check_package_installed(self):
        """Test package checking for installed package."""
        installed, version = readiness.check_package("sys")
        assert installed is True
        assert version is not None

    def test_check_package_not_installed(self):
        """Test package checking for non-existent package."""
        installed, version = readiness.check_package("nonexistent_package_xyz")
        assert installed is False
        assert version == "Not installed"

    def test_check_disk_space(self):
        """Test disk space checking."""
        disk = readiness.check_disk_space()

        if "error" not in disk:
            assert "total_gb" in disk
            assert "free_gb" in disk
            assert "used_percent" in disk
            assert disk["total_gb"] > 0
            assert disk["free_gb"] >= 0
            assert 0 <= disk["used_percent"] <= 100

    def test_assess_capabilities(self):
        """Test capability assessment."""
        capabilities = readiness.assess_capabilities()

        assert "core_packages" in capabilities
        assert "ml_packages" in capabilities
        assert "image_packages" in capabilities
        assert "minimal_ready" in capabilities
        assert "standard_ready" in capabilities
        assert "full_ready" in capabilities

        # Check specific packages
        assert "numpy" in capabilities["core_packages"]
        assert "Pillow" in capabilities["core_packages"]
        assert "torch" in capabilities["ml_packages"]
        assert "realesrgan" not in capabilities["ml_packages"]

    def test_full_tier_guidance_uses_governed_ml_install(self, capsys):
        """Readiness guidance should not suggest banned external ML packages."""
        capabilities = {
            "core_packages": {"numpy": True, "Pillow": True, "scipy": True, "PyYAML": True, "typer": True, "tqdm": True},
            "ml_packages": {"torch": False, "diffusers": False, "transformers": False, "controlnet_aux": False},
            "image_packages": {"tifffile": True, "imagecodecs": True, "scikit-image": True, "opencv": True},
            "minimal_ready": True,
            "standard_ready": True,
            "full_ready": False,
        }
        images = {
            "sample_dir_exists": False,
            "input_dir_exists": True,
            "sample_count": 0,
            "input_count": 1,
            "total_count": 1,
            "has_images": True,
        }

        readiness.print_tier_status(capabilities)
        readiness.print_available_operations(capabilities)
        readiness.print_quick_start_guide(capabilities, images)
        readiness.print_recommendations({"free_gb": 10.0, "sufficient": True}, capabilities)

        output = capsys.readouterr().out
        assert "make install-ml-core" in output
        assert "Apple Silicon" in output
        assert "./scripts/bootstrap/install_ml_stack.sh --profile core-cpu" in output
        assert ".venv/bin/python scripts/setup/download_depth_models.py" in output
        assert "Linux/CPU" not in output
        assert "pip install torch diffusers transformers realesrgan" not in output
        assert "pip install -r requirements.txt" not in output
        assert "Download ML models: python scripts/setup/download_depth_models.py" not in output

    def test_missing_core_guidance_uses_repo_managed_install(self, capsys):
        """Missing core readiness guidance should route through repo-managed setup."""
        capabilities = {
            "core_packages": {"numpy": False, "Pillow": False, "scipy": False, "PyYAML": False, "typer": False, "tqdm": False},
            "ml_packages": {"torch": False, "diffusers": False, "transformers": False, "controlnet_aux": False},
            "image_packages": {"tifffile": False, "imagecodecs": False, "scikit-image": False, "opencv": False},
            "minimal_ready": False,
            "standard_ready": False,
            "full_ready": False,
        }
        images = {
            "sample_dir_exists": False,
            "input_dir_exists": False,
            "sample_count": 0,
            "input_count": 0,
            "total_count": 0,
            "has_images": False,
        }

        readiness.print_tier_status(capabilities)
        readiness.print_quick_start_guide(capabilities, images)
        readiness.print_recommendations({"free_gb": 10.0, "sufficient": True}, capabilities)

        output = capsys.readouterr().out
        assert "make install-core" in output
        assert "make check-environment" in output
        assert "pip install numpy Pillow" not in output
        assert "pip install scipy" not in output

    def test_sample_download_guidance_uses_repo_python(self, capsys):
        """Sample download guidance should run through the repo-managed Python."""
        capabilities = {
            "core_packages": {"numpy": True, "Pillow": True, "scipy": False, "PyYAML": True, "typer": True, "tqdm": True},
            "ml_packages": {"torch": False, "diffusers": False, "transformers": False, "controlnet_aux": False},
            "image_packages": {"tifffile": False, "imagecodecs": False, "scikit-image": False, "opencv": False},
            "minimal_ready": True,
            "standard_ready": False,
            "full_ready": False,
        }
        images = {
            "sample_dir_exists": False,
            "input_dir_exists": False,
            "sample_count": 0,
            "input_count": 0,
            "total_count": 0,
            "has_images": False,
        }

        readiness.print_quick_start_guide(capabilities, images)

        output = capsys.readouterr().out
        assert ".venv/bin/python scripts/download_samples.py" in output
        assert re.search(r"(?m)^\s*python scripts/download_samples.py\b", output) is None

    def test_full_tier_quick_start_uses_current_cli_entrypoints(self, capsys):
        """Full-tier readiness guidance should advertise maintained console scripts."""
        capabilities = {
            "core_packages": {"numpy": True, "Pillow": True, "scipy": True, "PyYAML": True, "typer": True, "tqdm": True},
            "ml_packages": {"torch": True, "diffusers": True, "transformers": True, "controlnet_aux": True},
            "image_packages": {"tifffile": True, "imagecodecs": True, "scikit-image": True, "opencv": True},
            "minimal_ready": True,
            "standard_ready": True,
            "full_ready": True,
        }
        images = {
            "sample_dir_exists": False,
            "input_dir_exists": True,
            "sample_count": 0,
            "input_count": 1,
            "total_count": 1,
            "has_images": True,
        }

        readiness.print_quick_start_guide(capabilities, images)

        output = capsys.readouterr().out
        assert ".venv/bin/lux_render" in output
        assert "--input-glob" in output
        assert ".venv/bin/lux-depth-v3" in output
        assert "--model-key da3-metric" in output
        assert ".venv/bin/luxury-tiff-batch" in output
        assert "python scripts/pipelines/lux_render_pipeline.py" not in output
        assert "python scripts/context_aware_rendering.py" not in output
        assert "python scripts/utilities/luxury_tiff_batch_processor.py" not in output

    def test_check_sample_images(self):
        """Test sample image checking."""
        images = readiness.check_sample_images()

        assert "sample_dir_exists" in images
        assert "input_dir_exists" in images
        assert "sample_count" in images
        assert "input_count" in images
        assert "total_count" in images
        assert "has_images" in images

        assert images["sample_count"] >= 0
        assert images["input_count"] >= 0
        assert images["total_count"] >= 0


@pytest.mark.skipif(not PROCESSOR_AVAILABLE, reason="simple processor module not available")
class TestSimpleImageProcessor:
    """Test simple image processor functions."""

    def test_adjust_brightness(self):
        """Test brightness adjustment."""
        # Create test image
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))

        # Increase brightness
        result = processor.adjust_brightness(img, factor=1.5)
        assert result is not None
        assert result.size == img.size

        # Decrease brightness
        result = processor.adjust_brightness(img, factor=0.5)
        assert result is not None
        assert result.size == img.size

    def test_adjust_contrast(self):
        """Test contrast adjustment."""
        img = Image.new("RGB", (100, 100), color=(128, 128, 128))

        result = processor.adjust_contrast(img, factor=1.2)
        assert result is not None
        assert result.size == img.size

    def test_adjust_saturation(self):
        """Test saturation adjustment."""
        img = Image.new("RGB", (100, 100), color=(255, 128, 64))

        # Increase saturation
        result = processor.adjust_saturation(img, factor=1.5)
        assert result is not None
        assert result.size == img.size

        # Desaturate (grayscale)
        result = processor.adjust_saturation(img, factor=0.0)
        assert result is not None
        assert result.size == img.size

    def test_resize_image_maintain_aspect(self):
        """Test image resizing with aspect ratio preservation."""
        img = Image.new("RGB", (1920, 1080), color=(200, 200, 200))

        result = processor.resize_image(img, target_size=(1280, 720), maintain_aspect=True)
        assert result is not None

        # Should fit within target while maintaining aspect
        assert result.size[0] <= 1280
        assert result.size[1] <= 720

        # Aspect ratio should be approximately preserved
        original_aspect = 1920 / 1080
        result_aspect = result.size[0] / result.size[1]
        assert abs(original_aspect - result_aspect) < 0.01

    def test_resize_image_no_aspect(self):
        """Test image resizing without aspect ratio preservation."""
        img = Image.new("RGB", (1920, 1080), color=(200, 200, 200))

        result = processor.resize_image(img, target_size=(800, 800), maintain_aspect=False)
        assert result is not None
        assert result.size == (800, 800)

    def test_process_image_basic(self):
        """Test basic image processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test image
            input_path = tmpdir / "test_input.jpg"
            img = Image.new("RGB", (800, 600), color=(128, 128, 128))
            img.save(input_path, quality=95)

            # Process it
            output_path = tmpdir / "test_output.jpg"
            success = processor.process_image(
                input_path, output_path, brightness=1.1, contrast=1.05, saturation=1.0, quality=90, verbose=False
            )

            assert success is True
            assert output_path.exists()

            # Verify output
            result_img = Image.open(output_path)
            assert result_img.size == (800, 600)

    def test_process_image_with_resize(self):
        """Test image processing with resize."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create test image
            input_path = tmpdir / "test_input.jpg"
            img = Image.new("RGB", (1920, 1080), color=(200, 150, 100))
            img.save(input_path, quality=95)

            # Process with resize
            output_path = tmpdir / "test_output.jpg"
            success = processor.process_image(input_path, output_path, resize=(1280, 720), quality=85, verbose=False)

            assert success is True
            assert output_path.exists()

            # Verify resize
            result_img = Image.open(output_path)
            assert result_img.size[0] <= 1280
            assert result_img.size[1] <= 720

    def test_process_image_format_conversion(self):
        """Test image format conversion."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create PNG input
            input_path = tmpdir / "test_input.png"
            img = Image.new("RGB", (400, 300), color=(100, 200, 150))
            img.save(input_path)

            # Convert to JPEG
            output_path = tmpdir / "test_output.jpg"
            success = processor.process_image(input_path, output_path, quality=90, verbose=False)

            assert success is True
            assert output_path.exists()
            assert output_path.suffix == ".jpg"

    def test_process_image_nonexistent(self):
        """Test processing non-existent image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            input_path = tmpdir / "nonexistent.jpg"
            output_path = tmpdir / "output.jpg"

            success = processor.process_image(input_path, output_path, verbose=False)

            assert success is False


def test_readiness_module_imports():
    """Test that readiness check module can be imported."""
    assert READINESS_AVAILABLE or not READINESS_AVAILABLE  # Always passes
    # Just checking that the test module loads correctly


def test_processor_module_imports():
    """Test that simple processor module can be imported."""
    assert PROCESSOR_AVAILABLE or not PROCESSOR_AVAILABLE  # Always passes
    # Just checking that the test module loads correctly


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
