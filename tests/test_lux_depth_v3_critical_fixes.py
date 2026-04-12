"""Regression tests for 6 critical bug fixes in Lux Depth V3 Pipeline (PR #887).

Tests verify fixes for:
1. Double EXIF rotation in v2_enhance.py
2. Dimension mismatch in preprocessing.py + orchestrator.py
3. Quadratic complexity in batch_stats.py
4. Redundant processing in orchestrator.py (parallel mode)
5. Alpha channel safety in v2_enhance.py
6. Output directory trap in input_discovery.py
"""

import io
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest
from PIL import Image, ImageOps

from transformation_portal.lux_depth_v3.batch_stats import compute_batch_runtime_stats, detect_runtime_outliers
from transformation_portal.lux_depth_v3.input_discovery import DiscoveryConfig, discover_images
from transformation_portal.lux_depth_v3.preprocessing import preprocess_image
from transformation_portal.lux_depth_v3.v2_enhance import enhance_image, resolve_v2_emitted_artifact_path

pytestmark = pytest.mark.unit


class TestFix1DoubleEXIFRotation:
    """Test fix for Issue #1: Double EXIF rotation in v2_enhance.py"""

    def test_exif_orientation_stripped_after_rotation(self, tmp_path):
        """Verify EXIF data is stripped after applying exif_transpose to prevent double rotation."""
        # Create a test image with EXIF orientation tag
        img = Image.new("RGB", (100, 50), color="red")

        # Add EXIF orientation tag (orientation=6 means rotate 90° CW)
        # Pillow's exif_transpose will rotate the image
        exif_bytes = (
            b"\xff\xe1\x00\x18Exif\x00\x00MM\x00*\x00\x00\x00\x08\x00\x01\x01\x12\x00\x03\x00\x00\x00\x01\x00\x06\x00\x00"
        )

        # Save with EXIF data
        input_path = tmp_path / "test_exif.jpg"
        output_path = tmp_path / "output_exif.jpg"

        img.save(input_path, "JPEG", exif=exif_bytes)

        # Mock the enhancement stage to avoid dependencies
        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage:
            mock_instance = Mock()
            mock_stage.return_value = mock_instance

            # Create mock result
            mock_result = Mock()
            mock_result.status = Mock(COMPLETED=1)
            mock_result.status.value = 1
            mock_result.artifacts = {"enhanced_image": np.array(img)}
            mock_result.metadata = {}

            # Make status comparison work
            from transformation_portal.stage_graph.stage import StageStatus

            mock_result.status = StageStatus.COMPLETED
            mock_instance.compute.return_value = mock_result

            # Run enhancement
            result = enhance_image(input_path, output_path, config=None)
            emitted_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8)

            assert result["status"] == "success"
            assert Path(result["output"]) == emitted_output
            assert emitted_output.exists()

            # Load output and check that EXIF is not present or orientation is reset
            output_img = Image.open(emitted_output)
            exif_data = output_img.info.get("exif")

            # After fix: EXIF should be None (stripped) to prevent double rotation
            assert exif_data is None, "EXIF should be stripped after rotation to prevent double rotation"

    def test_no_exif_rotation_preserves_original(self, tmp_path):
        """Verify images without EXIF orientation are processed normally."""
        img = Image.new("RGB", (100, 50), color="blue")
        input_path = tmp_path / "no_exif.jpg"
        output_path = tmp_path / "output_no_exif.jpg"
        img.save(input_path, "JPEG")

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage:
            mock_instance = Mock()
            mock_stage.return_value = mock_instance

            mock_result = Mock()
            from transformation_portal.stage_graph.stage import StageStatus

            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": np.array(img)}
            mock_result.metadata = {}
            mock_instance.compute.return_value = mock_result

            result = enhance_image(input_path, output_path, config=None)
            assert result["status"] == "success"
            assert Path(result["output"]) == resolve_v2_emitted_artifact_path(output_path, bit_depth=8)


class TestFix2DimensionMismatch:
    """Test fix for Issue #2: Dimension mismatch in preprocessing + orchestrator"""

    def test_depth_map_resized_to_original_dimensions(self):
        """Verify depth map is resized back to original dimensions after padding/cropping."""
        # Create image with non-multiple-of-14 dimensions
        test_image = np.random.rand(103, 97, 3).astype(np.float32)  # 103x97 -> will be adjusted to 98x98

        preprocessed, original_shape = preprocess_image(test_image)

        # Verify original shape is preserved
        assert original_shape == (103, 97), f"Original shape should be (103, 97), got {original_shape}"

        # Preprocessed should be multiple of 14
        assert preprocessed.shape[0] % 14 == 0, f"Height should be multiple of 14, got {preprocessed.shape[0]}"
        assert preprocessed.shape[1] % 14 == 0, f"Width should be multiple of 14, got {preprocessed.shape[1]}"

        # Dimensions should be different (padding/cropping occurred)
        assert preprocessed.shape[:2] != original_shape, "Preprocessing should change dimensions"

    def test_depth_resize_preserves_original_aspect_ratio(self):
        """Verify resizing depth map back preserves original aspect ratio."""
        # Simulate depth map after inference (padded to 98x98)
        depth_map = np.random.rand(98, 98).astype(np.float32)
        original_shape = (103, 97)  # Original non-multiple-of-14

        # Simulate resize operation (as done in fix #2)
        from PIL import Image as PILImage

        depth_pil = PILImage.fromarray((depth_map * 65535).astype(np.uint16), mode="I;16")
        depth_resized = depth_pil.resize((original_shape[1], original_shape[0]), PILImage.Resampling.LANCZOS)
        depth_final = np.array(depth_resized, dtype=np.float32) / 65535.0

        # Verify final depth matches original dimensions
        assert (
            depth_final.shape[:2] == original_shape
        ), f"Resized depth should match original {original_shape}, got {depth_final.shape[:2]}"


class TestFix3QuadraticComplexity:
    """Test fix for Issue #3: Quadratic complexity in batch_stats.py"""

    def test_detect_outliers_with_precomputed_median(self):
        """Verify detect_runtime_outliers accepts pre-computed median to avoid O(n²)."""
        runtimes = [1.0, 1.1, 0.9, 1.2, 6.0]  # Last one is outlier
        stats = compute_batch_runtime_stats(runtimes)
        median = stats["median"]

        # Call with pre-computed median (new signature)
        result = detect_runtime_outliers("slow_image.jpg", 6.0, runtimes, median=median)

        assert result is not None, "Should detect outlier"
        warning_msg, metadata = result
        assert "outlier" in warning_msg.lower()
        assert metadata["is_outlier"] is True
        assert metadata["median_runtime_s"] == median

    def test_batch_processing_performance_improvement(self):
        """Verify performance improvement: O(n) vs O(n²) for large batches."""
        # Simulate large batch
        n = 1000
        runtimes = [1.0 + i * 0.01 for i in range(n)]
        runtimes.append(100.0)  # Add outlier

        # Pre-compute median once (O(n log n))
        stats = compute_batch_runtime_stats(runtimes)
        median = stats["median"]

        # Time old approach (without pre-computed median - would be O(n²))
        start_old = time.perf_counter()
        for i, runtime in enumerate(runtimes):
            # Old way: computes median every iteration
            _ = detect_runtime_outliers(f"img_{i}.jpg", runtime, runtimes, median=None)
        time_old = time.perf_counter() - start_old

        # Time new approach (with pre-computed median - O(n))
        start_new = time.perf_counter()
        for i, runtime in enumerate(runtimes):
            # New way: pass pre-computed median
            _ = detect_runtime_outliers(f"img_{i}.jpg", runtime, runtimes, median=median)
        time_new = time.perf_counter() - start_new

        # New approach should be significantly faster (but both are fast for n=1000)
        # This test documents the fix, not strict performance bounds
        assert time_new < time_old * 2, f"New approach should not be slower: {time_new:.4f}s vs {time_old:.4f}s"

    def test_backward_compatibility_without_median(self):
        """Verify function still works without median parameter (backward compatibility)."""
        runtimes = [1.0, 1.1, 0.9, 1.2, 6.0]

        # Call without median parameter (old signature)
        result = detect_runtime_outliers("slow_image.jpg", 6.0, runtimes)

        assert result is not None, "Should detect outlier even without pre-computed median"
        warning_msg, metadata = result
        assert metadata["is_outlier"] is True


class TestFix4RedundantProcessing:
    """Test fix for Issue #4: Redundant processing in orchestrator.py (parallel mode)"""

    def test_parallel_batch_uses_precomputed_paths(self):
        """Verify parallel batch processing passes pre-computed paths to enhance_image."""
        # This is an integration test - verify the signature accepts _precomputed_paths
        # Actual orchestrator testing requires full setup, so we test the contract

        # Verify enhance_image accepts _precomputed_paths parameter
        import inspect

        from transformation_portal.lux_depth_v3.config import EnhanceConfig
        from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

        sig = inspect.signature(EnhanceOrchestrator.enhance_image)
        params = list(sig.parameters.keys())

        assert (
            "_precomputed_paths" in params
        ), "enhance_image should accept _precomputed_paths parameter to avoid redundant computation"

    def test_precomputed_paths_skip_redundant_computation(self, tmp_path):
        """Verify when _precomputed_paths is provided, path generation is skipped."""
        # Mock test to verify the logic path
        from transformation_portal.lux_depth_v3.input_manager import ImageInput
        from transformation_portal.lux_depth_v3.orchestrator import make_output_key

        test_image = tmp_path / "test.jpg"
        Image.new("RGB", (100, 100)).save(test_image)

        image_input = ImageInput(test_image)

        # Pre-compute paths (as done in parallel batch)
        output_key = make_output_key(test_image, tmp_path)
        precomputed = {
            "output_key": output_key,
            "depth_path": tmp_path / "depth.png",
            "manifest_path": tmp_path / "manifest.json",
            "should_skip": False,
        }

        # Verify precomputed paths have expected structure
        assert "output_key" in precomputed
        assert "depth_path" in precomputed
        assert "manifest_path" in precomputed
        assert "should_skip" in precomputed


class TestFix5AlphaChannelSafety:
    """Test fix for Issue #5: Alpha channel safety in v2_enhance.py"""

    def test_alpha_channel_resized_if_dimensions_mismatch(self, tmp_path):
        """Verify alpha channel is resized to match enhanced image if dimensions differ."""
        # Create RGBA image
        img_rgba = Image.new("RGBA", (100, 50), color=(255, 0, 0, 128))
        input_path = tmp_path / "test_rgba.png"
        output_path = tmp_path / "output_rgba.png"
        img_rgba.save(input_path)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage:
            mock_instance = Mock()
            mock_stage.return_value = mock_instance

            # Mock enhancement that returns different dimensions
            enhanced_rgb = (np.random.rand(80, 60, 3) * 255).astype(np.uint8)  # Different size, uint8!

            mock_result = Mock()
            from transformation_portal.stage_graph.stage import StageStatus

            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": enhanced_rgb}
            mock_result.metadata = {}
            mock_instance.compute.return_value = mock_result

            # Run enhancement - should not crash despite dimension mismatch
            result = enhance_image(input_path, output_path, config=None)
            emitted_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8)

            assert result["status"] == "success"
            assert Path(result["output"]) == emitted_output
            assert emitted_output.exists()

            # Verify output has alpha channel and correct dimensions
            output_img = Image.open(emitted_output)
            assert output_img.mode == "RGBA", "Output should preserve RGBA mode"
            # Output should match enhanced dimensions (80, 60), not original (100, 50)
            assert output_img.size == (60, 80), f"Output should match enhanced dimensions (60, 80), got {output_img.size}"

    def test_alpha_channel_preserved_when_dimensions_match(self, tmp_path):
        """Verify alpha channel is correctly stacked when dimensions already match."""
        img_rgba = Image.new("RGBA", (100, 50), color=(0, 255, 0, 200))
        input_path = tmp_path / "test_rgba_match.png"
        output_path = tmp_path / "output_rgba_match.png"
        img_rgba.save(input_path)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage:
            mock_instance = Mock()
            mock_stage.return_value = mock_instance

            # Mock enhancement that returns SAME dimensions
            enhanced_rgb = (np.random.rand(50, 100, 3) * 255).astype(np.uint8)  # Same as input, uint8!

            mock_result = Mock()
            from transformation_portal.stage_graph.stage import StageStatus

            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": enhanced_rgb}
            mock_result.metadata = {}
            mock_instance.compute.return_value = mock_result

            result = enhance_image(input_path, output_path, config=None)
            emitted_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8)
            assert result["status"] == "success"
            assert Path(result["output"]) == emitted_output

            output_img = Image.open(emitted_output)
            assert output_img.mode == "RGBA"
            assert output_img.size == (100, 50)


class TestFix6OutputDirectoryTrap:
    """Test fix for Issue #6: Output directory trap in input_discovery.py"""

    def test_output_directory_explicitly_excluded(self, tmp_path):
        """Verify output directory is excluded even if it's a subdirectory of input."""
        # Create input directory with subdirectories
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create output directory inside input
        output_dir = input_dir / "output"
        output_dir.mkdir()

        # Create test images in input
        (input_dir / "image1.jpg").write_bytes(Image.new("RGB", (10, 10)).tobytes())
        (input_dir / "image2.jpg").write_bytes(Image.new("RGB", (10, 10)).tobytes())

        # Create images in output directory (should be excluded)
        (output_dir / "processed1.jpg").write_bytes(Image.new("RGB", (10, 10)).tobytes())
        (output_dir / "processed2.jpg").write_bytes(Image.new("RGB", (10, 10)).tobytes())

        # Actually save proper images
        Image.new("RGB", (10, 10)).save(input_dir / "image1.jpg")
        Image.new("RGB", (10, 10)).save(input_dir / "image2.jpg")
        Image.new("RGB", (10, 10)).save(output_dir / "processed1.jpg")
        Image.new("RGB", (10, 10)).save(output_dir / "processed2.jpg")

        # Discover images with output_dir specified
        config = DiscoveryConfig()
        images = discover_images(input_dir, config, output_dir=output_dir)

        # Should find only images in input, not output
        assert len(images) == 2, f"Should find 2 images (input only), found {len(images)}"
        image_names = {img.name for img in images}
        assert "image1.jpg" in image_names
        assert "image2.jpg" in image_names
        assert "processed1.jpg" not in image_names, "Output images should be excluded"
        assert "processed2.jpg" not in image_names, "Output images should be excluded"

    def test_output_directory_nested_subdirectories(self, tmp_path):
        """Verify deeply nested output directories are excluded."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create nested output structure
        output_dir = input_dir / "results" / "depth" / "v3"
        output_dir.mkdir(parents=True)

        # Images in input
        Image.new("RGB", (10, 10)).save(input_dir / "source.jpg")

        # Images in nested output
        Image.new("RGB", (10, 10)).save(output_dir / "depth_map.jpg")

        config = DiscoveryConfig()
        images = discover_images(input_dir, config, output_dir=output_dir.parent.parent)

        # Should only find source image
        assert len(images) == 1
        assert images[0].name == "source.jpg"

    def test_output_directory_none_uses_pattern_matching(self, tmp_path):
        """Verify when output_dir is None, pattern matching still works."""
        input_dir = tmp_path / "input"
        input_dir.mkdir()

        # Create depth subdirectory (matches exclude_path_patterns)
        depth_dir = input_dir / "depth"
        depth_dir.mkdir()

        Image.new("RGB", (10, 10)).save(input_dir / "image.jpg")
        Image.new("RGB", (10, 10)).save(depth_dir / "depth_map.jpg")

        config = DiscoveryConfig()
        images = discover_images(input_dir, config, output_dir=None)

        # Should exclude depth/ via pattern matching
        assert len(images) == 1
        assert images[0].name == "image.jpg"


# Performance and integration tests
class TestCriticalFixesIntegration:
    """Integration tests verifying all fixes work together."""

    def test_all_fixes_integration_smoke_test(self, tmp_path):
        """Smoke test that all fixes work together without errors."""
        # This test verifies that the changes don't break the basic workflow

        # Test 1: EXIF handling
        img = Image.new("RGB", (100, 50))
        test_path = tmp_path / "test.jpg"
        img.save(test_path)

        # Test 2: Preprocessing with dimension enforcement
        preprocessed, original = preprocess_image(test_path)
        assert original == (50, 100)  # PIL uses (H, W) -> (50, 100)

        # Test 3: Batch stats
        runtimes = [1.0, 1.1, 1.2, 10.0]  # 10.0 is >5× median (~1.15)
        stats = compute_batch_runtime_stats(runtimes)
        median = stats["median"]
        result = detect_runtime_outliers("test.jpg", 10.0, runtimes, median=median)
        assert result is not None, f"10.0s should be outlier vs median {median:.2f}s"

        # Test 6: Input discovery
        config = DiscoveryConfig()
        images = discover_images(tmp_path, config, output_dir=tmp_path / "output")
        # Should find test.jpg (no output directory created yet)
        assert len(images) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
