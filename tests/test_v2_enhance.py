"""Unit tests for V2 Enhancement Implementation.

Tests the main V2 enhancement logic, depth map loading, and integration
with EnhancementStage.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image

from transformation_portal.lux_depth_v3.v2_enhance import V2EnhancementError, enhance_image, find_depth_map, load_depth_map
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig
from transformation_portal.stage_graph.stage import StageStatus


class TestFindDepthMap:
    """Test depth map discovery logic."""

    def test_find_depth_map_standard_naming(self, tmp_path):
        """Test finding depth map with standard naming convention."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        # Create depth map with standard naming
        depth_path = depth_dir / "test_image_depth.png"
        depth_path.touch()

        found = find_depth_map(depth_dir, "test_image")
        assert found == depth_path

    def test_find_depth_map_u16_naming(self, tmp_path):
        """Test finding depth map with _u16 suffix."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        depth_path = depth_dir / "test_image_depth_u16.png"
        depth_path.touch()

        found = find_depth_map(depth_dir, "test_image")
        assert found == depth_path

    def test_find_depth_map_simple_naming(self, tmp_path):
        """Test finding depth map with simple naming (just stem.png)."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        depth_path = depth_dir / "test_image.png"
        depth_path.touch()

        found = find_depth_map(depth_dir, "test_image")
        assert found == depth_path

    def test_find_depth_map_not_found(self, tmp_path):
        """Test behavior when depth map not found."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        found = find_depth_map(depth_dir, "nonexistent")
        assert found is None

    def test_find_depth_map_no_directory(self):
        """Test behavior when depth_dir is None or doesn't exist."""
        assert find_depth_map(None, "test") is None
        assert find_depth_map(Path("/nonexistent"), "test") is None


class TestLoadDepthMap:
    """Test depth map loading and normalization."""

    def test_load_depth_map_uint8(self, tmp_path):
        """Test loading uint8 depth map (0-255)."""
        depth_path = tmp_path / "depth.png"

        # Create uint8 depth map
        depth_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        Image.fromarray(depth_data, mode="L").save(depth_path)

        loaded = load_depth_map(depth_path)

        assert loaded.shape == (100, 100)
        assert loaded.dtype == np.float32
        assert 0.0 <= loaded.min() <= loaded.max() <= 1.0

    def test_load_depth_map_uint16(self, tmp_path):
        """Test loading uint16 depth map (0-65535)."""
        depth_path = tmp_path / "depth.png"

        # Create uint16 depth map and save as 16-bit grayscale
        depth_data = np.random.randint(0, 65536, (100, 100), dtype=np.uint16)
        # PIL needs mode 'I;16' for 16-bit images
        Image.fromarray(depth_data).save(depth_path, format="PNG", bits=16)

        loaded = load_depth_map(depth_path)

        assert loaded.shape == (100, 100)
        assert loaded.dtype == np.float32
        assert 0.0 <= loaded.min() <= loaded.max() <= 1.0

    def test_load_depth_map_already_normalized(self, tmp_path):
        """Test loading depth map that's already normalized (0-1)."""
        depth_path = tmp_path / "depth.png"

        # Create normalized depth map
        depth_data = np.random.rand(100, 100).astype(np.float32)
        Image.fromarray((depth_data * 255).astype(np.uint8), mode="L").save(depth_path)

        loaded = load_depth_map(depth_path)

        assert loaded.shape == (100, 100)
        assert loaded.dtype == np.float32
        assert 0.0 <= loaded.min() <= loaded.max() <= 1.0

    def test_load_depth_map_rgb_converts_to_grayscale(self, tmp_path):
        """Test that RGB depth maps are converted to grayscale."""
        depth_path = tmp_path / "depth.png"

        # Create RGB image
        rgb_data = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(rgb_data, mode="RGB").save(depth_path)

        loaded = load_depth_map(depth_path)

        assert loaded.shape == (100, 100)  # Should be 2D grayscale
        assert loaded.dtype == np.float32

    def test_load_depth_map_nonexistent_file(self, tmp_path):
        """Test error handling for nonexistent file."""
        depth_path = tmp_path / "nonexistent.png"

        with pytest.raises(V2EnhancementError, match="Failed to load depth map"):
            load_depth_map(depth_path)


class TestEnhanceImage:
    """Test main enhance_image function."""

    def test_enhance_image_basic(self, tmp_path):
        """Test basic enhancement without depth map."""
        # Create test input image
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        output_path = tmp_path / "output.png"

        # Mock EnhancementStage to avoid actual processing
        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            # Mock successful enhancement
            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {
                "enhanced_image": test_image,
                "enhancement_metadata": {"test": "metadata"},
            }
            mock_result.metadata = {"processing_ms": 100}
            mock_stage.compute.return_value = mock_result

            # Run enhancement
            report = enhance_image(input_path, output_path)

            # Verify report
            assert report["status"] == "success"
            assert report["input"] == str(input_path)
            assert report["output"] == str(output_path)
            assert report["preset"] == "default"
            assert "runtime_s" in report
            assert output_path.exists()

    def test_enhance_image_with_depth_map(self, tmp_path):
        """Test enhancement with depth map."""
        # Create test input image
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        # Create depth map
        depth_path = tmp_path / "depth.png"
        depth_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        Image.fromarray(depth_data, mode="L").save(depth_path)

        output_path = tmp_path / "output.png"

        # Mock EnhancementStage
        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            # Run enhancement
            report = enhance_image(input_path, output_path, depth_map_path=depth_path)

            # Verify depth map was passed to stage
            assert report["depth_map"] == str(depth_path)

            # Check that compute was called with depth_map in context
            call_args = mock_stage.compute.call_args
            context = call_args[0][0]
            assert context.get_artifact("depth_map") is not None

    def test_enhance_image_with_custom_config(self, tmp_path):
        """Test enhancement with custom configuration."""
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        output_path = tmp_path / "output.png"

        config = V2EnhancementConfig(
            preset="luxury_estate", enhancement_strength=0.9, clarity_strength=0.8, material_strength=0.7
        )

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            report = enhance_image(input_path, output_path, config=config)

            # Verify config was used
            assert report["preset"] == "luxury_estate"
            assert report["config"]["enhancement_strength"] == 0.9

            # Verify EnhancementStage was initialized with config values
            mock_stage_cls.assert_called_once()
            call_kwargs = mock_stage_cls.call_args[1]
            assert call_kwargs["enhancement_strength"] == 0.9
            assert call_kwargs["clarity_strength"] == 0.8
            assert call_kwargs["material_strength"] == 0.7

    def test_enhance_image_none_preset_passthrough(self, tmp_path):
        """Test that 'none' preset skips enhancement (passthrough)."""
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        output_path = tmp_path / "output.png"

        config = V2EnhancementConfig.from_preset("none")

        # Should not call EnhancementStage
        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            report = enhance_image(input_path, output_path, config=config)

            # Verify passthrough
            assert report["status"] == "passthrough"
            assert report["preset"] == "none"
            assert "enhancement skipped" in report["message"]

            # Verify EnhancementStage was NOT called
            mock_stage_cls.assert_not_called()

            # Verify output file exists (copied from input)
            assert output_path.exists()

    def test_enhance_image_input_not_found(self, tmp_path):
        """Test error handling for nonexistent input."""
        input_path = tmp_path / "nonexistent.png"
        output_path = tmp_path / "output.png"

        with pytest.raises(FileNotFoundError, match="Input image not found"):
            enhance_image(input_path, output_path)

    def test_enhance_image_stage_failure(self, tmp_path):
        """Test error handling when EnhancementStage fails."""
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        output_path = tmp_path / "output.png"

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            # Mock stage failure
            mock_result = Mock()
            mock_result.status = StageStatus.FAILED
            mock_result.error = "Test error"
            mock_stage.compute.return_value = mock_result

            with pytest.raises(V2EnhancementError, match="Enhancement failed: Test error"):
                enhance_image(input_path, output_path)

    def test_enhance_image_creates_output_directory(self, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        # Output path in non-existent directory
        output_path = tmp_path / "subdir" / "nested" / "output.png"

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            report = enhance_image(input_path, output_path)

            # Verify output directory was created
            assert output_path.parent.exists()
            assert output_path.exists()

    def test_enhance_image_device_selection(self, tmp_path):
        """Test that device parameter is passed to stage context."""
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        output_path = tmp_path / "output.png"

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            # Test with different devices
            for device in ["cpu", "cuda", "mps"]:
                report = enhance_image(input_path, output_path, device=device)

                # Verify device was passed to context
                call_args = mock_stage.compute.call_args
                context = call_args[0][0]
                assert context.device == device
