"""
Quality Firewall tests for V2 Enhancement.

Tests the 16-bit preservation guarantees and Quality Firewall enforcement
to prevent silent 8-bit degradation.

Note: These tests use simplified mocking to avoid heavy dependencies.
For integration tests with real tifffile, see tests/integration/.
"""

from pathlib import Path
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = [pytest.mark.unit]

from transformation_portal.lux_depth_v3.v2_enhance import (
    V2EnhancementError,
    enhance_image,
    load_image_preserve_bit_depth,
    resolve_v2_emitted_artifact_path,
)
from transformation_portal.stage_graph.stage import StageStatus


@pytest.fixture
def temp_16bit_tiff(tmp_path):
    """Create a temporary 16-bit TIFF image for testing."""
    # Create 16-bit test image
    image_array = np.random.randint(0, 65536, size=(100, 100, 3), dtype=np.uint16)

    # Save with PIL in a way that preserves 16-bit tag
    tiff_path = tmp_path / "test_16bit.tif"

    # Use tifffile if available for proper 16-bit saving
    try:
        import tifffile

        tifffile.imwrite(tiff_path, image_array, photometric="rgb")
    except ImportError:
        # Fallback to PIL grayscale 16-bit
        pil_img = Image.fromarray(image_array[:, :, 0], mode="I;16")
        pil_img.save(tiff_path, format="TIFF")

    return tiff_path


@pytest.fixture
def temp_8bit_tiff(tmp_path):
    """Create a temporary 8-bit TIFF image for testing."""
    # Create 8-bit test image
    image_array = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)

    tiff_path = tmp_path / "test_8bit.tif"
    pil_img = Image.fromarray(image_array, mode="RGB")
    pil_img.save(tiff_path, format="TIFF")

    return tiff_path


class TestQualityFirewallLoad:
    """Test Quality Firewall enforcement on load path."""

    def test_tifffile_load_failure_blocks_when_firewall_active(self, temp_16bit_tiff, tmp_path):
        """tifffile load failure raises when input is 16-bit and --allow-8bit not set."""
        pytest.importorskip("tifffile")
        # Patch tifffile where it's imported (locally in the function)
        with patch("tifffile.imread") as mock_imread:
            mock_imread.side_effect = RuntimeError("Mock tifffile load failure")

            # Should raise V2EnhancementError due to Quality Firewall
            with pytest.raises(V2EnhancementError) as exc_info:
                load_image_preserve_bit_depth(temp_16bit_tiff, allow_8bit_output=False)

            # Check error message mentions Quality Firewall
            assert "Quality Firewall" in str(exc_info.value)
            assert "tifffile" in str(exc_info.value).lower()

    def test_tifffile_load_failure_allowed_with_flag(self, temp_16bit_tiff, tmp_path):
        """tifffile load failure falls back to PIL when --allow-8bit is set."""
        pytest.importorskip("tifffile")
        # Patch tifffile where it's imported
        with patch("tifffile.imread") as mock_imread:
            mock_imread.side_effect = RuntimeError("Mock tifffile load failure")

            # Should NOT raise - falls back to PIL
            try:
                image, bits, metadata = load_image_preserve_bit_depth(temp_16bit_tiff, allow_8bit_output=True)
                # Should have loaded via PIL fallback (8-bit)
                assert bits == 8
                assert image.dtype == np.uint8
            except V2EnhancementError:
                pytest.fail("Should not raise when allow_8bit_output=True")

    def test_explicit_16_bit_output_overrides_legacy_downgrade_permission(self, temp_16bit_tiff, tmp_path):
        """Canonical 16-bit output must fail closed even with the legacy flag."""
        pytest.importorskip("tifffile")
        output_path = tmp_path / "output.tif"

        with patch("tifffile.imread", side_effect=RuntimeError("Mock tifffile load failure")):
            with pytest.raises(V2EnhancementError, match="blocked by Quality Firewall"):
                enhance_image(
                    temp_16bit_tiff,
                    output_path,
                    allow_8bit_output=True,
                    output_bit_depth=16,
                )

        assert not output_path.exists()


class TestQualityFirewallMetadata:
    """Test metadata consistency and reporting."""

    def test_metadata_tracks_firewall_state_active(self, temp_8bit_tiff, tmp_path):
        """Metadata correctly reports Quality Firewall state when active."""
        output_path = tmp_path / "output" / "enhanced.tif"

        # For 16-bit testing, we'd need actual 16-bit input or sophisticated mocking
        # This test verifies metadata structure is correct
        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as MockStage:
            # Mock enhancement stage
            mock_stage_instance = MockStage.return_value
            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED

            enhanced_8bit = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
            mock_result.artifacts = {"enhanced_image": enhanced_8bit, "enhancement_metadata": {}}
            mock_result.metadata = {}
            mock_stage_instance.compute.return_value = mock_result

            result = enhance_image(temp_8bit_tiff, output_path, allow_8bit_output=False)

            # Check metadata structure
            assert "bit_depth" in result
            assert "input_bits_per_sample" in result["bit_depth"]
            assert "output_bits_per_sample" in result["bit_depth"]
            assert "quality_firewall_active" in result["bit_depth"]
            assert "bit_depth_preserved" in result["bit_depth"]
            assert "downgrade_allowed" in result["bit_depth"]


class TestDtypeConsistency:
    """Test dtype consistency throughout pipeline."""

    def test_target_dtype_propagates_to_enhancement_stage(self, temp_8bit_tiff, tmp_path):
        """Verify that target_dtype is passed correctly to EnhancementStage."""
        output_path = tmp_path / "output" / "enhanced.tif"

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as MockStage:
            mock_stage_instance = MockStage.return_value
            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED

            enhanced_8bit = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
            mock_result.artifacts = {"enhanced_image": enhanced_8bit, "enhancement_metadata": {}}
            mock_result.metadata = {}
            mock_stage_instance.compute.return_value = mock_result

            # Run enhancement
            result = enhance_image(temp_8bit_tiff, output_path, allow_8bit_output=False)

            # Verify EnhancementStage was initialized with correct dtype
            call_kwargs = MockStage.call_args[1]
            # For 8-bit input, target_dtype should be np.uint8
            assert call_kwargs["output_dtype"] == np.uint8


class TestPassthroughMode:
    """Test that passthrough mode (preset='none') bypasses enhancement correctly."""

    def test_passthrough_preserves_file_exactly(self, temp_8bit_tiff, tmp_path):
        """Preset 'none' should copy file without any processing."""
        from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig

        output_path = tmp_path / "output" / "enhanced.tif"

        # Create config with preset='none'
        config = V2EnhancementConfig(preset="none", enhancement_strength=0.0, clarity_strength=0.0, material_strength=0.0)

        result = enhance_image(temp_8bit_tiff, output_path, config=config, allow_8bit_output=False)

        # Verify passthrough
        assert result["status"] == "passthrough"
        assert output_path.exists()


# Minimal integration-style test to verify the code actually works
class TestRealProcessing:
    """Minimal tests with real (non-mocked) components where feasible."""

    def test_8bit_processing_works_end_to_end(self, temp_8bit_tiff, tmp_path):
        """Verify that 8-bit processing actually works (with mocked stage)."""
        output_path = tmp_path / "output" / "enhanced.tif"

        # Only mock the enhancement stage (heavy computation)
        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as MockStage:
            mock_stage_instance = MockStage.return_value
            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED

            # Load the real input to get correct dimensions
            from PIL import Image

            real_img = Image.open(temp_8bit_tiff)
            real_array = np.array(real_img)

            # Return array with same shape
            mock_result.artifacts = {"enhanced_image": real_array, "enhancement_metadata": {}}
            mock_result.metadata = {}
            mock_stage_instance.compute.return_value = mock_result

            # Run enhancement
            result = enhance_image(temp_8bit_tiff, output_path, allow_8bit_output=False)
            emitted_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8, materials_enabled=False)

            # Verify success
            assert result["status"] == "success"
            assert Path(result["output"]) == emitted_output
            assert emitted_output.exists()
            assert result["bit_depth"]["input_bits_per_sample"] == 8
            assert result["bit_depth"]["output_bits_per_sample"] == 8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
