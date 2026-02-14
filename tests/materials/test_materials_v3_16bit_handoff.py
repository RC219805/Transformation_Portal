"""Test 16-bit output path for Materials V3 → V2 handoff.

Validates that:
1. Materials V3 outputs 16-bit TIFF when emit flags are enabled
2. Materials V3 outputs 8-bit PNG when emit flags are disabled (Golden Path)
3. Bit depth tracking in manifest reflects actual artifacts created
4. File format and dtype verification
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator


@pytest.fixture
def mock_depth_backend():
    """Mock depth backend to avoid ML dependencies in integration tests."""
    with patch("transformation_portal.lux_depth_v3.orchestrator.DepthBackendRegistry"):
        yield


@pytest.fixture
def mock_da3_available():
    """Mock DA3Backend.ensure_available() to succeed in offline CI."""
    with patch("transformation_portal.depth.backends.da3.DA3Backend.ensure_available"):
        yield


@pytest.fixture
def test_image():
    """Create a simple test image."""
    from PIL import Image

    # Create gradient image
    img_array = np.zeros((256, 256, 3), dtype=np.uint8)
    for i in range(256):
        img_array[i, :, :] = int(255 * i / 256)

    img = Image.fromarray(img_array)
    return img


@pytest.mark.ml
def test_16bit_tiff_handoff_when_emit_flags_enabled(tmp_path, mock_depth_backend, mock_da3_available, test_image):
    """Test that Materials V3 outputs 16-bit TIFF when emit flags are enabled."""
    # Create input directory and save test image
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    test_img_path = input_dir / "test.png"
    test_image.save(test_img_path)

    # Configure with 16-bit flags enabled
    config = EnhanceConfig(
        enable_materials_v3=True,
        apply_pixel_ops=True,
        depth_device="cpu",
        enable_v2=False,  # Disable V2 to preserve handoff file
        depth_backend="synthetic",
        emit_master16=True,
        emit_upscaled16=True,
    )

    output_dir = tmp_path / "output_16bit"
    orchestrator = EnhanceOrchestrator(config, output_dir)

    # Mock depth backend compute method
    mock_result = MagicMock()
    mock_result.depth_map = np.ones((256, 256), dtype=np.float32) * 0.5
    mock_result.original_image = np.array(test_image)

    with patch.object(orchestrator.depth_backend, "compute", return_value=mock_result):
        # Mock Materials V3 to return an enhanced image
        with patch.object(orchestrator.materials_v3_engine, "process") as mock_process:
            # Create a modified image to simulate enhancement
            enhanced_image = np.ones((256, 256, 3), dtype=np.float32) * 0.6
            mock_process.return_value = {
                "enhanced_image": enhanced_image,
                "materials_v3_response_plan": {"per_class": {}},
                "materials_v3_pixel_ops": {"enabled": True, "applied": [], "blocked": [], "timing_ms": 10},
                "materials_v3_metadata": {"version": "3.1"},
                "material_masks": {},
            }

            # Process the image
            orchestrator.enhance_single_image(test_img_path)

    # Verify 16-bit TIFF handoff file was created
    temp_dir = output_dir / "temp"
    handoff_files = list(temp_dir.glob("*_materials_v3_enhanced.tif"))
    assert len(handoff_files) == 1, "Expected one 16-bit TIFF handoff file"

    handoff_path = handoff_files[0]
    assert handoff_path.suffix == ".tif", "Expected .tif extension"

    # Verify TIFF format and bit depth
    try:
        import tifffile

        img_array = tifffile.imread(handoff_path)
        assert img_array.dtype == np.uint16, f"Expected uint16, got {img_array.dtype}"
        assert img_array.shape == (256, 256, 3), f"Expected (256, 256, 3), got {img_array.shape}"
    except ImportError:
        pytest.skip("tifffile not available")

    # Verify manifest bit depth tracking
    manifest_path = output_dir / "test_manifest.json"
    if manifest_path.exists():
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest

        manifest = CombinedManifest.load(manifest_path)
        if manifest.materials_v3:
            assert manifest.materials_v3.output_bit_depth == 16, "Expected 16-bit output in manifest"
        if manifest.v2:
            assert manifest.v2.input_bit_depth == 16, "Expected 16-bit input to V2 in manifest"


@pytest.mark.ml
def test_8bit_png_handoff_when_emit_flags_disabled(tmp_path, mock_depth_backend, mock_da3_available, test_image):
    """Test that Materials V3 outputs 8-bit PNG when emit flags are disabled (Golden Path)."""
    # Create input directory and save test image
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    test_img_path = input_dir / "test.png"
    test_image.save(test_img_path)

    # Configure without 16-bit flags (Golden Path)
    config = EnhanceConfig(
        enable_materials_v3=True,
        apply_pixel_ops=True,
        depth_device="cpu",
        enable_v2=False,  # Disable V2 to preserve handoff file
        depth_backend="synthetic",
        emit_master16=False,  # Golden Path: 8-bit
        emit_upscaled16=False,
    )

    output_dir = tmp_path / "output_8bit"
    orchestrator = EnhanceOrchestrator(config, output_dir)

    # Mock depth backend compute method
    mock_result = MagicMock()
    mock_result.depth_map = np.ones((256, 256), dtype=np.float32) * 0.5
    mock_result.original_image = np.array(test_image)

    with patch.object(orchestrator.depth_backend, "compute", return_value=mock_result):
        # Mock Materials V3 to return an enhanced image
        with patch.object(orchestrator.materials_v3_engine, "process") as mock_process:
            # Create a modified image to simulate enhancement
            enhanced_image = np.ones((256, 256, 3), dtype=np.float32) * 0.6
            mock_process.return_value = {
                "enhanced_image": enhanced_image,
                "materials_v3_response_plan": {"per_class": {}},
                "materials_v3_pixel_ops": {"enabled": True, "applied": [], "blocked": [], "timing_ms": 10},
                "materials_v3_metadata": {"version": "3.1"},
                "material_masks": {},
            }

            # Process the image
            orchestrator.enhance_single_image(test_img_path)

    # Verify 8-bit PNG handoff file was created
    temp_dir = output_dir / "temp"
    handoff_files = list(temp_dir.glob("*_materials_v3_enhanced.png"))
    assert len(handoff_files) == 1, "Expected one 8-bit PNG handoff file"

    handoff_path = handoff_files[0]
    assert handoff_path.suffix == ".png", "Expected .png extension"

    # Verify PNG format and bit depth
    from PIL import Image

    img = Image.open(handoff_path)
    img_array = np.array(img)
    assert img_array.dtype == np.uint8, f"Expected uint8, got {img_array.dtype}"
    assert img_array.shape == (256, 256, 3), f"Expected (256, 256, 3), got {img_array.shape}"

    # Verify manifest bit depth tracking
    manifest_path = output_dir / "test_manifest.json"
    if manifest_path.exists():
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest

        manifest = CombinedManifest.load(manifest_path)
        if manifest.materials_v3:
            assert manifest.materials_v3.output_bit_depth == 8, "Expected 8-bit output in manifest"
        if manifest.v2:
            assert manifest.v2.input_bit_depth == 8, "Expected 8-bit input to V2 in manifest"


@pytest.mark.ml
def test_manifest_bit_depth_accuracy_when_no_enhanced_image(tmp_path, mock_depth_backend, mock_da3_available, test_image):
    """Test that manifest bit depth is 8-bit when Materials V3 runs but produces no enhanced image."""
    # Create input directory and save test image
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    test_img_path = input_dir / "test.png"
    test_image.save(test_img_path)

    # Configure with 16-bit flags enabled
    config = EnhanceConfig(
        enable_materials_v3=True,
        apply_pixel_ops=True,
        depth_device="cpu",
        enable_v2=False,
        depth_backend="synthetic",
        emit_master16=True,  # Flags enabled, but Materials V3 returns no enhanced image
        emit_upscaled16=True,
    )

    output_dir = tmp_path / "output_no_enhance"
    orchestrator = EnhanceOrchestrator(config, output_dir)

    # Mock depth backend compute method
    mock_result = MagicMock()
    mock_result.depth_map = np.ones((256, 256), dtype=np.float32) * 0.5
    mock_result.original_image = np.array(test_image)

    with patch.object(orchestrator.depth_backend, "compute", return_value=mock_result):
        # Mock Materials V3 to return NO enhanced image (enhanced_image=None)
        with patch.object(orchestrator.materials_v3_engine, "process") as mock_process:
            mock_process.return_value = {
                "enhanced_image": None,  # No enhancement
                "materials_v3_response_plan": {"per_class": {}},
                "materials_v3_pixel_ops": {"enabled": True, "applied": [], "blocked": [], "timing_ms": 10},
                "materials_v3_metadata": {"version": "3.1"},
                "material_masks": {},
            }

            # Process the image
            orchestrator.enhance_single_image(test_img_path)

    # Verify no handoff file was created (no enhanced image)
    temp_dir = output_dir / "temp"
    if temp_dir.exists():
        tiff_files = list(temp_dir.glob("*_materials_v3_enhanced.tif"))
        png_files = list(temp_dir.glob("*_materials_v3_enhanced.png"))
        assert len(tiff_files) == 0, "Expected no TIFF handoff file when enhanced_image is None"
        assert len(png_files) == 0, "Expected no PNG handoff file when enhanced_image is None"

    # Verify manifest bit depth tracking reflects the truth (8-bit, not 16-bit)
    manifest_path = output_dir / "test_manifest.json"
    if manifest_path.exists():
        from transformation_portal.lux_depth_v3.manifest import CombinedManifest

        manifest = CombinedManifest.load(manifest_path)
        if manifest.materials_v3:
            # Should be 8-bit because no enhanced image was produced
            assert manifest.materials_v3.output_bit_depth == 8, (
                "Expected 8-bit in manifest when enhanced_image is None, not 16-bit"
            )
        if manifest.v2:
            # V2 input should also be 8-bit (original input, not 16-bit handoff)
            assert manifest.v2.input_bit_depth == 8, (
                "Expected 8-bit V2 input when no enhanced image produced, not 16-bit"
            )
