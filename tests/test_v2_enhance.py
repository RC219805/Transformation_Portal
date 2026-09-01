"""Unit tests for V2 Enhancement Implementation.

Tests the main V2 enhancement logic, depth map loading, and integration
with EnhancementStage.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from PIL import Image, ImageOps

from transformation_portal.lux_depth_v3.v2_enhance import (
    V2EnhancementError,
    _apply_exif_orientation_to_array,
    _extract_icc_profile,
    _normalize_icc_profile_payload,
    canonical_asset_stem,
    emitted_v2_suffix_for_bit_depth,
    enhance_image,
    find_depth_map,
    load_depth_map,
    resolve_v2_emitted_artifact_path,
)
from transformation_portal.lux_depth_v3.v2_presets import V2EnhancementConfig
from transformation_portal.stage_graph.stage import StageStatus

pytestmark = pytest.mark.unit


class TestFindDepthMap:
    """Test depth map discovery logic."""

    def test_canonical_asset_stem_strips_known_derived_suffixes(self):
        """Derived suffixes should normalize back to canonical source stem."""
        stem = "750Picacho_Pool_master16__tiff_eb4924f8_materials_v3_enhanced"
        assert canonical_asset_stem(stem) == "750Picacho_Pool_master16__tiff_eb4924f8"

    def test_canonical_asset_stem_preserves_dotted_stem_segments(self):
        """Dot-containing stems should remain intact when already stem-like."""
        stem = "image.v1_materials_v3_enhanced"
        assert canonical_asset_stem(stem) == "image.v1"

    def test_canonical_asset_stem_strips_extension_for_path_like_input(self):
        """Path-like filename inputs should strip extension then derived suffixes."""
        stem = "/tmp/image.v1_materials_v3_enhanced.png"
        assert canonical_asset_stem(stem) == "image.v1"

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

    def test_find_depth_map_resolves_from_derived_stem(self, tmp_path):
        """Derived V2 stems should resolve to depth sidecar written for source stem."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        depth_path = depth_dir / "750Picacho_Pool_master16__tiff_eb4924f8_depth.png"
        depth_path.touch()

        found = find_depth_map(depth_dir, "750Picacho_Pool_master16__tiff_eb4924f8_materials_v3_enhanced")
        assert found == depth_path

    def test_find_depth_map_recurses_nested_depth_outputs(self, tmp_path):
        """Depth lookup should find sidecars in nested output_key directories."""
        depth_dir = tmp_path / "depth"
        nested_dir = depth_dir / "scene_a"
        nested_dir.mkdir(parents=True)

        depth_path = nested_dir / "test_image_depth.png"
        depth_path.touch()

        found = find_depth_map(depth_dir, "test_image")
        assert found == depth_path

    def test_find_depth_map_fails_closed_on_direct_ambiguity(self, tmp_path):
        """Multiple direct candidate sidecars should not silently choose one."""
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()

        (depth_dir / "test_image_depth.png").touch()
        (depth_dir / "test_image_depth_u16.png").touch()

        with pytest.raises(V2EnhancementError) as exc_info:
            find_depth_map(depth_dir, "test_image")

        message = str(exc_info.value)
        assert "Ambiguous depth map matches" in message
        assert "test_image_depth.png" in message
        assert "test_image_depth_u16.png" in message

    def test_find_depth_map_fails_closed_on_recursive_ambiguity(self, tmp_path):
        """Multiple nested candidate sidecars should not silently choose the first sorted hit."""
        depth_dir = tmp_path / "depth"
        nested_a = depth_dir / "scene_a"
        nested_b = depth_dir / "scene_b"
        nested_a.mkdir(parents=True)
        nested_b.mkdir(parents=True)

        (nested_a / "test_image_depth.png").touch()
        (nested_b / "test_image_depth.png").touch()

        with pytest.raises(V2EnhancementError) as exc_info:
            find_depth_map(depth_dir, "test_image")

        message = str(exc_info.value)
        assert "Ambiguous depth map matches" in message
        assert "scene_a/test_image_depth.png" in message
        assert "scene_b/test_image_depth.png" in message

    def test_find_depth_map_no_directory(self):
        """Test behavior when depth_dir is None or doesn't exist."""
        assert find_depth_map(None, "test") is None
        assert find_depth_map(Path("/nonexistent"), "test") is None


class TestEmittedArtifactPath:
    """Test canonical emitted V2 artifact naming."""

    @pytest.mark.parametrize(
        ("candidate_name", "bit_depth", "expected_name"),
        [
            ("scene.DNG", 16, "scene_materials_v3_enhanced.tif"),
            ("scene.CR2", 16, "scene_materials_v3_enhanced.tif"),
            ("scene_materials_v3_enhanced.nef", 16, "scene_materials_v3_enhanced.tif"),
            ("scene_v2_enhanced.png", 8, "scene_materials_v3_enhanced.png"),
            ("scene.jpg", 8, "scene_materials_v3_enhanced.png"),
        ],
    )
    def test_resolve_v2_emitted_artifact_path_normalizes_basename_and_suffix(
        self,
        tmp_path,
        candidate_name,
        bit_depth,
        expected_name,
    ):
        candidate_path = tmp_path / candidate_name

        resolved = resolve_v2_emitted_artifact_path(candidate_path, bit_depth=bit_depth)

        assert resolved == tmp_path / expected_name
        assert resolved.suffix == emitted_v2_suffix_for_bit_depth(bit_depth)

    def test_resolve_v2_emitted_artifact_path_uses_plain_v2_name_without_materials(self, tmp_path):
        candidate_path = tmp_path / "scene_v2_enhanced.png"

        resolved = resolve_v2_emitted_artifact_path(
            candidate_path,
            bit_depth=8,
            materials_enabled=False,
        )

        assert resolved == tmp_path / "scene_v2_enhanced.png"


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
        expected_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8, materials_enabled=False)

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
            assert report["output"] == str(expected_output)
            assert report["preset"] == "default"
            assert report["depth_consumed"] is False
            assert report["artifact_contract"] == "canonical_v2_emitted_artifact"
            assert report["is_canonical_emitted_artifact"] is True
            assert report["output_naming_policy"] == "canonical_v2_emitted_artifact"
            assert report["io"]["load_backend"] == "pil"
            assert report["io"]["save_backend"] == "pil"
            assert report["io"]["metadata_preservation_mode"] == "none"
            assert report["io"]["icc_preserved"] is False
            assert report["io"]["exif_preservation_mode"] == "none"
            assert report["io"]["exif_orientation_normalized"] is False
            assert report["io"]["source_exif_orientation"] is None
            assert report["io"]["save_degraded"] is False
            assert "runtime_s" in report
            assert expected_output.exists()
            reopened = np.asarray(Image.open(expected_output))
            assert reopened.dtype == np.uint8

    def test_explicit_16_bit_output_reopens_with_uint16_samples(self, tmp_path):
        """The canonical V2 setting must control encoded bytes, not metadata only."""
        tifffile = pytest.importorskip("tifffile")
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        Image.fromarray(np.full((4, 5, 3), 128, dtype=np.uint8), mode="RGB").save(input_path)
        enhanced = np.linspace(0, 65535, 60, dtype=np.uint16).reshape((4, 5, 3))

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as stage_cls:
            result = Mock(status=StageStatus.COMPLETED, metadata={})
            result.artifacts = {"enhanced_image": enhanced}
            stage_cls.return_value.compute.return_value = result
            report = enhance_image(input_path, output_path, output_bit_depth=16)

        reopened = tifffile.imread(report["output"])
        assert Path(report["output"]).suffix == ".tif"
        assert reopened.dtype == np.uint16
        assert int(reopened.max()) > 255
        assert report["bit_depth"]["output_bits_per_sample"] == 16
        assert report["bit_depth"]["output_dtype"] == "uint16"

    def test_explicit_8_bit_output_downconverts_16_bit_input(self, tmp_path):
        """The canonical 8-bit setting explicitly authorizes and encodes a downgrade."""
        tifffile = pytest.importorskip("tifffile")
        input_path = tmp_path / "input.tif"
        caller_path = tmp_path / "output.tif"
        source = np.linspace(0, 65535, 60, dtype=np.uint16).reshape((4, 5, 3))
        tifffile.imwrite(input_path, source, photometric="rgb")
        config = V2EnhancementConfig.from_preset("none")

        report = enhance_image(
            input_path,
            caller_path,
            config=config,
            output_bit_depth=8,
        )

        emitted_path = Path(report["output"])
        with Image.open(emitted_path) as reopened_image:
            reopened = np.asarray(reopened_image)
            assert reopened_image.format == "PNG"
        assert emitted_path.suffix == ".png"
        assert reopened.dtype == np.uint8
        assert int(reopened.max()) == 255
        assert report["bit_depth"]["input_bits_per_sample"] == 16
        assert report["bit_depth"]["output_bits_per_sample"] == 8
        assert report["bit_depth"]["downgrade_allowed"] is True
        assert report["io"]["save_degraded"] is True
        assert report["io"]["save_degradation_reason"] == "explicit_output_bit_depth"

    def test_explicit_16_bit_output_fails_closed_when_tiff_write_fails(self, tmp_path):
        """A failed TIFF write must never publish an 8-bit file under a 16-bit claim."""
        pytest.importorskip("tifffile")
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        Image.fromarray(np.full((3, 4, 3), 128, dtype=np.uint8), mode="RGB").save(input_path)
        enhanced = np.full((3, 4, 3), 32768, dtype=np.uint16)
        expected_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=16, materials_enabled=False)

        with (
            patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as stage_cls,
            patch("tifffile.imwrite", side_effect=RuntimeError("simulated disk failure")),
        ):
            result = Mock(status=StageStatus.COMPLETED, metadata={})
            result.artifacts = {"enhanced_image": enhanced}
            stage_cls.return_value.compute.return_value = result
            with pytest.raises(V2EnhancementError, match="publishing an 8-bit file"):
                enhance_image(input_path, output_path, output_bit_depth=16)

        assert not expected_output.exists()

    def test_explicit_16_bit_rgba_scales_alpha_to_uint16_range(self, tmp_path):
        """Up-encoding RGBA must scale alpha with RGB instead of leaving 8-bit values."""
        tifffile = pytest.importorskip("tifffile")
        input_path = tmp_path / "input_rgba.png"
        output_path = tmp_path / "output.png"
        alpha = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        rgba = np.dstack([np.full((2, 2, 3), 127, dtype=np.uint8), alpha])
        Image.fromarray(rgba, mode="RGBA").save(input_path)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as stage_cls:
            result = Mock(status=StageStatus.COMPLETED, metadata={})
            result.artifacts = {"enhanced_image": np.full((2, 2, 3), 32768, dtype=np.uint16)}
            stage_cls.return_value.compute.return_value = result
            report = enhance_image(input_path, output_path, output_bit_depth=16)

        reopened = tifffile.imread(report["output"])
        assert reopened.dtype == np.uint16
        np.testing.assert_array_equal(reopened[:, :, 3], alpha.astype(np.uint16) * 257)

    def test_enhance_image_reports_no_material_masks_supplied(self, tmp_path):
        """V2 metadata should separate mask handoff from pixel adjustments."""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        test_image = np.full((12, 12, 3), 128, dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        report = enhance_image(
            input_path,
            output_path,
            config=V2EnhancementConfig(
                enhancement_strength=0.1,
                clarity_strength=0.0,
                material_strength=1.0,
            ),
        )

        metadata = report["enhancement_metadata"]
        assert metadata["material_masks_supplied"] is False
        assert metadata["material_masks_supplied_count"] == 0
        assert metadata["v2_material_adjustments_applied"] is False
        assert metadata["materials_applied"] is False

    def test_enhance_image_reports_supported_material_adjustment_applied(self, tmp_path):
        """Supported non-empty masks should report actual V2 material adjustment."""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        test_image = np.full((12, 12, 3), 128, dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        report = enhance_image(
            input_path,
            output_path,
            material_masks={"glass": np.ones((12, 12), dtype=np.float32)},
            config=V2EnhancementConfig(
                enhancement_strength=0.0,
                clarity_strength=0.0,
                material_strength=1.0,
            ),
        )

        metadata = report["enhancement_metadata"]
        assert metadata["material_masks_supplied"] is True
        assert metadata["material_masks_supplied_count"] == 1
        assert metadata["v2_material_adjustments_applied"] is True
        assert metadata["materials_applied"] is True

    def test_enhance_image_reports_empty_material_mask_supplied_but_not_applied(self, tmp_path):
        """Empty masks count as supplied but cannot apply V2 pixel adjustment."""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        test_image = np.full((12, 12, 3), 128, dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        report = enhance_image(
            input_path,
            output_path,
            material_masks={"wood": np.zeros((12, 12), dtype=np.float32)},
            config=V2EnhancementConfig(
                enhancement_strength=0.0,
                clarity_strength=0.0,
                material_strength=1.0,
            ),
        )

        metadata = report["enhancement_metadata"]
        assert metadata["material_masks_supplied"] is True
        assert metadata["material_masks_supplied_count"] == 1
        assert metadata["v2_material_adjustments_applied"] is False
        assert metadata["materials_applied"] is False

    def test_enhance_image_reports_unsupported_mask_supplied_but_not_applied(self, tmp_path):
        """Unsupported masks from handoff are not V2 material adjustments."""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        test_image = np.full((12, 12, 3), 128, dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        report = enhance_image(
            input_path,
            output_path,
            material_masks={"water": np.ones((12, 12), dtype=np.float32)},
            config=V2EnhancementConfig(
                enhancement_strength=0.0,
                clarity_strength=0.0,
                material_strength=1.0,
            ),
        )

        metadata = report["enhancement_metadata"]
        assert metadata["material_masks_supplied"] is True
        assert metadata["material_masks_supplied_count"] == 1
        assert metadata["v2_material_adjustments_applied"] is False
        assert metadata["materials_applied"] is False

    def test_enhance_image_reports_masks_supplied_without_material_strength(self, tmp_path):
        """Material-strength-disabled handoffs should not imply V2 pixel adjustment."""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        test_image = np.full((12, 12, 3), 128, dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        report = enhance_image(
            input_path,
            output_path,
            material_masks={"glass": np.ones((12, 12), dtype=np.float32)},
            config=V2EnhancementConfig(
                enhancement_strength=0.1,
                clarity_strength=0.0,
                material_strength=0.0,
            ),
        )

        metadata = report["enhancement_metadata"]
        assert metadata["material_masks_supplied"] is True
        assert metadata["material_masks_supplied_count"] == 1
        assert metadata["v2_material_adjustments_applied"] is False
        assert metadata["materials_applied"] is False

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
            assert report["depth_consumed"] is True

            # Check that compute was called with depth_map in context
            call_args = mock_stage.compute.call_args
            context = call_args[0][0]
            assert context.get_artifact("depth_map") is not None

    def test_enhance_image_with_mismatched_depth_dimensions(self, tmp_path):
        """Depth map should be resized to match image dimensions before tone mapping."""
        input_path = tmp_path / "input.png"
        image = np.random.randint(0, 256, (80, 100, 3), dtype=np.uint8)
        Image.fromarray(image, mode="RGB").save(input_path)

        depth_path = tmp_path / "depth.png"
        depth_data = np.random.randint(0, 256, (82, 108), dtype=np.uint8)
        Image.fromarray(depth_data, mode="L").save(depth_path)

        output_path = tmp_path / "output.png"
        expected_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8, materials_enabled=False)
        config = V2EnhancementConfig(
            preset="default",
            enhancement_strength=0.7,
            clarity_strength=0.0,
            material_strength=0.0,
        )

        report = enhance_image(input_path, output_path, depth_map_path=depth_path, config=config)

        assert report["status"] == "success"
        assert report["depth_consumed"] is True
        assert expected_output.exists()

        output_image = Image.open(expected_output)
        assert output_image.size == (100, 80)

    def test_enhance_image_depth_consumed_prefers_stage_metadata(self, tmp_path):
        """Test that depth_consumed follows stage metadata when present."""
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        depth_path = tmp_path / "depth.png"
        depth_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        Image.fromarray(depth_data, mode="L").save(depth_path)

        output_path = tmp_path / "output.png"

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image}
            mock_result.metadata = {"has_depth": False}
            mock_stage.compute.return_value = mock_result

            report = enhance_image(input_path, output_path, depth_map_path=depth_path)

            assert report["depth_map"] == str(depth_path)
            assert report["depth_consumed"] is False

            # Verify structured depth block reflects stage override
            assert "depth" in report
            assert report["depth"]["loaded"] is True
            assert report["depth"]["supplied_to_stage"] is True
            assert report["depth"]["consumed"] is False
            assert report["depth"]["consumption_source"] == "stage_metadata"
            assert report["depth"]["stage_has_depth"] is False

    def test_enhance_image_depth_block_with_real_depth(self, tmp_path):
        """Test structured depth block when depth is loaded and consumed."""
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        depth_path = tmp_path / "depth.png"
        depth_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        Image.fromarray(depth_data, mode="L").save(depth_path)

        output_path = tmp_path / "output.png"

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image}
            mock_result.metadata = {}  # No explicit has_depth
            mock_stage.compute.return_value = mock_result

            report = enhance_image(input_path, output_path, depth_map_path=depth_path)

            # Verify structured depth block
            assert "depth" in report
            assert report["depth"]["requested"] is True
            assert report["depth"]["resolved_path"] == str(depth_path)
            assert report["depth"]["loaded"] is True
            assert report["depth"]["supplied_to_stage"] is True
            assert report["depth"]["consumed"] is True
            assert report["depth"]["consumption_source"] == "fallback_input_presence"
            assert report["depth"]["stage_has_depth"] is None

    def test_enhance_image_depth_block_no_depth_requested(self, tmp_path):
        """Test structured depth block when no depth is requested."""
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

            report = enhance_image(input_path, output_path, depth_map_path=None)

            # Verify structured depth block for no depth request
            assert "depth" in report
            assert report["depth"]["requested"] is False
            assert report["depth"]["resolved_path"] is None
            assert report["depth"]["loaded"] is False
            assert report["depth"]["supplied_to_stage"] is False
            assert report["depth"]["consumed"] is False
            assert report["depth"]["consumption_source"] == "not_requested"
            assert report["depth"]["stage_has_depth"] is None

    def test_enhance_image_depth_block_passthrough(self, tmp_path):
        """Test structured depth block in passthrough mode."""
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        depth_path = tmp_path / "depth.png"
        depth_data = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        Image.fromarray(depth_data, mode="L").save(depth_path)

        output_path = tmp_path / "output.png"

        # Use "none" preset for passthrough
        config = V2EnhancementConfig(preset="none")
        report = enhance_image(input_path, output_path, depth_map_path=depth_path, config=config)

        assert report["status"] == "passthrough"
        assert report["artifact_contract"] == "passthrough_exact_copy"
        assert report["is_canonical_emitted_artifact"] is False
        assert report["output_naming_policy"] == "caller_path_exact"
        assert "depth" in report
        assert report["depth"]["requested"] is True
        assert report["depth"]["loaded"] is False  # Not loaded in passthrough
        assert report["depth"]["supplied_to_stage"] is False
        assert report["depth"]["consumed"] is False
        assert report["depth"]["consumption_source"] == "passthrough"

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
            assert report["depth_consumed"] is False
            assert "enhancement skipped" in report["message"]
            assert report["artifact_contract"] == "passthrough_exact_copy"
            assert report["is_canonical_emitted_artifact"] is False
            assert report["output_naming_policy"] == "caller_path_exact"

            # Verify EnhancementStage was NOT called
            mock_stage_cls.assert_not_called()

            # Verify output file exists (copied from input)
            assert output_path.exists()
            assert not resolve_v2_emitted_artifact_path(output_path, bit_depth=8, materials_enabled=False).exists()

    @pytest.mark.parametrize("output_bit_depth", [8, 16])
    def test_none_preset_explicit_output_depth_uses_canonical_encoding(self, tmp_path, output_bit_depth):
        """Preset none skips adjustments, but never bypasses an explicit encoding contract."""
        input_path = tmp_path / "input.png"
        source = np.linspace(0, 255, 60, dtype=np.uint8).reshape((4, 5, 3))
        Image.fromarray(source, mode="RGB").save(input_path)
        caller_path = tmp_path / "output.tif"
        config = V2EnhancementConfig.from_preset("none")

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as stage_cls:
            report = enhance_image(
                input_path,
                caller_path,
                config=config,
                output_bit_depth=output_bit_depth,
            )

        stage_cls.assert_not_called()
        emitted_path = Path(report["output"])
        assert report["status"] == "success"
        assert report["artifact_contract"] == "canonical_v2_emitted_artifact"
        assert report["is_canonical_emitted_artifact"] is True
        assert report["bit_depth"]["output_bits_per_sample"] == output_bit_depth
        if output_bit_depth == 16:
            tifffile = pytest.importorskip("tifffile")
            reopened = tifffile.imread(emitted_path)
            assert emitted_path.suffix == ".tif"
            assert reopened.dtype == np.uint16
            assert int(reopened.max()) > 255
            assert emitted_path.read_bytes()[:4] in {b"II*\x00", b"MM\x00*"}
        else:
            with Image.open(emitted_path) as reopened_image:
                reopened = np.asarray(reopened_image)
                assert reopened_image.format == "PNG"
            assert emitted_path.suffix == ".png"
            assert reopened.dtype == np.uint8
            assert emitted_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    def test_enhance_image_normalizes_inherited_raw_suffix_for_8bit_output(self, tmp_path):
        """8-bit enhancement should overwrite inherited RAW suffixes before save."""
        input_path = tmp_path / "input.png"
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        Image.fromarray(test_image, mode="RGB").save(input_path)

        raw_suffix_output = tmp_path / "input.DNG"
        expected_output = tmp_path / "input_v2_enhanced.png"

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            report = enhance_image(input_path, raw_suffix_output)

        assert report["status"] == "success"
        assert report["output"] == str(expected_output)
        assert expected_output.exists()
        assert not raw_suffix_output.exists()

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
        expected_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8, materials_enabled=False)

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
            assert expected_output.exists()

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

    def test_enhance_image_rgba_preserves_alpha(self, tmp_path):
        """Test that RGBA inputs preserve alpha channel byte-for-byte."""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        expected_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8, materials_enabled=False)

        # Create RGBA test image with distinct alpha channel
        rgb = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        alpha = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        rgba_image = np.dstack([rgb, alpha])
        Image.fromarray(rgba_image, mode="RGBA").save(input_path)

        # Mock EnhancementStage to return enhanced RGB
        enhanced_rgb = np.clip(rgb.astype(np.float32) * 1.1, 0, 255).astype(np.uint8)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": enhanced_rgb}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            # Run enhancement
            report = enhance_image(input_path, output_path)

            # Verify output is RGBA
            output_image = Image.open(expected_output)
            assert output_image.mode == "RGBA"

            # Verify alpha channel preserved byte-for-byte
            output_array = np.array(output_image)
            assert output_array.shape[2] == 4
            np.testing.assert_array_equal(output_array[:, :, 3], alpha, err_msg="Alpha channel not preserved byte-for-byte")

            # Verify RGB was enhanced (EnhancementStage received RGB only)
            call_args = mock_stage.compute.call_args
            context = call_args[0][0]
            input_to_stage = context.get_artifact("image")
            assert input_to_stage.shape == (100, 100, 3)  # RGB only, no alpha

    def test_enhance_image_preserves_spatial_dimensions(self, tmp_path):
        """Test that enhancement never changes spatial dimensions."""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"
        expected_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8, materials_enabled=False)

        # Test with various image sizes
        test_sizes = [(100, 100), (640, 480), (1920, 1080), (99, 157)]  # Including odd dimensions

        for height, width in test_sizes:
            test_image = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            Image.fromarray(test_image, mode="RGB").save(input_path)

            with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
                mock_stage = Mock()
                mock_stage_cls.return_value = mock_stage

                mock_result = Mock()
                mock_result.status = StageStatus.COMPLETED
                mock_result.artifacts = {"enhanced_image": test_image}
                mock_result.metadata = {}
                mock_stage.compute.return_value = mock_result

                enhance_image(input_path, output_path)

                # Verify spatial dimensions preserved
                output_image = Image.open(expected_output)
                assert output_image.size == (width, height), f"Dimensions changed for {width}x{height}"

    def test_enhance_image_passthrough_sha256_identical(self, tmp_path):
        """Test that 'none' preset creates byte-identical copy including metadata."""
        import hashlib

        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.jpg"
        expected_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8, materials_enabled=False)

        # Create test JPEG with EXIF and ICC profile
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image, mode="RGB")

        # Add fake EXIF data
        exif_bytes = b"Exif\x00\x00MM\x00*\x00\x00\x00\x08"  # Minimal EXIF header
        # Add fake ICC profile
        icc_profile = b"ICC_PROFILE" * 100  # Fake ICC data

        pil_image.save(input_path, format="JPEG", quality=95, exif=exif_bytes, icc_profile=icc_profile)

        # Get input file SHA256
        with open(input_path, "rb") as f:
            input_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Run passthrough enhancement
        config = V2EnhancementConfig.from_preset("none")
        report = enhance_image(input_path, output_path, config=config)

        assert report["status"] == "passthrough"
        assert report["artifact_contract"] == "passthrough_exact_copy"
        assert report["is_canonical_emitted_artifact"] is False
        assert report["output_naming_policy"] == "caller_path_exact"
        assert not expected_output.exists()

        # Get output file SHA256
        with open(output_path, "rb") as f:
            output_sha256 = hashlib.sha256(f.read()).hexdigest()

        # Verify byte-identical copy
        assert input_sha256 == output_sha256, "Passthrough copy is not byte-identical"

    def test_enhance_image_preserves_icc_profile(self, tmp_path):
        """Test that ICC color profiles are preserved in enhanced images."""
        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.jpg"
        expected_output = resolve_v2_emitted_artifact_path(output_path, bit_depth=8, materials_enabled=False)

        # Create test image with ICC profile
        test_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image, mode="RGB")
        icc_profile = b"ICC_PROFILE_DATA" * 100  # Fake ICC profile

        pil_image.save(input_path, format="JPEG", quality=95, icc_profile=icc_profile)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            enhance_image(input_path, output_path)

            # Verify ICC profile preserved
            output_image = Image.open(expected_output)
            output_icc = output_image.info.get("icc_profile")

            assert output_icc is not None, "ICC profile was not preserved"
            assert output_icc == icc_profile, "ICC profile data differs"

    def test_enhance_image_normalizes_inherited_raw_suffix_for_16bit_output(self, tmp_path):
        """16-bit enhancement should overwrite inherited RAW suffixes before save."""
        tifffile = pytest.importorskip("tifffile")

        input_path = tmp_path / "input_16bit.tif"
        output_path = tmp_path / "input.CR2"
        expected_output = tmp_path / "input_v2_enhanced.tif"

        test_image_16 = np.random.randint(0, 65536, (32, 32, 3), dtype=np.uint16)
        tifffile.imwrite(input_path, test_image_16, photometric="rgb", compression=None)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image_16}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            report = enhance_image(input_path, output_path, allow_8bit_output=False)

        assert report["status"] == "success"
        assert report["output"] == str(expected_output)
        assert Path(report["output"]) == expected_output
        assert expected_output.exists()
        assert not output_path.exists()

    def test_enhance_image_preserves_icc_profile_16bit_tiff(self, tmp_path):
        """Test that ICC profiles are preserved in 16-bit TIFF output via tifffile.

        Regression test for the ICC profile preservation path added in the 16-bit
        TIFF code path using tifffile extratags (tag 34675).
        """
        tifffile = pytest.importorskip("tifffile")

        input_path = tmp_path / "input_16bit.tif"
        output_path = tmp_path / "output_16bit.tif"

        # Seed RNG for deterministic test behavior
        rng = np.random.default_rng(seed=42)

        # Create 16-bit test image with ICC profile
        test_image_16 = rng.integers(0, 65535, size=(100, 100, 3), dtype=np.uint16)

        # Create a minimal fake ICC profile for testing purposes.
        # Real ICC profiles have a 128-byte header; we use a small payload (~1.1KB)
        # sufficient to verify the preservation mechanism works.
        icc_profile = b"\x00\x00\x02\x10" + b"ICC_PROFILE_DATA_16BIT" * 50

        # Save input with ICC profile using tifffile extratags (uncompressed to avoid imagecodecs)
        tifffile.imwrite(
            input_path,
            test_image_16,
            photometric="rgb",
            compression=None,  # No compression to avoid imagecodecs dependency
            extratags=[(34675, "B", len(icc_profile), icc_profile, False)],
        )

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            # Return 16-bit enhanced image
            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": test_image_16}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            # Mock tifffile.imwrite at the tifffile module level (since it's imported locally)
            with patch("tifffile.imwrite") as mock_imwrite:
                enhance_image(input_path, output_path, allow_8bit_output=False)

                # Verify tifffile.imwrite was called
                assert mock_imwrite.called, "tifffile.imwrite was not called"

                # Get the call arguments
                call_kwargs = mock_imwrite.call_args[1]

                # Verify extratags contains ICC profile (tag 34675)
                extratags = call_kwargs.get("extratags")
                assert extratags is not None, "extratags not passed to tifffile.imwrite"

                # Find the ICC profile tag in extratags
                icc_tag_found = False
                for tag in extratags:
                    if tag[0] == 34675:  # ICC profile tag
                        icc_tag_found = True
                        # tag format: (tag_id, dtype, count, value, writeonce)
                        assert tag[3] == icc_profile, "ICC profile data differs"
                        break

                assert icc_tag_found, "ICC profile tag 34675 not found in extratags"

    def test_enhance_image_handles_exif_orientation(self, tmp_path):
        """Test that EXIF orientation is properly handled."""
        input_path = tmp_path / "input.jpg"
        output_path = tmp_path / "output.jpg"

        # Create test image (portrait orientation)
        test_image = np.random.randint(0, 256, (200, 100, 3), dtype=np.uint8)
        pil_image = Image.fromarray(test_image, mode="RGB")

        # Save with EXIF orientation tag (will be handled by ImageOps.exif_transpose)
        pil_image.save(input_path, format="JPEG", quality=95)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            # Patch ImageOps.exif_transpose to verify it's called
            with patch("transformation_portal.lux_depth_v3.v2_enhance.ImageOps.exif_transpose") as mock_transpose:
                mock_transpose.return_value = pil_image

                mock_stage = Mock()
                mock_stage_cls.return_value = mock_stage

                mock_result = Mock()
                mock_result.status = StageStatus.COMPLETED
                mock_result.artifacts = {"enhanced_image": test_image}
                mock_result.metadata = {}
                mock_stage.compute.return_value = mock_result

                enhance_image(input_path, output_path)

                # Verify exif_transpose was called
                mock_transpose.assert_called_once()

    @pytest.mark.parametrize("orientation", range(1, 9))
    def test_numpy_exif_orientation_matches_pillow_reference(self, orientation):
        """Numpy EXIF transforms should match ImageOps.exif_transpose geometry."""
        source = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
        image = Image.fromarray(source, mode="RGB")
        exif = image.getexif()
        exif[0x0112] = orientation
        image.info["exif"] = exif.tobytes()

        expected = np.asarray(ImageOps.exif_transpose(image))
        actual = _apply_exif_orientation_to_array(source, orientation)

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize("orientation", [2, 4, 5, 7])
    def test_16bit_tiff_applies_mirrored_exif_orientations(self, tmp_path, orientation):
        """The tifffile loader path should apply mirrored EXIF orientations too."""
        tifffile = pytest.importorskip("tifffile")

        input_path = tmp_path / f"input_orientation_{orientation}.tif"
        output_path = tmp_path / f"output_orientation_{orientation}.tif"
        source = np.arange(2 * 3 * 3, dtype=np.uint16).reshape(2, 3, 3)
        tifffile.imwrite(
            input_path,
            source,
            photometric="rgb",
            compression=None,
            extratags=[(274, "H", 1, orientation, False)],
        )

        with Image.open(input_path) as opened:
            assert opened.getexif().get(0x0112) == orientation

        if orientation == 2:
            expected = np.flip(source, axis=1)
        elif orientation == 4:
            expected = np.flip(source, axis=0)
        elif orientation == 5:
            expected = np.rot90(np.flip(source, axis=1), 1)
        else:
            expected = np.rot90(np.flip(source, axis=1), -1)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            def compute(context):
                image_from_context = context.get_artifact("image")
                mock_result = Mock()
                mock_result.status = StageStatus.COMPLETED
                mock_result.artifacts = {"enhanced_image": image_from_context.copy()}
                mock_result.metadata = {}
                return mock_result

            mock_stage.compute.side_effect = compute

            with patch("tifffile.imwrite"):
                report = enhance_image(input_path, output_path, allow_8bit_output=False)

        assert report["status"] == "success"
        assert report["io"]["load_backend"] == "tifffile"
        assert report["io"]["save_backend"] == "tifffile"
        assert report["io"]["metadata_preservation_mode"] == "none"
        assert report["io"]["exif_preservation_mode"] == "none"
        assert report["io"]["exif_orientation_normalized"] is True
        assert report["io"]["source_exif_orientation"] == orientation

        call_args = mock_stage.compute.call_args
        context = call_args[0][0]
        np.testing.assert_array_equal(context.get_artifact("image"), expected)

    def test_16bit_tiff_allowed_8bit_output_reports_normalized_exif(self, tmp_path):
        """PIL 8-bit output should not report normalized EXIF as fully preserved."""
        tifffile = pytest.importorskip("tifffile")

        input_path = tmp_path / "input_orientation_with_exif.tif"
        output_path = tmp_path / "output_orientation_with_exif.tif"
        source = np.arange(2 * 3 * 3, dtype=np.uint16).reshape(2, 3, 3)
        tifffile.imwrite(
            input_path,
            source,
            photometric="rgb",
            compression=None,
            extratags=[
                (274, "H", 1, 6, False),
                (315, "s", len("transformation-portal") + 1, "transformation-portal", False),
            ],
        )

        with Image.open(input_path) as opened:
            exif = opened.getexif()
            assert exif.get(0x0112) == 6
            assert exif.get(315) == "transformation-portal"

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            def compute(context):
                image_from_context = context.get_artifact("image")
                mock_result = Mock()
                mock_result.status = StageStatus.COMPLETED
                mock_result.artifacts = {"enhanced_image": (image_from_context / 257).astype(np.uint8)}
                mock_result.metadata = {}
                return mock_result

            mock_stage.compute.side_effect = compute

            report = enhance_image(input_path, output_path, allow_8bit_output=True)

        assert report["status"] == "success"
        assert report["io"]["load_backend"] == "tifffile"
        assert report["io"]["save_backend"] == "pil"
        assert report["io"]["save_degraded"] is True
        assert report["io"]["save_degradation_reason"] == "allow_8bit_output"
        assert report["io"]["metadata_preservation_mode"] == "partial"
        assert report["io"]["exif_preservation_mode"] == "normalized"
        assert report["io"]["exif_orientation_normalized"] is True
        assert report["io"]["source_exif_orientation"] == 6

    def test_extract_icc_profile_prefers_pil_info(self):
        """PIL-exposed ICC payload is the first source of truth."""
        icc_bytes = bytes(range(1, 12))
        image = Image.new("RGB", (1, 1))
        image.info["icc_profile"] = icc_bytes

        assert _extract_icc_profile(image) == icc_bytes

    def test_normalize_icc_profile_payload_tolerates_common_forms(self):
        """ICC fallback normalization accepts bytes, bytearrays, ints, and chunks."""
        assert _normalize_icc_profile_payload(b"abc") == b"abc"
        assert _normalize_icc_profile_payload(bytearray(b"abc")) == b"abc"
        assert _normalize_icc_profile_payload([65, 66, 67]) == b"ABC"
        assert _normalize_icc_profile_payload((b"ab", bytearray(b"cd"))) == b"abcd"
        assert _normalize_icc_profile_payload(b"") is None
        assert _normalize_icc_profile_payload([999]) is None
        assert _normalize_icc_profile_payload((b"ab", object())) is None

    def test_16bit_tiff_preserves_icc_from_tag_34675_and_strips_exif(self, tmp_path):
        """V2 TIFF output should preserve ICC only and report partial metadata."""
        tifffile = pytest.importorskip("tifffile")

        input_path = tmp_path / "input_with_icc_tag.tif"
        output_path = tmp_path / "output_with_icc_tag.tif"
        icc_bytes = b"test-icc-profile"
        source = np.full((4, 5, 3), 32768, dtype=np.uint16)
        tifffile.imwrite(
            input_path,
            source,
            photometric="rgb",
            compression=None,
            extratags=[
                (274, "H", 1, 1, False),
                (315, "s", len("transformation-portal") + 1, "transformation-portal", False),
                (34675, "B", len(icc_bytes), icc_bytes, False),
            ],
        )

        with Image.open(input_path) as opened:
            assert _extract_icc_profile(opened) == icc_bytes

        report = enhance_image(
            input_path,
            output_path,
            config=V2EnhancementConfig(
                enhancement_strength=0.1,
                clarity_strength=0.0,
                material_strength=0.0,
            ),
        )

        assert report["status"] == "success"
        assert report["io"]["save_backend"] == "tifffile"
        assert report["io"]["icc_preserved"] is True
        assert report["io"]["exif_preservation_mode"] == "none"
        assert report["io"]["metadata_preservation_mode"] == "partial"

        with tifffile.TiffFile(report["output"]) as tif:
            icc_tag = tif.pages[0].tags.get(34675)
            assert icc_tag is not None
            assert _normalize_icc_profile_payload(icc_tag.value) == icc_bytes
            assert tif.pages[0].tags.get(315) is None

    def test_enhance_image_handles_palette_mode(self, tmp_path):
        """Test that palette (P) mode images are converted to RGB."""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"

        # Create palette mode image
        test_image = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        pil_image = Image.fromarray(test_image, mode="L")
        pil_image = pil_image.convert("P")  # Convert to palette mode
        pil_image.save(input_path)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            # Enhancement should receive RGB
            rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": rgb_image}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            enhance_image(input_path, output_path)

            # Verify EnhancementStage received RGB (H, W, 3)
            call_args = mock_stage.compute.call_args
            context = call_args[0][0]
            input_to_stage = context.get_artifact("image")
            assert input_to_stage.shape == (100, 100, 3), "Palette image not converted to RGB"

    def test_enhance_image_handles_la_mode(self, tmp_path):
        """Test that LA (luminance + alpha) mode images are converted to RGBA."""
        input_path = tmp_path / "input.png"
        output_path = tmp_path / "output.png"

        # Create LA mode image
        la_array = np.random.randint(0, 256, (100, 100, 2), dtype=np.uint8)
        pil_image = Image.fromarray(la_array, mode="LA")
        pil_image.save(input_path)

        with patch("transformation_portal.lux_depth_v3.v2_enhance.EnhancementStage") as mock_stage_cls:
            mock_stage = Mock()
            mock_stage_cls.return_value = mock_stage

            # Enhancement should receive RGB
            rgb_image = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            mock_result = Mock()
            mock_result.status = StageStatus.COMPLETED
            mock_result.artifacts = {"enhanced_image": rgb_image}
            mock_result.metadata = {}
            mock_stage.compute.return_value = mock_result

            enhance_image(input_path, output_path)

            # Verify EnhancementStage received RGB (H, W, 3)
            call_args = mock_stage.compute.call_args
            context = call_args[0][0]
            input_to_stage = context.get_artifact("image")
            assert input_to_stage.shape == (100, 100, 3), "LA image not converted to RGB"

    def test_load_depth_map_handles_all_zeros(self, tmp_path):
        """Test that all-zeros depth maps are handled gracefully."""
        depth_path = tmp_path / "depth_zeros.png"

        # Create all-zeros depth map
        depth_data = np.zeros((100, 100), dtype=np.uint8)
        Image.fromarray(depth_data, mode="L").save(depth_path)

        # Should load without error
        loaded = load_depth_map(depth_path)

        assert loaded.shape == (100, 100)
        assert loaded.dtype == np.float32
        assert loaded.max() == 0.0
        assert loaded.min() == 0.0
