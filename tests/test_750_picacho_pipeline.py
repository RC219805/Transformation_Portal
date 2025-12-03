#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Production-readiness tests for the 750 Picacho Lane end-to-end pipeline.

This test suite validates:
- Pipeline module imports and configuration
- All 5 processing stages function correctly
- Production output files exist and meet quality standards
- Configuration presets are valid
- Error handling is robust

Note: This test file adds the projects directory to sys.path because the
picacho_pool_remediation_pipeline module is a standalone script in the
projects directory, not an installed package. This is consistent with
how the pipeline is used in production (directly executed from that directory).

Author: Transformation Portal
Date: 2025-12-03
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

# Add projects directory to path for imports - required because the pipeline
# is a standalone script in projects/750_picacho_lane, not an installed package
PROJECT_DIR = Path(__file__).parent.parent / "projects" / "750_picacho_lane"
sys.path.insert(0, str(PROJECT_DIR))

from picacho_pool_remediation_pipeline import (  # noqa: E402
    AtmosphericIntegrator,
    DepthPostProcessor,
    LightingStratification,
    MaterialSystemReconstructor,
    MaterialType,
    PBRMaterialProperties,
    PicachoPoolRemediationPipeline,
    StylingRectifier,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_image():
    """Create a sample RGB image for testing."""
    # Create a test image with various colors to test material detection
    img = np.zeros((256, 256, 3), dtype=np.float32)
    # Blue region (water-like)
    img[:128, :128, 2] = 0.8
    img[:128, :128, 1] = 0.4
    img[:128, :128, 0] = 0.2
    # Neutral region (stone-like)
    img[:128, 128:, :] = 0.6
    # Warm region (wood-like)
    img[128:, :128, 0] = 0.5
    img[128:, :128, 1] = 0.35
    img[128:, :128, 2] = 0.25
    # Bright region
    img[128:, 128:, :] = 0.9
    return img


@pytest.fixture
def sample_depth_map():
    """Create a sample depth map for testing."""
    depth = np.zeros((256, 256), dtype=np.float32)
    # Gradient from near (0) to far (1)
    for i in range(256):
        depth[i, :] = i / 255.0
    return depth


@pytest.fixture
def temp_image_file(sample_image, tmp_path):
    """Create a temporary image file."""
    image_path = tmp_path / "test_image.tif"
    img_uint8 = (sample_image * 255).astype(np.uint8)
    Image.fromarray(img_uint8, mode="RGB").save(image_path)
    return image_path


@pytest.fixture
def default_config():
    """Return default pipeline configuration."""
    return {
        "lighting_zones": 4,
        "darkness_preservation": 0.35,
        "scattering_threshold_m": 30.0,
        "enable_material_reconstruction": True,
        "enable_atmospheric_integration": True,
        "enable_lighting_stratification": True,
        "enable_styling_rectification": True,
        "enable_depth_processing": True,
    }


# ============================================================================
# Module Import Tests
# ============================================================================


class TestModuleImports:
    """Test that all pipeline modules can be imported correctly."""

    def test_pipeline_import(self):
        """Test main pipeline class can be imported."""
        assert PicachoPoolRemediationPipeline is not None

    def test_material_type_enum(self):
        """Test MaterialType enum has all expected values."""
        expected_materials = ["PLASTER", "STONE", "WOOD", "WATER", "GLASS", "METAL"]
        for material in expected_materials:
            assert hasattr(MaterialType, material)

    def test_pbr_material_properties(self):
        """Test PBRMaterialProperties dataclass."""
        props = PBRMaterialProperties(
            name="Test Material",
            albedo_color=(0.5, 0.5, 0.5),
            roughness=0.5,
        )
        assert props.name == "Test Material"
        assert props.albedo_color == (0.5, 0.5, 0.5)
        assert props.roughness == 0.5

    def test_stage_processors_import(self):
        """Test all stage processors can be imported."""
        assert MaterialSystemReconstructor is not None
        assert AtmosphericIntegrator is not None
        assert LightingStratification is not None
        assert StylingRectifier is not None
        assert DepthPostProcessor is not None


# ============================================================================
# Material System Reconstruction Tests (Stage 1)
# ============================================================================


class TestMaterialSystemReconstructor:
    """Tests for Stage 1: Material System Reconstruction."""

    def test_init(self):
        """Test MaterialSystemReconstructor initialization."""
        reconstructor = MaterialSystemReconstructor()
        assert reconstructor.material_masks == {}

    def test_detect_materials(self, sample_image):
        """Test material detection from image."""
        reconstructor = MaterialSystemReconstructor()
        masks = reconstructor.detect_materials(sample_image)

        assert MaterialType.WATER in masks
        assert MaterialType.STONE in masks
        assert MaterialType.WOOD in masks

        # Masks should be numpy arrays
        for mask in masks.values():
            assert isinstance(mask, np.ndarray)
            assert mask.shape == sample_image.shape[:2]

    def test_pbr_enhancement(self, sample_image):
        """Test PBR material enhancement."""
        reconstructor = MaterialSystemReconstructor()
        masks = reconstructor.detect_materials(sample_image)
        enhanced = reconstructor.apply_pbr_enhancement(sample_image, masks)

        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == sample_image.dtype
        # Values should be clipped to [0, 1]
        assert enhanced.min() >= 0
        assert enhanced.max() <= 1

    def test_materials_dictionary(self):
        """Test that MATERIALS dictionary has proper PBR properties."""
        reconstructor = MaterialSystemReconstructor()

        for mat_type in [MaterialType.PLASTER, MaterialType.STONE, MaterialType.WOOD, MaterialType.WATER]:
            assert mat_type in reconstructor.MATERIALS
            props = reconstructor.MATERIALS[mat_type]
            assert isinstance(props, PBRMaterialProperties)
            assert len(props.albedo_color) == 3
            assert 0 <= props.roughness <= 1
            assert 0 <= props.metallic <= 1


# ============================================================================
# Atmospheric Integration Tests (Stage 2)
# ============================================================================


class TestAtmosphericIntegrator:
    """Tests for Stage 2: Atmospheric Integration."""

    def test_init(self):
        """Test AtmosphericIntegrator initialization."""
        integrator = AtmosphericIntegrator()
        assert integrator.blue_hour_intensity == 0.7

    def test_init_custom_intensity(self):
        """Test AtmosphericIntegrator with custom intensity."""
        integrator = AtmosphericIntegrator(blue_hour_intensity=0.5)
        assert integrator.blue_hour_intensity == 0.5

    def test_blue_hour_lighting(self, sample_image):
        """Test blue hour lighting application."""
        integrator = AtmosphericIntegrator()
        enhanced = integrator.apply_blue_hour_lighting(sample_image)

        assert enhanced.shape == sample_image.shape
        assert enhanced.dtype == sample_image.dtype
        assert enhanced.min() >= 0
        assert enhanced.max() <= 1


# ============================================================================
# Lighting Stratification Tests (Stage 3)
# ============================================================================


class TestLightingStratification:
    """Tests for Stage 3: Lighting Stratification."""

    def test_init_defaults(self):
        """Test LightingStratification default initialization."""
        lighting = LightingStratification()
        assert lighting.num_zones == 4
        assert lighting.darkness_preservation == 0.35

    def test_init_custom(self):
        """Test LightingStratification with custom parameters."""
        lighting = LightingStratification(num_zones=6, darkness_preservation=0.5)
        assert lighting.num_zones == 6
        assert lighting.darkness_preservation == 0.5

    def test_multi_zone_lighting_no_depth(self, sample_image):
        """Test multi-zone lighting without depth map (synthetic depth)."""
        lighting = LightingStratification()
        enhanced = lighting.apply_multi_zone_lighting(sample_image)

        assert enhanced.shape == sample_image.shape
        assert enhanced.min() >= 0
        assert enhanced.max() <= 1

    def test_multi_zone_lighting_with_depth(self, sample_image, sample_depth_map):
        """Test multi-zone lighting with depth map."""
        lighting = LightingStratification()
        enhanced = lighting.apply_multi_zone_lighting(sample_image, sample_depth_map)

        assert enhanced.shape == sample_image.shape
        assert enhanced.min() >= 0
        assert enhanced.max() <= 1


# ============================================================================
# Styling Rectification Tests (Stage 4)
# ============================================================================


class TestStylingRectifier:
    """Tests for Stage 4: Styling Rectification."""

    def test_init(self):
        """Test StylingRectifier initialization."""
        rectifier = StylingRectifier()
        assert rectifier.prohibited_elements == []
        assert rectifier.accessories_added == []

    def test_styling_corrections(self, sample_image):
        """Test styling corrections application."""
        rectifier = StylingRectifier()
        corrected = rectifier.apply_styling_corrections(sample_image)

        assert corrected.shape == sample_image.shape
        # Should not modify too aggressively
        assert np.allclose(corrected, sample_image, atol=0.2)


# ============================================================================
# Depth Post-Processing Tests (Stage 5)
# ============================================================================


class TestDepthPostProcessor:
    """Tests for Stage 5: Post-Production Depth Processing."""

    def test_init_defaults(self):
        """Test DepthPostProcessor default initialization."""
        processor = DepthPostProcessor()
        assert processor.distance_threshold == 30.0

    def test_init_custom(self):
        """Test DepthPostProcessor with custom threshold."""
        processor = DepthPostProcessor(distance_threshold_m=50.0)
        assert processor.distance_threshold == 50.0

    def test_atmospheric_scattering_no_depth(self, sample_image):
        """Test atmospheric scattering without depth map."""
        processor = DepthPostProcessor()
        processed = processor.apply_atmospheric_scattering(sample_image)

        assert processed.shape == sample_image.shape
        assert processed.min() >= 0
        assert processed.max() <= 1

    def test_atmospheric_scattering_with_depth(self, sample_image, sample_depth_map):
        """Test atmospheric scattering with depth map."""
        processor = DepthPostProcessor()
        processed = processor.apply_atmospheric_scattering(sample_image, sample_depth_map)

        assert processed.shape == sample_image.shape
        assert processed.min() >= 0
        assert processed.max() <= 1


# ============================================================================
# Full Pipeline Tests
# ============================================================================


class TestPicachoPoolRemediationPipeline:
    """Tests for the complete remediation pipeline."""

    def test_init_default_config(self):
        """Test pipeline initialization with default config."""
        pipeline = PicachoPoolRemediationPipeline()

        assert pipeline.config is not None
        assert pipeline.config["lighting_zones"] == 4
        assert pipeline.config["darkness_preservation"] == 0.35
        assert pipeline.config["scattering_threshold_m"] == 30.0

    def test_init_custom_config(self, default_config):
        """Test pipeline initialization with custom config."""
        default_config["lighting_zones"] = 6
        pipeline = PicachoPoolRemediationPipeline(config=default_config)

        assert pipeline.config["lighting_zones"] == 6

    def test_default_config_method(self):
        """Test _default_config returns valid configuration."""
        pipeline = PicachoPoolRemediationPipeline()
        config = pipeline._default_config()

        required_keys = [
            "lighting_zones",
            "darkness_preservation",
            "scattering_threshold_m",
            "enable_material_reconstruction",
            "enable_atmospheric_integration",
            "enable_lighting_stratification",
            "enable_styling_rectification",
            "enable_depth_processing",
        ]
        for key in required_keys:
            assert key in config

    def test_stage_processors_initialized(self):
        """Test that all stage processors are initialized."""
        pipeline = PicachoPoolRemediationPipeline()

        assert isinstance(pipeline.material_reconstructor, MaterialSystemReconstructor)
        assert isinstance(pipeline.atmospheric_integrator, AtmosphericIntegrator)
        assert isinstance(pipeline.lighting_stratification, LightingStratification)
        assert isinstance(pipeline.styling_rectifier, StylingRectifier)
        assert isinstance(pipeline.depth_processor, DepthPostProcessor)

    def test_process_image(self, temp_image_file, tmp_path):
        """Test processing a single image through the pipeline."""
        output_path = tmp_path / "output.tif"
        pipeline = PicachoPoolRemediationPipeline()

        success = pipeline.process(temp_image_file, output_path)

        assert success is True
        assert output_path.exists()

    def test_process_nonexistent_file(self, tmp_path):
        """Test processing a nonexistent file."""
        input_path = tmp_path / "nonexistent.tif"
        output_path = tmp_path / "output.tif"
        pipeline = PicachoPoolRemediationPipeline()

        success = pipeline.process(input_path, output_path)

        assert success is False
        assert not output_path.exists()

    def test_process_with_stages_disabled(self, temp_image_file, tmp_path):
        """Test processing with some stages disabled."""
        config = {
            "lighting_zones": 4,
            "darkness_preservation": 0.35,
            "scattering_threshold_m": 30.0,
            "enable_material_reconstruction": False,
            "enable_atmospheric_integration": False,
            "enable_lighting_stratification": True,
            "enable_styling_rectification": False,
            "enable_depth_processing": True,
        }
        output_path = tmp_path / "output.tif"
        pipeline = PicachoPoolRemediationPipeline(config=config)

        success = pipeline.process(temp_image_file, output_path)

        assert success is True
        assert output_path.exists()


# ============================================================================
# Production Output Validation Tests
# ============================================================================


class TestProductionOutputs:
    """Tests to verify production output files exist and are valid."""

    @pytest.fixture
    def production_dir(self):
        """Get the production output directory."""
        return Path(__file__).parent.parent / "projects" / "750_picacho_lane" / "Final_Production_UltraQuality"

    def test_production_directory_exists(self, production_dir):
        """Test that production output directory exists."""
        assert production_dir.exists(), f"Production directory not found: {production_dir}"

    def test_all_room_outputs_exist(self, production_dir):
        """Test that all expected room outputs exist."""
        expected_files = [
            "750Picacho_Aerial_UltraQuality.tif",
            "750Picacho_GreatRoom_UltraQuality.tif",
            "750Picacho_Kitchen_UltraQuality.tif",
            "750Picacho_Pool_UltraQuality.tif",
            "750Picacho_PrimaryBathroom_UltraQuality.tif",
            "750Picacho_PrimaryBedroom_UltraQuality.tif",
        ]

        for filename in expected_files:
            file_path = production_dir / filename
            assert file_path.exists(), f"Missing production file: {filename}"

    def test_quality_report_exists(self, production_dir):
        """Test that quality report JSON exists."""
        report_path = production_dir / "ultra_quality_report.json"
        assert report_path.exists(), "Quality report not found"

    def test_quality_report_valid_json(self, production_dir):
        """Test that quality report is valid JSON."""
        report_path = production_dir / "ultra_quality_report.json"
        with open(report_path, "r") as f:
            data = json.load(f)

        assert "project" in data
        assert "processing_summary" in data
        assert "individual_results" in data

    def test_quality_report_all_images_processed(self, production_dir):
        """Test that quality report shows all images were processed."""
        report_path = production_dir / "ultra_quality_report.json"
        with open(report_path, "r") as f:
            data = json.load(f)

        assert data["processing_summary"]["total_images"] >= 6
        assert len(data["individual_results"]) >= 6

    def test_quality_scores_above_threshold(self, production_dir):
        """Test that enhanced quality scores are above minimum threshold."""
        report_path = production_dir / "ultra_quality_report.json"
        with open(report_path, "r") as f:
            data = json.load(f)

        min_threshold = 75.0  # Minimum acceptable quality score

        for result in data["individual_results"]:
            assert result["enhanced_score"] >= min_threshold, (
                f"Quality score below threshold for {result['filename']}: "
                f"{result['enhanced_score']}"
            )

    def test_all_images_improved(self, production_dir):
        """Test that all images show improvement from baseline."""
        report_path = production_dir / "ultra_quality_report.json"
        with open(report_path, "r") as f:
            data = json.load(f)

        for result in data["individual_results"]:
            assert result["improvement"] > 0, (
                f"No improvement for {result['filename']}: "
                f"baseline={result['baseline_score']}, enhanced={result['enhanced_score']}"
            )


# ============================================================================
# Configuration Preset Tests
# ============================================================================


class TestConfigurationPresets:
    """Tests for pipeline configuration presets."""

    @pytest.fixture
    def elite_preset_path(self):
        """Get path to elite preset configuration."""
        return Path(__file__).parent.parent / "config" / "750_picacho_elite_preset.yaml"

    @pytest.fixture
    def master_preset_path(self):
        """Get path to master preset configuration."""
        return Path(__file__).parent.parent / "config" / "750_picacho_master_preset.yaml"

    def test_elite_preset_exists(self, elite_preset_path):
        """Test that elite preset file exists."""
        assert elite_preset_path.exists(), f"Elite preset not found: {elite_preset_path}"

    def test_master_preset_exists(self, master_preset_path):
        """Test that master preset file exists."""
        assert master_preset_path.exists(), f"Master preset not found: {master_preset_path}"

    def test_elite_preset_valid_yaml(self, elite_preset_path):
        """Test that elite preset is valid YAML."""
        with open(elite_preset_path, "r") as f:
            config = yaml.safe_load(f)

        assert "name" in config
        assert "depth" in config
        assert "material_response" in config
        assert "color_grading" in config


# ============================================================================
# Documentation Tests
# ============================================================================


class TestDocumentation:
    """Tests for pipeline documentation completeness."""

    @pytest.fixture
    def docs_dir(self):
        """Get the documentation directory."""
        return Path(__file__).parent.parent / "projects" / "750_picacho_lane"

    def test_readme_exists(self, docs_dir):
        """Test that README file exists."""
        readme = docs_dir / "README_750Picacho_Pool.md"
        assert readme.exists(), "README file not found"

    def test_processing_summary_exists(self, docs_dir):
        """Test that processing summary exists."""
        summary = docs_dir / "PROCESSING_SUMMARY_750Picacho_Pool.md"
        assert summary.exists(), "Processing summary not found"

    def test_execution_report_exists(self, docs_dir):
        """Test that execution report exists."""
        report = docs_dir / "EXECUTION_REPORT.md"
        assert report.exists(), "Execution report not found"

    def test_batch_processing_complete_exists(self, docs_dir):
        """Test that batch processing complete report exists."""
        report = docs_dir / "BATCH_PROCESSING_COMPLETE.md"
        assert report.exists(), "Batch processing report not found"


# ============================================================================
# Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Tests for robust error handling."""

    def test_invalid_image_file(self, tmp_path):
        """Test handling of an invalid/corrupted image file."""
        # Create a file that's not a valid image
        invalid_path = tmp_path / "invalid.tif"
        invalid_path.write_text("This is not a valid image file")

        pipeline = PicachoPoolRemediationPipeline()
        output_path = tmp_path / "output.tif"

        # Should handle gracefully (return False, not crash)
        success = pipeline.process(invalid_path, output_path)
        assert success is False

    def test_grayscale_image_handling(self, tmp_path):
        """Test handling of grayscale (non-RGB) images."""
        # Create a grayscale image (2D instead of 3D)
        grayscale_img = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        image_path = tmp_path / "grayscale.tif"
        Image.fromarray(grayscale_img, mode="L").save(image_path)

        pipeline = PicachoPoolRemediationPipeline()
        output_path = tmp_path / "output.tif"

        # Should handle gracefully (return False since it's not RGB)
        success = pipeline.process(image_path, output_path)
        assert success is False

    def test_empty_masks_handling(self, sample_image):
        """Test handling of empty material masks."""
        reconstructor = MaterialSystemReconstructor()
        # Create empty masks
        empty_masks = {
            MaterialType.WATER: np.zeros(sample_image.shape[:2], dtype=np.float32),
            MaterialType.STONE: np.zeros(sample_image.shape[:2], dtype=np.float32),
        }

        # Should not crash with empty masks
        enhanced = reconstructor.apply_pbr_enhancement(sample_image, empty_masks)
        assert enhanced.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
