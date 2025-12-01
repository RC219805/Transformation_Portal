#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration Tests for Unified Pipeline.

End-to-end tests for the UnifiedPipeline orchestrator and related components.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


# Check if scipy is available (required by pipeline)
try:
    from scipy.ndimage import gaussian_filter
    HAS_SCIPY = True
    del gaussian_filter  # Clean up - not needed in this module
except ImportError:
    HAS_SCIPY = False

pytestmark = pytest.mark.skipif(
    not HAS_SCIPY,
    reason="scipy is required for unified pipeline"
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image():
    """Create sample RGB image for testing."""
    arr = np.zeros((600, 800, 3), dtype=np.uint8)
    for i in range(600):
        arr[i, :, 0] = int(255 * i / 600)
    for j in range(800):
        arr[:, j, 1] = int(255 * j / 800)
    arr[:, :, 2] = 128
    return Image.fromarray(arr, 'RGB')


@pytest.fixture
def sample_image_file(temp_dir, sample_image):
    """Save sample image to file."""
    image_path = temp_dir / "test_image.jpg"
    sample_image.save(image_path, quality=95)
    return image_path


@pytest.fixture
def sample_recipe(temp_dir):
    """Create a minimal test recipe."""
    recipe_content = """
name: "Test Recipe"
description: "Test recipe for unit testing"

stages:
  - material_response
  - color_grading
  - photo_finishing

material_response:
  enabled: true
  profile: "luxury_interior"
  texture_boost: 0.2

color_grading:
  enabled: true
  contrast: 1.05
  saturation: 1.02

photo_finishing:
  enabled: true
  aces: true
  bloom:
    enabled: true
    threshold: 0.85
    intensity: 0.2
  vignette:
    enabled: false
  grain:
    enabled: true
    amount: 0.01

output:
  format: "jpeg"
  quality: 90
"""
    recipe_path = temp_dir / "test_recipe.yaml"
    recipe_path.write_text(recipe_content)
    return recipe_path


class TestConfigLoader:
    """Test configuration loading functionality."""

    def test_load_recipe(self, sample_recipe):
        """Test loading a recipe file."""
        from transformation_portal.config_loader import load_recipe

        recipe = load_recipe(sample_recipe)

        assert recipe['name'] == "Test Recipe"
        assert 'stages' in recipe
        assert 'material_response' in recipe['stages']

    def test_load_nonexistent_recipe(self, temp_dir):
        """Test error on nonexistent recipe."""
        from transformation_portal.config_loader import load_recipe

        with pytest.raises(FileNotFoundError):
            load_recipe(temp_dir / "nonexistent.yaml")

    def test_validate_recipe(self, sample_recipe):
        """Test recipe validation."""
        from transformation_portal.config_loader import load_recipe, validate_recipe

        recipe = load_recipe(sample_recipe)
        is_valid, errors = validate_recipe(recipe)

        assert is_valid
        assert len(errors) == 0

    def test_validate_recipe_missing_name(self, temp_dir):
        """Test validation fails with missing name."""
        from transformation_portal.config_loader import validate_recipe

        invalid_recipe = {
            'stages': ['color_grading'],
        }
        is_valid, errors = validate_recipe(invalid_recipe)

        assert not is_valid
        assert any('name' in e.lower() for e in errors)

    def test_validate_recipe_missing_stages(self, temp_dir):
        """Test validation fails with missing stages."""
        from transformation_portal.config_loader import validate_recipe

        invalid_recipe = {
            'name': 'Test',
        }
        is_valid, errors = validate_recipe(invalid_recipe)

        assert not is_valid
        assert any('stages' in e.lower() for e in errors)

    def test_list_recipes(self, temp_dir, sample_recipe):
        """Test listing recipes in directory."""
        from transformation_portal.config_loader import list_recipes

        recipes = list_recipes(temp_dir)

        assert len(recipes) == 1
        assert recipes[0]['name'] == "Test Recipe"

    def test_environment_variable_expansion(self, temp_dir):
        """Test environment variable expansion in recipes."""
        import os
        from transformation_portal.config_loader import load_recipe

        os.environ['TEST_LUT_PATH'] = '/custom/luts'

        recipe_content = """
name: "Env Test"
stages:
  - color_grading
color_grading:
  lut: "${TEST_LUT_PATH}/test.cube"
"""
        recipe_path = temp_dir / "env_recipe.yaml"
        recipe_path.write_text(recipe_content)

        recipe = load_recipe(recipe_path)

        assert '/custom/luts' in recipe['color_grading']['lut']


class TestUnifiedPipeline:
    """Test UnifiedPipeline orchestrator."""

    def test_pipeline_initialization(self, sample_recipe):
        """Test pipeline can be initialized from recipe."""
        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(sample_recipe)

        assert pipeline.name == "Test Recipe"
        assert len(pipeline.stages) > 0

    def test_pipeline_stages_from_recipe(self, sample_recipe):
        """Test stages are correctly loaded from recipe."""
        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(sample_recipe)
        stage_names = [s.name for s in pipeline.stages]

        assert 'material_response' in stage_names
        assert 'color_grading' in stage_names
        assert 'photo_finishing' in stage_names

    def test_process_single(self, sample_recipe, sample_image_file, temp_dir):
        """Test processing a single image."""
        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(sample_recipe)
        pipeline.recipe['_output_dir'] = str(temp_dir / "output")

        result = pipeline.process_single(sample_image_file)

        assert result.success
        assert result.output_path is not None
        assert result.output_path.exists()
        assert len(result.stages_executed) > 0
        assert result.total_time > 0

    def test_process_single_nonexistent(self, sample_recipe, temp_dir):
        """Test error handling for nonexistent file."""
        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(sample_recipe)
        result = pipeline.process_single(temp_dir / "nonexistent.jpg")

        assert not result.success
        assert result.error_message is not None

    def test_process_batch_dry_run(self, sample_recipe, sample_image_file, temp_dir):
        """Test batch processing dry run."""
        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(sample_recipe)

        result = pipeline.process_batch(
            str(sample_image_file),
            temp_dir / "output",
            dry_run=True
        )

        assert result.dry_run
        assert len(result.results) == 1

    def test_process_batch_multiple_images(self, sample_recipe, temp_dir, sample_image):
        """Test batch processing multiple images."""
        from transformation_portal.pipeline_unified import UnifiedPipeline

        # Create multiple test images
        for i in range(3):
            path = temp_dir / f"test_{i}.jpg"
            sample_image.save(path, quality=90)

        pipeline = UnifiedPipeline.from_recipe(sample_recipe)

        result = pipeline.process_batch(
            str(temp_dir / "test_*.jpg"),
            temp_dir / "output",
            dry_run=False
        )

        assert result.successful_count == 3
        assert result.failed_count == 0
        assert len(result.results) == 3


class TestProcessingResult:
    """Test ProcessingResult dataclass."""

    def test_result_initialization(self, temp_dir):
        """Test ProcessingResult creation."""
        from transformation_portal.pipeline_unified import ProcessingResult

        result = ProcessingResult(
            input_path=temp_dir / "test.jpg",
            success=True,
            total_time=1.5,
        )

        assert result.success
        assert result.total_time == 1.5
        assert result.error_message is None

    def test_result_repr(self, temp_dir):
        """Test ProcessingResult string representation."""
        from transformation_portal.pipeline_unified import ProcessingResult

        result = ProcessingResult(
            input_path=temp_dir / "test.jpg",
            success=True,
            total_time=2.5,
        )

        repr_str = repr(result)
        assert "test.jpg" in repr_str
        assert "2.50" in repr_str


class TestBatchResult:
    """Test BatchResult dataclass."""

    def test_batch_result_summary(self, temp_dir):
        """Test batch result summary generation."""
        from transformation_portal.pipeline_unified import BatchResult, ProcessingResult

        results = [
            ProcessingResult(
                input_path=temp_dir / f"test_{i}.jpg",
                success=(i % 2 == 0),
                total_time=1.0,
                error_message=None if i % 2 == 0 else "Test error",
            )
            for i in range(4)
        ]

        batch = BatchResult(
            results=results,
            total_time=5.0,
            successful_count=2,
            failed_count=2,
        )

        summary = batch.summary()

        assert "Total images: 4" in summary
        assert "Successful: 2" in summary
        assert "Failed: 2" in summary


class TestRecipeValidator:
    """Test recipe validation utilities."""

    def test_validator_initialization(self):
        """Test RecipeValidator can be initialized."""
        from transformation_portal.utils.recipe_validator import RecipeValidator

        validator = RecipeValidator()

        assert validator.schema is not None

    def test_validate_valid_recipe(self, sample_recipe):
        """Test validation of valid recipe."""
        import yaml
        from transformation_portal.utils.recipe_validator import RecipeValidator

        with open(sample_recipe) as f:
            recipe_dict = yaml.safe_load(f)

        validator = RecipeValidator()
        is_valid, errors = validator.validate(recipe_dict)

        assert is_valid
        assert len(errors) == 0

    def test_validate_file(self, sample_recipe):
        """Test validating recipe file."""
        from transformation_portal.utils.recipe_validator import validate_recipe_file

        is_valid, errors = validate_recipe_file(sample_recipe)

        assert is_valid
        assert len(errors) == 0


class TestBuiltInRecipes:
    """Test built-in recipe files."""

    def test_signature_estate_recipe(self):
        """Test signature_estate.yaml is valid."""
        recipe_path = Path("config/recipes/signature_estate.yaml")
        if not recipe_path.exists():
            pytest.skip("Recipe file not found")

        from transformation_portal.config_loader import load_recipe, validate_recipe

        recipe = load_recipe(recipe_path)
        is_valid, errors = validate_recipe(recipe)

        assert is_valid, f"Validation errors: {errors}"
        assert recipe['name'] == "Signature Estate"

    def test_golden_hour_courtyard_recipe(self):
        """Test golden_hour_courtyard.yaml is valid."""
        recipe_path = Path("config/recipes/golden_hour_courtyard.yaml")
        if not recipe_path.exists():
            pytest.skip("Recipe file not found")

        from transformation_portal.config_loader import load_recipe, validate_recipe

        recipe = load_recipe(recipe_path)
        is_valid, errors = validate_recipe(recipe)

        assert is_valid, f"Validation errors: {errors}"
        assert recipe['name'] == "Golden Hour Courtyard"

    def test_interior_neutral_luxe_recipe(self):
        """Test interior_neutral_luxe.yaml is valid."""
        recipe_path = Path("config/recipes/interior_neutral_luxe.yaml")
        if not recipe_path.exists():
            pytest.skip("Recipe file not found")

        from transformation_portal.config_loader import load_recipe, validate_recipe

        recipe = load_recipe(recipe_path)
        is_valid, errors = validate_recipe(recipe)

        assert is_valid, f"Validation errors: {errors}"
        assert recipe['name'] == "Interior Neutral Luxe"

    def test_video_cinematic_hdr_recipe(self):
        """Test video_cinematic_hdr.yaml is valid."""
        recipe_path = Path("config/recipes/video_cinematic_hdr.yaml")
        if not recipe_path.exists():
            pytest.skip("Recipe file not found")

        from transformation_portal.config_loader import load_recipe, validate_recipe

        recipe = load_recipe(recipe_path)
        is_valid, errors = validate_recipe(recipe)

        assert is_valid, f"Validation errors: {errors}"
        assert recipe['name'] == "Video Cinematic HDR"


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_workflow(self, temp_dir, sample_image):
        """Test full pipeline workflow from recipe to output."""
        from transformation_portal.config_loader import load_recipe
        from transformation_portal.pipeline_unified import UnifiedPipeline

        # Create recipe
        recipe_content = """
name: "E2E Test Recipe"
description: "End-to-end test"
stages:
  - material_response
  - color_grading
  - photo_finishing
material_response:
  enabled: true
  texture_boost: 0.2
color_grading:
  enabled: true
  contrast: 1.05
photo_finishing:
  enabled: true
  aces: true
  bloom:
    enabled: false
  vignette:
    enabled: false
  grain:
    enabled: false
output:
  format: "jpeg"
  quality: 85
"""
        recipe_path = temp_dir / "e2e_recipe.yaml"
        recipe_path.write_text(recipe_content)

        # Create input image
        input_path = temp_dir / "input.jpg"
        sample_image.save(input_path, quality=95)

        # Load and validate recipe
        recipe = load_recipe(recipe_path)
        assert recipe['name'] == "E2E Test Recipe"

        # Process image
        pipeline = UnifiedPipeline.from_recipe(recipe_path)
        pipeline.recipe['_output_dir'] = str(temp_dir / "output")

        result = pipeline.process_single(input_path)

        # Verify result
        assert result.success, f"Pipeline failed: {result.error_message}"
        assert result.output_path is not None
        assert result.output_path.exists()

        # Verify output image is valid
        output_image = Image.open(result.output_path)
        assert output_image.size == sample_image.size
        assert output_image.mode == 'RGB'

    def test_pipeline_with_disabled_stages(self, temp_dir, sample_image):
        """Test pipeline with some stages disabled."""
        recipe_content = """
name: "Minimal Test"
stages:
  - color_grading
material_response:
  enabled: false
color_grading:
  enabled: true
  contrast: 1.0
photo_finishing:
  enabled: false
output:
  format: "jpeg"
  quality: 80
"""
        recipe_path = temp_dir / "minimal_recipe.yaml"
        recipe_path.write_text(recipe_content)

        input_path = temp_dir / "input.jpg"
        sample_image.save(input_path, quality=95)

        from transformation_portal.pipeline_unified import UnifiedPipeline

        pipeline = UnifiedPipeline.from_recipe(recipe_path)
        pipeline.recipe['_output_dir'] = str(temp_dir / "output")

        result = pipeline.process_single(input_path)

        assert result.success
        assert 'color_grading' in result.stages_executed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
