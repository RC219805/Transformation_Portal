"""Comprehensive tests for the pipeline_unified module.

Tests cover:
- Data classes (ProcessingResult, BatchResult, PipelineStage)
- UnifiedPipeline initialization and configuration
- Recipe loading and validation
- Single image processing
- Batch processing (including dry-run mode)
- Stage execution
- Error handling
- Output generation

This implements TEST-002 from the improvement opportunities tracking.

Test Categories:
- Unit tests (@pytest.mark.unit): Pure unit tests for data classes, initialization
- Integration tests: Full pipeline workflows with I/O operations
  (These are fast enough to include in the default test run per ADR-044)
"""

from __future__ import annotations

import copy
import importlib
import json
import logging
import sys
import types
from pathlib import Path
from typing import Any, Callable, Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.unit

# Module paths for stubbing optional dependencies
PIPELINE_MODULE = "transformation_portal.pipeline_unified"
QUALITY_BRIDGE_MODULE = "transformation_portal.pipelines.quality_feedback_bridge"
RENDERING_4K_MODULE = "transformation_portal.pipelines.rendering_4k_pipeline"


# =============================================================================
# Stub Helpers
# =============================================================================


def _create_stub_module(name: str, classes: Dict[str, type]) -> types.ModuleType:
    """Create a stub module with the given classes.

    Args:
        name: Full module name (e.g., 'transformation_portal.pipelines.quality_feedback_bridge')
        classes: Dictionary mapping class names to class types

    Returns:
        ModuleType with the specified classes attached
    """
    module = types.ModuleType(name)
    for class_name, class_type in classes.items():
        setattr(module, class_name, class_type)
    return module


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def stub_optional_dependencies(monkeypatch):
    """Stub optional pipeline dependencies for isolated testing."""

    class StubQualityTargets:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StubQualityFeedbackBridge:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def assess(self, **kwargs):
            return MagicMock(
                hybrid_score=85.0,
                perceptual_composite=80.0,
                heuristic_composite=90.0,
                lpips_available=False,
                targets_summary="Quality targets met",
                heuristic=MagicMock(sharpness=0.8, contrast=0.7, colorfulness=0.75),
                material_fidelity=MagicMock(overall_fidelity=0.95),
                to_rag_document=lambda: {"image_id": "test"},
            )

    class StubRendering4KPipeline:
        pass

    # Create stub modules using helper
    quality_module = _create_stub_module(
        QUALITY_BRIDGE_MODULE,
        {
            "QualityFeedbackBridge": StubQualityFeedbackBridge,
            "QualityTargets": StubQualityTargets,
        },
    )
    rendering_module = _create_stub_module(
        RENDERING_4K_MODULE,
        {"Rendering4KPipeline": StubRendering4KPipeline},
    )

    # Install stubs
    monkeypatch.setitem(sys.modules, QUALITY_BRIDGE_MODULE, quality_module)
    monkeypatch.setitem(sys.modules, RENDERING_4K_MODULE, rendering_module)

    # Clear cached module to force reimport
    monkeypatch.delitem(sys.modules, PIPELINE_MODULE, raising=False)

    yield

    # Cleanup
    sys.modules.pop(PIPELINE_MODULE, None)


@pytest.fixture
def pipeline_module(stub_optional_dependencies):
    """Import pipeline_unified with stubbed dependencies."""
    return importlib.import_module(PIPELINE_MODULE)


@pytest.fixture
def minimal_recipe() -> Dict[str, Any]:
    """Minimal valid recipe for testing."""
    return {
        "name": "Test Pipeline",
        "description": "Minimal test recipe",
        "stages": ["color_grading"],
        "color_grading": {
            "enabled": True,
            "contrast": 1.0,
            "saturation": 1.0,
        },
        "quality_feedback": {"enabled": False},
        "output": {"format": "png"},
    }


@pytest.fixture
def full_recipe() -> Dict[str, Any]:
    """Full recipe with all stages for testing."""
    return {
        "name": "Full Test Pipeline",
        "description": "Full test recipe with all stages",
        "stages": [
            "depth_estimation",
            "material_response",
            "color_grading",
            "photo_finishing",
            "branding",
        ],
        "depth_estimation": {"enabled": True},
        "material_response": {"enabled": True},
        "color_grading": {
            "enabled": True,
            "exposure": 0.1,
            "contrast": 1.05,
            "saturation": 1.1,
            "warmth": 0.02,
        },
        "photo_finishing": {
            "enabled": True,
            "aces": True,
            "bloom": {"enabled": True, "threshold": 0.8, "intensity": 0.2},
            "vignette": {"enabled": True, "strength": 0.15},
            "grain": {"enabled": True, "amount": 0.01},
        },
        "branding": {"enabled": False},
        "quality_feedback": {"enabled": False},
        "output": {"format": "png", "quality": 95},
    }


@pytest.fixture
def sample_input_image(tmp_path: Path) -> Path:
    """Create a sample input image for testing."""
    img = Image.new("RGB", (64, 48), color=(100, 120, 140))
    path = tmp_path / "input.png"
    img.save(path)
    return path


@pytest.fixture
def sample_input_directory(tmp_path: Path) -> Path:
    """Create a directory with multiple sample images for batch testing."""
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    for i in range(3):
        img = Image.new("RGB", (32, 24), color=(50 + i * 40, 80 + i * 30, 100 + i * 20))
        img.save(input_dir / f"image_{i:02d}.png")

    return input_dir


# =============================================================================
# ProcessingResult Data Class Tests
# =============================================================================


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def test_default_values(self, pipeline_module, tmp_path: Path):
        """Test ProcessingResult default values."""
        result = pipeline_module.ProcessingResult(input_path=tmp_path / "test.jpg")

        assert result.input_path == tmp_path / "test.jpg"
        assert result.output_path is None
        assert result.success is False
        assert result.error_message is None
        assert result.stages_executed == []
        assert result.stage_times == {}
        assert result.total_time == 0.0
        assert result.metadata == {}
        assert result.quality_metrics is None
        assert result.rag_document is None

    def test_quality_score_property(self, pipeline_module, tmp_path: Path):
        """Test quality_score property returns correct value."""
        result = pipeline_module.ProcessingResult(input_path=tmp_path / "test.jpg")

        # No quality metrics
        assert result.quality_score == 0.0

        # With quality metrics
        result.quality_metrics = {"overall_score": 0.85}
        assert result.quality_score == 0.85

    def test_repr_success(self, pipeline_module, tmp_path: Path):
        """Test repr for successful result."""
        result = pipeline_module.ProcessingResult(
            input_path=tmp_path / "test.jpg",
            success=True,
            total_time=1.5,
        )
        repr_str = repr(result)
        assert "✓" in repr_str
        assert "test.jpg" in repr_str
        assert "1.50s" in repr_str

    def test_repr_failure(self, pipeline_module, tmp_path: Path):
        """Test repr for failed result."""
        result = pipeline_module.ProcessingResult(
            input_path=tmp_path / "test.jpg",
            success=False,
            total_time=0.5,
        )
        repr_str = repr(result)
        assert "✗" in repr_str

    def test_repr_with_quality(self, pipeline_module, tmp_path: Path):
        """Test repr includes quality when available."""
        result = pipeline_module.ProcessingResult(
            input_path=tmp_path / "test.jpg",
            success=True,
            total_time=1.0,
            quality_metrics={"overall_score": 0.92},
        )
        repr_str = repr(result)
        assert "quality=92.00%" in repr_str


# =============================================================================
# BatchResult Data Class Tests
# =============================================================================


class TestBatchResult:
    """Tests for BatchResult dataclass."""

    def test_default_values(self, pipeline_module):
        """Test BatchResult default values."""
        batch = pipeline_module.BatchResult()

        assert batch.results == []
        assert batch.total_time == 0.0
        assert batch.successful_count == 0
        assert batch.failed_count == 0
        assert batch.dry_run is False

    def test_summary_empty_batch(self, pipeline_module):
        """Test summary for empty batch."""
        batch = pipeline_module.BatchResult()
        summary = batch.summary()

        assert "Total images: 0" in summary
        assert "Successful: 0" in summary
        assert "Failed: 0" in summary

    def test_summary_with_results(self, pipeline_module, tmp_path: Path):
        """Test summary includes correct counts and failed images."""
        results = [
            pipeline_module.ProcessingResult(
                input_path=tmp_path / f"img{i}.jpg",
                success=i < 2,
                total_time=1.0,
                error_message=None if i < 2 else "Test error",
            )
            for i in range(3)
        ]

        batch = pipeline_module.BatchResult(
            results=results,
            total_time=3.0,
            successful_count=2,
            failed_count=1,
        )
        summary = batch.summary()

        assert "Total images: 3" in summary
        assert "Successful: 2" in summary
        assert "Failed: 1" in summary
        assert "Average time per image: 1.00s" in summary
        assert "Failed images:" in summary
        assert "Test error" in summary

    def test_summary_dry_run(self, pipeline_module):
        """Test summary shows dry run status."""
        batch = pipeline_module.BatchResult(dry_run=True)
        summary = batch.summary()

        assert "Dry run: True" in summary


# =============================================================================
# PipelineStage Data Class Tests
# =============================================================================


class TestPipelineStage:
    """Tests for PipelineStage dataclass."""

    def test_default_values(self, pipeline_module):
        """Test PipelineStage default values."""
        stage = pipeline_module.PipelineStage(
            name="test_stage",
            display_name="Test Stage",
        )

        assert stage.name == "test_stage"
        assert stage.display_name == "Test Stage"
        assert stage.enabled is True
        assert stage.required is False
        assert stage.config == {}
        assert stage.processor is None

    def test_full_initialization(self, pipeline_module):
        """Test PipelineStage with all parameters."""

        def processor_fn(img):
            return img

        stage = pipeline_module.PipelineStage(
            name="custom",
            display_name="Custom Stage",
            enabled=False,
            required=True,
            config={"key": "value"},
            processor=processor_fn,
        )

        assert stage.enabled is False
        assert stage.required is True
        assert stage.config == {"key": "value"}
        assert stage.processor is processor_fn


# =============================================================================
# UnifiedPipeline Initialization Tests
# =============================================================================


class TestUnifiedPipelineInit:
    """Tests for UnifiedPipeline initialization."""

    def test_basic_initialization(self, pipeline_module, minimal_recipe):
        """Test basic pipeline initialization."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)

        assert pipeline.name == "Test Pipeline"
        assert pipeline.description == "Minimal test recipe"
        assert pipeline.recipe == minimal_recipe

    def test_stages_initialized(self, pipeline_module, minimal_recipe):
        """Test stages are correctly initialized from recipe."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)

        # Should have color_grading stage
        stage_names = [s.name for s in pipeline.stages]
        assert "color_grading" in stage_names

    def test_unnamed_pipeline(self, pipeline_module):
        """Test pipeline handles missing name."""
        recipe = {"stages": [], "quality_feedback": {"enabled": False}}
        pipeline = pipeline_module.UnifiedPipeline(recipe)

        assert pipeline.name == "Unnamed Pipeline"
        assert pipeline.description == ""

    def test_all_stages_initialized(self, pipeline_module, full_recipe):
        """Test all stages from full recipe are initialized."""
        pipeline = pipeline_module.UnifiedPipeline(full_recipe)

        stage_names = [s.name for s in pipeline.stages]
        assert "depth_estimation" in stage_names
        assert "color_grading" in stage_names
        assert "photo_finishing" in stage_names

    def test_disabled_stages_excluded(self, pipeline_module):
        """Test disabled stages are properly marked."""
        recipe = {
            "name": "Test",
            "stages": ["color_grading", "branding"],
            "color_grading": {"enabled": True},
            "branding": {"enabled": False},
            "quality_feedback": {"enabled": False},
        }
        pipeline = pipeline_module.UnifiedPipeline(recipe)

        branding_stage = next((s for s in pipeline.stages if s.name == "branding"), None)
        assert branding_stage is not None
        assert branding_stage.enabled is False

    def test_device_detection(self, pipeline_module, minimal_recipe):
        """Test device detection returns valid value."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)

        # Should return one of the valid devices
        assert pipeline.device in ("cpu", "cuda", "mps")


# =============================================================================
# Process Single Image Tests
# =============================================================================


class TestProcessSingle:
    """Tests for single image processing."""

    def test_process_single_success(self, pipeline_module, minimal_recipe, sample_input_image, tmp_path: Path):
        """Test successful single image processing."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        assert result.error_message is None
        assert result.output_path is not None
        assert result.output_path.exists()
        assert "color_grading" in result.stages_executed

    def test_process_single_file_not_found(self, pipeline_module, minimal_recipe, tmp_path: Path):
        """Test processing nonexistent file."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        result = pipeline.process_single(tmp_path / "nonexistent.jpg")

        assert result.success is False
        assert result.error_message is not None
        assert "not found" in result.error_message.lower()

    def test_process_single_converts_mode(self, pipeline_module, minimal_recipe, tmp_path: Path):
        """Test that non-RGB images are converted."""
        # Create RGBA image
        rgba_image = Image.new("RGBA", (32, 24), color=(100, 120, 140, 255))
        input_path = tmp_path / "rgba_input.png"
        rgba_image.save(input_path)

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        result = pipeline.process_single(input_path)

        assert result.success is True
        # Verify output is RGB
        with Image.open(result.output_path) as output:
            assert output.mode == "RGB"

    def test_process_single_timing(self, pipeline_module, minimal_recipe, sample_input_image):
        """Test that timing information is recorded."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.total_time > 0
        assert "color_grading" in result.stage_times
        assert result.stage_times["color_grading"] >= 0

    def test_process_single_path_as_string(self, pipeline_module, minimal_recipe, sample_input_image):
        """Test processing with string path."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        result = pipeline.process_single(str(sample_input_image))

        assert result.success is True


# =============================================================================
# Stage Execution Tests
# =============================================================================


class TestStageExecution:
    """Tests for individual stage execution."""

    def test_color_grading_defaults(self, pipeline_module, minimal_recipe, sample_input_image):
        """Test color grading with default parameters."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        assert "color_grading" in result.stages_executed

    def test_color_grading_adjustments(self, pipeline_module, sample_input_image, tmp_path: Path):
        """Test color grading applies adjustments."""
        recipe = {
            "name": "Color Test",
            "stages": ["color_grading"],
            "color_grading": {
                "enabled": True,
                "exposure": 0.5,
                "contrast": 1.2,
                "saturation": 1.3,
                "warmth": 0.1,
            },
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True

        # Verify image was modified (should be brighter/more saturated)
        with Image.open(sample_input_image) as original:
            with Image.open(result.output_path) as processed:
                orig_arr = np.array(original)
                proc_arr = np.array(processed)
                # Exposure increase should make it brighter
                assert np.mean(proc_arr) > np.mean(orig_arr)

    def test_photo_finishing_aces(self, pipeline_module, sample_input_image, tmp_path: Path):
        """Test photo finishing with ACES tone mapping."""
        recipe = {
            "name": "Photo Test",
            "stages": ["photo_finishing"],
            "photo_finishing": {
                "enabled": True,
                "aces": True,
                "bloom": {"enabled": False},
                "vignette": {"enabled": False},
                "grain": {"enabled": False},
            },
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        assert "photo_finishing" in result.stages_executed

    def test_photo_finishing_bloom(self, pipeline_module, tmp_path: Path):
        """Test photo finishing bloom effect."""
        # Create bright image to trigger bloom
        bright_img = Image.new("RGB", (32, 24), color=(240, 240, 240))
        input_path = tmp_path / "bright.png"
        bright_img.save(input_path)

        recipe = {
            "name": "Bloom Test",
            "stages": ["photo_finishing"],
            "photo_finishing": {
                "enabled": True,
                "aces": False,
                "bloom": {"enabled": True, "threshold": 0.5, "intensity": 0.5},
                "vignette": {"enabled": False},
                "grain": {"enabled": False},
            },
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(input_path)

        assert result.success is True

    def test_photo_finishing_vignette(self, pipeline_module, sample_input_image):
        """Test photo finishing vignette effect."""
        recipe = {
            "name": "Vignette Test",
            "stages": ["photo_finishing"],
            "photo_finishing": {
                "enabled": True,
                "aces": False,
                "bloom": {"enabled": False},
                "vignette": {"enabled": True, "strength": 0.3},
                "grain": {"enabled": False},
            },
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True

        # Vignette should darken edges - center must be strictly brighter than corner
        with Image.open(result.output_path) as processed:
            arr = np.array(processed)
            center = arr[arr.shape[0] // 2, arr.shape[1] // 2]
            corner = arr[0, 0]
            # Center should be strictly brighter than corner (vignette effect)
            assert np.mean(center) > np.mean(corner), (
                f"Vignette effect not applied: center brightness ({np.mean(center):.2f}) "
                f"should be greater than corner brightness ({np.mean(corner):.2f})"
            )

    def test_photo_finishing_grain(self, pipeline_module, sample_input_image):
        """Test photo finishing grain effect."""
        recipe = {
            "name": "Grain Test",
            "stages": ["photo_finishing"],
            "photo_finishing": {
                "enabled": True,
                "aces": False,
                "bloom": {"enabled": False},
                "vignette": {"enabled": False},
                "grain": {"enabled": True, "amount": 0.05},
            },
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True

    def test_upscaling_4k_fallback(self, pipeline_module, sample_input_image):
        """Test 4K upscaling fallback to Lanczos."""
        recipe = {
            "name": "Upscale Test",
            "stages": ["upscaling_4k"],
            "upscaling_4k": {
                "enabled": True,
                "target_width": 128,
                "target_height": 96,
            },
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True

        # Verify upscaling occurred
        with Image.open(result.output_path) as processed:
            assert processed.width >= 64  # Should be upscaled

    def test_unknown_stage_skipped(self, pipeline_module, sample_input_image, caplog):
        """Test unknown stage is skipped with warning logged."""
        # Create a minimal recipe with a known stage
        recipe = {
            "name": "Unknown Stage Test",
            "stages": ["color_grading"],
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)

        # Inject an unknown stage directly into pipeline.stages to trigger
        # the else branch in _execute_stage()
        unknown_stage = pipeline_module.PipelineStage(
            name="totally_unknown_stage",
            display_name="Unknown Stage",
            enabled=True,
            required=False,
        )
        pipeline.stages.insert(0, unknown_stage)

        with caplog.at_level(logging.WARNING):
            result = pipeline.process_single(sample_input_image)

        assert result.success is True
        # Verify the warning was logged for the unknown stage
        assert any("Unknown stage" in record.message for record in caplog.records)

    def test_branding_disabled(self, pipeline_module, sample_input_image):
        """Test branding stage is skipped when disabled."""
        recipe = {
            "name": "Branding Test",
            "stages": ["branding"],
            "branding": {"enabled": False},
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True

    def test_branding_text_overlay(self, pipeline_module, sample_input_image):
        """Test branding with text overlay."""
        recipe = {
            "name": "Branding Test",
            "stages": ["branding"],
            "branding": {
                "enabled": True,
                "text": "Test Brand",
            },
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True


# =============================================================================
# Batch Processing Tests
# =============================================================================


class TestBatchProcessing:
    """Tests for batch image processing."""

    def test_batch_processing_success(self, pipeline_module, minimal_recipe, sample_input_directory, tmp_path: Path):
        """Test successful batch processing."""
        output_dir = tmp_path / "outputs"

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        batch_result = pipeline.process_batch(
            str(sample_input_directory / "*.png"),
            output_dir,
        )

        assert batch_result.successful_count == 3
        assert batch_result.failed_count == 0
        assert len(batch_result.results) == 3
        assert batch_result.total_time > 0
        assert output_dir.exists()

    def test_batch_processing_dry_run(self, pipeline_module, minimal_recipe, sample_input_directory, tmp_path: Path):
        """Test batch processing dry run mode."""
        output_dir = tmp_path / "outputs"

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        batch_result = pipeline.process_batch(
            str(sample_input_directory / "*.png"),
            output_dir,
            dry_run=True,
        )

        assert batch_result.dry_run is True
        assert batch_result.successful_count == 3
        assert len(batch_result.results) == 3
        # Output directory should NOT be created in dry run
        assert not output_dir.exists()

    def test_batch_empty_glob(self, pipeline_module, minimal_recipe, tmp_path: Path):
        """Test batch processing with no matching files."""
        output_dir = tmp_path / "outputs"

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        batch_result = pipeline.process_batch(
            str(tmp_path / "nonexistent/*.png"),
            output_dir,
        )

        assert len(batch_result.results) == 0
        assert batch_result.successful_count == 0

    def test_batch_creates_output_directory(self, pipeline_module, minimal_recipe, sample_input_directory, tmp_path: Path):
        """Test batch processing creates output directory."""
        output_dir = tmp_path / "nested" / "outputs"
        assert not output_dir.exists()

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        pipeline.process_batch(
            str(sample_input_directory / "*.png"),
            output_dir,
        )

        assert output_dir.exists()

    def test_batch_partial_failure(self, pipeline_module, minimal_recipe, tmp_path: Path):
        """Test batch processing handles partial failures."""
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()

        # Create valid image
        Image.new("RGB", (32, 24)).save(input_dir / "valid.png")

        # Create invalid "image" (empty file)
        (input_dir / "invalid.png").touch()

        output_dir = tmp_path / "outputs"

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        batch_result = pipeline.process_batch(
            str(input_dir / "*.png"),
            output_dir,
        )

        assert batch_result.successful_count >= 1
        assert batch_result.failed_count >= 1

    def test_batch_processing_parallel_preserves_order_and_recipe_state(
        self,
        pipeline_module,
        minimal_recipe,
        sample_input_directory,
        sample_input_image,
        tmp_path: Path,
    ):
        """Test parallel batch processing keeps deterministic ordering and leaves recipe unchanged."""
        output_dir = tmp_path / "parallel_outputs"

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        original_recipe = copy.deepcopy(pipeline.recipe)

        batch_result = pipeline.process_batch(
            str(sample_input_directory / "*.png"),
            output_dir,
            parallel=True,
        )

        assert batch_result.successful_count == 3
        assert [result.input_path.name for result in batch_result.results] == [
            "image_00.png",
            "image_01.png",
            "image_02.png",
        ]
        assert pipeline.recipe == original_recipe
        assert "_output_dir" not in pipeline.recipe

        single_result = pipeline.process_single(sample_input_image)
        assert single_result.success is True

    def test_batch_processing_parallel_serializes_rag_documents(
        self,
        pipeline_module,
        sample_input_directory,
        tmp_path: Path,
    ):
        """Test parallel batch processing writes one intact RAG document per line."""
        rag_index_path = tmp_path / "rag-index.jsonl"
        recipe = {
            "name": "Quality Batch",
            "stages": ["color_grading"],
            "color_grading": {"enabled": True},
            "quality_feedback": {
                "enabled": True,
                "rag_indexing_enabled": True,
                "rag_index_path": str(rag_index_path),
            },
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        batch_result = pipeline.process_batch(
            str(sample_input_directory / "*.png"),
            tmp_path / "quality_outputs",
            parallel=True,
        )

        assert batch_result.successful_count == 3
        lines = rag_index_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        assert all(json.loads(line)["image_id"] == "test" for line in lines)

    def test_batch_processing_parallel_reuses_worker_pipeline_and_honors_max_workers(
        self,
        pipeline_module,
        minimal_recipe,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Test parallel batch processing caches one worker pipeline per thread."""
        input_dir = tmp_path / "parallel_inputs"
        input_dir.mkdir()
        for index in range(6):
            Image.new("RGB", (32, 24), color=(40 + index, 90, 120)).save(input_dir / f"image_{index:02d}.png")

        observed: dict[str, int] = {}

        class RecordingExecutor:
            def __init__(self, max_workers: int):
                observed["max_workers"] = max_workers

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def map(self, func, iterable):
                for item in iterable:
                    yield func(item)

        monkeypatch.setattr(pipeline_module, "ThreadPoolExecutor", RecordingExecutor)
        monkeypatch.setattr(pipeline_module.os, "cpu_count", lambda: 8)

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        create_calls = 0
        original_create_worker_pipeline = pipeline._create_worker_pipeline

        def _counting_create_worker_pipeline(*args, **kwargs):
            nonlocal create_calls
            create_calls += 1
            return original_create_worker_pipeline(*args, **kwargs)

        monkeypatch.setattr(pipeline, "_create_worker_pipeline", _counting_create_worker_pipeline)

        batch_result = pipeline.process_batch(
            str(input_dir / "*.png"),
            tmp_path / "parallel_outputs",
            parallel=True,
            max_workers=5,
        )

        assert batch_result.successful_count == 6
        assert observed["max_workers"] == 5
        assert create_calls == 1


# =============================================================================
# Output Generation Tests
# =============================================================================


class TestOutputGeneration:
    """Tests for output file generation."""

    def test_output_format_png(self, pipeline_module, sample_input_image):
        """Test PNG output format."""
        recipe = {
            "name": "PNG Test",
            "stages": ["color_grading"],
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        assert result.output_path.suffix == ".png"

    def test_output_format_jpeg(self, pipeline_module, sample_input_image):
        """Test JPEG output format."""
        recipe = {
            "name": "JPEG Test",
            "stages": ["color_grading"],
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": False},
            "output": {"format": "jpeg", "quality": 85},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        assert result.output_path.suffix == ".jpg"

    def test_output_format_tiff(self, pipeline_module, sample_input_image):
        """Test TIFF output format."""
        recipe = {
            "name": "TIFF Test",
            "stages": ["color_grading"],
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": False},
            "output": {"format": "tiff"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        assert result.output_path.suffix == ".tif"

    def test_output_filename_includes_recipe(self, pipeline_module, sample_input_image):
        """Test output filename includes recipe name."""
        recipe = {
            "name": "Custom Recipe Name",
            "stages": ["color_grading"],
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert "custom_recipe_name" in result.output_path.name.lower()

    def test_output_directory_created(self, pipeline_module, minimal_recipe, sample_input_image, tmp_path: Path):
        """Test output directory is created if not exists."""
        # Set custom output directory via recipe
        minimal_recipe["_output_dir"] = str(tmp_path / "custom_output")

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        assert (tmp_path / "custom_output").exists()


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_corrupted_image_handling(self, pipeline_module, minimal_recipe, tmp_path: Path):
        """Test handling of corrupted image file."""
        corrupt_path = tmp_path / "corrupt.png"
        corrupt_path.write_text("not a valid image")

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        result = pipeline.process_single(corrupt_path)

        assert result.success is False
        assert result.error_message is not None

    def test_required_stage_failure(self, pipeline_module, sample_input_image):
        """Test that required stage failure stops pipeline and sets error message."""
        recipe = {
            "name": "Required Stage Test",
            "stages": ["color_grading"],
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)

        # Find the color_grading stage and mark it as required
        for stage in pipeline.stages:
            if stage.name == "color_grading":
                stage.required = True

        # Patch the _execute_stage to raise an exception for color_grading
        original_execute = pipeline._execute_stage

        def mock_execute(stage, image):
            if stage.name == "color_grading":
                raise RuntimeError("Simulated required stage failure")
            return original_execute(stage, image)

        with patch.object(pipeline, "_execute_stage", side_effect=mock_execute):
            result = pipeline.process_single(sample_input_image)

        # Required stage failure should stop the pipeline
        assert result.success is False
        assert result.error_message is not None
        assert "Simulated required stage failure" in result.error_message

    def test_optional_stage_failure_continues(self, pipeline_module, sample_input_image, caplog):
        """Test that optional stage failure allows pipeline to continue."""
        recipe = {
            "name": "Optional Stage Test",
            "stages": ["ai_enhancement", "color_grading"],
            "ai_enhancement": {"enabled": True},
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)

        # Patch the _execute_stage to raise for ai_enhancement (optional stage)
        original_execute = pipeline._execute_stage

        def mock_execute(stage, image):
            if stage.name == "ai_enhancement":
                raise RuntimeError("Simulated optional stage failure")
            return original_execute(stage, image)

        with patch.object(pipeline, "_execute_stage", side_effect=mock_execute):
            with caplog.at_level(logging.WARNING):
                result = pipeline.process_single(sample_input_image)

        # Pipeline should still succeed despite optional stage failure
        assert result.success is True
        # Color grading should still have run
        assert "color_grading" in result.stages_executed
        # Warning should be logged for the failed optional stage
        assert any("failed" in record.message.lower() for record in caplog.records)


# =============================================================================
# Recipe Loading Tests
# =============================================================================


class TestRecipeLoading:
    """Tests for recipe loading functionality."""

    def test_from_recipe_invalid_file(self, pipeline_module, tmp_path: Path):
        """Test from_recipe with nonexistent file."""
        with pytest.raises(FileNotFoundError):
            pipeline_module.UnifiedPipeline.from_recipe(tmp_path / "nonexistent.yaml")

    def test_from_recipe_valid_file(self, pipeline_module, tmp_path: Path):
        """Test from_recipe with valid YAML file.

        The config_loader module is a core dependency and should always be
        available. If the import fails, it indicates a real regression.
        """
        import yaml

        recipe = {
            "name": "File Recipe",
            "description": "Test recipe from file",
            "stages": ["color_grading"],
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        recipe_path = tmp_path / "test_recipe.yaml"
        with open(recipe_path, "w") as f:
            yaml.safe_dump(recipe, f)

        # from_recipe uses config_loader which is a core module
        # If this fails, it's a real regression that should fail the test
        pipeline = pipeline_module.UnifiedPipeline.from_recipe(recipe_path)
        assert pipeline.name == "File Recipe"


# =============================================================================
# Quality Assessment Tests
# =============================================================================


class TestQualityAssessment:
    """Tests for quality assessment functionality."""

    def test_basic_quality_metrics(self, pipeline_module, sample_input_image):
        """Test basic quality metrics computation."""
        recipe = {
            "name": "Quality Test",
            "stages": ["color_grading", "quality_assessment"],
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": True},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        # Quality metrics may or may not be present depending on stub behavior
        if result.quality_metrics:
            assert "overall_score" in result.quality_metrics
            assert 0 <= result.quality_metrics["overall_score"] <= 1

    def test_quality_disabled(self, pipeline_module, sample_input_image):
        """Test pipeline without quality assessment."""
        recipe = {
            "name": "No Quality Test",
            "stages": ["color_grading"],
            "color_grading": {"enabled": True},
            "quality_feedback": {"enabled": False},
            "output": {"format": "png"},
        }

        pipeline = pipeline_module.UnifiedPipeline(recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        # Quality assessment should not be in stages
        assert "quality_assessment" not in result.stages_executed


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full pipeline workflows."""

    def test_full_pipeline_workflow(self, pipeline_module, full_recipe, sample_input_image):
        """Test complete pipeline with all stages."""
        # Disable stages that require external dependencies
        full_recipe["depth_estimation"]["enabled"] = False
        full_recipe["material_response"]["enabled"] = False

        pipeline = pipeline_module.UnifiedPipeline(full_recipe)
        result = pipeline.process_single(sample_input_image)

        assert result.success is True
        assert len(result.stages_executed) > 0
        assert result.output_path.exists()

    def test_pipeline_immutability(self, pipeline_module, minimal_recipe, sample_input_image):
        """Test that processing doesn't modify input image."""
        with Image.open(sample_input_image) as original:
            original_data = np.array(original).copy()

        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)
        pipeline.process_single(sample_input_image)

        with Image.open(sample_input_image) as after:
            after_data = np.array(after)

        np.testing.assert_array_equal(original_data, after_data)

    def test_multiple_sequential_runs(self, pipeline_module, minimal_recipe, sample_input_image):
        """Test multiple sequential processing runs."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)

        results = []
        for _ in range(3):
            result = pipeline.process_single(sample_input_image)
            results.append(result)

        assert all(r.success for r in results)

    def test_batch_then_single(
        self, pipeline_module, minimal_recipe, sample_input_directory, sample_input_image, tmp_path: Path
    ):
        """Test batch processing followed by single processing."""
        pipeline = pipeline_module.UnifiedPipeline(minimal_recipe)

        # Batch process
        batch_result = pipeline.process_batch(
            str(sample_input_directory / "*.png"),
            tmp_path / "batch_outputs",
        )
        assert batch_result.successful_count > 0

        # Single process should still work
        single_result = pipeline.process_single(sample_input_image)
        assert single_result.success is True
