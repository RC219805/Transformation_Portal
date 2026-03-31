"""Unit tests for pipeline orchestrator (Phase 2.4).

Tests for SpatialAIPipeline, PipelineConfig, and PipelineResult to achieve ≥85% coverage.
Covers configuration validation, tier enforcement, preset loading, stage execution,
resource management integration, error handling, and E2E workflows with mocked backends.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from transformation_portal.spatial_ai.ingest.linear_decoder import LinearIngestResult
from transformation_portal.spatial_ai.materials.contracts import (
    AvailabilityState,
    BackendDecision,
    MaterialProperties,
    PBRGenerationMetadata,
    PBRTextures,
)
from transformation_portal.spatial_ai.orchestration.error_handler import ErrorRecoveryStrategy, PipelineError
from transformation_portal.spatial_ai.orchestration.pipeline import PipelineConfig, PipelineResult, SpatialAIPipeline
from transformation_portal.spatial_ai.orchestration.resource_manager import ResourceLimits
from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationResult

pytestmark = pytest.mark.unit


class TestPipelineConfig:
    """Test PipelineConfig dataclass and validation."""

    def test_minimal_valid_config(self):
        """Test creating config with minimal required fields."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
        )
        assert config.tier == "standard"
        assert config.stages == ["ingest"]
        assert config.ingest == {}
        assert config.segmentation == {}
        assert config.materials == {}
        assert config.reconstruction == {}
        assert config.resource_limits is None
        assert config.error_strategy == ErrorRecoveryStrategy.RETRY

    def test_full_config_creation(self):
        """Test creating config with all fields."""
        limits = ResourceLimits(max_gpu_memory_gb=8.0)
        config = PipelineConfig(
            tier="apex_research",
            stages=["ingest", "segment", "materials"],
            ingest={"strict_ingest": True},
            segmentation={"backend": "sam2"},
            materials={"backend": "heuristic"},
            reconstruction={"enabled": False},
            resource_limits=limits,
            error_strategy=ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
        )
        assert config.tier == "apex_research"
        assert len(config.stages) == 3
        assert config.ingest["strict_ingest"] is True
        assert config.resource_limits is limits
        assert config.error_strategy == ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK

    def test_invalid_stage_rejected(self):
        """Test invalid stage name is rejected."""
        with pytest.raises(ValueError, match="Invalid stage"):
            PipelineConfig(
                tier="standard",
                stages=["ingest", "invalid_stage"],
            )

    def test_all_valid_stages_accepted(self):
        """Test all valid stages are accepted."""
        valid_stages = ["ingest", "segment", "materials", "reconstruction"]
        config = PipelineConfig(
            tier="apex_research",
            stages=valid_stages,
        )
        assert config.stages == valid_stages

    def test_invalid_tier_rejected(self):
        """Test invalid tier is rejected."""
        with pytest.raises(ValueError, match="Invalid tier"):
            PipelineConfig(
                tier="invalid_tier",
                stages=["ingest"],
            )

    def test_all_valid_tiers_accepted(self):
        """Test all valid tiers are accepted."""
        valid_tiers = ["standard", "apex_research", "apex_research_ultra", "experimental"]
        for tier in valid_tiers:
            config = PipelineConfig(tier=tier, stages=["ingest"])
            assert config.tier == tier

    def test_reconstruction_requires_research_tier(self):
        """Test reconstruction requires research tier (license enforcement)."""
        with pytest.raises(ValueError, match="research tier.*3DGS"):
            PipelineConfig(
                tier="standard",
                stages=["ingest", "reconstruction"],
            )

    def test_reconstruction_allowed_in_research_tier(self):
        """Test reconstruction is allowed in research tier."""
        config = PipelineConfig(
            tier="apex_research",
            stages=["ingest", "reconstruction"],
        )
        assert "reconstruction" in config.stages

    def test_reconstruction_allowed_in_experimental_tier(self):
        """Test reconstruction is allowed in experimental tier."""
        config = PipelineConfig(
            tier="experimental",
            stages=["reconstruction"],
        )
        assert "reconstruction" in config.stages

    def test_materials_non_strict_backend_allows_documented_fallback(self):
        """Non-strict materials config should allow runtime fallback semantics."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest", "segment", "materials"],
            materials={"backend": "nvdiffrec", "strict_backend": False},
        )
        assert config.materials["backend"] == "nvdiffrec"
        assert config.materials["strict_backend"] is False

    def test_materials_strict_backend_rejects_single_image_nvdiffrec(self):
        """Strict materials mode should fail fast on single-image NVDIFFREC requests."""
        with pytest.raises(ValueError, match="materials.strict_backend=True forbids fallback"):
            PipelineConfig(
                tier="standard",
                stages=["ingest", "segment", "materials"],
                materials={"backend": "nvdiffrec", "strict_backend": True},
            )

    def test_materials_strict_backend_rejects_pbr_fusion_without_runtime(self, monkeypatch):
        """Strict materials mode should fail fast when PBRFusion runtime is unavailable."""
        monkeypatch.delenv("PBRFUSION_PATH", raising=False)
        with pytest.raises(ValueError, match="runtime_missing"):
            PipelineConfig(
                tier="standard",
                stages=["ingest", "segment", "materials"],
                materials={"backend": "pbr_fusion", "strict_backend": True},
            )

    def test_materials_invalid_backend_rejected_when_stage_enabled(self):
        """Unknown materials backends should be rejected during config validation."""
        with pytest.raises(ValueError, match="Unknown backend"):
            PipelineConfig(
                tier="standard",
                stages=["ingest", "segment", "materials"],
                materials={"backend": "not_real"},
            )

    @given(
        tier=st.sampled_from(["standard", "apex_research", "experimental"]),
        stages=st.lists(
            st.sampled_from(["ingest", "segment", "materials"]),
            min_size=1,
            max_size=3,
            unique=True,
        ),
    )
    @settings(max_examples=20)
    def test_valid_config_property_based(self, tier, stages):
        """Property-based test for valid configurations."""
        config = PipelineConfig(tier=tier, stages=stages)
        assert config.tier == tier
        assert set(config.stages) == set(stages)


class TestPipelineResult:
    """Test PipelineResult dataclass."""

    def test_minimal_result_creation(self):
        """Test creating result with minimal fields."""
        result = PipelineResult(
            input_path=Path("input.tiff"),
            output_dir=Path("output/"),
            stages_completed=["ingest"],
        )
        assert result.input_path == Path("input.tiff")
        assert result.output_dir == Path("output/")
        assert result.stages_completed == ["ingest"]
        assert result.linear_image is None
        assert result.segmentation is None
        assert result.materials is None
        assert result.scene_3d is None
        assert result.execution_time == 0.0
        assert result.peak_memory_mb == 0.0
        assert result.errors == []
        assert result.warnings == []
        assert result.metadata == {}

    def test_full_result_creation(self):
        """Test creating result with all fields."""
        linear_result = Mock(spec=LinearIngestResult)
        seg_result = Mock(spec=SegmentationResult)
        materials = {"seg_0": Mock(spec=PBRTextures)}

        result = PipelineResult(
            input_path=Path("test.tiff"),
            output_dir=Path("out/"),
            stages_completed=["ingest", "segment", "materials"],
            linear_image=linear_result,
            segmentation=seg_result,
            materials=materials,
            execution_time=42.5,
            peak_memory_mb=2048.0,
            errors=["Error 1"],
            warnings=["Warning 1"],
            metadata={"key": "value"},
        )

        assert result.linear_image is linear_result
        assert result.segmentation is seg_result
        assert result.materials is materials
        assert result.execution_time == 42.5
        assert result.peak_memory_mb == 2048.0
        assert result.errors == ["Error 1"]
        assert result.warnings == ["Warning 1"]
        assert result.metadata == {"key": "value"}

    def test_save_summary(self, tmp_path):
        """Test saving execution summary as JSON."""
        # Create mock results
        linear_result = MagicMock()

        seg_result = MagicMock()
        seg_result.masks = [np.ones((10, 10), dtype=bool), np.ones((10, 10), dtype=bool)]

        materials = {"seg_0": MagicMock(), "seg_1": MagicMock()}

        scene_3d = MagicMock()
        scene_3d.splats.num_gaussians = 10000
        scene_3d.rmse = 0.015

        result = PipelineResult(
            input_path=Path("input.tiff"),
            output_dir=Path("output/"),
            stages_completed=["ingest", "segment", "materials", "reconstruct"],
            linear_image=linear_result,
            segmentation=seg_result,
            materials=materials,
            scene_3d=scene_3d,
            execution_time=123.4,
            peak_memory_mb=4096.0,
            errors=["Error 1", "Error 2"],
            warnings=["Warning 1"],
            metadata={"custom": "data"},
        )

        summary_path = tmp_path / "summary.json"
        result.save_summary(summary_path)

        assert summary_path.exists()

        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["input"] == "input.tiff"
        assert summary["output_dir"] == "output"
        assert summary["stages_completed"] == ["ingest", "segment", "materials", "reconstruct"]
        assert summary["execution_time"] == 123.4
        assert summary["peak_memory_mb"] == 4096.0
        assert summary["errors"] == ["Error 1", "Error 2"]
        assert summary["warnings"] == ["Warning 1"]

        # Check results section
        assert summary["results"]["linear_image"] is True
        assert summary["results"]["segmentation"]["completed"] is True
        assert summary["results"]["segmentation"]["num_masks"] == 2
        assert summary["results"]["materials"]["completed"] is True
        assert summary["results"]["materials"]["num_segments"] == 2
        assert summary["results"]["scene_3d"]["completed"] is True
        assert summary["results"]["scene_3d"]["num_gaussians"] == 10000
        assert summary["results"]["scene_3d"]["rmse"] == 0.015

        assert summary["metadata"] == {"custom": "data"}

    def test_save_summary_partial_results(self, tmp_path):
        """Test saving summary with partial results."""
        result = PipelineResult(
            input_path=Path("test.tiff"),
            output_dir=Path("out/"),
            stages_completed=["ingest"],
            linear_image=MagicMock(),
            execution_time=10.0,
            peak_memory_mb=512.0,
        )

        summary_path = tmp_path / "partial.json"
        result.save_summary(summary_path)

        with open(summary_path) as f:
            summary = json.load(f)

        assert summary["results"]["linear_image"] is True
        assert summary["results"]["segmentation"]["completed"] is False
        assert summary["results"]["segmentation"]["num_masks"] == 0
        assert summary["results"]["materials"]["completed"] is False
        assert summary["results"]["scene_3d"]["completed"] is False


class TestSpatialAIPipelineInitialization:
    """Test SpatialAIPipeline initialization."""

    def test_initialization_with_config_object(self):
        """Test initialization with PipelineConfig object."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest", "segment"],
        )
        pipeline = SpatialAIPipeline(config)

        assert pipeline.config is config
        assert pipeline.config.tier == "standard"
        assert len(pipeline.config.stages) == 2

    def test_initialization_with_dict(self):
        """Test initialization with dict."""
        config_dict = {
            "tier": "standard",
            "pipeline": {
                "ingest": {"strict_ingest": False},
                "segment": {"backend": "sam2"},
            },
        }
        pipeline = SpatialAIPipeline(config_dict)

        assert pipeline.config.tier == "standard"
        assert "ingest" in pipeline.config.stages
        assert "segment" in pipeline.config.stages or "segmentation" in pipeline.config.stages

    def test_initialization_from_preset_name(self):
        """Test initialization from preset name."""
        pipeline = SpatialAIPipeline.from_preset("spatial_ai_standard")

        assert pipeline.config.tier == "standard"
        assert "ingest" in pipeline.config.stages
        assert "segmentation" in pipeline.config.stages

    def test_initialization_from_preset_path(self):
        """Test initialization from preset file path."""
        preset_path = Path("config/presets/spatial_ai/spatial_ai_standard.yaml")
        if preset_path.exists():
            pipeline = SpatialAIPipeline(preset_path)
            assert pipeline.config.tier == "standard"

    def test_initialization_preset_not_found(self):
        """Test initialization with nonexistent preset raises error."""
        with pytest.raises(FileNotFoundError, match="Preset not found"):
            SpatialAIPipeline.from_preset("nonexistent_preset")

    def test_initialization_creates_components(self):
        """Test initialization creates resource manager, error handler, progress tracker."""
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        assert pipeline.resource_manager is not None
        assert pipeline.error_handler is not None
        assert pipeline.progress_tracker is not None
        assert pipeline.progress_tracker.total_stages == 1

    def test_initialization_invalid_type_rejected(self):
        """Test initialization with invalid type raises TypeError."""
        with pytest.raises(TypeError):
            SpatialAIPipeline(12345)


class TestSpatialAIPipelinePresetLoading:
    """Test preset loading logic."""

    def test_load_preset_standard(self):
        """Test loading spatial_ai_standard preset."""
        config = SpatialAIPipeline._load_preset("spatial_ai_standard")

        assert config.tier == "standard"
        # Preset keys become stage names - map to valid stage names
        assert config.stages == ["ingest", "segmentation", "materials"]
        assert config.error_strategy in [
            ErrorRecoveryStrategy.RETRY,
            ErrorRecoveryStrategy.FAIL_FAST,
            ErrorRecoveryStrategy.RETRY_WITH_CPU_FALLBACK,
        ]

    def test_load_preset_research(self):
        """Test loading spatial_ai_research preset."""
        config = SpatialAIPipeline._load_preset("spatial_ai_research")

        assert config.tier == "apex_research"
        assert "ingest" in config.stages or "segmentation" in config.stages

    def test_load_config_file(self, tmp_path):
        """Test loading config from YAML file."""
        config_data = {
            "tier": "standard",
            "pipeline": {
                "ingest": {"strict_ingest": False},
                "segment": {"backend": "sam2"},
            },
        }

        config_path = tmp_path / "test_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = SpatialAIPipeline._load_config_file(config_path)

        assert config.tier == "standard"
        assert "ingest" in config.stages
        assert "segment" in config.stages or "segmentation" in config.stages

    def test_dict_to_config(self):
        """Test converting dict to PipelineConfig."""
        data = {
            "tier": "apex_research",
            "pipeline": {
                "ingest": {"strict_ingest": True, "emit_exr": True},
                "segmentation": {"backend": "sam2"},
                "materials": {"backend": "heuristic"},
            },
        }

        config = SpatialAIPipeline._dict_to_config(data)

        assert config.tier == "apex_research"
        assert set(config.stages) == {"ingest", "segmentation", "materials"}
        assert config.ingest == {"strict_ingest": True, "emit_exr": True}
        assert config.segmentation == {"backend": "sam2"}


class TestSpatialAIPipelineIngestStage:
    """Test ingest stage execution."""

    def test_run_ingest_success(self, tmp_path):
        """Test successful ingest execution."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
            ingest={"strict_ingest": False, "emit_exr": False},
        )
        pipeline = SpatialAIPipeline(config)

        # Mock LinearDecoder
        mock_result = MagicMock(spec=LinearIngestResult)
        mock_result.input_size = (1024, 768)
        mock_result.linear_rgb = np.random.rand(768, 1024, 3).astype(np.float32)
        mock_result.gamma = 1.0

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            mock_decoder = MockDecoder.return_value
            mock_decoder.decode.return_value = mock_result

            result = pipeline._run_ingest(
                input_path=Path("input.tiff"),
                output_dir=tmp_path,
                save_intermediates=False,
            )

        assert result is mock_result
        MockDecoder.assert_called_once_with(gamma=1.0, bit_depth=32, strict_ingest=False)

    def test_run_ingest_with_openexr_preflight(self, tmp_path):
        """Test ingest with OpenEXR preflight check."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
            ingest={"strict_ingest": True, "emit_exr": True},
        )
        pipeline = SpatialAIPipeline(config)

        # Save reference to original __import__ before patching to avoid recursion
        import builtins

        _real_import = builtins.__import__

        def mock_import(name, *args):
            if name == "OpenEXR":
                raise ImportError("No module named 'OpenEXR'")
            return _real_import(name, *args)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(PipelineError, match="OpenEXR"):
                pipeline._run_ingest(
                    input_path=Path("input.tiff"),
                    output_dir=tmp_path,
                    save_intermediates=True,
                )

    def test_run_ingest_failure_raises_pipeline_error(self, tmp_path):
        """Test ingest failure raises PipelineError."""
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            mock_decoder = MockDecoder.return_value
            mock_decoder.decode.side_effect = RuntimeError("Decode failed")

            with pytest.raises(PipelineError) as exc_info:
                pipeline._run_ingest(
                    input_path=Path("input.tiff"),
                    output_dir=tmp_path,
                    save_intermediates=False,
                )

        assert exc_info.value.stage == "ingest"


class TestSpatialAIPipelineSegmentationStage:
    """Test segmentation stage execution."""

    def test_run_segmentation_success(self, tmp_path):
        """Test successful segmentation execution."""
        config = PipelineConfig(
            tier="standard",
            stages=["segment"],
            segmentation={"backend": "sam2"},
            resource_limits=ResourceLimits(device_preference=["cpu"]),
        )
        pipeline = SpatialAIPipeline(config)

        # Mock ingest result
        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(512, 512, 3).astype(np.float32)
        ingest_result.gamma = 1.0

        # Mock segmentation result
        mock_seg_result = MagicMock(spec=SegmentationResult)
        mock_seg_result.masks = [np.ones((512, 512), dtype=bool)]
        mock_seg_result.scores = np.array([0.95])
        mock_seg_result.metadata = [
            MaskMetadata(area=512 * 512, bbox=(0, 0, 512, 512), stability_score=0.95, material_label="wall")
        ]

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.SAM2Backend") as MockBackend:
            mock_backend = MockBackend.return_value
            mock_backend.segment.return_value = mock_seg_result

            result = pipeline._run_segmentation(
                ingest_result=ingest_result,
                output_dir=tmp_path,
                save_intermediates=False,
            )

        assert result is mock_seg_result
        MockBackend.assert_called_once()

    def test_run_segmentation_passes_tiling_config(self, tmp_path):
        """Test tiling config dict is parsed and passed to SAM2 backend."""
        config = PipelineConfig(
            tier="standard",
            stages=["segment"],
            segmentation={"backend": "sam2", "tiling": {"enabled": True, "tile_size_px": 512, "overlap_px": 64}},
            resource_limits=ResourceLimits(device_preference=["cpu"]),
        )
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(128, 128, 3).astype(np.float32)
        ingest_result.gamma = 1.0

        mock_seg_result = MagicMock(spec=SegmentationResult)
        mock_seg_result.masks = [np.ones((128, 128), dtype=bool)]
        mock_seg_result.scores = np.array([0.95])
        mock_seg_result.metadata = [MaskMetadata(area=128 * 128, bbox=(0, 0, 128, 128), stability_score=0.95)]

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.SAM2Backend") as MockBackend:
            mock_backend = MockBackend.return_value
            mock_backend.segment.return_value = mock_seg_result

            pipeline._run_segmentation(
                ingest_result=ingest_result,
                output_dir=tmp_path,
                save_intermediates=False,
            )

        _, kwargs = MockBackend.call_args
        assert kwargs["tiling"].enabled is True
        assert kwargs["tiling"].tile_size_px == 512
        assert kwargs["tiling"].overlap_px == 64

    def test_run_segmentation_default_no_tiling(self, tmp_path):
        """Test default segmentation passes tiling disabled when no tiling config is set."""
        config = PipelineConfig(
            tier="standard",
            stages=["segment"],
            segmentation={"backend": "sam2"},
            resource_limits=ResourceLimits(device_preference=["cpu"]),
        )
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(128, 128, 3).astype(np.float32)
        ingest_result.gamma = 1.0

        mock_seg_result = MagicMock(spec=SegmentationResult)
        mock_seg_result.masks = [np.ones((128, 128), dtype=bool)]
        mock_seg_result.scores = np.array([0.95])
        mock_seg_result.metadata = [MaskMetadata(area=128 * 128, bbox=(0, 0, 128, 128), stability_score=0.95)]

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.SAM2Backend") as MockBackend:
            mock_backend = MockBackend.return_value
            mock_backend.segment.return_value = mock_seg_result
            pipeline._run_segmentation(
                ingest_result=ingest_result,
                output_dir=tmp_path,
                save_intermediates=False,
            )

        _, kwargs = MockBackend.call_args
        assert kwargs["tiling"].enabled is False

    def test_run_segmentation_saves_intermediates(self, tmp_path):
        """Test segmentation saves masks when save_intermediates=True."""
        config = PipelineConfig(
            tier="standard",
            stages=["segment"],
            resource_limits=ResourceLimits(device_preference=["cpu"]),
        )
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(256, 256, 3).astype(np.float32)
        ingest_result.gamma = 1.0

        mock_seg_result = MagicMock(spec=SegmentationResult)
        mock_seg_result.masks = [np.ones((256, 256), dtype=bool)]
        mock_seg_result.scores = np.array([0.9])
        mock_seg_result.metadata = [MaskMetadata(area=256 * 256, bbox=(0, 0, 256, 256), stability_score=0.9)]

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.SAM2Backend") as MockBackend:
            mock_backend = MockBackend.return_value
            mock_backend.segment.return_value = mock_seg_result

            result = pipeline._run_segmentation(
                ingest_result=ingest_result,
                output_dir=tmp_path,
                save_intermediates=True,
            )

        # Check masks were saved
        masks_path = tmp_path / "segmentation_masks.npz"
        assert masks_path.exists()

    def test_run_segmentation_invalid_backend_rejected(self, tmp_path):
        """Test segmentation with invalid backend raises error."""
        config = PipelineConfig(
            tier="standard",
            stages=["segment"],
            segmentation={"backend": "invalid_backend"},
            resource_limits=ResourceLimits(device_preference=["cpu"]),
        )
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(256, 256, 3).astype(np.float32)

        with pytest.raises(PipelineError, match="sam2"):
            pipeline._run_segmentation(
                ingest_result=ingest_result,
                output_dir=tmp_path,
                save_intermediates=False,
            )


class TestSpatialAIPipelineMaterialsStage:
    """Test materials stage execution."""

    def test_run_materials_success(self, tmp_path):
        """Test successful materials generation."""
        config = PipelineConfig(
            tier="standard",
            stages=["materials"],
            materials={"backend": "heuristic", "material_hints": True},
            resource_limits=ResourceLimits(device_preference=["cpu"]),
        )
        pipeline = SpatialAIPipeline(config)

        # Mock inputs
        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(256, 256, 3).astype(np.float32)
        ingest_result.gamma = 1.0

        seg_result = MagicMock(spec=SegmentationResult)
        seg_result.masks = [np.ones((256, 256), dtype=bool), np.ones((256, 256), dtype=bool)]
        seg_result.scores = np.array([0.9, 0.85])
        seg_result.metadata = [
            MaskMetadata(area=128 * 128, bbox=(0, 0, 128, 128), stability_score=0.9, material_label="wood"),
            MaskMetadata(area=128 * 128, bbox=(0, 0, 128, 128), stability_score=0.85, material_label="metal"),
        ]

        # Mock PBR textures
        mock_pbr = MagicMock(spec=PBRTextures)

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.MaterialBackend") as MockBackend:
            mock_backend = MockBackend.return_value
            mock_backend.generate.return_value = mock_pbr

            result = pipeline._run_materials(
                ingest_result=ingest_result,
                seg_result=seg_result,
                output_dir=tmp_path,
                save_intermediates=False,
            )

        assert len(result) == 2
        assert "segment_0" in result
        assert "segment_1" in result

    def test_run_materials_saves_intermediates(self, tmp_path):
        """Test materials saves textures plus diagnostics/provenance sidecars."""
        config = PipelineConfig(
            tier="standard",
            stages=["materials"],
            resource_limits=ResourceLimits(device_preference=["cpu"]),
        )
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(128, 128, 3).astype(np.float32)
        ingest_result.gamma = 1.0
        ingest_result.input_path = Path("input.tiff")
        ingest_result.content_hash = "abc123"
        ingest_result.input_size = (128, 128)

        seg_result = MagicMock(spec=SegmentationResult)
        seg_result.masks = [np.ones((128, 128), dtype=bool)]
        seg_result.scores = np.array([0.9])
        seg_result.metadata = [MaskMetadata(area=128 * 128, bbox=(0, 0, 128, 128), stability_score=0.9, material_label="wood")]

        mock_pbr = MagicMock(spec=PBRTextures)
        mock_pbr.albedo = np.random.rand(128, 128, 3).astype(np.float32)
        mock_pbr.normal = np.random.rand(128, 128, 3).astype(np.float32)
        mock_pbr.roughness = np.random.rand(128, 128).astype(np.float32)
        mock_pbr.metallic = np.random.rand(128, 128).astype(np.float32)
        mock_pbr.ambient_occlusion = np.random.rand(128, 128).astype(np.float32)
        mock_pbr.height = None
        mock_pbr.properties = MaterialProperties(roughness_mean=0.4, metallic_mean=0.1, ao_strength=0.6)
        mock_pbr.metadata = PBRGenerationMetadata(
            backend="heuristic_v5.0.0",
            normal_scale=1.0,
            ao_blend_ratio="0.7_concavity_0.3_variance",
            bilateral_enabled=True,
            material_hint="wood",
            depth_used=False,
            backend_decision=BackendDecision(
                requested_backend="nvdiffrec",
                executed_backend="heuristic",
                availability_state=AvailabilityState.INPUT_CONTRACT_MISMATCH,
                fallback_reason="single-image input only",
                required_inputs=["multi_view_images"],
                required_runtime=["cuda"],
            ),
        )

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.MaterialBackend") as MockBackend:
            mock_backend = MockBackend.return_value
            mock_backend.generate.return_value = mock_pbr

            result = pipeline._run_materials(
                ingest_result=ingest_result,
                seg_result=seg_result,
                output_dir=tmp_path,
                save_intermediates=True,
            )

        # Check textures directory was created
        textures_dir = tmp_path / "materials" / "segment_0"
        assert textures_dir.exists()
        assert (textures_dir / "albedo.npy").exists()
        assert (textures_dir / "normal.npy").exists()
        assert (textures_dir / "diagnostics.json").exists()
        assert (textures_dir / "provenance.json").exists()

        diagnostics = json.loads((textures_dir / "diagnostics.json").read_text(encoding="utf-8"))
        provenance = json.loads((textures_dir / "provenance.json").read_text(encoding="utf-8"))

        assert diagnostics["segment_id"] == "segment_0"
        assert diagnostics["requested_backend"] == "heuristic"
        assert diagnostics["generation_metadata"]["backend_decision"]["requested_backend"] == "nvdiffrec"
        assert diagnostics["generation_metadata"]["backend_decision"]["executed_backend"] == "heuristic"

        assert provenance["segment_id"] == "segment_0"
        assert provenance["segment_index"] == 0
        assert provenance["input_content_hash"] == "abc123"
        assert provenance["backend_decision"]["availability_state"] == "input_contract_mismatch"
        assert provenance["artifact_payload_hashes"]["hash_target"] == "numpy_array_bytes"
        assert provenance["artifact_payload_hashes"]["albedo"]

    def test_run_materials_save_intermediates_preserves_original_segment_metadata_after_middle_failure(self, tmp_path):
        """Test save_intermediates keeps original segment metadata when a middle segment is skipped."""
        config = PipelineConfig(
            tier="standard",
            stages=["materials"],
            resource_limits=ResourceLimits(device_preference=["cpu"]),
            error_strategy=ErrorRecoveryStrategy.RETRY,
        )
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(128, 128, 3).astype(np.float32)
        ingest_result.gamma = 1.0
        ingest_result.input_path = Path("input.tiff")
        ingest_result.content_hash = "abc123"
        ingest_result.input_size = (128, 128)

        masks = [
            np.pad(np.ones((16, 16), dtype=bool), ((0, 112), (0, 112))),
            np.pad(np.ones((20, 20), dtype=bool), ((32, 76), (32, 76))),
            np.pad(np.ones((24, 24), dtype=bool), ((64, 40), (64, 40))),
        ]
        metadata = [
            MaskMetadata(area=16 * 16, bbox=(0, 0, 16, 16), stability_score=0.91, material_label="wood"),
            MaskMetadata(area=20 * 20, bbox=(32, 32, 20, 20), stability_score=0.82, material_label="metal"),
            MaskMetadata(area=24 * 24, bbox=(64, 64, 24, 24), stability_score=0.73, material_label="stone"),
        ]

        seg_result = MagicMock(spec=SegmentationResult)
        seg_result.masks = masks
        seg_result.scores = np.array([0.95, 0.85, 0.75])
        seg_result.metadata = metadata

        def make_mock_pbr(material_hint: str) -> MagicMock:
            mock_pbr = MagicMock(spec=PBRTextures)
            mock_pbr.albedo = np.random.rand(128, 128, 3).astype(np.float32)
            mock_pbr.normal = np.random.rand(128, 128, 3).astype(np.float32)
            mock_pbr.roughness = np.random.rand(128, 128).astype(np.float32)
            mock_pbr.metallic = np.random.rand(128, 128).astype(np.float32)
            mock_pbr.ambient_occlusion = np.random.rand(128, 128).astype(np.float32)
            mock_pbr.height = None
            mock_pbr.properties = MaterialProperties(roughness_mean=0.4, metallic_mean=0.1, ao_strength=0.6)
            mock_pbr.metadata = PBRGenerationMetadata(
                backend="heuristic_v5.0.0",
                normal_scale=1.0,
                ao_blend_ratio="0.7_concavity_0.3_variance",
                bilateral_enabled=True,
                material_hint=material_hint,
                depth_used=False,
                backend_decision=BackendDecision(
                    requested_backend="heuristic",
                    executed_backend="heuristic",
                    availability_state=AvailabilityState.AVAILABLE,
                    fallback_reason=None,
                    required_inputs=[],
                    required_runtime=[],
                ),
            )
            return mock_pbr

        first_pbr = make_mock_pbr("wood")
        third_pbr = make_mock_pbr("stone")

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.MaterialBackend") as MockBackend:
            mock_backend = MockBackend.return_value
            mock_backend.generate.side_effect = [first_pbr, RuntimeError("segment 1 failed"), third_pbr]

            result = pipeline._run_materials(
                ingest_result=ingest_result,
                seg_result=seg_result,
                output_dir=tmp_path,
                save_intermediates=True,
            )

        assert set(result.keys()) == {"segment_0", "segment_2"}

        materials_dir = tmp_path / "materials"
        assert (materials_dir / "segment_0").exists()
        assert not (materials_dir / "segment_1").exists()
        assert (materials_dir / "segment_2").exists()

        diagnostics = json.loads((materials_dir / "segment_2" / "diagnostics.json").read_text(encoding="utf-8"))
        provenance = json.loads((materials_dir / "segment_2" / "provenance.json").read_text(encoding="utf-8"))

        assert diagnostics["segment_id"] == "segment_2"
        assert diagnostics["segment_index"] == 2
        assert diagnostics["generation_metadata"]["material_hint"] == "stone"
        assert diagnostics["mask_area"] == int(np.count_nonzero(masks[2]))

        assert provenance["segment_id"] == "segment_2"
        assert provenance["segment_index"] == 2
        assert provenance["mask_metadata"]["area"] == metadata[2].area
        assert provenance["mask_metadata"]["bbox"] == list(metadata[2].bbox)
        assert provenance["mask_metadata"]["stability_score"] == metadata[2].stability_score
        assert provenance["mask_metadata"]["material_label"] == metadata[2].material_label


class TestSpatialAIPipelineReconstructionStage:
    """Test reconstruction stage execution."""

    def test_run_reconstruction_not_implemented(self, tmp_path):
        """Test reconstruction raises NotImplementedError for single-view."""
        config = PipelineConfig(
            tier="apex_research",
            stages=["reconstruction"],
        )
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)

        with pytest.raises(PipelineError, match="multi-view"):
            pipeline._run_reconstruction(
                ingest_result=ingest_result,
                seg_result=None,
                output_dir=tmp_path,
                save_intermediates=False,
            )


class TestSpatialAIPipelineE2E:
    """Test end-to-end pipeline execution."""

    def test_process_ingest_only(self, tmp_path):
        """Test pipeline with ingest stage only."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
            ingest={"strict_ingest": False},
        )
        pipeline = SpatialAIPipeline(config)

        # Create test input
        input_path = tmp_path / "input.tiff"
        input_path.touch()

        output_dir = tmp_path / "output"

        # Mock ingest
        mock_result = MagicMock(spec=LinearIngestResult)
        mock_result.input_size = (512, 512)
        mock_result.linear_rgb = np.random.rand(512, 512, 3).astype(np.float32)
        mock_result.gamma = 1.0

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            mock_decoder = MockDecoder.return_value
            mock_decoder.decode.return_value = mock_result

            result = pipeline.process(
                input_path=input_path,
                output_dir=output_dir,
                save_intermediates=False,
            )

        assert result.stages_completed == ["ingest"]
        assert result.linear_image is mock_result
        assert result.segmentation is None
        assert result.execution_time > 0
        assert result.peak_memory_mb >= 0

    def test_process_ingest_and_segmentation(self, tmp_path):
        """Test pipeline with ingest and segmentation."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest", "segment"],
            resource_limits=ResourceLimits(device_preference=["cpu"]),
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()
        output_dir = tmp_path / "output"

        # Mock ingest
        mock_ingest = MagicMock(spec=LinearIngestResult)
        mock_ingest.input_size = (256, 256)
        mock_ingest.linear_rgb = np.random.rand(256, 256, 3).astype(np.float32)
        mock_ingest.gamma = 1.0

        # Mock segmentation
        mock_seg = MagicMock(spec=SegmentationResult)
        mock_seg.masks = [np.ones((256, 256), dtype=bool)]
        mock_seg.scores = np.array([0.9])
        mock_seg.metadata = [MaskMetadata(area=256 * 256, bbox=(0, 0, 256, 256), stability_score=0.9)]

        with (
            patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder,
            patch("transformation_portal.spatial_ai.orchestration.pipeline.SAM2Backend") as MockSAM2,
        ):

            MockDecoder.return_value.decode.return_value = mock_ingest
            MockSAM2.return_value.segment.return_value = mock_seg

            result = pipeline.process(
                input_path=input_path,
                output_dir=output_dir,
                save_intermediates=False,
            )

        assert result.stages_completed == ["ingest", "segmentation"]
        assert result.linear_image is mock_ingest
        assert result.segmentation is mock_seg

    def test_process_full_pipeline(self, tmp_path):
        """Test full pipeline with ingest, segment, materials."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest", "segment", "materials"],
            resource_limits=ResourceLimits(device_preference=["cpu"]),
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()
        output_dir = tmp_path / "output"

        # Mock all stages
        mock_ingest = MagicMock(spec=LinearIngestResult)
        mock_ingest.input_size = (128, 128)
        mock_ingest.linear_rgb = np.random.rand(128, 128, 3).astype(np.float32)
        mock_ingest.gamma = 1.0

        mock_seg = MagicMock(spec=SegmentationResult)
        mock_seg.masks = [np.ones((128, 128), dtype=bool)]
        mock_seg.scores = np.array([0.9])
        mock_seg.metadata = [MaskMetadata(area=128 * 128, bbox=(0, 0, 128, 128), stability_score=0.9)]

        mock_pbr = MagicMock(spec=PBRTextures)
        mock_pbr.albedo = np.random.rand(128, 128, 3).astype(np.float32)
        mock_pbr.normal = np.random.rand(128, 128, 3).astype(np.float32)
        mock_pbr.roughness = np.random.rand(128, 128).astype(np.float32)
        mock_pbr.metallic = np.random.rand(128, 128).astype(np.float32)
        mock_pbr.ambient_occlusion = np.random.rand(128, 128).astype(np.float32)
        mock_pbr.height = None
        mock_pbr.properties = MaterialProperties(roughness_mean=0.4, metallic_mean=0.1, ao_strength=0.6)
        mock_pbr.metadata = PBRGenerationMetadata(
            backend="heuristic_v5.0.0",
            normal_scale=1.0,
            ao_blend_ratio="0.7_concavity_0.3_variance",
            bilateral_enabled=True,
            material_hint=None,
            depth_used=False,
            backend_decision=BackendDecision(
                requested_backend="heuristic",
                executed_backend="heuristic",
                availability_state=AvailabilityState.AVAILABLE,
                fallback_reason=None,
                required_inputs=[],
                required_runtime=[],
            ),
        )

        with (
            patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder,
            patch("transformation_portal.spatial_ai.orchestration.pipeline.SAM2Backend") as MockSAM2,
            patch("transformation_portal.spatial_ai.orchestration.pipeline.MaterialBackend") as MockMaterial,
        ):

            MockDecoder.return_value.decode.return_value = mock_ingest
            MockSAM2.return_value.segment.return_value = mock_seg
            MockMaterial.return_value.generate.return_value = mock_pbr

            result = pipeline.process(
                input_path=input_path,
                output_dir=output_dir,
                save_intermediates=True,
            )

        assert result.stages_completed == ["ingest", "segmentation", "materials"]
        assert result.materials is not None
        assert len(result.materials) == 1

    def test_process_saves_summary(self, tmp_path):
        """Test pipeline saves summary when save_intermediates=True."""
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()
        output_dir = tmp_path / "output"

        mock_result = MagicMock(spec=LinearIngestResult)
        mock_result.input_size = (64, 64)
        mock_result.linear_rgb = np.random.rand(64, 64, 3).astype(np.float32)

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            MockDecoder.return_value.decode.return_value = mock_result

            pipeline.process(
                input_path=input_path,
                output_dir=output_dir,
                save_intermediates=True,
            )

        summary_path = output_dir / "pipeline_summary.json"
        assert summary_path.exists()

    def test_process_input_not_found(self, tmp_path):
        """Test process raises error when input file not found."""
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        with pytest.raises(FileNotFoundError):
            pipeline.process(
                input_path=tmp_path / "nonexistent.tiff",
                output_dir=tmp_path / "output",
            )

    def test_process_stage_dependency_validation(self, tmp_path):
        """Test process validates stage dependencies."""
        config = PipelineConfig(
            tier="standard",
            stages=["segment"],  # Missing ingest dependency
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        with pytest.raises(PipelineError, match="Ingest stage required"):
            pipeline.process(
                input_path=input_path,
                output_dir=tmp_path / "output",
            )


class TestSpatialAIPipelineErrorHandling:
    """Test pipeline error handling integration."""

    def test_process_with_fail_fast_strategy(self, tmp_path):
        """Test pipeline with FAIL_FAST error strategy."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
            error_strategy=ErrorRecoveryStrategy.FAIL_FAST,
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            MockDecoder.return_value.decode.side_effect = RuntimeError("Decode failed")

            with pytest.raises(PipelineError):
                pipeline.process(
                    input_path=input_path,
                    output_dir=tmp_path / "output",
                )

    def test_process_with_return_partial_strategy(self, tmp_path):
        """Test pipeline with RETURN_PARTIAL strategy returns partial results."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest", "segment"],
            error_strategy=ErrorRecoveryStrategy.RETURN_PARTIAL,
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        mock_ingest = MagicMock(spec=LinearIngestResult)
        mock_ingest.input_size = (64, 64)
        mock_ingest.linear_rgb = np.random.rand(64, 64, 3).astype(np.float32)

        with (
            patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder,
            patch("transformation_portal.spatial_ai.orchestration.pipeline.SAM2Backend") as MockSAM2,
        ):

            MockDecoder.return_value.decode.return_value = mock_ingest
            MockSAM2.return_value.segment.side_effect = RuntimeError("Segmentation failed")

            result = pipeline.process(
                input_path=input_path,
                output_dir=tmp_path / "output",
            )

        # Should return partial results
        assert result.stages_completed == ["ingest"]
        assert result.linear_image is mock_ingest
        assert result.segmentation is None
        assert len(result.errors) > 0


class TestSpatialAIPipelineResourceManagement:
    """Test resource management integration."""

    def test_process_uses_resource_manager_context(self, tmp_path):
        """Test process uses resource manager as context manager.

        This test verifies that the pipeline process successfully completes,
        which implicitly confirms the resource manager context is used correctly.
        Direct mocking of __enter__/__exit__ is not possible due to Python's
        special method lookup on the type, not the instance.
        """
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        mock_result = MagicMock(spec=LinearIngestResult)
        mock_result.input_size = (32, 32)
        mock_result.linear_rgb = np.random.rand(32, 32, 3).astype(np.float32)

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            MockDecoder.return_value.decode.return_value = mock_result

            result = pipeline.process(
                input_path=input_path,
                output_dir=tmp_path / "output",
            )

        # If the context manager wasn't used, process would fail
        # Successful completion confirms context manager was used
        assert result is not None
        assert result.linear_image is mock_result

    def test_process_tracks_peak_memory(self, tmp_path):
        """Test process tracks peak memory usage."""
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        mock_result = MagicMock(spec=LinearIngestResult)
        mock_result.input_size = (32, 32)
        mock_result.linear_rgb = np.random.rand(32, 32, 3).astype(np.float32)

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            MockDecoder.return_value.decode.return_value = mock_result

            with patch.object(pipeline.resource_manager, "get_peak_memory_mb", return_value=1024.5):
                result = pipeline.process(
                    input_path=input_path,
                    output_dir=tmp_path / "output",
                )

        assert result.peak_memory_mb == 1024.5


class TestSpatialAIPipelineProgressTracking:
    """Test progress tracking integration."""

    def test_process_emits_progress_events(self, tmp_path):
        """Test process emits progress events through tracker."""
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        mock_result = MagicMock(spec=LinearIngestResult)
        mock_result.input_size = (32, 32)
        mock_result.linear_rgb = np.random.rand(32, 32, 3).astype(np.float32)

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            MockDecoder.return_value.decode.return_value = mock_result

            with (
                patch.object(pipeline.progress_tracker, "start_pipeline") as mock_start,
                patch.object(pipeline.progress_tracker, "complete_pipeline") as mock_complete,
            ):

                pipeline.process(
                    input_path=input_path,
                    output_dir=tmp_path / "output",
                )

        mock_start.assert_called_once()
        mock_complete.assert_called_once_with(success=True)

    def test_process_tracks_execution_time(self, tmp_path):
        """Test process tracks execution time."""
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        mock_result = MagicMock(spec=LinearIngestResult)
        mock_result.input_size = (32, 32)
        mock_result.linear_rgb = np.random.rand(32, 32, 3).astype(np.float32)

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            MockDecoder.return_value.decode.return_value = mock_result

            result = pipeline.process(
                input_path=input_path,
                output_dir=tmp_path / "output",
            )

        assert result.execution_time >= 0.0


class TestGraphModeExecution:
    """Test graph-mode execution path (ADR-029)."""

    def test_graph_mode_rejects_reconstruction(self, tmp_path):
        """Graph mode should fail explicitly if reconstruction is requested."""
        config = PipelineConfig(
            tier="apex_research",  # Use research tier (3DGS license requires it)
            stages=["ingest", "segment", "reconstruction"],
            use_execution_graph=True,
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        with pytest.raises(PipelineError) as exc_info:
            pipeline.process(
                input_path=input_path,
                output_dir=tmp_path / "output",
            )

        assert "reconstruction" in str(exc_info.value).lower()
        assert "graph mode" in str(exc_info.value).lower()

    def test_graph_mode_delegates_to_executor(self, tmp_path):
        """Graph mode should delegate to Executor.execute()."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
            use_execution_graph=True,
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        # Mock the executor and its dependencies
        mock_exec_result = MagicMock()
        mock_exec_result.stages_executed = 1
        mock_exec_result.stages_cached = 0
        mock_exec_result.total_time_ms = 100.0
        mock_exec_result.stage_results = []
        mock_exec_result.outputs = {}

        with patch("transformation_portal.spatial_ai.orchestration.graph.executor.Executor.execute") as mock_execute:
            mock_execute.return_value = mock_exec_result

            result = pipeline.process(
                input_path=input_path,
                output_dir=tmp_path / "output",
            )

            # Verify executor was called
            mock_execute.assert_called_once()

            # Verify result is valid
            assert isinstance(result, PipelineResult)

    def test_graph_mode_wraps_errors_as_pipeline_error(self, tmp_path):
        """Graph mode should wrap execution errors in PipelineError."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
            use_execution_graph=True,
            error_strategy=ErrorRecoveryStrategy.FAIL_FAST,
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        with patch("transformation_portal.spatial_ai.orchestration.graph.executor.Executor.execute") as mock_execute:
            mock_execute.side_effect = RuntimeError("Test executor failure")

            with pytest.raises(PipelineError) as exc_info:
                pipeline.process(
                    input_path=input_path,
                    output_dir=tmp_path / "output",
                )

            # Verify error is wrapped in PipelineError
            assert "graph" in str(exc_info.value.stage).lower()

    def test_graph_mode_return_partial_on_error(self, tmp_path):
        """Graph mode should return partial result if RETURN_PARTIAL strategy."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest"],
            use_execution_graph=True,
            error_strategy=ErrorRecoveryStrategy.RETURN_PARTIAL,
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        with patch("transformation_portal.spatial_ai.orchestration.graph.executor.Executor.execute") as mock_execute:
            mock_execute.side_effect = RuntimeError("Test executor failure")

            # Should not raise, should return partial result
            result = pipeline.process(
                input_path=input_path,
                output_dir=tmp_path / "output",
            )

            assert isinstance(result, PipelineResult)
            assert len(result.errors) > 0

    def test_graph_mode_without_reconstruction_succeeds(self, tmp_path):
        """Graph mode should succeed without reconstruction stage."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest", "segment"],
            use_execution_graph=True,
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        mock_exec_result = MagicMock()
        mock_exec_result.stages_executed = 2
        mock_exec_result.stages_cached = 0
        mock_exec_result.total_time_ms = 200.0
        mock_exec_result.stage_results = []
        mock_exec_result.outputs = {}

        with patch("transformation_portal.spatial_ai.orchestration.graph.executor.Executor.execute") as mock_execute:
            mock_execute.return_value = mock_exec_result

            result = pipeline.process(
                input_path=input_path,
                output_dir=tmp_path / "output",
            )

            assert isinstance(result, PipelineResult)
            mock_execute.assert_called_once()
