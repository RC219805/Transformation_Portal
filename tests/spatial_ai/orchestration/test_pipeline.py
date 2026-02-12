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
from transformation_portal.spatial_ai.materials.contracts import PBRTextures
from transformation_portal.spatial_ai.orchestration.error_handler import ErrorRecoveryStrategy, PipelineError
from transformation_portal.spatial_ai.orchestration.pipeline import PipelineConfig, PipelineResult, SpatialAIPipeline
from transformation_portal.spatial_ai.orchestration.resource_manager import ResourceLimits
from transformation_portal.spatial_ai.segmentation.contracts import MaskMetadata, SegmentationResult


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
        valid_stages = ["ingest", "segment", "materials", "reconstruct"]
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
                stages=["ingest", "reconstruct"],
            )

    def test_reconstruction_allowed_in_research_tier(self):
        """Test reconstruction is allowed in research tier."""
        config = PipelineConfig(
            tier="apex_research",
            stages=["ingest", "reconstruct"],
        )
        assert "reconstruct" in config.stages

    def test_reconstruction_allowed_in_experimental_tier(self):
        """Test reconstruction is allowed in experimental tier."""
        config = PipelineConfig(
            tier="experimental",
            stages=["reconstruct"],
        )
        assert "reconstruct" in config.stages

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
        assert "segment" in pipeline.config.stages

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
        assert "segment" in config.stages

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

        # Mock OpenEXR not available
        with patch.dict("sys.modules", {"OpenEXR": None}):
            with pytest.raises(RuntimeError, match="OpenEXR"):
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
        mock_seg_result.metadata = [MaskMetadata(material_label="wall")]

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

    def test_run_segmentation_saves_intermediates(self, tmp_path):
        """Test segmentation saves masks when save_intermediates=True."""
        config = PipelineConfig(tier="standard", stages=["segment"])
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(256, 256, 3).astype(np.float32)
        ingest_result.gamma = 1.0

        mock_seg_result = MagicMock(spec=SegmentationResult)
        mock_seg_result.masks = [np.ones((256, 256), dtype=bool)]
        mock_seg_result.scores = np.array([0.9])
        mock_seg_result.metadata = [MaskMetadata()]

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
        )
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(256, 256, 3).astype(np.float32)

        with pytest.raises(ValueError, match="sam2"):
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
            MaskMetadata(material_label="wood"),
            MaskMetadata(material_label="metal"),
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
        """Test materials saves textures when save_intermediates=True."""
        config = PipelineConfig(tier="standard", stages=["materials"])
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)
        ingest_result.linear_rgb = np.random.rand(128, 128, 3).astype(np.float32)
        ingest_result.gamma = 1.0

        seg_result = MagicMock(spec=SegmentationResult)
        seg_result.masks = [np.ones((128, 128), dtype=bool)]
        seg_result.scores = np.array([0.9])
        seg_result.metadata = [MaskMetadata()]

        mock_pbr = MagicMock(spec=PBRTextures)
        mock_pbr.albedo = np.random.rand(128, 128, 3).astype(np.float32)
        mock_pbr.normal = np.random.rand(128, 128, 3).astype(np.float32)
        mock_pbr.roughness = np.random.rand(128, 128).astype(np.float32)
        mock_pbr.metallic = np.random.rand(128, 128).astype(np.float32)
        mock_pbr.ambient_occlusion = np.random.rand(128, 128).astype(np.float32)
        mock_pbr.height = None

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


class TestSpatialAIPipelineReconstructionStage:
    """Test reconstruction stage execution."""

    def test_run_reconstruction_not_implemented(self, tmp_path):
        """Test reconstruction raises NotImplementedError for single-view."""
        config = PipelineConfig(
            tier="apex_research",
            stages=["reconstruct"],
        )
        pipeline = SpatialAIPipeline(config)

        ingest_result = MagicMock(spec=LinearIngestResult)

        with pytest.raises(NotImplementedError, match="multi-view"):
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
        mock_seg.metadata = [MaskMetadata()]

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

        assert result.stages_completed == ["ingest", "segment"]
        assert result.linear_image is mock_ingest
        assert result.segmentation is mock_seg

    def test_process_full_pipeline(self, tmp_path):
        """Test full pipeline with ingest, segment, materials."""
        config = PipelineConfig(
            tier="standard",
            stages=["ingest", "segment", "materials"],
        )
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()
        output_dir = tmp_path / "output"

        # Mock all stages
        mock_ingest = MagicMock(spec=LinearIngestResult)
        mock_ingest.linear_rgb = np.random.rand(128, 128, 3).astype(np.float32)
        mock_ingest.gamma = 1.0

        mock_seg = MagicMock(spec=SegmentationResult)
        mock_seg.masks = [np.ones((128, 128), dtype=bool)]
        mock_seg.scores = np.array([0.9])
        mock_seg.metadata = [MaskMetadata()]

        mock_pbr = MagicMock(spec=PBRTextures)

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

        assert result.stages_completed == ["ingest", "segment", "materials"]
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
        """Test process uses resource manager as context manager."""
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        mock_result = MagicMock(spec=LinearIngestResult)
        mock_result.linear_rgb = np.random.rand(32, 32, 3).astype(np.float32)

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            MockDecoder.return_value.decode.return_value = mock_result

            with (
                patch.object(pipeline.resource_manager, "__enter__") as mock_enter,
                patch.object(pipeline.resource_manager, "__exit__") as mock_exit,
            ):

                mock_enter.return_value = pipeline.resource_manager

                pipeline.process(
                    input_path=input_path,
                    output_dir=tmp_path / "output",
                )

        mock_enter.assert_called_once()
        mock_exit.assert_called_once()

    def test_process_tracks_peak_memory(self, tmp_path):
        """Test process tracks peak memory usage."""
        config = PipelineConfig(tier="standard", stages=["ingest"])
        pipeline = SpatialAIPipeline(config)

        input_path = tmp_path / "input.tiff"
        input_path.touch()

        mock_result = MagicMock(spec=LinearIngestResult)
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
        mock_result.linear_rgb = np.random.rand(32, 32, 3).astype(np.float32)

        with patch("transformation_portal.spatial_ai.orchestration.pipeline.LinearDecoder") as MockDecoder:
            MockDecoder.return_value.decode.return_value = mock_result

            result = pipeline.process(
                input_path=input_path,
                output_dir=tmp_path / "output",
            )

        assert result.execution_time >= 0.0
