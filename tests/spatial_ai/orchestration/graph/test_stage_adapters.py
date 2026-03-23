"""Tests for stage adapters (ADR-029 integration).

Tests the IngestStage, SegmentationStage, MaterialsStage adapters
that bridge legacy pipeline functionality with ExecutionGraph.
"""

from dataclasses import dataclass
from pathlib import Path

import pytest

from transformation_portal.spatial_ai.orchestration.graph.stage_adapters import (
    IngestStage,
    MaterialsStage,
    SegmentationStage,
    build_spatial_ai_graph,
)

pytestmark = pytest.mark.unit


@dataclass
class MockExecutionContext:
    """Mock execution context for testing."""

    device: str = "cpu"
    config: dict = None
    output_dir: Path = None
    enable_caching: bool = False

    def __post_init__(self):
        if self.config is None:
            self.config = {}
        if self.output_dir is None:
            import tempfile

            self.output_dir = Path(tempfile.gettempdir()) / "test_output"


class TestIngestStageMetadata:
    """Tests for IngestStage metadata."""

    def test_metadata_has_required_fields(self):
        """IngestStage metadata should have all required fields."""
        stage = IngestStage()
        meta = stage.metadata

        assert meta.name == "linear_ingest"
        assert meta.version == "1.0.0"
        assert meta.description
        assert meta.resource_requirements.gpu_memory_mb == 0
        assert meta.resource_requirements.cpu_memory_mb > 0
        assert meta.deterministic is True
        assert meta.idempotent is True

    def test_metadata_marks_stage_as_cpu_only(self):
        """IngestStage should not require GPU."""
        stage = IngestStage()
        assert stage.metadata.resource_requirements.gpu_required is False


class TestIngestStageCacheKey:
    """Tests for IngestStage cache key computation."""

    def test_cache_key_is_sha256(self):
        """Cache key should be 64 hex characters (SHA256)."""
        stage = IngestStage()
        context = MockExecutionContext()

        key = stage.compute_cache_key({"input_path": None}, context)

        assert len(key) == 64
        assert all(c in "0123456789abcdef" for c in key)

    def test_different_inputs_produce_different_keys(self, tmp_path):
        """Different input files should produce different cache keys."""
        stage = IngestStage()
        context = MockExecutionContext()

        # Create two different input files
        file1 = tmp_path / "test1.txt"
        file2 = tmp_path / "test2.txt"
        file1.write_text("content1")
        file2.write_text("content2")

        key1 = stage.compute_cache_key({"input_path": file1}, context)
        key2 = stage.compute_cache_key({"input_path": file2}, context)

        assert key1 != key2


class TestSegmentationStageMetadata:
    """Tests for SegmentationStage metadata."""

    def test_metadata_has_required_fields(self):
        """SegmentationStage metadata should have all required fields."""
        stage = SegmentationStage()
        meta = stage.metadata

        assert meta.name == "sam2_segmentation"
        assert meta.version == "2.1.0"
        assert meta.description
        assert meta.resource_requirements.gpu_memory_mb > 0
        assert meta.deterministic is True

    def test_model_size_affects_resource_requirements(self):
        """Larger model should require more GPU memory."""
        large_stage = SegmentationStage(model_size="large")
        base_stage = SegmentationStage(model_size="base")

        large_gpu = large_stage.metadata.resource_requirements.gpu_memory_mb
        base_gpu = base_stage.metadata.resource_requirements.gpu_memory_mb
        assert large_gpu > base_gpu


class TestMaterialsStageMetadata:
    """Tests for MaterialsStage metadata."""

    def test_metadata_has_required_fields(self):
        """MaterialsStage metadata should have all required fields."""
        stage = MaterialsStage()
        meta = stage.metadata

        assert meta.name == "pbr_materials"
        assert meta.version == "2.2.0"
        assert meta.description
        assert meta.deterministic is True
        assert meta.idempotent is True

    def test_heuristic_backend_has_no_gpu_requirement(self):
        """Heuristic backend should not require GPU."""
        stage = MaterialsStage(backend="heuristic")
        assert stage.metadata.resource_requirements.gpu_memory_mb == 0


class TestBuildSpatialAiGraph:
    """Tests for build_spatial_ai_graph factory function."""

    def test_builds_graph_with_default_stages(self):
        """Factory should create graph with default stages (ingest, segment)."""
        graph = build_spatial_ai_graph()

        assert graph.get_stage("ingest") is not None
        assert graph.get_stage("segment") is not None
        assert graph.get_stage("materials") is None

    def test_builds_graph_with_materials_stage(self):
        """Factory should create graph with materials stage when requested."""
        graph = build_spatial_ai_graph(
            stages=["ingest", "segment", "materials"],
        )

        assert graph.get_stage("ingest") is not None
        assert graph.get_stage("segment") is not None
        assert graph.get_stage("materials") is not None

    def test_graph_has_correct_dependencies(self):
        """Graph stages should have correct input dependencies."""
        graph = build_spatial_ai_graph(
            stages=["ingest", "segment", "materials"],
        )

        # Segment depends on ingest
        segment_node = graph.get_stage("segment")
        assert "linear_rgb" in segment_node.inputs
        assert segment_node.inputs["linear_rgb"] == "ingest.linear_rgb"

        # Materials depends on ingest and segment
        materials_node = graph.get_stage("materials")
        assert "linear_rgb" in materials_node.inputs
        assert "masks" in materials_node.inputs
        assert materials_node.inputs["linear_rgb"] == "ingest.linear_rgb"
        assert materials_node.inputs["masks"] == "segment.masks"

    def test_requires_ingest_for_segment(self):
        """Factory should reject segment without ingest."""
        with pytest.raises(ValueError, match="Segmentation requires ingest"):
            build_spatial_ai_graph(stages=["segment"])

    def test_requires_segment_for_materials(self):
        """Factory should reject materials without segmentation."""
        with pytest.raises(ValueError, match="Materials requires segmentation"):
            build_spatial_ai_graph(stages=["ingest", "materials"])

    def test_can_plan_graph(self):
        """Built graph should be plannable."""
        graph = build_spatial_ai_graph(
            stages=["ingest", "segment"],
        )

        plan = graph.plan()

        assert len(plan.stages) == 2
        # Ingest should come before segment
        stage_ids = [s.stage_id for s in plan.stages]
        assert stage_ids.index("ingest") < stage_ids.index("segment")
