"""
Tests for RAG Artifact Classifier.
"""

import sys
from pathlib import Path

import pytest

# Add agents directory to path
agents_path = Path(__file__).parent.parent / '.github' / 'agents'
sys.path.insert(0, str(agents_path))

from rag_system.classifier import (  # noqa: E402
    ArtifactClassifier,
    ArtifactType,
    PipelineType,
    ProcessingMetadata,
)


@pytest.fixture
def classifier():
    """Create a classifier instance."""
    return ArtifactClassifier()


class TestArtifactClassification:
    """Test artifact classification."""

    def test_classify_depth_map(self, classifier):
        """Test depth map classification."""
        result = classifier.classify_artifact("output/depth_map_2024-11-05.png")
        assert result == ArtifactType.DEPTH_MAP

    def test_classify_color_grade(self, classifier):
        """Test color grade classification."""
        result = classifier.classify_artifact("graded_final_lut_applied.jpg")
        assert result == ArtifactType.COLOR_GRADE

    def test_classify_hdr_output(self, classifier):
        """Test HDR output classification."""
        result = classifier.classify_artifact("hdr_tonemapped_output.exr")
        assert result == ArtifactType.HDR_OUTPUT

    def test_classify_metric(self, classifier):
        """Test metrics classification."""
        result = classifier.classify_artifact("performance_metrics.json")
        assert result == ArtifactType.METRIC

    def test_classify_log(self, classifier):
        """Test log classification."""
        result = classifier.classify_artifact("processing.log")
        assert result == ArtifactType.LOG

    def test_classify_unknown(self, classifier):
        """Test unknown artifact classification."""
        result = classifier.classify_artifact("random_file.xyz")
        assert result == ArtifactType.UNKNOWN


class TestPipelineDetection:
    """Test pipeline detection."""

    def test_detect_depth_pipeline(self, classifier):
        """Test depth pipeline detection."""
        result = classifier.detect_pipeline("depth_pipeline/output/image.png")
        assert result == PipelineType.DEPTH_PIPELINE

    def test_detect_lux_render(self, classifier):
        """Test lux render detection."""
        result = classifier.detect_pipeline("lux_render_output/enhanced.jpg")
        assert result == PipelineType.LUX_RENDER

    def test_detect_material_response(self, classifier):
        """Test material response detection."""
        result = classifier.detect_pipeline("material_response/processed.tif")
        assert result == PipelineType.MATERIAL_RESPONSE

    def test_detect_unknown_pipeline(self, classifier):
        """Test unknown pipeline detection."""
        result = classifier.detect_pipeline("random/path/file.jpg")
        assert result == PipelineType.UNKNOWN


class TestMetadataExtraction:
    """Test metadata extraction."""

    def test_extract_timestamp_from_filename(self, classifier):
        """Test timestamp extraction."""
        metadata = classifier.extract_metadata(
            "output_2024-11-05/depth_map.png",
            ArtifactType.DEPTH_MAP,
            PipelineType.DEPTH_PIPELINE,
        )
        assert metadata.timestamp is not None
        assert metadata.timestamp.year == 2024
        assert metadata.timestamp.month == 11
        assert metadata.timestamp.day == 5

    def test_extract_resolution(self, classifier):
        """Test resolution extraction."""
        metadata = classifier.extract_metadata(
            "image_1920x1080.jpg",
            ArtifactType.RENDER,
            PipelineType.CUSTOM,
        )
        assert metadata.resolution == (1920, 1080)

    def test_extract_from_json_content(self, classifier):
        """Test extraction from JSON content."""
        json_content = '''
        {
            "parameters": {"quality": "high", "denoise": 0.5},
            "processing_time": 45.2,
            "memory_usage": 2048.5,
            "gpu_utilization": 85.3,
            "success": true
        }
        '''
        metadata = classifier.extract_metadata(
            "metrics.json",
            ArtifactType.METRIC,
            PipelineType.DEPTH_PIPELINE,
            content=json_content,
        )
        assert metadata.parameters == {"quality": "high", "denoise": 0.5}
        assert metadata.processing_time == 45.2
        assert metadata.memory_usage == 2048.5
        assert metadata.gpu_utilization == 85.3
        assert metadata.success is True

    def test_extract_error_from_log(self, classifier):
        """Test error extraction from log."""
        log_content = "Processing started\nError: CUDA out of memory\nFailed to process"
        metadata = classifier.extract_metadata(
            "process.log",
            ArtifactType.LOG,
            PipelineType.DEPTH_PIPELINE,
            content=log_content,
        )
        assert metadata.success is False
        assert "CUDA out of memory" in metadata.error_message

    def test_extract_ai_model_info(self, classifier):
        """Test AI model extraction."""
        metadata = classifier.extract_metadata(
            "depth_anything_v2_output.png",
            ArtifactType.DEPTH_MAP,
            PipelineType.DEPTH_PIPELINE,
        )
        assert metadata.ai_model == "Depth Anything V2"

    def test_extract_color_space(self, classifier):
        """Test color space extraction."""
        metadata = classifier.extract_metadata(
            "image_srgb_16bit.tif",
            ArtifactType.RENDER,
            PipelineType.TIFF_PROCESSOR,
        )
        assert metadata.color_space == "SRGB"
        assert metadata.bit_depth == 16


class TestTagGeneration:
    """Test tag generation."""

    def test_generate_basic_tags(self, classifier):
        """Test basic tag generation."""
        metadata = ProcessingMetadata(
            pipeline=PipelineType.DEPTH_PIPELINE,
            artifact_type=ArtifactType.DEPTH_MAP,
            success=True,
        )
        tags = classifier.generate_tags(
            ArtifactType.DEPTH_MAP,
            metadata,
            "output/depth_map.png",
        )
        assert 'depth_map' in tags
        assert 'depth_pipeline' in tags
        assert 'success' in tags

    def test_generate_resolution_tags(self, classifier):
        """Test resolution tag generation."""
        metadata = ProcessingMetadata(
            pipeline=PipelineType.DEPTH_PIPELINE,
            artifact_type=ArtifactType.RENDER,
            resolution=(3840, 2160),
        )
        tags = classifier.generate_tags(
            ArtifactType.RENDER,
            metadata,
            "output/render.png",
        )
        assert 'resolution:3840x2160' in tags
        assert '4k_plus' in tags

    def test_generate_performance_tags(self, classifier):
        """Test performance tag generation."""
        # Fast processing
        metadata = ProcessingMetadata(
            pipeline=PipelineType.DEPTH_PIPELINE,
            artifact_type=ArtifactType.RENDER,
            processing_time=0.5,
        )
        tags = classifier.generate_tags(
            ArtifactType.RENDER,
            metadata,
            "output/fast.png",
        )
        assert 'fast_processing' in tags

        # Slow processing
        metadata.processing_time = 15.0
        tags = classifier.generate_tags(
            ArtifactType.RENDER,
            metadata,
            "output/slow.png",
        )
        assert 'slow_processing' in tags

    def test_generate_error_tags(self, classifier):
        """Test error tag generation."""
        metadata = ProcessingMetadata(
            pipeline=PipelineType.DEPTH_PIPELINE,
            artifact_type=ArtifactType.LOG,
            success=False,
            error_message="ValueError: invalid input shape",
        )
        tags = classifier.generate_tags(
            ArtifactType.LOG,
            metadata,
            "error.log",
        )
        assert 'has_error' in tags
        assert 'error_type:ValueError' in tags


class TestArtifactHierarchy:
    """Test artifact hierarchy and relationships."""

    def test_add_artifact(self, classifier):
        """Test adding an artifact."""
        artifact = classifier.add_artifact("test/image.png")
        assert artifact.artifact_id.startswith("artifact_")
        assert artifact.file_path == "test/image.png"
        assert artifact.artifact_id in classifier.artifacts

    def test_add_child_artifact(self, classifier):
        """Test adding a child artifact."""
        parent = classifier.add_artifact("original.jpg")
        child = classifier.add_artifact("processed.jpg", parent_id=parent.artifact_id)

        assert child.parent_id == parent.artifact_id
        assert child.artifact_id in parent.children_ids

    def test_link_related_artifacts(self, classifier):
        """Test linking related artifacts."""
        artifact1 = classifier.add_artifact("image1.png")
        artifact2 = classifier.add_artifact("image2.png")

        classifier.link_related_artifacts(artifact1.artifact_id, artifact2.artifact_id)

        assert artifact2.artifact_id in artifact1.related_ids
        assert artifact1.artifact_id in artifact2.related_ids

    def test_get_transformation_chain(self, classifier):
        """Test getting transformation chain."""
        # Create chain: original -> processed -> enhanced
        original = classifier.add_artifact("original.jpg")
        processed = classifier.add_artifact("processed.jpg", parent_id=original.artifact_id)
        enhanced = classifier.add_artifact("enhanced.jpg", parent_id=processed.artifact_id)

        chain = classifier.get_transformation_chain(processed.artifact_id)

        assert len(chain) >= 3
        assert chain[0].artifact_id == original.artifact_id
        assert processed.artifact_id in [c.artifact_id for c in chain]
        assert enhanced.artifact_id in [c.artifact_id for c in chain]


class TestArtifactSearch:
    """Test artifact search functionality."""

    def test_search_by_single_tag(self, classifier):
        """Test search by single tag."""
        # Add artifacts with different types
        classifier.add_artifact("depth_map_001.png")
        classifier.add_artifact("color_grade_001.jpg")
        classifier.add_artifact("depth_map_002.png")

        results = classifier.search_by_tags({'depth_map'}, require_all=False)
        assert len(results) >= 2
        assert all('depth_map' in r.tags for r in results)

    def test_search_by_multiple_tags_any(self, classifier):
        """Test search by multiple tags (any match)."""
        classifier.add_artifact("depth_pipeline/depth_map_4k.png")
        classifier.add_artifact("lux_render/output.jpg")

        results = classifier.search_by_tags(
            {'depth_pipeline', 'lux_render'},
            require_all=False
        )
        assert len(results) >= 2

    def test_search_by_multiple_tags_all(self, classifier):
        """Test search by multiple tags (all required)."""
        # Add artifact with both tags
        metadata_content = '{"success": true, "processing_time": 0.5}'
        classifier.add_artifact(
            "depth_pipeline/depth_map_1920x1080.png",
            content=metadata_content
        )

        results = classifier.search_by_tags(
            {'depth_map', 'full_hd'},
            require_all=True
        )

        # Should find artifacts that have both tags
        for result in results:
            assert 'depth_map' in result.tags
            assert 'full_hd' in result.tags or 'hd' in result.tags


class TestStatistics:
    """Test statistics generation."""

    def test_get_statistics(self, classifier):
        """Test statistics generation."""
        # Add various artifacts
        classifier.add_artifact("depth_map_001.png")
        classifier.add_artifact("depth_map_002.png")
        classifier.add_artifact("color_grade_001.jpg")

        stats = classifier.get_statistics()

        assert stats['total_artifacts'] >= 3
        assert 'by_type' in stats
        assert 'by_pipeline' in stats
        assert isinstance(stats['success_rate'], float)
        assert isinstance(stats['avg_processing_time'], float)

    def test_statistics_with_errors(self, classifier):
        """Test statistics with error artifacts."""
        log_content = "Error: Processing failed"
        classifier.add_artifact("error.log", content=log_content)

        stats = classifier.get_statistics()
        assert stats['artifacts_with_errors'] >= 1


class TestExport:
    """Test export functionality."""

    def test_export_to_json(self, classifier, tmp_path):
        """Test exporting to JSON."""
        # Add some artifacts
        classifier.add_artifact("test1.png")
        classifier.add_artifact("test2.png")

        output_file = tmp_path / "artifacts.json"
        classifier.export_to_json(str(output_file))

        assert output_file.exists()

        # Verify JSON structure
        import json
        with open(output_file) as f:
            data = json.load(f)

        assert 'artifacts' in data
        assert 'statistics' in data
        assert 'export_time' in data
        assert len(data['artifacts']) >= 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
