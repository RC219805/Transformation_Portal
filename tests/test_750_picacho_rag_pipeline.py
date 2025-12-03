#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for 750 Picacho Lane RAG-Enhanced Pipeline.

Comprehensive tests for:
- Property memory system
- RAG context retrieval
- Pipeline processing stages
- Learning feedback loop
- Batch processing
- Configuration loading
"""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Add project to path
project_path = Path(__file__).parent.parent / 'projects' / '750_picacho_lane'
sys.path.insert(0, str(project_path))

from property_memory import (  # noqa: E402
    MaterialType,
    PropertyKnowledge,
    PropertyMemory,
    RoomConfiguration,
    SceneType,
)
from rag_enhanced_pipeline import (  # noqa: E402
    KnowledgeIntegrationBridge,
    PipelineConfig,
    ProcessingMetrics,
    RAGContext,
    RAGEnhancedPipeline,
    load_config_from_yaml,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_image(temp_dir):
    """Create a sample test image."""
    # Create a simple test image (100x100 RGB)
    img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(img_array, mode='RGB')

    img_path = temp_dir / "750Picacho_Pool_test.jpg"
    img.save(img_path, format='JPEG')

    return img_path


@pytest.fixture
def sample_images(temp_dir):
    """Create multiple sample test images."""
    images = []
    scene_names = ["Pool", "GreatRoom", "Kitchen", "Aerial"]

    for scene in scene_names:
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        img_path = temp_dir / f"750Picacho_{scene}_test.jpg"
        img.save(img_path, format='JPEG')
        images.append(img_path)

    return images


@pytest.fixture
def property_memory(temp_dir):
    """Create a PropertyMemory instance with temp storage."""
    memory_path = temp_dir / "memory" / "test_memory.json"
    return PropertyMemory(memory_path)


@pytest.fixture
def pipeline_config(temp_dir):
    """Create a PipelineConfig for testing."""
    return PipelineConfig(
        input_dir=temp_dir / "input",
        output_dir=temp_dir / "output",
        quality_mode="fast",
        enable_depth=False,  # Disable for faster tests
        enable_ai_enhancement=False,
        enable_learning=True,
        preserve_16bit=False,  # Use 8-bit for testing (avoids tifffile dep)
    )


@pytest.fixture
def pipeline(pipeline_config, temp_dir):
    """Create a RAGEnhancedPipeline for testing."""
    memory_path = temp_dir / "memory" / "pipeline_memory.json"
    return RAGEnhancedPipeline(config=pipeline_config, memory_path=memory_path)


# ============================================================================
# Property Memory Tests
# ============================================================================


class TestPropertyMemory:
    """Tests for PropertyMemory class."""

    def test_initialization_creates_defaults(self, property_memory):
        """Test that PropertyMemory initializes with default configurations."""
        assert len(property_memory.room_configs) > 0
        assert SceneType.POOL in property_memory.room_configs
        assert SceneType.GREAT_ROOM in property_memory.room_configs

    def test_get_room_config_returns_config(self, property_memory):
        """Test getting room configuration."""
        config = property_memory.get_room_config(SceneType.POOL)

        assert config is not None
        assert config.scene_type == SceneType.POOL
        assert len(config.materials) > 0
        assert MaterialType.WATER in config.materials

    def test_get_optimal_parameters(self, property_memory):
        """Test getting optimal parameters for a scene."""
        params = property_memory.get_optimal_parameters(SceneType.POOL)

        assert isinstance(params, dict)
        assert 'water_enhance' in params
        assert params['water_enhance'] is True

    def test_get_materials(self, property_memory):
        """Test getting materials for a scene."""
        materials = property_memory.get_materials(SceneType.KITCHEN)

        assert isinstance(materials, list)
        assert MaterialType.METAL in materials
        assert MaterialType.STONE in materials

    def test_add_processing_result(self, property_memory):
        """Test adding a processing result."""
        property_memory.add_processing_result(
            scene_type=SceneType.POOL,
            input_path="/test/input.jpg",
            output_path="/test/output.tif",
            parameters={'contrast': 1.1},
            quality_score=0.85,
            processing_time=2.5,
            success=True,
        )

        config = property_memory.get_room_config(SceneType.POOL)
        assert len(config.processing_history) > 0
        assert config.processing_history[-1].quality_score == 0.85

    def test_add_user_feedback(self, property_memory):
        """Test adding user feedback."""
        property_memory.add_user_feedback(
            scene_type=SceneType.GREAT_ROOM,
            feedback="Increase wood warmth",
            rating=0.9,
            suggested_parameters={'warmth': 12},
        )

        assert len(property_memory.feedback_records) > 0

    def test_learn_from_results_insufficient_data(self, property_memory):
        """Test learning with insufficient data."""
        result = property_memory.learn_from_results(SceneType.POOL, min_samples=10)

        assert result['status'] == 'insufficient_data'
        assert result['samples'] < 10

    def test_learn_from_results_with_data(self, property_memory):
        """Test learning with sufficient data."""
        # Add multiple processing results
        for i in range(5):
            property_memory.add_processing_result(
                scene_type=SceneType.POOL,
                input_path=f"/test/input_{i}.jpg",
                output_path=f"/test/output_{i}.tif",
                parameters={'contrast': 1.1 + i * 0.01, 'saturation': 1.05},
                quality_score=0.8 + i * 0.02,
                processing_time=2.5,
                success=True,
            )

        result = property_memory.learn_from_results(SceneType.POOL, min_samples=3)

        assert result['status'] == 'success'
        assert result['samples_analyzed'] >= 3
        assert 'learned_parameters' in result
        assert 'quality_trend' in result

    def test_get_property_knowledge(self, property_memory):
        """Test getting property knowledge summary."""
        knowledge = property_memory.get_property_knowledge()

        assert isinstance(knowledge, PropertyKnowledge)
        assert knowledge.property_name == "750 Picacho Lane"
        assert knowledge.total_scenes > 0

    def test_scene_type_from_filename(self, property_memory):
        """Test scene type detection from filename."""
        assert property_memory.get_scene_type_from_filename("750Picacho_Pool.exr") == SceneType.POOL
        assert property_memory.get_scene_type_from_filename("kitchen_render.jpg") == SceneType.KITCHEN
        assert property_memory.get_scene_type_from_filename("aerial_view.tif") == SceneType.AERIAL
        assert property_memory.get_scene_type_from_filename("unknown.jpg") is None

    def test_export_knowledge(self, property_memory, temp_dir):
        """Test exporting knowledge to file."""
        export_path = temp_dir / "knowledge_export.json"
        property_memory.export_knowledge(export_path)

        assert export_path.exists()

        with open(export_path, 'r') as f:
            data = json.load(f)

        assert 'property_knowledge' in data
        assert 'room_configurations' in data

    def test_persistence(self, temp_dir):
        """Test that memory persists across instances."""
        memory_path = temp_dir / "persist_test.json"

        # Create first instance and add data
        memory1 = PropertyMemory(memory_path)
        memory1.add_processing_result(
            scene_type=SceneType.POOL,
            input_path="/test/input.jpg",
            output_path="/test/output.tif",
            parameters={'test': True},
            quality_score=0.9,
            processing_time=1.0,
            success=True,
        )

        # Create second instance and verify data
        memory2 = PropertyMemory(memory_path)
        config = memory2.get_room_config(SceneType.POOL)

        assert len(config.processing_history) > 0


class TestRoomConfiguration:
    """Tests for RoomConfiguration dataclass."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = RoomConfiguration(
            scene_type=SceneType.POOL,
            materials=[MaterialType.WATER, MaterialType.STONE],
            optimal_parameters={'contrast': 1.1},
            notes="Test config",
        )

        data = config.to_dict()

        assert data['scene_type'] == 'pool'
        assert 'water' in data['materials']
        assert data['optimal_parameters']['contrast'] == 1.1

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            'scene_type': 'pool',
            'materials': ['water', 'stone'],
            'optimal_parameters': {'contrast': 1.1},
            'quality_baseline': 0.8,
            'processing_history': [],
            'notes': "Test config",
        }

        config = RoomConfiguration.from_dict(data)

        assert config.scene_type == SceneType.POOL
        assert MaterialType.WATER in config.materials
        assert config.optimal_parameters['contrast'] == 1.1


# ============================================================================
# Pipeline Tests
# ============================================================================


class TestPipelineConfig:
    """Tests for PipelineConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = PipelineConfig()

        assert config.quality_mode == "premium"
        assert config.enable_depth is True
        assert config.enable_learning is True
        assert 'tiff' in config.output_formats
        assert 'jpg' in config.output_formats

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PipelineConfig(quality_mode="fast")
        data = config.to_dict()

        assert data['quality_mode'] == "fast"
        assert isinstance(data['input_dir'], str)


class TestRAGEnhancedPipeline:
    """Tests for RAGEnhancedPipeline class."""

    def test_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.session_id is not None
        assert pipeline.processed_count == 0
        assert pipeline.memory is not None
        assert pipeline.knowledge is not None

    def test_detect_scene_type(self, pipeline, sample_image):
        """Test scene type detection."""
        scene_type = pipeline._detect_scene_type(sample_image)

        # Filename contains "Pool"
        assert scene_type == SceneType.POOL

    def test_get_rag_context(self, pipeline, sample_image):
        """Test RAG context retrieval."""
        context = pipeline._get_rag_context(SceneType.POOL, sample_image)

        assert isinstance(context, RAGContext)
        assert context.confidence > 0
        assert isinstance(context.recommended_parameters, dict)

    def test_load_image(self, pipeline, sample_image):
        """Test image loading."""
        img_array = pipeline._load_image(sample_image)

        assert isinstance(img_array, np.ndarray)
        assert img_array.dtype == np.float32
        assert img_array.min() >= 0
        assert img_array.max() <= 1
        assert len(img_array.shape) == 3
        assert img_array.shape[2] == 3

    def test_apply_color_grading(self, pipeline, sample_image):
        """Test color grading application."""
        img_array = pipeline._load_image(sample_image)
        params = {'contrast': 1.1, 'saturation': 1.05, 'temperature': 5}

        graded = pipeline._apply_color_grading(img_array, params)

        assert graded.shape == img_array.shape
        assert graded.dtype == np.float32

    def test_apply_material_response(self, pipeline, sample_image):
        """Test material response application."""
        img_array = pipeline._load_image(sample_image)
        materials = [MaterialType.WATER, MaterialType.STONE]
        params = {'water_enhance': True, 'water_saturation': 1.25}

        enhanced = pipeline._apply_material_response(img_array, materials, params)

        assert enhanced.shape == img_array.shape
        assert enhanced.dtype == np.float32

    def test_calculate_quality_score(self, pipeline, sample_image):
        """Test quality score calculation."""
        original = pipeline._load_image(sample_image)
        # Enhance contrast for processed version
        processed = np.clip((original - 0.5) * 1.2 + 0.5, 0, 1)

        score = pipeline._calculate_quality_score(original, processed)

        assert 0 <= score <= 1

    def test_process_image(self, pipeline, sample_image, temp_dir):
        """Test full image processing."""
        output_dir = temp_dir / "output"

        outputs, metrics = pipeline.process_image(sample_image, output_dir)

        assert isinstance(outputs, dict)
        assert isinstance(metrics, ProcessingMetrics)
        assert metrics.processing_time > 0
        assert len(metrics.stages_completed) > 0
        assert pipeline.processed_count == 1

    def test_process_image_saves_outputs(self, pipeline, sample_image, temp_dir):
        """Test that processing saves output files."""
        output_dir = temp_dir / "output"

        outputs, _ = pipeline.process_image(sample_image, output_dir)

        if 'tiff' in outputs:
            assert outputs['tiff'].exists()
        if 'jpg' in outputs:
            assert outputs['jpg'].exists()
        if 'thumbnail' in outputs:
            assert outputs['thumbnail'].exists()

    def test_batch_process(self, pipeline, sample_images, temp_dir):
        """Test batch processing."""
        output_dir = temp_dir / "batch_output"

        all_outputs, all_metrics = pipeline.batch_process(sample_images, output_dir)

        assert len(all_outputs) == len(sample_images)
        assert len(all_metrics) == len(sample_images)
        assert pipeline.processed_count == len(sample_images)

    def test_batch_process_with_callback(self, pipeline, sample_images, temp_dir):
        """Test batch processing with progress callback."""
        output_dir = temp_dir / "batch_output"
        progress_calls = []

        def callback(current, total, path):
            progress_calls.append((current, total, path))

        pipeline.batch_process(sample_images, output_dir, progress_callback=callback)

        assert len(progress_calls) == len(sample_images)
        # Verify order
        for i, (current, total, _) in enumerate(progress_calls, 1):
            assert current == i
            assert total == len(sample_images)

    def test_get_session_summary(self, pipeline, sample_image, temp_dir):
        """Test session summary generation."""
        output_dir = temp_dir / "output"

        # Process an image first
        pipeline.process_image(sample_image, output_dir)

        summary = pipeline.get_session_summary()

        assert summary['processed'] == 1
        assert summary['successful'] == 1
        assert summary['failed'] == 0
        assert summary['total_time'] > 0
        assert 'avg_quality_score' in summary

    def test_get_recommendations(self, pipeline, sample_image, temp_dir):
        """Test recommendation generation."""
        output_dir = temp_dir / "output"

        pipeline.process_image(sample_image, output_dir)

        recommendations = pipeline.get_recommendations()

        assert isinstance(recommendations, list)

    def test_learning_records_result(self, pipeline, sample_image, temp_dir):
        """Test that learning records processing results."""
        output_dir = temp_dir / "output"

        pipeline.process_image(sample_image, output_dir)

        # Check that result was recorded in memory
        config = pipeline.memory.get_room_config(SceneType.POOL)
        assert len(config.processing_history) > 0

    def test_process_image_failure_handling(self, pipeline, temp_dir):
        """Test handling of processing failures."""
        # Non-existent image
        fake_path = temp_dir / "nonexistent.jpg"

        outputs, metrics = pipeline.process_image(fake_path)

        assert len(outputs) == 0
        assert len(metrics.errors) > 0


class TestKnowledgeIntegrationBridge:
    """Tests for KnowledgeIntegrationBridge class."""

    def test_initialization(self):
        """Test bridge initialization."""
        bridge = KnowledgeIntegrationBridge()

        assert bridge._engine is None
        assert bridge._initialized is False

    def test_initialize(self):
        """Test initialization attempt."""
        bridge = KnowledgeIntegrationBridge()
        result = bridge.initialize()

        # Should return True (with or without engine)
        assert result is True
        assert bridge._initialized is True

    def test_is_available_without_engine(self):
        """Test availability check without engine."""
        bridge = KnowledgeIntegrationBridge()
        bridge._initialized = True

        assert bridge.is_available is False

    def test_add_feedback_without_engine(self):
        """Test feedback addition without engine (should not raise)."""
        bridge = KnowledgeIntegrationBridge()
        bridge.initialize()

        # Should not raise
        bridge.add_feedback(
            pipeline="test",
            artifact_id="test001",
            success=True,
            processing_time=1.0,
            parameters={},
        )

    def test_analyze_patterns_without_engine(self):
        """Test pattern analysis without engine."""
        bridge = KnowledgeIntegrationBridge()
        bridge.initialize()

        result = bridge.analyze_patterns("test")

        assert isinstance(result, dict)

    def test_query_knowledge_without_engine(self):
        """Test knowledge query works even when RAG engine is not available."""
        bridge = KnowledgeIntegrationBridge()
        bridge.initialize()
        # The bridge should work in fallback mode (without RAG engine)

        result = bridge.query_knowledge("Test query")

        # Should return some response (either "not available" or the engine's response)
        assert isinstance(result, str)
        assert len(result) > 0


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config_from_yaml(self, temp_dir):
        """Test loading configuration from YAML."""
        yaml_content = """
name: "Test Pipeline"
input:
  directory: "test_input"
output:
  directory: "test_output"
  formats:
    - tiff
    - jpg
  preserve_16bit: true
  jpeg_quality: 90
processing:
  quality_mode: "balanced"
  enable_depth: true
  enable_material_response: true
depth:
  model: "depth_anything_v2"
  zones: 3
  atmospheric_haze: false
material_response:
  strength: 0.8
  preserve_highlights: true
color_grading:
  lut_strength: 0.65
  saturation: 1.1
  contrast: 1.05
  temperature: 3.0
rag:
  enabled: true
  top_k: 10
learning:
  enabled: true
  threshold: 0.9
  min_samples: 5
"""
        config_path = temp_dir / "test_config.yaml"
        config_path.write_text(yaml_content)

        config = load_config_from_yaml(config_path)

        assert config.quality_mode == "balanced"
        assert config.depth_zones == 3
        assert config.material_strength == 0.8
        assert config.lut_strength == 0.65
        assert config.rag_top_k == 10
        assert config.min_samples_for_learning == 5


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for the full pipeline."""

    def test_end_to_end_processing(self, temp_dir):
        """Test complete end-to-end processing workflow."""
        # Create input directory and images
        input_dir = temp_dir / "input"
        output_dir = temp_dir / "output"
        input_dir.mkdir(parents=True)

        # Create test images
        for scene in ["Pool", "Kitchen"]:
            img_array = np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8)
            img = Image.fromarray(img_array, mode='RGB')
            img_path = input_dir / f"750Picacho_{scene}.jpg"
            img.save(img_path, format='JPEG')

        # Configure pipeline
        config = PipelineConfig(
            input_dir=input_dir,
            output_dir=output_dir,
            quality_mode="fast",
            enable_depth=False,
            enable_learning=True,
            preserve_16bit=False,  # Use 8-bit for testing
        )

        # Create pipeline
        memory_path = temp_dir / "memory" / "integration_memory.json"
        pipeline = RAGEnhancedPipeline(config=config, memory_path=memory_path)

        # Process images
        image_paths = list(input_dir.glob("*.jpg"))
        all_outputs, all_metrics = pipeline.batch_process(image_paths, output_dir)

        # Verify results
        assert len(all_outputs) == 2
        assert len(all_metrics) == 2
        assert all(len(m.errors) == 0 for m in all_metrics)

        # Verify session summary
        summary = pipeline.get_session_summary()
        assert summary['processed'] == 2
        assert summary['successful'] == 2

        # Verify memory was updated
        pool_config = pipeline.memory.get_room_config(SceneType.POOL)
        assert len(pool_config.processing_history) > 0

    def test_learning_improves_over_time(self, temp_dir):
        """Test that learning feedback loop improves parameters."""
        memory_path = temp_dir / "learning_memory.json"
        memory = PropertyMemory(memory_path)

        # Simulate multiple processing runs with improving quality
        for i in range(5):
            quality = 0.7 + i * 0.05  # Improving quality
            memory.add_processing_result(
                scene_type=SceneType.POOL,
                input_path=f"/test/input_{i}.jpg",
                output_path=f"/test/output_{i}.tif",
                parameters={'contrast': 1.0 + i * 0.02},
                quality_score=quality,
                processing_time=2.0,
                success=True,
            )

        # Trigger learning
        result = memory.learn_from_results(SceneType.POOL, min_samples=3)

        assert result['status'] == 'success'
        assert result['quality_trend'] in ['improving', 'stable']

        # Verify optimal parameters were updated
        config = memory.get_room_config(SceneType.POOL)
        assert config.quality_baseline > 0.7

    def test_rag_context_influences_processing(self, temp_dir):
        """Test that RAG context affects processing decisions."""
        # Create test image
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        img_path = temp_dir / "750Picacho_Pool_test.jpg"
        img.save(img_path, format='JPEG')

        output_dir = temp_dir / "output"

        # Configure pipeline
        config = PipelineConfig(
            output_dir=output_dir,
            quality_mode="fast",
            enable_depth=False,
            enable_learning=True,
            preserve_16bit=False,  # Use 8-bit for testing
        )

        # Create pipeline
        memory_path = temp_dir / "memory" / "rag_test_memory.json"
        pipeline = RAGEnhancedPipeline(config=config, memory_path=memory_path)

        # Add historical data to memory
        for i in range(3):
            pipeline.memory.add_processing_result(
                scene_type=SceneType.POOL,
                input_path=f"/test/input_{i}.jpg",
                output_path=f"/test/output_{i}.tif",
                parameters={'contrast': 1.15, 'saturation': 1.12},
                quality_score=0.9,
                processing_time=2.0,
                success=True,
            )

        # Process with RAG context
        outputs, metrics = pipeline.process_image(img_path, output_dir)

        # Verify RAG context was used
        assert metrics.rag_context_used is True


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error conditions."""

    def test_empty_batch(self, pipeline, temp_dir):
        """Test processing empty batch."""
        output_dir = temp_dir / "output"

        all_outputs, all_metrics = pipeline.batch_process([], output_dir)

        assert len(all_outputs) == 0
        assert len(all_metrics) == 0

    def test_unknown_scene_type(self, pipeline, temp_dir):
        """Test handling of unknown scene type."""
        # Create image with non-matching name
        img_array = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        img_path = temp_dir / "unknown_scene.jpg"
        img.save(img_path, format='JPEG')

        scene_type = pipeline._detect_scene_type(img_path)

        # Should default to EXTERIOR
        assert scene_type == SceneType.EXTERIOR

    def test_corrupted_memory_file(self, temp_dir):
        """Test handling of corrupted memory file."""
        memory_path = temp_dir / "memory" / "corrupted.json"
        memory_path.parent.mkdir(parents=True)
        memory_path.write_text("{ invalid json }")

        # Should initialize with defaults instead of crashing
        memory = PropertyMemory(memory_path)

        assert len(memory.room_configs) > 0

    def test_very_small_image(self, temp_dir):
        """Test processing very small image."""
        # Create tiny image
        img_array = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='RGB')
        img_path = temp_dir / "750Picacho_Pool_tiny.jpg"
        img.save(img_path, format='JPEG')

        # Create pipeline with 8-bit output
        config = PipelineConfig(
            output_dir=temp_dir / "output",
            quality_mode="fast",
            enable_depth=False,
            preserve_16bit=False,  # Use 8-bit to avoid tifffile dependency
        )
        memory_path = temp_dir / "memory" / "tiny_memory.json"
        pipeline = RAGEnhancedPipeline(config=config, memory_path=memory_path)

        output_dir = temp_dir / "output"
        outputs, metrics = pipeline.process_image(img_path, output_dir)

        # Should still process successfully
        assert len(metrics.errors) == 0

    def test_grayscale_image(self, temp_dir):
        """Test processing grayscale image (converted to RGB)."""
        img_array = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = Image.fromarray(img_array, mode='L')
        img_path = temp_dir / "750Picacho_Pool_gray.jpg"
        img.save(img_path, format='JPEG')

        # Create pipeline with 8-bit output
        config = PipelineConfig(
            output_dir=temp_dir / "output",
            quality_mode="fast",
            enable_depth=False,
            preserve_16bit=False,  # Use 8-bit to avoid tifffile dependency
        )
        memory_path = temp_dir / "memory" / "gray_memory.json"
        pipeline = RAGEnhancedPipeline(config=config, memory_path=memory_path)

        output_dir = temp_dir / "output"
        outputs, metrics = pipeline.process_image(img_path, output_dir)

        # Should convert and process successfully
        assert len(metrics.errors) == 0


# ============================================================================
# Main
# ============================================================================


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
