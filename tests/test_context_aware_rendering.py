#!/usr/bin/env python3
"""Tests for context-aware rendering pipeline integration.

Tests the integration of context-aware rendering with actual processing pipelines.
These tests mock the architectural_context_extractor to avoid PyMuPDF dependency.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from PIL import Image

# Add scripts directory to path for importing context_aware_rendering
scripts_dir = Path(__file__).parent.parent / "scripts"
sys.path.insert(0, str(scripts_dir))


# Mock classes for architectural_context_extractor
@dataclass
class MockRoomContext:
    """Mock room context for testing."""
    name: str
    dimensions: Optional[Tuple[float, float]] = None
    floor_level: Optional[str] = None
    ceiling_height: Optional[float] = None
    materials: List[str] = None
    features: List[str] = None
    adjacent_rooms: List[str] = None

    def __post_init__(self):
        if self.materials is None:
            self.materials = []
        if self.features is None:
            self.features = []
        if self.adjacent_rooms is None:
            self.adjacent_rooms = []


@dataclass
class MockProjectContext:
    """Mock project context for testing."""
    project_name: str
    project_number: Optional[str] = None
    address: Optional[str] = None
    architect: Optional[str] = None
    total_sqft: Optional[float] = None
    floors: List[str] = None
    rooms: Dict[str, MockRoomContext] = None
    materials_palette: List[str] = None
    design_style: Optional[str] = None
    extracted_images: List[str] = None
    raw_text: Optional[str] = None


@pytest.fixture
def sample_image(tmp_path):
    """Create a sample test image."""
    img_path = tmp_path / "test_kitchen.jpg"
    # Create a simple RGB image (100x100) with fixed seed for reproducibility
    rng = np.random.default_rng(seed=42)
    arr = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    img.save(img_path)
    return img_path


@pytest.fixture
def sample_context():
    """Create a sample project context."""
    rooms = {
        "kitchen_main": MockRoomContext(
            name="Kitchen",
            materials=["metal", "stone", "wood"],
            features=["island", "pantry"],
        ),
        "living_room": MockRoomContext(
            name="Living Room",
            materials=["wood", "fabric"],
            features=["fireplace"],
        ),
    }

    return MockProjectContext(
        project_name="Test Project",
        project_number="TP-001",
        rooms=rooms,
        materials_palette=["wood", "metal", "stone", "glass"],
        design_style="Modern Contemporary",
    )


@pytest.fixture
def pipeline(sample_context, tmp_path):
    """Create a ContextAwareRenderingPipeline instance."""
    # Mock the architectural_context_extractor module
    mock_module = MagicMock()
    mock_module.ProjectContext = MockProjectContext
    mock_module.RoomContext = MockRoomContext
    mock_module.ArchitecturalContextExtractor = MagicMock()

    with patch.dict(sys.modules, {'architectural_context_extractor': mock_module}):
        # Import after mocking
        from context_aware_rendering import ContextAwareRenderingPipeline
        return ContextAwareRenderingPipeline(
            project_context=sample_context,
            output_dir=tmp_path / "output",
        )


class TestRenderingStrategy:
    """Tests for RenderingStrategy dataclass."""

    def test_strategy_creation(self):
        """Test creating a RenderingStrategy."""
        # Mock the architectural_context_extractor module before import
        mock_module = MagicMock()
        mock_module.ProjectContext = MockProjectContext
        mock_module.RoomContext = MockRoomContext
        mock_module.ArchitecturalContextExtractor = MagicMock()

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_module}):
            from context_aware_rendering import RenderingStrategy

            strategy = RenderingStrategy(
                room_type="kitchen",
                primary_materials=["metal", "stone"],
                lighting_style="bright",
                depth_emphasis="balanced",
                color_temperature="neutral",
                enhancement_strength=0.75,
                lut_preset="signature_estate",
            )

            assert strategy.room_type == "kitchen"
            assert "metal" in strategy.primary_materials
            assert strategy.enhancement_strength == 0.75


class TestContextAwareRenderingPipeline:
    """Tests for ContextAwareRenderingPipeline class."""

    def test_init(self, pipeline, sample_context):
        """Test pipeline initialization."""
        assert pipeline.context == sample_context
        assert pipeline.output_dir.exists()

    def test_identify_room_from_filename_direct(self, pipeline):
        """Test room identification from filename - direct matches."""
        assert pipeline.identify_room_from_filename(Path("kitchen_render.jpg")) == "kitchen"
        assert pipeline.identify_room_from_filename(Path("bathroom_view.png")) == "bathroom"
        assert pipeline.identify_room_from_filename(Path("bedroom_main.tif")) == "bedroom"
        assert pipeline.identify_room_from_filename(Path("living_room.jpg")) == "living"
        assert pipeline.identify_room_from_filename(Path("outdoor_pool.jpg")) == "outdoor"

    def test_identify_room_from_filename_alias(self, pipeline):
        """Test room identification from filename - alias matches."""
        assert pipeline.identify_room_from_filename(Path("pool_view.jpg")) == "outdoor"
        assert pipeline.identify_room_from_filename(Path("master_suite.jpg")) == "bedroom"
        assert pipeline.identify_room_from_filename(Path("powder_room.jpg")) == "bathroom"

    def test_identify_room_from_filename_unknown(self, pipeline):
        """Test room identification returns None for unknown rooms."""
        assert pipeline.identify_room_from_filename(Path("unknown_space.jpg")) is None

    def test_derive_strategy_kitchen(self, pipeline, sample_image):
        """Test strategy derivation for kitchen image."""
        strategy = pipeline.derive_strategy(sample_image)

        assert strategy.room_type == "kitchen"
        assert strategy.lighting_style == "bright"
        assert strategy.color_temperature == "neutral"

    def test_derive_strategy_with_design_style(self, pipeline, tmp_path):
        """Test strategy adapts to design style."""
        # Create bedroom image (traditional style would affect it) with fixed seed
        img_path = tmp_path / "bedroom_test.jpg"
        rng = np.random.default_rng(seed=42)
        arr = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)

        strategy = pipeline.derive_strategy(img_path)

        # Modern style should keep neutral temperature
        assert strategy.color_temperature == "neutral"

    def test_generate_depth_config(self, pipeline):
        """Test depth configuration generation."""
        from context_aware_rendering import RenderingStrategy

        strategy = RenderingStrategy(
            room_type="kitchen",
            primary_materials=["metal"],
            lighting_style="bright",
            depth_emphasis="balanced",
            color_temperature="neutral",
            enhancement_strength=0.75,
        )

        config = pipeline.generate_depth_config(strategy)

        assert "zone_weights" in config
        assert config["tone_map"] == "reinhard"  # bright lighting
        assert config["zone_weights"]["midground"] == 1.0

    def test_generate_depth_config_atmospheric(self, pipeline):
        """Test depth config for atmospheric emphasis."""
        from context_aware_rendering import RenderingStrategy

        strategy = RenderingStrategy(
            room_type="outdoor",
            primary_materials=["stone"],
            lighting_style="natural",
            depth_emphasis="atmospheric",
            color_temperature="neutral",
            enhancement_strength=0.8,
        )

        config = pipeline.generate_depth_config(strategy)

        assert config["zone_weights"]["background"] == 1.0
        assert config["zone_weights"]["foreground"] == 0.6

    def test_generate_material_config(self, pipeline):
        """Test material response configuration generation."""
        from context_aware_rendering import RenderingStrategy

        strategy = RenderingStrategy(
            room_type="kitchen",
            primary_materials=["metal", "stone", "wood"],
            lighting_style="bright",
            depth_emphasis="balanced",
            color_temperature="neutral",
            enhancement_strength=0.75,
        )

        config = pipeline.generate_material_config(strategy)

        assert config["enabled_surfaces"] == ["metal", "stone", "wood"]
        assert config["global_strength"] == 0.75
        assert "material_strengths" in config
        assert config["material_strengths"]["metal"] >= config["material_strengths"]["wood"]

    def test_generate_color_config(self, pipeline):
        """Test color grading configuration generation."""
        from context_aware_rendering import RenderingStrategy

        strategy = RenderingStrategy(
            room_type="bedroom",
            primary_materials=["wood", "fabric"],
            lighting_style="soft",
            depth_emphasis="atmospheric",
            color_temperature="warm",
            enhancement_strength=0.6,
            lut_preset="warm_invitation",
        )

        config = pipeline.generate_color_config(strategy)

        assert config["lut_preset"] == "warm_invitation"
        assert config["saturation"] == 1.08  # warm temperature
        assert config["tint"] == 5  # warm temperature

    def test_process_render_saves_strategy(self, pipeline, sample_image):
        """Test that process_render saves strategy JSON."""
        result = pipeline.process_render(
            sample_image,
            apply_depth=False,  # Skip depth for faster test
            apply_material=False,
            apply_color=False,
        )

        # Check result structure
        assert isinstance(result, dict)
        assert "strategy_path" in result
        assert "output_path" in result

        # Check strategy file was created
        strategy_path = result["strategy_path"]
        assert strategy_path.exists()

        # Validate strategy JSON content
        with open(strategy_path) as f:
            strategy_data = json.load(f)

        assert "strategy" in strategy_data
        assert strategy_data["strategy"]["room_type"] == "kitchen"

    def test_process_render_with_color_grading(self, pipeline, sample_image):
        """Test process_render applies color grading when available."""
        result = pipeline.process_render(
            sample_image,
            apply_depth=False,
            apply_material=False,
            apply_color=True,
        )

        assert "output_path" in result
        assert result["output_path"].exists()

        # Check if color grading was applied
        if "color_grading" in result.get("processing_applied", []):
            # Verify output image exists and is valid
            with Image.open(result["output_path"]) as img:
                assert img.size == (100, 100)


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_image_to_array(self, sample_image):
        """Test image loading as normalized array."""
        # Mock the architectural_context_extractor module before import
        mock_module = MagicMock()
        mock_module.ProjectContext = MockProjectContext
        mock_module.RoomContext = MockRoomContext
        mock_module.ArchitecturalContextExtractor = MagicMock()

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_module}):
            from context_aware_rendering import _image_to_array

            arr = _image_to_array(sample_image)

            assert arr.dtype == np.float32
            assert arr.min() >= 0.0
            assert arr.max() <= 1.0
            assert arr.shape == (100, 100, 3)

    def test_array_to_image(self, tmp_path):
        """Test saving array as image."""
        # Mock the architectural_context_extractor module before import
        mock_module = MagicMock()
        mock_module.ProjectContext = MockProjectContext
        mock_module.RoomContext = MockRoomContext
        mock_module.ArchitecturalContextExtractor = MagicMock()

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_module}):
            from context_aware_rendering import _array_to_image

            # Use fixed seed for reproducibility
            rng = np.random.default_rng(seed=42)
            arr = rng.random((50, 50, 3)).astype(np.float32)
            output_path = tmp_path / "output.png"

            _array_to_image(arr, output_path)

            assert output_path.exists()
            with Image.open(output_path) as img:
                assert img.size == (50, 50)

    def test_check_depth_pipeline(self):
        """Test depth pipeline availability check."""
        # Mock the architectural_context_extractor module before import
        mock_module = MagicMock()
        mock_module.ProjectContext = MockProjectContext
        mock_module.RoomContext = MockRoomContext
        mock_module.ArchitecturalContextExtractor = MagicMock()

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_module}):
            from context_aware_rendering import _check_depth_pipeline

            # Should return True or False without raising
            result = _check_depth_pipeline()
            assert isinstance(result, bool)

    def test_check_tiff_processor(self):
        """Test TIFF processor availability check."""
        # Mock the architectural_context_extractor module before import
        mock_module = MagicMock()
        mock_module.ProjectContext = MockProjectContext
        mock_module.RoomContext = MockRoomContext
        mock_module.ArchitecturalContextExtractor = MagicMock()

        with patch.dict(sys.modules, {'architectural_context_extractor': mock_module}):
            from context_aware_rendering import _check_tiff_processor

            # Should return True when luxury_tiff_batch_processor is installed
            result = _check_tiff_processor()
            assert isinstance(result, bool)


class TestDepthPipelineIntegration:
    """Tests for depth pipeline integration."""

    def test_build_depth_pipeline_config(self, pipeline):
        """Test depth pipeline config building."""
        depth_config = {
            "model_size": "small",
            "device": "pytorch_cpu",
            "tone_map": "agx",
            "zone_weights": {
                "foreground": 0.8,
                "midground": 1.0,
                "background": 0.8,
            },
        }

        full_config = pipeline._build_depth_pipeline_config(depth_config)

        assert "depth_model" in full_config
        assert full_config["depth_model"]["variant"] == "small"
        assert "processing" in full_config
        assert full_config["processing"]["zone_tone_mapping"]["enabled"] is True

    def test_apply_depth_processing_unavailable(self, pipeline, sample_image):
        """Test graceful handling when depth pipeline unavailable."""
        from context_aware_rendering import _image_to_array

        arr = _image_to_array(sample_image)

        # Mock unavailable pipeline
        with patch("context_aware_rendering._check_depth_pipeline", return_value=False):
            result = pipeline._apply_depth_processing(arr, {}, sample_image)

        assert result is None


class TestColorGradingIntegration:
    """Tests for color grading integration."""

    def test_get_temp_from_strategy(self, pipeline):
        """Test temperature mapping from strategy."""
        from context_aware_rendering import RenderingStrategy

        warm_strategy = RenderingStrategy(
            room_type="bedroom",
            primary_materials=["wood"],
            lighting_style="soft",
            depth_emphasis="atmospheric",
            color_temperature="warm",
            enhancement_strength=0.6,
        )

        assert pipeline._get_temp_from_strategy(warm_strategy) == 5600.0

        neutral_strategy = RenderingStrategy(
            room_type="kitchen",
            primary_materials=["metal"],
            lighting_style="bright",
            depth_emphasis="balanced",
            color_temperature="neutral",
            enhancement_strength=0.75,
        )

        assert pipeline._get_temp_from_strategy(neutral_strategy) == 6500.0

    def test_apply_color_grading_with_processor(self, pipeline, sample_image):
        """Test color grading when TIFF processor is available."""
        from context_aware_rendering import RenderingStrategy, _image_to_array

        arr = _image_to_array(sample_image)
        strategy = RenderingStrategy(
            room_type="kitchen",
            primary_materials=["metal"],
            lighting_style="bright",
            depth_emphasis="balanced",
            color_temperature="neutral",
            enhancement_strength=0.75,
        )
        color_config = {"saturation": 1.05, "tint": 0}

        result = pipeline._apply_color_grading(arr, color_config, strategy)

        # If processor is available, result should be a numpy array
        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.shape == arr.shape


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_execution(self, pipeline, sample_image):
        """Test complete pipeline execution."""
        result = pipeline.process_render(
            sample_image,
            apply_depth=False,  # Skip depth (requires ML models)
            apply_material=True,
            apply_color=True,
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "output_path" in result
        assert "strategy_path" in result
        assert "processing_applied" in result
        assert "strategy" in result

        # Verify output file exists
        assert result["output_path"].exists()

        # Verify strategy is captured
        assert result["strategy"].room_type == "kitchen"

    def test_pipeline_with_unknown_room(self, pipeline, tmp_path):
        """Test pipeline handles unknown room gracefully."""
        # Create image with unrecognizable name using fixed seed
        img_path = tmp_path / "random_space_xyz.jpg"
        rng = np.random.default_rng(seed=42)
        arr = rng.integers(0, 255, (50, 50, 3), dtype=np.uint8)
        Image.fromarray(arr).save(img_path)

        result = pipeline.process_render(
            img_path,
            apply_depth=False,
            apply_material=False,
            apply_color=True,
        )

        # Should still produce output
        assert result["output_path"].exists()
        assert result["strategy"].room_type == "unknown"
