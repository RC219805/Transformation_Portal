"""Tests for vlm/scene_analyzer.py module (Phase 5 coverage).

Tests for:
- SpaceType enum
- RoomType enum
- ArchitecturalStyle enum
- SceneAnalysis dataclass
- SceneAnalyzer class (mocked)

All tests use mocks - no ML model downloads or GPU requirements.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Skip all tests if torch not available (required by LLaVA/VLM)
torch = pytest.importorskip("torch", reason="torch required for VLM tests")

pytestmark = [pytest.mark.unit, pytest.mark.ml]


class TestSpaceType:
    """Test SpaceType enum."""

    def test_all_types_defined(self):
        """Test all space types are defined."""
        from transformation_portal.vlm.scene_analyzer import SpaceType

        assert SpaceType.INTERIOR.value == "interior"
        assert SpaceType.EXTERIOR.value == "exterior"
        assert SpaceType.AERIAL.value == "aerial"
        assert SpaceType.UNKNOWN.value == "unknown"


class TestRoomType:
    """Test RoomType enum."""

    def test_all_types_defined(self):
        """Test all room types are defined."""
        from transformation_portal.vlm.scene_analyzer import RoomType

        assert RoomType.LIVING.value == "living_room"
        assert RoomType.KITCHEN.value == "kitchen"
        assert RoomType.BEDROOM.value == "bedroom"
        assert RoomType.BATHROOM.value == "bathroom"
        assert RoomType.DINING.value == "dining_room"
        assert RoomType.OFFICE.value == "office"
        assert RoomType.POOL_AREA.value == "pool_area"
        assert RoomType.COURTYARD.value == "courtyard"
        assert RoomType.ENTRY.value == "entry"
        assert RoomType.UNKNOWN.value == "unknown"


class TestArchitecturalStyle:
    """Test ArchitecturalStyle enum."""

    def test_all_styles_defined(self):
        """Test all architectural styles are defined."""
        from transformation_portal.vlm.scene_analyzer import ArchitecturalStyle

        assert ArchitecturalStyle.MODERN.value == "modern"
        assert ArchitecturalStyle.CONTEMPORARY.value == "contemporary"
        assert ArchitecturalStyle.TRADITIONAL.value == "traditional"
        assert ArchitecturalStyle.MEDITERRANEAN.value == "mediterranean"
        assert ArchitecturalStyle.COASTAL.value == "coastal"
        assert ArchitecturalStyle.TRANSITIONAL.value == "transitional"
        assert ArchitecturalStyle.INDUSTRIAL.value == "industrial"
        assert ArchitecturalStyle.MINIMALIST.value == "minimalist"
        assert ArchitecturalStyle.LUXURY_ESTATE.value == "luxury_estate"
        assert ArchitecturalStyle.UNKNOWN.value == "unknown"


class TestSceneAnalysis:
    """Test SceneAnalysis dataclass."""

    def test_basic_creation(self):
        """Test basic scene analysis creation."""
        from transformation_portal.vlm.scene_analyzer import (
            ArchitecturalStyle,
            RoomType,
            SceneAnalysis,
            SpaceType,
        )

        analysis = SceneAnalysis(
            space_type=SpaceType.INTERIOR,
            room_type=RoomType.KITCHEN,
            architectural_style=ArchitecturalStyle.MODERN,
            materials=["marble", "stainless steel", "glass"],
            luxury_features=["high ceiling", "designer fixtures"],
            lighting_conditions="natural light",
            confidence=0.9,
            raw_analysis="Full analysis text...",
        )

        assert analysis.space_type == SpaceType.INTERIOR
        assert analysis.room_type == RoomType.KITCHEN
        assert len(analysis.materials) == 3
        assert analysis.confidence == 0.9

    def test_exterior_analysis_no_room_type(self):
        """Test exterior analysis has no room type."""
        from transformation_portal.vlm.scene_analyzer import (
            ArchitecturalStyle,
            SceneAnalysis,
            SpaceType,
        )

        analysis = SceneAnalysis(
            space_type=SpaceType.EXTERIOR,
            room_type=None,
            architectural_style=ArchitecturalStyle.MEDITERRANEAN,
            materials=["stone", "stucco"],
            luxury_features=["fountain", "landscaping"],
            lighting_conditions="golden hour",
            confidence=0.85,
            raw_analysis="...",
        )

        assert analysis.room_type is None
        assert analysis.space_type == SpaceType.EXTERIOR


class TestSceneAnalyzerMocked:
    """Test SceneAnalyzer with mocked LLaVA processor."""

    @pytest.fixture
    def mock_llava_processor(self):
        """Create mocked LLaVA processor."""
        mock = MagicMock()
        mock.analyze_image = MagicMock(return_value="""
1. SPACE TYPE: Interior

2. ROOM TYPE: This is a modern kitchen with open layout

3. ARCHITECTURAL STYLE: Modern contemporary design

4. MATERIALS: Marble countertops, stainless steel appliances, glass backsplash, hardwood floors

5. LUXURY FEATURES: High ceiling, designer fixtures, premium appliances, custom cabinetry

6. LIGHTING: Natural light from large windows
""")
        return mock

    def test_analyzer_initialization(self, mock_llava_processor):
        """Test analyzer initialization."""
        from transformation_portal.vlm.scene_analyzer import SceneAnalyzer

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)

        assert analyzer.processor == mock_llava_processor

    def test_analyze_returns_scene_analysis(self, mock_llava_processor, tmp_path):
        """Test analyze returns SceneAnalysis object."""
        from transformation_portal.vlm.scene_analyzer import SceneAnalysis, SceneAnalyzer

        # Create test image
        img_path = tmp_path / "test.png"
        Image.new("RGB", (100, 100), color="red").save(img_path)

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            result = analyzer.analyze(img_path)

        assert isinstance(result, SceneAnalysis)

    def test_extract_space_type_interior(self, mock_llava_processor):
        """Test space type extraction - interior."""
        from transformation_portal.vlm.scene_analyzer import SceneAnalyzer, SpaceType

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            space_type = analyzer._extract_space_type("This is an interior view of a room")

        assert space_type == SpaceType.INTERIOR

    def test_extract_space_type_exterior(self, mock_llava_processor):
        """Test space type extraction - exterior."""
        from transformation_portal.vlm.scene_analyzer import SceneAnalyzer, SpaceType

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            space_type = analyzer._extract_space_type("This is an exterior facade view")

        assert space_type == SpaceType.EXTERIOR

    def test_extract_space_type_aerial(self, mock_llava_processor):
        """Test space type extraction - aerial."""
        from transformation_portal.vlm.scene_analyzer import SceneAnalyzer, SpaceType

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            space_type = analyzer._extract_space_type("Drone aerial shot of the property")

        assert space_type == SpaceType.AERIAL

    def test_extract_room_type_kitchen(self, mock_llava_processor):
        """Test room type extraction - kitchen."""
        from transformation_portal.vlm.scene_analyzer import RoomType, SceneAnalyzer

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            room_type = analyzer._extract_room_type("This is a gourmet kitchen with island")

        assert room_type == RoomType.KITCHEN

    def test_extract_room_type_bathroom(self, mock_llava_processor):
        """Test room type extraction - bathroom."""
        from transformation_portal.vlm.scene_analyzer import RoomType, SceneAnalyzer

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            room_type = analyzer._extract_room_type("Spa-like master bathroom")

        assert room_type == RoomType.BATHROOM

    def test_extract_room_type_living(self, mock_llava_processor):
        """Test room type extraction - living room."""
        from transformation_portal.vlm.scene_analyzer import RoomType, SceneAnalyzer

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            room_type = analyzer._extract_room_type("Open living room with fireplace")

        assert room_type == RoomType.LIVING

    def test_extract_style_modern(self, mock_llava_processor):
        """Test architectural style extraction - modern."""
        from transformation_portal.vlm.scene_analyzer import ArchitecturalStyle, SceneAnalyzer

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            style = analyzer._extract_style("Modern architectural design with clean lines")

        assert style == ArchitecturalStyle.MODERN

    def test_extract_style_mediterranean(self, mock_llava_processor):
        """Test architectural style extraction - mediterranean."""
        from transformation_portal.vlm.scene_analyzer import ArchitecturalStyle, SceneAnalyzer

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            style = analyzer._extract_style("Mediterranean villa with terracotta tiles")

        assert style == ArchitecturalStyle.MEDITERRANEAN

    def test_extract_materials(self, mock_llava_processor):
        """Test material extraction."""
        from transformation_portal.vlm.scene_analyzer import SceneAnalyzer

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            materials = analyzer._extract_materials("Marble countertops, oak hardwood floors, glass windows")

        assert "marble" in materials
        assert "oak" in materials
        assert "glass" in materials

    def test_extract_luxury_features(self, mock_llava_processor):
        """Test luxury feature extraction."""
        from transformation_portal.vlm.scene_analyzer import SceneAnalyzer

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            features = analyzer._extract_luxury_features(
                "High ceiling with chandelier, custom built-in, smart home automation"
            )

        assert "high ceiling" in features
        assert "chandelier" in features

    def test_extract_lighting(self, mock_llava_processor):
        """Test lighting extraction."""
        from transformation_portal.vlm.scene_analyzer import SceneAnalyzer

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            lighting = analyzer._extract_lighting("Lighting: Golden hour natural light")

        assert "golden hour" in lighting

    def test_get_processing_recommendations_kitchen(self, mock_llava_processor):
        """Test processing recommendations for kitchen."""
        from transformation_portal.vlm.scene_analyzer import (
            ArchitecturalStyle,
            RoomType,
            SceneAnalysis,
            SceneAnalyzer,
            SpaceType,
        )

        analysis = SceneAnalysis(
            space_type=SpaceType.INTERIOR,
            room_type=RoomType.KITCHEN,
            architectural_style=ArchitecturalStyle.MODERN,
            materials=["marble"],
            luxury_features=[],
            lighting_conditions="natural light",
            confidence=0.9,
            raw_analysis="...",
        )

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            recs = analyzer.get_processing_recommendations(analysis)

        assert recs["suggested_preset"] == "kitchen-bright"
        assert recs["scene_type"] == "interior"

    def test_get_processing_recommendations_pool(self, mock_llava_processor):
        """Test processing recommendations for pool area."""
        from transformation_portal.vlm.scene_analyzer import (
            ArchitecturalStyle,
            RoomType,
            SceneAnalysis,
            SceneAnalyzer,
            SpaceType,
        )

        analysis = SceneAnalysis(
            space_type=SpaceType.INTERIOR,
            room_type=RoomType.POOL_AREA,
            architectural_style=ArchitecturalStyle.LUXURY_ESTATE,
            materials=["tile"],
            luxury_features=["infinity pool"],
            lighting_conditions="afternoon",
            confidence=0.85,
            raw_analysis="...",
        )

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            recs = analyzer.get_processing_recommendations(analysis)

        assert recs["suggested_preset"] == "pool-luxury"
        assert recs["atmospheric_effects"] is True

    def test_get_processing_recommendations_aerial(self, mock_llava_processor):
        """Test processing recommendations for aerial view."""
        from transformation_portal.vlm.scene_analyzer import (
            ArchitecturalStyle,
            SceneAnalysis,
            SceneAnalyzer,
            SpaceType,
        )

        analysis = SceneAnalysis(
            space_type=SpaceType.AERIAL,
            room_type=None,
            architectural_style=ArchitecturalStyle.LUXURY_ESTATE,
            materials=["stone"],
            luxury_features=["panoramic view"],
            lighting_conditions="midday",
            confidence=0.9,
            raw_analysis="...",
        )

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            recs = analyzer.get_processing_recommendations(analysis)

        assert recs["suggested_preset"] == "aerial-estate"
        assert recs["atmospheric_effects"] is True

    def test_get_processing_recommendations_golden_hour(self, mock_llava_processor):
        """Test processing recommendations for golden hour lighting."""
        from transformation_portal.vlm.scene_analyzer import (
            ArchitecturalStyle,
            RoomType,
            SceneAnalysis,
            SceneAnalyzer,
            SpaceType,
        )

        analysis = SceneAnalysis(
            space_type=SpaceType.INTERIOR,
            room_type=RoomType.LIVING,
            architectural_style=ArchitecturalStyle.COASTAL,
            materials=["wood"],
            luxury_features=[],
            lighting_conditions="golden hour natural light",
            confidence=0.9,
            raw_analysis="...",
        )

        with patch("transformation_portal.vlm.scene_analyzer.LLaVAProcessor"):
            analyzer = SceneAnalyzer(llava_processor=mock_llava_processor)
            recs = analyzer.get_processing_recommendations(analysis)

        assert "color_grading" in recs
        assert "golden-hour" in recs["color_grading"]


class TestPromptConstants:
    """Test prompt constant definitions."""

    def test_structured_prompt_exists(self):
        """Test structured analysis prompt exists."""
        from transformation_portal.vlm.scene_analyzer import SceneAnalyzer

        assert hasattr(SceneAnalyzer, "STRUCTURED_ANALYSIS_PROMPT")
        prompt = SceneAnalyzer.STRUCTURED_ANALYSIS_PROMPT

        assert "SPACE TYPE" in prompt
        assert "ROOM TYPE" in prompt
        assert "ARCHITECTURAL STYLE" in prompt
        assert "MATERIALS" in prompt
        assert "LUXURY FEATURES" in prompt
        assert "LIGHTING" in prompt
