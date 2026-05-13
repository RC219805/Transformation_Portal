"""CPU/core parser tests for VLM scene analysis."""

from __future__ import annotations

from collections import deque
from typing import Iterable

import pytest

from transformation_portal.vlm.scene_analyzer import (
    ArchitecturalStyle,
    RoomType,
    SceneAnalysis,
    SceneAnalyzer,
    SpaceType,
)

pytestmark = [pytest.mark.unit]


class FakeProcessor:
    """Deterministic processor stub that never touches model runtimes."""

    def __init__(self, responses: str | Iterable[str]):
        if isinstance(responses, str):
            responses = [responses]
        self._responses = deque(responses)
        self.calls: list[dict[str, object]] = []

    def analyze_image(self, image, **kwargs):
        self.calls.append({"image": image, **kwargs})
        return self._responses.popleft()


def _analyzer(response: str = "Interior kitchen with marble and natural light.") -> SceneAnalyzer:
    return SceneAnalyzer(llava_processor=FakeProcessor(response))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Interior room with indoor seating", SpaceType.INTERIOR),
        ("Exterior facade and outdoor courtyard", SpaceType.EXTERIOR),
        ("Drone aerial overhead estate view", SpaceType.AERIAL),
        ("Abstract architectural mood board", SpaceType.UNKNOWN),
    ],
)
def test_extract_space_type_matrix(text: str, expected: SpaceType) -> None:
    assert _analyzer()._extract_space_type(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("chef kitchen with island", RoomType.KITCHEN),
        ("spa bathroom with stone tub", RoomType.BATHROOM),
        ("primary bedroom suite", RoomType.BEDROOM),
        ("open living room lounge", RoomType.LIVING),
        ("formal dining area", RoomType.DINING),
        ("library and office", RoomType.OFFICE),
        ("poolside pool area", RoomType.POOL_AREA),
        ("courtyard patio terrace", RoomType.COURTYARD),
        ("foyer entryway entrance", RoomType.ENTRY),
        ("gallery corridor", RoomType.UNKNOWN),
    ],
)
def test_extract_room_type_matrix(text: str, expected: RoomType) -> None:
    assert _analyzer()._extract_room_type(text) is expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Spanish Mediterranean villa with terracotta roof", ArchitecturalStyle.MEDITERRANEAN),
        ("Modern home with clean mid-century lines", ArchitecturalStyle.MODERN),
        ("Coastal beach estate by the sea", ArchitecturalStyle.COASTAL),
        ("Palatial luxury estate with grand entry", ArchitecturalStyle.LUXURY_ESTATE),
        ("Unclassified architectural reference", ArchitecturalStyle.UNKNOWN),
    ],
)
def test_extract_style_matrix(text: str, expected: ArchitecturalStyle) -> None:
    assert _analyzer()._extract_style(text) is expected


def test_extract_materials_covers_luxury_material_keywords() -> None:
    text = "Marble counters, walnut wood floors, glass walls, metal railings, stone fireplace, fabric drapery."

    materials = _analyzer()._extract_materials(text)

    assert {"marble", "wood", "glass", "metal", "stone", "fabric"}.issubset(materials)


def test_extract_luxury_features_covers_expected_keywords() -> None:
    text = "High ceilings, designer fixtures, ocean view, and smart home automation define the room."

    features = _analyzer()._extract_luxury_features(text)

    assert {"high ceiling", "designer", "ocean view", "smart home"}.issubset(features)


def test_extract_lighting_prefers_explicit_section() -> None:
    text = "LIGHTING: Soft natural light from clerestory windows\nOther notes follow."

    assert _analyzer()._extract_lighting(text) == "soft natural light from clerestory windows"


def test_extract_lighting_falls_back_to_keyword() -> None:
    assert _analyzer()._extract_lighting("The room is lit by dramatic light across stone.") == "dramatic light"


def test_analyze_uses_fake_processor_and_parses_structured_response() -> None:
    fake = FakeProcessor("""
1. SPACE TYPE: Interior
2. ROOM TYPE: Kitchen
3. ARCHITECTURAL STYLE: Modern coastal
4. MATERIALS: marble, wood, glass, metal
5. LUXURY FEATURES: high ceilings, designer fixtures, ocean view, smart home
6. LIGHTING: Natural light from large windows
""")
    analyzer = SceneAnalyzer(llava_processor=fake)

    analysis = analyzer.analyze("image-token")

    assert analysis.space_type is SpaceType.INTERIOR
    assert analysis.room_type is RoomType.KITCHEN
    assert analysis.architectural_style is ArchitecturalStyle.MODERN
    assert {"marble", "wood", "glass", "metal"}.issubset(analysis.materials)
    assert {"high ceiling", "designer", "ocean view", "smart home"}.issubset(analysis.luxury_features)
    assert analysis.lighting_conditions == "natural light from large windows"
    assert fake.calls[0]["image"] == "image-token"
    assert fake.calls[0]["temperature"] == 0.1


@pytest.mark.parametrize(
    ("analysis", "expected"),
    [
        (
            SceneAnalysis(
                space_type=SpaceType.INTERIOR,
                room_type=RoomType.KITCHEN,
                architectural_style=ArchitecturalStyle.MODERN,
                materials=["marble"],
                luxury_features=[],
                lighting_conditions="standard lighting",
                confidence=0.85,
                raw_analysis="",
            ),
            {
                "suggested_preset": "kitchen-bright",
                "enhancement_strength": pytest.approx(0.405),
                "material_response_strength": 0.75,
            },
        ),
        (
            SceneAnalysis(
                space_type=SpaceType.INTERIOR,
                room_type=RoomType.BATHROOM,
                architectural_style=ArchitecturalStyle.UNKNOWN,
                materials=["stone"],
                luxury_features=[],
                lighting_conditions="standard lighting",
                confidence=0.85,
                raw_analysis="",
            ),
            {
                "suggested_preset": "bathroom-spa",
                "enhancement_strength": 0.4,
                "material_response_strength": 0.7,
            },
        ),
        (
            SceneAnalysis(
                space_type=SpaceType.INTERIOR,
                room_type=RoomType.BEDROOM,
                architectural_style=ArchitecturalStyle.UNKNOWN,
                materials=["fabric"],
                luxury_features=[],
                lighting_conditions="standard lighting",
                confidence=0.85,
                raw_analysis="",
            ),
            {"suggested_preset": "bedroom-cozy", "enhancement_strength": 0.35},
        ),
        (
            SceneAnalysis(
                space_type=SpaceType.INTERIOR,
                room_type=RoomType.POOL_AREA,
                architectural_style=ArchitecturalStyle.UNKNOWN,
                materials=["tile"],
                luxury_features=["infinity pool"],
                lighting_conditions="standard lighting",
                confidence=0.85,
                raw_analysis="",
            ),
            {
                "suggested_preset": "pool-luxury",
                "enhancement_strength": 0.5,
                "atmospheric_effects": True,
            },
        ),
        (
            SceneAnalysis(
                space_type=SpaceType.EXTERIOR,
                room_type=None,
                architectural_style=ArchitecturalStyle.MEDITERRANEAN,
                materials=["stone"],
                luxury_features=[],
                lighting_conditions="golden hour",
                confidence=0.85,
                raw_analysis="",
            ),
            {
                "atmospheric_effects": True,
                "color_grading": "california-golden-hour",
            },
        ),
        (
            SceneAnalysis(
                space_type=SpaceType.AERIAL,
                room_type=None,
                architectural_style=ArchitecturalStyle.LUXURY_ESTATE,
                materials=["stone"],
                luxury_features=["ocean view"],
                lighting_conditions="natural light",
                confidence=0.85,
                raw_analysis="",
            ),
            {
                "suggested_preset": "aerial-estate",
                "atmospheric_effects": True,
                "preserve_lighting": True,
            },
        ),
    ],
)
def test_get_processing_recommendations_matrix(
    analysis: SceneAnalysis,
    expected: dict[str, object],
) -> None:
    recommendations = _analyzer().get_processing_recommendations(analysis)

    for key, expected_value in expected.items():
        assert recommendations[key] == expected_value
