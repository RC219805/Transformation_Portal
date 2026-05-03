from __future__ import annotations

import pytest

from transformation_portal.vlm_captioning.parser import parse_fastvlm_caption

pytestmark = pytest.mark.unit


def test_parse_05b_malformed_colon_xml_output() -> None:
    raw = """
    <answer>
    SCENE: Swimming pool patio; MATERIALS: stone, tile, concrete;
    FEATURES: pool edge, steps; NATURAL: sky, trees; LIGHTING: daylight;
    ISSUES: no visible quality issues; UNCERTAIN: distant material labels.
    </answer>
    """

    parsed = parse_fastvlm_caption(raw)

    assert parsed.validated is True
    assert parsed.caption["scene"] == "Swimming pool patio"
    assert parsed.caption["materials"] == ["stone", "tile", "concrete"]
    assert parsed.caption["features"] == ["pool edge", "steps"]
    assert parsed.caption["natural"] == ["sky", "trees"]
    assert parsed.caption["lighting"] == "daylight"
    assert parsed.caption["issues"] == ["no visible quality issues"]
    assert parsed.caption["uncertain"] == ["distant material labels"]


def test_parse_15b_clean_output() -> None:
    raw = (
        "SCENE=Swimming pool; MATERIALS=Concrete, tiles, metal; "
        "FEATURES=Architectural details, landscaping; NATURAL=Greenery, sky; "
        "LIGHTING=Daylight; ISSUES=No apparent issues; UNCERTAIN=No apparent issues."
    )

    parsed = parse_fastvlm_caption(raw)

    assert parsed.validated is True
    assert parsed.caption == {
        "scene": "Swimming pool",
        "materials": ["Concrete", "tiles", "metal"],
        "features": ["Architectural details", "landscaping"],
        "natural": ["Greenery", "sky"],
        "lighting": "Daylight",
        "issues": ["No apparent issues"],
        "uncertain": ["No apparent issues"],
    }


def test_parse_7b_clean_output_with_pipe_delimiters() -> None:
    raw = (
        "SCENE=Luxury exterior pool | MATERIALS=stone, plaster, glass, metal | "
        "FEATURES=pool, terrace, railing | NATURAL=trees, hillside, sky | "
        "LIGHTING=soft daylight | ISSUES=none apparent | UNCERTAIN=small distant objects"
    )

    parsed = parse_fastvlm_caption(raw)

    assert parsed.validated is True
    assert parsed.caption["scene"] == "Luxury exterior pool"
    assert parsed.caption["materials"] == ["stone", "plaster", "glass", "metal"]


def test_missing_keys_marks_unvalidated_without_fabricating_fields() -> None:
    parsed = parse_fastvlm_caption("SCENE=Patio; MATERIALS=stone.")

    assert parsed.validated is False
    assert parsed.missing_keys == ["features", "natural", "lighting", "issues", "uncertain"]
    assert parsed.caption == {"scene": "Patio", "materials": ["stone"]}
