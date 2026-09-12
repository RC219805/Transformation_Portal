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


@pytest.mark.parametrize("generation", ["", "<SCENE>Modern residential building with white walls; outdoor seating"])
def test_verbose_prompt_echo_cannot_authorize_malformed_generation(generation: str) -> None:
    # Captured failure shape from the governed 1.5B runtime: the old parser
    # accepted all seven keys from the echoed prompt, not the assistant output.
    raw = (
        "Loading vision tower\n==========\n"
        "Prompt: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\n<image>\n"
        "Return exactly one line: SCENE=<short>; MATERIALS=<materials>; FEATURES=<features>; "
        "NATURAL=<nature>; LIGHTING=<light>; ISSUES=<issues>; UNCERTAIN=<uncertainty>. "
        "Use the uppercase keys exactly as shown. Do not add commentary.<|im_end|>\n"
        f"<|im_start|>assistant\n\n{generation}\n==========\n"
        "Prompt: 441 tokens\nGeneration: 120 tokens\n"
    )

    parsed = parse_fastvlm_caption(raw)

    assert parsed.validated is False
    assert parsed.caption == {}
    assert parsed.raw_text == raw


def test_verbose_output_parses_only_generated_caption() -> None:
    generated = (
        "SCENE=Patio; MATERIALS=stone; FEATURES=steps; NATURAL=trees; " "LIGHTING=daylight; ISSUES=none; UNCERTAIN=none"
    )
    raw = (
        "Prompt: <|im_start|>user\nSCENE=Wrong; MATERIALS=wrong; FEATURES=wrong; "
        "NATURAL=wrong; LIGHTING=wrong; ISSUES=wrong; UNCERTAIN=wrong<|im_end|>\n"
        f"<|im_start|>assistant\n{generated}\n==========\nPrompt: 10 tokens"
    )

    parsed = parse_fastvlm_caption(raw)

    assert parsed.validated is True
    assert parsed.caption["scene"] == "Patio"
    assert parsed.caption["uncertain"] == ["none"]
    assert "wrong" not in str(parsed.caption).lower()


def test_prompt_only_or_instructions_cannot_supply_caption_fields() -> None:
    template = (
        "Return these fields: SCENE=short; MATERIALS=materials; FEATURES=features; "
        "NATURAL=nature; LIGHTING=light; ISSUES=issues; UNCERTAIN=uncertainty"
    )
    for raw in (template, "Prompt: " + template, "<|im_start|>user\n" + template):
        parsed = parse_fastvlm_caption(raw)
        assert parsed.validated is False
        assert parsed.caption == {}


def test_empty_required_values_do_not_validate() -> None:
    parsed = parse_fastvlm_caption(
        "SCENE=<short>; MATERIALS=<materials>; FEATURES=<features>; NATURAL=<nature>; "
        "LIGHTING=<light>; ISSUES=<issues>; UNCERTAIN=<uncertainty>"
    )

    assert parsed.validated is False
    assert parsed.missing_keys == []
    assert any("empty required fields" in warning for warning in parsed.warnings)
