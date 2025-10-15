from __future__ import annotations

import pytest

from temporal_evolution import TemporalEvolutionRoadmap


def _sample_payload() -> dict[str, object]:
    return {
        "temporal_evolution": {
            "predictive_refactoring": [
                {"Monitor": "Industry trend APIs, competitor repositories, academic papers"},
                {"Predict": "Future architectural needs using transformer models"},
                {"Pre-adapt": "Generate compatibility layers before standards emerge"},
            ],
            "quantum_branching": [
                "Maintain parallel reality branches of the codebase",
                "Each branch optimized for different future scenarios",
                "Collapse to optimal branch when future becomes present",
            ],
        }
    }


def test_from_mapping_parses_nested_temporal_evolution_block() -> None:
    roadmap = TemporalEvolutionRoadmap.from_mapping(_sample_payload())

    assert [discipline.name for discipline in roadmap.disciplines] == [
        "predictive_refactoring",
        "quantum_branching",
    ]

    predictive = roadmap.disciplines[0]
    assert [directive.summary for directive in predictive.directives] == [
        "Monitor",
        "Predict",
        "Pre-adapt",
    ]
    assert predictive.directives[0].detail.startswith("Industry trend APIs")

    quantum = roadmap.disciplines[1]
    assert [directive.detail for directive in quantum.directives] == [None, None, None]
    assert "parallel reality" in quantum.directives[0].summary

    markdown = roadmap.to_markdown()
    assert "### Predictive Refactoring" in markdown
    assert "**Monitor**" in markdown
    assert "- Maintain parallel reality branches" in markdown


@pytest.mark.parametrize(
    "invalid_directives",
    [
        "plain string",
        {"unexpected": "mapping"},
        [123],
        [{"Monitor": 10}],
        [{"Monitor": "ok", "Extra": "nope"}],
        [{"   ": "detail"}],
    ],
)
def test_from_mapping_validates_directive_structure(invalid_directives) -> None:
    payload = {"temporal_evolution": {"predictive_refactoring": invalid_directives}}

    with pytest.raises(TypeError):
        TemporalEvolutionRoadmap.from_mapping(payload)


def test_serialise_round_trip() -> None:
    payload = _sample_payload()
    roadmap = TemporalEvolutionRoadmap.from_mapping(payload)

    assert roadmap.serialise() == payload["temporal_evolution"]


@pytest.mark.parametrize(
    "invalid_name",
    [42, "   "],
)
def test_from_mapping_validates_discipline_names(invalid_name) -> None:
    payload = {"temporal_evolution": {invalid_name: []}}

    with pytest.raises(TypeError):
        TemporalEvolutionRoadmap.from_mapping(payload)

