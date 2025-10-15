from __future__ import annotations

from synthetic_viewer import ACUScore, JourneyMoment, SyntheticViewer


def build_demo_stream() -> list[dict[str, float]]:
    return [
        {"technical": 0.92, "emotional": 0.88, "memorability": 0.82, "desire": 0.9},
        {"technical": 0.95, "emotional": 0.9, "memorability": 0.86, "desire": 0.94},
        {"technical": 0.9, "emotional": 0.87, "memorability": 0.84, "desire": 0.93},
    ]


def test_experience_content_returns_consensus_score() -> None:
    viewer = SyntheticViewer()
    score = viewer.experience_content(build_demo_stream())

    assert isinstance(score, ACUScore)
    assert 0.0 <= score.technical <= 1.0
    assert 0.0 <= score.emotional <= 1.0
    assert 0.0 <= score.memorability <= 1.0
    assert 0.0 <= score.desire_quotient <= 1.0
    assert score.emotional > 0.85  # strong emotional response from the sample
    assert score.overall == score.as_dict()["overall"]


def test_journey_moment_normalisation() -> None:
    moment = JourneyMoment(technical=1.5, emotional=-0.2, memorability=0.5, desire=0.3)
    assert moment.technical == 1.0
    assert moment.emotional == 0.0


def test_consensus_matches_manual_average() -> None:
    viewer = SyntheticViewer()
    journey = viewer.consciousness.traverse(build_demo_stream())

    base = viewer.aesthetic_cortex.score(journey, viewer.archetype)
    alternates = [
        viewer.aesthetic_cortex.score(journey, "minimalist_millennial"),
        viewer.aesthetic_cortex.score(journey, "traditional_luxury_connoisseur"),
        viewer.aesthetic_cortex.score(journey, "futurist_tech_executive"),
    ]
    manual = viewer.reach_aesthetic_consensus([base, *alternates])

    automated = viewer.experience_content(journey)
    assert automated == manual
