"""Tests for the CognitiveMaterialResponse orchestration layer."""

from material_response import (
    CognitiveMaterialResponse,
    EmotionalResonance,
    GlobalLuxurySemantics,
    LightingProfile,
    MaterialAestheticProfile,
    NeuroAestheticEngine,
    ViewerProfile,
)


def test_perception_model_generates_clamped_scores() -> None:
    engine = NeuroAestheticEngine()

    resonance = engine.predict_limbic_response(
        texture="Velvet chaise with polished brass inlay",
        warmth=0.72,
        cultural_background="Mediterranean",
    )

    assert isinstance(resonance, EmotionalResonance)
    assert 0.0 <= resonance.awe <= 1.0
    assert 0.0 <= resonance.comfort <= 1.0
    assert 0.0 <= resonance.focus <= 1.0
    assert resonance.cultural_background == "mediterranean"


def test_global_luxury_semantics_reacts_to_culture() -> None:
    material = MaterialAestheticProfile(
        name="Polished Travertine",
        texture="polished stone",
        rarity=0.8,
        craftsmanship=0.7,
        innovation=0.35,
    )
    resonance = EmotionalResonance(
        awe=0.55,
        comfort=0.48,
        focus=0.52,
        cultural_background="Scandinavian",
    )

    semantics = GlobalLuxurySemantics()
    contextualized = semantics.recontextualize(material, resonance)

    assert contextualized.scores["focus"] >= contextualized.scores["comfort"]
    assert "scandinavian" in contextualized.narrative


def test_global_luxury_semantics_alias_normalization() -> None:
    material = MaterialAestheticProfile(
        name="Lacquered Shoji",
        texture="lacquer and rice paper",
        rarity=0.65,
        craftsmanship=0.85,
        innovation=0.42,
    )
    resonance = EmotionalResonance(
        awe=0.62,
        comfort=0.51,
        focus=0.58,
        cultural_background="Nordic",
    )

    semantics = GlobalLuxurySemantics()
    contextualized = semantics.recontextualize(material, resonance)

    assert "scandinavian" in contextualized.narrative
    assert contextualized.scores["focus"] >= contextualized.scores["comfort"]


def test_cognitive_material_response_pipeline() -> None:
    material = MaterialAestheticProfile(
        name="Midnight Velvet Chaise",
        texture="handcrafted velvet",
        rarity=0.72,
        craftsmanship=0.9,
        innovation=0.58,
    )
    lighting = LightingProfile(warmth=0.63, intensity=0.55, diffusion=0.68)
    viewer = ViewerProfile(
        cultural_background="Mediterranean",
        novelty_preference=0.6,
        heritage_affinity=0.55,
    )

    cognitive_response = CognitiveMaterialResponse()
    result = cognitive_response.process(material, lighting, viewer)

    assert 0.0 <= result["luxury_index"] <= 1.0
    assert result["future_alignment"] >= 0.5
    assert (result["emotional_resonance"]["comfort"] >=
            result["emotional_resonance"]["focus"])
    assert result["recommendations"]
