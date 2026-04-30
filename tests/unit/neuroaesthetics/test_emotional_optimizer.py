"""Unit tests for neuroaesthetics.emotional_optimizer.EmotionalOptimizer.

Covers EmotionalTarget enum, EmotionalProfile field contract, analyze()
return type, overall_quality bounds, and optimize_for_emotion() across all
seven emotional targets — using in-process numpy/PIL images.
"""

from __future__ import annotations

import types

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("sklearn")

pytestmark = [pytest.mark.unit]


@pytest.fixture(autouse=True)
def _stub_cv2_saliency(monkeypatch):
    """Stub cv2.saliency so tests run without opencv-contrib-python."""
    stub = types.SimpleNamespace(
        StaticSaliencyFineGrained_create=lambda: types.SimpleNamespace(computeSaliency=lambda img: (False, None))
    )
    monkeypatch.setattr(cv2, "saliency", stub, raising=False)


# ---------------------------------------------------------------------------
# EmotionalTarget enum
# ---------------------------------------------------------------------------


class TestEmotionalTargetEnum:
    def test_all_targets_exist(self):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalTarget

        values = {t.value for t in EmotionalTarget}
        for expected in ("nostalgia", "aspiration", "desire", "luxury", "comfort", "energy", "serenity"):
            assert expected in values

    def test_serenity_value(self):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalTarget

        assert EmotionalTarget.SERENITY.value == "serenity"

    def test_luxury_value(self):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalTarget

        assert EmotionalTarget.LUXURY.value == "luxury"

    def test_seven_targets_defined(self):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalTarget

        assert len(EmotionalTarget) == 7


# ---------------------------------------------------------------------------
# EmotionalOptimizer.analyze() — return type and profile contract
# ---------------------------------------------------------------------------


class TestEmotionalOptimizerAnalyze:
    def test_returns_emotional_profile(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer, EmotionalProfile

        result = EmotionalOptimizer().analyze(sample_rgb_image)
        assert isinstance(result, EmotionalProfile)

    def test_profile_has_golden_ratio_analysis(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer

        result = EmotionalOptimizer().analyze(sample_rgb_image)
        assert result.golden_ratio_analysis is not None

    def test_profile_has_color_harmony_analysis(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer

        result = EmotionalOptimizer().analyze(sample_rgb_image)
        assert result.color_harmony_analysis is not None

    def test_profile_has_spatial_frequency_analysis(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer

        result = EmotionalOptimizer().analyze(sample_rgb_image)
        assert result.spatial_frequency_analysis is not None

    def test_overall_quality_bounded(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer

        result = EmotionalOptimizer().analyze(sample_rgb_image)
        assert 0.0 <= result.overall_quality <= 1.0

    def test_emotional_scores_is_dict(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer

        result = EmotionalOptimizer().analyze(sample_rgb_image)
        assert isinstance(result.emotional_scores, dict)

    def test_emotional_scores_not_empty(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer

        result = EmotionalOptimizer().analyze(sample_rgb_image)
        assert len(result.emotional_scores) > 0

    def test_optimization_priority_is_list(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer

        result = EmotionalOptimizer().analyze(sample_rgb_image)
        assert isinstance(result.optimization_priority, list)

    def test_enhancement_strategy_is_dict(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer

        result = EmotionalOptimizer().analyze(sample_rgb_image)
        assert isinstance(result.enhancement_strategy, dict)

    def test_accepts_pil_image(self, sample_rgb_pil):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer, EmotionalProfile

        result = EmotionalOptimizer().analyze(sample_rgb_pil)
        assert isinstance(result, EmotionalProfile)


# ---------------------------------------------------------------------------
# EmotionalOptimizer.optimize_for_emotion()
# ---------------------------------------------------------------------------


class TestOptimizeForEmotion:
    def test_returns_dict(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer, EmotionalTarget

        result = EmotionalOptimizer().optimize_for_emotion(sample_rgb_image, EmotionalTarget.SERENITY)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "target",
        ["nostalgia", "aspiration", "desire", "luxury", "comfort", "energy", "serenity"],
    )
    def test_all_targets_return_dict(self, sample_rgb_image, target):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer, EmotionalTarget

        result = EmotionalOptimizer().optimize_for_emotion(sample_rgb_image, EmotionalTarget(target))
        assert isinstance(result, dict)

    def test_result_not_empty(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.emotional_optimizer import EmotionalOptimizer, EmotionalTarget

        result = EmotionalOptimizer().optimize_for_emotion(sample_rgb_image, EmotionalTarget.LUXURY)
        assert len(result) > 0
