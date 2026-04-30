"""Unit tests for neuroaesthetics.color_harmony.ColorHarmonyAnalyzer.

Covers HarmonyType enum, ColorHarmonyAnalyzer construction, analyze() output
contract, score bounds, palette structure, and temperature/emotional profile
fields — using in-process numpy/PIL images.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("sklearn")

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# HarmonyType enum
# ---------------------------------------------------------------------------


class TestHarmonyTypeEnum:
    def test_all_harmony_types_exist(self):
        from transformation_portal.neuroaesthetics.color_harmony import HarmonyType

        values = {t.value for t in HarmonyType}
        for expected in (
            "analogous",
            "complementary",
            "triadic",
            "tetradic",
            "monochromatic",
            "warm",
            "cool",
            "neutral",
        ):
            assert expected in values

    def test_complementary_value(self):
        from transformation_portal.neuroaesthetics.color_harmony import HarmonyType

        assert HarmonyType.COMPLEMENTARY.value == "complementary"

    def test_monochromatic_value(self):
        from transformation_portal.neuroaesthetics.color_harmony import HarmonyType

        assert HarmonyType.MONOCHROMATIC.value == "monochromatic"


# ---------------------------------------------------------------------------
# ColorHarmonyAnalyzer construction
# ---------------------------------------------------------------------------


class TestColorHarmonyAnalyzerInit:
    def test_default_num_colors(self):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        assert ColorHarmonyAnalyzer().num_colors == 5

    def test_custom_num_colors(self):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        assert ColorHarmonyAnalyzer(num_colors=3).num_colors == 3

    def test_default_min_proportion(self):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        assert ColorHarmonyAnalyzer().min_proportion == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# analyze() — return type and contract
# ---------------------------------------------------------------------------


class TestColorHarmonyAnalyze:
    def test_returns_harmony_analysis(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer, HarmonyAnalysis

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result, HarmonyAnalysis)

    def test_harmony_score_bounded(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_image)
        assert 0.0 <= result.harmony_score <= 1.0

    def test_harmony_type_is_enum(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer, HarmonyType

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result.harmony_type, HarmonyType)

    def test_palette_has_correct_structure(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer, ColorPalette

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result.palette, ColorPalette)
        assert isinstance(result.palette.colors_rgb, np.ndarray)
        assert isinstance(result.palette.proportions, np.ndarray)

    def test_palette_proportions_sum_to_one(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_image)
        assert result.palette.proportions.sum() == pytest.approx(1.0, abs=0.01)

    def test_recommendations_are_strings(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_image)
        assert all(isinstance(r, str) for r in result.recommendations)

    def test_emotional_profile_is_dict(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result.emotional_profile, dict)

    def test_disharmony_factors_is_list(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result.disharmony_factors, list)

    def test_temperature_is_float(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result.temperature, float)

    def test_accepts_pil_image(self, sample_rgb_pil):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer, HarmonyAnalysis

        result = ColorHarmonyAnalyzer().analyze(sample_rgb_pil)
        assert isinstance(result, HarmonyAnalysis)

    def test_warm_dominant_image_does_not_raise(self):
        """Image with dominant warm tones but enough variance for KMeans to converge."""
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        rng = np.random.default_rng(42)
        warm = rng.integers(150, 220, (60, 60, 3), dtype=np.uint8)
        warm[:, :, 1] = rng.integers(20, 70, (60, 60), dtype=np.uint8)
        warm[:, :, 2] = rng.integers(20, 70, (60, 60), dtype=np.uint8)
        result = ColorHarmonyAnalyzer().analyze(warm)
        assert 0.0 <= result.harmony_score <= 1.0
