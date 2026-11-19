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


# ---------------------------------------------------------------------------
# Regression: warm/cool classification, degenerate images, determinism
# ---------------------------------------------------------------------------


def _jittered_solid(rgb: tuple[int, int, int], size: int = 48, jitter: int = 6) -> np.ndarray:
    """Near-solid colour patch with enough variance for two K-means clusters."""
    rng = np.random.default_rng(0)
    noise = rng.integers(-jitter, jitter + 1, (size, size, 3))
    return np.clip(np.asarray(rgb)[None, None, :] + noise, 0, 255).astype(np.uint8)


class TestColorTemperatureClassification:
    """Warm/cool ranges are colour-wheel hues; they must not be read off the CIELAB hue angle."""

    @pytest.mark.parametrize(
        ("name", "rgb"),
        [("blue", (20, 40, 220)), ("navy", (10, 20, 120)), ("sky", (90, 160, 235)), ("cyan", (0, 200, 220))],
    )
    def test_blues_and_cyans_are_cool(self, name, rgb):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        result = ColorHarmonyAnalyzer(num_colors=2).analyze(_jittered_solid(rgb), sample_fraction=1.0)

        assert result.temperature == pytest.approx(-1.0), name
        assert result.emotional_profile["nostalgia"] == 0.0
        assert result.emotional_profile["serenity"] > 0.0

    @pytest.mark.parametrize(
        ("name", "rgb"),
        [("red", (220, 30, 30)), ("orange", (240, 140, 20)), ("yellow", (240, 220, 30))],
    )
    def test_reds_oranges_and_yellows_are_warm(self, name, rgb):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        result = ColorHarmonyAnalyzer(num_colors=2).analyze(_jittered_solid(rgb), sample_fraction=1.0)

        assert result.temperature == pytest.approx(1.0), name
        assert result.emotional_profile["nostalgia"] > 0.0

    @pytest.mark.parametrize(("name", "rgb"), [("green", (30, 180, 60)), ("grey", (128, 128, 128))])
    def test_greens_and_greys_are_neutral(self, name, rgb):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        result = ColorHarmonyAnalyzer(num_colors=2).analyze(_jittered_solid(rgb), sample_fraction=1.0)

        assert result.temperature == pytest.approx(0.0), name

    def test_achromatic_palette_has_no_hue_relationship(self):
        """Hue of a grey is numerical noise; it must not be read as complementary/triadic."""
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer, HarmonyType

        result = ColorHarmonyAnalyzer(num_colors=2).analyze(_jittered_solid((128, 128, 128)), sample_fraction=1.0)

        assert result.harmony_type is HarmonyType.MONOCHROMATIC


class TestDegenerateImages:
    """Images with fewer distinct colours or pixels than requested clusters must still analyse."""

    @pytest.mark.parametrize(
        ("columns", "expected_colors"),
        [
            ([(255, 255, 255)], 1),
            ([(30, 30, 30), (220, 220, 220)], 2),
            ([(200, 30, 30), (30, 200, 30), (30, 30, 200)], 3),
        ],
        ids=["flat", "two-tone", "three-colour"],
    )
    def test_fewer_distinct_colours_than_clusters(self, columns, expected_colors):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        image = np.concatenate([np.full((48, 16, 3), color, dtype=np.uint8) for color in columns], axis=1)

        result = ColorHarmonyAnalyzer(num_colors=5).analyze(image, sample_fraction=1.0)

        assert len(result.palette.colors_rgb) == expected_colors
        assert len(result.palette.proportions) == expected_colors
        assert float(result.palette.proportions.sum()) == pytest.approx(1.0)
        assert {tuple(int(c) for c in color) for color in result.palette.colors_rgb} == set(columns)

    @pytest.mark.parametrize("shape", [(6, 6, 3), (2, 2, 3), (1, 1, 3)], ids=["6x6", "2x2", "1x1"])
    def test_tiny_images_keep_enough_samples_for_kmeans(self, shape):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        image = np.random.default_rng(1).integers(0, 255, shape, dtype=np.uint8)

        result = ColorHarmonyAnalyzer(num_colors=5).analyze(image)  # default 10% sampling

        assert 1 <= len(result.palette.colors_rgb) <= min(5, shape[0] * shape[1])
        assert float(result.palette.proportions.sum()) == pytest.approx(1.0)

    def test_empty_image_is_rejected_clearly(self):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        with pytest.raises(ValueError, match="empty image"):
            ColorHarmonyAnalyzer()._extract_palette(np.zeros((0, 0, 3), dtype=np.uint8), 0.1)


class TestSamplingBehaviour:
    def test_analysis_is_deterministic_and_leaves_global_rng_untouched(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer

        analyzer = ColorHarmonyAnalyzer()
        np.random.seed(123)
        state_before = np.random.get_state()[1].copy()

        first = analyzer.analyze(sample_rgb_image)
        second = analyzer.analyze(sample_rgb_image)

        assert np.array_equal(first.palette.colors_rgb, second.palette.colors_rgb)
        assert np.array_equal(first.palette.proportions, second.palette.proportions)
        assert first.harmony_score == second.harmony_score
        assert np.array_equal(state_before, np.random.get_state()[1])

    def test_subsample_is_bounded_for_large_images(self, monkeypatch):
        from transformation_portal.neuroaesthetics import color_harmony

        fitted_sizes: list[int] = []
        real_kmeans = color_harmony.KMeans

        class RecordingKMeans(real_kmeans):
            def fit(self, X, *args, **kwargs):
                fitted_sizes.append(len(X))
                return super().fit(X, *args, **kwargs)

        monkeypatch.setattr(color_harmony, "KMeans", RecordingKMeans)
        monkeypatch.setattr(color_harmony.ColorHarmonyAnalyzer, "MAX_PALETTE_SAMPLES", 500)
        image = np.random.default_rng(2).integers(0, 255, (200, 200, 3), dtype=np.uint8)

        color_harmony.ColorHarmonyAnalyzer().analyze(image)  # 10% of 40,000 = 4,000 > cap

        assert fitted_sizes == [500]

    def test_clashing_saturated_colours_are_reported_once(self):
        from transformation_portal.neuroaesthetics.color_harmony import ColorHarmonyAnalyzer, ColorPalette

        # Four saturated colours whose pairwise hue distances mostly fall in the clash band.
        hues = np.array([0.0, 100.0, 200.0, 300.0])
        palette = ColorPalette(
            colors_rgb=np.zeros((4, 3), dtype=np.uint8),
            colors_lab=np.zeros((4, 3), dtype=np.float32),
            proportions=np.full(4, 0.25),
            hues=hues,
            saturations=np.full(4, 80.0),
            lightnesses=np.full(4, 50.0),
        )

        issues = ColorHarmonyAnalyzer()._identify_disharmony(palette)

        assert issues.count("Clashing saturated colors detected") == 1
