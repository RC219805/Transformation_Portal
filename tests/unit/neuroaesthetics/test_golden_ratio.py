"""Unit tests for neuroaesthetics.golden_ratio.GoldenRatioAnalyzer.

Covers PHI constant, analyzer construction, analyze() output contract,
score bounds, grid/alignment geometry, and the optimal-crop helper —
using in-process numpy/PIL images with no filesystem access.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# PHI constant
# ---------------------------------------------------------------------------


class TestPhiConstant:
    def test_phi_approximately_1618(self):
        from transformation_portal.neuroaesthetics.golden_ratio import PHI

        assert PHI == pytest.approx(1.6180339887, rel=1e-6)

    def test_phi_satisfies_golden_equation(self):
        from transformation_portal.neuroaesthetics.golden_ratio import PHI

        assert PHI**2 == pytest.approx(PHI + 1, rel=1e-6)

    def test_phi_greater_than_one(self):
        from transformation_portal.neuroaesthetics.golden_ratio import PHI

        assert PHI > 1.0


# ---------------------------------------------------------------------------
# GoldenRatioAnalyzer construction
# ---------------------------------------------------------------------------


class TestGoldenRatioAnalyzerInit:
    def test_default_tolerance(self):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        assert GoldenRatioAnalyzer().tolerance == pytest.approx(0.05)

    def test_custom_tolerance(self):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        assert GoldenRatioAnalyzer(tolerance=0.1).tolerance == pytest.approx(0.1)

    def test_default_min_feature_strength(self):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        assert GoldenRatioAnalyzer().min_feature_strength == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# analyze() — return type and contract
# ---------------------------------------------------------------------------


class TestGoldenRatioAnalyze:
    def test_returns_dataclass(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalysis, GoldenRatioAnalyzer

        result = GoldenRatioAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result, GoldenRatioAnalysis)

    def test_score_bounded_zero_to_one(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        result = GoldenRatioAnalyzer().analyze(sample_rgb_image)
        assert 0.0 <= result.score <= 1.0

    def test_grid_points_is_ndarray(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        result = GoldenRatioAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result.grid_points, np.ndarray)

    def test_feature_positions_is_list(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        result = GoldenRatioAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result.feature_positions, list)

    def test_alignments_is_list(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        result = GoldenRatioAnalyzer().analyze(sample_rgb_image)
        assert isinstance(result.alignments, list)

    def test_recommendations_are_strings(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        result = GoldenRatioAnalyzer().analyze(sample_rgb_image)
        assert all(isinstance(r, str) for r in result.recommendations)

    def test_accepts_pil_image(self, sample_rgb_pil):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalysis, GoldenRatioAnalyzer

        result = GoldenRatioAnalyzer().analyze(sample_rgb_pil)
        assert isinstance(result, GoldenRatioAnalysis)

    def test_uniform_gray_image_does_not_raise(self):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        gray = np.full((100, 100, 3), 128, dtype=np.uint8)
        result = GoldenRatioAnalyzer().analyze(gray)
        assert 0.0 <= result.score <= 1.0

    def test_without_feature_detection(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalysis, GoldenRatioAnalyzer

        result = GoldenRatioAnalyzer().analyze(sample_rgb_image, detect_features=False)
        assert isinstance(result, GoldenRatioAnalysis)
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# get_optimal_crop()
# ---------------------------------------------------------------------------


class TestGetOptimalCrop:
    def test_returns_four_values(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        crop = GoldenRatioAnalyzer().get_optimal_crop(sample_rgb_image)
        assert len(crop) == 4

    def test_crop_within_image_bounds(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        x, y, w, h = GoldenRatioAnalyzer().get_optimal_crop(sample_rgb_image)
        img_h, img_w = sample_rgb_image.shape[:2]
        assert x >= 0 and y >= 0
        assert x + w <= img_w
        assert y + h <= img_h

    def test_crop_positive_dimensions(self, sample_rgb_image):
        from transformation_portal.neuroaesthetics.golden_ratio import GoldenRatioAnalyzer

        x, y, w, h = GoldenRatioAnalyzer().get_optimal_crop(sample_rgb_image)
        assert w > 0 and h > 0
