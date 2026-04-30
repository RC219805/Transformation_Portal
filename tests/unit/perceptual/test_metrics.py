"""Unit tests for perceptual.metrics.

Covers MetricType enum, PerceptualScore dataclass (including is_better_than),
module-level compute_psnr/compute_ssim/compute_mse functions, and the
QualityMetrics class methods — using CPU torch tensors with an inline
lightweight substrate mock, no GPU or pretrained model required.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_substrate(device=None):
    """Create a minimal substrate duck-type accepted by QualityMetrics."""
    if device is None:
        device = torch.device("cpu")
    return type(
        "_Substrate",
        (),
        {
            "get_device": lambda self=None: device,
            "to_device": lambda self, x: x.to(device),
        },
    )()


def _rand_tensor(shape=(1, 3, 32, 32)):
    return torch.rand(*shape, dtype=torch.float32)


# ---------------------------------------------------------------------------
# MetricType enum
# ---------------------------------------------------------------------------


class TestMetricTypeEnum:
    def test_all_metric_types_exist(self):
        from transformation_portal.perceptual.metrics import MetricType

        values = {m.value for m in MetricType}
        for expected in ("LPIPS", "FID", "BRISQUE", "NIQE", "PSNR", "SSIM", "MSE"):
            assert expected in values

    def test_psnr_value(self):
        from transformation_portal.perceptual.metrics import MetricType

        assert MetricType.PSNR.value == "PSNR"

    def test_ssim_value(self):
        from transformation_portal.perceptual.metrics import MetricType

        assert MetricType.SSIM.value == "SSIM"


# ---------------------------------------------------------------------------
# PerceptualScore dataclass
# ---------------------------------------------------------------------------


class TestPerceptualScore:
    def test_fields_stored(self):
        from transformation_portal.perceptual.metrics import MetricType, PerceptualScore

        score = PerceptualScore(
            metric_type=MetricType.PSNR,
            score=35.0,
            higher_is_better=True,
            normalized_score=0.8,
            metadata={},
        )
        assert score.score == pytest.approx(35.0)
        assert score.higher_is_better is True
        assert score.normalized_score == pytest.approx(0.8)

    def test_is_better_than_higher_is_better(self):
        from transformation_portal.perceptual.metrics import MetricType, PerceptualScore

        better = PerceptualScore(MetricType.PSNR, 40.0, True, 0.9, {})
        worse = PerceptualScore(MetricType.PSNR, 30.0, True, 0.7, {})
        assert better.is_better_than(worse) is True
        assert worse.is_better_than(better) is False

    def test_is_better_than_lower_is_better(self):
        from transformation_portal.perceptual.metrics import MetricType, PerceptualScore

        better = PerceptualScore(MetricType.LPIPS, 0.1, False, 0.9, {})
        worse = PerceptualScore(MetricType.LPIPS, 0.5, False, 0.5, {})
        assert better.is_better_than(worse) is True
        assert worse.is_better_than(better) is False

    def test_equal_scores_not_better(self):
        from transformation_portal.perceptual.metrics import MetricType, PerceptualScore

        s1 = PerceptualScore(MetricType.MSE, 0.01, False, 0.9, {})
        s2 = PerceptualScore(MetricType.MSE, 0.01, False, 0.9, {})
        assert s1.is_better_than(s2) is False


# ---------------------------------------------------------------------------
# Module-level compute_psnr
# ---------------------------------------------------------------------------


class TestModuleComputePsnr:
    def test_identical_images_return_inf(self):
        from transformation_portal.perceptual.metrics import compute_psnr

        t = _rand_tensor()
        result = compute_psnr(t, t)
        assert math.isinf(result)

    def test_noisy_image_returns_positive_finite(self):
        from transformation_portal.perceptual.metrics import compute_psnr

        ref = _rand_tensor()
        noisy = ref + 0.1 * torch.randn_like(ref)
        noisy = noisy.clamp(0, 1)
        result = compute_psnr(noisy, ref)
        assert math.isfinite(result)
        assert result > 0

    def test_higher_noise_gives_lower_psnr(self):
        from transformation_portal.perceptual.metrics import compute_psnr

        ref = torch.ones(1, 3, 32, 32) * 0.5
        low_noise = ref + 0.01 * torch.randn_like(ref)
        high_noise = ref + 0.2 * torch.randn_like(ref)
        assert compute_psnr(low_noise, ref) > compute_psnr(high_noise, ref)

    def test_symmetry(self):
        from transformation_portal.perceptual.metrics import compute_psnr

        a = _rand_tensor()
        b = _rand_tensor()
        assert compute_psnr(a, b) == pytest.approx(compute_psnr(b, a), rel=1e-4)


# ---------------------------------------------------------------------------
# QualityMetrics.compute_psnr / compute_ssim / compute_mse
# ---------------------------------------------------------------------------


class TestQualityMetricsComputePsnr:
    def test_identical_images_psnr_infinite(self):
        from transformation_portal.perceptual.metrics import QualityMetrics

        metrics = QualityMetrics(_make_substrate())
        t = _rand_tensor()
        result = metrics.compute_psnr(t, t)
        assert math.isinf(result.score)

    def test_returns_perceptual_score(self):
        from transformation_portal.perceptual.metrics import MetricType, PerceptualScore, QualityMetrics

        metrics = QualityMetrics(_make_substrate())
        t = _rand_tensor()
        result = metrics.compute_psnr(t, t)
        assert isinstance(result, PerceptualScore)
        assert result.metric_type == MetricType.PSNR

    def test_psnr_higher_is_better(self):
        from transformation_portal.perceptual.metrics import QualityMetrics

        metrics = QualityMetrics(_make_substrate())
        t = _rand_tensor()
        result = metrics.compute_psnr(t, t)
        assert result.higher_is_better is True


class TestQualityMetricsComputeSsim:
    def test_identical_images_ssim_is_one(self):
        from transformation_portal.perceptual.metrics import QualityMetrics

        metrics = QualityMetrics(_make_substrate())
        t = _rand_tensor()
        result = metrics.compute_ssim(t, t)
        assert result.score == pytest.approx(1.0, abs=1e-3)

    def test_different_images_ssim_less_than_one(self):
        from transformation_portal.perceptual.metrics import QualityMetrics

        metrics = QualityMetrics(_make_substrate())
        a = torch.zeros(1, 3, 32, 32)
        b = torch.ones(1, 3, 32, 32)
        result = metrics.compute_ssim(a, b)
        assert result.score < 1.0

    def test_ssim_higher_is_better(self):
        from transformation_portal.perceptual.metrics import QualityMetrics

        metrics = QualityMetrics(_make_substrate())
        t = _rand_tensor()
        result = metrics.compute_ssim(t, t)
        assert result.higher_is_better is True


class TestQualityMetricsComputeMse:
    def test_identical_images_mse_is_zero(self):
        from transformation_portal.perceptual.metrics import QualityMetrics

        metrics = QualityMetrics(_make_substrate())
        t = _rand_tensor()
        result = metrics.compute_mse(t, t)
        assert result.score == pytest.approx(0.0, abs=1e-7)

    def test_mse_lower_is_better(self):
        from transformation_portal.perceptual.metrics import QualityMetrics

        metrics = QualityMetrics(_make_substrate())
        t = _rand_tensor()
        result = metrics.compute_mse(t, t)
        assert result.higher_is_better is False

    def test_different_images_mse_positive(self):
        from transformation_portal.perceptual.metrics import QualityMetrics

        metrics = QualityMetrics(_make_substrate())
        a = torch.zeros(1, 3, 16, 16)
        b = torch.ones(1, 3, 16, 16)
        result = metrics.compute_mse(a, b)
        assert result.score > 0
