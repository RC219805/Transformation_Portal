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
        for expected in ("lpips", "fid", "brisque", "niqe", "psnr", "ssim", "mse"):
            assert expected in values

    def test_psnr_value(self):
        from transformation_portal.perceptual.metrics import MetricType

        assert MetricType.PSNR.value == "psnr"

    def test_ssim_value(self):
        from transformation_portal.perceptual.metrics import MetricType

        assert MetricType.SSIM.value == "ssim"


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


# ---------------------------------------------------------------------------
# Module-level convenience wrappers (regression: M-04)
# ---------------------------------------------------------------------------


@pytest.mark.ml
@pytest.mark.regression
class TestModuleConvenienceWrappers:
    """The wrappers build their own substrate; it must accept a bound call."""

    def test_compute_ssim_identical_is_one(self):
        from transformation_portal.perceptual.metrics import compute_ssim

        img = _rand_tensor((1, 3, 32, 32))
        assert compute_ssim(img, img.clone()) == pytest.approx(1.0, abs=1e-4)

    def test_compute_ssim_different_below_one(self):
        from transformation_portal.perceptual.metrics import compute_ssim

        torch.manual_seed(0)
        assert compute_ssim(_rand_tensor(), _rand_tensor()) < 1.0

    def test_compute_brisque_returns_finite_float(self):
        from transformation_portal.perceptual.metrics import compute_brisque

        score = compute_brisque(_rand_tensor((3, 32, 32)))
        assert isinstance(score, float)
        assert math.isfinite(score)

    def test_compute_niqe_returns_finite_float(self):
        from transformation_portal.perceptual.metrics import compute_niqe

        score = compute_niqe(_rand_tensor((3, 32, 32)))
        assert isinstance(score, float)
        assert math.isfinite(score)

    def test_compute_lpips_identical_is_zero(self):
        # Needs the real LPIPS model: without the lpips package the metric is
        # unavailable (M-14) instead of silently falling back to MSE.
        pytest.importorskip("lpips")
        from transformation_portal.perceptual.metrics import compute_lpips

        img = _rand_tensor((1, 3, 32, 32))
        score = compute_lpips(img, img.clone(), torch.device("cpu"))
        assert score == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------------
# Fallback / unavailable metrics (regression: M-13, M-14)
# ---------------------------------------------------------------------------


def _sharp_and_blurry():
    import torch.nn.functional as F

    torch.manual_seed(0)
    sharp = torch.rand(3, 64, 64)
    blurry = F.avg_pool2d(sharp.unsqueeze(0), 9, stride=1, padding=4)[0]
    return sharp, blurry


@pytest.fixture
def _no_brisque_models(monkeypatch):
    for name in ("TP_BRISQUE_MODEL_PATH", "TP_BRISQUE_RANGE_PATH", "BRISQUE_MODEL_PATH", "BRISQUE_RANGE_PATH"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def _no_lpips(monkeypatch):
    import sys

    # A None entry makes ``import lpips`` raise ImportError whether or not it is installed.
    monkeypatch.setitem(sys.modules, "lpips", None)


@pytest.mark.ml
@pytest.mark.regression
class TestBrisqueFallbackDirection:
    """The sharpness proxy is higher-is-better and must be labelled so (M-13)."""

    def test_fallback_prefers_sharp_image(self, _no_brisque_models):
        from transformation_portal.perceptual.metrics import QualityMetrics

        qm = QualityMetrics(_make_substrate())
        sharp, blurry = _sharp_and_blurry()
        s_sharp, s_blurry = qm.compute_brisque(sharp), qm.compute_brisque(blurry)

        assert s_sharp.higher_is_better is True
        assert s_sharp.is_better_than(s_blurry) is True
        assert s_blurry.is_better_than(s_sharp) is False
        # normalized_score agrees with the comparison direction.
        assert s_sharp.normalized_score > s_blurry.normalized_score

    def test_fallback_normalized_score_is_bounded_and_tagged(self, _no_brisque_models):
        from transformation_portal.perceptual.metrics import QualityMetrics

        qm = QualityMetrics(_make_substrate())
        sharp, _ = _sharp_and_blurry()
        result = qm.compute_brisque(sharp)

        assert result.score > 1.0  # raw gradient magnitude exceeds 1 for this image
        assert 0.0 <= result.normalized_score <= 1.0
        assert result.metadata["fallback"] == "gradient_sharpness"

    def test_scores_with_different_directions_are_not_comparable(self):
        from transformation_portal.perceptual.metrics import MetricType, PerceptualScore

        real = PerceptualScore(MetricType.BRISQUE, 20.0, False, 0.8, {})
        proxy = PerceptualScore(MetricType.BRISQUE, 0.5, True, 0.5, {"fallback": "gradient_sharpness"})
        with pytest.raises(ValueError, match="direction"):
            real.is_better_than(proxy)

    def test_analyzer_compare_picks_sharp_image_on_brisque(self, _no_brisque_models):
        from pathlib import Path

        from transformation_portal.perceptual.analyzer import PerceptualAnalyzer
        from transformation_portal.perceptual.image_loader import ImageMetadata

        def _meta(name):
            return ImageMetadata(
                path=Path(name),
                image_type=None,
                width=64,
                height=64,
                channels=3,
                format="PNG",
                size_bytes=0,
                bit_depth=8,
                color_space="sRGB",
                mean_intensity=0.5,
                std_intensity=0.2,
                dynamic_range=1.0,
                tags={},
            )

        sharp, blurry = _sharp_and_blurry()
        result = PerceptualAnalyzer(_make_substrate()).compare(sharp, blurry, _meta("sharp.png"), _meta("blurry.png"))
        assert result["metric_comparisons"]["brisque"]["winner"] == 1


@pytest.mark.ml
@pytest.mark.regression
class TestLpipsUnavailable:
    """Without the lpips package LPIPS is reported unavailable, not as MSE (M-14)."""

    def test_compute_lpips_raises_unavailable(self, _no_lpips):
        from transformation_portal.perceptual.metrics import MetricUnavailableError, QualityMetrics

        qm = QualityMetrics(_make_substrate())
        sharp, blurry = _sharp_and_blurry()
        with pytest.raises(MetricUnavailableError, match="lpips"):
            qm.compute_lpips(blurry, sharp)

    def test_module_wrapper_raises_unavailable(self, _no_lpips):
        from transformation_portal.perceptual.metrics import MetricUnavailableError, compute_lpips

        img = _rand_tensor((1, 3, 32, 32))
        with pytest.raises(MetricUnavailableError):
            compute_lpips(img, img.clone(), torch.device("cpu"))

    def test_compute_all_omits_lpips_key(self, _no_lpips):
        from transformation_portal.perceptual.metrics import MetricType, QualityMetrics

        qm = QualityMetrics(_make_substrate())
        sharp, blurry = _sharp_and_blurry()
        scores = qm.compute_all(blurry, sharp)

        assert MetricType.LPIPS not in scores
        assert scores[MetricType.MSE].metric_type is MetricType.MSE
        assert all(score.metric_type is key for key, score in scores.items())
