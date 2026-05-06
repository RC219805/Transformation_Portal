"""Unit tests for extracted rendering_4k quality and memory helpers."""

from __future__ import annotations

import numpy as np
import pytest

from transformation_portal.pipelines.rendering_4k.types import DeviceType, QualityFeedbackConfig, QualityMetrics

pytestmark = [pytest.mark.unit]


def _float32_image(height: int = 16, width: int = 16) -> np.ndarray:
    rng = np.random.default_rng(123)
    return rng.random((height, width, 3), dtype=np.float32)


def test_gpu_memory_manager_cpu_is_noop() -> None:
    from transformation_portal.pipelines.rendering_4k.quality import GPUMemoryManager

    manager = GPUMemoryManager(DeviceType.CPU)

    assert manager.get_memory_stats() == {}
    assert manager.check_memory_threshold(0.01) is True
    assert manager.clear_cache() is None
    assert manager.log_memory_status() is None


def test_quality_assessor_heuristic_metrics_stay_bounded() -> None:
    from transformation_portal.pipelines.rendering_4k.quality import QualityAssessor

    metrics = QualityAssessor(QualityFeedbackConfig(use_lpips=False)).assess(_float32_image())

    assert 0.0 <= metrics.sharpness <= 1.0
    assert 0.0 <= metrics.contrast <= 1.0
    assert 0.0 <= metrics.colorfulness <= 1.0
    assert 0.0 <= metrics.exposure_balance <= 1.0
    assert 0.0 <= metrics.noise_level <= 1.0
    assert 0.0 <= metrics.overall_score <= 1.0


def test_quality_assessor_uses_numpy_fallbacks_without_scipy(monkeypatch: pytest.MonkeyPatch) -> None:
    from transformation_portal.pipelines.rendering_4k import quality

    monkeypatch.setattr(quality, "HAS_SCIPY", False)
    monkeypatch.setattr(quality, "convolve", None)
    monkeypatch.setattr(quality, "median_filter", None)

    metrics = quality.QualityAssessor(QualityFeedbackConfig(use_lpips=False)).assess(_float32_image())

    assert isinstance(metrics, QualityMetrics)
    assert 0.0 <= metrics.sharpness <= 1.0
    assert 0.0 <= metrics.noise_level <= 1.0
    assert 0.0 <= metrics.overall_score <= 1.0


def test_quality_assessor_lpips_disabled_falls_back_to_heuristics() -> None:
    from transformation_portal.pipelines.rendering_4k.quality import QualityAssessor

    image = _float32_image()
    assessor = QualityAssessor(QualityFeedbackConfig(use_lpips=False))

    assert assessor._get_perceptual_assessor() is None

    metrics = assessor.assess(image, reference=image.copy())

    assert metrics.lpips_score == 0.0
    assert metrics.lpips_percentile == 0.0
    assert metrics.material_fidelity == 0.0
    assert metrics.perceptual_quality == 0.0
    assert 0.0 <= metrics.overall_score <= 1.0


def test_quality_assessor_suggests_adjustments_at_thresholds() -> None:
    from transformation_portal.pipelines.rendering_4k.quality import QualityAssessor

    assessor = QualityAssessor(QualityFeedbackConfig(use_lpips=False))

    adjustments = assessor.suggest_adjustments(
        QualityMetrics(
            sharpness=0.49,
            contrast=0.39,
            colorfulness=0.39,
            exposure_balance=0.30,
            noise_level=0.31,
        )
    )

    assert adjustments == {
        "clarity_boost": 0.2,
        "contrast_increase": 0.1,
        "saturation_boost": 0.05,
        "exposure_adjust": 0.1,
        "denoise_strength": 0.2,
    }


def test_quality_assessor_leaves_good_metrics_unadjusted() -> None:
    from transformation_portal.pipelines.rendering_4k.quality import QualityAssessor

    assessor = QualityAssessor(QualityFeedbackConfig(use_lpips=False))

    assert (
        assessor.suggest_adjustments(
            QualityMetrics(
                sharpness=0.50,
                contrast=0.40,
                colorfulness=0.40,
                exposure_balance=0.50,
                noise_level=0.30,
            )
        )
        == {}
    )
