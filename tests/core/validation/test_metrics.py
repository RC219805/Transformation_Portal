"""Tests for metrics computation."""

import pytest
import numpy as np

from src.transformation_portal.core.validation.metrics import (
    MetricsComputer,
    QualityMetrics
)


def test_quality_metrics_to_dict():
    """Test converting metrics to dictionary."""
    metrics = QualityMetrics(ssim=0.95, psnr=35.2, mae=0.05)
    
    data = metrics.to_dict()
    assert data["ssim"] == 0.95
    assert data["psnr"] == 35.2
    assert data["mae"] == 0.05
    assert "lpips" not in data  # None values excluded


def test_quality_metrics_str():
    """Test string representation."""
    metrics = QualityMetrics(ssim=0.95, psnr=35.2)
    
    string = str(metrics)
    assert "SSIM: 0.9500" in string
    assert "PSNR: 35.20" in string


def test_metrics_computer_init():
    """Test metrics computer initialization."""
    computer = MetricsComputer()
    
    assert computer.weights["ssim"] == 0.3
    assert computer.weights["psnr"] == 0.2


def test_compute_identical_images():
    """Test metrics for identical images."""
    computer = MetricsComputer()
    
    # Create identical images
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    proc = ref.copy()
    
    metrics = computer.compute(ref, proc, metrics=["ssim", "psnr", "mae", "mse"])
    
    assert metrics.ssim > 0.99  # Should be ~1.0
    assert metrics.psnr > 40.0  # Very high for identical
    assert metrics.mae < 0.01   # Very low
    assert metrics.mse < 0.0001


def test_compute_different_images():
    """Test metrics for different images."""
    computer = MetricsComputer()
    
    # Create different images
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    proc = np.random.rand(100, 100, 3).astype(np.float32)
    
    metrics = computer.compute(ref, proc, metrics=["ssim", "psnr", "mae"])

    # SSIM can be negative for very different images
    assert -1.0 <= metrics.ssim <= 1.0
    assert metrics.psnr > 0.0
    assert metrics.mae > 0.0


def test_compute_with_noise():
    """Test metrics with added noise."""
    computer = MetricsComputer()
    
    # Original image
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    
    # Add small noise
    noise = np.random.randn(100, 100, 3).astype(np.float32) * 0.01
    proc = np.clip(ref + noise, 0, 1)
    
    metrics = computer.compute(ref, proc, metrics=["ssim", "psnr", "mae"])
    
    # Should have high similarity
    assert metrics.ssim > 0.9
    assert metrics.psnr > 30.0
    assert metrics.mae < 0.02


def test_normalize_image_uint8():
    """Test image normalization from uint8."""
    computer = MetricsComputer()
    
    # uint8 image (0-255)
    img_uint8 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    normalized = computer._normalize_image(img_uint8)
    
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_image_float():
    """Test image normalization from float."""
    computer = MetricsComputer()
    
    # float32 image (0-1)
    img_float = np.random.rand(100, 100, 3).astype(np.float32)
    
    normalized = computer._normalize_image(img_float)
    
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_image_channel_first():
    """Test normalization of (C, H, W) format."""
    computer = MetricsComputer()
    
    # (C, H, W) format
    img_chw = np.random.rand(3, 100, 100).astype(np.float32)
    
    normalized = computer._normalize_image(img_chw)
    
    # Should be converted to (H, W, C)
    assert normalized.shape == (100, 100, 3)


def test_compute_ssim():
    """Test SSIM computation."""
    computer = MetricsComputer()
    
    # Create test images
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    
    # Identical
    ssim_identical = computer._compute_ssim(ref, ref)
    assert ssim_identical > 0.99
    
    # Different
    proc = np.random.rand(100, 100, 3).astype(np.float32)
    ssim_diff = computer._compute_ssim(ref, proc)
    # SSIM can be slightly negative with simple approximation
    assert -0.1 <= ssim_diff <= 1.0


def test_compute_psnr():
    """Test PSNR computation."""
    computer = MetricsComputer()
    
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    
    # Identical images
    psnr_identical = computer._compute_psnr(ref, ref)
    assert psnr_identical > 80.0
    
    # With noise
    noise = np.random.randn(100, 100, 3).astype(np.float32) * 0.01
    proc = np.clip(ref + noise, 0, 1)
    psnr_noise = computer._compute_psnr(ref, proc)
    assert 30.0 < psnr_noise < 50.0


def test_compute_mae():
    """Test MAE computation."""
    computer = MetricsComputer()
    
    ref = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    
    # Identical
    mae_identical = computer._compute_mae(ref, ref)
    assert mae_identical < 1e-6
    
    # With offset
    proc = ref + 0.1
    mae_offset = computer._compute_mae(ref, proc)
    assert abs(mae_offset - 0.1) < 0.01


def test_compute_mse():
    """Test MSE computation."""
    computer = MetricsComputer()
    
    ref = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    
    # Identical
    mse_identical = computer._compute_mse(ref, ref)
    assert mse_identical < 1e-10
    
    # With offset
    proc = ref + 0.1
    mse_offset = computer._compute_mse(ref, proc)
    assert abs(mse_offset - 0.01) < 0.001


def test_compute_weighted_score():
    """Test weighted quality score."""
    computer = MetricsComputer()
    
    # Perfect metrics
    metrics = QualityMetrics(ssim=1.0, psnr=50.0, lpips=0.0, nima=10.0)
    score = computer.compute_weighted_score(metrics)
    
    assert 0.0 <= score <= 1.0
    assert score > 0.9  # Should be near perfect
    
    # Poor metrics
    metrics_poor = QualityMetrics(ssim=0.5, psnr=20.0, lpips=0.5, nima=3.0)
    score_poor = computer.compute_weighted_score(metrics_poor)
    
    assert score_poor < score


def test_compute_with_grayscale():
    """Test metrics with grayscale images."""
    computer = MetricsComputer()
    
    # Grayscale images (H, W)
    ref = np.random.rand(100, 100).astype(np.float32)
    proc = ref + np.random.randn(100, 100).astype(np.float32) * 0.01
    proc = np.clip(proc, 0, 1)
    
    metrics = computer.compute(ref, proc, metrics=["ssim", "psnr"])
    
    assert metrics.ssim > 0.9
    assert metrics.psnr > 30.0


def test_compute_subset_metrics():
    """Test computing only specific metrics."""
    computer = MetricsComputer()
    
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    proc = ref.copy()
    
    # Only SSIM
    metrics = computer.compute(ref, proc, metrics=["ssim"])
    assert metrics.ssim is not None
    assert metrics.psnr is None
    assert metrics.mae is None


@pytest.mark.skipif(not pytest.importorskip("skimage", reason="scikit-image not available"),
                    reason="scikit-image not available")
def test_compute_ssim_with_skimage():
    """Test SSIM with scikit-image."""
    computer = MetricsComputer()
    
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    ssim = computer._compute_ssim(ref, ref)
    
    # Should use scikit-image implementation
    assert ssim > 0.99
