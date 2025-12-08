"""Tests for validation module."""

import numpy as np
import pytest
from pathlib import Path
from unittest.mock import Mock, patch

from lux_depth_v2.validation import QualityValidator, ValidationReport, ComparisonReport
from lux_depth_v2.validation import metrics, degradation


def test_compute_ssim():
    """Test SSIM computation."""
    # Perfect match
    img1 = np.random.rand(100, 100, 3).astype(np.float32)
    ssim_perfect = metrics.compute_ssim(img1, img1)
    assert 0.99 <= ssim_perfect <= 1.0
    
    # Different images (fallback implementation can produce slightly negative values)
    img2 = np.random.rand(100, 100, 3).astype(np.float32)
    ssim_diff = metrics.compute_ssim(img1, img2)
    assert -0.1 <= ssim_diff < 0.99  # Wider range for fallback implementation


def test_compute_psnr():
    """Test PSNR computation."""
    # Perfect match
    img1 = np.random.rand(100, 100, 3).astype(np.float32)
    psnr_perfect = metrics.compute_psnr(img1, img1)
    assert psnr_perfect > 60.0  # Should be very high
    
    # Different images (random noise can have very low PSNR)
    img2 = np.random.rand(100, 100, 3).astype(np.float32)
    psnr_diff = metrics.compute_psnr(img1, img2)
    assert 5.0 <= psnr_diff <= 50.0  # Wider range for random images


def test_heuristic_aesthetic_score():
    """Test heuristic aesthetic scoring."""
    # High quality image (good dynamic range, contrast)
    img_hq = np.linspace(0, 1, 10000).reshape(100, 100, 1).repeat(3, axis=2).astype(np.float32)
    score_hq = metrics._heuristic_aesthetic_score(img_hq)
    assert 1.0 <= score_hq <= 10.0
    
    # Low quality image (flat, no contrast)
    img_lq = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    score_lq = metrics._heuristic_aesthetic_score(img_lq)
    assert 1.0 <= score_lq <= 10.0
    assert score_lq < score_hq  # High quality should score higher


def test_apply_downsample_degradation():
    """Test downsampling degradation."""
    img = np.random.rand(100, 100, 3).astype(np.float32)
    downsampled = degradation.apply_downsample_degradation(img, scale=2)
    assert downsampled.shape[0] == 50
    assert downsampled.shape[1] == 50


def test_apply_noise_degradation():
    """Test noise degradation."""
    img = np.ones((100, 100, 3), dtype=np.float32) * 0.5
    noisy = degradation.apply_noise_degradation(img, noise_level=0.1)
    assert noisy.shape == img.shape
    assert np.all(noisy >= 0.0)
    assert np.all(noisy <= 1.0)
    assert not np.allclose(noisy, img)  # Should be different


def test_create_synthetic_pair():
    """Test synthetic pair creation."""
    original = np.random.rand(200, 200, 3).astype(np.float32)
    degraded, reference = degradation.create_synthetic_pair(
        original,
        degradations=["downsample", "noise"]
    )
    
    # Degraded should be smaller
    assert degraded.shape[0] < original.shape[0]
    assert degraded.shape[1] < original.shape[1]
    
    # Reference should be same as original
    assert np.allclose(reference, original)


def test_quality_validator_init():
    """Test QualityValidator initialization."""
    validator = QualityValidator(device="cpu")
    assert validator.device == "cpu"
    assert validator.default_weights is not None
    assert "ssim" in validator.default_weights


def test_validation_report_to_dict():
    """Test ValidationReport serialization."""
    report = ValidationReport(
        mode="real",
        test_images=["img1.tif", "img2.tif"],
        metrics_scores={"ssim": 0.95, "psnr": 38.2},
        composite_score=0.87
    )
    
    report_dict = report.to_dict()
    assert report_dict["mode"] == "real"
    assert len(report_dict["test_images"]) == 2
    assert report_dict["metrics_scores"]["ssim"] == 0.95
    assert report_dict["composite_score"] == 0.87


def test_comparison_report_to_dict():
    """Test ComparisonReport serialization."""
    report = ComparisonReport(
        our_method="LuxDepthV2",
        baseline_method="Topaz",
        test_images=["img1.tif"],
        our_scores={"ssim": 0.95},
        baseline_scores={"ssim": 0.92},
        our_wins=1,
        baseline_wins=0,
        ties=0
    )
    
    report_dict = report.to_dict()
    assert report_dict["our_method"] == "LuxDepthV2"
    assert report_dict["our_wins"] == 1


def test_compute_composite_score():
    """Test composite score computation."""
    validator = QualityValidator()
    
    metric_scores = {
        "ssim": 0.95,
        "psnr": 40.0,
        "lpips": 0.10,
        "nima": 8.0
    }
    
    weights = {
        "ssim": 0.25,
        "psnr": 0.15,
        "lpips": 0.35,
        "nima": 0.25
    }
    
    composite = validator._compute_composite_score(metric_scores, weights)
    
    # Composite should be in [0, 1]
    assert 0.0 <= composite <= 1.0
    
    # With high scores, composite should be high
    assert composite > 0.7


def test_compare_metrics():
    """Test metric comparison logic."""
    validator = QualityValidator()
    
    # Our method better
    ours = {"ssim": 0.95, "psnr": 40.0, "lpips": 0.10}
    baseline = {"ssim": 0.90, "psnr": 35.0, "lpips": 0.15}
    result = validator._compare_metrics(ours, baseline)
    assert result > 0  # We win
    
    # Baseline better
    ours = {"ssim": 0.85, "psnr": 32.0, "lpips": 0.20}
    baseline = {"ssim": 0.95, "psnr": 40.0, "lpips": 0.10}
    result = validator._compare_metrics(ours, baseline)
    assert result < 0  # Baseline wins
    
    # Tie (similar scores)
    ours = {"ssim": 0.90, "psnr": 38.0}
    baseline = {"ssim": 0.90, "psnr": 38.0}
    result = validator._compare_metrics(ours, baseline)
    assert result == 0  # Tie


@pytest.mark.parametrize("mode", ["real", "synthetic"])
def test_validate_batch_mode(mode, tmp_path):
    """Test validation in different modes."""
    validator = QualityValidator(device="cpu")
    
    # Create mock images
    test_images = []
    for i in range(3):
        img_path = tmp_path / f"test_{i}.tif"
        # Create dummy image file (will be mocked in _load_image)
        img_path.touch()
        test_images.append(img_path)
    
    # Mock image loading
    with patch.object(validator, '_load_image') as mock_load:
        mock_load.return_value = np.random.rand(100, 100, 3).astype(np.float32)
        
        # Mock reference finding for synthetic mode
        with patch.object(validator, '_find_reference') as mock_find_ref:
            if mode == "synthetic":
                mock_find_ref.return_value = tmp_path / "reference.tif"
            else:
                mock_find_ref.return_value = None
            
            report = validator.validate_batch(
                test_images=test_images,
                output_dir=tmp_path / "validation",
                mode=mode
            )
    
    assert report.mode == mode
    assert len(report.test_images) == 3
    assert len(report.per_image_scores) == 3
    assert report.composite_score is not None
    assert 0.0 <= report.composite_score <= 1.0


def test_aggregate_scores():
    """Test score aggregation."""
    validator = QualityValidator()
    
    per_image_scores = [
        {"image": "img1.tif", "metrics": {"ssim": 0.95, "psnr": 40.0}},
        {"image": "img2.tif", "metrics": {"ssim": 0.90, "psnr": 38.0}},
        {"image": "img3.tif", "metrics": {"ssim": 0.92, "psnr": 39.0}},
    ]
    
    aggregated = validator._aggregate_scores(per_image_scores)
    
    assert "ssim" in aggregated
    assert "psnr" in aggregated
    assert abs(aggregated["ssim"] - 0.923333) < 0.01  # Mean of 0.95, 0.90, 0.92
    assert abs(aggregated["psnr"] - 39.0) < 0.1  # Mean of 40, 38, 39


def test_get_timestamp():
    """Test timestamp generation."""
    timestamp = QualityValidator._get_timestamp()
    assert isinstance(timestamp, str)
    assert "UTC" in timestamp
    assert len(timestamp) > 10  # Should be reasonable length


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
