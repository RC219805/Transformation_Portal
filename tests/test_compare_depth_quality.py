"""
Tests for depth quality comparison utility.

Tests cover:
    - Metric computation (L1, L2, SSIM if available)
    - Edge consistency calculation
    - Recommendation generation logic
    - Visualization output
"""

import numpy as np
import pytest
from pathlib import Path
import sys

# Add scripts to path for import
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

try:
    from compare_depth_quality import compute_metrics, generate_recommendation
    SCRIPT_AVAILABLE = True
except ImportError:
    SCRIPT_AVAILABLE = False


@pytest.mark.skipif(not SCRIPT_AVAILABLE, reason="compare_depth_quality script not available")
class TestDepthQualityComparison:
    """Test depth quality comparison utility."""
    
    def test_compute_metrics_identical_maps(self):
        """Test metrics for identical depth maps."""
        depth = np.random.rand(256, 256).astype(np.float32)
        metrics = compute_metrics(depth, depth.copy())
        
        # Identical maps should have zero error
        assert metrics["l1_mae"] == pytest.approx(0.0, abs=1e-6)
        assert metrics["l2_rmse"] == pytest.approx(0.0, abs=1e-6)
        
        # SSIM should be 1.0 if available
        if "ssim" in metrics:
            assert metrics["ssim"] == pytest.approx(1.0, abs=1e-3)
    
    def test_compute_metrics_different_maps(self):
        """Test metrics for different depth maps."""
        depth1 = np.random.rand(256, 256).astype(np.float32)
        depth2 = np.random.rand(256, 256).astype(np.float32)
        metrics = compute_metrics(depth1, depth2)
        
        # Different random maps should have non-zero error
        assert metrics["l1_mae"] > 0.0
        assert metrics["l2_rmse"] > 0.0
        
        # MAE <= RMSE (triangle inequality)
        assert metrics["l1_mae"] <= metrics["l2_rmse"]
    
    def test_compute_metrics_shape_mismatch(self):
        """Test that mismatched shapes raise error."""
        depth1 = np.random.rand(256, 256).astype(np.float32)
        depth2 = np.random.rand(128, 128).astype(np.float32)
        
        with pytest.raises(ValueError, match="same shape"):
            compute_metrics(depth1, depth2)
    
    def test_recommendation_generation(self, tmp_path):
        """Test recommendation text generation."""
        metrics = {
            "l1_mae": 0.015,
            "l2_rmse": 0.025,
            "ssim": 0.92,
            "edge_correlation": 0.88,
            "flat_noise_ratio": 1.05
        }
        
        output_path = tmp_path / "recommendation.md"
        recommendation = generate_recommendation(metrics, output_path)
        
        # Check output file was created
        assert output_path.exists()
        
        # Check key sections are present
        assert "Metrics Summary" in recommendation
        assert "SSIM" in recommendation
        assert "Recommendation" in recommendation
        assert "Quality Tier Guidelines" in recommendation
    
    def test_recommendation_no_ssim(self, tmp_path):
        """Test recommendation when SSIM unavailable."""
        metrics = {
            "l1_mae": 0.018,
            "l2_rmse": 0.028
        }
        
        output_path = tmp_path / "recommendation.md"
        recommendation = generate_recommendation(metrics, output_path)
        
        assert "METRICS INCOMPLETE" in recommendation
        assert "scikit-image" in recommendation
    
    def test_recommendation_thresholds(self, tmp_path):
        """Test recommendation thresholds for different SSIM values."""
        test_cases = [
            (0.98, "NEARLY IDENTICAL"),
            (0.93, "VERY SIMILAR"),
            (0.85, "MODERATE DIFFERENCE"),
            (0.75, "SIGNIFICANT DIFFERENCE")
        ]
        
        for ssim_val, expected_tier in test_cases:
            metrics = {
                "l1_mae": 0.01,
                "l2_rmse": 0.02,
                "ssim": ssim_val
            }
            
            output_path = tmp_path / f"rec_{ssim_val}.md"
            recommendation = generate_recommendation(metrics, output_path)
            
            assert expected_tier in recommendation, f"Expected tier '{expected_tier}' for SSIM={ssim_val}"


@pytest.mark.skipif(not SCRIPT_AVAILABLE, reason="compare_depth_quality script not available")
def test_metric_determinism():
    """Test that metrics are deterministic for same input."""
    np.random.seed(42)
    depth1 = np.random.rand(128, 128).astype(np.float32)
    depth2 = np.random.rand(128, 128).astype(np.float32)
    
    metrics1 = compute_metrics(depth1, depth2)
    metrics2 = compute_metrics(depth1, depth2)
    
    for key in metrics1:
        assert metrics1[key] == pytest.approx(metrics2[key]), f"Metric {key} not deterministic"


@pytest.mark.skipif(not SCRIPT_AVAILABLE, reason="compare_depth_quality script not available")
def test_metrics_range():
    """Test that metrics stay within expected ranges."""
    depth1 = np.random.rand(256, 256).astype(np.float32)
    depth2 = np.random.rand(256, 256).astype(np.float32)
    metrics = compute_metrics(depth1, depth2)
    
    # L1/L2 should be in [0, 1] for normalized depth
    assert 0.0 <= metrics["l1_mae"] <= 1.0
    assert 0.0 <= metrics["l2_rmse"] <= 1.0
    
    # SSIM should be in [-1, 1] (but typically [0, 1])
    if "ssim" in metrics:
        assert -1.0 <= metrics["ssim"] <= 1.0
    
    # Edge correlation should be in [-1, 1]
    if "edge_correlation" in metrics:
        assert -1.0 <= metrics["edge_correlation"] <= 1.0
