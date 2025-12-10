"""Tests for baseline comparison."""

import pytest
from pathlib import Path
import json

from src.transformation_portal.core.validation.comparison import (
    BaselineComparator,
    ComparisonResult,
    ComparisonStatus
)


def test_baseline_comparator_no_baseline(tmp_path):
    """Test comparator with no baseline file."""
    comparator = BaselineComparator(tmp_path)
    
    metrics = {"ssim": 0.95, "psnr": 35.0}
    result = comparator.compare("test_preset", metrics)
    
    assert result.status == ComparisonStatus.NO_BASELINE
    assert result.current == metrics


def test_baseline_comparator_with_baseline(tmp_path):
    """Test comparator with existing baseline."""
    # Create baseline file
    baseline_data = {
        "test_preset": {
            "ssim": 0.90,
            "psnr": 33.0
        }
    }
    
    baseline_path = tmp_path / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)
    
    comparator = BaselineComparator(tmp_path)
    
    # Compare metrics
    metrics = {"ssim": 0.95, "psnr": 35.0}
    result = comparator.compare("test_preset", metrics)
    
    assert result.status == ComparisonStatus.IMPROVEMENT
    assert abs(result.delta["ssim"] - 0.05) < 0.001
    assert abs(result.delta["psnr"] - 2.0) < 0.001


def test_baseline_comparator_regression(tmp_path):
    """Test detection of regression."""
    baseline_data = {
        "test_preset": {
            "ssim": 0.95,
            "psnr": 35.0
        }
    }
    
    baseline_path = tmp_path / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)
    
    comparator = BaselineComparator(tmp_path, threshold=0.05)
    
    # Worse metrics (regression)
    metrics = {"ssim": 0.85, "psnr": 30.0}
    result = comparator.compare("test_preset", metrics)
    
    assert result.status == ComparisonStatus.REGRESSION
    assert result.delta["ssim"] < -0.05


def test_baseline_comparator_stable(tmp_path):
    """Test stable comparison (within threshold)."""
    baseline_data = {
        "test_preset": {
            "ssim": 0.95,
            "psnr": 35.0
        }
    }
    
    baseline_path = tmp_path / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)
    
    comparator = BaselineComparator(tmp_path, threshold=0.05)
    
    # Similar metrics (within threshold)
    metrics = {"ssim": 0.950, "psnr": 35.01}  # Extremely close to baseline
    result = comparator.compare("test_preset", metrics)
    
    assert result.status == ComparisonStatus.STABLE


def test_baseline_comparator_update(tmp_path):
    """Test updating baseline."""
    comparator = BaselineComparator(tmp_path)
    
    # Update baseline
    metrics = {"ssim": 0.95, "psnr": 35.0}
    comparator.update_baseline("test_preset", metrics)
    
    # Verify saved
    baseline_path = tmp_path / "baseline_metrics.json"
    assert baseline_path.exists()
    
    with open(baseline_path) as f:
        data = json.load(f)
    
    assert data["test_preset"]["ssim"] == 0.95
    assert data["test_preset"]["psnr"] == 35.0


def test_baseline_comparator_list_presets(tmp_path):
    """Test listing presets."""
    baseline_data = {
        "preset1": {"ssim": 0.95},
        "preset2": {"ssim": 0.90}
    }
    
    baseline_path = tmp_path / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)
    
    comparator = BaselineComparator(tmp_path)
    
    presets = comparator.list_presets()
    assert "preset1" in presets
    assert "preset2" in presets


def test_baseline_comparator_get_baseline(tmp_path):
    """Test getting baseline for preset."""
    baseline_data = {
        "test_preset": {"ssim": 0.95, "psnr": 35.0}
    }
    
    baseline_path = tmp_path / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)
    
    comparator = BaselineComparator(tmp_path)
    
    baseline = comparator.get_baseline("test_preset")
    assert baseline["ssim"] == 0.95
    
    missing = comparator.get_baseline("nonexistent")
    assert missing is None


def test_baseline_comparator_has_baseline(tmp_path):
    """Test checking if baseline exists."""
    baseline_data = {
        "test_preset": {"ssim": 0.95}
    }
    
    baseline_path = tmp_path / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)
    
    comparator = BaselineComparator(tmp_path)
    
    assert comparator.has_baseline("test_preset")
    assert not comparator.has_baseline("nonexistent")


def test_comparison_result_to_dict():
    """Test converting comparison result to dict."""
    result = ComparisonResult(
        status=ComparisonStatus.IMPROVEMENT,
        delta={"ssim": 0.05},
        baseline={"ssim": 0.90},
        current={"ssim": 0.95},
        threshold=0.05
    )
    
    data = result.to_dict()
    
    assert data["status"] == "improvement"
    assert data["delta"]["ssim"] == 0.05
    assert data["threshold"] == 0.05


def test_comparison_result_str_no_baseline():
    """Test string representation with no baseline."""
    result = ComparisonResult(
        status=ComparisonStatus.NO_BASELINE,
        delta={},
        baseline={},
        current={"ssim": 0.95},
        threshold=0.05
    )
    
    string = str(result)
    assert "no_baseline" in string
    assert "No baseline available" in string


def test_comparison_result_str_with_changes():
    """Test string representation with changes."""
    result = ComparisonResult(
        status=ComparisonStatus.IMPROVEMENT,
        delta={"ssim": 0.05, "psnr": 2.0},
        baseline={"ssim": 0.90, "psnr": 33.0},
        current={"ssim": 0.95, "psnr": 35.0},
        threshold=0.05
    )
    
    string = str(result)
    assert "improvement" in string
    assert "ssim" in string
    assert "psnr" in string


def test_baseline_comparator_custom_threshold(tmp_path):
    """Test comparator with custom threshold."""
    baseline_data = {
        "test_preset": {"ssim": 0.95}
    }
    
    baseline_path = tmp_path / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)
    
    # Strict threshold (1%)
    comparator = BaselineComparator(tmp_path, threshold=0.01)
    
    # Small improvement should be detected
    metrics = {"ssim": 0.96}  # +0.01
    result = comparator.compare("test_preset", metrics)
    
    # With 1% threshold, 0.01 improvement should be detected
    assert result.status in (ComparisonStatus.IMPROVEMENT, ComparisonStatus.STABLE)


def test_baseline_comparator_partial_metrics(tmp_path):
    """Test comparison with partial metrics."""
    baseline_data = {
        "test_preset": {
            "ssim": 0.95,
            "psnr": 35.0,
            "mae": 0.05
        }
    }
    
    baseline_path = tmp_path / "baseline_metrics.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_data, f)
    
    comparator = BaselineComparator(tmp_path)
    
    # Only some metrics
    metrics = {"ssim": 0.96, "psnr": 36.0}
    result = comparator.compare("test_preset", metrics)
    
    # Should only compare available metrics
    assert "ssim" in result.delta
    assert "psnr" in result.delta
    assert "mae" not in result.delta
