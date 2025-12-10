"""Tests for processing reports."""

import pytest
from pathlib import Path
import json
import tempfile

from src.transformation_portal.core.validation.report import (
    ProcessingReport,
    GitInfo,
    DeviceInfo,
    ModelInfo
)


def test_git_info_capture():
    """Test capturing git information."""
    git_info = GitInfo.capture()
    
    if git_info is not None:
        assert isinstance(git_info.commit, str)
        assert len(git_info.commit) == 40  # SHA-1 hash
        assert isinstance(git_info.branch, str)
        assert isinstance(git_info.is_dirty, bool)


def test_git_info_not_in_repo(tmp_path):
    """Test git info capture outside repository."""
    # Change to temp directory (not a git repo)
    git_info = GitInfo.capture(tmp_path)
    assert git_info is None


def test_device_info_capture():
    """Test capturing device information."""
    device_info = DeviceInfo.capture()
    
    assert device_info.device_type in ("cpu", "cuda", "mps")
    assert device_info.device_name is not None
    assert device_info.python_version is not None
    assert device_info.platform is not None
    assert device_info.cpu_count >= 1


def test_model_info_from_weights(tmp_path):
    """Test model info from weights file."""
    # Create dummy weights file
    weights_path = tmp_path / "model.pth"
    weights_path.write_bytes(b"dummy weights data")
    
    model_info = ModelInfo.from_weights("test_model", weights_path)
    
    assert model_info.model_name == "test_model"
    assert model_info.checkpoint_sha256 is not None
    assert len(model_info.checkpoint_sha256) == 64  # SHA-256 hash


def test_model_info_missing_weights():
    """Test model info without weights file."""
    model_info = ModelInfo.from_weights("test_model", Path("/nonexistent.pth"))
    
    assert model_info.model_name == "test_model"
    assert model_info.checkpoint_sha256 is None


def test_processing_report_create():
    """Test creating processing report."""
    config = {
        "preset": "test_preset",
        "param1": "value1",
        "param2": 42
    }
    
    report = ProcessingReport.create(
        config=config,
        input_path=Path("input.jpg"),
        output_path=Path("output.jpg"),
        duration_ms=100.5,
        metrics={"ssim": 0.95, "psnr": 35.2},
        success=True
    )
    
    assert report.preset == "test_preset"
    assert report.input_path == "input.jpg"
    assert report.output_path == "output.jpg"
    assert report.duration_ms == 100.5
    assert report.metrics["ssim"] == 0.95
    assert report.success is True
    assert report.config_hash is not None
    assert len(report.config_hash) == 64  # SHA-256


def test_processing_report_save_load(tmp_path):
    """Test saving and loading report."""
    config = {"preset": "test"}
    
    report = ProcessingReport.create(
        config=config,
        input_path=Path("input.jpg"),
        output_path=Path("output.jpg"),
        duration_ms=100.0,
        metrics={"ssim": 0.95}
    )
    
    # Save report
    report_path = tmp_path / "report.json"
    report.save(report_path)
    
    assert report_path.exists()
    
    # Load report
    loaded = ProcessingReport.load(report_path)
    
    assert loaded.preset == report.preset
    assert loaded.duration_ms == report.duration_ms
    assert loaded.metrics == report.metrics
    assert loaded.config_hash == report.config_hash


def test_processing_report_to_dict():
    """Test converting report to dictionary."""
    config = {"preset": "test"}
    
    report = ProcessingReport.create(
        config=config,
        input_path=Path("input.jpg"),
        output_path=Path("output.jpg"),
        duration_ms=100.0,
        metrics={"ssim": 0.95}
    )
    
    data = report.to_dict()
    
    assert isinstance(data, dict)
    assert "device_info" in data
    assert "metrics" in data
    assert "timestamp" in data


def test_processing_report_with_error():
    """Test report with error information."""
    config = {"preset": "test"}
    
    report = ProcessingReport.create(
        config=config,
        input_path=Path("input.jpg"),
        output_path=Path("output.jpg"),
        duration_ms=50.0,
        metrics={},
        success=False,
        error="Processing failed: out of memory"
    )
    
    assert report.success is False
    assert report.error == "Processing failed: out of memory"


def test_processing_report_with_metadata():
    """Test report with additional metadata."""
    config = {"preset": "test"}
    metadata = {
        "batch_id": "batch_123",
        "user": "test_user",
        "tags": ["test", "validation"]
    }
    
    report = ProcessingReport.create(
        config=config,
        input_path=Path("input.jpg"),
        output_path=Path("output.jpg"),
        duration_ms=100.0,
        metrics={"ssim": 0.95},
        metadata=metadata
    )
    
    assert report.metadata["batch_id"] == "batch_123"
    assert report.metadata["tags"] == ["test", "validation"]


def test_processing_report_with_model_info():
    """Test report with model information."""
    config = {"preset": "test"}
    model_info = ModelInfo(
        model_name="depth_model",
        model_version="v2.0",
        checkpoint_sha256="abc123"
    )
    
    report = ProcessingReport.create(
        config=config,
        input_path=Path("input.jpg"),
        output_path=Path("output.jpg"),
        duration_ms=100.0,
        metrics={"ssim": 0.95},
        model_info=model_info
    )
    
    assert report.model_info.model_name == "depth_model"
    assert report.model_info.model_version == "v2.0"
