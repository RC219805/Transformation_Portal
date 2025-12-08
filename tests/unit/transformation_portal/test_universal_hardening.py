"""
Unit tests for universal hardening wrapper.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from transformation_portal.hardening import UniversalHardenedWrapper, Pipeline, wrap_function


class MockPipeline:
    """Mock pipeline for testing."""
    
    def __init__(self, should_fail=False):
        self.should_fail = should_fail
        self.process_called = False
    
    def process(self, input_path: Path, **kwargs):
        self.process_called = True
        if self.should_fail:
            raise RuntimeError("Pipeline failed")
        return {"processed": True, "path": str(input_path)}


@pytest.fixture
def mock_policy():
    """Create mock hardening policy."""
    policy = Mock()
    policy.max_input_bytes = 50 * 1024 * 1024  # 50MB
    policy.allowed_input_exts = [".tif", ".tiff", ".png", ".jpg"]
    return policy


@pytest.fixture
def test_image_path(tmp_path):
    """Create a test image file."""
    image_path = tmp_path / "test.tif"
    image_path.write_bytes(b"fake image data")
    return image_path


def test_wrapper_initialization():
    """Test wrapper can be initialized."""
    pipeline = MockPipeline()
    wrapper = UniversalHardenedWrapper(
        pipeline,
        policy=None,
        enable_profiling=True,
        enable_stamping=True,
        enable_input_validation=False
    )
    
    assert wrapper.pipeline is pipeline
    assert wrapper.enable_profiling is True
    assert wrapper.enable_stamping is True
    assert wrapper.enable_input_validation is False


def test_wrapper_processes_successfully(test_image_path, mock_policy):
    """Test wrapper processes successfully."""
    pipeline = MockPipeline()
    
    with patch('transformation_portal.hardening.universal.validate_input_path', return_value=test_image_path):
        wrapper = UniversalHardenedWrapper(
            pipeline,
            policy=mock_policy,
            enable_input_validation=False
        )
        
        result = wrapper.process(test_image_path, preset="test")
        
        assert result["success"] is True
        assert result["result"]["processed"] is True
        assert pipeline.process_called is True


def test_wrapper_handles_pipeline_failure(test_image_path, mock_policy):
    """Test wrapper handles pipeline failures gracefully."""
    pipeline = MockPipeline(should_fail=True)
    
    wrapper = UniversalHardenedWrapper(
        pipeline,
        policy=mock_policy,
        enable_input_validation=False
    )
    
    result = wrapper.process(test_image_path)
    
    assert result["success"] is False
    assert result["result"] is None


def test_wrapper_includes_report_when_enabled(test_image_path, mock_policy):
    """Test wrapper includes report when stamping enabled."""
    pipeline = MockPipeline()
    
    wrapper = UniversalHardenedWrapper(
        pipeline,
        policy=mock_policy,
        enable_stamping=True,
        enable_input_validation=False
    )
    
    result = wrapper.process(test_image_path, preset="test")
    
    assert "report" in result
    assert result["report"].run_id is not None
    assert result["report"].config_hash is not None
    assert result["report"].success is True


def test_wrapper_measures_duration_when_profiling(test_image_path, mock_policy):
    """Test wrapper measures duration when profiling enabled."""
    pipeline = MockPipeline()
    
    wrapper = UniversalHardenedWrapper(
        pipeline,
        policy=mock_policy,
        enable_profiling=True,
        enable_stamping=True,
        enable_input_validation=False
    )
    
    result = wrapper.process(test_image_path)
    
    assert result["report"].duration_ms is not None
    assert result["report"].duration_ms > 0


def test_wrapper_no_report_when_disabled(test_image_path, mock_policy):
    """Test wrapper doesn't include report when stamping disabled."""
    pipeline = MockPipeline()
    
    wrapper = UniversalHardenedWrapper(
        pipeline,
        policy=mock_policy,
        enable_stamping=False,
        enable_input_validation=False
    )
    
    result = wrapper.process(test_image_path)
    
    assert "report" not in result
    assert result["success"] is True


def test_wrap_function():
    """Test wrap_function utility."""
    def mock_process_func(input_path, **kwargs):
        return {"processed": True, "path": str(input_path)}
    
    wrapper = wrap_function(
        mock_process_func,
        policy=None,
        enable_input_validation=False
    )
    
    assert isinstance(wrapper, UniversalHardenedWrapper)


def test_config_hash_is_deterministic(test_image_path, mock_policy):
    """Test config hash is deterministic for same config."""
    pipeline = MockPipeline()
    wrapper = UniversalHardenedWrapper(
        pipeline,
        policy=mock_policy,
        enable_input_validation=False
    )
    
    config1 = {"preset": "test", "value": 42}
    config2 = {"value": 42, "preset": "test"}  # Different order
    
    hash1 = wrapper._compute_config_hash(config1)
    hash2 = wrapper._compute_config_hash(config2)
    
    assert hash1 == hash2  # Should be same despite different key order


def test_wrapper_validates_input_when_enabled(test_image_path, mock_policy):
    """Test wrapper validates input when validation enabled."""
    pipeline = MockPipeline()
    
    with patch('transformation_portal.hardening.universal.validate_input_path') as mock_validate:
        mock_validate.return_value = test_image_path
        
        wrapper = UniversalHardenedWrapper(
            pipeline,
            policy=mock_policy,
            enable_input_validation=True
        )
        
        result = wrapper.process(test_image_path)
        
        mock_validate.assert_called_once()
        assert result["success"] is True


def test_wrapper_handles_validation_failure(test_image_path, mock_policy):
    """Test wrapper handles validation failures."""
    pipeline = MockPipeline()
    
    with patch('transformation_portal.hardening.universal.validate_input_path') as mock_validate:
        mock_validate.side_effect = ValueError("Invalid file")
        
        wrapper = UniversalHardenedWrapper(
            pipeline,
            policy=mock_policy,
            enable_input_validation=True
        )
        
        result = wrapper.process(test_image_path)
        
        assert result["success"] is False
        assert "Invalid file" in result["report"].error


@pytest.mark.parametrize("enable_profiling,enable_stamping", [
    (True, True),
    (True, False),
    (False, True),
    (False, False),
])
def test_wrapper_feature_combinations(test_image_path, mock_policy, enable_profiling, enable_stamping):
    """Test wrapper works with all feature combinations."""
    pipeline = MockPipeline()
    
    wrapper = UniversalHardenedWrapper(
        pipeline,
        policy=mock_policy,
        enable_profiling=enable_profiling,
        enable_stamping=enable_stamping,
        enable_input_validation=False
    )
    
    result = wrapper.process(test_image_path)
    
    assert result["success"] is True
    
    if enable_stamping:
        assert "report" in result
        if enable_profiling:
            assert result["report"].duration_ms is not None
        else:
            assert result["report"].duration_ms is None
    else:
        assert "report" not in result
