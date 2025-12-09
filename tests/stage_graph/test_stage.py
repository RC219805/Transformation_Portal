"""
Tests for core stage abstraction.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from src.transformation_portal.stage_graph.stage import (
    Stage,
    StageContext,
    StageResult,
    StageStatus,
)


class SimpleStage(Stage):
    """Simple test stage for testing."""
    
    def compute(self, context: StageContext) -> StageResult:
        """Double the input value."""
        value = context.get_artifact("input_value", 0)
        
        return StageResult(
            stage_name=self.name,
            stage_version=self.version,
            status=StageStatus.COMPLETED,
            artifacts={"output_value": value * 2},
        )
    
    def get_cache_key(self, context: StageContext) -> str:
        """Cache key based on input value."""
        value = context.get_artifact("input_value", 0)
        return f"simple_{self.version}_{value}"


class FailingStage(Stage):
    """Stage that always fails for testing error handling."""
    
    def compute(self, context: StageContext) -> StageResult:
        """Raise an error."""
        raise ValueError("Intentional failure for testing")
    
    def get_cache_key(self, context: StageContext) -> str:
        """Cache key."""
        return "failing_stage"


def test_stage_basic_execution():
    """Test basic stage execution."""
    stage = SimpleStage(name="test", version="1.0.0")
    
    context = StageContext(
        artifacts={"input_value": 5},
        cache_enabled=False,
    )
    
    result = stage.execute(context)
    
    assert result.is_success()
    assert result.get_artifact("output_value") == 10
    assert result.stage_name == "test"
    assert result.stage_version == "1.0.0"
    assert result.duration_ms > 0


def test_stage_caching():
    """Test stage caching functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        stage = SimpleStage(name="test", version="1.0.0")
        
        context = StageContext(
            artifacts={"input_value": 5},
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        
        # First execution - cache miss
        result1 = stage.execute(context)
        assert result1.is_success()
        assert not result1.cache_hit
        assert result1.get_artifact("output_value") == 10
        
        # Second execution - cache hit
        result2 = stage.execute(context)
        assert result2.is_success()
        assert result2.cache_hit
        assert result2.get_artifact("output_value") == 10


def test_stage_cache_invalidation():
    """Test cache invalidation on input change."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        stage = SimpleStage(name="test", version="1.0.0")
        
        # First execution with value=5
        context1 = StageContext(
            artifacts={"input_value": 5},
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        result1 = stage.execute(context1)
        assert not result1.cache_hit
        
        # Second execution with value=10 - different cache key
        context2 = StageContext(
            artifacts={"input_value": 10},
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        result2 = stage.execute(context2)
        assert not result2.cache_hit  # Different input
        assert result2.get_artifact("output_value") == 20


def test_stage_error_handling():
    """Test stage error handling."""
    stage = FailingStage(name="fail", version="1.0.0")
    
    context = StageContext(cache_enabled=False)
    
    result = stage.execute(context)
    
    assert not result.is_success()
    assert result.status == StageStatus.FAILED
    assert "Intentional failure" in result.error
    assert result.error_traceback is not None


def test_stage_context():
    """Test stage context operations."""
    context = StageContext()
    
    # Test artifact operations
    context.set_artifact("key1", "value1")
    assert context.get_artifact("key1") == "value1"
    assert context.get_artifact("missing", "default") == "default"
    
    # Test config operations
    context.config = {"param1": 42}
    assert context.get_config("param1") == 42
    assert context.get_config("missing", 0) == 0


def test_stage_result():
    """Test stage result operations."""
    result = StageResult(
        stage_name="test",
        stage_version="1.0.0",
        status=StageStatus.COMPLETED,
        artifacts={"output": 42},
    )
    
    assert result.is_success()
    assert result.get_artifact("output") == 42
    assert result.get_artifact("missing", 0) == 0
    
    # Test failed result
    failed_result = StageResult(
        stage_name="test",
        stage_version="1.0.0",
        status=StageStatus.FAILED,
        error="Test error",
    )
    
    assert not failed_result.is_success()


def test_stage_metadata():
    """Test stage metadata tracking."""
    stage = SimpleStage(name="test", version="1.0.0")
    
    context = StageContext(
        artifacts={"input_value": 5},
        run_id="test-run-123",
        metadata={"user": "test_user"},
    )
    
    result = stage.execute(context)
    
    assert result.stage_name == "test"
    assert result.stage_version == "1.0.0"
    assert result.timestamp > 0
    assert result.duration_ms > 0


def test_stage_cache_disabled():
    """Test stage execution with caching disabled."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        stage = SimpleStage(name="test", version="1.0.0")
        
        context = StageContext(
            artifacts={"input_value": 5},
            cache_enabled=False,  # Disabled
            cache_dir=cache_dir,
        )
        
        # Multiple executions - should never hit cache
        result1 = stage.execute(context)
        assert not result1.cache_hit
        
        result2 = stage.execute(context)
        assert not result2.cache_hit


def test_stage_version_invalidation():
    """Test cache invalidation on version change."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_dir = Path(tmpdir)
        
        # First version
        stage_v1 = SimpleStage(name="test", version="1.0.0")
        context = StageContext(
            artifacts={"input_value": 5},
            cache_enabled=True,
            cache_dir=cache_dir,
        )
        result1 = stage_v1.execute(context)
        assert not result1.cache_hit
        
        # Second version - cache miss due to version change
        stage_v2 = SimpleStage(name="test", version="2.0.0")
        result2 = stage_v2.execute(context)
        assert not result2.cache_hit  # Version changed
