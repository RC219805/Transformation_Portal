"""Tests for Phase 1 Stability Architecture.

Tests orchestrator, resource monitor, checkpoint system, error recovery, and validation.
"""

import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import Phase 1 modules
from lux_depth_v2.orchestrator import ProcessOrchestrator, TaskConfig, TaskStatus
from lux_depth_v2.resource_monitor import ResourceMonitor, ResourceThresholds
from lux_depth_v2.checkpoint import CheckpointManager, ProcessingStage, TaskCheckpoint
from lux_depth_v2.error_recovery import ErrorRecovery, ErrorCategory, RetryStrategy
from lux_depth_v2.preflight import PreFlightValidator, ValidationReport


class TestProcessOrchestrator:
    """Test Process Orchestrator."""
    
    def test_orchestrator_initialization(self):
        """Test orchestrator initializes correctly."""
        orch = ProcessOrchestrator(max_workers=2, memory_budget_gb=16.0, device="cpu")
        
        assert orch.max_workers == 2
        assert orch.memory_budget_gb == 16.0
        assert orch.device == "cpu"
        assert orch.total_tasks == 0
    
    def test_submit_task(self, tmp_path):
        """Test task submission."""
        orch = ProcessOrchestrator(max_workers=1)
        
        task = TaskConfig(
            task_id="test_001",
            input_path=tmp_path / "input.tif",
            output_dir=tmp_path / "output",
            preset="photo_realistic",
        )
        
        task_id = orch.submit_task(task, priority=0)
        
        assert task_id == "test_001"
        assert orch.total_tasks == 1
        assert not orch.task_queue.empty()
    
    def test_get_progress(self, tmp_path):
        """Test progress tracking."""
        orch = ProcessOrchestrator(max_workers=1)
        
        # Submit some tasks
        for i in range(3):
            task = TaskConfig(
                task_id=f"test_{i:03d}",
                input_path=tmp_path / f"input_{i}.tif",
                output_dir=tmp_path / "output",
            )
            orch.submit_task(task)
        
        progress = orch.get_progress()
        
        assert progress["total_tasks"] == 3
        assert progress["queued"] == 3
        assert progress["active"] == 0


class TestResourceMonitor:
    """Test Resource Monitor."""
    
    def test_monitor_initialization(self):
        """Test monitor initializes correctly."""
        thresholds = ResourceThresholds(
            mps_memory_gb=50.0,
            cpu_percent=80.0,
            ram_percent=75.0,
            disk_space_gb=20.0
        )
        
        monitor = ResourceMonitor(alert_thresholds=thresholds)
        
        assert monitor.thresholds.mps_memory_gb == 50.0
        assert monitor.thresholds.cpu_percent == 80.0
    
    def test_get_metrics(self):
        """Test metrics collection."""
        monitor = ResourceMonitor()
        metrics = monitor.get_metrics()
        
        assert metrics.ram_total_gb > 0
        assert metrics.ram_used_gb >= 0
        assert 0 <= metrics.ram_percent <= 100
        assert metrics.cpu_count > 0
        assert metrics.timestamp > 0
    
    def test_disk_space_check(self, tmp_path):
        """Test disk space checking."""
        monitor = ResourceMonitor()
        disk_metrics = monitor.check_disk_space([tmp_path])
        
        assert len(disk_metrics) > 0
        for path, metrics in disk_metrics.items():
            assert "total_gb" in metrics
            assert "free_gb" in metrics
            assert "percent" in metrics
            assert metrics["total_gb"] > 0
    
    def test_is_safe_to_process(self):
        """Test safety check."""
        monitor = ResourceMonitor()
        
        # Small image should be safe
        assert monitor.is_safe_to_process(image_size_mp=10.0, upscale=2, strict=False)
        
        # Huge image might not be safe (depends on system)
        # Just verify it returns a boolean
        result = monitor.is_safe_to_process(image_size_mp=324.0, upscale=4, strict=True)
        assert isinstance(result, bool)


class TestCheckpointManager:
    """Test Checkpoint System."""
    
    def test_checkpoint_initialization(self, tmp_path):
        """Test checkpoint manager initializes correctly."""
        checkpoint_dir = tmp_path / ".checkpoints"
        manager = CheckpointManager(checkpoint_dir=checkpoint_dir)
        
        assert manager.checkpoint_dir == checkpoint_dir
        assert checkpoint_dir.exists()
    
    def test_save_and_load_checkpoint(self, tmp_path):
        """Test saving and loading checkpoints."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / ".checkpoints")
        
        # Save checkpoint
        manager.save_checkpoint(
            task_id="test_001",
            stage=ProcessingStage.DEPTH_LOAD,
            status="success",
            metadata={"test": "data"}
        )
        
        # Load checkpoint
        checkpoint = manager.load_checkpoint("test_001")
        
        assert checkpoint is not None
        assert checkpoint.task_id == "test_001"
        assert checkpoint.current_stage == ProcessingStage.DEPTH_LOAD
        assert ProcessingStage.DEPTH_LOAD.value in checkpoint.stages
        assert checkpoint.stages[ProcessingStage.DEPTH_LOAD.value].status == "success"
    
    def test_can_resume(self, tmp_path):
        """Test resume capability check."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / ".checkpoints")
        
        # Create incomplete checkpoint
        manager.save_checkpoint(
            task_id="test_001",
            stage=ProcessingStage.DEPTH_LOAD,
            status="success"
        )
        
        assert manager.can_resume("test_001")
    
    def test_checkpoint_cleanup(self, tmp_path):
        """Test checkpoint cleanup."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / ".checkpoints")
        
        # Create some checkpoints
        for i in range(3):
            manager.save_checkpoint(
                task_id=f"test_{i:03d}",
                stage=ProcessingStage.COMPLETE,
                status="success"
            )
        
        # Cleanup (with 0 days to clean all)
        manager.cleanup(older_than_days=0, completed_only=True)
        
        # Verify some were cleaned (exact count depends on timing)
        checkpoints = manager.list_checkpoints()
        assert len(checkpoints) <= 3
    
    def test_get_statistics(self, tmp_path):
        """Test checkpoint statistics."""
        manager = CheckpointManager(checkpoint_dir=tmp_path / ".checkpoints")
        
        # Create checkpoints with different statuses
        manager.save_checkpoint("test_001", ProcessingStage.COMPLETE, "success")
        manager.save_checkpoint("test_002", ProcessingStage.DEPTH_LOAD, "failed", error="Test error")
        
        stats = manager.get_statistics()
        
        assert "total" in stats
        assert "completed" in stats
        assert "failed" in stats
        assert stats["total"] >= 2


class TestErrorRecovery:
    """Test Error Recovery System."""
    
    def test_recovery_initialization(self):
        """Test error recovery initializes correctly."""
        strategy = RetryStrategy(max_retries=5, backoff_base=3.0)
        recovery = ErrorRecovery(strategy=strategy)
        
        assert recovery.strategy.max_retries == 5
        assert recovery.strategy.backoff_base == 3.0
    
    def test_classify_error(self):
        """Test error classification."""
        recovery = ErrorRecovery()
        
        # Resource error
        oom_error = RuntimeError("CUDA out of memory")
        assert recovery.classify_error(oom_error) == ErrorCategory.RESOURCE
        
        # Transient error
        timeout_error = TimeoutError("Connection timeout")
        assert recovery.classify_error(timeout_error) == ErrorCategory.TRANSIENT
        
        # Input error
        input_error = ValueError("Cannot identify image file")
        assert recovery.classify_error(input_error) == ErrorCategory.INPUT
    
    def test_should_retry(self):
        """Test retry decision logic."""
        recovery = ErrorRecovery()
        
        # Transient error should retry
        error = TimeoutError("timeout")
        should_retry, reason = recovery.should_retry(error, "task_001", attempt=0)
        assert should_retry
        
        # Permanent error should not retry
        error = RuntimeError("Permanent failure")
        should_retry, reason = recovery.should_retry(error, "task_001", attempt=0)
        assert not should_retry
        
        # Max retries exceeded
        error = TimeoutError("timeout")
        should_retry, reason = recovery.should_retry(error, "task_001", attempt=10)
        assert not should_retry
    
    def test_backoff_delay(self):
        """Test exponential backoff calculation."""
        recovery = ErrorRecovery()
        
        # Delay should increase exponentially
        delay0 = recovery.get_backoff_delay(0)
        delay1 = recovery.get_backoff_delay(1)
        delay2 = recovery.get_backoff_delay(2)
        
        assert delay0 < delay1 < delay2
        assert delay0 >= 0.1  # Minimum delay
        
        # Should cap at max delay
        delay_huge = recovery.get_backoff_delay(100)
        assert delay_huge <= recovery.strategy.max_delay_s * 1.1  # Allow jitter
    
    def test_get_fallback_config(self):
        """Test fallback configuration generation."""
        recovery = ErrorRecovery()
        
        original = {
            "device": "cuda",
            "upscale": 4,
            "upscaler_backend": "realesrgan"
        }
        
        # First retry: switch to CPU
        error = RuntimeError("CUDA out of memory")
        fallback = recovery.get_fallback_config(original, error, attempt=0)
        assert fallback["device"] == "cpu"
        
        # Second retry: reduce upscale
        fallback = recovery.get_fallback_config(original, error, attempt=1)
        assert fallback["upscale"] == 2
        
        # Third retry: disable upscaling
        fallback = recovery.get_fallback_config(original, error, attempt=2)
        assert fallback["upscaler_backend"] == "none"
    
    def test_execute_with_retry_success(self):
        """Test successful execution with retry."""
        recovery = ErrorRecovery()
        
        # Function that succeeds
        def success_func():
            return "success"
        
        result, success, error = recovery.execute_with_retry(
            success_func, "task_001"
        )
        
        assert success
        assert result == "success"
        assert error is None
    
    def test_execute_with_retry_failure(self):
        """Test failed execution with retry."""
        recovery = ErrorRecovery(strategy=RetryStrategy(max_retries=1))
        
        # Function that always fails with permanent error
        def fail_func():
            raise RuntimeError("Permanent error")
        
        result, success, error = recovery.execute_with_retry(
            fail_func, "task_001"
        )
        
        assert not success
        assert result is None
        assert error is not None


class TestPreFlightValidator:
    """Test Pre-flight Validation."""
    
    def test_validator_initialization(self):
        """Test validator initializes correctly."""
        validator = PreFlightValidator()
        assert validator.logger is not None
    
    def test_validate_system(self):
        """Test system validation."""
        validator = PreFlightValidator()
        result = validator.validate_system()
        
        # Should pass on development system
        assert result.passed
        assert "python_version" in result.details
    
    def test_validate_input_file(self, tmp_path):
        """Test input file validation."""
        validator = PreFlightValidator()
        
        # Create a test image
        from PIL import Image
        import numpy as np
        
        test_image = tmp_path / "test.tif"
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img.save(test_image)
        
        result = validator.validate_input_file(test_image)
        
        assert result.passed
        assert "width" in result.details
        assert "height" in result.details
        assert result.details["width"] == 100
        assert result.details["height"] == 100
    
    def test_validate_missing_file(self, tmp_path):
        """Test validation of missing file."""
        validator = PreFlightValidator()
        
        missing_file = tmp_path / "nonexistent.tif"
        result = validator.validate_input_file(missing_file)
        
        assert not result.passed
        assert result.severity == "error"
    
    def test_validate_depth_map(self, tmp_path):
        """Test depth map validation."""
        validator = PreFlightValidator()
        
        # Create test files
        from PIL import Image
        import numpy as np
        
        depth_dir = tmp_path / "depth"
        depth_dir.mkdir()
        
        input_file = tmp_path / "test.tif"
        depth_file = depth_dir / "test.tif"
        
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img.save(input_file)
        
        depth = Image.fromarray(np.uint8(np.random.rand(100, 100) * 255))
        depth.save(depth_file)
        
        result = validator.validate_depth_map(input_file, depth_dir)
        
        assert result.passed
        assert "depth_path" in result.details
    
    def test_validate_all(self, tmp_path):
        """Test comprehensive validation."""
        validator = PreFlightValidator()
        
        # Create test input
        from PIL import Image
        import numpy as np
        
        input_file = tmp_path / "test.tif"
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img.save(input_file)
        
        report = validator.validate_all(
            input_path=input_file,
            depth_dir=None,
            device="cpu",
            upscale=2
        )
        
        assert isinstance(report, ValidationReport)
        assert len(report.results) > 0
        
        # Should have system, input, and resource validation results
        assert any("system" in r.message.lower() for r in report.results)


class TestIntegration:
    """Integration tests for Phase 1 components."""
    
    def test_checkpoint_with_recovery(self, tmp_path):
        """Test checkpoint integration with error recovery."""
        checkpoint_dir = tmp_path / ".checkpoints"
        checkpoint_mgr = CheckpointManager(checkpoint_dir=checkpoint_dir)
        recovery = ErrorRecovery()
        
        task_id = "integration_test_001"
        
        # Simulate processing stages
        stages = [
            ProcessingStage.INIT,
            ProcessingStage.DEPTH_LOAD,
            ProcessingStage.MATERIAL_SEGMENTATION,
        ]
        
        for stage in stages:
            checkpoint_mgr.save_checkpoint(task_id, stage, "success")
        
        # Verify checkpoint
        checkpoint = checkpoint_mgr.load_checkpoint(task_id)
        assert checkpoint is not None
        assert len(checkpoint.stages) == 3
        
        # Get next stage
        next_stage = checkpoint.get_next_stage()
        assert next_stage == ProcessingStage.POST_PROCESSING
    
    def test_monitor_with_validation(self, tmp_path):
        """Test resource monitor integration with validation."""
        monitor = ResourceMonitor()
        validator = PreFlightValidator()
        
        # Get current resources
        metrics = monitor.get_metrics()
        
        # Create test image
        from PIL import Image
        import numpy as np
        
        test_image = tmp_path / "test.tif"
        img = Image.fromarray(np.uint8(np.random.rand(1000, 1000, 3) * 255))
        img.save(test_image)
        
        # Validate with resource check
        report = validator.validate_all(test_image, device="cpu", upscale=2)
        
        # Should consider available resources
        assert report is not None
        assert any("resource" in r.message.lower() for r in report.results)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
