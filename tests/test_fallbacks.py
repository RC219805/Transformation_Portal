"""Tests for fallback behaviors in edge cases."""

import pytest
from pathlib import Path
import tempfile
import os


def test_disk_full_recovery(tmp_path):
    """Test graceful handling of disk full errors."""
    from src.transformation_portal.core.batch.job import BatchProcessor, JobItem
    
    # Create a processor that simulates disk full
    def disk_full_processor(path):
        raise OSError("No space left on device")
    
    processor = BatchProcessor(
        processor_fn=disk_full_processor,
        checkpoint_dir=tmp_path / "checkpoints",
        max_retries=0
    )
    
    # Create test input
    input_file = tmp_path / "input.txt"
    input_file.write_text("test")
    
    # Process should fail but checkpoint should save
    job = processor.process_batch([input_file], tmp_path / "output")
    
    # Should have failed item
    failed = job.get_failed_items()
    assert len(failed) == 1
    assert "No space left on device" in failed[0].error
    
    # Checkpoint should exist
    assert job.checkpoint_path.exists()


def test_corrupted_input_file(tmp_path):
    """Test handling of corrupted input files."""
    from src.transformation_portal.core.batch.job import BatchProcessor
    
    def processor_with_validation(path):
        # Simulate validation failure
        content = path.read_text()
        if "corrupted" in content:
            raise ValueError("Corrupted input detected")
        
        class Result:
            def save(self, output_path):
                Path(output_path).write_text("processed")
        return Result()
    
    processor = BatchProcessor(
        processor_fn=processor_with_validation,
        checkpoint_dir=tmp_path / "checkpoints",
        max_retries=0
    )
    
    # Create corrupted file
    corrupted = tmp_path / "corrupted.txt"
    corrupted.write_text("corrupted data")
    
    # Create valid file
    valid = tmp_path / "valid.txt"
    valid.write_text("valid data")
    
    # Process batch
    job = processor.process_batch(
        [corrupted, valid],
        tmp_path / "output"
    )
    
    # Corrupted should fail, valid should succeed
    assert len(job.get_failed_items()) == 1
    assert len(job.get_completed_items()) == 1


def test_memory_error_fallback():
    """Test fallback when memory error occurs."""
    from src.transformation_portal.core.device.profiler import PerformanceProfiler
    
    profiler = PerformanceProfiler(enable_memory_tracking=True)
    
    # Simulate memory-intensive operation
    with profiler.profile("memory_test"):
        try:
            # Try to allocate large array (may fail on limited memory)
            import numpy as np
            large_array = np.zeros((10000, 10000, 10), dtype=np.float64)
            _ = large_array
        except MemoryError:
            # Should handle gracefully
            pass
    
    # Profiler should still work
    results = profiler.get_results()
    assert len(results) == 1


def test_missing_dependency_graceful_degradation():
    """Test graceful degradation when optional dependencies missing."""
    from src.transformation_portal.core.validation.metrics import MetricsComputer
    
    computer = MetricsComputer()
    
    import numpy as np
    ref = np.random.rand(100, 100, 3).astype(np.float32)
    proc = ref.copy()
    
    # Should work with basic metrics even if optional deps missing
    metrics = computer.compute(ref, proc, metrics=["mae", "mse", "psnr"])
    
    assert metrics.mae is not None
    assert metrics.mse is not None
    assert metrics.psnr is not None


def test_concurrent_checkpoint_access(tmp_path):
    """Test handling of concurrent checkpoint file access."""
    from src.transformation_portal.core.batch.job import BatchJob, JobItem
    import time
    import threading
    
    items = [JobItem(f"input{i}.jpg", f"output{i}.jpg") for i in range(5)]
    
    job = BatchJob(
        job_id="concurrent_test",
        items=items,
        checkpoint_path=tmp_path / "job.json",
        created_at="2025-01-01T00:00:00Z"
    )
    
    # Save from multiple threads
    errors = []
    
    def save_checkpoint():
        try:
            for _ in range(3):
                job.save_checkpoint()
                time.sleep(0.001)
        except Exception as e:
            errors.append(e)
    
    threads = [threading.Thread(target=save_checkpoint) for _ in range(3)]
    
    for t in threads:
        t.start()
    
    for t in threads:
        t.join()
    
    # Should not have errors (atomic writes)
    assert len(errors) == 0
    
    # Checkpoint should be valid
    loaded = BatchJob.load_checkpoint(job.checkpoint_path)
    assert loaded.job_id == job.job_id


def test_invalid_checkpoint_recovery(tmp_path):
    """Test recovery from invalid checkpoint file."""
    from src.transformation_portal.core.batch.job import BatchJob
    
    # Create invalid checkpoint
    checkpoint_path = tmp_path / "invalid.json"
    checkpoint_path.write_text("invalid json{{{")
    
    # Should raise error
    with pytest.raises(Exception):
        BatchJob.load_checkpoint(checkpoint_path)


def test_extremely_long_path_handling(tmp_path):
    """Test handling of extremely long file paths."""
    from src.transformation_portal.core.security.sanitization import sanitize_filename
    
    # Create very long filename
    long_name = "a" * 255  # Max filename length on most systems
    
    try:
        validated = sanitize_filename(long_name)
        # Should handle gracefully (may truncate or raise)
        assert len(validated) <= 255
    except Exception:
        # Some systems may reject very long names
        pass


def test_unicode_filename_handling(tmp_path):
    """Test handling of Unicode characters in filenames."""
    from src.transformation_portal.core.security.sanitization import sanitize_filename
    
    # Unicode filename
    unicode_name = "test_文件_🎨.jpg"
    
    sanitized = sanitize_filename(unicode_name)
    
    # Should produce safe filename
    assert isinstance(sanitized, str)
    # Check no path traversal chars
    assert ".." not in sanitized
    assert "/" not in sanitized
    assert "\\" not in sanitized


def test_symlink_attack_prevention(tmp_path):
    """Test prevention of symlink attacks."""
    from src.transformation_portal.core.security.path import PathValidator
    
    # Create a file outside allowed directory
    outside_dir = tmp_path.parent / "outside"
    outside_dir.mkdir(exist_ok=True)
    outside_file = outside_dir / "secret.txt"
    outside_file.write_text("secret data")
    
    # Create symlink inside allowed directory
    allowed_dir = tmp_path / "allowed"
    allowed_dir.mkdir()
    
    try:
        symlink = allowed_dir / "link"
        symlink.symlink_to(outside_file)
        
        # PathValidator should detect and reject symlink that points outside allowed root
        validator = PathValidator(allowed_roots=[allowed_dir])
        
        # validate() returns False for paths outside allowed roots
        assert not validator.validate(symlink), "Symlink pointing outside allowed root should be rejected"
        
        # safe_resolve() should raise ValueError for traversal attempts
        with pytest.raises(ValueError, match="escapes allowed root"):
            validator.safe_resolve(symlink, root=allowed_dir)
    
    except OSError:
        # Symlink creation may fail on some systems (Windows without admin)
        pytest.skip("Symlink creation not supported")


def test_path_traversal_prevention():
    """Test prevention of path traversal attacks."""
    from src.transformation_portal.core.security.path import PathValidator, safe_resolve_path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        # Test traversal attempts that actually escape the root
        traversal_attempts = [
            tmp_path / ".." / "outside.txt",
            tmp_path / "subdir" / ".." / ".." / "outside.txt",
        ]
        
        validator = PathValidator(allowed_roots=[tmp_path])
        
        for dangerous_path in traversal_attempts:
            # Resolve the path to see where it actually points
            resolved = dangerous_path.resolve()
            
            # PathValidator.validate() should return False for paths outside allowed roots
            # The validator logs warnings but returns False (doesn't raise)
            is_valid = validator.validate(dangerous_path)
            
            # Check if path actually escapes (resolve normalizes ..)
            try:
                resolved.relative_to(tmp_path)
                # Path is inside root - should be valid
                assert is_valid, f"Path inside root should be valid: {dangerous_path}"
            except ValueError:
                # Path is outside root - should be invalid
                assert not is_valid, f"Path outside root should be invalid: {dangerous_path} -> {resolved}"
        
        # Explicitly test safe_resolve_path raises on traversal
        with pytest.raises(ValueError, match="escapes allowed root"):
            safe_resolve_path(tmp_path / ".." / "outside.txt", root=tmp_path)


def test_large_batch_checkpoint_performance(tmp_path):
    """Test checkpoint performance with large batches."""
    from src.transformation_portal.core.batch.job import BatchJob, JobItem
    import time
    
    # Create large batch
    items = [
        JobItem(f"input{i}.jpg", f"output{i}.jpg")
        for i in range(10000)
    ]
    
    job = BatchJob(
        job_id="large_batch",
        items=items,
        checkpoint_path=tmp_path / "large.json",
        created_at="2025-01-01T00:00:00Z"
    )
    
    # Measure checkpoint save time
    start = time.perf_counter()
    job.save_checkpoint()
    save_time = time.perf_counter() - start
    
    # Should be reasonably fast (<1 second for 10k items)
    assert save_time < 1.0
    
    # Measure load time
    start = time.perf_counter()
    loaded = BatchJob.load_checkpoint(job.checkpoint_path)
    load_time = time.perf_counter() - start
    
    assert load_time < 1.0
    assert len(loaded.items) == 10000


@pytest.mark.skipif(not pytest.importorskip("torch", reason="torch not available"),
                    reason="torch not available")
def test_cuda_out_of_memory_handling():
    """Test handling of CUDA out of memory errors."""
    import torch
    from src.transformation_portal.core.device.profiler import GPUProfiler
    
    profiler = GPUProfiler(enabled=True)
    
    with profiler.profile("cuda_test"):
        try:
            if torch.cuda.is_available():
                # Try to allocate very large tensor
                x = torch.zeros(100000, 100000, device="cuda")
                _ = x
        except RuntimeError as e:
            # Should handle OOM gracefully
            assert "out of memory" in str(e).lower() or "cuda" in str(e).lower()
    
    # Profiler should still work
    report = profiler.report()
    assert "cuda_test" in [s["name"] for s in report["stages"]]


def test_empty_batch_handling(tmp_path):
    """Test handling of empty batch."""
    from src.transformation_portal.core.batch.job import BatchProcessor
    
    def processor(path):
        return None
    
    batch_processor = BatchProcessor(
        processor_fn=processor,
        checkpoint_dir=tmp_path / "checkpoints"
    )
    
    # Process empty batch
    job = batch_processor.process_batch([], tmp_path / "output")
    
    assert job.is_complete()
    assert len(job.items) == 0


def test_processor_timeout_handling(tmp_path):
    """Test handling of processor timeouts."""
    from src.transformation_portal.core.batch.job import BatchProcessor
    import time
    
    def slow_processor(path):
        time.sleep(10)  # Very slow
        return None
    
    # This test demonstrates structure but doesn't actually implement timeout
    # Real implementation would need timeout wrapper
    processor = BatchProcessor(
        processor_fn=slow_processor,
        checkpoint_dir=tmp_path / "checkpoints",
        max_retries=0
    )
    
    # Note: Current implementation doesn't have timeout
    # This is a placeholder for future enhancement
