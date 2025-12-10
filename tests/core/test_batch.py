"""Tests for batch processing with checkpoint/resume."""

import pytest
from pathlib import Path
import json
import time

from src.transformation_portal.core.batch.job import (
    BatchJob,
    JobItem,
    JobStatus,
    BatchProcessor
)


def test_job_item_creation():
    """Test creating job item."""
    item = JobItem(
        input_path="input.jpg",
        output_path="output.jpg"
    )
    
    assert item.status == JobStatus.PENDING
    assert item.error is None
    assert item.duration_ms is None
    assert item.attempt == 0


def test_job_item_to_dict():
    """Test converting job item to dict."""
    item = JobItem(
        input_path="input.jpg",
        output_path="output.jpg",
        status=JobStatus.COMPLETED,
        duration_ms=123.4
    )
    
    data = item.to_dict()
    
    assert data["input_path"] == "input.jpg"
    assert data["status"] == "completed"
    assert data["duration_ms"] == 123.4


def test_job_item_from_dict():
    """Test creating job item from dict."""
    data = {
        "input_path": "input.jpg",
        "output_path": "output.jpg",
        "status": "completed",
        "duration_ms": 123.4,
        "error": None,
        "attempt": 1,
        "metadata": None
    }
    
    item = JobItem.from_dict(data)
    
    assert item.input_path == "input.jpg"
    assert item.status == JobStatus.COMPLETED
    assert item.duration_ms == 123.4


def test_batch_job_creation(tmp_path):
    """Test creating batch job."""
    items = [
        JobItem("input1.jpg", "output1.jpg"),
        JobItem("input2.jpg", "output2.jpg")
    ]
    
    job = BatchJob(
        job_id="test_job",
        items=items,
        checkpoint_path=tmp_path / "job.json",
        created_at="2025-01-01T00:00:00Z"
    )
    
    assert job.job_id == "test_job"
    assert len(job.items) == 2


def test_batch_job_save_load_checkpoint(tmp_path):
    """Test saving and loading checkpoint."""
    items = [
        JobItem("input1.jpg", "output1.jpg"),
        JobItem("input2.jpg", "output2.jpg", status=JobStatus.COMPLETED)
    ]
    
    checkpoint_path = tmp_path / "job.json"
    
    job = BatchJob(
        job_id="test_job",
        items=items,
        checkpoint_path=checkpoint_path,
        created_at="2025-01-01T00:00:00Z"
    )
    
    # Save checkpoint
    job.save_checkpoint()
    assert checkpoint_path.exists()
    
    # Load checkpoint
    loaded = BatchJob.load_checkpoint(checkpoint_path)
    
    assert loaded.job_id == job.job_id
    assert len(loaded.items) == len(job.items)
    assert loaded.items[1].status == JobStatus.COMPLETED


def test_batch_job_get_pending_items():
    """Test getting pending items."""
    items = [
        JobItem("input1.jpg", "output1.jpg", status=JobStatus.PENDING),
        JobItem("input2.jpg", "output2.jpg", status=JobStatus.COMPLETED),
        JobItem("input3.jpg", "output3.jpg", status=JobStatus.PENDING)
    ]
    
    job = BatchJob(
        job_id="test",
        items=items,
        checkpoint_path=Path("test.json"),
        created_at="2025-01-01T00:00:00Z"
    )
    
    pending = job.get_pending_items()
    assert len(pending) == 2
    assert all(item.status == JobStatus.PENDING for item in pending)


def test_batch_job_get_failed_items():
    """Test getting failed items."""
    items = [
        JobItem("input1.jpg", "output1.jpg", status=JobStatus.COMPLETED),
        JobItem("input2.jpg", "output2.jpg", status=JobStatus.FAILED),
        JobItem("input3.jpg", "output3.jpg", status=JobStatus.FAILED)
    ]
    
    job = BatchJob(
        job_id="test",
        items=items,
        checkpoint_path=Path("test.json"),
        created_at="2025-01-01T00:00:00Z"
    )
    
    failed = job.get_failed_items()
    assert len(failed) == 2


def test_batch_job_mark_completed(tmp_path):
    """Test marking item as completed."""
    item = JobItem("input.jpg", "output.jpg")
    
    job = BatchJob(
        job_id="test",
        items=[item],
        checkpoint_path=tmp_path / "job.json",
        created_at="2025-01-01T00:00:00Z"
    )
    
    job.mark_completed(item, duration_ms=123.4)
    
    assert item.status == JobStatus.COMPLETED
    assert item.duration_ms == 123.4
    assert (tmp_path / "job.json").exists()


def test_batch_job_mark_failed(tmp_path):
    """Test marking item as failed."""
    item = JobItem("input.jpg", "output.jpg")
    
    job = BatchJob(
        job_id="test",
        items=[item],
        checkpoint_path=tmp_path / "job.json",
        created_at="2025-01-01T00:00:00Z"
    )
    
    job.mark_failed(item, error="Processing failed")
    
    assert item.status == JobStatus.FAILED
    assert item.error == "Processing failed"


def test_batch_job_is_complete():
    """Test checking if job is complete."""
    items = [
        JobItem("input1.jpg", "output1.jpg", status=JobStatus.COMPLETED),
        JobItem("input2.jpg", "output2.jpg", status=JobStatus.FAILED),
        JobItem("input3.jpg", "output3.jpg", status=JobStatus.SKIPPED)
    ]
    
    job = BatchJob(
        job_id="test",
        items=items,
        checkpoint_path=Path("test.json"),
        created_at="2025-01-01T00:00:00Z"
    )
    
    assert job.is_complete()
    
    # Add pending item
    items.append(JobItem("input4.jpg", "output4.jpg"))
    assert not job.is_complete()


def test_batch_job_get_stats():
    """Test getting job statistics."""
    items = [
        JobItem("input1.jpg", "output1.jpg", status=JobStatus.COMPLETED, duration_ms=100),
        JobItem("input2.jpg", "output2.jpg", status=JobStatus.COMPLETED, duration_ms=200),
        JobItem("input3.jpg", "output3.jpg", status=JobStatus.FAILED),
        JobItem("input4.jpg", "output4.jpg", status=JobStatus.PENDING)
    ]
    
    job = BatchJob(
        job_id="test",
        items=items,
        checkpoint_path=Path("test.json"),
        created_at="2025-01-01T00:00:00Z"
    )
    
    stats = job.get_stats()
    
    assert stats["total"] == 4
    assert stats["completed"] == 2
    assert stats["failed"] == 1
    assert stats["pending"] == 1
    assert stats["avg_duration_ms"] == 150.0
    assert stats["total_duration_ms"] == 300.0


def test_batch_processor_creation(tmp_path):
    """Test creating batch processor."""
    def dummy_processor(path):
        return {"processed": True}
    
    processor = BatchProcessor(
        processor_fn=dummy_processor,
        checkpoint_dir=tmp_path,
        max_retries=3
    )
    
    assert processor.max_retries == 3
    assert processor.checkpoint_dir.exists()


def test_batch_processor_process_batch(tmp_path):
    """Test processing batch."""
    # Create test input files
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    input_files = []
    for i in range(3):
        f = input_dir / f"input{i}.txt"
        f.write_text(f"test {i}")
        input_files.append(f)
    
    output_dir = tmp_path / "output"
    
    # Simple processor
    def processor(path):
        class Result:
            def save(self, output_path):
                Path(output_path).write_text(f"processed: {path.read_text()}")
        return Result()
    
    batch_processor = BatchProcessor(
        processor_fn=processor,
        checkpoint_dir=tmp_path / "checkpoints"
    )
    
    # Process batch
    job = batch_processor.process_batch(input_files, output_dir)
    
    assert job.is_complete()
    assert len(job.get_completed_items()) == 3
    
    # Verify outputs
    for i in range(3):
        output_file = output_dir / f"input{i}.txt"
        assert output_file.exists()


def test_batch_processor_resume(tmp_path):
    """Test resuming from checkpoint."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    input_files = []
    for i in range(3):
        f = input_dir / f"input{i}.txt"
        f.write_text(f"test {i}")
        input_files.append(f)
    
    output_dir = tmp_path / "output"
    checkpoint_dir = tmp_path / "checkpoints"
    
    # Processor that fails on second file
    call_count = [0]
    
    def failing_processor(path):
        call_count[0] += 1
        if call_count[0] == 2:
            raise RuntimeError("Simulated failure")
        
        class Result:
            def save(self, output_path):
                Path(output_path).write_text(f"processed: {path.read_text()}")
        return Result()
    
    batch_processor = BatchProcessor(
        processor_fn=failing_processor,
        checkpoint_dir=checkpoint_dir,
        max_retries=0  # No retries
    )
    
    # Process batch (will fail on second file)
    job = batch_processor.process_batch(input_files, output_dir, job_id="test_job")
    
    # Should have 1 completed, 1 failed, 1 pending
    assert len(job.get_completed_items()) >= 1
    assert len(job.get_failed_items()) >= 1
    
    # Resume with working processor
    def working_processor(path):
        class Result:
            def save(self, output_path):
                Path(output_path).write_text(f"processed: {path.read_text()}")
        return Result()
    
    batch_processor2 = BatchProcessor(
        processor_fn=working_processor,
        checkpoint_dir=checkpoint_dir
    )
    
    checkpoint_path = checkpoint_dir / "test_job.json"
    resumed_job = batch_processor2.process_batch(
        [],
        output_dir,
        resume_from=checkpoint_path
    )
    
    # Should complete pending items
    assert len(resumed_job.get_pending_items()) == 0


def test_batch_processor_skip_existing(tmp_path):
    """Test skipping existing outputs."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    input_file = input_dir / "input.txt"
    input_file.write_text("test")
    
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    
    # Create existing output
    output_file = output_dir / "input.txt"
    output_file.write_text("existing")
    
    def processor(path):
        raise RuntimeError("Should not be called")
    
    batch_processor = BatchProcessor(
        processor_fn=processor,
        checkpoint_dir=tmp_path / "checkpoints",
        skip_existing=True
    )
    
    job = batch_processor.process_batch([input_file], output_dir)
    
    # Should be skipped
    assert len(job.get_completed_items()) == 0
    assert job.items[0].status == JobStatus.SKIPPED


def test_batch_processor_retry_failed(tmp_path):
    """Test retrying failed items."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    
    input_file = input_dir / "input.txt"
    input_file.write_text("test")
    
    output_dir = tmp_path / "output"
    
    # Create job with failed item
    items = [
        JobItem(
            input_path=str(input_file),
            output_path=str(output_dir / "input.txt"),
            status=JobStatus.FAILED,
            error="Previous failure"
        )
    ]
    
    checkpoint_path = tmp_path / "checkpoints" / "job.json"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    job = BatchJob(
        job_id="test",
        items=items,
        checkpoint_path=checkpoint_path,
        created_at="2025-01-01T00:00:00Z"
    )
    job.save_checkpoint()
    
    # Retry with working processor
    def processor(path):
        class Result:
            def save(self, output_path):
                Path(output_path).write_text("processed")
        return Result()
    
    batch_processor = BatchProcessor(
        processor_fn=processor,
        checkpoint_dir=tmp_path / "checkpoints"
    )
    
    # Load job from checkpoint and retry with fixed processor
    loaded_job = BatchJob.load_checkpoint(checkpoint_path)
    
    # Verify initial state: 1 failed item
    assert len(loaded_job.get_failed_items()) == 1
    assert len(loaded_job.get_completed_items()) == 0
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    retried_job = batch_processor.retry_failed(loaded_job)
    
    # Should now be completed
    completed = retried_job.get_completed_items()
    failed = retried_job.get_failed_items()
    
    # Retry semantics: previously failed item should now complete with working processor
    assert len(completed) == 1, "Failed item should complete after retry with working processor"
    assert len(failed) == 0, "No items should fail with working processor"
    assert completed[0].status == JobStatus.COMPLETED
    assert completed[0].duration_ms > 0, "Completed item should have timing data"
