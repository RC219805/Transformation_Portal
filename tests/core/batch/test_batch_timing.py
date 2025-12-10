"""Tests for batch job timing instrumentation."""

import pytest
import tempfile
import time
from pathlib import Path
from src.transformation_portal.core.batch.job import (
    BatchJob, BatchProcessor, JobItem, JobStatus
)


def test_job_item_timing_s_field():
    """Test that JobItem has timing_s field."""
    item = JobItem(
        input_path="/test/input.jpg",
        output_path="/test/output.jpg",
        timing_s={"load": 0.1, "process": 0.5}
    )
    
    assert item.timing_s is not None
    assert "load" in item.timing_s
    assert "process" in item.timing_s


def test_job_item_timing_s_serialization():
    """Test that timing_s survives serialization."""
    item = JobItem(
        input_path="/test/input.jpg",
        output_path="/test/output.jpg",
        timing_s={"load": 0.1, "depth": 0.2, "export": 0.05}
    )
    
    # Serialize
    data = item.to_dict()
    assert "timing_s" in data
    assert data["timing_s"]["load"] == 0.1
    
    # Deserialize
    restored = JobItem.from_dict(data)
    assert restored.timing_s is not None
    assert restored.timing_s["load"] == 0.1
    assert restored.timing_s["depth"] == 0.2


def test_batch_job_mark_completed_with_timing():
    """Test that mark_completed accepts timing_s."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test.json"
        
        item = JobItem(
            input_path="/test/input.jpg",
            output_path="/test/output.jpg"
        )
        
        job = BatchJob(
            job_id="test_job",
            items=[item],
            checkpoint_path=checkpoint_path,
            created_at="2025-01-01T00:00:00Z"
        )
        
        timing_s = {"load": 0.1, "depth": 0.3, "export": 0.05}
        job.mark_completed(item, duration_ms=450.0, timing_s=timing_s)
        
        # Verify item updated
        assert item.status == JobStatus.COMPLETED
        assert item.duration_ms == 450.0
        assert item.timing_s == timing_s


def test_batch_job_checkpoint_persistence_with_timing():
    """Test that timing_s is persisted in checkpoints."""
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "test.json"
        
        item = JobItem(
            input_path="/test/input.jpg",
            output_path="/test/output.jpg"
        )
        
        job = BatchJob(
            job_id="test_job",
            items=[item],
            checkpoint_path=checkpoint_path,
            created_at="2025-01-01T00:00:00Z"
        )
        
        timing_s = {"load": 0.1, "depth": 0.3, "export": 0.05}
        job.mark_completed(item, duration_ms=450.0, timing_s=timing_s)
        
        # Checkpoint should exist
        assert checkpoint_path.exists()
        
        # Load checkpoint
        restored_job = BatchJob.load_checkpoint(checkpoint_path)
        
        # Verify timing_s persisted
        restored_item = restored_job.items[0]
        assert restored_item.timing_s is not None
        assert restored_item.timing_s["load"] == 0.1
        assert restored_item.timing_s["depth"] == 0.3


def test_batch_processor_extracts_timing_from_result():
    """Test that BatchProcessor extracts timing_s from processor result."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create dummy input file
        input_file = tmpdir / "input.txt"
        input_file.write_text("test")
        
        checkpoint_dir = tmpdir / "checkpoints"
        output_dir = tmpdir / "output"
        
        # Processor function that returns timing_s
        def processor_fn(input_path: Path) -> dict:
            time.sleep(0.01)
            return {
                "status": "success",
                "timing_s": {
                    "load": 0.005,
                    "process": 0.004,
                    "export": 0.001
                }
            }
        
        processor = BatchProcessor(
            processor_fn=processor_fn,
            checkpoint_dir=checkpoint_dir,
            skip_existing=False
        )
        
        job = processor.process_batch(
            input_paths=[input_file],
            output_dir=output_dir
        )
        
        # Verify timing_s propagated
        completed = job.get_completed_items()
        assert len(completed) == 1
        item = completed[0]
        assert item.timing_s is not None
        assert "load" in item.timing_s
        assert "process" in item.timing_s
        assert "export" in item.timing_s


def test_batch_processor_backward_compat_stage_times_sec():
    """Test backward compatibility with stage_times_sec."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        input_file = tmpdir / "input.txt"
        input_file.write_text("test")
        
        checkpoint_dir = tmpdir / "checkpoints"
        output_dir = tmpdir / "output"
        
        # Processor returns old format (stage_times_sec)
        def processor_fn(input_path: Path) -> dict:
            return {
                "status": "success",
                "stage_times_sec": {
                    "load": 0.1,
                    "depth": 0.2
                }
            }
        
        processor = BatchProcessor(
            processor_fn=processor_fn,
            checkpoint_dir=checkpoint_dir,
            skip_existing=False
        )
        
        job = processor.process_batch(
            input_paths=[input_file],
            output_dir=output_dir
        )
        
        # Should convert stage_times_sec to timing_s
        completed = job.get_completed_items()
        item = completed[0]
        assert item.timing_s is not None
        assert item.timing_s["load"] == 0.1
        assert item.timing_s["depth"] == 0.2


def test_batch_job_timing_optional():
    """Test that timing_s is optional (None by default)."""
    item = JobItem(
        input_path="/test/input.jpg",
        output_path="/test/output.jpg"
    )
    
    assert item.timing_s is None
    
    # Should serialize without error
    data = item.to_dict()
    assert "timing_s" in data
    assert data["timing_s"] is None


def test_batch_processor_handles_missing_timing():
    """Test that processor works when result has no timing_s."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        input_file = tmpdir / "input.txt"
        input_file.write_text("test")
        
        checkpoint_dir = tmpdir / "checkpoints"
        output_dir = tmpdir / "output"
        
        # Processor with no timing info
        def processor_fn(input_path: Path) -> dict:
            return {"status": "success"}
        
        processor = BatchProcessor(
            processor_fn=processor_fn,
            checkpoint_dir=checkpoint_dir,
            skip_existing=False
        )
        
        job = processor.process_batch(
            input_paths=[input_file],
            output_dir=output_dir
        )
        
        # Should complete without timing_s
        completed = job.get_completed_items()
        item = completed[0]
        assert item.status == JobStatus.COMPLETED
        assert item.timing_s is None
