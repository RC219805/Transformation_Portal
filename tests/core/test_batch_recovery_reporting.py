"""Persisted RUNNING batch items stay unresolved and must not imply completion."""

from __future__ import annotations

import logging
from dataclasses import asdict

import pytest

from transformation_portal.core.batch import BatchJob, BatchProcessor, JobItem, JobStatus

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.mark.parametrize("include_pending", [False, True], ids=["running-only", "mixed-pending-running"])
def test_checkpoint_running_items_report_incomplete_without_replay(tmp_path, caplog, include_pending) -> None:
    checkpoint = tmp_path / "batch.json"
    running = JobItem(
        id="interrupted",
        input_path="source.tif",
        output_path="partial.tif",
        status=JobStatus.RUNNING,
        execution_time=1.25,
        metadata={"attempt": "original"},
    )
    items = [running]
    if include_pending:
        items.append(JobItem(id="pending", input_path="next.tif", output_path="next-output.tif"))
    BatchJob(name="resume", output_dir=str(tmp_path), items=items).save(checkpoint)
    checkpoint_before = checkpoint.read_bytes()
    loaded = BatchJob.load(checkpoint)
    running_before = asdict(loaded.get_item("interrupted"))
    executed = []

    with caplog.at_level(logging.INFO, logger="transformation_portal.core.batch.job"):
        result = BatchProcessor(max_workers=1).process(loaded, lambda item: executed.append(item.id), checkpoint)

    assert result is loaded
    assert executed == (["pending"] if include_pending else [])
    assert asdict(result.get_item("interrupted")) == running_before
    assert asdict(BatchJob.load(checkpoint).get_item("interrupted")) == running_before
    assert result.progress < 1.0
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("incomplete" in warning and "RUNNING" in warning and "not replayed" in warning for warning in warnings)
    assert "Job complete" not in caplog.text
    if include_pending:
        assert result.get_item("pending").status == JobStatus.COMPLETED
        assert BatchJob.load(checkpoint).get_item("pending").status == JobStatus.COMPLETED
    else:
        assert checkpoint.read_bytes() == checkpoint_before
