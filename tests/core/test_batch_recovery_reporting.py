"""Legacy batch execution never implicitly replays uncertain checkpoint work."""

from __future__ import annotations

import logging
from dataclasses import asdict

import pytest

from transformation_portal.core.batch import (
    BatchJob,
    BatchProcessor,
    BatchRecoveryRequiredError,
    JobItem,
    JobStatus,
    RecoveryIdentity,
)

pytestmark = [pytest.mark.unit, pytest.mark.regression]


@pytest.mark.parametrize("include_pending", [False, True], ids=["running-only", "mixed-pending-running"])
def test_checkpoint_running_items_require_recovery_without_replay(tmp_path, caplog, include_pending) -> None:
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

    with caplog.at_level(logging.INFO), pytest.raises(BatchRecoveryRequiredError):
        BatchProcessor(max_workers=1).process(loaded, lambda item: executed.append(item.id), checkpoint)

    assert executed == []
    assert asdict(loaded.get_item("interrupted")) == running_before
    assert asdict(BatchJob.load(checkpoint).get_item("interrupted")) == running_before
    assert loaded.progress < 1.0
    assert "Job complete" not in caplog.text
    if include_pending:
        assert loaded.get_item("pending").status == JobStatus.PENDING
    assert checkpoint.read_bytes() == checkpoint_before


@pytest.mark.parametrize("status", [JobStatus.RUNNING, JobStatus.FAILED])
def test_uncertain_in_memory_work_cannot_be_replayed_without_checkpoint(tmp_path, status):
    checkpoint = tmp_path / "absent.json"
    job = BatchJob(name="uncertain", output_dir=str(tmp_path), items=[JobItem("item", "in.tif", "out.tif", status=status)])
    before = asdict(job.items[0])
    with pytest.raises(BatchRecoveryRequiredError):
        BatchProcessor().process(job, lambda item: pytest.fail("uncertain legacy replay"), checkpoint)
    assert asdict(job.items[0]) == before
    assert not checkpoint.exists()


def test_existing_pending_checkpoint_requires_explicit_guarded_recovery(tmp_path):
    checkpoint = tmp_path / "batch.json"
    job = BatchJob(name="existing", output_dir=str(tmp_path), items=[JobItem("item", "in.tif", "out.tif")])
    job.save(checkpoint)
    before = checkpoint.read_bytes()
    with pytest.raises(BatchRecoveryRequiredError):
        BatchProcessor().process(job, lambda item: pytest.fail("legacy checkpoint execution"), checkpoint)
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize("entrypoint", ["process", "start_guarded"])
@pytest.mark.parametrize("move_original", [False, True], ids=["different-target", "renamed-source"])
def test_loaded_legacy_work_cannot_be_made_fresh_by_changing_checkpoint_path(tmp_path, entrypoint, move_original):
    original = tmp_path / "legacy.json"
    BatchJob(name="legacy", output_dir=str(tmp_path), items=[JobItem("pending", "in.tif", "out.tif")]).save(original)
    loaded = BatchJob.load(original)
    before = original.read_bytes()
    if move_original:
        retained = original.rename(tmp_path / "retained-legacy.json")
        target = original
    else:
        retained = original
        target = tmp_path / "new-guarded.json"
    executed = []

    with pytest.raises(BatchRecoveryRequiredError):
        if entrypoint == "process":
            BatchProcessor().process(loaded, lambda item: executed.append(item.id), target)
        else:
            BatchProcessor().start_guarded(
                loaded,
                lambda item, context: executed.append(item.id),
                target,
                identity=RecoveryIdentity(processor="test-v1", inputs_sha256="a" * 64),
            )
    assert executed == []
    assert not target.exists()
    assert retained.read_bytes() == before
    assert loaded.items[0].status == JobStatus.PENDING


@pytest.mark.parametrize("status", [JobStatus.COMPLETED, JobStatus.SKIPPED])
@pytest.mark.parametrize("new_target", [False, True], ids=["existing-target", "new-target"])
def test_terminal_only_loaded_legacy_call_is_a_harmless_no_op(tmp_path, status, new_target):
    checkpoint = tmp_path / "batch.json"
    job = BatchJob(name="done", output_dir=str(tmp_path), items=[JobItem("item", "in.tif", "out.tif", status=status)])
    job.save(checkpoint)
    before = checkpoint.read_bytes()
    loaded = BatchJob.load(checkpoint)
    target = tmp_path / "new.json" if new_target else checkpoint
    result = BatchProcessor().process(loaded, lambda item: pytest.fail("terminal item executed"), target)
    assert result is loaded
    assert checkpoint.read_bytes() == before
    if new_target:
        assert not target.exists()


def test_fresh_legacy_wrapper_executes_pending_items_once_and_keeps_plain_json(tmp_path):
    checkpoint = tmp_path / "batch.json"
    job = BatchJob(
        name="fresh",
        output_dir=str(tmp_path),
        items=[JobItem("pending", "in.tif", "out.tif"), JobItem("done", "old.tif", "old-out.tif", status=JobStatus.COMPLETED)],
    )
    executed = []

    def execute(item):
        executed.append(item.id)
        item.metadata["processed"] = True

    result = BatchProcessor(max_workers=1).process(job, execute, checkpoint)
    assert result is job
    assert executed == ["pending"]
    loaded = BatchJob.load(checkpoint)
    assert loaded.items[0].status == JobStatus.COMPLETED
    assert loaded.items[0].metadata["processed"] is True
    assert loaded.items[1].status == JobStatus.COMPLETED
