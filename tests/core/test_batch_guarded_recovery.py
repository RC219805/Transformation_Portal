"""Adversarial contracts for explicitly owned, opt-in batch recovery."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict
from pathlib import Path

import pytest

from transformation_portal.core.batch import (
    BatchCheckpointError,
    BatchJob,
    BatchOwnershipError,
    BatchProcessor,
    BatchRecoveryRequiredError,
    IdempotencyContract,
    JobItem,
    JobStatus,
    RecoveryIdentity,
    recovery,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.regression,
    pytest.mark.skipif(os.name != "posix", reason="Guarded batch ownership requires POSIX flock"),
]

REPO_ROOT = Path(__file__).resolve().parents[2]
IDENTITY = RecoveryIdentity(processor="test-processor-v1", inputs_sha256="a" * 64)
IDEMPOTENCY = IdempotencyContract(namespace="test-effects", version="1")


def _job(tmp_path: Path, count: int = 1) -> BatchJob:
    return BatchJob(
        name="guarded-test",
        output_dir=str(tmp_path),
        items=[
            JobItem(id=f"item-{index}", input_path=f"source-{index}.tif", output_path=f"out-{index}.tif")
            for index in range(count)
        ],
    )


def _read(checkpoint: Path) -> dict:
    return json.loads(checkpoint.read_text(encoding="utf-8"))


def _wait_for(predicate, *, process=None, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate():
        if process is not None and process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"Child exited before synchronization: {process.returncode}\n{stdout}\n{stderr}")
        if time.monotonic() >= deadline:
            pytest.fail("Timed out waiting for batch synchronization")
        time.sleep(0.01)


@pytest.fixture
def children():
    """Always reap processes, including when an ownership assertion fails."""
    processes = []

    def launch(script: str, tmp_path: Path):
        process = subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        return process

    yield launch
    for process in processes:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


CHILD_IMPORTS = """
import json
import os
import sys
import time
from pathlib import Path
from transformation_portal.core.batch import (
    BatchJob, BatchProcessor, IdempotencyContract, JobItem, RecoveryIdentity,
)
root = Path(sys.argv[1])
checkpoint = root / "batch.json"
identity = RecoveryIdentity(processor="test-processor-v1", inputs_sha256="a" * 64)
contract = IdempotencyContract(namespace="test-effects", version="1")
job = BatchJob(name="guarded-test", output_dir=str(root), items=[
    JobItem(id="item-0", input_path="source-0.tif", output_path="out-0.tif")
])
"""


def test_running_is_durable_before_detached_callback_and_metadata_is_merged(tmp_path):
    checkpoint = tmp_path / "batch.json"
    job = _job(tmp_path)
    original = job.items[0]
    original.metadata = {"nested": {"values": ["before"]}}
    observed = []

    def execute(item, context):
        document = _read(checkpoint)
        assert document["schema"] == "tp.batch.checkpoint.v1"
        assert document["job"]["items"][0]["status"] == "RUNNING"
        assert document["attempts"][item.id]["attempt"] == context.attempt == 1
        assert document["attempts"][item.id]["outcome"] == "running"
        assert item is not original
        item.metadata["nested"]["values"].append("after")
        assert original.metadata == {"nested": {"values": ["before"]}}
        observed.append(context.idempotency_key)

    result = BatchProcessor(max_workers=1).start_guarded(job, execute, checkpoint, identity=IDENTITY)

    assert result is job
    assert len(observed) == 1 and observed[0]
    assert job.items[0].status == JobStatus.COMPLETED
    assert job.items[0].metadata == {"nested": {"values": ["before", "after"]}}
    assert _read(checkpoint)["job"]["items"][0]["metadata"] == job.items[0].metadata


@pytest.mark.parametrize("failed_publication", [1, 2], ids=["initial-checkpoint", "running-before-callback"])
def test_failed_checkpoint_publication_prevents_callback(tmp_path, monkeypatch, failed_publication):
    checkpoint = tmp_path / "batch.json"
    real_replace = os.replace
    publications = []
    executed = []

    def fail_selected_write(source, destination):
        publications.append(Path(destination))
        if len(publications) == failed_publication:
            raise OSError("injected checkpoint publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_selected_write)
    with pytest.raises(BatchCheckpointError):
        BatchProcessor(max_workers=1).start_guarded(
            _job(tmp_path), lambda item, context: executed.append(item.id), checkpoint, identity=IDENTITY
        )

    assert executed == []
    if checkpoint.exists():
        assert _read(checkpoint)["job"]["items"][0]["status"] != "COMPLETED"


def test_active_process_owns_checkpoint_and_plain_save_cannot_bypass_lock(tmp_path, children):
    checkpoint = tmp_path / "batch.json"
    ready, release = tmp_path / "ready", tmp_path / "release"
    child = children(
        CHILD_IMPORTS + """
def execute(item, context):
    (root / "ready").write_text("running")
    while not (root / "release").exists():
        time.sleep(0.01)
    item.metadata["finished"] = True
BatchProcessor(max_workers=1).start_guarded(job, execute, checkpoint, identity=identity, idempotency=contract)
""",
        tmp_path,
    )
    _wait_for(ready.exists, process=child)
    lock_path = checkpoint.with_name(checkpoint.name + ".lock")
    os.utime(lock_path, (1, 1))
    before = checkpoint.read_bytes()
    executed = []
    processor = BatchProcessor(max_workers=1)
    callback = lambda item, context: executed.append(item.id)

    with pytest.raises(BatchOwnershipError):
        processor.resume_guarded(checkpoint, callback, identity=IDENTITY, idempotency=IDEMPOTENCY)
    with pytest.raises(BatchOwnershipError):
        processor.start_guarded(_job(tmp_path), callback, checkpoint, identity=IDENTITY)
    with pytest.raises(BatchOwnershipError):
        _job(tmp_path).save(checkpoint)
    assert checkpoint.read_bytes() == before
    assert executed == []

    release.write_text("finish", encoding="utf-8")
    stdout, stderr = child.communicate(timeout=15)
    assert child.returncode == 0, (stdout, stderr)
    result = processor.resume_guarded(checkpoint, callback, identity=IDENTITY, idempotency=IDEMPOTENCY)
    assert result.items[0].status == JobStatus.COMPLETED
    assert result.items[0].metadata["finished"] is True
    assert executed == []


def test_process_death_releases_ownership_but_does_not_authorize_replay(tmp_path, children):
    checkpoint, ready = tmp_path / "batch.json", tmp_path / "ready"
    child = children(
        CHILD_IMPORTS + """
def execute(item, context):
    (root / "ready").write_text("running")
    while True:
        time.sleep(0.01)
BatchProcessor(max_workers=1).start_guarded(job, execute, checkpoint, identity=identity)
""",
        tmp_path,
    )
    _wait_for(ready.exists, process=child)
    lock_path = checkpoint.with_name(checkpoint.name + ".lock")
    original_inode = lock_path.stat().st_ino
    child.kill()
    child.communicate(timeout=10)
    before = checkpoint.read_bytes()
    executed = []

    with pytest.raises(BatchRecoveryRequiredError):
        BatchProcessor().resume_guarded(checkpoint, lambda item, context: executed.append(item.id), identity=IDENTITY)
    assert checkpoint.read_bytes() == before
    assert lock_path.stat().st_ino == original_inode
    assert executed == []


def test_crash_after_side_effect_reuses_precommitted_key_for_idempotent_recovery(tmp_path, children):
    checkpoint, ledger = tmp_path / "batch.json", tmp_path / "effects.json"
    child = children(
        CHILD_IMPORTS + """
def execute(item, context):
    (root / "effects.json").write_text(json.dumps({
        "key": context.idempotency_key, "attempt": context.attempt, "effects": 1,
    }))
    os._exit(79)
BatchProcessor(max_workers=1).start_guarded(job, execute, checkpoint, identity=identity, idempotency=contract)
""",
        tmp_path,
    )
    stdout, stderr = child.communicate(timeout=15)
    assert child.returncode == 79, (stdout, stderr)
    assert _read(checkpoint)["job"]["items"][0]["status"] == "RUNNING"
    original_effect = _read(ledger)
    attempts = []

    def deduplicating_callback(item, context):
        effect = _read(ledger)
        assert effect["key"] == context.idempotency_key
        assert context.attempt == 2
        attempts.append(context.attempt)
        item.metadata["already_applied"] = True

    processor = BatchProcessor(max_workers=1)
    result = processor.resume_guarded(checkpoint, deduplicating_callback, identity=IDENTITY, idempotency=IDEMPOTENCY)
    assert result.items[0].status == JobStatus.COMPLETED
    assert result.items[0].metadata["already_applied"] is True
    assert _read(ledger) == original_effect
    assert attempts == [2]
    processor.resume_guarded(checkpoint, deduplicating_callback, identity=IDENTITY, idempotency=IDEMPOTENCY)
    assert attempts == [2]


def test_unattempted_pending_checkpoint_can_resume_without_replay_contract(tmp_path, monkeypatch):
    checkpoint = tmp_path / "batch.json"
    real_replace = os.replace
    writes = 0

    def stop_before_first_attempt(source, destination):
        nonlocal writes
        writes += 1
        if writes > 1:
            raise OSError("simulated interruption before starting work")
        return real_replace(source, destination)

    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", stop_before_first_attempt)
        with pytest.raises(BatchCheckpointError):
            BatchProcessor(max_workers=1).start_guarded(
                _job(tmp_path),
                lambda item, context: pytest.fail("callback ran before checkpoint"),
                checkpoint,
                identity=IDENTITY,
            )

    assert _read(checkpoint)["attempts"] == {}
    attempted = []
    result = BatchProcessor(max_workers=1).resume_guarded(
        checkpoint, lambda item, context: attempted.append(context.attempt), identity=IDENTITY
    )
    assert attempted == [1]
    assert result.items[0].status == JobStatus.COMPLETED


@pytest.fixture
def completed_checkpoint(tmp_path):
    checkpoint = tmp_path / "batch.json"
    BatchProcessor(max_workers=1).start_guarded(
        _job(tmp_path), lambda item, context: None, checkpoint, identity=IDENTITY, idempotency=IDEMPOTENCY
    )
    return checkpoint


@pytest.mark.parametrize(
    "corruption",
    [
        "unknown-schema",
        "unknown-field",
        "missing-identity",
        "duplicate-items",
        "bad-status",
        "attempts-list",
        "bool-attempt",
        "unknown-outcome",
        "outcome-list",
        "pending-with-attempt",
    ],
)
def test_corrupt_envelopes_fail_closed_before_callback(completed_checkpoint, corruption):
    checkpoint = completed_checkpoint
    document = _read(checkpoint)
    if corruption == "unknown-schema":
        document["schema"] = "tp.batch.checkpoint.v999"
    elif corruption == "unknown-field":
        document["unrecognized"] = True
    elif corruption == "missing-identity":
        del document["identity"]
    elif corruption == "duplicate-items":
        document["job"]["items"].append(copy.deepcopy(document["job"]["items"][0]))
    elif corruption == "bad-status":
        document["job"]["items"][0]["status"] = "SUCCEEDED"
    elif corruption == "attempts-list":
        document["attempts"] = []
    elif corruption == "bool-attempt":
        document["attempts"]["item-0"]["attempt"] = True
    elif corruption == "unknown-outcome":
        document["attempts"]["item-0"]["outcome"] = "maybe"
    elif corruption == "outcome-list":
        document["attempts"]["item-0"]["outcome"] = []
    elif corruption == "pending-with-attempt":
        document["job"]["items"][0]["status"] = "PENDING"
    checkpoint.write_text(json.dumps(document), encoding="utf-8")
    before = checkpoint.read_bytes()
    executed = []

    with pytest.raises(BatchCheckpointError):
        BatchProcessor().resume_guarded(
            checkpoint, lambda item, context: executed.append(item.id), identity=IDENTITY, idempotency=IDEMPOTENCY
        )
    assert checkpoint.read_bytes() == before
    assert executed == []


@pytest.mark.parametrize("raw", ["{not json", "null", "[]", '{"schema":"tp.batch.checkpoint.v1"}'])
def test_unreadable_or_incomplete_checkpoint_is_not_reconstructed(tmp_path, raw):
    checkpoint = tmp_path / "batch.json"
    checkpoint.write_text(raw, encoding="utf-8")
    executed = []
    with pytest.raises(BatchCheckpointError):
        BatchProcessor().resume_guarded(checkpoint, lambda item, context: executed.append(item.id), identity=IDENTITY)
    assert checkpoint.read_text(encoding="utf-8") == raw
    assert executed == []


def test_legacy_json_round_trip_does_not_grant_guarded_resume_authority(tmp_path):
    checkpoint = tmp_path / "batch.json"
    job = _job(tmp_path)
    job.items[0].metadata = {"ordinary": [1, 2]}
    job.save(checkpoint)
    loaded = BatchJob.load(checkpoint)
    assert asdict(loaded.items[0]) == asdict(job.items[0])
    assert "schema" not in _read(checkpoint)
    before = checkpoint.read_bytes()

    with pytest.raises(BatchRecoveryRequiredError):
        BatchProcessor().resume_guarded(checkpoint, lambda item, context: pytest.fail("legacy replay"), identity=IDENTITY)
    assert checkpoint.read_bytes() == before


def test_start_never_overwrites_existing_guarded_checkpoint(completed_checkpoint):
    checkpoint = completed_checkpoint
    before = checkpoint.read_bytes()
    with pytest.raises(BatchRecoveryRequiredError):
        BatchProcessor().start_guarded(
            _job(checkpoint.parent), lambda item, context: pytest.fail("duplicate start"), checkpoint, identity=IDENTITY
        )
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize(
    "identity",
    [
        RecoveryIdentity(processor="different-processor", inputs_sha256="a" * 64),
        RecoveryIdentity(processor="test-processor-v1", inputs_sha256="b" * 64),
    ],
    ids=["processor", "inputs"],
)
def test_resume_rejects_changed_execution_identity(completed_checkpoint, identity):
    checkpoint = completed_checkpoint
    before = checkpoint.read_bytes()
    with pytest.raises(BatchRecoveryRequiredError):
        BatchProcessor().resume_guarded(
            checkpoint,
            lambda item, context: pytest.fail("mismatched execution identity"),
            identity=identity,
            idempotency=IDEMPOTENCY,
        )
    assert checkpoint.read_bytes() == before


@pytest.mark.parametrize(
    "recorded,requested",
    [
        (None, None),
        (None, IDEMPOTENCY),
        (IDEMPOTENCY, None),
        (IDEMPOTENCY, IdempotencyContract(namespace="different-effects", version="1")),
        (IDEMPOTENCY, IdempotencyContract(namespace="test-effects", version="2")),
    ],
    ids=["no-contract", "retroactive-contract", "missing-contract", "changed-namespace", "changed-version"],
)
def test_failed_attempts_require_the_original_precommitted_replay_contract(tmp_path, recorded, requested):
    checkpoint = tmp_path / "batch.json"

    def fail_after_attempt(item, context):
        raise RuntimeError("side effects may already have happened")

    failed = BatchProcessor(max_workers=1).start_guarded(
        _job(tmp_path), fail_after_attempt, checkpoint, identity=IDENTITY, idempotency=recorded
    )
    assert failed.items[0].status == JobStatus.FAILED
    before = checkpoint.read_bytes()
    with pytest.raises(BatchRecoveryRequiredError):
        BatchProcessor().resume_guarded(
            checkpoint, lambda item, context: pytest.fail("unauthorized replay"), identity=IDENTITY, idempotency=requested
        )
    assert checkpoint.read_bytes() == before


def test_failed_attempt_retries_only_with_same_contract_and_key(tmp_path):
    checkpoint = tmp_path / "batch.json"
    attempts = []

    def fail(item, context):
        attempts.append(context)
        raise RuntimeError("retry requires an explicit caller contract")

    BatchProcessor(max_workers=1).start_guarded(_job(tmp_path), fail, checkpoint, identity=IDENTITY, idempotency=IDEMPOTENCY)
    result = BatchProcessor(max_workers=1).resume_guarded(
        checkpoint, lambda item, context: attempts.append(context), identity=IDENTITY, idempotency=IDEMPOTENCY
    )
    assert result.items[0].status == JobStatus.COMPLETED
    assert [context.attempt for context in attempts] == [1, 2]
    assert attempts[0].idempotency_key == attempts[1].idempotency_key


def test_checkpoint_failure_drains_an_already_started_callback_before_unlock(tmp_path, monkeypatch):
    checkpoint = tmp_path / "batch.json"
    job = _job(tmp_path, 3)
    started, release, write_failed = threading.Event(), threading.Event(), threading.Event()
    executed, errors = [], []
    real_replace = os.replace
    writes = 0

    def fail_second_claim(source, destination):
        nonlocal writes
        writes += 1
        if writes == 3:
            assert started.wait(10)
            write_failed.set()
            raise OSError("second callback claim could not be persisted")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_second_claim)

    def execute(item, context):
        executed.append(item.id)
        started.set()
        assert release.wait(10)
        item.metadata["drained"] = True

    def run():
        try:
            BatchProcessor(max_workers=2).start_guarded(job, execute, checkpoint, identity=IDENTITY)
        except BaseException as error:
            errors.append(error)

    coordinator = threading.Thread(target=run, daemon=True)
    coordinator.start()
    try:
        assert write_failed.wait(10)
        assert coordinator.is_alive()
        with pytest.raises(BatchOwnershipError):
            BatchProcessor().resume_guarded(checkpoint, lambda item, context: pytest.fail("owner escaped"), identity=IDENTITY)
    finally:
        release.set()
        coordinator.join(timeout=15)
    assert not coordinator.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], BatchCheckpointError)
    assert executed == ["item-0"]
    document = _read(checkpoint)
    assert document["job"]["items"][0]["status"] == "COMPLETED"
    assert document["job"]["items"][0]["metadata"]["drained"] is True
    assert document["job"]["items"][1]["status"] != "COMPLETED"
    assert document["job"]["items"][2]["status"] == "PENDING"


def test_interrupted_submit_and_repeated_drain_interrupt_keep_started_work_owned(tmp_path, monkeypatch):
    checkpoint = tmp_path / "batch.json"
    job = _job(tmp_path, 2)
    started, release, drain_retried = threading.Event(), threading.Event(), threading.Event()
    executed, errors = [], []
    real_submit = recovery.ThreadPoolExecutor.submit
    real_wait = recovery.wait
    drain_interrupts = 0

    def interrupted_submit(executor, *args, **kwargs):
        real_submit(executor, *args, **kwargs)
        assert started.wait(10)
        raise KeyboardInterrupt("submission interrupted after enqueue")

    def interrupted_drain(*args, **kwargs):
        nonlocal drain_interrupts
        if drain_interrupts == 0:
            drain_interrupts += 1
            raise KeyboardInterrupt("second interrupt during drain")
        drain_retried.set()
        return real_wait(*args, **kwargs)

    monkeypatch.setattr(recovery.ThreadPoolExecutor, "submit", interrupted_submit)
    monkeypatch.setattr(recovery, "wait", interrupted_drain)

    def execute(item, context):
        executed.append(item.id)
        started.set()
        assert release.wait(10)
        item.metadata["drained"] = True

    def run():
        try:
            BatchProcessor(max_workers=2).start_guarded(job, execute, checkpoint, identity=IDENTITY)
        except BaseException as error:
            errors.append(error)

    coordinator = threading.Thread(target=run, daemon=True)
    coordinator.start()
    try:
        assert drain_retried.wait(10), "The coordinator did not survive its second interruption"
        assert coordinator.is_alive()
        with pytest.raises(BatchOwnershipError):
            BatchProcessor().resume_guarded(checkpoint, lambda item, context: pytest.fail("owner escaped"), identity=IDENTITY)
    finally:
        release.set()
        coordinator.join(timeout=15)
    assert not coordinator.is_alive()
    assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt)
    assert str(errors[0]) == "submission interrupted after enqueue"
    assert executed == ["item-0"]
    document = _read(checkpoint)
    assert document["job"]["items"][0]["status"] == "COMPLETED"
    assert document["job"]["items"][0]["metadata"]["drained"] is True
    assert document["job"]["items"][1]["status"] == "PENDING"
    assert "item-1" not in document["attempts"]


@pytest.mark.parametrize("interrupted", [False, True], ids=["stop-on-error", "keyboard-interrupt"])
def test_failure_drains_started_workers_before_checkpoint_unlock(tmp_path, interrupted):
    checkpoint = tmp_path / "batch.json"
    job = _job(tmp_path, 3)
    peer_started, release_peer, failure_raised = threading.Event(), threading.Event(), threading.Event()
    executed, errors, returned = [], [], []

    def execute(item, context):
        executed.append(item.id)
        if item.id == "item-0":
            assert peer_started.wait(10), "Second bounded worker did not start"
            failure_raised.set()
            if interrupted:
                raise KeyboardInterrupt("injected interrupt")
            raise RuntimeError("injected failure")
        if item.id == "item-1":
            peer_started.set()
            assert release_peer.wait(10), "Coordinator did not release the peer worker"
            item.metadata["drained"] = True

    def run():
        try:
            returned.append(
                BatchProcessor(max_workers=2, checkpoint_interval=1, stop_on_errors=True).start_guarded(
                    job, execute, checkpoint, identity=IDENTITY
                )
            )
        except BaseException as error:  # The test must retain an intentional KeyboardInterrupt from its thread.
            errors.append(error)

    coordinator = threading.Thread(target=run, daemon=True)
    coordinator.start()
    try:
        assert failure_raised.wait(10)
        assert coordinator.is_alive()
        with pytest.raises(BatchOwnershipError):
            BatchProcessor().resume_guarded(checkpoint, lambda item, context: pytest.fail("parallel owner"), identity=IDENTITY)
    finally:
        release_peer.set()
        coordinator.join(timeout=15)
    assert not coordinator.is_alive()
    if interrupted:
        assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt)
    else:
        assert errors == []
        assert returned == [job]
    assert set(executed) == {"item-0", "item-1"}
    document = _read(checkpoint)
    assert document["job"]["items"][1]["status"] == "COMPLETED"
    assert document["job"]["items"][1]["metadata"]["drained"] is True
    assert document["job"]["items"][2]["status"] == "PENDING"
    assert "item-2" not in document["attempts"]
    with pytest.raises(BatchRecoveryRequiredError):
        BatchProcessor().resume_guarded(checkpoint, lambda item, context: pytest.fail("implicit replay"), identity=IDENTITY)
