"""Shared coordinator contracts independent of the HTTP application."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy.exc import SQLAlchemyError

from tests.lux_depth_v5.test_pipeline import request_case  # noqa: F401 - controlled native inference fixture
from transformation_portal.orchestrator import execution_dispatch, execution_runtime
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.artifact_store.local import LocalArtifactStore
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator, _dispatch_fence
from transformation_portal.orchestrator.execution_runtime import create_managed_execution_service
from transformation_portal.orchestrator.photography_adapter import prepare_photography_dispatch
from transformation_portal.orchestrator.queue.base import JobEnqueueRequest
from transformation_portal.orchestrator.storage.base import RepositoryError
from transformation_portal.orchestrator.storage.memory import MemoryJobEventStore, MemoryJobRepository
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost
from transformation_portal.orchestrator.worker import RetryableExecutorUnavailable

pytestmark = pytest.mark.unit


@pytest.fixture
def managed(request_case, monkeypatch, tmp_path):
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    monkeypatch.setenv("TP_ALLOWED_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("TP_ALLOWED_OUTPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "private"))
    monkeypatch.setenv("TP_CANCEL_GRACE_SECONDS", "0.1")
    monkeypatch.setenv("TP_PILOT_CONTROL_PLANE_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_LUX_V5_CACHE_DIR", str(tmp_path / "cache"))
    request_case = replace(request_case, cache_dir=tmp_path / "cache" / "tenant")
    prepared = prepare_photography_dispatch(
        request_case, publisher=GenerationPublisher(artifact_store=None, record_store=None)
    )
    repository, events = MemoryJobRepository(), MemoryJobEventStore()
    artifacts = LocalArtifactStore(root_dir=tmp_path / "artifacts")
    locator = DispatchLocator("job_shared", "attempt", "dispatch", hashlib.sha256(prepared.plan_bytes).hexdigest(), "tenant")
    output = tmp_path / "requested" / ".tp-attempts" / (locator.job_id + "-" + "a" * 32)
    fence = DispatchFence(locator, "worker", 1, time.time() + 300, str(output), str(tmp_path / "requested"))
    job = execution_runtime.WorkerJob(
        id=locator.job_id, created_at=time.time(), effective_request={"pipeline": "lux-depth-v5"}
    )

    class Records:
        authority_valid = True
        commits: list[dict] = []
        finishes: list[dict] = []

        async def assert_dispatch_authority(self, observed):
            assert observed == fence
            if not self.authority_valid:
                raise DispatchAuthorityLost("dispatch was canceled")

        async def fetch_plan(self, observed):
            assert observed == locator
            return prepared.plan_bytes

        async def fetch_execution_bindings(self, observed):
            assert observed == locator
            return prepared.bindings_bytes

        async def commit_generation(self, observed, **kwargs):
            assert observed == fence
            self.commits.append(kwargs)
            await repository.update(
                job.id,
                state=kwargs["state"],
                exit_code=kwargs["exit_code"],
                artifacts=kwargs["artifacts"],
                run_summary=kwargs["run_summary"],
                finished_at=time.time(),
            )
            return json.loads(kwargs["manifest_bytes"])

        async def finish_dispatch(self, observed, **kwargs):
            assert observed == fence
            self.finishes.append(kwargs)
            await repository.update(job.id, **kwargs, finished_at=time.time())

        async def generation_cleanup_state(self, _job_id):
            return None

    records = Records()
    monkeypatch.setattr(execution_runtime, "get_job_repository", lambda: repository)
    monkeypatch.setattr(execution_runtime, "get_job_event_store", lambda: events)
    monkeypatch.setattr(execution_runtime, "get_operational_record_store", lambda: records)
    monkeypatch.setattr(execution_runtime, "get_artifact_store", lambda: artifacts)
    return create_managed_execution_service(), job, repository, events, artifacts, records, fence, prepared


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [False, True])
async def test_managed_service_runs_graph_and_publishes_independently_verified_v5_inventory(managed, monkeypatch, tamper):
    service, job, repository, _events, artifacts, records, fence, prepared = managed
    await repository.create(job)
    commands = []

    async def controlled_child(*argv, **kwargs):
        commands.append(argv)
        assert kwargs["start_new_session"] is True
        assert "TP_API_KEY" not in kwargs["env"]
        output = Path(fence.output_root)
        # Native inference alone is controlled by request_case; execute the real
        # dispatch graph, serializers and parent-side semantic publication.
        assert (
            execution_dispatch.execute_dispatch_plan(
                prepared.plan_bytes,
                execution_bindings=prepared.bindings_bytes,
                output_root=output,
                execution_workspace=Path(argv[argv.index("--execution-workspace") + 1]),
            )
            == 0
        )
        if tamper:
            (output / "input-0000/delivery.tif").write_bytes(b"forged photographic output")

        class Process:
            pid = None
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                return self.returncode

        process = Process()
        process.stdout.feed_data(b"progress=100% token=must-not-leak\n")
        process.stdout.feed_eof()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", controlled_child)
    monkeypatch.setenv("TP_API_KEY", "private")
    token = _dispatch_fence.set(fence)
    try:
        assert await service.execute(fence.locator, asyncio.Event()) == 0
    finally:
        _dispatch_fence.reset(token)
    if tamper:
        assert not records.commits
        assert len(records.finishes) == 1
        assert records.finishes[0]["state"] == "failed"
        assert (await repository.get(job.id)).state == "failed"
        return
    assert len(commands) == len(records.commits) == 1
    assert not records.finishes
    committed = await repository.get(job.id)
    assert committed.state == "succeeded"
    assert committed.progress == 100
    assert committed.logs_tail == ["progress=100% token=<redacted>"]
    assert committed.run_summary["pipeline"] == "lux_depth_v5"
    assert committed.run_summary["production_acceptance"] == "pending"
    manifest = json.loads(records.commits[0]["manifest_bytes"])
    descriptors = {item["path"]: item for item in committed.artifacts["items"]}
    assert set(descriptors) == {item["path"] for item in manifest["files"]}
    assert committed.artifacts["indexed_count"] == len(manifest["files"])
    for item in manifest["files"]:
        descriptor = descriptors[item["path"]]
        assert descriptor["size_bytes"] == item["size_bytes"]
        assert descriptor["sha256"] == item["sha256"]
        assert descriptor["fingerprint_status"] == "ok"
        assert descriptor["relative_path"] == item["path"]
    assert descriptors["input-0000/delivery.tif"]["browser_previewable"] is False
    assert "preview_url" not in descriptors["input-0000/delivery.tif"]
    delivery = next(item for item in manifest["files"] if item["path"] == "input-0000/delivery.tif")
    stream = await artifacts.open_bytes(job.id, delivery["storage_path"])
    data = b"".join([chunk async for chunk in stream])
    assert hashlib.sha256(data).hexdigest() == delivery["sha256"]
    assert not list((Path(fence.output_root).parents[2] / "private").iterdir())


@pytest.mark.asyncio
async def test_silent_managed_subprocess_is_reaped_on_cancellation(managed, monkeypatch):
    service, job, repository, _events, _artifacts, records, fence, _prepared = managed
    await repository.create(job)
    started = asyncio.Event()
    child = []
    original_spawn = asyncio.create_subprocess_exec

    async def spawn(*_argv, **kwargs):
        proc = await original_spawn(sys.executable, "-c", "import time; time.sleep(30)", **kwargs)
        child.append(proc)
        started.set()
        return proc

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    cancel = asyncio.Event()
    token = _dispatch_fence.set(fence)
    try:
        task = asyncio.create_task(service.execute(fence.locator, cancel))
        await asyncio.wait_for(started.wait(), timeout=5)
        cancel.set()
        await asyncio.wait_for(task, timeout=5)
    finally:
        _dispatch_fence.reset(token)
    assert child[0].returncode is not None
    assert not records.commits
    assert len(records.finishes) == 1
    assert records.finishes[0]["state"] == "canceled"


@pytest.mark.asyncio
async def test_managed_worker_rejects_raw_command_without_spawning(managed, monkeypatch):
    service, job, repository, *_rest = managed
    await repository.create(job)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned untrusted argv"))
    with pytest.raises(DispatchAuthorityLost, match="immutable dispatch locator"):
        await service.execute(JobEnqueueRequest(job.id, ["malicious"]), asyncio.Event())


@pytest.mark.asyncio
async def test_managed_worker_rejects_unclaimed_locator_without_workspace(managed):
    service, job, repository, _events, _artifacts, _records, fence, _prepared = managed
    await repository.create(job)
    with pytest.raises(DispatchAuthorityLost, match="matching claim"):
        await service.execute(fence.locator, asyncio.Event())
    assert not Path(fence.output_root).exists()


@pytest.mark.asyncio
async def test_repository_outage_keeps_dispatch_retryable(managed):
    service, _job, _repository, _events, _artifacts, _records, fence, _prepared = managed

    async def unavailable(_job_id):
        raise execution_runtime.ExecutionRepositoryUnavailable("database unavailable")

    service.runtime = replace(service.runtime, load_job=unavailable)
    with pytest.raises(RetryableExecutorUnavailable):
        await service.execute(fence.locator, asyncio.Event())


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["fetch_plan", "fetch_execution_bindings", "assert_dispatch_authority"])
@pytest.mark.parametrize("error_type", [OSError, RepositoryError, SQLAlchemyError])
async def test_prelaunch_authority_outage_cannot_publish_terminal_result(managed, monkeypatch, boundary, error_type):
    service, job, repository, _events, _artifacts, records, fence, _prepared = managed
    job.state = "running"  # The operational claim has already committed.
    await repository.create(job)
    failure = error_type("temporary authority read failure")

    async def unavailable(_authority):
        raise failure

    monkeypatch.setattr(records, boundary, unavailable)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned without authority"))
    token = _dispatch_fence.set(fence)
    try:
        with pytest.raises(RetryableExecutorUnavailable) as rejected:
            await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    assert rejected.value.__cause__ is failure
    assert not records.commits
    assert not records.finishes
    current = await repository.get(job.id)
    assert current.state == "running"
    assert current.finished_at is None
    assert current.exit_code is None
    assert current.error is None
    private_root = Path(fence.output_root).parents[2] / "private"
    assert not private_root.exists() or not list(private_root.iterdir())


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["fetch_plan", "fetch_execution_bindings", "assert_dispatch_authority"])
async def test_prelaunch_authority_revocation_is_not_reclassified_as_outage(managed, monkeypatch, boundary):
    service, job, repository, _events, _artifacts, records, fence, _prepared = managed
    await repository.create(job)

    async def revoked(_authority):
        raise DispatchAuthorityLost("immutable authority rejected")

    monkeypatch.setattr(records, boundary, revoked)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned after revocation"))
    token = _dispatch_fence.set(fence)
    try:
        if boundary == "assert_dispatch_authority":
            assert await service.execute(fence.locator, asyncio.Event()) == 0
        else:
            with pytest.raises(DispatchAuthorityLost, match="immutable authority rejected"):
                await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    assert not records.commits
    assert not records.finishes


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["fetch_plan", "fetch_execution_bindings"])
async def test_invalid_admitted_content_is_not_reclassified_as_outage(managed, monkeypatch, boundary):
    service, job, repository, _events, _artifacts, records, fence, _prepared = managed
    await repository.create(job)

    async def invalid_content(_authority):
        return b"{}"

    monkeypatch.setattr(records, boundary, invalid_content)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned invalid content"))
    token = _dispatch_fence.set(fence)
    try:
        with pytest.raises(ValueError):
            await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    assert not Path(fence.output_root).exists()
    assert not records.commits
    assert not records.finishes


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["fetch_execution_bindings", "assert_dispatch_authority"])
async def test_worker_retains_dispatch_lease_after_prelaunch_authority_outage(managed, monkeypatch, boundary):
    from tests.orchestrator.test_worker_runner_unit import FakeBroker
    from transformation_portal.orchestrator import worker as worker_module
    from transformation_portal.orchestrator.queue.base import JobLease
    from transformation_portal.orchestrator.worker import WorkerConfig, WorkerRunner

    service, job, repository, _events, _artifacts, records, fence, _prepared = managed
    job.state = "running"
    await repository.create(job)

    async def claim(*_args, **_kwargs):
        return fence

    async def unavailable(_authority):
        raise OSError("temporary authority connection failure")

    monkeypatch.setattr(records, "claim_dispatch", claim, raising=False)
    monkeypatch.setattr(records, boundary, unavailable)
    monkeypatch.setattr(worker_module, "get_operational_record_store", lambda: records)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned without authority"))
    broker = FakeBroker(leases=[JobLease(job.id, "worker", 0.0, fence.locator)])
    runner = WorkerRunner(
        broker=broker,
        config=WorkerConfig(worker_id="worker", lease_seconds=30.0, heartbeat_interval_seconds=100.0),
        executor=service.execute,
    )
    assert await runner.step() is True
    assert broker.released == []
    assert not records.finishes
    assert not records.commits
    assert (await repository.get(job.id)).state == "running"


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="requires POSIX process groups and fork")
async def test_managed_cancellation_reaps_resistant_inherited_native_descendant(managed, monkeypatch):
    service, job, repository, _events, _artifacts, records, fence, _prepared = managed
    await repository.create(job)
    ready = asyncio.Event()
    witness_read, witness_write = os.pipe()
    os.set_blocking(witness_read, False)
    original_spawn = asyncio.create_subprocess_exec
    original_publish = service.runtime.publish_event
    children = []
    program = """
import os, signal, time
ready_read, ready_write = os.pipe()
pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    os.close(ready_read)
    os.write(ready_write, b'1')
    os.close(ready_write)
    time.sleep(30)
else:
    os.close(ready_write)
    os.read(ready_read, 1)
    os.close(ready_read)
    print('native ready', flush=True)
    time.sleep(30)
"""

    async def spawn(*_argv, **kwargs):
        proc = await original_spawn(sys.executable, "-c", program, pass_fds=(witness_write,), **kwargs)
        children.append(proc)
        return proc

    async def publish(job_id, event, payload):
        await original_publish(job_id, event, payload)
        if event == "log" and payload["line"] == "native ready":
            ready.set()

    service.runtime = replace(service.runtime, publish_event=publish)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    cancel = asyncio.Event()
    token = _dispatch_fence.set(fence)
    task = asyncio.create_task(service.execute(fence.locator, cancel))
    try:
        await asyncio.wait_for(ready.wait(), timeout=5)
        os.close(witness_write)
        witness_write = -1
        cancel.set()
        await asyncio.wait_for(task, timeout=5)
        assert children[0].returncode is not None
        assert os.read(witness_read, 1) == b""
        assert records.finishes[0]["state"] == "canceled"
        assert not records.commits
    finally:
        cancel.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        _dispatch_fence.reset(token)
        if witness_write >= 0:
            os.close(witness_write)
        os.close(witness_read)


@pytest.mark.asyncio
async def test_changed_publisher_limits_reject_pickup_before_workspace_or_spawn(managed, monkeypatch):
    from transformation_portal.orchestrator.artifact_store import generation

    service, job, repository, _events, _artifacts, records, fence, _prepared = managed
    await repository.create(job)
    monkeypatch.setattr(generation, "MAX_GENERATION_FILES", generation.MAX_GENERATION_FILES - 1)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned with stale limits"))
    token = _dispatch_fence.set(fence)
    try:
        with pytest.raises(ValueError, match="Publisher limits changed"):
            await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    assert not Path(fence.output_root).exists()
    assert not records.commits


@pytest.mark.asyncio
async def test_direct_managed_runner_cannot_bypass_dispatch_preparation(managed, monkeypatch):
    service, job, _repository, _events, _artifacts, _records, fence, _prepared = managed
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned without prepared plan"))
    token = _dispatch_fence.set(fence)
    try:
        with pytest.raises(DispatchAuthorityLost, match="prepared dispatch context"):
            await service.run(job, ["untrusted"])
    finally:
        _dispatch_fence.reset(token)


@pytest.mark.asyncio
async def test_managed_photography_revocation_rejects_pickup_before_spawn(managed, monkeypatch):
    service, job, repository, _events, _artifacts, records, fence, _prepared = managed
    await repository.create(job)
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "0")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned after revocation"))
    token = _dispatch_fence.set(fence)
    try:
        with pytest.raises(DispatchAuthorityLost, match="disabled"):
            await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    assert not Path(fence.output_root).exists()
    assert not records.commits


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["verification", "manifest_staging"])
async def test_observed_cancellation_during_publication_cannot_commit_success(managed, monkeypatch, boundary):
    import threading

    from transformation_portal.orchestrator import photography_adapter

    service, job, repository, _events, artifacts, records, fence, prepared = managed
    await repository.create(job)
    blocked = asyncio.Event()
    resume_stage = asyncio.Event()
    resume_verification = threading.Event()
    cancellation = asyncio.Event()
    loop = asyncio.get_running_loop()

    async def completed_child(*argv, **_kwargs):
        assert (
            execution_dispatch.execute_dispatch_plan(
                prepared.plan_bytes,
                execution_bindings=prepared.bindings_bytes,
                output_root=Path(fence.output_root),
                execution_workspace=Path(argv[argv.index("--execution-workspace") + 1]),
            )
            == 0
        )

        class Process:
            pid = None
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                return 0

        process = Process()
        process.stdout.feed_eof()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", completed_child)
    if boundary == "verification":
        original_verify = photography_adapter.verify_photography_dispatch_result

        def delayed_verification(*args, **kwargs):
            loop.call_soon_threadsafe(blocked.set)
            assert resume_verification.wait(timeout=5), "parent verifier was not released"
            return original_verify(*args, **kwargs)

        monkeypatch.setattr(photography_adapter, "verify_photography_dispatch_result", delayed_verification)
    else:
        original_write = artifacts.write_immutable_file

        async def delayed_manifest_write(job_id, path, source, **kwargs):
            if path.endswith("/.manifest.json"):
                blocked.set()
                await resume_stage.wait()
            return await original_write(job_id, path, source, **kwargs)

        monkeypatch.setattr(artifacts, "write_immutable_file", delayed_manifest_write)
    token = _dispatch_fence.set(fence)
    task = asyncio.create_task(service.execute(fence.locator, cancellation))
    try:
        await asyncio.wait_for(blocked.wait(), timeout=5)
        cancellation.set()
        await asyncio.sleep(0)  # Let the bridge observe loss before releasing publication I/O.
        resume_verification.set()
        resume_stage.set()
        await asyncio.wait_for(task, timeout=5)
    finally:
        resume_verification.set()
        resume_stage.set()
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        _dispatch_fence.reset(token)
    assert not records.commits
    assert len(records.finishes) == 1
    assert records.finishes[0]["state"] == "canceled"
    assert (await repository.get(job.id)).state == "canceled"


@pytest.mark.asyncio
async def test_substituted_output_ancestor_cannot_redirect_execution(managed, monkeypatch, tmp_path):
    service, job, repository, _events, _artifacts, records, fence, _prepared = managed
    await repository.create(job)
    admitted_output = Path(fence.output_root)
    admitted_output.parent.parent.mkdir()
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    admitted_output.parent.symlink_to(redirected, target_is_directory=True)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned outside admitted root"))
    token = _dispatch_fence.set(fence)
    try:
        with pytest.raises((DispatchAuthorityLost, ValueError)):
            await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    assert not list(redirected.iterdir())
    assert not (tmp_path / "private").exists()
    assert not records.commits


@pytest.mark.asyncio
async def test_cancellation_blocks_legacy_failure_promoted_to_partial_publication(managed):
    service, job, _repository, _events, _artifacts, records, fence, _prepared = managed
    job.state, job.exit_code, job.cancel_requested = "failed", 1, True

    def refresh_summary(current):
        current.state = "partial"
        current.run_summary = {"partial": True}
        return current.run_summary

    service.runtime = replace(service.runtime, index_artifacts=lambda _job: [], refresh_summary=refresh_summary)
    with pytest.raises(DispatchAuthorityLost, match="canceled before generation commit"):
        await service._publish_fenced_result(job, fence)
    assert not records.commits


@pytest.mark.asyncio
async def test_durable_cancellation_during_plan_fetch_is_checked_before_native_spawn(managed, monkeypatch):
    service, job, repository, _events, _artifacts, records, fence, prepared = managed
    await repository.create(job)

    async def cancel_during_fetch(observed):
        assert observed == fence.locator
        records.authority_valid = False
        await repository.update(job.id, state="canceled", cancel_requested=True, finished_at=time.time())
        return prepared.bindings_bytes

    monkeypatch.setattr(records, "fetch_execution_bindings", cancel_during_fetch)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned after durable cancel"))
    token = _dispatch_fence.set(fence)
    try:
        assert await service.execute(fence.locator, asyncio.Event()) == 0
    finally:
        _dispatch_fence.reset(token)
    assert not records.commits
    assert not records.finishes
    assert (await repository.get(job.id)).state == "canceled"


@pytest.mark.parametrize("name", ["TP_DATABASE_URL", "TP_REDIS_URL", "DATABASE_URL", "REDIS_URL"])
def test_native_child_does_not_inherit_operational_credentials(monkeypatch, name):
    from transformation_portal.orchestrator.execution_process import _sanitized_child_env

    monkeypatch.setenv(name, "postgresql://operator:private@database/authority")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", "/trusted/runtime/bin/python")
    child = _sanitized_child_env()
    assert name not in child
    assert child["TRANSFORMATION_PORTAL_DA3_PYTHON"] == "/trusted/runtime/bin/python"
