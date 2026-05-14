"""Phase 2.C/2.D/2.E - end-to-end contract for broker-mediated job dispatch.

Exercises the HTTP boundary against the always-on broker substrate:
the orchestrator must enqueue via the broker, the in-process
``WorkerRunner`` pool must lease the job, and a monkey-patched
``_run_job`` body must drive the existing in-process ``Job`` to a
terminal state. Cancellation routes through ``broker.cancel`` for
both pre-lease (queued, never picked up by a worker) and in-flight
(leased by a worker, killed mid-run) paths. Phase 2.D's reclaim
reconciler drives jobs to ``worker_lost`` when leases expire.

The Phase 2.A queue-broker contract tests and the Phase 1.B
repository contract tests cover their respective layers in
isolation. This file is the integration seam: it proves the
orchestrator + broker + worker wiring actually does what the
hardening plan describes for §5.2.
"""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytest
from fastapi.testclient import TestClient

import app as orchestrator_app
from transformation_portal.orchestrator import reset_singletons
from transformation_portal.orchestrator.queue import reset_singleton as reset_queue_singleton
from transformation_portal.orchestrator.runtime_handles import reset_runtime_registry

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_orchestrator_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reset orchestrator + broker singletons for the test.

    Drops every cached singleton (job repository, queue broker,
    runtime registry) so a clean broker + worker pool boots inside
    the ``TestClient`` lifespan. Broker dispatch is always-on after
    Phase 2.E; no env-var or constant flip is needed.
    """
    monkeypatch.setattr(orchestrator_app, "API_KEY_SECRET", "contract-secret")
    monkeypatch.setattr(orchestrator_app, "ENFORCE_JOB_API_KEY", True)
    # Tighten the worker pool's lease/heartbeat so the in-flight cancel
    # test does not have to wait the production default (30s heartbeat).
    monkeypatch.setattr(orchestrator_app, "WORKER_LEASE_SECONDS", 5.0)
    monkeypatch.setattr(orchestrator_app, "WORKER_HEARTBEAT_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(orchestrator_app, "WORKER_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(orchestrator_app, "MAX_CONCURRENT_JOBS", 2)
    # The DA3 runtime probe blocks the dispatch path on a real install;
    # point it at the test interpreter so admission proceeds.
    import sys

    monkeypatch.setattr(orchestrator_app, "_resolve_lux_depth_canary_runtime", lambda: Path(sys.executable))
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    reset_singletons()
    reset_queue_singleton()
    reset_runtime_registry()
    yield
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    reset_singletons()
    reset_queue_singleton()
    reset_runtime_registry()


def _build_payload(tmp_path: Path, suffix: str = "") -> Dict[str, Any]:
    input_dir = (tmp_path / f"in{suffix}").resolve()
    output_dir = (tmp_path / f"out{suffix}").resolve()
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    (input_dir / "frame.jpg").write_bytes(b"fixture-image")
    return {
        "pipeline": "lux-depth-v3",
        "args": {
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "preset": "custom",
            "quality_tier": "standard",
            "depth_backend": "da3",
            "non_commercial_ok": True,
        },
    }


@pytest.fixture
def client() -> TestClient:
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as test_client:
        yield test_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_state(job_id: str, *, expected: set[str], timeout: float = 2.0) -> str:
    """Poll the in-process ``Job`` for one of ``expected`` states.

    The worker pool drives state transitions asynchronously, so the
    HTTP handler returns 200 before the executor has finished. Use
    ``Job.state`` rather than another HTTP call so the assertion
    survives without re-entering FastAPI's request loop.
    """
    deadline = time.monotonic() + timeout
    last: Optional[str] = None
    while time.monotonic() < deadline:
        job = orchestrator_app.JOBS.get(job_id)
        if job is not None:
            last = job.state
            if job.state in expected:
                return job.state
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached one of {expected}; last seen state={last!r}")


def _wait_for_finished(job_id: str, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = orchestrator_app.JOBS.get(job_id)
        if job is not None and job.finished_at is not None:
            return
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} never finished")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_orchestrator_dispatches_jobs_through_broker(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Happy path: POST -> broker.enqueue -> worker leases -> executor runs."""
    dispatched: list[str] = []

    async def fake_run_job(job, argv) -> None:  # noqa: ANN001
        dispatched.append(job.id)
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    response = client.post("/v1/jobs", json=_build_payload(tmp_path))
    assert response.status_code == 200, response.text
    job_id = response.json()["data"]["id"]

    _wait_for_finished(job_id)
    job = orchestrator_app.JOBS[job_id]
    assert job.state == "succeeded"
    assert job.exit_code == 0
    assert dispatched == [job_id], "executor should have been invoked exactly once"


def test_worker_executor_hydrates_job_from_repository_when_runtime_cache_is_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[str] = []

    async def fake_run_job(job, argv) -> None:  # noqa: ANN001
        seen.append(job.id)
        assert argv == ["runner", "--flag"]
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.finished_at = now
        job.done_published_at = now

    async def scenario() -> int:
        job = orchestrator_app.Job(
            id="job_repo_only_worker",
            created_at=orchestrator_app._now(),
            state="queued",
            request={"pipeline": "lux-depth-v3"},
            effective_request={"pipeline": "lux-depth-v3", "args": {}},
        )
        await orchestrator_app._job_repository().create(orchestrator_app._record_from_job(job))
        orchestrator_app.JOBS.clear()
        request = orchestrator_app.JobEnqueueRequest(
            job_id=job.id,
            argv=["runner", "--flag"],
            api_version="v1",
            metadata={"pipeline": "lux-depth-v3"},
        )
        return await orchestrator_app._orchestrator_job_executor(request, asyncio.Event())

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    exit_code = asyncio.run(scenario())

    assert exit_code == 0
    assert seen == ["job_repo_only_worker"]
    assert orchestrator_app.JOBS["job_repo_only_worker"].state == "succeeded"


def test_pre_lease_cancel_drops_queue_and_publishes_terminal_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cancel while the job is still queued: broker.cancel returns True,
    no worker ever picks it up, and the orchestrator publishes the
    terminal cancelled-done event itself."""

    # Block the worker pool: monkey-patch ``_run_job`` to hang until
    # the test releases it. The pool size is 2, so we enqueue two
    # jobs blocking the slots, then a third that sits pre-lease in
    # the broker's queue, eligible for pre-lease cancel.
    #
    # ``TestClient`` runs the app on a separate thread, so the test
    # thread and the app's event-loop thread must coordinate across
    # threads. ``asyncio.Event`` is not thread-safe; use
    # ``threading.Event`` and bridge into the executor via
    # ``asyncio.to_thread`` so the event-loop thread can block on a
    # thread-safe primitive without spinning.
    release = threading.Event()

    async def fake_run_job(job, _argv) -> None:  # noqa: ANN001
        job.state = "running"
        await asyncio.to_thread(release.wait)
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        blocking_ids: list[str] = []
        for i in range(orchestrator_app.MAX_CONCURRENT_JOBS):
            response = client.post("/v1/jobs", json=_build_payload(tmp_path, f"-block{i}"))
            assert response.status_code == 200, response.text
            blocking_ids.append(response.json()["data"]["id"])
        # Wait until both blocking jobs are leased by the worker pool
        # — at that point the queue is empty and the next enqueue will
        # sit pre-lease until a worker slot frees.
        for jid in blocking_ids:
            _wait_for_state(jid, expected={"running"})

        # Bump the admission cap so we can enqueue one more without
        # tripping the 429 path; the broker still sees three jobs but
        # only two are leased.
        orchestrator_app.MAX_CONCURRENT_JOBS = 3
        response = client.post("/v1/jobs", json=_build_payload(tmp_path, "-queued"))
        assert response.status_code == 200, response.text
        queued_id = response.json()["data"]["id"]

        # Assert pre-lease state via the in-process ``Job`` (visible
        # from the test thread without crossing event loops). The
        # broker would also work, but driving its coroutines from a
        # foreign loop is unsafe — ``MemoryQueueBroker``'s
        # ``asyncio.Lock`` is bound to the app's event loop. The
        # ``Job.state`` transition is the same observable in
        # practice: state stays "queued" until the worker leases.
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            job = orchestrator_app.JOBS.get(queued_id)
            if job is not None and job.state == "queued":
                break
            time.sleep(0.01)
        else:
            raise AssertionError(f"job {queued_id} never reached pre-lease queued state")

        cancel_resp = client.post(f"/v1/jobs/{queued_id}/cancel")
        assert cancel_resp.status_code in (200, 202), cancel_resp.text

        _wait_for_finished(queued_id)
        cancelled_job = orchestrator_app.JOBS[queued_id]
        assert cancelled_job.state == "canceled"
        assert cancelled_job.cancel_requested is True

        # Unblock the workers so the TestClient shutdown does not hang
        # waiting for the executor tasks. ``threading.Event.set`` is
        # safe to call from the test thread.
        release.set()
        for jid in blocking_ids:
            _wait_for_finished(jid)


def test_inflight_cancel_routes_through_broker_to_executor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cancel while a worker holds the lease: the heartbeat reports
    LeaseStatus.cancelled, the executor's cancellation_event fires,
    Job.cancel_requested flips True, and the fake runner exits."""

    saw_cancel_request = asyncio.Event()

    async def fake_run_job(job, _argv) -> None:  # noqa: ANN001
        job.state = "running"
        # Wait for the broker-driven cancel to propagate via the
        # executor's bridge_task into Job.cancel_requested.
        deadline = time.monotonic() + 5.0
        while not job.cancel_requested and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        if job.cancel_requested:
            saw_cancel_request.set()
        job.state = "canceled"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post("/v1/jobs", json=_build_payload(tmp_path))
        assert response.status_code == 200, response.text
        job_id = response.json()["data"]["id"]
        _wait_for_state(job_id, expected={"running"})

        cancel_resp = client.post(f"/v1/jobs/{job_id}/cancel")
        assert cancel_resp.status_code in (200, 202), cancel_resp.text

        _wait_for_finished(job_id)
        assert saw_cancel_request.is_set(), "executor's cancellation bridge did not set Job.cancel_requested"
        job = orchestrator_app.JOBS[job_id]
        assert job.state == "canceled"


# ---------------------------------------------------------------------------
# Phase 2.D - worker_lost via broker lease-reclaim reconciler
# ---------------------------------------------------------------------------


def test_lease_reclaim_marks_job_worker_lost(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Worker dies holding the lease → reclaim sweep marks Job worker_lost.

    Pin a very short lease + very fast reclaim interval, monkey-patch the
    worker's heartbeat to a no-op so the lease genuinely expires while the
    executor is still hanging, then assert:

    - the broker reclaim returned the job_id,
    - ``Job.state`` transitioned to ``worker_lost``,
    - ``Job.error.retriable is True``,
    - the terminal ``done`` event was published.

    The executor's terminal-state guard means even after the broker
    re-queues the dispatch payload (Phase 2.A contract), a stale lease
    pickup is a no-op rather than a second run.
    """
    # Short lease so reclaim fires quickly inside the test.
    monkeypatch.setattr(orchestrator_app, "WORKER_LEASE_SECONDS", 0.2)
    monkeypatch.setattr(orchestrator_app, "WORKER_HEARTBEAT_INTERVAL_SECONDS", 0.05)
    monkeypatch.setattr(orchestrator_app, "WORKER_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(orchestrator_app, "RECLAIM_SWEEP_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(orchestrator_app, "MAX_CONCURRENT_JOBS", 1)

    # Suppress the heartbeat so the lease expires unobserved by the
    # worker — the moral equivalent of a worker process dying.
    from transformation_portal.orchestrator import worker as worker_module

    original_heartbeat = worker_module.WorkerRunner._heartbeat_loop

    async def _no_heartbeat(self, job_id, cancellation_event):  # noqa: ANN001
        # Park forever; the lease will expire and the reclaim sweep
        # will mark the in-process Job worker_lost.
        await cancellation_event.wait()

    monkeypatch.setattr(worker_module.WorkerRunner, "_heartbeat_loop", _no_heartbeat)

    # Executor blocks long enough for the lease to expire.
    release = threading.Event()

    async def fake_run_job(job, _argv) -> None:  # noqa: ANN001
        job.state = "running"
        await asyncio.to_thread(release.wait)
        # If the reclaim sweep already wrote terminal state, do not stomp.
        if job.finished_at is not None:
            return
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    try:
        with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
            try:
                response = client.post("/v1/jobs", json=_build_payload(tmp_path))
                assert response.status_code == 200, response.text
                job_id = response.json()["data"]["id"]
                _wait_for_state(job_id, expected={"running"})

                # Wait for the reclaim sweep to drive the Job terminal.
                _wait_for_finished(job_id, timeout=5.0)
                job = orchestrator_app.JOBS[job_id]
                assert job.state == "worker_lost"
                assert isinstance(job.error, dict)
                assert job.error.get("retriable") is True
                assert job.error.get("code") == orchestrator_app.WORKER_LOST_REASON_RECLAIMED
            finally:
                # Always unblock the executor thread so an assertion
                # failure above cannot leak the ``asyncio.to_thread``
                # worker — pytest cancellation does not stop the
                # underlying ``threading.Event.wait`` once entered.
                release.set()
    finally:
        # Restore heartbeat for subsequent tests in the same session.
        monkeypatch.setattr(worker_module.WorkerRunner, "_heartbeat_loop", original_heartbeat)


def test_executor_failure_marks_error_non_retriable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Phase 2.D — executor-level failures carry ``error.retriable=False``.

    Distinguishes the wire shape from broker-level ``worker_lost``
    (retriable=True) so operator tooling and future auto-retry
    policy can branch on the field.
    """

    async def fake_run_job(job, _argv) -> None:  # noqa: ANN001
        job.state = "failed"
        job.exit_code = 17
        job.error = orchestrator_app._error_obj(
            "RUNNER_EXIT_NONZERO",
            "runner exited with code 17",
            {"exit_code": 17},
            retriable=False,
        )
        now = orchestrator_app._now()
        job.done_published_at = now
        job.finished_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post("/v1/jobs", json=_build_payload(tmp_path))
        assert response.status_code == 200, response.text
        job_id = response.json()["data"]["id"]
        _wait_for_finished(job_id)
        job = orchestrator_app.JOBS[job_id]
        assert job.state == "failed"
        assert job.error is not None
        assert job.error.get("retriable") is False


# ---------------------------------------------------------------------------
# Phase 2.E - fail-closed contract when the broker is unavailable
# ---------------------------------------------------------------------------


def test_broker_construction_failure_returns_503_without_running_job(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Phase 2.E regression: broker unavailable → 503 ``QUEUE_UNAVAILABLE``.

    Phase 2.E removed the legacy in-band ``asyncio.create_task(_run_job(...))``
    fallback so the orchestrator now fail-closes when ``get_queue_broker()``
    raises (e.g. ``TP_ORCHESTRATOR_QUEUE_BACKEND=redis`` with
    ``TP_REDIS_URL`` unset). Pin the new contract:

    - The HTTP handler returns 503 with ``error.code=QUEUE_UNAVAILABLE``.
    - ``_run_job`` is never invoked, so no subprocess is spawned and the
      orphaned dispatch surface that the fallback exposed is gone.
    - The JOBS entry is rolled back so the admission slot is freed
      immediately (a successful retry must not 429 against a stale
      placeholder).
    - The empty output_dir that ``_materialize_dispatch_output_dir``
      created during admission is cleaned up so a misconfigured broker
      cannot accumulate empty directories per admitted-then-rejected
      request.
    """
    run_job_calls: list[str] = []

    async def fake_run_job(job, _argv) -> None:  # noqa: ANN001 - test stub
        run_job_calls.append(job.id)

    monkeypatch.setattr(orchestrator_app, "_run_job", fake_run_job)

    def _explode_get_queue_broker() -> Any:
        raise RuntimeError("simulated broker outage")

    monkeypatch.setattr(orchestrator_app, "get_queue_broker", _explode_get_queue_broker)

    payload = _build_payload(tmp_path)
    output_dir = Path(payload["args"]["output_dir"]).resolve()
    # Drop the auto-created output_dir from the helper so we can observe
    # whether ``_create_job``'s admission path leaves an empty directory
    # behind on the 503 rollback path.
    if output_dir.exists():
        # remove any contents the helper seeded
        for child in output_dir.iterdir():
            child.unlink()
        output_dir.rmdir()
    assert not output_dir.exists()

    initial_jobs_count = len(orchestrator_app.JOBS)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post("/v1/jobs", json=payload)
        rollback_record = asyncio.run(
            orchestrator_app.app.state.job_repository.get(response.json()["error"]["details"]["job_id"])
        )

    assert response.status_code == 503, response.text
    body = response.json()
    assert body["success"] is False
    assert body["error"]["code"] == "QUEUE_UNAVAILABLE"
    # No subprocess spawned: the in-band fallback is truly gone.
    assert run_job_calls == [], "broker outage must NOT silently fall back to in-band dispatch"
    # JOBS rollback: the admission slot is free, no stale placeholder.
    assert len(orchestrator_app.JOBS) == initial_jobs_count
    assert rollback_record is None
    # Empty output_dir cleaned up on rollback.
    assert not output_dir.exists(), "rollback path must reclaim the output_dir it created during admission"


def test_broker_construction_failure_preserves_preexisting_output_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Admission rollback must not remove a caller-owned output directory."""

    def _explode_get_queue_broker() -> Any:
        raise RuntimeError("simulated broker outage")

    monkeypatch.setattr(orchestrator_app, "get_queue_broker", _explode_get_queue_broker)
    payload = _build_payload(tmp_path)
    output_dir = Path(payload["args"]["output_dir"]).resolve()
    assert output_dir.is_dir()

    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post("/v1/jobs", json=payload)
        rollback_record = asyncio.run(
            orchestrator_app.app.state.job_repository.get(response.json()["error"]["details"]["job_id"])
        )

    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "QUEUE_UNAVAILABLE"
    assert rollback_record is None
    assert output_dir.is_dir()
