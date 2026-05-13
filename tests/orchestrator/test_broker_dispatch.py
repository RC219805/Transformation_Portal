"""Phase 2.C - end-to-end contract for broker-mediated job dispatch.

Exercises the HTTP boundary with ``TP_ORCHESTRATOR_USE_QUEUE_BROKER=1``:
the orchestrator must enqueue via the broker, the in-process
``WorkerRunner`` pool must lease the job, and a monkey-patched
``_run_job`` body must drive the existing in-process ``Job`` to a
terminal state. Cancellation routes through ``broker.cancel`` for
both pre-lease (queued, never picked up by a worker) and in-flight
(leased by a worker, killed mid-run) paths.

The Phase 2.A queue-broker contract tests and the Phase 1.B
repository contract tests cover their respective layers in
isolation. This file is the integration seam: it proves the
orchestrator + broker + worker wiring actually does what the
hardening plan describes for §5.2 Phase 2.C.
"""

from __future__ import annotations

import asyncio
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
    """Flip ``USE_QUEUE_BROKER`` on for the duration of the test.

    Also drops every cached singleton (job repository, queue broker,
    runtime registry) so a clean broker + worker pool boots inside
    the ``TestClient`` lifespan.
    """
    monkeypatch.setattr(orchestrator_app, "USE_QUEUE_BROKER", True)
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


def test_pre_lease_cancel_drops_queue_and_publishes_terminal_event(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cancel while the job is still queued: broker.cancel returns True,
    no worker ever picks it up, and the orchestrator publishes the
    terminal cancelled-done event itself."""

    # Block the worker pool: monkey-patch ``_run_job`` to hang until
    # the test releases it. The pool size is 2, so we enqueue two
    # jobs blocking the slots, then a third that sits pre-lease in
    # the broker's queue, eligible for pre-lease cancel.
    release = asyncio.Event()

    async def fake_run_job(job, _argv) -> None:  # noqa: ANN001
        job.state = "running"
        await release.wait()
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

        # The queued job has not been leased — assert via the broker
        # before issuing the cancel.
        broker = orchestrator_app.app.state.queue_broker
        # Give the worker pool a moment to skip it (it has no free slot).
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            queued_ids = asyncio.run(broker.queued_job_ids())
            if queued_id in queued_ids:
                break
            time.sleep(0.01)
        else:
            raise AssertionError(f"job {queued_id} never reached the broker queue")

        cancel_resp = client.post(f"/v1/jobs/{queued_id}/cancel")
        assert cancel_resp.status_code in (200, 202), cancel_resp.text

        _wait_for_finished(queued_id)
        cancelled_job = orchestrator_app.JOBS[queued_id]
        assert cancelled_job.state == "canceled"
        assert cancelled_job.cancel_requested is True

        # Unblock the workers so the TestClient shutdown does not hang
        # waiting for the executor tasks.
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
