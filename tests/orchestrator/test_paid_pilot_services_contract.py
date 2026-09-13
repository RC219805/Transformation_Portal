"""Opt-in Phase 5.A paid-pilot managed-services contract.

This test composes the durable pilot stack through FastAPI routes:
Postgres ``JobRepository`` + Redis ``QueueBroker`` + S3-compatible
``ArtifactStore``. It is skipped in offline CI unless the explicit
service URLs are present; ``make test-paid-pilot-services-contract``
performs the fail-closed environment validation before invoking it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

import app as orchestrator_app
from transformation_portal.orchestrator import reset_singletons
from transformation_portal.orchestrator.artifact_store import reset_singleton as reset_artifact_store_singleton
from transformation_portal.orchestrator.queue import reset_singleton as reset_queue_singleton
from transformation_portal.orchestrator.recovery import sweep_orphaned_jobs
from transformation_portal.orchestrator.runtime_handles import reset_runtime_registry

pytestmark = [pytest.mark.unit]

_REQUIRED_ENV = (
    "TP_DATABASE_URL",
    "TP_TEST_POSTGRES_URL",
    "TP_REDIS_URL",
    "TP_TEST_REDIS_URL",
    "TP_FRONTDOOR_REDIS_URL",
    "TP_ARTIFACT_BUCKET",
    "TP_ARTIFACT_ENDPOINT_URL",
    "TP_TEST_S3_URL",
    "TP_TEST_S3_BUCKET",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
)

_EXPECTED_SELECTORS = {
    "TP_ORCHESTRATOR_STATE_BACKEND": "postgres",
    "TP_ORCHESTRATOR_QUEUE_BACKEND": "redis",
    "TP_FRONTDOOR_SESSION_STORE": "redis",
    "TP_ARTIFACT_STORE": "s3",
}

_RUN_ENV = "TP_RUN_PAID_PILOT_SERVICES_CONTRACT"


def _missing_env() -> list[str]:
    return [name for name in _REQUIRED_ENV if not os.getenv(name, "").strip()]


def _require_paid_pilot_env() -> None:
    if os.getenv(_RUN_ENV, "").strip() != "1":
        pytest.skip(f"set {_RUN_ENV}=1 to run the paid-pilot managed-services smoke")
    missing = _missing_env()
    if missing:
        pytest.fail(
            "paid-pilot managed-services smoke was explicitly requested but "
            f"required service env is missing: {', '.join(missing)}",
            pytrace=False,
        )
    for name, expected in _EXPECTED_SELECTORS.items():
        observed = os.getenv(name, "").strip()
        if observed != expected:
            pytest.fail(
                f"{name} must be {expected!r} for the paid-pilot managed-services smoke; " f"got {observed!r}",
                pytrace=False,
            )
    try:
        import transformation_portal.orchestrator.storage.postgres  # noqa: F401
        from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore  # noqa: F401
        from transformation_portal.orchestrator.queue.redis import RedisQueueBroker  # noqa: F401
    except ImportError as exc:
        pytest.fail(
            "paid-pilot service env is set, but required Postgres/Redis/S3 dependencies " f"are unavailable: {exc}",
            pytrace=False,
        )


def test_paid_pilot_smoke_fails_when_run_flag_set_but_required_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(_RUN_ENV, "1")
    for name in _REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(pytest.fail.Exception) as exc_info:
        _require_paid_pilot_env()

    assert "explicitly requested" in str(exc_info.value)
    assert _REQUIRED_ENV[0] in str(exc_info.value)


async def _reset_external_state(
    *,
    database_url: str,
    redis_url: str,
    queue_prefix: str,
    bucket: str,
    artifact_prefix: str,
    endpoint_url: str,
    region_name: str,
) -> None:
    from alembic import command
    from alembic.config import Config
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.schema import CreateSchema, DropSchema

    from transformation_portal.orchestrator.artifact_store.s3 import S3ArtifactStore
    from transformation_portal.orchestrator.queue.locator import RedisLocatorQueueBroker
    from transformation_portal.orchestrator.storage.postgres import _SharedEngine

    assert database_url == os.environ["TP_TEST_POSTGRES_URL"], "only the explicit disposable test database may be reset"
    await _SharedEngine.dispose(database_url)
    # Component gates rebuild ORM tables. Recreate the disposable schema from
    # migrations so this vertical slice proves the real authority triggers too.
    engine = create_async_engine(database_url)
    try:
        async with engine.begin() as connection:
            await connection.execute(DropSchema("public", cascade=True, if_exists=True))
            await connection.execute(CreateSchema("public"))
    finally:
        await engine.dispose()
    repo_root = Path(__file__).resolve().parents[2]
    config = Config(str(repo_root / "migrations" / "alembic.ini"))
    config.set_main_option("script_location", str(repo_root / "migrations"))
    await asyncio.to_thread(command.upgrade, config, "head")
    broker = RedisLocatorQueueBroker(redis_url=redis_url, key_prefix=queue_prefix.rstrip(":") + ":dispatch:v1:")
    store = S3ArtifactStore(
        bucket=bucket,
        prefix=artifact_prefix,
        endpoint_url=endpoint_url,
        region_name=region_name,
    )
    try:
        await broker.reset()
        await store.reset()
    finally:
        await broker.close()
        await store.close()


@pytest.fixture
def paid_pilot_stack(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[dict[str, Any]]:
    _require_paid_pilot_env()
    run_id = uuid.uuid4().hex[:12]
    queue_prefix = f"tp:paid-pilot:test:{run_id}:queue:"
    artifact_prefix = f"tp/paid-pilot/test/{run_id}"
    database_url = os.environ["TP_TEST_POSTGRES_URL"]
    redis_url = os.environ["TP_TEST_REDIS_URL"]
    bucket = os.environ["TP_TEST_S3_BUCKET"]
    endpoint_url = os.environ["TP_TEST_S3_URL"]
    region_name = os.getenv("TP_ARTIFACT_REGION", "").strip() or "us-east-1"

    monkeypatch.setenv("TP_DATABASE_URL", database_url)
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "execution-storage"))
    monkeypatch.setenv("TP_REDIS_URL", redis_url)
    monkeypatch.setenv("TP_REDIS_KEY_PREFIX", queue_prefix)
    monkeypatch.setenv("TP_ARTIFACT_BUCKET", bucket)
    monkeypatch.setenv("TP_ARTIFACT_ENDPOINT_URL", endpoint_url)
    monkeypatch.setenv("TP_ARTIFACT_PREFIX", artifact_prefix)
    monkeypatch.setenv("TP_ARTIFACT_REGION", region_name)
    monkeypatch.setenv("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "") or region_name)
    monkeypatch.setattr(orchestrator_app, "API_KEY_SECRET", "contract-secret")
    monkeypatch.setattr(orchestrator_app, "ENFORCE_JOB_API_KEY", True)
    monkeypatch.setattr(orchestrator_app, "ALLOW_SSE_QUERY_API_KEY", False)
    monkeypatch.setattr(orchestrator_app, "WORKER_LEASE_SECONDS", 5.0)
    monkeypatch.setattr(orchestrator_app, "WORKER_HEARTBEAT_INTERVAL_SECONDS", 0.02)
    monkeypatch.setattr(orchestrator_app, "WORKER_POLL_INTERVAL_SECONDS", 0.01)
    monkeypatch.setattr(orchestrator_app, "MAX_CONCURRENT_JOBS", 1)
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", False)
    monkeypatch.setattr(orchestrator_app, "PILOT_MAX_ACTIVE_JOBS_PER_TENANT", 1)
    monkeypatch.setattr(orchestrator_app, "ORCHESTRATOR_IN_PROCESS_WORKERS_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "RECLAIM_SWEEP_INTERVAL_SECONDS", 60.0)
    monkeypatch.setattr(orchestrator_app, "RATE_LIMIT_PER_MINUTE", 0)

    reset_singletons()
    reset_queue_singleton()
    reset_artifact_store_singleton()
    reset_runtime_registry()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()

    asyncio.run(
        _reset_external_state(
            database_url=database_url,
            redis_url=redis_url,
            queue_prefix=queue_prefix,
            bucket=bucket,
            artifact_prefix=artifact_prefix,
            endpoint_url=endpoint_url,
            region_name=region_name,
        )
    )

    try:
        yield {
            "artifact_endpoint": endpoint_url,
            "bucket": bucket,
            "queue_prefix": queue_prefix,
            "artifact_prefix": artifact_prefix,
            "tmp_path": tmp_path,
        }
    finally:
        orchestrator_app.JOBS.clear()
        orchestrator_app.EVENT_SUBSCRIBERS.clear()
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()
        reset_singletons()
        reset_queue_singleton()
        reset_artifact_store_singleton()
        reset_runtime_registry()
        orchestrator_app.app.state.job_repository = None
        orchestrator_app.app.state.job_repository_unavailable = False
        asyncio.run(
            _reset_external_state(
                database_url=database_url,
                redis_url=redis_url,
                queue_prefix=queue_prefix,
                bucket=bucket,
                artifact_prefix=artifact_prefix,
                endpoint_url=endpoint_url,
                region_name=region_name,
            )
        )


def _wait_for_terminal_detail(client: TestClient, job_id: str, *, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_body: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        last_body = response.json()["data"]
        if last_body.get("state") in {"succeeded", "failed", "canceled", "worker_lost"}:
            return last_body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not reach terminal state; last body={last_body!r}")


def _wait_for_durable_artifacts(client: TestClient, job_id: str, *, timeout: float = 10.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_body: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        orchestrator_app.JOBS.clear()
        response = client.get(f"/v1/jobs/{job_id}")
        assert response.status_code == 200, response.text
        last_body = response.json()["data"]
        items = last_body.get("artifacts", {}).get("items", [])
        if last_body.get("state") == "succeeded" and items:
            return last_body
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not persist terminal artifacts; last body={last_body!r}")


def _collect_sse_frames(response: Any) -> list[tuple[str | None, str, dict[str, Any]]]:
    frames: list[tuple[str | None, str, dict[str, Any]]] = []
    current_id: str | None = None
    current_event = ""
    for line in response.iter_lines():
        if not line:
            continue
        if line.startswith("id: "):
            current_id = line.split("id: ", 1)[1].strip()
            continue
        if line.startswith("event: "):
            current_event = line.split("event: ", 1)[1].strip()
            continue
        if line.startswith("data: "):
            payload = json.loads(line.split("data: ", 1)[1])
            frames.append((current_id, current_event, payload))
            current_id = None
            if current_event == "done":
                break
    return frames


def test_paid_pilot_backend_services_compose_end_to_end(
    paid_pilot_stack: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tmp_path: Path = paid_pilot_stack["tmp_path"]
    fixture_root = tmp_path / "archive"
    shutil.copytree(Path(__file__).resolve().parents[1] / "fixtures" / "archive_small", fixture_root)
    input_dir = (fixture_root / "archive_root").resolve()
    output_dir = (tmp_path / "output").resolve()

    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(orchestrator_app, "ALLOWED_OUTPUT_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(orchestrator_app, "ALLOWED_PATH_ROOTS", [tmp_path.resolve()])

    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        create_response = client.post(
            "/v1/jobs",
            json={
                "pipeline": "archive-gate-a",
                "args": {
                    "input_dir": str(input_dir),
                    "output_dir": str(output_dir),
                    "archive_command": "fixity-scan",
                    "archive_index": str(fixture_root / "archive_index_normalized.csv.gz"),
                    "validate_schemas": False,
                },
            },
        )
        assert create_response.status_code == 200, create_response.text
        job_id = create_response.json()["data"]["id"]

        terminal = _wait_for_terminal_detail(client, job_id)
        assert terminal["state"] == "succeeded"
        assert terminal["exit_code"] == 0

        orchestrator_app.JOBS.clear()
        durable_body = _wait_for_durable_artifacts(client, job_id)
        assert durable_body["artifacts"]["items"], "terminal artifact metadata should persist in Postgres"

        orchestrator_app.JOBS.clear()
        with client.stream("GET", f"/v1/jobs/{job_id}/events", headers={"Last-Event-ID": "0"}) as stream_response:
            assert stream_response.status_code == 200
            replayed_frames = _collect_sse_frames(stream_response)
        replayed_events = [(event_id, event_name, payload.get("id")) for event_id, event_name, payload in replayed_frames]
        assert replayed_events[-1][1:] == ("done", job_id)
        event_names = [name for _id, name, _job in replayed_events]
        assert event_names[0] == "state"
        assert "log" in event_names
        assert event_names.count("done") == 1
        assert all(event_id is not None for event_id, _event_name, _payload_id in replayed_events)
        sequences = [int(event_id) for event_id, _name, _payload in replayed_events]
        assert sequences == sorted(set(sequences))

        records = orchestrator_app.get_operational_record_store()
        locator = client.portal.call(records.get_locator, job_id)
        assert locator is not None
        plan_bytes = client.portal.call(records.fetch_plan, locator)
        assert hashlib.sha256(plan_bytes).hexdigest() == locator.plan_digest
        assert json.loads(plan_bytes)["planned_backend"] == "archive"
        manifest = client.portal.call(records.committed_manifest, job_id, "default")
        assert manifest and manifest["files"]
        item = manifest["files"][0]
        artifact_path = f"/v1/jobs/{job_id}/artifacts/{item['path']}"

        artifact_response = client.get(
            artifact_path,
            follow_redirects=False,
        )
        assert artifact_response.status_code in {302, 303, 307}
        location = artifact_response.headers.get("location", "")
        assert paid_pilot_stack["artifact_endpoint"] in location
        assert paid_pilot_stack["bucket"] in location
        downloaded = httpx.get(location, timeout=10.0)
        assert downloaded.status_code == 200
        assert hashlib.sha256(downloaded.content).hexdigest() == item["sha256"]

        delete_response = client.delete(f"/v1/jobs/{job_id}/artifacts")
        assert delete_response.status_code == 200, delete_response.text

        orchestrator_app.JOBS.clear()
        deleted_response = client.get(
            artifact_path,
            follow_redirects=False,
        )
        assert deleted_response.status_code == 410
        assert deleted_response.json()["error"]["code"] == "ARTIFACT_DELETED"

        assert client.portal.call(records.committed_manifest, job_id, "default") is None
        repo = orchestrator_app._job_repository()
        abandoned = orchestrator_app.Job(
            id=f"paid_pilot_worker_lost_{uuid.uuid4().hex[:8]}",
            created_at=orchestrator_app._now(),
            state="queued",
            request={"pipeline": "archive-gate-a"},
            effective_request={"pipeline": "archive-gate-a", "tenant_id": "default", "args": {}},
        )

        async def claim_abandoned():
            attempt = await records.admit(
                orchestrator_app._record_from_job(abandoned),
                plan_bytes,
                tenant_id="default",
                output_root=str(output_dir / ".tp-attempts" / abandoned.id),
                requested_output_root=str(output_dir),
                global_limit=1,
                tenant_limit=1,
            )
            await records.claim_dispatch(attempt, "lost-test-worker", lease_seconds=0.1)
            # Restart sweeping has no authority over a durable live attempt.
            assert await sweep_orphaned_jobs(repo) == []
            await asyncio.sleep(0.15)
            assert await records.expire_dispatches() == [abandoned.id]

        client.portal.call(claim_abandoned)
        orchestrator_app.JOBS.clear()

        swept_response = client.get(f"/v1/jobs/{abandoned.id}")
        assert swept_response.status_code == 200, swept_response.text
        swept_body = swept_response.json()["data"]
        assert swept_body["state"] == "worker_lost"
        assert swept_body["error"]["code"] == "worker_lost_via_lease_reclaim"
        assert client.portal.call(records.committed_manifest, abandoned.id, "default") is None
