"""Opt-in Postgres app-route authority smoke for Phase 1.E.

These tests exercise the FastAPI route handlers against the real
Postgres ``JobRepository`` backend. They are intentionally excluded from
offline CI unless ``TP_TEST_POSTGRES_URL`` is provided by the operator.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncIterator

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

import app as orchestrator_app
from transformation_portal.orchestrator import reset_singletons
from transformation_portal.orchestrator.artifact_store import reset_singleton as reset_artifact_store_singleton
from transformation_portal.orchestrator.queue import reset_singleton as reset_queue_singleton
from transformation_portal.orchestrator.recovery import WORKER_LOST_REASON_CODE, sweep_orphaned_jobs

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]

_POSTGRES_URL_ENV = "TP_TEST_POSTGRES_URL"


@pytest_asyncio.fixture
async def postgres_app_client(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> AsyncIterator[tuple[AsyncClient, Any]]:
    database_url = os.getenv(_POSTGRES_URL_ENV, "").strip()
    if not database_url:
        pytest.skip(f"{_POSTGRES_URL_ENV} is not set; skipping Postgres app authority smoke")
    try:
        import transformation_portal.orchestrator.storage.postgres  # noqa: F401
    except ImportError:
        pytest.skip("sqlalchemy[asyncio] not installed; skipping postgres")

    monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
    monkeypatch.setenv("TP_DATABASE_URL", database_url)
    monkeypatch.setenv("TP_ARTIFACT_STORE", "local")
    monkeypatch.setenv("TP_ARTIFACT_LOCAL_ROOT", str(tmp_path / "artifact-store"))
    monkeypatch.setattr(orchestrator_app, "API_KEY_SECRET", "contract-secret")
    monkeypatch.setattr(orchestrator_app, "ENFORCE_JOB_API_KEY", True)
    monkeypatch.setattr(orchestrator_app, "ALLOW_SSE_QUERY_API_KEY", False)

    reset_singletons()
    reset_queue_singleton()
    reset_artifact_store_singleton()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()

    repo = orchestrator_app._job_repository()
    await repo.reset()

    transport = ASGITransport(app=orchestrator_app.app)
    try:
        async with AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers={"x-api-key": "contract-secret"},
        ) as client:
            yield client, repo
    finally:
        orchestrator_app.JOBS.clear()
        orchestrator_app.EVENT_SUBSCRIBERS.clear()
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()
        await repo.reset()
        await repo.close()
        reset_singletons()
        reset_queue_singleton()
        reset_artifact_store_singleton()
        orchestrator_app.app.state.job_repository = None
        orchestrator_app.app.state.job_repository_unavailable = False


@pytest.mark.parametrize("api_prefix", ["/v1", "/v2"])
async def test_postgres_create_list_detail_survive_runtime_cache_clear(
    postgres_app_client: tuple[AsyncClient, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mark_da3_runtime_available: None,
    api_prefix: str,
) -> None:
    client, _ = postgres_app_client
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    (input_dir / "frame.jpg").write_bytes(b"fixture")
    monkeypatch.setattr(orchestrator_app, "ALLOWED_INPUT_ROOTS", [tmp_path.resolve()])
    monkeypatch.setattr(orchestrator_app, "ALLOWED_OUTPUT_ROOTS", [tmp_path.resolve()])

    async def _noop_dispatch(*_args: object, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(orchestrator_app, "_dispatch_job", _noop_dispatch)

    create_response = await client.post(
        f"{api_prefix}/jobs",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "non_commercial_ok": True,
            },
        },
    )
    assert create_response.status_code == 200
    job_id = create_response.json()["data"]["id"]
    orchestrator_app.JOBS.clear()

    list_response = await client.get(f"{api_prefix}/jobs")
    detail_response = await client.get(f"{api_prefix}/jobs/{job_id}")

    assert list_response.status_code == 200
    assert any(job["id"] == job_id for job in list_response.json()["data"]["jobs"])
    assert detail_response.status_code == 200
    detail = detail_response.json()["data"]
    assert detail["id"] == job_id
    assert detail["state"] == "queued"


@pytest.mark.parametrize("api_prefix", ["/v1", "/v2"])
async def test_postgres_cancel_writes_repository_after_runtime_cache_clear(
    postgres_app_client: tuple[AsyncClient, Any],
    api_prefix: str,
) -> None:
    client, repo = postgres_app_client
    job = orchestrator_app.Job(
        id=f"pg_cancel_{api_prefix.strip('/')}",
        created_at=orchestrator_app._now(),
        state="queued",
        request={"pipeline": "lux-depth-v3"},
    )
    await repo.create(orchestrator_app._record_from_job(job))
    orchestrator_app.JOBS.clear()

    response = await client.post(f"{api_prefix}/jobs/{job.id}/cancel")

    assert response.status_code == 200
    assert response.json()["data"] == {"id": job.id, "state": "queued"}
    record = await repo.get(job.id)
    assert record is not None
    assert record.cancel_requested is True


@pytest.mark.parametrize("api_prefix", ["/v1", "/v2"])
async def test_postgres_artifact_fetch_and_delete_use_repository_metadata_after_runtime_cache_clear(
    postgres_app_client: tuple[AsyncClient, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    api_prefix: str,
) -> None:
    client, repo = postgres_app_client
    output_dir = tmp_path / f"artifact-{api_prefix.strip('/')}"
    artifact_path = output_dir / "renders" / "hero.txt"
    artifact_path.parent.mkdir(parents=True)
    artifact_path.write_bytes(b"postgres artifact")
    monkeypatch.setattr(orchestrator_app, "ALLOWED_OUTPUT_ROOTS", [tmp_path.resolve()])
    job = orchestrator_app.Job(
        id=f"pg_artifact_{api_prefix.strip('/')}",
        created_at=orchestrator_app._now(),
        finished_at=orchestrator_app._now(),
        done_published_at=orchestrator_app._now(),
        state="succeeded",
        progress=100,
        exit_code=0,
        request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
        effective_request={"pipeline": "lux-depth-v3", "args": {"output_dir": str(output_dir)}},
    )
    orchestrator_app._index_job_artifacts(job)
    await repo.create(orchestrator_app._record_from_job(job))
    await repo.set_artifacts(job.id, job.artifacts, job.artifact_lookup)
    orchestrator_app.JOBS.clear()

    fetch_response = await client.get(f"{api_prefix}/jobs/{job.id}/artifacts/renders/hero.txt")

    assert fetch_response.status_code == 200
    assert fetch_response.content == b"postgres artifact"
    orchestrator_app.JOBS.clear()

    delete_response = await client.delete(f"{api_prefix}/jobs/{job.id}/artifacts")

    assert delete_response.status_code == 200
    record = await repo.get(job.id)
    assert record is not None
    assert record.artifacts["lifecycle"]["deletion_status"] == "deleted"
    orchestrator_app.JOBS.clear()
    refetch_response = await client.get(f"{api_prefix}/jobs/{job.id}/artifacts/renders/hero.txt")
    assert refetch_response.status_code == 410
    assert refetch_response.json()["error"]["code"] == "ARTIFACT_DELETED"


async def test_postgres_restart_sweep_exposes_worker_lost_after_runtime_cache_clear(
    postgres_app_client: tuple[AsyncClient, Any],
) -> None:
    client, repo = postgres_app_client
    job = orchestrator_app.Job(
        id="pg_worker_lost_app",
        created_at=orchestrator_app._now(),
        started_at=orchestrator_app._now(),
        state="running",
        request={"pipeline": "lux-depth-v3"},
    )
    await repo.create(orchestrator_app._record_from_job(job))
    orchestrator_app.JOBS.clear()

    swept = await sweep_orphaned_jobs(repo)
    response = await client.get(f"/v1/jobs/{job.id}")

    assert swept == [job.id]
    assert response.status_code == 200
    body = response.json()["data"]
    assert body["state"] == "worker_lost"
    assert body["error"]["code"] == WORKER_LOST_REASON_CODE
