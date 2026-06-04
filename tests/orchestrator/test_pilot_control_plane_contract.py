"""Focused contracts for opt-in managed pilot control-plane behavior."""

from __future__ import annotations

import asyncio
import importlib
from typing import Any

import pytest
from fastapi.testclient import TestClient

from transformation_portal.orchestrator import reset_singletons
from transformation_portal.orchestrator.queue import reset_singleton as reset_queue_singleton

orchestrator_app = importlib.import_module("app")

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _reset_pilot_state(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_app, "API_KEY_SECRET", "contract-secret")
    monkeypatch.setattr(orchestrator_app, "ENFORCE_JOB_API_KEY", True)
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", False)
    monkeypatch.setattr(orchestrator_app, "PILOT_ALLOWED_TENANTS", set())
    monkeypatch.setattr(orchestrator_app, "PILOT_ALLOWED_PIPELINES", {"lux-depth-v3"})
    monkeypatch.setattr(orchestrator_app, "PILOT_MAX_ACTIVE_JOBS_PER_TENANT", 0)
    monkeypatch.setattr(orchestrator_app, "_PILOT_TENANT_MANAGER", None)
    reset_singletons()
    reset_queue_singleton()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    yield
    reset_singletons()
    reset_queue_singleton()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()


async def _audit_noop(**_kwargs: Any) -> None:
    return None


def _seed_job(job: Any) -> None:
    asyncio.run(orchestrator_app._job_repository().create(orchestrator_app._record_from_job(job)))


def test_pilot_tenant_mode_requires_tenant_for_config_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)

    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post("/v1/config-preview", json={"pipeline": "lux-depth-v3", "args": {}})

    assert response.status_code == 400
    body = response.json()
    assert body["error"]["code"] == "INVALID_ARGUMENT"
    assert body["error"]["details"] == {
        "field": "x-tp-tenant-id",
        "reason": "tenant_required",
    }


def test_pilot_tenant_mode_filters_job_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "PILOT_ALLOWED_TENANTS", {"tenant_a", "tenant_b"})
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)

    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        _seed_job(
            orchestrator_app.Job(
                id="job_tenant_a",
                created_at=1.0,
                state="succeeded",
                request={"pipeline": "lux-depth-v3"},
                effective_request={"pipeline": "lux-depth-v3", "args": {}, "tenant_id": "tenant_a"},
            )
        )
        _seed_job(
            orchestrator_app.Job(
                id="job_tenant_b",
                created_at=2.0,
                state="succeeded",
                request={"pipeline": "lux-depth-v3"},
                effective_request={"pipeline": "lux-depth-v3", "args": {}, "tenant_id": "tenant_b"},
            )
        )

        list_response = client.get("/v1/jobs", headers={"x-tp-tenant-id": "tenant_a"})
        cross_tenant_response = client.get("/v1/jobs/job_tenant_b", headers={"x-tp-tenant-id": "tenant_a"})

    assert list_response.status_code == 200
    body = list_response.json()
    assert body["data"]["total"] == 1
    assert [job["id"] for job in body["data"]["jobs"]] == ["job_tenant_a"]
    assert cross_tenant_response.status_code == 404
    assert cross_tenant_response.json()["error"]["code"] == "NOT_FOUND"


def test_pilot_artifact_storage_job_id_uses_tenant_prefix() -> None:
    job = orchestrator_app.Job(
        id="job_artifacts",
        created_at=1.0,
        request={"pipeline": "lux-depth-v3"},
        effective_request={"pipeline": "lux-depth-v3", "args": {}, "tenant_id": "tenant_a"},
    )

    assert orchestrator_app._artifact_storage_job_id(job) == "tenant_a__job_artifacts"
