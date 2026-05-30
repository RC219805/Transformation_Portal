"""Jobs route factory for the FastAPI portal origin.

The business logic remains owned by ``app.py`` for this seam. This module only
owns the route decorators so route-family extraction can happen without changing
wire contracts or endpoint behavior.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Request
from starlette.responses import Response

from transformation_portal.api.v1 import (
    JobEnvelope,
    JobsListEnvelope,
    JobStatusEnvelope,
)


@dataclass(frozen=True)
class JobRouteHandlers:
    """Injected app-owned handlers for the jobs route family."""

    create_job_http: Callable[[Request, dict[str, Any]], Awaitable[Response]]
    create_job_v2_http: Callable[[Request, dict[str, Any]], Awaitable[Response]]
    list_jobs: Callable[[int], Awaitable[Response]]
    list_jobs_v2: Callable[[int], Awaitable[Response]]
    get_job: Callable[[str, bool], Awaitable[Response]]
    get_job_v2: Callable[[str, bool], Awaitable[Response]]
    get_job_artifact: Callable[[str, str], Awaitable[Response]]
    get_job_artifact_v2: Callable[[str, str], Awaitable[Response]]
    delete_job_artifacts: Callable[[str], Awaitable[Response]]
    delete_job_artifacts_v2: Callable[[str], Awaitable[Response]]
    cancel_job: Callable[[str], Awaitable[Response]]
    cancel_job_v2: Callable[[str], Awaitable[Response]]
    job_events: Callable[[Request, str], Awaitable[Response]]
    job_events_v2: Callable[[Request, str], Awaitable[Response]]


def create_jobs_router(handlers: JobRouteHandlers, *, job_list_limit: int) -> APIRouter:
    """Create the dual-version jobs router without owning route behavior."""
    router = APIRouter()

    @router.post("/v1/jobs", response_model=JobEnvelope)
    async def create_job_http(request: Request, payload: dict[str, Any]) -> Response:
        return await handlers.create_job_http(request, payload)

    @router.post("/v2/jobs", response_model=JobEnvelope)
    async def create_job_v2_http(request: Request, payload: dict[str, Any]) -> Response:
        return await handlers.create_job_v2_http(request, payload)

    @router.get("/v1/jobs", response_model=JobsListEnvelope)
    async def list_jobs(limit: int = job_list_limit) -> Response:
        return await handlers.list_jobs(limit)

    @router.get("/v2/jobs", response_model=JobsListEnvelope)
    async def list_jobs_v2(limit: int = job_list_limit) -> Response:
        return await handlers.list_jobs_v2(limit)

    @router.get("/v1/jobs/{job_id}", response_model=JobStatusEnvelope)
    async def get_job(job_id: str, include_logs: bool = True) -> Response:
        return await handlers.get_job(job_id, include_logs)

    @router.get("/v2/jobs/{job_id}", response_model=JobStatusEnvelope)
    async def get_job_v2(job_id: str, include_logs: bool = True) -> Response:
        return await handlers.get_job_v2(job_id, include_logs)

    @router.get("/v1/jobs/{job_id}/artifacts/{artifact_path:path}")
    async def get_job_artifact(job_id: str, artifact_path: str) -> Response:
        return await handlers.get_job_artifact(job_id, artifact_path)

    @router.get("/v2/jobs/{job_id}/artifacts/{artifact_path:path}")
    async def get_job_artifact_v2(job_id: str, artifact_path: str) -> Response:
        return await handlers.get_job_artifact_v2(job_id, artifact_path)

    @router.delete("/v1/jobs/{job_id}/artifacts", response_model=JobStatusEnvelope)
    async def delete_job_artifacts(job_id: str) -> Response:
        return await handlers.delete_job_artifacts(job_id)

    @router.delete("/v2/jobs/{job_id}/artifacts", response_model=JobStatusEnvelope)
    async def delete_job_artifacts_v2(job_id: str) -> Response:
        return await handlers.delete_job_artifacts_v2(job_id)

    @router.post("/v1/jobs/{job_id}/cancel", response_model=JobEnvelope)
    async def cancel_job(job_id: str) -> Response:
        return await handlers.cancel_job(job_id)

    @router.post("/v2/jobs/{job_id}/cancel", response_model=JobEnvelope)
    async def cancel_job_v2(job_id: str) -> Response:
        return await handlers.cancel_job_v2(job_id)

    @router.get("/v1/jobs/{job_id}/events")
    async def job_events(request: Request, job_id: str) -> Response:
        return await handlers.job_events(request, job_id)

    @router.get("/v2/jobs/{job_id}/events")
    async def job_events_v2(request: Request, job_id: str) -> Response:
        return await handlers.job_events_v2(request, job_id)

    return router
