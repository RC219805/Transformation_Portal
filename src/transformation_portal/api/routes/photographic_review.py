"""Additive authenticated photographic review routes; existing routes are stable."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any, Awaitable, Callable, Literal

from fastapi import APIRouter, Query, Request
from starlette.responses import Response

from transformation_portal.api.v1.envelopes import ApiEnvelope
from transformation_portal.api.v1.photographic_review import (
    DeliveryHistoryPage,
    DeliveryRequest,
    LightingGroupsRequest,
    ReviewDecisionRequest,
    RevisionRequest,
)
from transformation_portal.orchestrator.photographic_admission import ReviewOverload
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewInputError, ReviewRuntime
from transformation_portal.orchestrator.photographic_review_service import PhotographicReviewService, envelope
from transformation_portal.orchestrator.photographic_review_store import ReviewConflict

LOGGER = logging.getLogger(__name__)
HashQuery = Annotated[str, Query(pattern=r"^[0-9a-f]{64}$")]
OptionalHashQuery = Annotated[str | None, Query(pattern=r"^[0-9a-f]{64}$")]
PathQuery = Annotated[str, Query(min_length=1, max_length=1024)]


def create_photographic_review_router(runtime: ReviewRuntime) -> APIRouter:
    service = PhotographicReviewService()

    @asynccontextmanager
    async def lifespan(_app: Any) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await service.close()

    router = APIRouter(tags=["Photographic review"], lifespan=lifespan)
    versioned = APIRouter(prefix="/jobs/{job_id}/photographic-review")

    async def invoke(
        request: Request,
        job_id: str,
        method: Callable[..., Awaitable[dict[str, Any] | Response]],
        *args: Any,
        lightweight_authorization: bool = False,
        **kwargs: Any,
    ) -> Response:
        try:
            authorize = runtime.authorize
            if lightweight_authorization:
                authorize = runtime.authorize_history or authorize
            elif service.admission.config.enabled:
                if runtime.authorize_admission is None:
                    raise RuntimeError("Photographic admission requires a lightweight authenticated adapter")
                authorize = runtime.authorize_admission
            if service.admission.config.enabled and not lightweight_authorization:
                async with service.admission.authorization():
                    context = await authorize(request, job_id)
            else:
                context = await authorize(request, job_id)
            if isinstance(context, Response):
                return context
            result = (
                await method(context, *args, **kwargs)
                if lightweight_authorization
                else await service.execute(context, method, *args, **kwargs)
            )
            return result if isinstance(result, Response) else envelope(result)
        except ReviewOverload as exc:
            response = runtime.error_response(
                exc.status,
                code="RESOURCE_EXHAUSTED",
                message=str(exc),
                details={"job_id": job_id, "reason": exc.reason, "scope": "process"},
                retriable=exc.retriable,
            )
            if exc.retriable:
                response.headers["Retry-After"] = "1"
            return response
        except ReviewConflict as exc:
            return runtime.error_response(409, code="REVIEW_CONFLICT", message=str(exc), details={"job_id": job_id})
        except ReviewAccessError as exc:
            return runtime.error_response(exc.status, code=exc.code, message=str(exc), details={"job_id": job_id})
        except ReviewInputError as exc:
            return runtime.error_response(400, code="INVALID_ARGUMENT", message=str(exc)[:500], details={"job_id": job_id})
        except ValueError:
            LOGGER.info("Invalid photographic input for job %s", job_id, exc_info=True)
            return runtime.error_response(
                400,
                code="INVALID_ARGUMENT",
                message="Photographic input is invalid; check its encoding, geometry, recipe, and current review state",
                details={"job_id": job_id},
            )
        except Exception:
            LOGGER.exception("Photographic review operation unavailable for job %s", job_id)
            return runtime.error_response(
                503,
                code="SERVICE_UNAVAILABLE",
                message="photographic review is temporarily unavailable",
                details={"job_id": job_id},
                retriable=True,
            )

    @versioned.get("", response_model=ApiEnvelope[dict])
    async def catalogue(request: Request, job_id: str) -> Response:
        return await invoke(request, job_id, service.catalogue)

    @versioned.get("/render")
    async def render(
        request: Request,
        job_id: str,
        artifact_path: PathQuery,
        sha256: HashQuery,
        max_edge: int = Query(default=1600, ge=1, le=1600),
        source_identity_sha256: OptionalHashQuery = None,
        view: Literal["recorded", "clip_srgb"] = "clip_srgb",
    ) -> Response:
        return await invoke(
            request,
            job_id,
            service.render,
            artifact_path,
            sha256,
            max_edge=max_edge,
            source_identity_sha256=source_identity_sha256,
            view=view,
        )

    @versioned.get("/native-crop")
    async def crop(
        request: Request,
        job_id: str,
        artifact_path: PathQuery,
        sha256: HashQuery,
        x: int = Query(ge=0),
        y: int = Query(ge=0),
        width: int = Query(ge=1, le=2048),
        height: int = Query(ge=1, le=2048),
        source_identity_sha256: OptionalHashQuery = None,
        view: Literal["recorded", "clip_srgb"] = "clip_srgb",
    ) -> Response:
        return await invoke(
            request,
            job_id,
            service.render,
            artifact_path,
            sha256,
            crop=(x, y, width, height),
            source_identity_sha256=source_identity_sha256,
            view=view,
        )

    @versioned.get("/quality", response_model=ApiEnvelope[dict])
    async def quality(
        request: Request,
        job_id: str,
        artifact_path: PathQuery,
        sha256: HashQuery,
        reference_path: PathQuery,
        reference_sha256: HashQuery,
        source_identity_sha256: OptionalHashQuery = None,
        reference_identity_sha256: OptionalHashQuery = None,
    ) -> Response:
        return await invoke(
            request,
            job_id,
            service.quality,
            artifact_path,
            sha256,
            reference_path,
            reference_sha256,
            source_identity_sha256=source_identity_sha256,
            reference_identity_sha256=reference_identity_sha256,
        )

    @versioned.post("/decisions", response_model=ApiEnvelope[dict])
    async def decision(request: Request, job_id: str, payload: ReviewDecisionRequest) -> Response:
        return await invoke(request, job_id, service.decision, payload)

    @versioned.post("/lighting-groups", response_model=ApiEnvelope[dict])
    async def groups(request: Request, job_id: str, payload: LightingGroupsRequest) -> Response:
        return await invoke(request, job_id, service.groups, payload)

    @versioned.post("/revisions", response_model=ApiEnvelope[dict])
    async def revision(request: Request, job_id: str, payload: RevisionRequest) -> Response:
        return await invoke(request, job_id, service.revision, payload)

    @versioned.post("/delivery", response_model=ApiEnvelope[dict])
    async def delivery(request: Request, job_id: str, payload: DeliveryRequest) -> Response:
        return await invoke(request, job_id, service.delivery, payload)

    @versioned.get("/deliveries", response_model=ApiEnvelope[DeliveryHistoryPage])
    async def deliveries(
        request: Request,
        job_id: str,
        limit: int = Query(default=20, ge=1, le=64),
        cursor: str | None = Query(default=None, min_length=1, max_length=1024),
        request_correlation_id: str | None = Query(default=None, min_length=1, max_length=64, pattern=r"^[A-Za-z0-9._:-]+$"),
    ) -> Response:
        return await invoke(
            request,
            job_id,
            service.deliveries,
            lightweight_authorization=True,
            limit=limit,
            cursor=cursor,
            request_correlation_id=request_correlation_id,
        )

    @versioned.get("/files/{identity}/{filename}")
    async def download(request: Request, job_id: str, identity: str, filename: str) -> Response:
        return await invoke(request, job_id, service.download, identity, filename)

    router.include_router(versioned, prefix="/v1")
    router.include_router(versioned, prefix="/v2")
    return router
