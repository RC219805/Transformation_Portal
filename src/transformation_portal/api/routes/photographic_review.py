"""Additive authenticated photographic review routes; existing routes are stable."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Query, Request
from starlette.responses import Response

from transformation_portal.api.v1.envelopes import ApiEnvelope
from transformation_portal.api.v1.photographic_review import (
    DeliveryRequest,
    LightingGroupsRequest,
    ReviewDecisionRequest,
    RevisionRequest,
)
from transformation_portal.orchestrator.photographic_review_access import ReviewAccessError, ReviewInputError, ReviewRuntime
from transformation_portal.orchestrator.photographic_review_service import PhotographicReviewService, envelope
from transformation_portal.orchestrator.photographic_review_store import ReviewConflict

LOGGER = logging.getLogger(__name__)
HashQuery = Annotated[str, Query(pattern=r"^[0-9a-f]{64}$")]
OptionalHashQuery = Annotated[str | None, Query(pattern=r"^[0-9a-f]{64}$")]
PathQuery = Annotated[str, Query(min_length=1, max_length=1024)]


def create_photographic_review_router(runtime: ReviewRuntime) -> APIRouter:
    router = APIRouter(prefix="/v1/jobs/{job_id}/photographic-review", tags=["Photographic review"])
    service = PhotographicReviewService()

    async def invoke(request, job_id, method, *args, **kwargs):
        try:
            context = await runtime.authorize(request, job_id)
            if isinstance(context, Response):
                return context
            result = await method(context, *args, **kwargs)
            return result if isinstance(result, Response) else envelope(result)
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

    @router.get("", response_model=ApiEnvelope[dict])
    async def catalogue(request: Request, job_id: str):
        return await invoke(request, job_id, service.catalogue)

    @router.get("/render")
    async def render(
        request: Request,
        job_id: str,
        artifact_path: PathQuery,
        sha256: HashQuery,
        max_edge: int = Query(default=1600, ge=1, le=1600),
        source_identity_sha256: OptionalHashQuery = None,
    ):
        return await invoke(
            request,
            job_id,
            service.render,
            artifact_path,
            sha256,
            max_edge=max_edge,
            source_identity_sha256=source_identity_sha256,
        )

    @router.get("/native-crop")
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
    ):
        return await invoke(
            request,
            job_id,
            service.render,
            artifact_path,
            sha256,
            crop=(x, y, width, height),
            source_identity_sha256=source_identity_sha256,
        )

    @router.get("/quality", response_model=ApiEnvelope[dict])
    async def quality(
        request: Request,
        job_id: str,
        artifact_path: PathQuery,
        sha256: HashQuery,
        reference_path: PathQuery,
        reference_sha256: HashQuery,
        source_identity_sha256: OptionalHashQuery = None,
        reference_identity_sha256: OptionalHashQuery = None,
    ):
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

    @router.post("/decisions", response_model=ApiEnvelope[dict])
    async def decision(request: Request, job_id: str, payload: ReviewDecisionRequest):
        return await invoke(request, job_id, service.decision, payload)

    @router.post("/lighting-groups", response_model=ApiEnvelope[dict])
    async def groups(request: Request, job_id: str, payload: LightingGroupsRequest):
        return await invoke(request, job_id, service.groups, payload)

    @router.post("/revisions", response_model=ApiEnvelope[dict])
    async def revision(request: Request, job_id: str, payload: RevisionRequest):
        return await invoke(request, job_id, service.revision, payload)

    @router.post("/delivery", response_model=ApiEnvelope[dict])
    async def delivery(request: Request, job_id: str, payload: DeliveryRequest):
        return await invoke(request, job_id, service.delivery, payload)

    @router.get("/files/{identity}/{filename}")
    async def download(request: Request, job_id: str, identity: str, filename: str):
        return await invoke(request, job_id, service.download, identity, filename)

    return router
