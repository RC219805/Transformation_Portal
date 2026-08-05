"""Focused tests for the extracted jobs router factory seam."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from starlette.responses import JSONResponse, Response

from transformation_portal.api.routes.jobs import DEFAULT_JOB_LIST_LIMIT, JobRouteHandlers, create_jobs_router
from transformation_portal.api.v1 import JobEnvelope, JobsListEnvelope, JobStatusEnvelope

pytestmark = [pytest.mark.unit]


@dataclass
class _RecordingHandlers:
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def _record(self, name: str, **kwargs: Any) -> Response:
        self.calls.append((name, kwargs))
        return JSONResponse({"handler": name, **kwargs})

    async def create_job_http(self, request: Request, payload: dict[str, Any]) -> Response:
        return self._record("create_job_http", path=request.url.path, payload=payload)

    async def create_job_v2_http(self, request: Request, payload: dict[str, Any]) -> Response:
        return self._record("create_job_v2_http", path=request.url.path, payload=payload)

    async def list_jobs(self, request: Request, limit: int) -> Response:  # noqa: ARG002
        return self._record("list_jobs", limit=limit)

    async def list_jobs_v2(self, request: Request, limit: int) -> Response:  # noqa: ARG002
        return self._record("list_jobs_v2", limit=limit)

    async def get_job(self, request: Request, job_id: str, include_logs: bool) -> Response:  # noqa: ARG002
        return self._record("get_job", job_id=job_id, include_logs=include_logs)

    async def get_job_v2(self, request: Request, job_id: str, include_logs: bool) -> Response:  # noqa: ARG002
        return self._record("get_job_v2", job_id=job_id, include_logs=include_logs)

    async def get_job_artifact(self, request: Request, job_id: str, artifact_path: str) -> Response:  # noqa: ARG002
        return self._record("get_job_artifact", job_id=job_id, artifact_path=artifact_path)

    async def get_job_artifact_v2(self, request: Request, job_id: str, artifact_path: str) -> Response:  # noqa: ARG002
        return self._record("get_job_artifact_v2", job_id=job_id, artifact_path=artifact_path)

    async def delete_job_artifacts(self, request: Request, job_id: str) -> Response:  # noqa: ARG002
        return self._record("delete_job_artifacts", job_id=job_id)

    async def delete_job_artifacts_v2(self, request: Request, job_id: str) -> Response:  # noqa: ARG002
        return self._record("delete_job_artifacts_v2", job_id=job_id)

    async def cancel_job(self, request: Request, job_id: str) -> Response:  # noqa: ARG002
        return self._record("cancel_job", job_id=job_id)

    async def cancel_job_v2(self, request: Request, job_id: str) -> Response:  # noqa: ARG002
        return self._record("cancel_job_v2", job_id=job_id)

    async def job_events(self, request: Request, job_id: str) -> Response:
        return self._record("job_events", path=request.url.path, job_id=job_id)

    async def job_events_v2(self, request: Request, job_id: str) -> Response:
        return self._record("job_events_v2", path=request.url.path, job_id=job_id)


def _build_handlers(recording: _RecordingHandlers) -> JobRouteHandlers:
    return JobRouteHandlers(
        create_job_http=recording.create_job_http,
        create_job_v2_http=recording.create_job_v2_http,
        list_jobs=recording.list_jobs,
        list_jobs_v2=recording.list_jobs_v2,
        get_job=recording.get_job,
        get_job_v2=recording.get_job_v2,
        get_job_artifact=recording.get_job_artifact,
        get_job_artifact_v2=recording.get_job_artifact_v2,
        delete_job_artifacts=recording.delete_job_artifacts,
        delete_job_artifacts_v2=recording.delete_job_artifacts_v2,
        cancel_job=recording.cancel_job,
        cancel_job_v2=recording.cancel_job_v2,
        job_events=recording.job_events,
        job_events_v2=recording.job_events_v2,
    )


def _build_test_app(*, job_list_limit: int = 37) -> tuple[TestClient, _RecordingHandlers, FastAPI]:
    recording = _RecordingHandlers()
    test_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    test_app.include_router(create_jobs_router(_build_handlers(recording), job_list_limit=job_list_limit))
    return TestClient(test_app), recording, test_app


def test_jobs_router_default_factory_call_uses_portal_list_limit_contract() -> None:
    recording = _RecordingHandlers()
    test_app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    test_app.include_router(create_jobs_router(_build_handlers(recording)))
    client = TestClient(test_app)

    assert client.get("/v1/jobs").json() == {
        "handler": "list_jobs",
        "limit": DEFAULT_JOB_LIST_LIMIT,
    }
    assert client.get("/v2/jobs").json() == {
        "handler": "list_jobs_v2",
        "limit": DEFAULT_JOB_LIST_LIMIT,
    }
    assert recording.calls == [
        ("list_jobs", {"limit": DEFAULT_JOB_LIST_LIMIT}),
        ("list_jobs_v2", {"limit": DEFAULT_JOB_LIST_LIMIT}),
    ]


def test_jobs_router_pins_paths_methods_endpoint_names_and_response_models() -> None:
    _client, _recording, test_app = _build_test_app()

    schema_refs = {
        JobEnvelope: {"$ref": "#/components/schemas/ApiEnvelope_JobBriefData_"},
        JobsListEnvelope: {"$ref": "#/components/schemas/ApiEnvelope_JobsListData_"},
        JobStatusEnvelope: {"$ref": "#/components/schemas/ApiEnvelope_JobStatusData_"},
        None: {},
    }
    expected = {
        ("POST", "/v1/jobs"): ("create_job_http", JobEnvelope),
        ("POST", "/v2/jobs"): ("create_job_v2_http", JobEnvelope),
        ("GET", "/v1/jobs"): ("list_jobs", JobsListEnvelope),
        ("GET", "/v2/jobs"): ("list_jobs_v2", JobsListEnvelope),
        ("GET", "/v1/jobs/{job_id}"): ("get_job", JobStatusEnvelope),
        ("GET", "/v2/jobs/{job_id}"): ("get_job_v2", JobStatusEnvelope),
        ("GET", "/v1/jobs/{job_id}/artifacts/{artifact_path}"): ("get_job_artifact", None),
        ("GET", "/v2/jobs/{job_id}/artifacts/{artifact_path}"): ("get_job_artifact_v2", None),
        ("DELETE", "/v1/jobs/{job_id}/artifacts"): ("delete_job_artifacts", JobStatusEnvelope),
        ("DELETE", "/v2/jobs/{job_id}/artifacts"): ("delete_job_artifacts_v2", JobStatusEnvelope),
        ("POST", "/v1/jobs/{job_id}/cancel"): ("cancel_job", JobEnvelope),
        ("POST", "/v2/jobs/{job_id}/cancel"): ("cancel_job_v2", JobEnvelope),
        ("GET", "/v1/jobs/{job_id}/events"): ("job_events", None),
        ("GET", "/v2/jobs/{job_id}/events"): ("job_events_v2", None),
    }

    openapi_paths = test_app.openapi()["paths"]
    tracked = {
        (method.upper(), path): operation
        for path, path_item in openapi_paths.items()
        for method, operation in path_item.items()
        if method.upper() in {"GET", "POST", "DELETE"}
    }
    assert set(tracked) == set(expected)
    for route, (_endpoint_name, response_model) in expected.items():
        assert tracked[route]["responses"]["200"]["content"]["application/json"]["schema"] == schema_refs[response_model]

    for (method, path), (endpoint_name, _response_model) in expected.items():
        values = {"job_id": "job-1", "artifact_path": "nested/report.json"}
        expected_path = path.format(**values)
        route_values = {name: value for name, value in values.items() if "{" + name + "}" in path}
        assert str(test_app.url_path_for(endpoint_name, **route_values)) == expected_path


def test_jobs_router_delegates_list_limits_to_the_injected_handlers() -> None:
    client, recording, _test_app = _build_test_app(job_list_limit=41)

    assert client.get("/v1/jobs").json() == {"handler": "list_jobs", "limit": 41}
    assert client.get("/v2/jobs", params={"limit": 9}).json() == {"handler": "list_jobs_v2", "limit": 9}
    assert recording.calls == [
        ("list_jobs", {"limit": 41}),
        ("list_jobs_v2", {"limit": 9}),
    ]


@pytest.mark.parametrize("jobs_path", ["/v1/jobs", "/v2/jobs"])
def test_jobs_router_rejects_non_object_create_body_before_handler(
    jobs_path: str,
) -> None:
    client, recording, _test_app = _build_test_app()

    response = client.post(jobs_path, json=["not", "an", "object"])

    assert response.status_code == 422
    assert recording.calls == []


@pytest.mark.parametrize("jobs_path", ["/v1/jobs", "/v2/jobs"])
def test_jobs_router_rejects_missing_create_body_before_handler(
    jobs_path: str,
) -> None:
    client, recording, _test_app = _build_test_app()

    response = client.post(jobs_path)

    assert response.status_code == 422
    assert recording.calls == []


@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/v1/jobs", {"limit": "not-an-int"}),
        ("/v2/jobs", {"limit": "not-an-int"}),
        ("/v1/jobs/job-1", {"include_logs": "not-a-bool"}),
        ("/v2/jobs/job-2", {"include_logs": "not-a-bool"}),
    ],
)
def test_jobs_router_rejects_invalid_query_parameters_before_handler(
    path: str,
    params: dict[str, str],
) -> None:
    client, recording, _test_app = _build_test_app()

    response = client.get(path, params=params)

    assert response.status_code == 422
    assert recording.calls == []


@pytest.mark.parametrize(
    ("method", "path", "json_payload", "expected_handler", "expected_kwargs"),
    [
        (
            "post",
            "/v1/jobs",
            {"pipeline": "lux-depth-v3", "args": {"input_dir": "in"}},
            "create_job_http",
            {"path": "/v1/jobs", "payload": {"pipeline": "lux-depth-v3", "args": {"input_dir": "in"}}},
        ),
        (
            "post",
            "/v2/jobs",
            {"pipeline": "lux-depth-v3", "args": {"output_dir": "out"}},
            "create_job_v2_http",
            {"path": "/v2/jobs", "payload": {"pipeline": "lux-depth-v3", "args": {"output_dir": "out"}}},
        ),
        ("get", "/v1/jobs/job-1?include_logs=false", None, "get_job", {"job_id": "job-1", "include_logs": False}),
        ("get", "/v2/jobs/job-2", None, "get_job_v2", {"job_id": "job-2", "include_logs": True}),
        (
            "get",
            "/v1/jobs/job-1/artifacts/nested/report.json",
            None,
            "get_job_artifact",
            {"job_id": "job-1", "artifact_path": "nested/report.json"},
        ),
        (
            "get",
            "/v2/jobs/job-2/artifacts/nested/report.json",
            None,
            "get_job_artifact_v2",
            {"job_id": "job-2", "artifact_path": "nested/report.json"},
        ),
        ("delete", "/v1/jobs/job-1/artifacts", None, "delete_job_artifacts", {"job_id": "job-1"}),
        ("delete", "/v2/jobs/job-2/artifacts", None, "delete_job_artifacts_v2", {"job_id": "job-2"}),
        ("post", "/v1/jobs/job-1/cancel", None, "cancel_job", {"job_id": "job-1"}),
        ("post", "/v2/jobs/job-2/cancel", None, "cancel_job_v2", {"job_id": "job-2"}),
        (
            "get",
            "/v1/jobs/job-1/events",
            None,
            "job_events",
            {"path": "/v1/jobs/job-1/events", "job_id": "job-1"},
        ),
        (
            "get",
            "/v2/jobs/job-2/events",
            None,
            "job_events_v2",
            {"path": "/v2/jobs/job-2/events", "job_id": "job-2"},
        ),
    ],
)
def test_jobs_router_delegates_each_route_shape_without_rewriting_arguments(
    method: str,
    path: str,
    json_payload: dict[str, Any] | None,
    expected_handler: str,
    expected_kwargs: dict[str, Any],
) -> None:
    client, recording, _test_app = _build_test_app()

    if json_payload is None:
        response = getattr(client, method)(path)
    else:
        response = getattr(client, method)(path, json=json_payload)

    assert response.status_code == 200
    assert response.json() == {"handler": expected_handler, **expected_kwargs}
    assert recording.calls == [(expected_handler, expected_kwargs)]
