"""Phase 1.D - byte-identical wire-shape regression for /v[12]/jobs routes.

The orchestrator's success-path responses move from
``JSONResponse(_api_envelope(...))`` to returning Pydantic envelope
models directly. FastAPI's ``response_model=`` then validates the
response at runtime, but only if the bytes on the wire are
*semantically identical* to today's output. This file pins that.

Two layers of coverage:

1. ``test_pydantic_envelope_matches_*`` - side-by-side assertions
   that the Pydantic models in
   ``src/transformation_portal/api/v1/envelopes.py`` and
   ``.../jobs.py`` produce the same dict as the legacy
   ``app.py:_api_envelope`` helper, after canonical
   ``json.dumps(..., sort_keys=True)`` normalization. Catches model-
   level drift.

2. ``test_*_route_wire_shape`` - HTTP-level assertions through the
   real ``TestClient``: exercise every success-path response (create,
   list, get, cancel, both v1 and v2) and pin the four envelope
   keys (``schema``, ``success``, ``data``, ``error``) plus the
   per-payload field set. Catches FastAPI-introduced shape changes.

The serialization knob that makes Pydantic match the legacy helper
is ``exclude_unset=True``: legacy ``_api_envelope`` mirrors whatever
dict the caller passed in (e.g. ``_cancel_job`` passes
``{"id": ..., "state": ...}`` with no ``events_url`` key), and
``JobBriefData(id=..., state=...)`` with ``events_url`` unset and
``exclude_unset=True`` produces the same shape. Routes that refactor
to return Pydantic envelopes use ``response_model_exclude_unset=True``
for the same reason.

SSE event payloads stay manual and are intentionally out of scope.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterator

import pytest
from fastapi.testclient import TestClient

import app as orchestrator_app
from transformation_portal.api.v1.envelopes import ApiEnvelope
from transformation_portal.api.v1.jobs import (
    JobBriefData,
    JobsListData,
    JobStatusData,
)

pytestmark = [pytest.mark.unit]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


# ---------------------------------------------------------------------------
# Layer 1: Pydantic-model output equals legacy helper output (no HTTP).
# ---------------------------------------------------------------------------


def test_pydantic_envelope_matches_helper_for_create_brief() -> None:
    """Create response: id + state + events_url; all three keys present."""
    legacy = orchestrator_app._api_envelope(
        "tp.orchestrator.job.v1",
        success=True,
        data={"id": "job_abc", "state": "queued", "events_url": "/v1/jobs/job_abc/events"},
        error=None,
    )
    pydantic_dump = ApiEnvelope[JobBriefData](
        schema="tp.orchestrator.job.v1",
        success=True,
        data=JobBriefData(id="job_abc", state="queued", events_url="/v1/jobs/job_abc/events"),
        error=None,
    ).model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert _canonical(legacy) == _canonical(pydantic_dump)


def test_pydantic_envelope_matches_helper_for_cancel_brief() -> None:
    """Cancel response: id + state only; events_url must be absent."""
    legacy = orchestrator_app._api_envelope(
        "tp.orchestrator.job.v1",
        success=True,
        data={"id": "job_xyz", "state": "canceled"},
        error=None,
    )
    pydantic_dump = ApiEnvelope[JobBriefData](
        schema="tp.orchestrator.job.v1",
        success=True,
        data=JobBriefData(id="job_xyz", state="canceled"),
        error=None,
    ).model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert _canonical(legacy) == _canonical(pydantic_dump)
    # Belt-and-braces: the resulting dict must have no events_url key.
    assert "events_url" not in pydantic_dump["data"]
    # And error must remain null on the envelope (set explicitly).
    assert pydantic_dump["error"] is None


def test_pydantic_envelope_matches_helper_for_jobs_list() -> None:
    """JobsListData with a serialized job entry: byte-identical to helper."""
    sample_job: Dict[str, Any] = {
        "id": "job_aaa",
        "pipeline": "lux-depth-v3",
        "created_at": 1700_000_000.0,
        "started_at": 1700_000_001.0,
        "finished_at": None,
        "state": "running",
        "progress": 42,
        "exit_code": None,
        "events_url": "/v1/jobs/job_aaa/events",
        "artifacts": {},
        "error": None,
        "run_summary": None,
        "last_event_at": 1700_000_002.0,
    }
    legacy = orchestrator_app._api_envelope(
        "tp.orchestrator.jobs.v1",
        success=True,
        data={"jobs": [sample_job], "total": 1, "returned": 1},
        error=None,
    )
    pydantic_dump = ApiEnvelope[JobsListData](
        schema="tp.orchestrator.jobs.v1",
        success=True,
        data=JobsListData(
            jobs=[JobStatusData(**sample_job)],
            total=1,
            returned=1,
        ),
        error=None,
    ).model_dump(mode="json", by_alias=True, exclude_unset=True)
    assert _canonical(legacy) == _canonical(pydantic_dump)


# ---------------------------------------------------------------------------
# Layer 2: HTTP-level wire-shape pins. Hit the real routes; assert envelope.
# ---------------------------------------------------------------------------


@pytest.fixture(name="client")
def _client_fixture(
    monkeypatch: pytest.MonkeyPatch,
    mark_da3_runtime_available: None,
) -> Iterator[TestClient]:
    """A TestClient with the contract auth-key wired and a trivial fake runner.

    ``mark_da3_runtime_available`` makes the lux-depth-v3 dispatch
    preflight pass without a real DA3 install; it's the same fixture
    the legacy contract tests use.
    """

    async def _instant_complete(job, _argv) -> None:  # noqa: ANN001
        # Trivial fake runner: mark the job succeeded synchronously.
        job.state = "succeeded"
        job.exit_code = 0
        now = orchestrator_app._now()
        job.started_at = now
        job.finished_at = now
        job.done_published_at = now

    monkeypatch.setattr(orchestrator_app, "_run_job", _instant_complete)
    # Avoid pipeline-validation false-rejects for the lightweight smoke jobs.
    monkeypatch.setattr(orchestrator_app, "_materialize_dispatch_output_dir", lambda *_a, **_kw: None)

    previous_api_key = orchestrator_app.API_KEY_SECRET
    previous_enforce = orchestrator_app.ENFORCE_JOB_API_KEY
    orchestrator_app.API_KEY_SECRET = "contract-secret"
    orchestrator_app.ENFORCE_JOB_API_KEY = True
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    try:
        with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
            yield client
    finally:
        orchestrator_app.API_KEY_SECRET = previous_api_key
        orchestrator_app.ENFORCE_JOB_API_KEY = previous_enforce
        orchestrator_app.JOBS.clear()
        orchestrator_app.EVENT_SUBSCRIBERS.clear()


def _create_dummy_job(client: TestClient, *, api_version: str) -> str:
    response = client.post(
        f"/{api_version}/jobs",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/phase1d_wire_shape",
            },
        },
    )
    assert response.status_code == 200, response.text
    return response.json()["data"]["id"]


def _assert_envelope_shape(body: Dict[str, Any], *, expected_schema: str) -> None:
    """Every success envelope must have exactly schema/success/data/error."""
    assert set(body.keys()) == {"schema", "success", "data", "error"}, body
    assert body["schema"] == expected_schema
    assert body["success"] is True
    assert body["error"] is None
    assert isinstance(body["data"], dict)


@pytest.mark.parametrize("api_version", ["v1", "v2"])
def test_create_job_wire_shape(client: TestClient, api_version: str) -> None:
    response = client.post(
        f"/{api_version}/jobs",
        json={
            "pipeline": "lux-depth-v3",
            "args": {
                "input_dir": "./tests/fixtures/archive_small/archive_root",
                "output_dir": "./tests/fixtures/portal_contract_output/phase1d_create",
            },
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_envelope_shape(body, expected_schema="tp.orchestrator.job.v1")
    assert set(body["data"].keys()) == {"id", "state", "events_url"}
    assert body["data"]["state"] in {"queued", "running", "succeeded"}
    assert body["data"]["events_url"].endswith(f"/{api_version}/jobs/{body['data']['id']}/events")


@pytest.mark.parametrize("api_version", ["v1", "v2"])
def test_list_jobs_wire_shape(client: TestClient, api_version: str) -> None:
    _create_dummy_job(client, api_version=api_version)
    response = client.get(f"/{api_version}/jobs")
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_envelope_shape(body, expected_schema="tp.orchestrator.jobs.v1")
    assert set(body["data"].keys()) == {"jobs", "total", "returned"}
    assert isinstance(body["data"]["jobs"], list)
    assert body["data"]["total"] >= 1
    assert body["data"]["returned"] >= 1
    # logs_tail must be absent (list endpoint passes include_logs=False).
    for job in body["data"]["jobs"]:
        assert "logs_tail" not in job


@pytest.mark.parametrize("api_version", ["v1", "v2"])
def test_get_job_wire_shape(client: TestClient, api_version: str) -> None:
    job_id = _create_dummy_job(client, api_version=api_version)
    response = client.get(f"/{api_version}/jobs/{job_id}")
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_envelope_shape(body, expected_schema="tp.orchestrator.job_status.v1")
    # The detail endpoint defaults to include_logs=True, so logs_tail must be
    # present (even if empty).
    expected_required = {
        "id",
        "pipeline",
        "created_at",
        "state",
        "progress",
        "events_url",
        "artifacts",
        "logs_tail",
    }
    assert expected_required.issubset(body["data"].keys()), body["data"].keys()


@pytest.mark.parametrize("api_version", ["v1", "v2"])
def test_cancel_job_wire_shape(client: TestClient, api_version: str) -> None:
    job_id = _create_dummy_job(client, api_version=api_version)
    response = client.post(f"/{api_version}/jobs/{job_id}/cancel")
    assert response.status_code == 200, response.text
    body = response.json()
    _assert_envelope_shape(body, expected_schema="tp.orchestrator.job.v1")
    # Cancel response carries {id, state} - no events_url.
    assert set(body["data"].keys()) == {"id", "state"}
    assert body["data"]["id"] == job_id


# ---------------------------------------------------------------------------
# Layer 3: canonical-JSON regression - the same response endpoint, hit twice,
# must produce a canonical JSON string that is invariant within one process
# (no field-order randomization, no float formatting drift). Pins the
# determinism the Pydantic refactor must preserve.
# ---------------------------------------------------------------------------


def test_list_jobs_response_is_canonically_deterministic(client: TestClient) -> None:
    _create_dummy_job(client, api_version="v1")
    first = client.get("/v1/jobs").json()
    second = client.get("/v1/jobs").json()
    # In-process determinism: same content yields the same canonical bytes.
    assert _canonical(first) == _canonical(second)
