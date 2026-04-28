"""Unit tests for the job lifecycle response/request models (Phase 1.2 PR C).

These tests verify the typed models in
``transformation_portal.api.v1.jobs`` accept and reject the wire shapes that
``app.py``'s job-lifecycle handlers produce. They complement (do not replace)
the existing wire-contract tests in
``tests/test_app_orchestrator_contract_http.py`` and
``tests/test_app_orchestrator_runtime.py`` — those exercise the routes
end-to-end; these exercise the models in isolation so a regression there is
caught before the contract tests get to it.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from transformation_portal.api.v1 import (
    ErrorObject,
    JobBriefData,
    JobCreateRequest,
    JobEnvelope,
    JobsListData,
    JobsListEnvelope,
    JobStatusData,
    JobStatusEnvelope,
)

# ---------------------------------------------------------------------------
# JobBriefData — payload for tp.orchestrator.job.v1
# ---------------------------------------------------------------------------


class TestJobBriefData:
    """Used by POST /v1/jobs (with events_url) and POST /v1/jobs/{id}/cancel
    (without events_url). Both shapes must validate."""

    def test_create_response_shape_with_events_url(self) -> None:
        # Mirrors app.py:_create_job line 8514-8518
        data = JobBriefData(
            id="job_abcd1234",
            state="queued",
            events_url="/v1/jobs/job_abcd1234/events",
        )
        dumped = data.model_dump(mode="json")
        assert dumped["id"] == "job_abcd1234"
        assert dumped["state"] == "queued"
        assert dumped["events_url"] == "/v1/jobs/job_abcd1234/events"

    def test_cancel_response_shape_without_events_url(self) -> None:
        # Mirrors app.py:_cancel_job line 8699: data={"id": ..., "state": ...}
        # The model serialises events_url=None on the Python side; the runtime
        # wire shape from the handler omits the key entirely (handler returns
        # JSONResponse directly, bypassing response_model serialization). The
        # OpenAPI schema treats events_url as optional.
        data = JobBriefData(id="job_abcd1234", state="canceled")
        dumped = data.model_dump(mode="json")
        assert dumped["id"] == "job_abcd1234"
        assert dumped["state"] == "canceled"
        assert dumped["events_url"] is None

    def test_unknown_state_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JobBriefData(id="x", state="canceling")  # type: ignore[arg-type]

    def test_extra_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JobBriefData(id="x", state="queued", extra="nope")  # type: ignore[call-arg]

    @pytest.mark.parametrize(
        "state",
        ["queued", "running", "succeeded", "partial", "failed", "canceled"],
    )
    def test_every_known_state_accepted(self, state: str) -> None:
        data = JobBriefData(id="x", state=state)  # type: ignore[arg-type]
        assert data.state == state


# ---------------------------------------------------------------------------
# JobStatusData — payload for tp.orchestrator.job_status.v1 + entries in jobs.v1
# ---------------------------------------------------------------------------


class TestJobStatusData:
    """Mirrors app.py:_serialize_job (line 6530)."""

    def test_minimal_fresh_job(self) -> None:
        # A queued job that hasn't started: started_at/finished_at/exit_code
        # all None; no error; logs_tail absent (list endpoint sets
        # include_logs=False).
        data = JobStatusData(
            id="job_x",
            pipeline="lux-depth-v3",
            created_at=1.0,
            state="queued",
            progress=0,
            events_url="/v1/jobs/job_x/events",
        )
        dumped = data.model_dump(mode="json")
        assert dumped["id"] == "job_x"
        assert dumped["pipeline"] == "lux-depth-v3"
        assert dumped["state"] == "queued"
        assert dumped["progress"] == 0
        assert dumped["started_at"] is None
        assert dumped["finished_at"] is None
        assert dumped["exit_code"] is None
        assert dumped["error"] is None
        assert dumped["run_summary"] is None
        assert dumped["last_event_at"] is None
        assert dumped["artifacts"] == {}
        assert dumped["logs_tail"] is None  # list endpoint shape

    def test_running_job_with_logs_tail_and_indexed_artifacts(self) -> None:
        # GET /v1/jobs/{id}?include_logs=true with artifacts already indexed.
        # The artifacts dict shape comes from _index_job_artifacts at
        # app.py:6430 — {output_dir, items, indexed_count, truncated} — and
        # the model accepts it as dict[str, Any].
        artifacts_payload = {
            "output_dir": "/tmp/job_x/output",
            "items": [
                {"path": "output.tif", "size_bytes": 1024, "sha256": "abc..."},
                {"path": "summary.json", "size_bytes": 256, "sha256": "def..."},
            ],
            "indexed_count": 2,
            "truncated": False,
        }
        data = JobStatusData(
            id="job_x",
            pipeline="archive-gate-a",
            created_at=1.0,
            started_at=1.5,
            state="running",
            progress=50,
            events_url="/v1/jobs/job_x/events",
            logs_tail=["[INFO] started", "[INFO] step 1"],
            artifacts=artifacts_payload,
            last_event_at=2.0,
        )
        dumped = data.model_dump(mode="json")
        assert dumped["logs_tail"] == ["[INFO] started", "[INFO] step 1"]
        assert dumped["artifacts"] == artifacts_payload
        assert dumped["artifacts"]["indexed_count"] == 2
        assert dumped["last_event_at"] == 2.0

    def test_failed_job_with_error(self) -> None:
        # _serialize_job sets data["error"] = job.error which is a dict
        # matching ErrorObject's shape (set by _error_obj). The model
        # accepts both the dict form and an ErrorObject directly.
        data = JobStatusData(
            id="job_x",
            pipeline="lux-depth-v3",
            created_at=1.0,
            started_at=1.5,
            finished_at=10.0,
            state="failed",
            progress=42,
            exit_code=1,
            events_url="/v1/jobs/job_x/events",
            error={"code": "RUNNER_EXIT_NONZERO", "message": "boom", "details": {"exit_code": 1}},
        )
        dumped = data.model_dump(mode="json")
        assert dumped["error"]["code"] == "RUNNER_EXIT_NONZERO"
        assert dumped["error"]["details"] == {"exit_code": 1}
        assert dumped["exit_code"] == 1

    def test_error_field_accepts_error_object_instance(self) -> None:
        # Construction via ErrorObject also works (used in tests + future
        # internal call sites).
        data = JobStatusData(
            id="job_x",
            pipeline="lux-depth-v3",
            created_at=1.0,
            state="failed",
            progress=0,
            events_url="/v1/jobs/job_x/events",
            error=ErrorObject(code="RUNNER_NOT_FOUND", message="missing runner"),
        )
        dumped = data.model_dump(mode="json")
        assert dumped["error"]["code"] == "RUNNER_NOT_FOUND"
        assert dumped["error"]["details"] == {}

    def test_extra_top_level_keys_accepted(self) -> None:
        # extra="allow" — _serialize_job may grow new keys in later PRs.
        data = JobStatusData(
            id="job_x",
            pipeline="lux-depth-v3",
            created_at=1.0,
            state="queued",
            progress=0,
            events_url="/v1/jobs/job_x/events",
            future_telemetry_field="not-yet-modeled",
        )
        assert data.model_dump(mode="json")["future_telemetry_field"] == "not-yet-modeled"


# ---------------------------------------------------------------------------
# JobsListData — payload for tp.orchestrator.jobs.v1
# ---------------------------------------------------------------------------


class TestJobsListData:
    def test_empty_list(self) -> None:
        data = JobsListData(jobs=[], total=0, returned=0)
        assert data.model_dump(mode="json") == {"jobs": [], "total": 0, "returned": 0}

    def test_list_with_entries(self) -> None:
        # Per app.py:_list_jobs (line 8534), entries come from
        # _serialize_job(..., include_logs=False) — logs_tail will always be
        # absent / None for list entries.
        entry = JobStatusData(
            id="job_x",
            pipeline="lux-depth-v3",
            created_at=1.0,
            state="succeeded",
            progress=100,
            events_url="/v1/jobs/job_x/events",
        )
        data = JobsListData(jobs=[entry, entry], total=2, returned=2)
        dumped = data.model_dump(mode="json")
        assert len(dumped["jobs"]) == 2
        assert dumped["total"] == 2
        assert dumped["returned"] == 2
        # logs_tail absent / null for list entries
        assert dumped["jobs"][0]["logs_tail"] is None

    def test_extra_keys_rejected(self) -> None:
        with pytest.raises(ValidationError):
            JobsListData(jobs=[], total=0, returned=0, page=1)  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# JobCreateRequest — request model for POST /v[12]/jobs (not yet wired)
# ---------------------------------------------------------------------------


class TestJobCreateRequest:
    def test_minimal_required_pipeline(self) -> None:
        req = JobCreateRequest(pipeline="lux-depth-v3")
        assert req.pipeline == "lux-depth-v3"
        assert req.args == {}

    def test_pipeline_required(self) -> None:
        with pytest.raises(ValidationError):
            JobCreateRequest()  # type: ignore[call-arg]

    def test_args_passes_through(self) -> None:
        req = JobCreateRequest(
            pipeline="lux-depth-v3",
            args={"input_dir": "/fixtures/in", "output_dir": "/fixtures/out", "device": "cpu"},
        )
        assert req.args["input_dir"] == "/fixtures/in"

    def test_extra_top_level_keys_passthrough(self) -> None:
        # Existing handler accepts extra fields liberally; the model preserves
        # that contract via extra="allow". When a future PR wires this model
        # as the handler param, this guarantees no existing caller breaks.
        req = JobCreateRequest(pipeline="lux-depth-v3", overrides={"foo": "bar"})
        assert req.model_dump(mode="json")["overrides"] == {"foo": "bar"}


# ---------------------------------------------------------------------------
# Envelope aliases — full round-trip wraps for each schema
# ---------------------------------------------------------------------------


class TestJobEnvelopes:
    def test_create_envelope_matches_handler_shape(self) -> None:
        # Mirrors app.py:_create_job line 8510-8521
        env = JobEnvelope(
            schema="tp.orchestrator.job.v1",
            success=True,
            data=JobBriefData(
                id="job_x",
                state="queued",
                events_url="/v1/jobs/job_x/events",
            ),
        )
        dumped = env.model_dump(mode="json")
        assert list(dumped.keys()) == ["schema", "success", "data", "error"]
        assert dumped["schema"] == "tp.orchestrator.job.v1"
        assert dumped["data"]["id"] == "job_x"
        assert dumped["error"] is None

    def test_cancel_envelope_matches_handler_shape(self) -> None:
        # Mirrors app.py:_cancel_job line 8695-8702
        env = JobEnvelope(
            schema="tp.orchestrator.job.v1",
            success=True,
            data=JobBriefData(id="job_x", state="canceled"),
        )
        dumped = env.model_dump(mode="json")
        # events_url serialises to null in this Python-side model_dump; the
        # actual wire shape from the handler omits the key (JSONResponse
        # bypass). Either is wire-compatible.
        assert dumped["data"]["state"] == "canceled"

    def test_status_envelope_round_trip(self) -> None:
        # Mirrors app.py:_get_job line 8578-8585
        status = JobStatusData(
            id="job_x",
            pipeline="lux-depth-v3",
            created_at=1.0,
            state="running",
            progress=50,
            events_url="/v1/jobs/job_x/events",
        )
        env = JobStatusEnvelope(schema="tp.orchestrator.job_status.v1", success=True, data=status)
        dumped = env.model_dump(mode="json")
        assert dumped["schema"] == "tp.orchestrator.job_status.v1"
        assert dumped["data"]["progress"] == 50

    def test_list_envelope_round_trip(self) -> None:
        # Mirrors app.py:_list_jobs line 8544-8554
        env = JobsListEnvelope(
            schema="tp.orchestrator.jobs.v1",
            success=True,
            data=JobsListData(jobs=[], total=0, returned=0),
        )
        dumped = env.model_dump(mode="json")
        assert dumped["schema"] == "tp.orchestrator.jobs.v1"
        assert dumped["data"]["total"] == 0

    def test_envelope_aliases_carry_typed_data_payloads(self) -> None:
        # Behavioral check that each alias's `data` field is bound to the
        # right payload type. We avoid `assert JobEnvelope is ApiEnvelope[T]`
        # — that relies on Pydantic's internal generic-specialization cache,
        # which isn't part of the public contract and could change across
        # Pydantic versions without breaking real semantics. Per-route alias
        # design is still documented inline; this test just verifies the
        # observable behavior (rejects wrong-shape payloads, accepts
        # right-shape payloads) instead of object identity.

        # JobEnvelope.data must be a valid JobBriefData (requires id + state).
        with pytest.raises(ValidationError):
            JobEnvelope(
                schema="tp.orchestrator.job.v1",
                success=True,
                data={"not": "a job"},  # missing id, state
            )
        # JobsListEnvelope.data must be a valid JobsListData (jobs/total/returned).
        with pytest.raises(ValidationError):
            JobsListEnvelope(
                schema="tp.orchestrator.jobs.v1",
                success=True,
                data={"jobs": []},  # missing total, returned
            )
        # JobStatusEnvelope.data must be a valid JobStatusData.
        with pytest.raises(ValidationError):
            JobStatusEnvelope(
                schema="tp.orchestrator.job_status.v1",
                success=True,
                data={"id": "x"},  # missing pipeline, created_at, state, ...
            )

        # And the corresponding right-shape payloads must succeed:
        JobEnvelope(
            schema="tp.orchestrator.job.v1",
            success=True,
            data=JobBriefData(id="x", state="queued"),
        )
        JobsListEnvelope(
            schema="tp.orchestrator.jobs.v1",
            success=True,
            data=JobsListData(jobs=[], total=0, returned=0),
        )
        JobStatusEnvelope(
            schema="tp.orchestrator.job_status.v1",
            success=True,
            data=JobStatusData(
                id="x",
                pipeline="lux-depth-v3",
                created_at=1.0,
                state="queued",
                progress=0,
                events_url="/v1/jobs/x/events",
            ),
        )

    def test_unrecognised_schema_string_rejected_at_literal_level(self) -> None:
        # SchemaName Literal still rejects unknown strings even though the
        # alias doesn't pin to a single value.
        with pytest.raises(ValidationError):
            JobEnvelope(
                schema="tp.orchestrator.not_a_real_schema.v1",
                success=True,
                data=JobBriefData(id="x", state="queued"),
            )
