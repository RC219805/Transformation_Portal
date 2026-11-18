"""End-to-end rejection-path tests for the FastAPI orchestrator.

The lower-level helpers (``_has_valid_api_key``, ``_is_rate_limited``,
``_is_protected_api_key_endpoint``) are unit-tested in
``test_app_orchestrator_runtime.py``. These tests exercise the
*middleware-integrated* rejection paths through ``TestClient`` so the
contract surface — status codes, error envelope shape, header echo —
is pinned down for:

* API-key missing or wrong on a protected endpoint (401 UNAUTHORIZED)
* API-key authentication misconfigured (503 AUTH_CONFIGURATION_ERROR)
* TrustedHost middleware rejecting an off-allowlist Host header (400)
* Rate-limit middleware returning 429 RATE_LIMITED beyond the bucket
* Public liveness routes (``/healthz``, ``/ready``) bypassing auth

Tests follow the fixture conventions of
``test_app_orchestrator_contract_http.py``: an autouse fixture
snapshots and restores the configuration knobs that the rejection
logic reads.
"""

from __future__ import annotations

import asyncio
import importlib
import json

import pytest
from fastapi.testclient import TestClient

from transformation_portal.orchestrator import reset_singletons

pytestmark = pytest.mark.unit

orchestrator_app = importlib.import_module("app")


@pytest.fixture(autouse=True)
def _reset_orchestrator_globals():
    previous_api_key = orchestrator_app.API_KEY_SECRET
    previous_enforce = orchestrator_app.ENFORCE_JOB_API_KEY
    previous_rate_limit = orchestrator_app.RATE_LIMIT_PER_MINUTE
    orchestrator_app.API_KEY_SECRET = "rejection-secret"
    orchestrator_app.ENFORCE_JOB_API_KEY = True
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    reset_singletons()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False
    try:
        yield
    finally:
        orchestrator_app.API_KEY_SECRET = previous_api_key
        orchestrator_app.ENFORCE_JOB_API_KEY = previous_enforce
        orchestrator_app.RATE_LIMIT_PER_MINUTE = previous_rate_limit
        orchestrator_app.RATE_LIMIT_BUCKETS.clear()
        orchestrator_app.JOBS.clear()
        orchestrator_app.EVENT_SUBSCRIBERS.clear()
        reset_singletons()
        orchestrator_app.app.state.job_repository = None
        orchestrator_app.app.state.job_repository_unavailable = False


@pytest.fixture(name="anon_client")
def _anon_client_fixture():
    with TestClient(orchestrator_app.app) as client:
        yield client


@pytest.fixture(name="auth_client")
def _auth_client_fixture():
    with TestClient(
        orchestrator_app.app,
        headers={"x-api-key": "rejection-secret"},
    ) as client:
        yield client


class TestApiKeyRejection:
    def test_missing_key_on_protected_endpoint_returns_401(self, anon_client):
        response = anon_client.get("/v1/jobs")
        assert response.status_code == 401
        body = response.json()
        # Error envelope contract: code + message + details
        error = body.get("error") or body
        assert error.get("code") == "UNAUTHORIZED"
        assert "invalid or missing api key" in error.get("message", "").lower()
        assert error.get("details", {}).get("path") == "/v1/jobs"

    def test_wrong_key_on_protected_endpoint_returns_401(self, anon_client):
        response = anon_client.get(
            "/v1/jobs",
            headers={"x-api-key": "this-is-not-the-key"},
        )
        assert response.status_code == 401
        error = response.json().get("error") or response.json()
        assert error.get("code") == "UNAUTHORIZED"

    def test_valid_bearer_token_passes_auth_layer(self, anon_client):
        # /v1/jobs requires auth; with a valid bearer it must reach the route
        # handler and return the documented envelope. A weaker `!= 401`
        # assertion would silently accept a 5xx regression in routing.
        response = anon_client.get(
            "/v1/jobs",
            headers={"authorization": "Bearer rejection-secret"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body.get("schema") == "tp.orchestrator.jobs.v1"
        assert body.get("success") is True
        assert body.get("error") is None
        data = body.get("data", {})
        assert "jobs" in data
        assert "total" in data
        assert "returned" in data

    def test_protected_endpoint_with_no_secret_configured_returns_503(self, anon_client):
        orchestrator_app.API_KEY_SECRET = ""
        # Enforcement still on, but no secret to compare against.
        response = anon_client.get("/v1/jobs")
        assert response.status_code == 503
        error = response.json().get("error") or response.json()
        assert error.get("code") == "AUTH_CONFIGURATION_ERROR"
        assert error.get("details", {}).get("env") == "TP_API_KEY"

    @pytest.mark.parametrize("path", ["/healthz", "/ready"])
    def test_public_liveness_bypasses_auth(self, anon_client, path):
        response = anon_client.get(path)
        assert response.status_code == 200


class TestTrustedHostRejection:
    def test_off_allowlist_host_returns_400(self, auth_client):
        response = auth_client.get("/healthz", headers={"host": "evil.example.com"})
        assert response.status_code == 400
        # Starlette's TrustedHostMiddleware returns plain text "Invalid host header".
        assert "invalid host" in response.text.lower()

    def test_localhost_is_allowed(self, auth_client):
        response = auth_client.get("/healthz", headers={"host": "localhost"})
        assert response.status_code == 200

    def test_loopback_ipv4_is_allowed(self, auth_client):
        response = auth_client.get("/healthz", headers={"host": "127.0.0.1"})
        assert response.status_code == 200


class TestRateLimitRejection:
    def test_burst_above_threshold_returns_429(self, auth_client):
        orchestrator_app.RATE_LIMIT_PER_MINUTE = 3
        # First 3 requests succeed; the 4th from the same client should 429.
        for _ in range(3):
            assert auth_client.get("/healthz").status_code == 200
        response = auth_client.get("/healthz")
        assert response.status_code == 429
        error = response.json().get("error") or response.json()
        assert error.get("code") == "RATE_LIMITED"
        assert "rate limit exceeded" in error.get("message", "").lower()
        assert "client_ip" in error.get("details", {})

    def test_disabled_rate_limit_never_rejects(self, auth_client):
        orchestrator_app.RATE_LIMIT_PER_MINUTE = 0
        for _ in range(20):
            assert auth_client.get("/healthz").status_code == 200

    def test_429_emits_retry_and_ratelimit_headers(self, auth_client):
        """The per-IP rate-limit 429 path emits the contract header set:
        Retry-After + X-RateLimit-{Limit,Remaining,Reset}. Header values are
        validated by parse + range, not exact times, to keep the contract
        stable across clock jitter.
        """
        import time as _time

        orchestrator_app.RATE_LIMIT_PER_MINUTE = 3
        for _ in range(3):
            assert auth_client.get("/healthz").status_code == 200
        before_send = int(_time.time())
        response = auth_client.get("/healthz")
        assert response.status_code == 429

        retry_after_raw = response.headers.get("Retry-After")
        assert retry_after_raw is not None, "Retry-After header missing on 429"
        retry_after = int(retry_after_raw)
        assert retry_after >= 1, "Retry-After must be a positive integer"

        limit_raw = response.headers.get("X-RateLimit-Limit")
        assert limit_raw is not None, "X-RateLimit-Limit header missing on 429"
        assert int(limit_raw) == 3

        remaining_raw = response.headers.get("X-RateLimit-Remaining")
        assert remaining_raw is not None, "X-RateLimit-Remaining header missing on 429"
        assert int(remaining_raw) == 0

        reset_raw = response.headers.get("X-RateLimit-Reset")
        assert reset_raw is not None, "X-RateLimit-Reset header missing on 429"
        reset_epoch = int(reset_raw)
        # Reset is when the oldest bucket entry exits the sliding window.
        # Loose bound: must not be in the past relative to when we sent.
        assert reset_epoch >= before_send

    def test_429_envelope_body_unchanged_by_header_addition(self, auth_client):
        """Adding rate-limit headers must remain additive: the ApiEnvelope
        error body keeps its existing schema/code/message/details shape.
        """
        orchestrator_app.RATE_LIMIT_PER_MINUTE = 1
        assert auth_client.get("/healthz").status_code == 200
        response = auth_client.get("/healthz")
        assert response.status_code == 429
        body = response.json()
        # Envelope shape: schema + success + data + error
        assert body.get("schema") == "tp.orchestrator.error.v1"
        assert body.get("success") is False
        assert body.get("data") is None
        error = body.get("error") or {}
        assert error.get("code") == "RATE_LIMITED"
        assert "rate limit exceeded" in error.get("message", "").lower()
        assert "client_ip" in (error.get("details") or {})

    def test_failed_api_key_attempts_are_rate_limited(self, anon_client):
        """Wrong-key guesses must consume the per-client budget (fail closed):
        once it is spent, further guesses get the documented 429 envelope with
        Retry-After instead of an unlimited stream of 401s.
        """
        orchestrator_app.RATE_LIMIT_PER_MINUTE = 3
        statuses = [anon_client.get("/v1/jobs", headers={"x-api-key": f"guess-{index}"}).status_code for index in range(3)]
        assert statuses == [401, 401, 401]

        response = anon_client.get("/v1/jobs", headers={"x-api-key": "guess-final"})
        assert response.status_code == 429
        body = response.json()
        assert body.get("schema") == "tp.orchestrator.error.v1"
        assert (body.get("error") or {}).get("code") == "RATE_LIMITED"
        assert int(response.headers["Retry-After"]) >= 1
        assert int(response.headers["X-RateLimit-Limit"]) == 3

        # The budget is per client, so a valid key from the same client is
        # throttled too until the window frees a slot.
        valid = anon_client.get("/v1/jobs", headers={"x-api-key": "rejection-secret"})
        assert valid.status_code == 429

    def test_non_429_responses_do_not_emit_ratelimit_headers(self, auth_client):
        """The contract is additive only on rejection. Successful liveness
        responses must not advertise rate-limit headers, since that would
        imply a budget the route did not actually consume.
        """
        orchestrator_app.RATE_LIMIT_PER_MINUTE = 60
        response = auth_client.get("/healthz")
        assert response.status_code == 200
        assert response.headers.get("Retry-After") is None
        assert response.headers.get("X-RateLimit-Limit") is None
        assert response.headers.get("X-RateLimit-Remaining") is None
        assert response.headers.get("X-RateLimit-Reset") is None


class TestNonJsonCompliantPayloadRejection:
    """Valid-JSON literals that the strict response encoder cannot echo
    (``NaN``, ``Infinity``, ``1e400``, unpaired ``\\ud800`` escapes) and
    integers too large for ``float`` must never produce a bare 500: the config
    preview flags the arg as a field error and echoes it as ``null`` (the
    managed photography preview contract), and job creation is refused with
    the 400 INVALID_ARGUMENT envelope.
    """

    _ARGS = '"input_dir": "/nonexistent/in", "output_dir": "/nonexistent/out"'
    _CASES = [
        ("lux-depth-v3", '"max_workers": 1e400', "max_workers"),
        ("lux-depth-v3", '"reconstruction_iterations": Infinity', "reconstruction_iterations"),
        ("lux-depth-v3", '"v2_preset": "\\ud800"', "v2_preset"),
        ("lux-depth-v3", '"\\udc00": 1', "args"),
        ("archive-gate-b", '"note": NaN', "note"),
        ("archive-gate-b", '"nested": [{"value": -1e400}]', "nested"),
        ("archive-gate-b", '"note": "\\ud800"', "note"),
        ("archive-gate-a", '"archive_command": "fixity-verify", "workers": 1e400', "workers"),
    ]

    @staticmethod
    def _post(client, path, raw):
        return client.post(path, headers={"content-type": "application/json"}, content=raw.encode("utf-8"))

    @pytest.mark.parametrize(("pipeline", "extra", "field"), _CASES)
    def test_preview_flags_unechoable_args_and_echoes_null(self, auth_client, pipeline, extra, field):
        raw = '{"pipeline": "%s", "args": {%s, %s}}' % (pipeline, self._ARGS, extra)
        response = self._post(auth_client, "/v1/config-preview", raw)

        assert response.status_code == 200
        preview = response.json()["data"]
        assert field in [issue.get("field") for issue in preview["field_errors"]]
        assert preview["argv_preview"] == ""
        # The echo must be strict JSON: no NaN/Infinity literal, no lone surrogate.
        json.dumps(preview, allow_nan=False).encode("utf-8")

    @pytest.mark.parametrize("path", ["/v1/jobs", "/v2/jobs"])
    @pytest.mark.parametrize(("pipeline", "extra", "field"), _CASES)
    def test_job_create_refuses_unechoable_args(self, auth_client, path, pipeline, extra, field):
        raw = '{"pipeline": "%s", "args": {%s, %s}}' % (pipeline, self._ARGS, extra)
        response = self._post(auth_client, path, raw)

        assert response.status_code == 400
        body = response.json()
        assert body.get("schema") == "tp.orchestrator.error.v1"
        assert (body.get("error") or {}).get("code") == "INVALID_ARGUMENT"
        assert orchestrator_app.JOBS == {}

    @pytest.mark.parametrize("path", ["/v1/jobs", "/v2/jobs"])
    def test_otherwise_valid_archive_job_with_nan_arg_is_not_accepted(self, auth_client, tmp_path, path):
        input_dir = tmp_path / "in"
        output_dir = tmp_path / "out"
        input_dir.mkdir()
        output_dir.mkdir()
        manifest = output_dir / "m.jsonl"
        manifest.write_text("{}\n", encoding="utf-8")
        args = json.dumps({"input_dir": str(input_dir), "output_dir": str(output_dir), "manifest_jsonl": str(manifest)})
        valid = self._post(auth_client, "/v1/config-preview", '{"pipeline": "archive-gate-b", "args": %s}' % args)
        assert valid.status_code == 200
        assert valid.json()["data"]["field_errors"] == []

        raw = '{"pipeline": "archive-gate-b", "args": %s, "note": NaN}}' % args[:-1]
        response = self._post(auth_client, path, raw)

        assert response.status_code == 400
        error = response.json().get("error") or {}
        assert error.get("code") == "INVALID_ARGUMENT"
        assert error.get("details") == {"field": "note", "reason": "non_finite_number"}
        assert orchestrator_app.JOBS == {}

    @pytest.mark.parametrize("leaf", ["NaN", '"\\ud800"'])
    def test_deeply_nested_unechoable_arg_is_a_field_error_not_500(self, auth_client, leaf):
        # The JSON parser accepts nesting deeper than the interpreter recursion
        # limit; the null-echo copy of the preview must not recurse per level.
        depth = 2000
        raw = '{"pipeline": "archive-gate-b", "args": {%s, "note": %s%s%s}}' % (
            self._ARGS,
            "[" * depth,
            leaf,
            "]" * depth,
        )

        preview_response = self._post(auth_client, "/v1/config-preview", raw)

        assert preview_response.status_code == 200
        preview = preview_response.json()["data"]
        assert "note" in [issue.get("field") for issue in preview["field_errors"]]
        node = preview["normalized_args"]["note"]
        for _ in range(depth):
            node = node[0]
        assert node is None

        job_response = self._post(auth_client, "/v1/jobs", raw)
        assert job_response.status_code == 400
        assert (job_response.json().get("error") or {}).get("code") == "INVALID_ARGUMENT"
        assert orchestrator_app.JOBS == {}

    def test_unechoable_arg_is_the_reported_error_when_otherwise_valid(self):
        preview = orchestrator_app._reject_unechoable_preview_args(
            "archive-gate-b",
            {"note": float("nan"), "label": "ok"},
            {
                "pipeline": "archive-gate-b",
                "normalized_args": {"note": float("nan"), "label": "ok"},
                "execution_args": {"note": float("nan"), "label": "ok"},
                "argv_preview": "tp archive --note nan",
                "field_errors": [],
                "field_warnings": [],
                "readiness": None,
            },
        )

        assert preview["field_errors"] == [
            {
                "field": "note",
                "code": "non_finite_number",
                "message": orchestrator_app.PORTAL_SAFE_ERROR_MESSAGES["non_finite_number"],
            }
        ]
        assert preview["normalized_args"] == {"note": None, "label": "ok"}
        assert preview["execution_args"] == {"note": None, "label": "ok"}
        assert preview["argv_preview"] == ""
        assert preview["next_best_action"]

    def test_huge_integer_probability_is_a_field_error_not_500(self, auth_client):
        raw = '{"pipeline": "lux-depth-v3", "args": {%s, "sam2_pred_iou_thresh": 1%s}}' % (self._ARGS, "0" * 400)
        response = self._post(auth_client, "/v1/config-preview", raw)

        assert response.status_code == 200
        codes = [issue.get("code") for issue in response.json()["data"]["field_errors"]]
        assert "invalid_sam2_pred_iou_thresh" in codes

    def test_huge_integer_rum_value_returns_400(self, auth_client, monkeypatch):
        monkeypatch.setenv("TP_PORTAL_RUM_ENABLED", "1")
        monkeypatch.setenv("TP_PORTAL_RUM_ROLLOUT_PERCENT", "100")
        raw = (
            '{"event_type": "core_web_vital", "route": "/portal", "view": "overview", "unit": "ms",'
            ' "metric": "lcp", "value": 1%s}' % ("0" * 400)
        )
        response = self._post(auth_client, "/v1/portal/rum", raw)

        assert response.status_code == 400
        error = response.json().get("error") or {}
        assert error.get("details") == {"field": "payload", "reason": "invalid_value"}

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_integer_normalisers_record_errors_for_overflowing_values(self, value):
        errors: list = []
        assert orchestrator_app._normalize_optional_positive_int(value, "max_workers", errors) is None
        assert orchestrator_app._normalize_optional_non_negative_int(value, "verify_sample", errors) is None
        assert [issue.get("field") for issue in errors] == ["max_workers", "verify_sample"]

    def test_probability_normaliser_records_error_for_huge_integer(self):
        errors: list = []
        assert orchestrator_app._normalize_optional_probability(10**400, "sam2_pred_iou_thresh", errors) is None
        assert [issue.get("field") for issue in errors] == ["sam2_pred_iou_thresh"]


class TestArtifactDeletionRejection:
    @pytest.mark.parametrize("path", ["/v1/jobs/{job_id}/artifacts", "/v2/jobs/{job_id}/artifacts"])
    def test_active_job_artifact_delete_returns_409(self, auth_client, path):
        job = orchestrator_app.Job(
            id="reject_active_artifact_delete",
            created_at=orchestrator_app._now(),
            request={"pipeline": "lux-depth-v3", "args": {}},
            state="running",
        )
        asyncio.run(orchestrator_app._job_repository().create(orchestrator_app._record_from_job(job)))
        orchestrator_app.JOBS[job.id] = job

        response = auth_client.delete(path.format(job_id=job.id))

        assert response.status_code == 409
        error = response.json().get("error") or response.json()
        assert error.get("code") == "CONFLICT"
        assert error.get("details", {}).get("state") == "running"
