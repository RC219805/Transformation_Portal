"""Durable submission HTTP recovery never repeats mutable admission work."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from pathlib import Path

import httpx
import pytest
from starlette.requests import Request

import app as application
from transformation_portal.orchestrator.submissions import (
    SubmissionIdentity,
    SubmissionReceipt,
    submission_intent,
    submission_owner,
    validate_submission_key,
    verified_actor_owner,
)

pytestmark = pytest.mark.unit
ACTOR = {"username": "portal-a", "accessEmail": "shared@example.com", "role": "admin"}
BODY = {"pipeline": "lux-depth-v3", "args": {"input_dir": "/removed/source", "output_dir": "/removed/output"}}


def _headers(method, path, *, actor=None, key=None):
    payload = {"v": 1, "iat": int(time.time()), "method": method, "target": path, "actor": actor or ACTOR}
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    signature = hmac.new(
        application.FRONTDOOR_IDENTITY_SECRET.encode(), b"tp.frontdoor.actor.v1\n" + encoded.encode(), hashlib.sha256
    ).hexdigest()
    result = {"x-api-key": "contract-secret", "x-tp-actor-assertion": encoded + "." + signature}
    if key is not None:
        result["Idempotency-Key"] = key
    return result


@pytest.fixture
def receipts(monkeypatch, tmp_path):
    monkeypatch.setattr(application, "API_KEY_SECRET", "contract-secret")
    monkeypatch.setattr(application, "ENFORCE_JOB_API_KEY", True)
    monkeypatch.setattr(application, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(application, "FRONTDOOR_IDENTITY_SECRET", "submission-test-identity-secret-32bytes")
    monkeypatch.setattr(application, "PILOT_ACTOR_TENANTS_JSON", '{"shared@example.com":"tenant_a"}')
    monkeypatch.setattr(application, "PILOT_ALLOWED_PIPELINES", {"lux-depth-v3"})
    monkeypatch.setattr(application, "PILOT_ALLOWED_TENANTS", {"tenant_a"})
    monkeypatch.setattr(application, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "workspaces")
    monkeypatch.setattr(application, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(application, "_PILOT_TENANT_MANAGER", None)
    monkeypatch.setattr(application, "RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
    monkeypatch.setenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("TP_DATABASE_URL", "postgresql+asyncpg://unused/receipt_test")
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "execution"))
    monkeypatch.delenv("TP_JOB_SUBMISSION_RECOVERY_ENABLED", raising=False)
    saved = {}

    class Store:
        async def submission_receipt(self, tenant, owner, key):
            return saved.get((tenant, owner, key))

    monkeypatch.setattr(application, "get_operational_record_store", lambda: Store())

    async def audit(**_kwargs):
        return None

    monkeypatch.setattr(application, "_record_pilot_audit", audit)
    return saved


def _seed(saved, *, key="attempt-1", state="running", retired=False):
    receipt = SubmissionReceipt(key, submission_intent(BODY), "job_original", "lux-depth-v3", state, retired)
    saved[("tenant_a", submission_owner("tenant_a", ACTOR), key)] = receipt
    return receipt


def _deny_tenant_mkdir(monkeypatch):
    calls = []
    original = Path.mkdir
    roots = (application.PILOT_TENANT_WORKSPACE_ROOT, application.PILOT_TENANT_CAS_ROOT)

    def read_only(path, *args, **kwargs):
        if any(path.is_relative_to(root) for root in roots):
            calls.append(path)
            raise PermissionError("tenant storage is read-only")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", read_only)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("operation", ["replay", "lookup"])
@pytest.mark.parametrize("manager_state", ["cold", "warm_without_tenant"])
async def test_accepted_receipt_access_never_provisions_tenant_storage(
    receipts, monkeypatch, version, operation, manager_state
):
    _seed(receipts, state="succeeded")
    if manager_state == "warm_without_tenant":
        assert application._pilot_tenant_manager().get_tenant("tenant_a") is None
    mkdirs = _deny_tenant_mkdir(monkeypatch)
    path = f"/{version}/jobs" if operation == "replay" else f"/{version}/job-submissions/attempt-1"
    method = "POST" if operation == "replay" else "GET"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application.app, raise_app_exceptions=False), base_url="http://testserver"
    ) as client:
        response = await client.request(
            method,
            path,
            headers=_headers(method, path, key="attempt-1" if operation == "replay" else None),
            **({"json": BODY} if operation == "replay" else {}),
        )
    assert response.status_code == 200, response.text
    result = response.json()["data"]
    job = result if operation == "replay" else result["job"]
    assert job == {"id": "job_original", "state": "succeeded", "events_url": f"/{version}/jobs/job_original/events"}
    assert mkdirs == []
    if manager_state == "cold":
        assert application._PILOT_TENANT_MANAGER is None


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("key", [None, "new-attempt"])
async def test_new_and_keyless_http_work_still_requires_tenant_provisioning(receipts, monkeypatch, version, key):
    monkeypatch.setenv("TP_JOB_SUBMISSION_RECOVERY_ENABLED", "1")
    mkdirs = _deny_tenant_mkdir(monkeypatch)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("failed provisioning must not reach path checks or preview")

    monkeypatch.setattr(application, "_pilot_enforce_request_paths", forbidden)
    monkeypatch.setattr(application, "_build_config_preview_threaded", forbidden)
    path = f"/{version}/jobs"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application.app, raise_app_exceptions=False), base_url="http://testserver"
    ) as client:
        response = await client.post(path, json=BODY, headers=_headers("POST", path, key=key))
    assert response.status_code >= 400
    assert mkdirs, "New work must still materialize its authorized tenant, even when storage is unavailable"


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["replay", "lookup"])
@pytest.mark.parametrize("policy_source", ["configured", "cached"])
async def test_read_only_receipts_still_enforce_current_pipeline_policy(receipts, monkeypatch, operation, policy_source):
    _seed(receipts)
    if policy_source == "cached":
        manager = application._pilot_tenant_manager()
        manager.create_tenant("tenant_a", policy=application.TenantPolicy(allowed_node_types={"archive-gate-a"}))
    else:
        monkeypatch.setattr(application, "PILOT_ALLOWED_PIPELINES", {"archive-gate-a"})
    mkdirs = _deny_tenant_mkdir(monkeypatch)
    path = "/v1/jobs" if operation == "replay" else "/v1/job-submissions/attempt-1"
    method = "POST" if operation == "replay" else "GET"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application.app, raise_app_exceptions=False), base_url="http://testserver"
    ) as client:
        response = await client.request(
            method,
            path,
            headers=_headers(method, path, key="attempt-1" if operation == "replay" else None),
            **({"json": BODY} if operation == "replay" else {}),
        )
    assert response.status_code == 403, response.text
    assert response.json()["error"]["details"]["reason"] == "pipeline_not_allowed"
    assert mkdirs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["replay", "lookup"])
async def test_read_only_scope_rejects_conflicting_tenant_selector(receipts, monkeypatch, operation):
    _seed(receipts)
    mkdirs = _deny_tenant_mkdir(monkeypatch)
    path = "/v1/jobs" if operation == "replay" else "/v1/job-submissions/attempt-1"
    method = "POST" if operation == "replay" else "GET"
    headers = {
        **_headers(method, path, key="attempt-1" if operation == "replay" else None),
        application.PILOT_TENANT_HEADER: "tenant_b",
    }
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.request(method, path, headers=headers, **({"json": BODY} if operation == "replay" else {}))
    assert response.status_code == 403
    assert response.json()["error"]["details"]["reason"] == "tenant_actor_mismatch"
    assert mkdirs == []


@pytest.mark.asyncio
async def test_read_only_scope_still_rejects_invalid_tenant_membership_configuration(receipts, monkeypatch):
    monkeypatch.setattr(application, "PILOT_ACTOR_TENANTS_JSON", '{"shared@example.com":"../outside"}')
    monkeypatch.setattr(application, "PILOT_ALLOWED_TENANTS", {"../outside"})
    mkdirs = _deny_tenant_mkdir(monkeypatch)
    path = "/v1/job-submissions/attempt-1"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application.app, raise_app_exceptions=False), base_url="http://testserver"
    ) as client:
        response = await client.get(path, headers=_headers("GET", path))
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AUTH_CONFIGURATION_ERROR"
    assert mkdirs == []


def test_intent_is_order_stable_and_changes_with_every_request_field():
    assert submission_intent(BODY) == submission_intent({"args": BODY["args"], "pipeline": BODY["pipeline"]})
    assert submission_intent(BODY) != submission_intent({**BODY, "additional": "new intent"})


@pytest.mark.parametrize("value", [float("nan"), Path("/private"), b"bytes", {1: "non-json key"}])
def test_intent_rejects_non_json_and_nonfinite_values(value):
    with pytest.raises(ValueError):
        submission_intent({"args": value})


def test_intent_rejects_excessive_nesting_and_size():
    value = {}
    for _ in range(40):
        value = {"next": value}
    for payload in (value, {"large": "x" * (1024 * 1024)}):
        with pytest.raises(ValueError):
            submission_intent(payload)


def test_composite_owner_separates_shared_email_logins_tenants_and_operation():
    owner = submission_owner("tenant_a", ACTOR)
    assert owner != submission_owner("tenant_a", {**ACTOR, "username": "portal-b"})
    assert owner != submission_owner("tenant_b", ACTOR)
    assert owner == submission_owner("tenant_a", {**ACTOR, "role": "viewer"})
    assert owner != verified_actor_owner("tenant_a", ACTOR)


@pytest.mark.parametrize("value", ["", "../bad", "a/b", "x" * 65, "in valid"])
def test_submission_keys_are_bounded_portable_opaque_values(value):
    with pytest.raises(ValueError):
        validate_submission_key(value)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_replay_returns_canonical_state_before_paths_preview_readiness_and_quota(receipts, monkeypatch, version):
    _seed(receipts, state="succeeded")

    async def forbidden(*_args, **_kwargs):
        pytest.fail("accepted replay must not re-enter mutable admission")

    monkeypatch.setattr(application, "_pilot_enforce_request_paths", forbidden)
    monkeypatch.setattr(application, "_create_job", forbidden)
    path = f"/{version}/jobs"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.post(path, json=BODY, headers=_headers("POST", path, key="attempt-1"))
    assert response.status_code == 200, response.text
    assert response.json() == {
        "schema": "tp.orchestrator.job.v1",
        "success": True,
        "error": None,
        "data": {"id": "job_original", "state": "succeeded", "events_url": f"/{version}/jobs/job_original/events"},
    }


def _direct_request(version="v1", *, actor=None, key="attempt-1"):
    path = f"/{version}/jobs"
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "query_string": b"",
            "headers": [
                (name.lower().encode(), value.encode()) for name, value in _headers("POST", path, actor=actor, key=key).items()
            ],
        }
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
@pytest.mark.parametrize("prepared", [False, True])
async def test_direct_entry_repeats_receipt_lookup_before_mutable_checks(receipts, monkeypatch, version, prepared):
    _seed(receipts, state="succeeded")
    mkdirs = _deny_tenant_mkdir(monkeypatch)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("direct accepted replay must not re-enter mutable admission")

    monkeypatch.setattr(application, "_build_config_preview_threaded", forbidden)
    monkeypatch.setattr(application, "_pilot_enforce_request_paths", forbidden)
    identity = SubmissionIdentity("attempt-1", submission_owner("tenant_a", ACTOR), submission_intent(BODY))
    response = await getattr(application, "create_job" if version == "v1" else "create_job_v2")(
        BODY, request=_direct_request(version), **({"submission": identity} if prepared else {})
    )
    assert response.status_code == 200
    assert json.loads(response.body)["data"] == {
        "id": "job_original",
        "state": "succeeded",
        "events_url": f"/{version}/jobs/job_original/events",
    }
    assert mkdirs == []


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["v1", "v2"])
async def test_route_to_direct_entry_receipt_commit_race_replays_before_preview(receipts, monkeypatch, version):
    monkeypatch.setenv("TP_JOB_SUBMISSION_RECOVERY_ENABLED", "1")
    checked = []
    blocked_mkdirs = []

    async def commit_after_route_lookup(*_args, **_kwargs):
        checked.append("paths")
        tenant = _args[0]
        assert tenant.tenant_workspace.is_dir() and tenant.tenant_cas.is_dir()
        _seed(receipts, state="running")
        monkeypatch.setattr(application, "_PILOT_TENANT_MANAGER", None)
        blocked_mkdirs.append(_deny_tenant_mkdir(monkeypatch))
        return None

    async def forbidden(*_args, **_kwargs):
        pytest.fail("receipt committed after route lookup must replay before preview")

    monkeypatch.setattr(application, "_pilot_enforce_request_paths", commit_after_route_lookup)
    monkeypatch.setattr(application, "_build_config_preview_threaded", forbidden)
    path = f"/{version}/jobs"
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=application.app, raise_app_exceptions=False), base_url="http://testserver"
    ) as client:
        response = await client.post(path, json=BODY, headers=_headers("POST", path, key="attempt-1"))
    assert response.status_code == 200
    assert response.json()["data"]["id"] == "job_original"
    assert checked == ["paths"]
    assert blocked_mkdirs == [[]]


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["key", "owner", "intent"])
async def test_direct_entry_rejects_prepared_identity_not_bound_to_original_request(receipts, monkeypatch, changed):
    _seed(receipts)
    fields = {
        "key": "attempt-1",
        "owner_sha256": submission_owner("tenant_a", ACTOR),
        "intent_sha256": submission_intent(BODY),
    }
    fields[{"key": "key", "owner": "owner_sha256", "intent": "intent_sha256"}[changed]] = (
        "other-key" if changed == "key" else "a" * 64
    )

    async def forbidden(*_args, **_kwargs):
        pytest.fail("mismatched prepared identity must fail before mutable admission")

    monkeypatch.setattr(application, "_build_config_preview_threaded", forbidden)
    response = await application.create_job(BODY, request=_direct_request(), submission=SubmissionIdentity(**fields))
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["code"] == "INVALID_ARGUMENT"


@pytest.mark.asyncio
@pytest.mark.parametrize("revoked", ["actor", "tenant", "pipeline"])
async def test_direct_entry_rechecks_current_authority_before_receipt_replay(receipts, monkeypatch, revoked):
    _seed(receipts)
    request = _direct_request()
    tenant, denied = await application._pilot_tenant_from_request(request, action="job_create")
    assert denied is None
    if revoked == "actor":
        request = _direct_request(actor={**ACTOR, "accessEmail": "revoked@example.com"})
    elif revoked == "tenant":
        monkeypatch.setattr(application, "PILOT_ACTOR_TENANTS_JSON", '{"shared@example.com":"tenant_b"}')
    else:
        monkeypatch.setattr(application, "PILOT_ALLOWED_PIPELINES", {"archive-gate-a"})
        monkeypatch.setattr(application, "_PILOT_TENANT_MANAGER", None)

    async def forbidden(*_args, **_kwargs):
        pytest.fail("revoked authority must fail before mutable admission")

    monkeypatch.setattr(application, "_build_config_preview_threaded", forbidden)
    response = await application.create_job(BODY, request=request, pilot_tenant=tenant, portal_actor=ACTOR)
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_direct_new_key_retains_path_authority_before_preview(receipts, monkeypatch):
    monkeypatch.setenv("TP_JOB_SUBMISSION_RECOVERY_ENABLED", "1")
    called = []

    async def denied(*_args, **_kwargs):
        called.append("paths")
        return application._error_response(403, code="FORBIDDEN", message="path not allowed")

    async def forbidden(*_args, **_kwargs):
        pytest.fail("new keyed work must authorize paths before preview")

    monkeypatch.setattr(application, "_pilot_enforce_request_paths", denied)
    monkeypatch.setattr(application, "_build_config_preview_threaded", forbidden)
    response = await application.create_job(BODY, request=_direct_request())
    assert response.status_code == 403 and called == ["paths"]


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", [None, "keyless"])
async def test_trusted_direct_and_keyless_entries_keep_existing_preview_validation(monkeypatch, entry):
    seen = []

    async def preview(payload, **kwargs):
        seen.append((payload, kwargs))
        raise ValueError("unsupported pipeline")

    async def forbidden(*_args, **_kwargs):
        pytest.fail("keyless trusted entry must not require submission scope")

    monkeypatch.setattr(application, "_build_config_preview_threaded", preview)
    monkeypatch.setattr(application, "_prepare_submission", forbidden)
    monkeypatch.setattr(application, "_pilot_tenant_from_request", forbidden)
    payload = {"pipeline": "unsupported", "args": {}}
    response = await application.create_job(payload, request=_direct_request(key=None) if entry else None)
    assert response.status_code == 400
    assert json.loads(response.body)["error"]["code"] == "INVALID_ARGUMENT"
    assert seen == [(payload, {"archive_index_scan_mode": "full"})]


@pytest.mark.asyncio
async def test_direct_keyed_entry_requires_signed_actor_before_replay(receipts, monkeypatch):
    _seed(receipts)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/jobs",
            "query_string": b"",
            "headers": [(b"idempotency-key", b"attempt-1"), (b"x-api-key", b"contract-secret")],
        }
    )

    async def forbidden(*_args, **_kwargs):
        pytest.fail("unsigned direct entry must not reach mutable admission")

    monkeypatch.setattr(application, "_build_config_preview_threaded", forbidden)
    response = await application.create_job(BODY, request=request)
    assert response.status_code == 401


@pytest.mark.asyncio
async def test_prepared_identity_without_request_fails_closed_before_preview(monkeypatch):
    async def forbidden(*_args, **_kwargs):
        pytest.fail("a prepared identity without a signed request must not reach preview")

    monkeypatch.setattr(application, "_build_config_preview_threaded", forbidden)
    response = await application.create_job(
        BODY, submission=SubmissionIdentity("attempt-1", submission_owner("tenant_a", ACTOR), submission_intent(BODY))
    )
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_cross_version_lookup_has_exact_typed_receipt_and_role_is_not_owner(receipts):
    _seed(receipts)
    path = "/v2/job-submissions/attempt-1"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.get(path, headers=_headers("GET", path, actor={**ACTOR, "role": "viewer"}))
    assert response.status_code == 200
    assert response.json()["schema"] == "tp.orchestrator.job_submission.v1"
    assert response.json()["data"] == {
        "request_id": "attempt-1",
        "intent_sha256": submission_intent(BODY),
        "status": "accepted",
        "job": {"id": "job_original", "state": "running", "events_url": "/v2/jobs/job_original/events"},
    }


@pytest.mark.asyncio
async def test_shared_access_email_does_not_reveal_another_portal_owners_receipt(receipts):
    _seed(receipts)
    path = "/v1/job-submissions/attempt-1"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.get(path, headers=_headers("GET", path, actor={**ACTOR, "username": "portal-b"}))
    assert response.status_code == 404
    assert response.json()["error"]["details"]["reason"] == "submission_not_committed"


@pytest.mark.asyncio
async def test_different_intent_conflicts_without_mutable_admission(receipts):
    _seed(receipts)
    path = "/v1/jobs"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.post(
            path, json={**BODY, "extra": "different"}, headers=_headers("POST", path, key="attempt-1")
        )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "CONFLICT"


@pytest.mark.asyncio
async def test_retired_projection_retains_consumed_key_and_typed_lookup(receipts, monkeypatch):
    _seed(receipts, state="canceled", retired=True)
    mkdirs = _deny_tenant_mkdir(monkeypatch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        post = await client.post("/v1/jobs", json=BODY, headers=_headers("POST", "/v1/jobs", key="attempt-1"))
        path = "/v2/job-submissions/attempt-1"
        read = await client.get(path, headers=_headers("GET", path))
    assert post.status_code == 410
    assert post.json()["error"]["code"] == "SUBMISSION_EXPIRED"
    assert post.json()["error"]["details"]["job_id"] == "job_original"
    assert read.json()["data"]["status"] == "expired"
    assert "events_url" not in read.json()["data"]["job"]
    assert mkdirs == []


@pytest.mark.asyncio
async def test_new_key_is_default_off_while_old_receipts_remain_readable(receipts, monkeypatch):
    mkdirs = _deny_tenant_mkdir(monkeypatch)
    path = "/v1/jobs"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.post(path, json=BODY, headers=_headers("POST", path, key="new-attempt"))
    assert response.status_code == 503
    assert response.json()["error"]["details"]["reason"] == "submission_recovery_unavailable"
    assert mkdirs == []


@pytest.mark.asyncio
async def test_provider_failure_is_never_treated_as_absence_or_unkeyed_submission(receipts, monkeypatch):
    class Broken:
        async def submission_receipt(self, *_args):
            raise RuntimeError("database unavailable")

    monkeypatch.setattr(application, "get_operational_record_store", lambda: Broken())
    path = "/v1/jobs"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.post(path, json=BODY, headers=_headers("POST", path, key="attempt-1"))
    assert response.status_code == 503


@pytest.mark.asyncio
async def test_current_pipeline_revocation_prevents_receipt_replay(receipts, monkeypatch):
    _seed(receipts)
    monkeypatch.setattr(application, "PILOT_ALLOWED_PIPELINES", {"archive-gate-a"})
    path = "/v1/jobs"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.post(path, json=BODY, headers=_headers("POST", path, key="attempt-1"))
    assert response.status_code == 403


@pytest.mark.asyncio
async def test_unsigned_actor_headers_cannot_authorize_receipts(receipts, monkeypatch):
    _seed(receipts)
    mkdirs = _deny_tenant_mkdir(monkeypatch)
    path = "/v1/job-submissions/attempt-1"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.get(path, headers={"x-api-key": "contract-secret", "x-tp-actor": ACTOR["username"]})
        unauthenticated = await client.get(path)
    assert response.status_code == 401
    assert unauthenticated.status_code == 401
    assert mkdirs == []


@pytest.mark.asyncio
async def test_bootstrap_exposes_only_verified_composite_context_without_feature_enablement(receipts, monkeypatch):
    mkdirs = _deny_tenant_mkdir(monkeypatch)
    path = "/portal/bootstrap"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.get(path, headers=_headers("GET", path))
    assert response.json()["recoveryContext"] == {
        "schema": "tp.portal.recovery_context.v1",
        "tenantId": "tenant_a",
        "ownerId": verified_actor_owner("tenant_a", ACTOR),
    }
    assert response.json()["features"]["jobSubmissionReceipts"] is False
    assert mkdirs == []
    assert application._PILOT_TENANT_MANAGER is None


@pytest.mark.asyncio
async def test_invalid_canonical_receipt_state_is_a_typed_provider_failure(receipts):
    _seed(receipts, state="not-a-job-state")
    path = "/v1/job-submissions/attempt-1"
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application.app), base_url="http://testserver") as client:
        response = await client.get(path, headers=_headers("GET", path))
    assert response.status_code == 503
    assert response.json()["schema"] == "tp.orchestrator.error.v1"


@pytest.mark.asyncio
async def test_keyed_internal_admission_requires_a_real_request_and_tenant():
    from transformation_portal.orchestrator.submissions import SubmissionIdentity

    response = await application._create_distributed_job(
        payload=BODY,
        pipeline="lux-depth-v3",
        execution_args={},
        argv=[],
        trusted_output_dir=None,
        pilot_tenant=None,
        portal_actor=None,
        request=None,
        api_version="v1",
        submission=SubmissionIdentity("key", "a" * 64, submission_intent(BODY)),
    )
    assert response.status_code == 503
