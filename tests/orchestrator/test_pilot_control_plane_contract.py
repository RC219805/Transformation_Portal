"""Focused contracts for opt-in managed pilot control-plane behavior."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import importlib
import json
import time
from pathlib import Path
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
    monkeypatch.setattr(orchestrator_app, "FRONTDOOR_IDENTITY_SECRET", "frontdoor-test-identity-secret-32bytes")
    monkeypatch.setattr(
        orchestrator_app, "PILOT_ACTOR_TENANTS_JSON", '{"a@example.com":"tenant_a","b@example.com":"tenant_b"}'
    )
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
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()
    yield
    reset_singletons()
    reset_queue_singleton()
    orchestrator_app.app.state.job_repository = None
    orchestrator_app.app.state.job_repository_unavailable = False
    orchestrator_app.JOBS.clear()
    orchestrator_app.EVENT_SUBSCRIBERS.clear()
    orchestrator_app.RATE_LIMIT_BUCKETS.clear()


async def _audit_noop(**_kwargs: Any) -> None:
    return None


def _identity_headers(method: str, target: str, email: str = "a@example.com") -> dict[str, str]:
    payload = {
        "v": 1,
        "iat": int(time.time()),
        "method": method,
        "target": target,
        "actor": {"username": "actor", "accessEmail": email, "role": "admin"},
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode()).decode().rstrip("=")
    signature = hmac.new(
        orchestrator_app.FRONTDOOR_IDENTITY_SECRET.encode(), b"tp.frontdoor.actor.v1\n" + encoded.encode(), hashlib.sha256
    ).hexdigest()
    return {"x-tp-actor-assertion": encoded + "." + signature}


def _seed_job(job: Any) -> None:
    asyncio.run(orchestrator_app._job_repository().create(orchestrator_app._record_from_job(job)))


def test_pilot_tenant_mode_requires_authenticated_actor_for_config_preview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)

    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post("/v1/config-preview", json={"pipeline": "lux-depth-v3", "args": {}})

    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHORIZED"
    assert body["error"]["details"] == {
        "field": "x-tp-tenant-id",
        "reason": "authenticated_actor_required",
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

        list_response = client.get("/v1/jobs", headers=_identity_headers("GET", "/v1/jobs"))
        cross_tenant_response = client.get("/v1/jobs/job_tenant_b", headers=_identity_headers("GET", "/v1/jobs/job_tenant_b"))

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


def test_pilot_dispatch_filesystem_preflight_requires_tenant_scoped_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "workspaces")
    monkeypatch.setattr(orchestrator_app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(orchestrator_app, "_PILOT_TENANT_MANAGER", None)

    tenant = orchestrator_app._pilot_tenant_manager().create_tenant(
        "tenant_a",
        policy=orchestrator_app._pilot_tenant_policy(),
    )
    input_dir = tenant.tenant_workspace / "input"
    input_dir.mkdir(parents=True)
    output_dir = tenant.tenant_workspace / "output"
    outside = tmp_path / "outside"
    outside.mkdir()

    orchestrator_app._pilot_enforce_dispatch_filesystem_tenant(
        tenant,
        orchestrator_app.DispatchFilesystemPreflight(input_dir=input_dir, output_dir=output_dir),
        pipeline="lux-depth-v3",
    )

    with pytest.raises(orchestrator_app.JobPreflightError) as exc_info:
        orchestrator_app._pilot_enforce_dispatch_filesystem_tenant(
            tenant,
            orchestrator_app.DispatchFilesystemPreflight(input_dir=outside, output_dir=output_dir),
            pipeline="lux-depth-v3",
        )

    exc = exc_info.value
    assert exc.reason == "tenant_path_outside_workspace"
    assert exc.field == "input_dir"
    assert exc.status_code == 403
    assert exc.extra == {"pipeline": "lux-depth-v3", "tenant_id": "tenant_a"}


@pytest.mark.parametrize(
    "method,path",
    [
        ("GET", "/v1/jobs/job_tenant_b"),
        ("GET", "/v1/jobs/job_tenant_b/events"),
        ("GET", "/v1/jobs/job_tenant_b/artifacts/image.png"),
        ("DELETE", "/v1/jobs/job_tenant_b/artifacts"),
        ("POST", "/v1/jobs/job_tenant_b/cancel"),
    ],
)
def test_authenticated_tenant_cannot_access_other_job_surfaces(monkeypatch, method, path):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        _seed_job(
            orchestrator_app.Job(
                id="job_tenant_b",
                created_at=time.time(),
                state="succeeded",
                request={"pipeline": "lux-depth-v3"},
                effective_request={"pipeline": "lux-depth-v3", "args": {}, "tenant_id": "tenant_b"},
            )
        )
        response = client.request(method, path, headers=_identity_headers(method, path))
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "NOT_FOUND"


@pytest.mark.parametrize("method,path", [("GET", "/v1/jobs"), ("POST", "/v1/jobs"), ("POST", "/v1/config-preview")])
def test_tenant_selector_cannot_override_authenticated_membership(monkeypatch, method, path):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)
    headers = _identity_headers(method, path)
    headers["x-tp-tenant-id"] = "tenant_b"
    headers["x-tp-actor-email"] = "b@example.com"
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.request(method, path, headers=headers, json={"pipeline": "lux-depth-v3", "args": {}})
    assert response.status_code == 403
    assert response.json()["error"]["details"]["reason"] == "tenant_actor_mismatch"


def test_submission_receives_only_verified_actor_tenant(monkeypatch):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)

    async def capture(payload, **kwargs):
        assert kwargs["pilot_tenant"].tenant_id == "tenant_a"
        assert kwargs["portal_actor"]["accessEmail"] == "a@example.com"
        return orchestrator_app.JSONResponse({"captured": True})

    monkeypatch.setattr(orchestrator_app, "create_job", capture)
    headers = _identity_headers("POST", "/v1/jobs")
    headers["x-tp-actor-email"] = "b@example.com"
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post("/v1/jobs", json={"pipeline": "lux-depth-v3", "tenant_id": "tenant_b"}, headers=headers)
    assert response.status_code == 200


@pytest.mark.parametrize("email", ["missing@example.com", "b@example.com"])
def test_membership_mapping_and_allowlist_fail_closed(monkeypatch, email):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "PILOT_ALLOWED_TENANTS", {"tenant_a"})
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.get("/v1/jobs", headers=_identity_headers("GET", "/v1/jobs", email))
    assert response.status_code == 403


def test_assertion_does_not_replace_backend_api_key(monkeypatch):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    with TestClient(orchestrator_app.app) as client:
        response = client.get("/v1/jobs", headers=_identity_headers("GET", "/v1/jobs"))
    assert response.status_code == 401


@pytest.mark.parametrize(
    "secret,mapping",
    [
        ("", "{}"),
        ("short", '{"a@example.com":"tenant_a"}'),
        ("contract-secret", '{"a@example.com":"tenant_a"}'),
        ("x" * 32, '{"a@example.com":"../escape"}'),
    ],
)
def test_tenant_identity_config_failure_is_closed(monkeypatch, secret, mapping):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "FRONTDOOR_IDENTITY_SECRET", secret)
    monkeypatch.setattr(orchestrator_app, "PILOT_ACTOR_TENANTS_JSON", mapping)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.get("/v1/jobs", headers={"x-tp-tenant-id": "tenant_a"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AUTH_CONFIGURATION_ERROR"


@pytest.mark.parametrize("mapping,secret", [("{}", "x" * 32), ('{"a@example.com":"tenant_a"}', "")])
def test_readiness_fails_closed_for_missing_tenant_identity_configuration(monkeypatch, mapping, secret):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "PILOT_ACTOR_TENANTS_JSON", mapping)
    monkeypatch.setattr(orchestrator_app, "FRONTDOOR_IDENTITY_SECRET", secret)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        assert client.get("/ready").json()["ok"] is False
        response = client.get("/v1/readiness")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "AUTH_CONFIGURATION_ERROR"


def test_unsigned_actor_and_tenant_headers_cannot_authorize_membership(monkeypatch):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.get(
            "/v1/jobs",
            headers={
                "x-tp-tenant-id": "tenant_a",
                "x-tp-actor-email": "a@example.com",
                "x-tp-actor": "admin",
                "x-tp-actor-role": "admin",
            },
        )
    assert response.status_code == 401
    assert response.json()["error"]["details"]["reason"] == "authenticated_actor_required"
    assert orchestrator_app._PILOT_TENANT_MANAGER is None


@pytest.mark.parametrize("route", ["/v1/jobs", "/v2/jobs", "/v1/config-preview"])
@pytest.mark.parametrize(
    "field",
    [
        "manifest_jsonl",
        "manifestJsonl",
        "archive_root",
        "archiveIndex",
        "policy_yaml",
        "cameras_sidecar_path",
        "vlmCaptioningModel",
        "sam2CheckpointPath",
        "fastvlmPythonExecutable",
        "fastvlm_mlx_vlm_dir",
    ],
)
def test_cross_tenant_auxiliary_input_is_denied_before_preview(monkeypatch, tmp_path, route, field):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "workspaces")
    monkeypatch.setattr(orchestrator_app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)

    async def forbidden_preview(*args, **kwargs):
        pytest.fail("cross-tenant path reached content-inspecting preview")

    monkeypatch.setattr(orchestrator_app, "_build_config_preview_threaded", forbidden_preview)
    owned = tmp_path / "workspaces" / "tenant_a"
    other = tmp_path / "workspaces" / "tenant_b" / "secret.jsonl"
    headers = _identity_headers("POST", route)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post(
            route,
            headers=headers,
            json={
                "pipeline": "archive-gate-c",
                "args": {"input_dir": str(owned), "output_dir": str(owned / "output"), field: str(other)},
            },
        )
    assert response.status_code == 403
    assert response.json()["error"]["details"]["reason"] == "tenant_path_outside_workspace"
    assert not other.exists()


@pytest.mark.parametrize("kind", ["outside", "prefix_sibling", "parent", "relative", "tilde", "nul", "returning_symlink"])
def test_tenant_path_is_confined_before_user_path_filesystem_operations(monkeypatch, tmp_path, kind):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "workspaces")
    monkeypatch.setattr(orchestrator_app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(orchestrator_app, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)
    owned = tmp_path / "workspaces" / "tenant_a"
    other = tmp_path / "workspaces" / "tenant_b"
    if kind == "returning_symlink":
        other.mkdir(parents=True)
        (other / "returning").symlink_to(owned, target_is_directory=True)
    raw = {
        "outside": str(other / "secret.jsonl"),
        "prefix_sibling": str(owned.with_name("tenant_a_extra") / "secret.jsonl"),
        "parent": str(owned / ".." / "tenant_a" / "manifest.jsonl"),
        "relative": "workspaces/tenant_b/secret.jsonl",
        "tilde": "~other_user/secret.jsonl",
        "nul": str(owned / "bad\x00path"),
        "returning_symlink": str(other / "returning" / "manifest.jsonl"),
    }[kind]
    original_resolve, original_expanduser = Path.resolve, Path.expanduser

    def guarded_resolve(path, *args, **kwargs):
        if str(path) == raw:
            raise AssertionError("unconfined request path reached filesystem resolution")
        return original_resolve(path, *args, **kwargs)

    def guarded_expanduser(path):
        if str(path) == raw:
            raise AssertionError("unconfined request path reached home-directory expansion")
        return original_expanduser(path)

    async def forbidden_preview(*args, **kwargs):
        pytest.fail("unconfined request path reached content-inspecting preview")

    monkeypatch.setattr(Path, "resolve", guarded_resolve)
    monkeypatch.setattr(Path, "expanduser", guarded_expanduser)
    monkeypatch.setattr(orchestrator_app, "_build_config_preview_threaded", forbidden_preview)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post(
            "/v1/config-preview",
            headers=_identity_headers("POST", "/v1/config-preview"),
            json={"pipeline": "archive-gate-a", "args": {"archive_index": raw}},
        )
    assert response.status_code == 403
    assert response.json()["error"]["details"]["reason"] == "tenant_path_outside_workspace"


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("escapes", [False, True])
def test_confined_links_preserve_user_path_and_shared_runtime_identity(monkeypatch, tmp_path, shared, escapes):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "workspaces")
    monkeypatch.setattr(orchestrator_app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)
    root = tmp_path / "runtime" if shared else tmp_path / "workspaces" / "tenant_a"
    root.mkdir(parents=True)
    target = (tmp_path if escapes else root) / "image with spaces — 1.tif"
    target.write_bytes(b"fixture")
    selector = root / "selected link"
    selector.symlink_to(target)
    if shared:
        configured_alias = tmp_path / "configured_runtime"
        configured_alias.symlink_to(root, target_is_directory=True)
        monkeypatch.setattr(orchestrator_app, "default_fastvlm_runtime_root", lambda: configured_alias)
    field = "fastvlm_python_executable" if shared else "archive_index"

    async def existing_preview(payload, **kwargs):
        assert not escapes, "escaping symlink reached content-inspecting preview"
        assert payload["args"][field] == str(selector)
        return {"pipeline": "lux-depth-v3", "errors": [{"field": field, "reason": "existing_runtime_validation"}]}

    monkeypatch.setattr(orchestrator_app, "_build_config_preview_threaded", existing_preview)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post(
            "/v1/config-preview",
            headers=_identity_headers("POST", "/v1/config-preview"),
            json={"pipeline": "lux-depth-v3", "args": {field: str(selector)}},
        )
    if escapes:
        assert response.status_code == 403
        assert response.json()["error"]["details"]["reason"] == "tenant_path_outside_workspace"
    else:
        assert response.status_code == 200
        assert response.json()["data"]["errors"][0]["reason"] == "existing_runtime_validation"


@pytest.mark.parametrize("value", [2026, ["secret.jsonl"], {"path": "secret.jsonl"}, False])
def test_nonstring_tenant_path_is_rejected_before_preview(monkeypatch, value):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)

    async def forbidden_preview(*args, **kwargs):
        pytest.fail("non-string path reached content-inspecting preview")

    monkeypatch.setattr(orchestrator_app, "_build_config_preview_threaded", forbidden_preview)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post(
            "/v1/config-preview",
            headers=_identity_headers("POST", "/v1/config-preview"),
            json={"pipeline": "archive-gate-a", "args": {"archive_index": value}},
        )
    assert response.status_code == 400
    assert response.json()["error"]["details"] == {"field": "archive_index", "reason": "invalid_request"}


@pytest.mark.parametrize(
    "field", ["sam2_checkpoint_path", "fastvlm_python_executable", "fastvlm_mlx_vlm_dir", "vlm_captioning_model"]
)
def test_shared_runtime_paths_reach_existing_provenance_validation(monkeypatch, tmp_path, field):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)
    shared = tmp_path / "server_runtime"
    monkeypatch.setattr(orchestrator_app, "MANAGED_SAM2_TRUSTED_ROOTS", [shared])
    monkeypatch.setattr(orchestrator_app, "default_fastvlm_runtime_root", lambda: shared)
    path = str(shared / "selected_model")
    reached = []

    async def existing_preview(payload, **kwargs):
        reached.append(payload["args"][field])
        return {"pipeline": "lux-depth-v3", "errors": [{"field": field, "reason": "provenance_test_rejection"}]}

    monkeypatch.setattr(orchestrator_app, "_build_config_preview_threaded", existing_preview)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post(
            "/v1/config-preview",
            headers=_identity_headers("POST", "/v1/config-preview"),
            json={"pipeline": "lux-depth-v3", "args": {field: path}},
        )
    assert response.status_code == 200
    assert reached == [path]
    assert response.json()["data"]["errors"][0]["reason"] == "provenance_test_rejection"


@pytest.mark.parametrize("field", ["policy_yaml", "policyYaml", "manifest_jsonl"])
@pytest.mark.parametrize("operation", ["rights-apply", "manifest-build"])
def test_governed_rights_policy_is_shared_only_for_its_operation_and_field(monkeypatch, tmp_path, field, operation):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "PILOT_ALLOWED_PIPELINES", {"archive-gate-a"})
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)
    governed = tmp_path / "governed_policy"
    monkeypatch.setattr(orchestrator_app, "ARCHIVE_RIGHTS_POLICY_ROOT", governed)
    policy = governed / "rights_flags.yml"
    allowed = operation == "rights-apply" and field in {"policy_yaml", "policyYaml"}
    reached = []

    async def existing_preview(payload, **kwargs):
        reached.append(payload["args"][field])
        return {"pipeline": "archive-gate-a", "errors": []}

    monkeypatch.setattr(orchestrator_app, "_build_config_preview_threaded", existing_preview)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post(
            "/v1/config-preview",
            headers=_identity_headers("POST", "/v1/config-preview"),
            json={"pipeline": "archive-gate-a", "args": {"archive_command": operation, field: str(policy)}},
        )
    assert response.status_code == (200 if allowed else 403)
    assert reached == ([str(policy)] if allowed else [])


@pytest.mark.parametrize("selector", ["default", "review", "smoke"])
def test_server_selected_model_roles_remain_available_in_tenant_mode(monkeypatch, selector):
    monkeypatch.setattr(orchestrator_app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(orchestrator_app, "_record_pilot_audit", _audit_noop)

    async def existing_preview(payload, **kwargs):
        return {"pipeline": "lux-depth-v3", "selected": payload["args"]["vlmCaptioningModel"]}

    monkeypatch.setattr(orchestrator_app, "_build_config_preview_threaded", existing_preview)
    with TestClient(orchestrator_app.app, headers={"x-api-key": "contract-secret"}) as client:
        response = client.post(
            "/v1/config-preview",
            headers=_identity_headers("POST", "/v1/config-preview"),
            json={"pipeline": "lux-depth-v3", "args": {"vlmCaptioningModel": selector}},
        )
    assert response.status_code == 200
    assert response.json()["data"]["selected"] == selector
