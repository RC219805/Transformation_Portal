"""Signed managed access and tenant confinement for both unified workflows."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

import app
from tests.orchestrator import test_managed_lux_depth_api as unified_api
from tests.orchestrator.test_pilot_control_plane_contract import _identity_headers

pytestmark = pytest.mark.unit
configured = unified_api.configured
_ROUTES = ("/v1/config-preview", "/v1/jobs", "/v2/jobs")


@pytest.fixture
def managed(configured, monkeypatch, tmp_path):
    monkeypatch.setattr(app, "API_KEY_SECRET", "unified-access-test-key")
    monkeypatch.setattr(app, "ENFORCE_JOB_API_KEY", True)
    monkeypatch.setattr(app, "RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(app, "FRONTDOOR_IDENTITY_SECRET", "unified-frontdoor-test-identity-secret-32bytes")
    monkeypatch.setattr(app, "PILOT_ACTOR_TENANTS_JSON", '{"a@example.com":"tenant_a","b@example.com":"tenant_b"}')
    monkeypatch.setattr(app, "PILOT_ALLOWED_TENANTS", {"tenant_a"})
    monkeypatch.setattr(app, "PILOT_ALLOWED_PIPELINES", {"lux-depth"})
    monkeypatch.setattr(app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "tenants")
    monkeypatch.setattr(app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(app, "PILOT_MAX_ACTIVE_JOBS_PER_TENANT", 1)
    monkeypatch.setattr(app, "_PILOT_TENANT_MANAGER", None)
    monkeypatch.setattr(app, "_record_pilot_audit", AsyncMock(return_value=None))
    monkeypatch.setattr(app, "_job_repository", lambda: object())
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "execution"))
    own_root = tmp_path / "tenants" / "tenant_a"
    own_input = own_root / "input"
    own_input.mkdir(parents=True)
    configured["args"] = {"input_dir": str(own_input), "output_dir": str(own_root / "output")}
    return configured


@pytest.mark.parametrize("workflow", ["process", "infer"])
@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    "identity,status,reason",
    [
        ("missing", 401, "authenticated_actor_required"),
        ("unsigned", 401, "authenticated_actor_required"),
        ("wrong_target", 401, "authenticated_actor_required"),
        ("unmapped", 403, "actor_tenant_not_allowed"),
        ("tenant_denied", 403, "tenant_not_allowed"),
        ("conflicting_selector", 403, "tenant_actor_mismatch"),
    ],
)
def test_unified_authentication_and_membership_fail_before_preview(
    managed, monkeypatch, workflow, route, identity, status, reason
):
    managed["args"]["workflow"] = workflow
    preview = AsyncMock(side_effect=AssertionError("unauthorized actor reached preview"))
    monkeypatch.setattr(app, "_build_config_preview_threaded", preview)
    email = (
        "missing@example.com"
        if identity == "unmapped"
        else "b@example.com" if identity == "tenant_denied" else "a@example.com"
    )
    headers = _identity_headers("POST", "/wrong-target" if identity == "wrong_target" else route, email)
    if identity in {"missing", "unsigned"}:
        headers = {}
    if identity == "unsigned":
        headers.update({"x-tp-actor-email": "a@example.com", "x-tp-actor-role": "admin", "x-tp-tenant-id": "tenant_a"})
    if identity == "conflicting_selector":
        headers["x-tp-tenant-id"] = "tenant_b"
    response = TestClient(app.app, headers={"x-api-key": "unified-access-test-key"}).post(route, headers=headers, json=managed)
    assert response.status_code == status
    assert response.json()["error"]["details"]["reason"] == reason
    preview.assert_not_called()
    assert not Path(managed["args"]["output_dir"]).exists()


@pytest.mark.parametrize("workflow", ["process", "infer"])
@pytest.mark.parametrize("route", _ROUTES)
def test_unified_requires_its_exact_pipeline_grant(managed, monkeypatch, workflow, route):
    managed["args"]["workflow"] = workflow
    monkeypatch.setattr(app, "PILOT_ALLOWED_PIPELINES", {"lux-depth-v5", "lux-depth-v6"})
    admission = AsyncMock(side_effect=AssertionError("native pipeline grant authorized unified admission"))
    monkeypatch.setattr(app, "_create_distributed_job", admission)
    response = TestClient(app.app, headers={"x-api-key": "unified-access-test-key"}).post(
        route, headers=_identity_headers("POST", route), json=managed
    )
    assert response.status_code == 403
    assert response.json()["error"]["details"]["reason"] == "pipeline_not_allowed"
    admission.assert_not_called()
    assert not Path(managed["args"]["output_dir"]).exists()


@pytest.mark.parametrize("workflow", ["process", "infer"])
@pytest.mark.parametrize("route", _ROUTES)
def test_signed_unified_request_retains_verified_actor_tenant_and_workflow(managed, monkeypatch, workflow, route):
    managed["args"]["workflow"] = workflow
    managed["tenant_id"] = "tenant_b"
    admission = AsyncMock(return_value=app.JSONResponse({"captured": True}))
    monkeypatch.setattr(app, "_create_distributed_job", admission)
    headers = _identity_headers("POST", route)
    headers["x-tp-actor-email"] = "b@example.com"
    response = TestClient(app.app, headers={"x-api-key": "unified-access-test-key"}).post(route, headers=headers, json=managed)
    assert response.status_code == 200, response.text
    if route.endswith("config-preview"):
        assert not response.json()["data"]["field_errors"]
        assert response.json()["data"]["execution_args"]["workflow"] == workflow
    else:
        call = admission.await_args.kwargs
        assert call["pilot_tenant"].tenant_id == "tenant_a"
        assert call["portal_actor"]["accessEmail"] == "a@example.com"
        assert call["pipeline"] == "lux-depth" and call["execution_args"]["workflow"] == workflow
    assert not Path(managed["args"]["output_dir"]).exists()


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize(
    "workflow,field",
    [(workflow, field) for workflow in ("process", "infer") for field in ("input_dir", "output_dir", "companions_manifest")]
    + [("infer", "materials_manifest")],
)
def test_unified_rejects_foreign_paths_before_dereference(managed, monkeypatch, tmp_path, route, workflow, field):
    managed["args"].update(workflow=workflow, **{field: str(tmp_path / "tenants" / "tenant_b" / "private")})
    preview = AsyncMock(side_effect=AssertionError("foreign tenant path reached preview"))
    monkeypatch.setattr(app, "_build_config_preview_threaded", preview)
    original_resolve = Path.resolve

    def confined_resolve(path, *args, **kwargs):
        assert "tenant_b" not in path.parts, "foreign tenant namespace was dereferenced"
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", confined_resolve)
    response = TestClient(app.app, headers={"x-api-key": "unified-access-test-key"}).post(
        route, headers=_identity_headers("POST", route), json=managed
    )
    assert response.status_code == 403
    assert response.json()["error"]["details"] == {"field": field, "reason": "tenant_path_outside_workspace"}
    preview.assert_not_called()


@pytest.mark.parametrize("route", ["/v1/jobs", "/v2/jobs"])
@pytest.mark.parametrize(
    "workflow,field", [("process", "companions_manifest"), ("infer", "companions_manifest"), ("infer", "materials_manifest")]
)
def test_unified_reauthorizes_manifest_swap_before_preparation(managed, monkeypatch, tmp_path, route, workflow, field):
    from transformation_portal.orchestrator import lux_depth_adapter

    own_root = Path(managed["args"]["input_dir"]).parent
    foreign_root = tmp_path / "tenants" / "tenant_b"
    foreign_root.mkdir()
    own_manifest, foreign_manifest = own_root / "evidence.json", foreign_root / "evidence.json"
    own_manifest.write_text("{}", encoding="utf-8")
    foreign_manifest.write_text("{}", encoding="utf-8")
    selector = own_root / "selected.json"
    selector.symlink_to(own_manifest)
    managed["args"].update(workflow=workflow, **{field: str(selector)})
    monkeypatch.setattr(app, "_artifact_store", lambda: object())
    monkeypatch.setattr(app, "get_operational_record_store", lambda: object())
    prepare = Mock(side_effect=AssertionError("foreign manifest reached unified preparation"))
    monkeypatch.setattr(lux_depth_adapter, "prepare_unified_dispatch", prepare)
    original_preview = app._build_config_preview_threaded

    async def swap_then_preview(*args, **kwargs):
        selector.unlink()
        selector.symlink_to(foreign_manifest)
        result = await original_preview(*args, **kwargs)
        assert result["execution_args"][field] == str(foreign_manifest)
        return result

    monkeypatch.setattr(app, "_build_config_preview_threaded", swap_then_preview)
    response = TestClient(app.app, headers={"x-api-key": "unified-access-test-key"}).post(
        route, headers=_identity_headers("POST", route), json=managed
    )
    assert response.status_code == 403
    assert response.json()["error"]["details"] == {"field": field, "reason": "tenant_path_outside_workspace"}
    prepare.assert_not_called()
    assert not Path(managed["args"]["output_dir"]).exists()
    assert not (tmp_path / "execution").exists()
