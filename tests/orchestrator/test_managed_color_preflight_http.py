"""Authenticated photography routes authorize bounded color inspection."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageCms

import app
from tests.orchestrator import test_pilot_control_plane_contract as pilot_contract
from transformation_portal.lux_depth_v4 import input_preflight

pytestmark = pytest.mark.unit
_reset_pilot_state = pilot_contract._reset_pilot_state

_ROUTES = ["/v1/config-preview", "/v1/jobs", "/v2/jobs"]
_WORKFLOWS = [("lux-depth-v5", None), ("lux-depth-v6", None), ("lux-depth", "process"), ("lux-depth", "infer")]


@pytest.fixture
def configured_pilot(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(app, "PILOT_ALLOWED_PIPELINES", {"lux-depth", "lux-depth-v5", "lux-depth-v6"})
    monkeypatch.setattr(app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "workspaces")
    monkeypatch.setattr(app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(app, "ALLOWED_INPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(app, "ALLOWED_OUTPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(app, "_record_pilot_audit", pilot_contract._audit_noop)
    admission = AsyncMock(side_effect=AssertionError("Invalid color reached expensive dispatch preparation"))
    monkeypatch.setattr(app, "_create_distributed_job", admission)
    tenant = app._pilot_tenant_manager().create_tenant("tenant_a", policy=app._pilot_tenant_policy())
    source = tenant.tenant_workspace / "input"
    source.mkdir(parents=True)
    output = tenant.tenant_workspace / "output"
    # A valid non-sRGB ICC fixture exercises the same unsupported-profile
    # rejection as Adobe RGB without depending on a host-installed profile.
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("LAB")).tobytes()
    Image.new("RGB", (14, 14)).save(source / "photo.jpg", icc_profile=profile)
    return {"input_dir": str(source), "output_dir": str(output)}, admission


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("pipeline, workflow", _WORKFLOWS)
def test_authenticated_color_preflight_rejects_unsupported_profile_before_preparation(
    configured_pilot, route, pipeline, workflow
):
    args, admission = configured_pilot
    if workflow is not None:
        args["workflow"] = workflow
    # Omitting the lifespan keeps this HTTP boundary test independent of live
    # Postgres/Redis; rejection occurs before either service is needed.
    client = TestClient(app.app, headers={"x-api-key": "contract-secret"})
    response = client.post(
        route, json={"pipeline": pipeline, "args": args}, headers=pilot_contract._identity_headers("POST", route)
    )

    body = response.json()
    if route == "/v1/config-preview":
        assert response.status_code == 200
        assert body["success"] is True
        assert body["schema"] == "tp.orchestrator.config_preview.v1"
        error = body["data"]["field_errors"][0]
        assert error["field"] == "input_color"
        assert error["code"] == "input_color_unsupported_icc"
        assert body["data"]["normalized_args"]["input_color"] == "auto"
    else:
        assert response.status_code == 400
        assert body["success"] is False
        error = body["error"]
        assert error["code"] == "INVALID_ARGUMENT"
        assert error["details"] == {"field": "input_color", "reason": "input_color_unsupported_icc"}
    assert "Convert the photographs to sRGB" in error["message"]
    assert "does not convert" in error["message"]
    admission.assert_not_called()
    assert not Path(args["output_dir"]).exists()


@pytest.mark.parametrize("route", _ROUTES)
@pytest.mark.parametrize("pipeline, workflow", _WORKFLOWS)
def test_foreign_tenant_input_is_rejected_before_color_content_probe(
    configured_pilot, monkeypatch, tmp_path, route, pipeline, workflow
):
    args, admission = configured_pilot
    args["input_dir"] = str(tmp_path / "workspaces" / "tenant_b" / "input")
    if workflow is not None:
        args["workflow"] = workflow
    monkeypatch.setattr(
        input_preflight, "validate_input_directory_colors", lambda *_args, **_kwargs: pytest.fail("Read foreign input")
    )
    client = TestClient(app.app, headers={"x-api-key": "contract-secret"})
    response = client.post(
        route, json={"pipeline": pipeline, "args": args}, headers=pilot_contract._identity_headers("POST", route)
    )

    assert response.status_code == 403
    assert response.json()["error"]["details"]["reason"] == "tenant_path_outside_workspace"
    admission.assert_not_called()
    assert not Path(args["output_dir"]).exists()


@pytest.mark.parametrize("route", _ROUTES)
def test_missing_actor_cannot_authorize_color_content_probe(configured_pilot, monkeypatch, route):
    args, admission = configured_pilot
    monkeypatch.setattr(
        input_preflight, "validate_input_directory_colors", lambda *_args, **_kwargs: pytest.fail("Read unauthenticated input")
    )
    client = TestClient(app.app, headers={"x-api-key": "contract-secret"})
    response = client.post(route, json={"pipeline": "lux-depth", "args": args})
    assert response.status_code == 401
    assert response.json()["error"]["details"]["reason"] == "authenticated_actor_required"
    admission.assert_not_called()
