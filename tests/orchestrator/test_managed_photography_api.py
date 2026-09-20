"""Managed V5 is a closed opt-in API path with server-owned physical bindings."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient

import app
from tests.orchestrator.test_photography_adapter import admitted, request_case  # noqa: F401
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.materials_v4.engine import ResponsePolicy
from transformation_portal.orchestrator.dispatch import DispatchLocator
from transformation_portal.orchestrator.execution_policy import server_photography_cache_root
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost
from transformation_portal.portal.photography_jobs import (
    PhotographyJobArgs,
    managed_photography_readiness,
    photography_request,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
    monkeypatch.setenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    monkeypatch.delenv("TP_LUX_V5_CACHE_DIR", raising=False)
    monkeypatch.setattr(app, "ALLOWED_INPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(app, "ALLOWED_OUTPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", False)
    return {"pipeline": "lux-depth-v5", "args": {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}}


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_python", "/tmp/untrusted"),
        ("raw_python", "/tmp/untrusted"),
        ("cache_dir", "/tmp/untrusted"),
        ("argv", ["id"]),
        ("target_size", True),
        ("target_size", 519),
        ("strength", float("nan")),
        ("strength", True),
        ("precision", "auto"),
        ("model_key", "da3-research"),
        ("device", "cuda"),
        ("preview_maps", "true"),
        ("wall_time_seconds", 0),
        ("inputDir", "/tmp/alias"),
    ],
)
def test_preview_rejects_unclosed_or_coercible_options_before_admission(configured, field, value):
    configured["args"][field] = value
    preview = app._build_config_preview(configured)
    assert any(issue["field"] == field for issue in preview["field_errors"])
    assert not Path(configured["args"]["output_dir"]).exists()


def test_preview_keeps_response_shape_and_opt_in_defaults(configured):
    from transformation_portal.api.v1 import ConfigPreviewData

    preview = app._build_config_preview(configured)
    ConfigPreviewData.model_validate(preview)
    assert not preview["field_errors"]
    assert preview["readiness"]["status"] == "ready"
    assert preview["normalized_args"]["target_size"] == 518
    assert preview["normalized_args"]["precision"] == "fp32"
    assert preview["argv_preview"] == ""


@pytest.mark.parametrize("field", ["strength", "materials_policy", "unexpected"])
def test_http_preview_serializes_rejected_overflow_values(configured, monkeypatch, field):
    monkeypatch.setattr(app, "API_KEY_SECRET", "contract-secret")
    monkeypatch.setattr(app, "RATE_LIMIT_PER_MINUTE", 0)
    sentinel = "overflow-number"
    if field == "materials_policy":
        policy = ResponsePolicy().to_payload()
        policy["operations"][0]["strength"] = sentinel
        configured["args"][field] = policy
        configured["args"]["materials_manifest"] = str(Path(configured["args"]["input_dir"]) / "materials.json")
    elif field == "unexpected":
        configured["args"][field] = {"nested": [sentinel]}
    else:
        configured["args"][field] = sentinel
    # Valid JSON syntax can overflow Python's float parser before validation.
    raw = canonicalize_json(configured).replace(canonicalize_json(sentinel), b"1e309")
    client = TestClient(app.app, headers={"x-api-key": "contract-secret"})
    response = client.post("/v1/config-preview", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 200
    body = response.json()
    assert body["schema"] == "tp.orchestrator.config_preview.v1"
    assert body["success"] is True
    preview = body["data"]
    assert any(issue["field"] == field for issue in preview["field_errors"])
    rejected = preview["normalized_args"][field]
    if field == "materials_policy":
        rejected = rejected["operations"][0]["strength"]
    elif field == "unexpected":
        rejected = rejected["nested"][0]
    assert rejected is None
    assert preview["execution_args"] == preview["normalized_args"]
    assert not Path(configured["args"]["output_dir"]).exists()


@pytest.mark.parametrize(
    "name,value",
    [
        ("TP_LUX_V5_MANAGED_ENABLED", "0"),
        ("TP_ORCHESTRATOR_STATE_BACKEND", "memory"),
        ("TP_ORCHESTRATOR_QUEUE_BACKEND", "memory"),
        ("TRANSFORMATION_PORTAL_DA3_PYTHON", "/absent/python"),
    ],
)
@pytest.mark.asyncio
async def test_disabled_or_unsupported_topology_cannot_admit(configured, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    admit = AsyncMock(side_effect=AssertionError("Blocked configuration reached admission"))
    monkeypatch.setattr(app, "_create_distributed_job", admit)
    result = await app._create_job(configured)
    assert result.status_code == 400
    admit.assert_not_called()
    assert managed_photography_readiness()["status"] == "blocked"


def test_server_only_cache_and_runtime_preserve_venv_spelling(configured, monkeypatch, tmp_path):
    python = tmp_path / "venv-python"
    python.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", str(python))
    monkeypatch.setenv("TP_LUX_V5_CACHE_DIR", str(tmp_path / "cache"))
    request = photography_request(configured["args"], tenant_id="tenant_a")
    assert request.runtime_python == str(python)
    assert request.cache_dir == tmp_path / "cache" / "tenant_a"
    assert server_photography_cache_root("tenant_b") != request.cache_dir


@pytest.mark.parametrize("field", ["companions_manifest", "materials_manifest"])
@pytest.mark.asyncio
async def test_tenant_optional_input_rejected_before_resolving_foreign_namespace(configured, monkeypatch, tmp_path, field):
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "tenants")
    monkeypatch.setattr(app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(app, "_PILOT_TENANT_MANAGER", None)
    tenant = app._pilot_tenant_manager().create_tenant("tenant_a")
    configured["args"] = {field: str(tmp_path / "tenants" / "tenant_b" / "evidence.json")}
    error = AsyncMock(return_value="rejected")
    monkeypatch.setattr(app, "_pilot_tenant_error", error)
    resolve = Path.resolve

    def guarded(path, *args, **kwargs):
        assert "tenant_b" not in path.parts, "Foreign namespace was dereferenced"
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", guarded)
    assert await app._pilot_enforce_request_paths(tenant, configured, action="job_create", request=None) == "rejected"
    assert error.call_args.kwargs["field"] == field


@pytest.mark.parametrize("field", ["companions_manifest", "materials_manifest"])
@pytest.mark.asyncio
async def test_swapped_optional_manifest_is_reauthorized_before_preparation(configured, monkeypatch, tmp_path, field):
    from transformation_portal.orchestrator import photography_adapter

    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "tenants")
    monkeypatch.setattr(app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(app, "PILOT_ALLOWED_PIPELINES", {"lux-depth-v5"})
    monkeypatch.setattr(app, "PILOT_MAX_ACTIVE_JOBS_PER_TENANT", 1)
    monkeypatch.setattr(app, "_PILOT_TENANT_MANAGER", None)
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "execution"))
    tenant = app._pilot_tenant_manager().create_tenant("tenant_a")
    own_root = tmp_path / "tenants" / "tenant_a"
    foreign_root = tmp_path / "tenants" / "tenant_b"
    own_input = own_root / "in"
    own_input.mkdir(parents=True, exist_ok=True)
    foreign_root.mkdir(parents=True, exist_ok=True)
    own_manifest = own_root / "evidence.json"
    foreign_manifest = foreign_root / "evidence.json"
    own_manifest.write_text("{}", encoding="utf-8")
    foreign_manifest.write_text("{}", encoding="utf-8")
    selector = own_root / "selected.json"
    selector.symlink_to(own_manifest)
    configured["args"] = {"input_dir": str(own_input), "output_dir": str(own_root / "out"), field: str(selector)}
    monkeypatch.setattr(app, "_pilot_tenant_from_request", AsyncMock(return_value=(tenant, None)))
    monkeypatch.setattr(app, "_record_pilot_audit", AsyncMock(return_value=None))
    monkeypatch.setattr(app, "_job_repository", lambda: object())
    monkeypatch.setattr(app, "_artifact_store", lambda: object())
    monkeypatch.setattr(app, "get_operational_record_store", lambda: object())
    prepare = Mock(side_effect=AssertionError("Foreign manifest reached preparation"))
    monkeypatch.setattr(photography_adapter, "prepare_photography_dispatch", prepare)
    preview = app._build_config_preview_threaded

    async def swap_then_preview(*args, **kwargs):
        # The HTTP entrypoint has already authorized the original tenant path.
        selector.unlink()
        selector.symlink_to(foreign_manifest)
        result = await preview(*args, **kwargs)
        assert result["execution_args"][field] == str(foreign_manifest)
        return result

    monkeypatch.setattr(app, "_build_config_preview_threaded", swap_then_preview)
    result = await app.create_job_http(request=None, payload=configured)
    assert result.status_code == 403
    assert json.loads(result.body)["error"]["details"] == {
        "field": field,
        "reason": "tenant_path_outside_workspace",
    }
    prepare.assert_not_called()
    assert not (own_root / "out").exists()
    assert not (tmp_path / "execution").exists()


def test_worker_revokes_changed_cache_binding_before_output(admitted, request_case, configured, monkeypatch):
    import hashlib

    bound = json.loads(admitted.bindings_bytes)
    if bound["cache_root"] is not None:
        monkeypatch.setenv("TP_LUX_V5_CACHE_DIR", str(Path(bound["cache_root"]).parent))
        tenant_id = Path(bound["cache_root"]).name
    else:
        tenant_id = "default"
    locator = DispatchLocator("job", "attempt", "dispatch", hashlib.sha256(admitted.plan_bytes).hexdigest(), tenant_id)
    policy = app._execution_policy()
    policy.validate_dispatch_paths(
        locator, admitted.plan_bytes, request_case.output_dir, execution_bindings=admitted.bindings_bytes
    )
    monkeypatch.setenv("TP_LUX_V5_CACHE_DIR", str(request_case.output_dir.parent / "changed-cache"))
    with pytest.raises(DispatchAuthorityLost, match="cache selection changed"):
        policy.validate_dispatch_paths(
            locator, admitted.plan_bytes, request_case.output_dir, execution_bindings=admitted.bindings_bytes
        )
    assert not request_case.output_dir.exists()


def test_public_resource_limits_match_the_governed_plan_schema():
    from importlib.resources import files

    schema = json.loads(files("transformation_portal.schemas.execution").joinpath("plan.v2.schema.json").read_text())
    public = PhotographyJobArgs.model_json_schema()["properties"]
    for section in ("configuration", "resources"):
        for field, contract in schema["properties"][section]["properties"].items():
            if field == "inference_slots":
                continue
            for limit in ("minimum", "maximum", "multipleOf", "enum", "type"):
                if limit in contract:
                    assert public[field][limit] == contract[limit], field


def test_v5_cannot_enter_legacy_raw_command_builder(configured):
    with pytest.raises(ValueError, match="immutable distributed dispatch"):
        app._argv_from_request(configured)


@pytest.mark.asyncio
async def test_readiness_exposes_v5_as_a_separate_opt_in_pipeline(configured, monkeypatch):
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "0")
    response = await app.readiness(request=None)
    payload = json.loads(response.body)
    assert payload["data"]["pipelines"]["lux-depth-v5"]["status"] == "blocked"
    assert "lux-depth-v3" in payload["data"]["pipelines"]
