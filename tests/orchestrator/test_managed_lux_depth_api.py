"""Unified HTTP admission preserves closed workflows, tenant scope, and native authority."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

import app
from transformation_portal.api.v1 import ConfigMetadataData, ConfigPreviewData
from transformation_portal.portal.lux_depth_jobs import (
    LuxDepthInferArgs,
    LuxDepthProcessArgs,
    lux_depth_request,
    managed_lux_depth_readiness,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_LUX_DEPTH_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
    monkeypatch.setenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    monkeypatch.delenv("TP_LUX_V5_CACHE_DIR", raising=False)
    monkeypatch.setattr(app, "ALLOWED_INPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(app, "ALLOWED_OUTPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", False)
    (tmp_path / "in").mkdir()
    return {"pipeline": "lux-depth", "args": {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}}


@pytest.mark.parametrize("workflow", ["process", "infer"])
def test_preview_and_request_select_exact_native_workflow(configured, workflow):
    configured["args"]["workflow"] = workflow
    preview = app._build_config_preview(configured)
    ConfigPreviewData.model_validate(preview)
    assert not preview["field_errors"]
    assert preview["pipeline"] == "lux-depth"
    assert preview["normalized_args"] == preview["execution_args"]
    assert preview["execution_args"]["workflow"] == workflow
    assert preview["argv_preview"] == ""
    assert preview["readiness"]["status"] == "ready"
    assert preview["readiness"]["canonical_command"] == f"lux-depth {workflow}"
    assert preview["next_best_action"]["label"] == "Dispatch Lux Depth Unified"
    native = lux_depth_request(preview["execution_args"], tenant_id="default")
    if workflow == "process":
        from transformation_portal.lux_depth import PhotographyRequest

        assert type(native) is PhotographyRequest
        assert native.depth_maps.refinement == "guided_bilinear_v4"
    else:
        from transformation_portal.lux_depth import InferenceRequest

        assert type(native) is InferenceRequest
        assert "exposure_stops" not in preview["execution_args"]
    assert not Path(configured["args"]["output_dir"]).exists()


def test_process_is_explicit_normalized_default(configured):
    assert app._build_config_preview(configured)["execution_args"]["workflow"] == "process"


@pytest.mark.parametrize(
    "workflow,field,value",
    [
        ("process", "workflow", "finish"),
        ("process", "workflow", "depth-pro"),
        ("process", "workflow", None),
        ("process", "workflow", ["process"]),
        ("process", "materials_manifest", "/tmp/evidence.json"),
        ("process", "materials_policy", {}),
        ("process", "depth_maps", False),
        ("process", "exposure_stops", float("nan")),
        ("process", "white_balance", [1, True, 1]),
        ("infer", "exposure_stops", 1.0),
        ("infer", "depth_refinement", "bilinear"),
        ("infer", "runtime_python", "/tmp/python"),
        ("infer", "cache_dir", "/tmp/cache"),
        ("infer", "argv", ["id"]),
        ("infer", "model_key", "depth-pro"),
        ("infer", "target_size", 519),
        ("infer", "memory_mib", True),
    ],
)
def test_invalid_workflow_controls_fail_before_side_effects(configured, workflow, field, value):
    configured["args"].update(workflow=workflow)
    configured["args"][field] = value
    preview = app._build_config_preview(configured)
    assert any(issue["field"] == field for issue in preview["field_errors"])
    json.dumps(preview, allow_nan=False)
    with pytest.raises(ValidationError):
        lux_depth_request(configured["args"], tenant_id="default")
    assert not Path(configured["args"]["output_dir"]).exists()


@pytest.mark.parametrize("workflow", ["process", "infer"])
@pytest.mark.parametrize(
    "name,value,reason",
    [
        ("TP_LUX_DEPTH_MANAGED_ENABLED", "0", "lux_depth_disabled"),
        ("TP_ORCHESTRATOR_STATE_BACKEND", "memory", "photography_dispatch_required"),
        ("TP_ORCHESTRATOR_QUEUE_BACKEND", "memory", "photography_dispatch_required"),
        ("TRANSFORMATION_PORTAL_DA3_PYTHON", "/nonexistent/python", "photography_runtime_unavailable"),
    ],
)
@pytest.mark.asyncio
async def test_unified_readiness_blocks_admission_independently(configured, monkeypatch, workflow, name, value, reason):
    configured["args"]["workflow"] = workflow
    monkeypatch.setenv(name, value)
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "1")
    admit = AsyncMock(side_effect=AssertionError("blocked Unified reached admission"))
    monkeypatch.setattr(app, "_create_distributed_job", admit)
    readiness = managed_lux_depth_readiness(workflow)
    assert readiness["status"] == "blocked"
    assert reason in {item["reason"] for item in readiness["missing_prerequisites"]}
    response = await app._create_job(configured)
    assert response.status_code == 400
    admit.assert_not_called()


def test_closed_metadata_is_available_through_authenticated_http(configured, monkeypatch):
    monkeypatch.setattr(app, "API_KEY_SECRET", "unified-contract-secret")
    monkeypatch.setattr(app, "RATE_LIMIT_PER_MINUTE", 0)
    client = TestClient(app.app)
    assert client.get("/v1/config-metadata?pipeline=lux-depth").status_code == 401
    headers = {"x-api-key": "unified-contract-secret"}
    response = client.get("/v1/config-metadata?pipeline=lux-depth", headers=headers)
    assert response.status_code == 200
    data = response.json()["data"]
    ConfigMetadataData.model_validate(data)
    assert data["default_workflow"] == "process"
    for workflow, model in (("process", LuxDepthProcessArgs), ("infer", LuxDepthInferArgs)):
        schema = data["workflows"][workflow]["args_schema"]
        assert schema == model.model_json_schema()
        assert schema["additionalProperties"] is False
        assert "runtime_python" not in schema["properties"]
    readiness = client.get("/v1/readiness", headers=headers).json()["data"]["pipelines"]
    assert readiness["lux-depth"]["status"] == "ready"
    assert readiness["lux-depth-v5"]["status"] == readiness["lux-depth-v6"]["status"] == "blocked"
    assert client.post("/v1/config-preview", json=configured).status_code == 401
    assert client.post("/v1/jobs", json=configured).status_code == 401
    assert client.post("/v1/config-preview", json=configured, headers=headers).json()["data"]["pipeline"] == "lux-depth"


def test_unified_cannot_enter_raw_command_builder(configured):
    with pytest.raises(ValueError, match="immutable distributed dispatch"):
        app._argv_from_request(configured)


@pytest.mark.parametrize("workflow", ["process", "infer"])
@pytest.mark.asyncio
async def test_admission_preserves_unified_identity_and_bound_workflow(configured, monkeypatch, tmp_path, workflow):
    from transformation_portal.orchestrator import lux_depth_adapter

    configured["args"]["workflow"] = workflow
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "execution"))
    monkeypatch.setattr(app, "PILOT_MAX_ACTIVE_JOBS_PER_TENANT", 1)
    monkeypatch.setattr(app, "_artifact_store", lambda: object())
    monkeypatch.setattr(app, "_record_pilot_audit", AsyncMock(return_value=None))
    monkeypatch.setattr(app, "_flush_operational_outbox_best_effort", AsyncMock())
    monkeypatch.setattr(app, "_cache_runtime_job", lambda _job: None)
    records = SimpleNamespace(admit=AsyncMock())
    monkeypatch.setattr(app, "get_operational_record_store", lambda: records)
    prepare = Mock(return_value=SimpleNamespace(plan_bytes=b"native-plan", bindings_bytes=b"unified-bindings"))
    monkeypatch.setattr(lux_depth_adapter, "prepare_unified_dispatch", prepare)
    result = await app._create_distributed_job(
        payload=configured,
        pipeline="lux-depth",
        execution_args=app._build_config_preview(configured)["execution_args"],
        argv=[],
        trusted_output_dir=Path(configured["args"]["output_dir"]),
        pilot_tenant=None,
        portal_actor=None,
        request=None,
        api_version="v1",
    )
    assert result.status_code == 200
    assert prepare.call_args.kwargs["workflow"] == workflow
    assert records.admit.await_args.args[1] == b"native-plan"
    assert records.admit.await_args.kwargs["execution_bindings"] == b"unified-bindings"
    recorded = records.admit.await_args.args[0]
    assert recorded.effective_request["pipeline"] == "lux-depth"
    assert recorded.effective_request["args"]["workflow"] == workflow
    assert not Path(configured["args"]["output_dir"]).exists()


def test_readiness_and_metadata_do_not_import_numerical_or_inference_runtimes():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from transformation_portal.portal.lux_depth_jobs import "
            "managed_lux_depth_readiness, lux_depth_config_metadata; "
            "managed_lux_depth_readiness(); lux_depth_config_metadata(); "
            "assert not {'torch', 'numpy', 'transformation_portal.lux_depth_v5.lifecycle', "
            "'transformation_portal.lux_depth_v6.managed'} & sys.modules.keys()",
        ],
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode()
