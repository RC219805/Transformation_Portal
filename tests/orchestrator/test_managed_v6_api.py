"""Managed V6 preserves HTTP and tenant guards while freezing every finishing control."""

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
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.portal.photography_v6_jobs import PhotographyV6JobArgs, managed_v6_readiness, v6_request

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("version", ["V5", "V6"])
def test_ready_photography_preview_names_its_pipeline(version):
    action = app._preview_next_best_action(
        pipeline=f"lux-depth-{version.lower()}", errors=[], warnings=[], readiness_snapshot={"status": "ready"}
    )
    assert action["action"] == "dispatch_ready"
    assert action["label"] == f"Dispatch LuxDepth{version} photography"


@pytest.fixture
def configured(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
    monkeypatch.setenv("TP_ORCHESTRATOR_QUEUE_BACKEND", "redis")
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    monkeypatch.delenv("TP_LUX_V5_CACHE_DIR", raising=False)
    monkeypatch.setattr(app, "ALLOWED_INPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(app, "ALLOWED_OUTPUT_ROOTS", [tmp_path])
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", False)
    (tmp_path / "in").mkdir()
    return {"pipeline": "lux-depth-v6", "args": {"input_dir": str(tmp_path / "in"), "output_dir": str(tmp_path / "out")}}


@pytest.mark.parametrize(
    "field,value",
    [
        ("runtime_python", "/tmp/untrusted"),
        ("cache_dir", "/tmp/untrusted"),
        ("argv", ["id"]),
        ("depth_maps", False),
        ("max_pixels", 100_000_001),
        ("target_size", 519),
        ("exposure_stops", True),
        ("exposure_stops", 8.1),
        ("exposure_stops", float("nan")),
        ("white_balance", [1, 1]),
        ("white_balance", [1, 1, True]),
        ("white_balance", [1, 1, float("inf")]),
        ("white_balance", [1, 1, 4.1]),
        ("contrast", 0.24),
        ("pivot", 0),
        ("saturation", 2.1),
        ("render", "auto"),
        ("shoulder", 0.96),
        ("depth_refinement", "guided_bilinear"),
        ("materials_manifest", "/tmp/materials.json"),
        ("materials_policy", {}),
    ],
)
def test_preview_rejects_invalid_controls_without_side_effects(configured, field, value):
    configured["args"][field] = value
    preview = app._build_config_preview(configured)
    assert any(issue["field"] == field for issue in preview["field_errors"])
    with pytest.raises(ValidationError):
        v6_request(configured["args"], tenant_id="default")
    assert not Path(configured["args"]["output_dir"]).exists()


def test_preview_has_closed_v6_defaults_and_independent_readiness(configured):
    from transformation_portal.api.v1 import ConfigPreviewData

    preview = app._build_config_preview(configured)
    ConfigPreviewData.model_validate(preview)
    assert preview["pipeline"] == "lux-depth-v6"
    assert not preview["field_errors"]
    assert preview["readiness"]["status"] == "ready"
    assert preview["readiness"]["runner_details"]["plan_schema"] == "tp.execution.plan.v5"
    assert preview["normalized_args"] == preview["execution_args"]
    assert preview["normalized_args"]["white_balance"] == [1.0, 1.0, 1.0]
    assert preview["normalized_args"]["render"] == "perceptual_srgb"
    assert preview["normalized_args"]["depth_refinement"] == "guided_bilinear_v4"
    assert preview["argv_preview"] == ""
    assert preview["next_best_action"]["label"] == "Dispatch LuxDepthV6 photography"
    assert PhotographyV6JobArgs.model_json_schema()["additionalProperties"] is False


@pytest.mark.parametrize("field", ["input_dir", "companions_manifest"])
def test_preview_preserves_path_guards_without_reading_media(configured, tmp_path, monkeypatch, field):
    configured["args"][field] = str(tmp_path / "absent")
    monkeypatch.setattr(Path, "read_bytes", lambda _path: pytest.fail("Preview read file contents"))
    preview = app._build_config_preview(configured)
    assert any(issue["field"] == field for issue in preview["field_errors"])
    assert not Path(configured["args"]["output_dir"]).exists()


def test_readiness_does_not_import_inference_or_finishing_runtime():
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from transformation_portal.portal.photography_v6_jobs import managed_v6_readiness; "
            "managed_v6_readiness(); "
            "assert 'transformation_portal.lux_depth_v5.lifecycle' not in sys.modules; "
            "assert 'transformation_portal.lux_depth_v6.managed' not in sys.modules; "
            "assert 'transformation_portal.materials_v4.engine' not in sys.modules",
        ],
        env={**os.environ, "PYTHONPATH": str(repo / "src")},
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr.decode()


@pytest.mark.parametrize("field", ["exposure_stops", "white_balance", "unexpected"])
def test_http_rejected_nonfinite_values_remain_json_serializable(configured, monkeypatch, field):
    monkeypatch.setattr(app, "API_KEY_SECRET", "contract-secret")
    monkeypatch.setattr(app, "RATE_LIMIT_PER_MINUTE", 0)
    sentinel = "overflow-number"
    configured["args"][field] = [1, sentinel, 1] if field == "white_balance" else sentinel
    if field == "unexpected":
        configured["args"][field] = {"nested": [sentinel]}
    raw = canonicalize_json(configured).replace(canonicalize_json(sentinel), b"1e309")
    client = TestClient(app.app, headers={"x-api-key": "contract-secret"})
    response = client.post("/v1/config-preview", content=raw, headers={"content-type": "application/json"})
    assert response.status_code == 200
    preview = response.json()["data"]
    assert any(issue["field"] == field for issue in preview["field_errors"])
    rejected = preview["normalized_args"][field]
    if field == "white_balance":
        rejected = rejected[1]
    elif field == "unexpected":
        rejected = rejected["nested"][0]
    assert rejected is None


@pytest.mark.parametrize(
    "name,value",
    [
        ("TP_LUX_V6_MANAGED_ENABLED", "0"),
        ("TP_ORCHESTRATOR_STATE_BACKEND", "memory"),
        ("TP_ORCHESTRATOR_QUEUE_BACKEND", "memory"),
        ("TRANSFORMATION_PORTAL_DA3_PYTHON", "/absent/python"),
    ],
)
@pytest.mark.asyncio
async def test_blocked_readiness_cannot_admit(configured, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    admit = AsyncMock(side_effect=AssertionError("Blocked request reached admission"))
    monkeypatch.setattr(app, "_create_distributed_job", admit)
    result = await app._create_job(configured)
    assert result.status_code == 400
    admit.assert_not_called()
    assert managed_v6_readiness()["status"] == "blocked"


@pytest.mark.parametrize("field,value", [("materials_manifest", "/tmp/materials.json"), ("materials_policy", {})])
@pytest.mark.asyncio
async def test_materials_cannot_reach_admission(configured, monkeypatch, field, value):
    configured["args"][field] = value
    admit = AsyncMock(side_effect=AssertionError("Materials reached admission"))
    monkeypatch.setattr(app, "_create_distributed_job", admit)
    result = await app._create_job(configured)
    assert result.status_code == 400
    admit.assert_not_called()


@pytest.mark.parametrize("render", ["perceptual_srgb", "soft_srgb", "clip_srgb"])
@pytest.mark.parametrize("depth_refinement", ["guided_bilinear_v4", "guided_bilinear_v3", "bilinear"])
def test_typed_request_preserves_every_control_and_server_owned_bindings(
    configured, monkeypatch, tmp_path, render, depth_refinement
):
    python = tmp_path / "venv-python"
    python.symlink_to(Path(sys.executable).resolve())
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", str(python))
    monkeypatch.setenv("TP_LUX_V5_CACHE_DIR", str(tmp_path / "cache"))
    configured["args"].update(
        exposure_stops=1.25,
        white_balance=[0.75, 1, 1.5],
        contrast=1.2,
        pivot=0.2,
        saturation=0.8,
        render=render,
        shoulder=0.7,
        depth_refinement=depth_refinement,
        input_color="srgb",
        strength=0.5,
        clarity=0.1,
        target_size=1008,
    )
    request = v6_request(configured["args"], tenant_id="tenant_a")
    assert request.inference.runtime_python == str(python)
    assert request.inference.cache_dir == tmp_path / "cache" / "tenant_a"
    assert request.inference.input_color == "srgb"
    assert request.inference.target_size == 1008
    assert request.inference.strength == 0.5
    assert request.inference.clarity == 0.1
    assert request.grade.to_payload() == {
        "schema": "tp.lux.grade.v1",
        "exposure_stops": 1.25,
        "white_balance": [0.75, 1.0, 1.5],
        "contrast": 1.2,
        "pivot": 0.2,
        "saturation": 0.8,
    }
    assert request.render.mode == render
    assert request.render.shoulder == 0.7
    assert request.depth_maps.refinement == depth_refinement


def test_v6_cannot_enter_legacy_raw_command_builder(configured):
    with pytest.raises(ValueError, match="immutable distributed dispatch"):
        app._argv_from_request(configured)


@pytest.mark.asyncio
async def test_readiness_keeps_v3_v5_and_independent_v6(configured):
    payload = json.loads((await app.readiness(request=None)).body)
    pipelines = payload["data"]["pipelines"]
    assert pipelines["lux-depth-v6"]["status"] == "ready"
    assert pipelines["lux-depth-v5"]["status"] == "blocked"
    assert "lux-depth-v3" in pipelines


@pytest.mark.asyncio
async def test_admission_uses_only_typed_v6_plan_and_bindings(configured, monkeypatch, tmp_path):
    from transformation_portal.orchestrator import photography_adapter, photography_v6_adapter

    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "execution"))
    monkeypatch.setattr(app, "PILOT_MAX_ACTIVE_JOBS_PER_TENANT", 1)
    monkeypatch.setattr(app, "_artifact_store", lambda: object())
    monkeypatch.setattr(app, "_record_pilot_audit", AsyncMock(return_value=None))
    monkeypatch.setattr(app, "_flush_operational_outbox_best_effort", AsyncMock())
    monkeypatch.setattr(app, "_cache_runtime_job", lambda _job: None)
    records = SimpleNamespace(admit=AsyncMock())
    monkeypatch.setattr(app, "get_operational_record_store", lambda: records)
    prepare = Mock(return_value=SimpleNamespace(plan_bytes=b"canonical-v6", bindings_bytes=b"physical-bindings"))
    monkeypatch.setattr(photography_v6_adapter, "prepare_v6_dispatch", prepare)
    monkeypatch.setattr(
        photography_adapter, "prepare_photography_dispatch", Mock(side_effect=AssertionError("V5 adapter called"))
    )
    result = await app._create_distributed_job(
        payload=configured,
        pipeline="lux-depth-v6",
        execution_args=app._build_config_preview(configured)["execution_args"],
        argv=[],
        trusted_output_dir=Path(configured["args"]["output_dir"]),
        pilot_tenant=None,
        portal_actor=None,
        request=None,
        api_version="v1",
    )
    assert result.status_code == 200
    assert prepare.call_args.args[0].depth_maps.refinement == "guided_bilinear_v4"
    assert records.admit.await_args.args[1] == b"canonical-v6"
    assert records.admit.await_args.kwargs["execution_bindings"] == b"physical-bindings"
    assert not Path(configured["args"]["output_dir"]).exists()


@pytest.mark.asyncio
async def test_tenant_manifest_swap_is_reauthorized_before_v6_preparation(configured, monkeypatch, tmp_path):
    from transformation_portal.orchestrator import photography_v6_adapter

    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", True)
    monkeypatch.setattr(app, "PILOT_TENANT_WORKSPACE_ROOT", tmp_path / "tenants")
    monkeypatch.setattr(app, "PILOT_TENANT_CAS_ROOT", tmp_path / "cas")
    monkeypatch.setattr(app, "PILOT_ALLOWED_PIPELINES", {"lux-depth-v6"})
    monkeypatch.setattr(app, "PILOT_MAX_ACTIVE_JOBS_PER_TENANT", 1)
    monkeypatch.setattr(app, "_PILOT_TENANT_MANAGER", None)
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "execution"))
    tenant = app._pilot_tenant_manager().create_tenant("tenant_a")
    own_root = tmp_path / "tenants" / "tenant_a"
    foreign_root = tmp_path / "tenants" / "tenant_b"
    own_input = own_root / "in"
    own_input.mkdir(parents=True, exist_ok=True)
    foreign_root.mkdir(parents=True, exist_ok=True)
    own_manifest = own_root / "companions.json"
    foreign_manifest = foreign_root / "companions.json"
    own_manifest.write_text("{}", encoding="utf-8")
    foreign_manifest.write_text("{}", encoding="utf-8")
    selector = own_root / "selected.json"
    selector.symlink_to(own_manifest)
    configured["args"] = {
        "input_dir": str(own_input),
        "output_dir": str(own_root / "out"),
        "companions_manifest": str(selector),
    }
    monkeypatch.setattr(app, "_pilot_tenant_from_request", AsyncMock(return_value=(tenant, None)))
    monkeypatch.setattr(app, "_record_pilot_audit", AsyncMock(return_value=None))
    monkeypatch.setattr(app, "_job_repository", lambda: object())
    monkeypatch.setattr(app, "_artifact_store", lambda: object())
    monkeypatch.setattr(app, "get_operational_record_store", lambda: object())
    prepare = Mock(side_effect=AssertionError("Foreign manifest reached preparation"))
    monkeypatch.setattr(photography_v6_adapter, "prepare_v6_dispatch", prepare)
    preview = app._build_config_preview_threaded

    async def swap_then_preview(*args, **kwargs):
        selector.unlink()
        selector.symlink_to(foreign_manifest)
        result = await preview(*args, **kwargs)
        assert result["execution_args"]["companions_manifest"] == str(foreign_manifest)
        return result

    monkeypatch.setattr(app, "_build_config_preview_threaded", swap_then_preview)
    result = await app.create_job_http(request=None, payload=configured)
    assert result.status_code == 403
    assert json.loads(result.body)["error"]["details"] == {
        "field": "companions_manifest",
        "reason": "tenant_path_outside_workspace",
    }
    prepare.assert_not_called()
    assert not (own_root / "out").exists()
    assert not (tmp_path / "execution").exists()
