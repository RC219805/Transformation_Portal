"""Unified admission retains native plans and immutable route-specific authority."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace

import pytest

from tests.lux_depth_v5 import test_pipeline as controlled_pipeline
from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.dispatch import DispatchLocator
from transformation_portal.orchestrator.execution_dispatch import (
    command_from_dispatch_plan,
    execute_dispatch_plan,
    validate_dispatch_plan,
)
from transformation_portal.orchestrator.execution_policy import load_execution_policy
from transformation_portal.orchestrator.lux_depth_adapter import UnifiedBindings, prepare_unified_dispatch
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost

pytestmark = pytest.mark.unit
request_case = controlled_pipeline.request_case


@pytest.fixture(params=["process", "infer"])
def workflow(request):
    return request.param


@pytest.fixture
def admitted(request_case, workflow, monkeypatch):
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    request = ManagedLuxDepthV6Request(request_case) if workflow == "process" else request_case
    return prepare_unified_dispatch(
        request, workflow=workflow, publisher=GenerationPublisher(artifact_store=None, record_store=None)
    )


def test_unified_admission_preserves_native_bytes_and_binds_route(admitted, workflow, request_case, tmp_path):
    bound = UnifiedBindings(admitted.bindings_bytes)
    payload = bound.to_payload()
    plan = validate_dispatch_plan(admitted.plan_bytes, admitted.bindings_bytes)
    native = validate_dispatch_plan(admitted.plan_bytes, bound.photography_bindings_bytes)
    assert plan.to_payload() == native.to_payload()
    assert plan.schema == ("tp.execution.plan.v5" if workflow == "process" else "tp.execution.plan.v4")
    assert payload["pipeline"] == "lux-depth"
    assert bound.workflow == workflow
    assert payload["photography_bindings"]["input_root"] == str(request_case.input_dir)
    assert not request_case.output_dir.exists()
    command = command_from_dispatch_plan(
        admitted.plan_bytes,
        output_root=tmp_path / "attempt",
        plan_path=tmp_path / "plan.json",
        execution_bindings=admitted.bindings_bytes,
        bindings_path=tmp_path / "bindings.json",
    )
    assert command[command.index("--bindings-sha256") + 1] == hashlib.sha256(admitted.bindings_bytes).hexdigest()
    assert command[command.index("--plan-sha256") + 1] == hashlib.sha256(admitted.plan_bytes).hexdigest()


@pytest.mark.parametrize(
    "field,value",
    [
        ("pipeline", "lux-depth-v6"),
        ("workflow", "legacy"),
        ("workflow", []),
        ("argv", ["id"]),
        ("output_root", "/tmp/untrusted"),
        ("module", "os"),
        ("photography_bindings", None),
    ],
)
def test_unified_binding_schema_rejects_extra_authority(admitted, field, value):
    payload = json.loads(admitted.bindings_bytes)
    payload[field] = value
    with pytest.raises(ExecutionPlanError, match="closed"):
        validate_dispatch_plan(admitted.plan_bytes, canonicalize_json(payload))


def test_unified_workflow_cannot_select_another_native_plan(admitted, workflow):
    payload = json.loads(admitted.bindings_bytes)
    payload["workflow"] = "infer" if workflow == "process" else "process"
    with pytest.raises(ExecutionPlanError, match="plan family"):
        validate_dispatch_plan(admitted.plan_bytes, canonicalize_json(payload))


@pytest.mark.parametrize("tamper", ["plan", "strip_unified_envelope"])
def test_worker_child_rejects_carrier_replacement_before_execution(admitted, tmp_path, monkeypatch, tamper):
    from transformation_portal.orchestrator import execution_dispatch

    plan_path, bindings_path = tmp_path / "plan.json", tmp_path / "bindings.json"
    plan_path.write_bytes(admitted.plan_bytes)
    bindings_path.write_bytes(admitted.bindings_bytes)
    command = command_from_dispatch_plan(
        admitted.plan_bytes,
        output_root=tmp_path / "attempt",
        plan_path=plan_path,
        execution_bindings=admitted.bindings_bytes,
        bindings_path=bindings_path,
    )
    if tamper == "plan":
        plan_path.write_bytes(admitted.plan_bytes + b"\n")
    else:
        bindings_path.write_bytes(UnifiedBindings(admitted.bindings_bytes).photography_bindings_bytes)
    monkeypatch.setattr(sys, "argv", command[2:])
    monkeypatch.setattr(
        execution_dispatch, "execute_dispatch_plan", lambda *_a, **_kw: pytest.fail("executed changed carrier")
    )
    with pytest.raises(ExecutionPlanError, match="carrier digest does not match the claim"):
        execution_dispatch.main()
    assert not (tmp_path / "attempt").exists()


@pytest.mark.parametrize("invalid", ["noncanonical", "oversized", "inner_authority"])
def test_unified_bindings_require_bounded_canonical_inner_authority(admitted, invalid):
    raw = admitted.bindings_bytes
    if invalid == "noncanonical":
        raw += b"\n"
    elif invalid == "oversized":
        raw += b" " * 65536
    else:
        payload = json.loads(raw)
        payload["photography_bindings"]["argv"] = ["id"]
        raw = canonicalize_json(payload)
    with pytest.raises(ExecutionPlanError):
        validate_dispatch_plan(admitted.plan_bytes, raw)


@pytest.mark.parametrize("workflow", ["process", "infer", "finish", "depth-pro", "legacy"])
def test_unified_admission_rejects_wrong_carriers_before_preparation(request_case, workflow, monkeypatch):
    from transformation_portal.lux_depth import lifecycle

    monkeypatch.setattr(lifecycle, "prepare", lambda *_a, **_kw: pytest.fail("unexpected plan preparation"))
    request = request_case if workflow == "process" else ManagedLuxDepthV6Request(request_case)
    with pytest.raises((TypeError, ExecutionPlanError)):
        prepare_unified_dispatch(
            request, workflow=workflow, publisher=GenerationPublisher(artifact_store=None, record_store=None)
        )


def test_unified_dispatch_runs_exact_native_plan_without_repreparing(admitted, workflow, tmp_path, monkeypatch):
    from transformation_portal.lux_depth import lifecycle

    monkeypatch.setattr(lifecycle, "prepare", lambda *_a, **_kw: pytest.fail("worker re-prepared admitted intent"))
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "private"))
    output = tmp_path / "attempt"
    assert execute_dispatch_plan(admitted.plan_bytes, execution_bindings=admitted.bindings_bytes, output_root=output) == 0
    assert (output / "execution-plan.json").read_bytes() == admitted.plan_bytes
    verified = lifecycle.verify(output, expected_plan_sha256=hashlib.sha256(admitted.plan_bytes).hexdigest())
    assert verified.output_root == output
    prefix = "v6/" if workflow == "process" else ""
    assert (output / prefix / "input-0000/delivery.tif").is_file()
    assert (output / prefix / "input-0000/preview.png").is_file()
    assert not list((tmp_path / "private").iterdir())


def test_unified_worker_feature_gate_cannot_borrow_native_grants(admitted, request_case, tmp_path, monkeypatch):
    monkeypatch.setenv("TP_ALLOWED_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("TP_ALLOWED_OUTPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("TP_PILOT_CONTROL_PLANE_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_DEPTH_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V5_CACHE_DIR", str(request_case.cache_dir.parent))
    locator = DispatchLocator("job_unified", "attempt", "dispatch", hashlib.sha256(admitted.plan_bytes).hexdigest(), "cache")
    policy = load_execution_policy()
    policy.validate_dispatch_paths(
        locator, admitted.plan_bytes, tmp_path / "attempt", execution_bindings=admitted.bindings_bytes
    )
    native = UnifiedBindings(admitted.bindings_bytes).photography_bindings_bytes
    with pytest.raises(DispatchAuthorityLost, match="disabled"):
        policy.validate_dispatch_paths(locator, admitted.plan_bytes, tmp_path / "attempt", execution_bindings=native)
    monkeypatch.setenv("TP_LUX_DEPTH_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "1")
    with pytest.raises(DispatchAuthorityLost, match="disabled"):
        policy.validate_dispatch_paths(
            locator, admitted.plan_bytes, tmp_path / "attempt", execution_bindings=admitted.bindings_bytes
        )
    assert not (tmp_path / "attempt").exists()


@pytest.mark.parametrize("revoke", ["pipeline", "tenant"])
def test_unified_worker_rechecks_exact_tenant_pipeline_grant(admitted, tmp_path, monkeypatch, revoke):
    monkeypatch.setenv("TP_LUX_DEPTH_MANAGED_ENABLED", "1")
    locator = DispatchLocator("job_unified", "attempt", "dispatch", hashlib.sha256(admitted.plan_bytes).hexdigest(), "tenant")
    policy = replace(
        load_execution_policy(),
        pilot_enabled=True,
        allowed_pipelines=frozenset({"lux-depth-v5", "lux-depth-v6"} if revoke == "pipeline" else {"lux-depth"}),
        allowed_tenants=frozenset({"other" if revoke == "tenant" else "tenant"}),
    )
    with pytest.raises(DispatchAuthorityLost, match="no longer allowed"):
        policy.validate_dispatch_paths(
            locator, admitted.plan_bytes, tmp_path / "attempt", execution_bindings=admitted.bindings_bytes
        )
    assert not (tmp_path / "attempt").exists()
