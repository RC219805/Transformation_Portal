"""Frozen raw-photo V6 dispatch, independent feature policy and portal projection."""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from tests.lux_depth_v5 import test_pipeline as controlled_pipeline
from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v6.evidence import verify_execution_evidence
from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request
from transformation_portal.orchestrator.artifact_store.base import ArtifactStoreError
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.dispatch import DispatchFence, DispatchLocator
from transformation_portal.orchestrator.execution_dispatch import (
    command_from_dispatch_plan,
    execute_dispatch_plan,
    validate_dispatch_plan,
)
from transformation_portal.orchestrator.execution_policy import load_execution_policy
from transformation_portal.orchestrator.photography_adapter import PhotographyBindings
from transformation_portal.orchestrator.photography_v6_adapter import (
    ManagedV6PhotographyPublisher,
    consume_v6_dispatch,
    prepare_v6_dispatch,
)
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost

pytestmark = pytest.mark.unit
request_case = controlled_pipeline.request_case


@pytest.fixture
def admitted(request_case, monkeypatch):
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    monkeypatch.delenv("TRANSFORMATION_PORTAL_RAW_PYTHON", raising=False)
    return prepare_v6_dispatch(
        ManagedLuxDepthV6Request(request_case), publisher=GenerationPublisher(artifact_store=None, record_store=None)
    )


def test_v6_admission_freezes_both_stages_without_embedding_physical_paths(admitted, request_case):
    plan = validate_dispatch_plan(admitted.plan_bytes, admitted.bindings_bytes)
    bindings = PhotographyBindings(admitted.bindings_bytes).to_payload()
    assert plan.schema == "tp.execution.plan.v5"
    assert plan.to_payload()["pipeline"] == "lux_depth_v6"
    assert plan.to_payload()["inference"]["schema"] == "tp.execution.plan.v4"
    assert plan.plan_fingerprint_sha256 != hashlib.sha256(admitted.plan_bytes).hexdigest()
    assert bindings["input_root"] == str(request_case.input_dir)
    assert bindings["runtime_python"] == sys.executable
    assert str(request_case.input_dir).encode() not in admitted.plan_bytes
    assert not request_case.output_dir.exists()
    assert len(admitted.plan_bytes) <= 1024**2


@pytest.mark.parametrize("field,value", [("argv", ["id"]), ("module", "os"), ("output_root", "/tmp/untrusted")])
def test_v6_bindings_cannot_gain_executable_or_output_authority(admitted, field, value):
    payload = json.loads(admitted.bindings_bytes)
    payload[field] = value
    with pytest.raises(ExecutionPlanError, match="closed"):
        validate_dispatch_plan(admitted.plan_bytes, canonicalize_json(payload))


def test_v6_requires_exact_canonical_bindings_and_plan(admitted):
    with pytest.raises(ExecutionPlanError, match="immutable photography bindings"):
        validate_dispatch_plan(admitted.plan_bytes)
    with pytest.raises(ExecutionPlanError, match="canonical"):
        validate_dispatch_plan(admitted.plan_bytes + b"\n", admitted.bindings_bytes)
    with pytest.raises(ExecutionPlanError, match="closed"):
        validate_dispatch_plan(admitted.plan_bytes, admitted.bindings_bytes + b"\n")


def test_v6_worker_hydrates_without_input_discovery_or_preparation(admitted, tmp_path, monkeypatch):
    from transformation_portal.lux_depth_v5 import lifecycle
    from transformation_portal.lux_depth_v6 import managed

    monkeypatch.setattr(lifecycle, "prepare", lambda *_a, **_kw: pytest.fail("worker rediscovered inputs"))
    monkeypatch.setattr(managed, "prepare", lambda *_a, **_kw: pytest.fail("worker rebuilt V6 policy"))
    prepared = consume_v6_dispatch(admitted.plan_bytes, admitted.bindings_bytes, execution_root=tmp_path / "attempt")
    assert prepared.plan.canonical_bytes == admitted.plan_bytes
    assert prepared.output_root == tmp_path / "attempt"
    assert prepared.inference.output_root == tmp_path / "attempt/source-v5"
    assert prepared.inference.plan.to_payload() == json.loads(admitted.plan_bytes)["inference"]
    assert not prepared.output_root.exists()


def test_v6_worker_rejects_changed_server_runtime_before_output(admitted, tmp_path, monkeypatch):
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", "/tmp/different-python")
    with pytest.raises(ExecutionPlanError, match="current server policy"):
        consume_v6_dispatch(admitted.plan_bytes, admitted.bindings_bytes, execution_root=tmp_path / "attempt")
    assert not (tmp_path / "attempt").exists()


def test_v6_admission_rejects_client_runtime_selection(request_case, monkeypatch):
    monkeypatch.setenv("TRANSFORMATION_PORTAL_DA3_PYTHON", sys.executable)
    with pytest.raises(ExecutionPlanError, match="selected by the server"):
        prepare_v6_dispatch(
            ManagedLuxDepthV6Request(replace(request_case, runtime_python="/tmp/untrusted-python")),
            publisher=GenerationPublisher(artifact_store=None, record_store=None),
        )


def test_v6_dispatch_exports_both_independently_verifiable_generations(admitted, tmp_path, monkeypatch):
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(tmp_path / "private"))
    output = tmp_path / "attempt"
    assert execute_dispatch_plan(admitted.plan_bytes, execution_bindings=admitted.bindings_bytes, output_root=output) == 0
    assert (output / "execution-plan.json").read_bytes() == admitted.plan_bytes
    assert (output / "source-v5/execution-evidence.json").is_file()
    assert (output / "v6/input-0000/delivery.tif").is_file()
    assert (output / "v6/input-0000/preview.png").is_file()
    assert (output / "v6/input-0000/depth-relative.tif").is_file()
    verify_execution_evidence(output / "v6", source_root=output / "source-v5")
    assert not list((tmp_path / "private").iterdir())


def test_v6_fixed_command_keeps_plan_and_bindings_digest_fences(admitted, tmp_path):
    command = command_from_dispatch_plan(
        admitted.plan_bytes,
        output_root=tmp_path / "attempt",
        plan_path=tmp_path / "plan.json",
        execution_bindings=admitted.bindings_bytes,
        bindings_path=tmp_path / "bindings.json",
    )
    assert command[:3] == [sys.executable, "-m", "transformation_portal.orchestrator.execution_dispatch"]
    assert command[command.index("--plan-sha256") + 1] == hashlib.sha256(admitted.plan_bytes).hexdigest()
    assert command[command.index("--bindings-sha256") + 1] == hashlib.sha256(admitted.bindings_bytes).hexdigest()


def test_v6_policy_flag_is_independent_of_v5_and_revocable(admitted, request_case, tmp_path, monkeypatch):
    monkeypatch.setenv("TP_ALLOWED_INPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("TP_ALLOWED_OUTPUT_ROOTS", str(tmp_path))
    monkeypatch.setenv("TP_PILOT_CONTROL_PLANE_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V5_CACHE_DIR", str(request_case.cache_dir.parent))
    locator = DispatchLocator("job_v6", "attempt", "dispatch", hashlib.sha256(admitted.plan_bytes).hexdigest(), "cache")
    policy = load_execution_policy()
    policy.validate_dispatch_paths(
        locator, admitted.plan_bytes, tmp_path / "attempt", execution_bindings=admitted.bindings_bytes
    )
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "1")
    with pytest.raises(DispatchAuthorityLost, match="disabled"):
        policy.validate_dispatch_paths(
            locator, admitted.plan_bytes, tmp_path / "attempt", execution_bindings=admitted.bindings_bytes
        )


@pytest.mark.parametrize("revoke", ["pipeline", "tenant"])
def test_v6_policy_requires_its_own_tenant_pipeline_authorization(admitted, tmp_path, monkeypatch, revoke):
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "1")
    locator = DispatchLocator("job_v6", "attempt", "dispatch", hashlib.sha256(admitted.plan_bytes).hexdigest(), "tenant")
    policy = replace(
        load_execution_policy(),
        pilot_enabled=True,
        allowed_pipelines=frozenset({"lux-depth-v5" if revoke == "pipeline" else "lux-depth-v6"}),
        allowed_tenants=frozenset({"other" if revoke == "tenant" else "tenant"}),
    )
    with pytest.raises(DispatchAuthorityLost, match="no longer allowed"):
        policy.validate_dispatch_paths(
            locator, admitted.plan_bytes, tmp_path / "attempt", execution_bindings=admitted.bindings_bytes
        )
    assert not (tmp_path / "attempt").exists()


@pytest.mark.asyncio
async def test_v6_projection_prioritizes_photographs_and_keeps_depth_separate(tmp_path, monkeypatch):
    async def project_only(_publisher, _fence, _files, **kwargs):
        return kwargs["artifacts"]

    monkeypatch.setattr(GenerationPublisher, "publish", project_only)
    paths = [
        "execution-plan.json",
        "execution-evidence.json",
        "source-v5/execution-evidence.json",
        "source-v5/input-0000/delivery.tif",
        "source-v5/input-0000/preview.png",
        "source-v5/input-0000/native-depth.npy",
        "source-v5/input-0001/delivery.tif",
        "v6/input-0000/delivery.tif",
        "v6/input-0000/preview.png",
        "v6/input-0000/depth-relative.tif",
        "v6/input-0000/depth-preview.png",
        "v6/input-0000/depth-preview-valid.png",
        "v6/input-0001/delivery.tif",
    ]
    fence = DispatchFence(
        DispatchLocator("job_v6", "attempt", "dispatch", "a" * 64, "tenant"), "worker", 1, 100, str(tmp_path), str(tmp_path)
    )
    projected = await ManagedV6PhotographyPublisher(artifact_store=None, record_store=None).publish(
        fence,
        {path: tmp_path / path for path in paths},
        state="succeeded",
        exit_code=0,
        artifacts={"schema": "tp.lux.delivery.v4", "paths": paths, "execution_evidence": "execution-evidence.json"},
        run_summary={},
        expected_file_integrity={path: {"size_bytes": 1, "sha256": "a" * 64} for path in paths},
    )
    items = {item["path"]: item for item in projected["items"]}
    photograph = items["v6/input-0000/delivery.tif"]
    depth = items["v6/input-0000/depth-relative.tif"]
    assert photograph["preview_url"].endswith("/v6/input-0000/preview.png")
    assert depth["preview_url"].endswith("/v6/input-0000/depth-preview.png")
    assert photograph["display_hint"]["priority"] > depth["display_hint"]["priority"]
    assert not photograph["browser_previewable"]
    assert "preview_url" not in items["v6/input-0001/delivery.tif"]
    assert "preview_url" not in items["source-v5/input-0000/delivery.tif"]
    assert items["source-v5/input-0000/delivery.tif"]["display_hint"]["priority"] < depth["display_hint"]["priority"]
    # Exact explicit groups override filename/extension heuristics in Review.
    expected_groups = {
        "source-v5/execution-evidence.json": "lux-depth-v6|source-v5",
        "source-v5/input-0000/delivery.tif": "lux-depth-v6|input-0000|source-v5",
        "source-v5/input-0000/preview.png": "lux-depth-v6|input-0000|source-v5",
        "source-v5/input-0000/native-depth.npy": "lux-depth-v6|input-0000|source-v5",
        "source-v5/input-0001/delivery.tif": "lux-depth-v6|input-0001|source-v5",
        "v6/input-0000/delivery.tif": "lux-depth-v6|input-0000|photograph",
        "v6/input-0000/preview.png": "lux-depth-v6|input-0000|photograph",
        "v6/input-0000/depth-relative.tif": "lux-depth-v6|input-0000|relative-depth",
        "v6/input-0000/depth-preview.png": "lux-depth-v6|input-0000|relative-depth",
        "v6/input-0000/depth-preview-valid.png": "lux-depth-v6|input-0000|depth-validity",
        "v6/input-0001/delivery.tif": "lux-depth-v6|input-0001|photograph",
    }
    for relative, group in expected_groups.items():
        assert items[relative]["display_hint"]["compare_group"] == group
        if relative.startswith("source-v5/"):
            assert items[relative]["display_hint"]["role"] == "file"


@pytest.mark.asyncio
async def test_v6_projection_rejects_the_v5_delivery_schema(tmp_path):
    fence = DispatchFence(
        DispatchLocator("job_v6", "attempt", "dispatch", "a" * 64, "tenant"), "worker", 1, 100, str(tmp_path), str(tmp_path)
    )
    with pytest.raises(ArtifactStoreError, match="exact verified V6 inventory"):
        await ManagedV6PhotographyPublisher(artifact_store=None, record_store=None).publish(
            fence,
            {},
            state="succeeded",
            exit_code=0,
            artifacts={"schema": "tp.lux.delivery.v3", "paths": [], "execution_evidence": "execution-evidence.json"},
            run_summary={},
            expected_file_integrity={},
        )
