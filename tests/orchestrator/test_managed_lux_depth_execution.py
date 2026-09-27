"""The unified route uses the shared worker and independently verified publication."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.orchestrator import test_job_execution_service as service_fixture
from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request
from transformation_portal.orchestrator import execution_dispatch
from transformation_portal.orchestrator.dispatch import _dispatch_fence
from transformation_portal.orchestrator.lux_depth_adapter import prepare_unified_dispatch
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost

pytestmark = pytest.mark.unit
request_case = service_fixture.request_case


@pytest.fixture(params=["process", "infer"])
def workflow(request):
    return request.param


@pytest.fixture
def managed_unified(request_case, workflow, monkeypatch, tmp_path):
    def prepare(request, *, publisher):
        request = ManagedLuxDepthV6Request(request) if workflow == "process" else request
        return prepare_unified_dispatch(request, workflow=workflow, publisher=publisher)

    monkeypatch.setattr(service_fixture, "prepare_photography_dispatch", prepare)
    monkeypatch.setenv("TP_LUX_DEPTH_MANAGED_ENABLED", "1")
    context = service_fixture.managed.__wrapped__(request_case, monkeypatch, tmp_path)
    context[1].effective_request = {"pipeline": "lux-depth", "args": {"workflow": workflow}}
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "0")
    return context


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [False, True])
async def test_unified_coordinator_verifies_before_native_artifact_publication(managed_unified, workflow, monkeypatch, tamper):
    service, job, repository, _events, _store, records, fence, prepared = managed_unified
    await repository.create(job)
    prefix = "v6/" if workflow == "process" else ""

    async def controlled_child(*argv, **kwargs):
        assert kwargs["start_new_session"] is True
        assert "TP_API_KEY" not in kwargs["env"]
        assert (
            execution_dispatch.execute_dispatch_plan(
                prepared.plan_bytes,
                execution_bindings=prepared.bindings_bytes,
                output_root=Path(fence.output_root),
                execution_workspace=Path(argv[argv.index("--execution-workspace") + 1]),
            )
            == 0
        )
        if tamper:
            (Path(fence.output_root) / prefix / "input-0000/delivery.tif").write_bytes(b"changed after execution")

        class Process:
            pid = None
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                return self.returncode

        process = Process()
        process.stdout.feed_eof()
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", controlled_child)
    token = _dispatch_fence.set(fence)
    try:
        await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    if tamper:
        assert not records.commits
        assert records.finishes[0]["state"] == "failed"
        return
    assert len(records.commits) == 1
    assert not records.finishes
    result = await repository.get(job.id)
    assert result.state == "succeeded"
    assert result.run_summary["pipeline"] == "lux_depth"
    assert result.run_summary["workflow"] == workflow
    assert result.run_summary["engine_pipeline"] == ("lux_depth_v6" if workflow == "process" else "lux_depth_v5")
    assert result.run_summary["plan_schema"] == ("tp.execution.plan.v5" if workflow == "process" else "tp.execution.plan.v4")
    assert result.artifacts["schema"] == ("tp.lux.delivery.v4" if workflow == "process" else "tp.lux.delivery.v3")
    items = {item["path"]: item for item in result.artifacts["items"]}
    assert items[prefix + "input-0000/delivery.tif"]["preview_url"] == items[prefix + "input-0000/preview.png"]["url"]
    assert not list((Path(fence.output_root).parents[2] / "private").iterdir())


@pytest.mark.asyncio
async def test_unified_revocation_prevents_pickup_despite_native_feature_grants(managed_unified, monkeypatch):
    service, job, repository, _events, _store, records, fence, _prepared = managed_unified
    await repository.create(job)
    monkeypatch.setenv("TP_LUX_DEPTH_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "1")
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "1")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned after route revocation"))
    token = _dispatch_fence.set(fence)
    try:
        with pytest.raises(DispatchAuthorityLost, match="disabled"):
            await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    assert not Path(fence.output_root).exists()
    assert not records.commits
