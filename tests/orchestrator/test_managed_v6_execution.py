"""The shared coordinator publishes only complete independently verified V6 jobs."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from tests.orchestrator import test_job_execution_service as service_fixture
from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request
from transformation_portal.orchestrator import execution_dispatch
from transformation_portal.orchestrator.dispatch import _dispatch_fence
from transformation_portal.orchestrator.photography_v6_adapter import prepare_v6_dispatch
from transformation_portal.orchestrator.storage.operational import DispatchAuthorityLost

pytestmark = pytest.mark.unit
request_case = service_fixture.request_case


@pytest.fixture
def managed_v6(request_case, monkeypatch, tmp_path):
    def prepare(request, *, publisher):
        return prepare_v6_dispatch(ManagedLuxDepthV6Request(request), publisher=publisher)

    monkeypatch.setattr(service_fixture, "prepare_photography_dispatch", prepare)
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "1")
    context = service_fixture.managed.__wrapped__(request_case, monkeypatch, tmp_path)
    context[1].effective_request["pipeline"] = "lux-depth-v6"
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "0")
    return context


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper", [False, True])
async def test_managed_v6_service_runs_both_stages_and_verifies_before_publication(managed_v6, monkeypatch, tamper):
    service, job, repository, _events, store, records, fence, prepared = managed_v6
    await repository.create(job)
    commands = []

    async def controlled_child(*argv, **kwargs):
        commands.append(argv)
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
            (Path(fence.output_root) / "v6/input-0000/delivery.tif").write_bytes(b"changed after child completion")

        class Process:
            pid = None
            returncode = 0
            stdout = asyncio.StreamReader()

            async def wait(self):
                return self.returncode

        process = Process()
        process.stdout.feed_data(b"progress=100%\n")
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
    assert len(commands) == len(records.commits) == 1
    assert not records.finishes
    result = await repository.get(job.id)
    assert result.state == "succeeded"
    assert result.run_summary["pipeline"] == "lux_depth_v6"
    assert result.artifacts["schema"] == "tp.lux.delivery.v4"
    items = {item["path"]: item for item in result.artifacts["items"]}
    assert items["v6/input-0000/delivery.tif"]["preview_url"] == items["v6/input-0000/preview.png"]["url"]
    assert "v6/input-0000/depth-relative.tif" in items
    assert "source-v5/input-0000/native-depth.npy" in items
    manifest = json.loads(records.commits[0]["manifest_bytes"])
    assert set(items) == {entry["path"] for entry in manifest["files"]}
    delivery = next(entry for entry in manifest["files"] if entry["path"] == "v6/input-0000/delivery.tif")
    stream = await store.open_bytes(job.id, delivery["storage_path"])
    data = b"".join([chunk async for chunk in stream])
    assert hashlib.sha256(data).hexdigest() == delivery["sha256"]
    assert not list((Path(fence.output_root).parents[2] / "private").iterdir())


@pytest.mark.asyncio
async def test_v6_revocation_rejects_pickup_even_when_v5_is_enabled(managed_v6, monkeypatch):
    service, job, repository, _events, _store, records, fence, _prepared = managed_v6
    await repository.create(job)
    monkeypatch.setenv("TP_LUX_V6_MANAGED_ENABLED", "0")
    monkeypatch.setenv("TP_LUX_V5_MANAGED_ENABLED", "1")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned after V6 revocation"))
    token = _dispatch_fence.set(fence)
    try:
        with pytest.raises(DispatchAuthorityLost, match="disabled"):
            await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    assert not Path(fence.output_root).exists()
    assert not records.commits


@pytest.mark.asyncio
async def test_v6_changed_publication_limits_fail_before_workspace_or_spawn(managed_v6, monkeypatch):
    from transformation_portal.orchestrator.artifact_store import generation

    service, job, repository, _events, _store, records, fence, _prepared = managed_v6
    await repository.create(job)
    monkeypatch.setattr(generation, "MAX_GENERATION_FILES", generation.MAX_GENERATION_FILES - 1)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", lambda *_a, **_kw: pytest.fail("spawned with changed limits"))
    token = _dispatch_fence.set(fence)
    try:
        with pytest.raises(ValueError, match="[Pp]ublisher.*changed|[Ll]imits.*changed"):
            await service.execute(fence.locator, asyncio.Event())
    finally:
        _dispatch_fence.reset(token)
    assert not Path(fence.output_root).exists()
    assert not records.commits
