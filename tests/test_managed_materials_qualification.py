"""Managed qualification preserves budgets, retained sources and actor authority."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import os
import signal
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.photographic_delivery import managed_qualification as managed

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_barrier_surfaces_completed_http_failure_without_waiting_for_timeout():
    async def rejected():
        raise RuntimeError("Managed request rejected: 413 governed profile did not match")

    operation = asyncio.create_task(rejected())
    with pytest.raises(RuntimeError, match="413 governed profile"):
        await managed._wait_for_barrier(operation, asyncio.Event(), "inspection", timeout=30)
    assert operation.done()


@pytest.mark.asyncio
async def test_barrier_cannot_accept_request_completion_without_observed_phase():
    operation = asyncio.create_task(asyncio.sleep(0))
    with pytest.raises(RuntimeError, match="completed before"):
        await managed._wait_for_barrier(operation, asyncio.Event(), "inspection")


@pytest.mark.asyncio
async def test_barrier_keeps_the_actual_operation_owned_on_success_or_deadline():
    finish, reached = asyncio.Event(), asyncio.Event()
    operation = asyncio.create_task(finish.wait())
    try:
        reached.set()
        await managed._wait_for_barrier(operation, reached, "inspection")
        assert not operation.done()
        with pytest.raises(TimeoutError, match="did not reach"):
            await managed._wait_for_barrier(operation, asyncio.Event(), "publication", timeout=0.01)
        assert not operation.done()
    finally:
        finish.set()
        await operation


@pytest.mark.asyncio
@pytest.mark.parametrize("memory,outstanding", [(0, 1), (6 * managed.offline.GIB, 0)])
async def test_transport_probe_rejects_inner_response_release_before_outer_quiescence(memory, outstanding):
    finish = asyncio.Event()
    operation = asyncio.create_task(finish.wait())
    observer = SimpleNamespace(
        service=SimpleNamespace(
            admission=SimpleNamespace(snapshot=lambda: {"memory_reserved_bytes": memory, "outstanding": outstanding})
        )
    )
    try:
        with pytest.raises(RuntimeError, match="before outer send"):
            managed._require_transport_reservation(observer, operation)
    finally:
        finish.set()
        await operation


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_observed_transport_blocks_outside_app_and_drains_repeated_cancel(cancel):
    phases, sent = [], []
    observer = SimpleNamespace(
        service=SimpleNamespace(
            admission=SimpleNamespace(snapshot=lambda: {"memory_reserved_bytes": 6 * managed.offline.GIB, "outstanding": 1})
        ),
        pause_transport=True,
        scenario="cancel_transport" if cancel else "success",
        transport_entered=asyncio.Event(),
        transport_release=asyncio.Event(),
        transport_cancelled=asyncio.Event(),
        transport_cancel_release=asyncio.Event(),
        inner_response_returned=asyncio.Event(),
        capture=phases.append,
    )

    async def app(_scope, _receive, send):
        observer.inner_response_returned.set()
        await send({"type": "http.response.body", "body": b"unaltered"})

    async def send(message):
        sent.append(message)

    task = asyncio.create_task(
        managed._ObservedTransport(app, observer)({"type": "http", "method": "POST", "path": "/v1/revisions"}, None, send)
    )
    try:
        await managed._wait_for_owned_transport(observer, task)
        assert sent == [] and not task.done()
        if cancel:
            task.cancel()
            await managed._wait_for_barrier(task, observer.transport_cancelled, "cancelled send")
            task.cancel()
            await asyncio.sleep(0)
            managed._require_transport_reservation(observer, task)
            observer.transport_cancel_release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert sent == []
        else:
            observer.transport_release.set()
            await task
            assert sent == [{"type": "http.response.body", "body": b"unaltered"}]
        assert phases[-1] == "native_outer_transport_quiescent"
    finally:
        observer.transport_release.set()
        observer.transport_cancel_release.set()
        await asyncio.gather(task, return_exceptions=True)


def test_deployed_application_never_reuses_a_preimported_global_app(monkeypatch):
    monkeypatch.setitem(managed.sys.modules, "app", SimpleNamespace())
    with pytest.raises(RuntimeError, match="fresh deployed application"):
        managed._deployed_application(object())


@pytest.mark.parametrize("kind", ["deployed", "plain", "failed_import"])
def test_deployed_application_captures_real_router_service_and_always_restores_constructor(monkeypatch, kind):
    from transformation_portal.api.routes import photographic_review as routes

    monkeypatch.delitem(managed.sys.modules, "app", raising=False)
    constructor, service = routes.PhotographicReviewService, object()

    class DeployedApplication:
        pass

    module = SimpleNamespace(
        app=DeployedApplication() if kind == "deployed" else object(), _PhotographicReviewApplication=DeployedApplication
    )

    def imported(name):
        assert name == "app" and routes.PhotographicReviewService() is service
        if kind == "failed_import":
            raise RuntimeError("controlled import failure")
        return module

    monkeypatch.setattr(managed.importlib, "import_module", imported)
    if kind == "deployed":
        assert managed._deployed_application(service) is module
    else:
        with pytest.raises(RuntimeError, match="controlled import|deployed outer"):
            managed._deployed_application(service)
    assert routes.PhotographicReviewService is constructor


@pytest.fixture(name="case")
def fixture_case(tmp_path, monkeypatch):
    root, output = tmp_path / "retained", tmp_path / "evidence"
    root.mkdir()
    output.mkdir()
    source = root / "source.tif"
    tifffile.imwrite(source, np.full((64, 96, 3), 24000, np.uint16), photometric="rgb", iccprofile=output_srgb_icc())
    recipe = {
        "schema": "tp.photo.materials_recipe.v1",
        "input_stage": "display_referred",
        "regions": [{"id": "preservation", "label": "stone", "region": [0, 0, 1, 1], "strength": 0}],
    }
    row = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "recipe": recipe}
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"schema": managed.MANIFEST_SCHEMA, "sources": [row]}))
    monkeypatch.setenv("TP_MANAGED_QUALIFICATION_DATABASE_URL", "postgresql+asyncpg://localhost/qualification_test")
    monkeypatch.setattr(managed, "runtime_identity", lambda: {"managed_profile_identity": "a" * 64})
    monkeypatch.setattr(managed, "_headroom", lambda *_args: (managed.offline.GIB, []))
    return SimpleNamespace(source=source, manifest=manifest, row=row, output=output / "fresh")


@pytest.mark.parametrize(
    "database", ["", "postgresql://localhost/production", "sqlite:///qualification_test", "not-a-url-secret"]
)
def test_database_must_be_an_explicit_dedicated_postgres_test_database(database):
    with pytest.raises(ValueError, match="dedicated Postgres") as error:
        managed._database_url({"TP_MANAGED_QUALIFICATION_DATABASE_URL": database})
    assert database not in str(error.value) or not database


def test_preflight_is_metadata_only_and_does_not_launch_or_change_environment(case, monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Preflight must not decode pixels, connect to Postgres or launch a worker")

    monkeypatch.setattr(tifffile.TiffPage, "asarray", forbidden)
    monkeypatch.setattr(managed, "_run_process", forbidden)
    before, raw = dict(os.environ), case.source.read_bytes()
    report = managed.qualify(case.manifest, case.output, "a" * 64, preflight_only=True)
    assert report["schema"] == managed.SCHEMA and report["status"] == "planned"
    assert (
        report["production_flags_changed"]
        is report["model_inference_performed"]
        is report["aws_gpu_validation_performed"]
        is False
    )
    assert report["photographic_acceptance"] == "not_established"
    assert report["config"]["memory_bytes"] == 12 * managed.offline.GIB
    assert dict(os.environ) == before and case.source.read_bytes() == raw
    assert set(path.name for path in case.output.iterdir()) == {"qualification.json"}


def test_runtime_profile_mismatch_never_launches_a_worker(case, monkeypatch):
    monkeypatch.setattr(managed, "_run_process", lambda *_args, **_kwargs: pytest.fail("Blocked runtime must not launch"))
    report = managed.qualify(case.manifest, case.output, "b" * 64)
    assert report["status"] == "blocked"
    assert report["blocking_reasons"] == ["governed_runtime_profile_mismatch"]


@pytest.mark.parametrize("mutation", ["unknown_field", "duplicate", "relative", "changed_hash", "symlink"])
def test_closed_manifest_and_retained_identity_reject_before_output_creation(case, monkeypatch, mutation):
    document = json.loads(case.manifest.read_text())
    if mutation == "unknown_field":
        document["sources"][0]["memory_bytes"] = 1
    elif mutation == "duplicate":
        document["sources"].append(document["sources"][0].copy())
    elif mutation == "relative":
        document["sources"][0]["path"] = "source.tif"
    elif mutation == "changed_hash":
        document["sources"][0]["sha256"] = "b" * 64
    else:
        path = case.source.parent / "linked.tif"
        path.symlink_to(case.source)
        document["sources"][0]["path"] = str(path)
    case.manifest.write_text(json.dumps(document))
    monkeypatch.setattr(tifffile.TiffPage, "asarray", lambda *_args, **_kwargs: pytest.fail("Must not decode"))
    with pytest.raises(ValueError):
        managed.qualify(case.manifest, case.output, "a" * 64, preflight_only=True)
    assert not case.output.exists()


def test_existing_output_and_retained_namespace_are_preserved(case):
    case.output.mkdir()
    sentinel = case.output / "retained.txt"
    sentinel.write_text("preserve")
    with pytest.raises(ValueError, match="fresh"):
        managed.qualify(case.manifest, case.output, "a" * 64, preflight_only=True)
    assert sentinel.read_text() == "preserve"
    with pytest.raises(ValueError, match="separate"):
        managed.qualify(case.manifest, case.source.parent / "new", "a" * 64, preflight_only=True)


def test_signed_fixture_headers_are_method_target_bound_and_not_forged_actor_headers(monkeypatch):
    from transformation_portal.core.security.frontdoor_identity import verify_actor_assertion

    monkeypatch.setenv("TP_API_KEY", "private-key")
    monkeypatch.setenv("TP_FRONTDOOR_IDENTITY_SECRET", "dedicated-test-identity-secret-32bytes")
    headers = managed._headers("GET", "/v1/jobs/private/photographic-review")
    actor = verify_actor_assertion(
        headers["x-tp-actor-assertion"],
        secret=os.environ["TP_FRONTDOOR_IDENTITY_SECRET"],
        method="GET",
        target="/v1/jobs/private/photographic-review",
    )
    assert actor["username"] == "qualification" and "x-tp-actor" not in headers
    with pytest.raises(ValueError):
        verify_actor_assertion(
            headers["x-tp-actor-assertion"],
            secret=os.environ["TP_FRONTDOOR_IDENTITY_SECRET"],
            method="POST",
            target="/v1/jobs/private/photographic-review",
        )


@pytest.mark.parametrize("incremental,qualified", [(4 * managed.offline.GIB, True), (5 * managed.offline.GIB, False)])
def test_complete_managed_margin_cannot_raise_fixed_profile_and_verifier_is_sequential(
    case, monkeypatch, incremental, qualified
):
    phases = []

    def run(root, mode, **kwargs):
        assert kwargs["memory_limit"] == 12 * managed.offline.GIB
        phases.append(mode)
        if mode == "_worker":
            managed.offline._write_json(
                root / "managed-result.json",
                {
                    "runtime": {"managed_profile_identity": "a" * 64},
                    "baseline_rss_bytes": 0,
                    "peak_rss_bytes": incremental,
                    "incremental_peak_rss_bytes": incremental,
                    "transport_ownership_verified": True,
                },
            )
        else:
            assert phases == ["_worker", "_verify"]
            managed.offline._write_json(root / "verification.json", {"scope": "independent_after_managed_worker_exit"})
        return {}

    monkeypatch.setattr(managed, "_run_process", run)
    report = managed.qualify(case.manifest, case.output, "a" * 64, repeats=1)
    assert report["status"] == ("measured" if qualified else "not_qualified")
    assert report["measured_incremental_peak_plus_50_percent_bytes"] == incremental * 3 // 2
    assert report["config"]["memory_bytes"] == 12 * managed.offline.GIB
    assert report["source_unchanged"]
    assert report["encoded_products_repeatable"] is None
    assert report["managed_profile_qualification"] == "not_established"


@pytest.mark.parametrize(
    "repeats,concurrent,repeatable,transport_verified,passed",
    [
        (1, True, True, True, False),
        (2, False, True, True, False),
        (2, True, True, True, True),
        (2, True, False, True, False),
        (2, True, True, False, False),
    ],
)
def test_profile_gate_requires_repeatability_and_real_same_tenant_inspection(
    case, monkeypatch, repeats, concurrent, repeatable, transport_verified, passed
):
    def run(root, mode, **_kwargs):
        if mode == "_worker":
            managed.offline._write_json(
                root / "managed-result.json",
                {
                    "runtime": {"managed_profile_identity": "a" * 64},
                    "baseline_rss_bytes": 0,
                    "peak_rss_bytes": 4 * managed.offline.GIB,
                    "incremental_peak_rss_bytes": 4 * managed.offline.GIB,
                    "transport_ownership_verified": transport_verified,
                },
            )
        else:
            results = [{"source_index": 0, "product_identity_sha256": "a" * 64} for _ in range(repeats)]
            if not repeatable:
                results[-1]["product_identity_sha256"] = "b" * 64
            managed.offline._write_json(root / "verification.json", {"results": results})
        return {}

    monkeypatch.setattr(managed, "_run_process", run)
    report = managed.qualify(case.manifest, case.output, "a" * 64, repeats=repeats, concurrent_inspection=concurrent)
    assert (report["managed_profile_qualification"] == "passed_for_declared_cohort") is passed
    if repeats == 1:
        assert report["encoded_products_repeatable"] is None
    else:
        assert report["encoded_products_repeatable"] is repeatable
    if not repeatable:
        assert report["status"] == "not_qualified" and report["blocking_reasons"] == ["encoded_products_not_repeatable"]
    if not transport_verified:
        assert report["status"] == "not_qualified"
        assert report["blocking_reasons"] == ["deployed_transport_ownership_unverified"]


@pytest.mark.parametrize("repeat", [0, 4, True])
def test_repeats_are_bounded_before_namespace_creation(case, repeat):
    with pytest.raises(ValueError, match="one to three"):
        managed.qualify(case.manifest, case.output, "a" * 64, repeats=repeat)
    assert not case.output.exists()


def test_worker_incremental_peak_uses_actual_initial_rss_not_initial_high_water(tmp_path, monkeypatch):
    import resource

    psutil = pytest.importorskip("psutil")

    spec = tmp_path / "specification.json"
    runtime = {"managed_profile_identity": "a" * 64}
    managed.offline._write_json(spec, {"runtime": runtime})
    multiplier = 1 if managed.sys.platform == "darwin" else 1024
    peaks = iter([200, 300])
    monkeypatch.setattr(resource, "getrusage", lambda _who: SimpleNamespace(ru_maxrss=next(peaks)))
    monkeypatch.setattr(psutil, "Process", lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=100 * multiplier)))
    monkeypatch.setattr(managed.signal, "pthread_sigmask", lambda *_args: set())
    monkeypatch.setattr(managed, "_configure_worker", lambda _root: None)
    monkeypatch.setattr(managed, "runtime_identity", lambda: runtime)

    async def run(*_args):
        return {"results": []}

    monkeypatch.setattr(managed, "_managed_run", run)
    managed._worker(spec)
    result = managed.offline._read_json(tmp_path / "managed-result.json")
    assert result["baseline_rss_bytes"] == 100 * multiplier
    assert result["initial_process_high_water_bytes"] == 200 * multiplier
    assert result["peak_rss_bytes"] == 300 * multiplier
    assert result["incremental_peak_rss_bytes"] == 200 * multiplier
    assert result["baseline_semantics"] == "actual_rss_after_module_imports_before_managed_setup"


def _committed_fixture():
    identity = "a" * 32
    source = {"bound": {"sha256": "b" * 64}}
    result = {"tenant_id": "qualification", "job_id": "job"}
    variant = {
        "kind": "review_asset",
        "scene_id": "input-0000",
        "artifact_path": f"revisions/{identity}/master.tif",
        "storage_path": f"photographic-review/{identity}/master.tif",
        "sha256": "c" * 64,
        "size_bytes": 1234,
    }
    products = [
        {
            "kind": "review_asset",
            "filename": name,
            "storage_path": f"photographic-review/{identity}/{name}",
            "sha256": "d" * 64,
            "size_bytes": 123,
            "content_type": mime,
        }
        for name, mime in (
            ("materials-preview.png", "image/png"),
            ("materials-overlay.png", "image/png"),
            ("materials-diagnostics.json", "application/json"),
        )
    ]
    committed = {
        "id": identity,
        "variant": variant,
        "native_products": products,
        "native_source": {
            "artifact_path": "v6/input-0000/delivery.tif",
            "sha256": "b" * 64,
            "source_identity_sha256": "b" * 64,
        },
        "receipt": {"revision_sha256": "c" * 64},
    }
    return result, committed, source


def test_verifier_reconstructs_storage_paths_sizes_hashes_and_content_types_from_committed_state(tmp_path):
    result, committed, source = _committed_fixture()
    products = managed._committed_products(tmp_path, result, committed, source, 0)
    assert products[0] == {
        "path": str(tmp_path / "artifacts/qualification__job/photographic-review" / ("a" * 32) / "master.tif"),
        "sha256": "c" * 64,
        "size_bytes": 1234,
        "content_type": "image/tiff",
    }
    assert {row["content_type"] for row in products} == {"image/tiff", "image/png", "application/json"}
    changed = copy.deepcopy(committed)
    changed["variant"]["size_bytes"] += 1
    changed["native_products"][0]["sha256"] = "e" * 64
    assert managed._committed_products(tmp_path, result, changed, source, 0) != products


@pytest.mark.parametrize(
    "field",
    [
        "master_path",
        "master_hash",
        "master_size",
        "source",
        "source_path",
        "diagnostic_path",
        "diagnostic_mime",
        "diagnostic_duplicate",
    ],
)
def test_committed_inventory_rejects_foreign_or_incomplete_authority(tmp_path, field):
    result, committed, source = _committed_fixture()
    if field == "master_path":
        committed["variant"]["storage_path"] = "photographic-review/foreign/master.tif"
    elif field == "master_hash":
        committed["variant"]["sha256"] = "e" * 64
    elif field == "master_size":
        committed["variant"]["size_bytes"] = 0
    elif field == "source":
        committed["native_source"]["source_identity_sha256"] = "e" * 64
    elif field == "source_path":
        committed["native_source"]["artifact_path"] = "v6/input-0001/delivery.tif"
    elif field == "diagnostic_path":
        committed["native_products"][0]["storage_path"] = "photographic-review/foreign/materials-preview.png"
    elif field == "diagnostic_mime":
        committed["native_products"][0]["content_type"] = "image/jpeg"
    else:
        committed["native_products"][0] = committed["native_products"][1].copy()
    with pytest.raises(ValueError, match="Durable managed"):
        managed._committed_products(tmp_path, result, committed, source, 0)


@pytest.mark.parametrize("which", ["input", "output"])
def test_actual_native_master_content_must_match_the_receipt(which):
    from transformation_portal.core.image_artifact import ImageMaster

    before = ImageMaster(np.full((2, 3, 3), 0.25, np.float32), "a" * 64, 16)
    after = ImageMaster(np.full((2, 3, 3), 0.3, np.float32), "b" * 64, 16)
    receipt = {"input_master_content_sha256": before.content_hash(), "output_master_content_sha256": after.content_hash()}
    managed._verify_master_content(before, after, receipt)
    receipt[which + "_master_content_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="Actual managed master content"):
        managed._verify_master_content(before, after, receipt)


@pytest.mark.parametrize("failure", ["memory", "scratch", "timeout", "children", "observer", "cancel"])
def test_supervisor_drains_only_owned_process_group_on_every_failure(tmp_path, monkeypatch, failure):
    psutil = pytest.importorskip("psutil")
    state = SimpleNamespace(returncode=None, killed=[], waited=False)

    class Process:
        pid = 424242

        def poll(self):
            return state.returncode

        def wait(self, timeout=None):
            state.waited = True
            return state.returncode

    def launch(*args, **kwargs):
        assert kwargs["start_new_session"] and kwargs["stdin"] == managed.subprocess.DEVNULL
        assert args[0][1:3] == ["-m", managed.MODULE]
        assert kwargs["env"]["TMPDIR"].startswith(str(tmp_path))
        assert "UNRELATED_SECRET" not in kwargs["env"]
        return Process()

    def kill(pid, event):
        assert pid == Process.pid
        if event == 0:
            raise ProcessLookupError()
        state.killed.append((pid, event))
        state.returncode = -event

    def observe(pid):
        assert pid == Process.pid
        if failure == "observer":
            raise PermissionError("Owned process inspection denied")
        if failure == "cancel":
            raise KeyboardInterrupt()
        return SimpleNamespace(
            memory_info=lambda: SimpleNamespace(rss=20 * managed.offline.GIB if failure == "memory" else 1),
            children=lambda **_kwargs: [object()] if failure == "children" else [],
        )

    clock = iter([0])
    monkeypatch.setenv("UNRELATED_SECRET", "must-stay-local")
    monkeypatch.setattr(managed.subprocess, "Popen", launch)
    monkeypatch.setattr(managed.os, "killpg", kill)
    monkeypatch.setattr(psutil, "Process", observe)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=64 * managed.offline.GIB))
    monkeypatch.setattr(managed.time, "monotonic", lambda: next(clock, 1000 if failure == "timeout" else 0))
    monkeypatch.setattr(
        managed.offline,
        "_scratch_usage",
        lambda *_args: {"accounted_bytes": 20 * managed.offline.GIB if failure == "scratch" else 0},
    )
    expected = {"observer": PermissionError, "cancel": KeyboardInterrupt}.get(failure, RuntimeError)
    with pytest.raises(expected):
        managed._run_process(tmp_path, "_worker", memory_limit=12 * managed.offline.GIB, scratch_limit=managed.offline.GIB)
    assert state.waited
    assert state.killed == [(Process.pid, signal.SIGTERM), (Process.pid, signal.SIGKILL)]


@pytest.mark.integration
@pytest.mark.parametrize("scenario", sorted(managed.SCENARIOS))
def test_real_postgres_managed_child_rgba_cohort_faults_and_independent_restart_verifier(tmp_path, scenario):
    if not os.environ.get("TP_MANAGED_QUALIFICATION_DATABASE_URL"):
        pytest.skip("Managed CPU integration requires an explicit dedicated migrated Postgres *_test database")
    profile = managed._profile_identity()
    if profile is None:
        pytest.skip("The current host/source bytes do not match the governed native resource profile")
    retained, evidence = tmp_path / "retained", tmp_path / "evidence"
    retained.mkdir()
    evidence.mkdir()
    source = retained / "source.tif"
    samples = np.full((4000, 5000, 4), 24000, np.uint16)
    samples[..., 3] = 65535
    samples[:, :8, 3] = 0
    samples[:, 8:16, 3] = 32768
    tifffile.imwrite(
        source, samples, photometric="rgb", extrasamples="unassalpha", rowsperstrip=64, iccprofile=output_srgb_icc()
    )
    del samples  # The parent must not retain native arrays while measuring its child.
    recipe = {
        "schema": "tp.photo.materials_recipe.v1",
        "input_stage": "display_referred",
        "regions": [
            {"id": f"preservation-{index}", "label": "stone", "region": rectangle, "strength": 0}
            for index, rectangle in enumerate(([0, 0, 0.5, 0.5], [0.5, 0, 1, 0.5], [0, 0.5, 0.5, 1], [0.5, 0.5, 1, 1]))
        ],
    }
    manifest = retained / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": managed.MANIFEST_SCHEMA,
                "sources": [
                    {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "recipe": recipe}
                ],
            }
        )
    )
    report = managed.qualify(
        manifest, evidence / "fresh", profile, repeats=2 if scenario == "success" else 1, scenario=scenario
    )
    assert report["status"] == "measured", report["blocking_reasons"]
    result = report["complete_managed_result"]
    phases = [entry["phase"] for entry in result["phases"]]
    if scenario in {"success", "cancel_transport"}:
        expected = 2 if scenario == "success" else 1
        assert len(result["results"]) == expected and len(report["independent_verification"]["results"]) == expected
        assert result["transport_ownership_verified"] is True
        assert all(
            row["products_verified"] == 5 and row["maximum_16bit_code_delta"] <= 1
            for row in report["independent_verification"]["results"]
        )
        assert "native_outer_transport_holds_reservation_after_inner_return" in phases
        assert (
            phases.index("publication_written")
            < phases.index("publication_verified")
            < phases.index("before_cas_after_product_drop")
            < phases.index("native_outer_transport_blocked")
        )
        if scenario == "success":
            assert "same_tenant_inspection_completed_while_native_held" in phases
            assert "same_tenant_inspection_completed_while_outer_transport_blocked" in phases
            helper_crop = next(
                row["inspection_crop"]
                for row in result["phases"]
                if row["phase"] == "same_tenant_inspection_completed_while_native_held"
            )
            transport_crop = next(
                row["inspection_crop"]
                for row in result["phases"]
                if row["phase"] == "same_tenant_inspection_completed_while_outer_transport_blocked"
            )
            assert helper_crop != transport_crop
            outer_start = phases.index("native_outer_transport_holds_reservation_after_inner_return")
            outer_end = phases.index("same_tenant_inspection_completed_while_outer_transport_blocked")
            assert "source_loaded" in phases[outer_start + 1 : outer_end]
            assert phases.index("native_inner_response_returned") < phases.index("response_sent")
        else:
            assert "repeated_transport_cancellation_holds_owned_memory" in phases
            assert "cancelled_transport_committed_revision_recovered" in phases
    else:
        assert result["results"] == report["independent_verification"]["results"] == []
        if scenario.startswith("cancel_"):
            assert "repeated_cancellation_holds_owned_memory" in phases
    assert result["final_admission"]["memory_reserved_bytes"] == result["final_admission"]["scratch_reserved_bytes"] == 0
