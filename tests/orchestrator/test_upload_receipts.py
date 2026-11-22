"""Source-bound upload recovery contracts without service substitution."""

from __future__ import annotations

import asyncio
import hashlib
import io
import json
import os
import threading
import uuid
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from transformation_portal.core.archive_execution_plan import ARCHIVE_OPERATIONS
from transformation_portal.ingest.upload_recovery import (
    MARKER_FILENAME,
    UploadBatchLock,
    initialize_batch,
    seal_completion,
    verify_completion,
)
from transformation_portal.ingest.upload_staging import (
    IncomingUpload,
    UploadStagingError,
    cleanup_expired_batches,
    stage_upload_batch,
)
from transformation_portal.orchestrator.photographic_admission import drain_owned
from transformation_portal.orchestrator.storage.base import JobRecord
from transformation_portal.orchestrator.upload_receipts import (
    UploadOwnershipLost,
    UploadReservation,
    bind_upload_references,
    upload_owner,
    upload_source_paths,
    validate_upload_request_id,
)

pytestmark = pytest.mark.unit


def reservation(root: Path, *, owner: str = "a" * 64) -> UploadReservation:
    batch = "upload_" + uuid.uuid4().hex
    return UploadReservation(
        batch, "tenant_a", owner, str(uuid.uuid4()), uuid.uuid4().hex, str(root / batch), "staging", 1.0, 10.0
    )


def sealed(root: Path):
    item = reservation(root)
    initialize_batch(item)
    result = stage_upload_batch(
        upload_root=root,
        uploads=[IncomingUpload("nested/test.txt", io.BytesIO(b"original"))],
        now=item.created_at,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    seal_completion(item, result)
    return item, result


def partial_multipart(boundary: bytes) -> bytes:
    return (
        b"--" + boundary + b'\r\nContent-Disposition: form-data; name="files"; filename="one.txt"'
        b"\r\nContent-Type: application/octet-stream\r\n\r\nfirst\r\n--"
        + boundary
        + b'\r\nContent-Disposition: form-data; name="files"; filename="two.txt"'
        b"\r\nContent-Type: application/octet-stream\r\n\r\npartial"
    )


def material_descriptor_fixture(root: Path, relative: str = "bundle/evidence.json"):
    import numpy as np

    from transformation_portal.materials_v4.artifacts import write_evidence
    from transformation_portal.materials_v4.contracts import MaterialEvidence, RegionEvidence

    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    evidence = MaterialEvidence("a" * 64, (1, 1), (RegionEvidence("wall", "wall", np.ones((1, 1), dtype=np.float32)),))
    write_evidence(evidence, path)
    payload = json.loads(path.read_bytes())
    mask = payload["regions"][0]["mask"]
    original_mask = path.parent / mask["path"]
    mask["path"] = "nested/" + mask["path"]
    mask_path = path.parent / mask["path"]
    mask_path.parent.mkdir()
    original_mask.rename(mask_path)
    data = json.dumps(payload).encode()
    path.write_bytes(data)
    receipt = {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "source_sha256": evidence.source_sha256,
        "shape": [1, 1],
        "content_sha256": evidence.content_hash(),
    }
    mask_path.unlink()  # The read-set collector must never open/decode NPY.
    return SimpleNamespace(path=path, mask=mask_path, payload=payload, receipt=receipt)


@pytest.fixture
def relative_read_app(tmp_path, monkeypatch):
    import app
    from tests.core.test_execution_plan import _valid_payload
    from transformation_portal.core.execution_plan import CanonicalExecutionPlan
    from transformation_portal.orchestrator import execution_dispatch
    from transformation_portal.orchestrator.storage.postgres import PostgresJobRepository

    repo_root, cwd, client_root = (tmp_path / name for name in ("repo", "cwd", "client-root"))
    source = repo_root / "uploads" / "upload_relative" / "input"
    source.mkdir(parents=True)
    (repo_root / "output").mkdir()
    cwd.mkdir()
    client_root.mkdir()
    calls, commands = [], []
    repository = PostgresJobRepository(database_url="postgresql+asyncpg://unused/read_anchor_test")

    async def create(record, **kwargs):
        calls.append(("legacy", record, kwargs))

    class Store:
        async def admit(self, record, _plan, **kwargs):
            calls.append(("distributed", record, kwargs))
            return SimpleNamespace(job_id=record.id)

    async def no_error(*_args, **_kwargs):
        return None

    async def no_active(*_args, **_kwargs):
        return 0

    async def dispatch(_job, argv, **_kwargs):
        commands.append(argv)

    def prepare_plan(_request, *, trusted_argv):
        # Runtime/model preparation is outside this path-authority fixture.
        assert Path(trusted_argv[trusted_argv.index("--input-dir") + 1]) == source
        return CanonicalExecutionPlan.from_payload(_valid_payload()).to_canonical_json().encode()

    monkeypatch.setattr(app, "REPO_ROOT", repo_root)
    monkeypatch.setattr(app, "PILOT_CONTROL_PLANE_ENABLED", False)
    monkeypatch.setattr(app, "PILOT_MAX_ACTIVE_JOBS_PER_TENANT", 8)
    monkeypatch.setattr(app, "MAX_CONCURRENT_JOBS", 8)
    for name in ("ALLOWED_INPUT_ROOTS", "ALLOWED_OUTPUT_ROOTS", "ALLOWED_PATH_ROOTS"):
        monkeypatch.setattr(app, name, [repo_root])
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("TP_ORCHESTRATOR_EXECUTION_ROOT", str(repo_root / "execution"))
    monkeypatch.setattr(app, "_evaluate_pipeline_readiness", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(app, "_enforce_job_readiness_preflight", lambda *_args: None)
    monkeypatch.setattr(app, "_active_job_count", no_active)
    monkeypatch.setattr(app, "_pilot_active_job_count_for_tenant", no_active)
    monkeypatch.setattr(app, "_cleanup_expired_jobs", no_error)
    monkeypatch.setattr(app, "_record_pilot_audit", no_error)
    monkeypatch.setattr(app, "_cache_runtime_job", lambda *_args: None)
    monkeypatch.setattr(app, "_dispatch_job", dispatch)
    monkeypatch.setattr(app, "_flush_operational_outbox_best_effort", no_error)
    monkeypatch.setattr(app, "_job_repository", lambda: repository)
    monkeypatch.setattr(repository, "create", create)
    monkeypatch.setattr(app, "get_operational_record_store", Store)
    monkeypatch.setattr(execution_dispatch, "prepare_dispatch_plan", prepare_plan)
    payload = {
        "pipeline": "lux-depth-v3",
        "upload_read_base": str(client_root),
        "args": {
            "input_dir": "./uploads/upload_relative/input",
            "output_dir": "./output",
            "enable_segmentation": False,
            "enable_reconstruction": False,
            "upload_read_base": str(client_root),
            "repo_root": str(client_root),
        },
    }
    return SimpleNamespace(app=app, root=repo_root, source=source, client_root=client_root, payload=payload, calls=calls)


@pytest.mark.asyncio
@pytest.mark.parametrize("distributed", [False, True])
async def test_real_relative_preview_forwards_server_anchor_to_every_postgres_path(
    relative_read_app, monkeypatch, distributed
):
    from starlette.requests import Request

    case = relative_read_app
    monkeypatch.setattr(case.app, "_uses_dispatch_authority", lambda: distributed)
    preview = case.app._build_config_preview(case.payload)
    assert preview["field_errors"] == []
    assert preview["execution_args"]["input_dir"] == "./uploads/upload_relative/input"
    argv = case.app._argv_from_request(case.payload, execution_args=preview["execution_args"])
    assert Path(argv[argv.index("--input-dir") + 1]) == case.source
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/jobs",
            "headers": [(b"x-tp-upload-read-base", str(case.client_root).encode())],
        }
    )
    response = await case.app._create_job(case.payload, pilot_tenant=SimpleNamespace(tenant_id="tenant_a"), request=request)
    assert response.status_code == 200, response.body
    assert len(case.calls) == 1
    mode, record, options = case.calls[0]
    assert mode == ("distributed" if distributed else "legacy")
    assert record.effective_request["args"]["input_dir"] == "./uploads/upload_relative/input"
    assert options.get("upload_read_base") == case.root
    assert options["upload_read_base"] != case.client_root


def test_retained_relative_preview_uses_server_anchor_and_ignores_client_roots(relative_read_app):
    case = relative_read_app
    preview = case.app._build_config_preview(case.payload)
    job = case.app.Job(
        id="job_retained",
        created_at=1.0,
        request=case.payload,
        effective_request={"pipeline": "lux-depth-v3", "args": preview["execution_args"]},
    )
    retained = set()
    case.app._add_staged_input_dir(retained, job)
    assert str(case.source) in retained
    assert not any(Path(path).is_relative_to(case.client_root) for path in retained)


@pytest.mark.asyncio
@pytest.mark.parametrize("distributed", [False, True])
async def test_client_anchor_spoof_cannot_authorize_relative_escape(relative_read_app, monkeypatch, distributed):
    from starlette.requests import Request

    case = relative_read_app
    monkeypatch.setattr(case.app, "_uses_dispatch_authority", lambda: distributed)
    (case.client_root / "private").mkdir()
    case.payload["args"]["input_dir"] = "../client-root/private"
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/jobs",
            "headers": [(b"x-tp-upload-read-base", str(case.client_root).encode())],
        }
    )
    response = await case.app._create_job(case.payload, pilot_tenant=SimpleNamespace(tenant_id="tenant_a"), request=request)
    assert response.status_code == 400
    assert case.calls == []


def test_client_root_field_cannot_replace_missing_server_base_for_a_real_relative_path(tmp_path):
    with pytest.raises(UploadOwnershipLost):
        upload_source_paths({"args": {"input_dir": "default", "upload_read_base": str(tmp_path)}})


def caption_selector_paths(field, value, **kwargs):
    if field == "selector":
        from transformation_portal.orchestrator.upload_receipts import _admitted_source_paths

        plan = {
            "input_selection": {"root": "/frozen/input", "files": [{"id": "selected", "path": "photo.jpg"}]},
            "nodes": [{"configuration": {"captioning": {"enabled": True, "selector": value}}}],
        }
        return _admitted_source_paths(json.dumps(plan).encode(), None, **kwargs)
    return upload_source_paths({"pipeline": "lux-depth-v3", "args": {field: value}}, **kwargs)


@pytest.mark.parametrize("field", ["vlm_captioning_model", "vlmCaptioningModel", "selector"])
@pytest.mark.parametrize("role", ["default", "review"])
def test_closed_caption_roles_are_symbolic_only_in_caption_selector_fields(monkeypatch, field, role):
    from transformation_portal.orchestrator import upload_receipts
    from transformation_portal.vlm_captioning.fastvlm_runtime import FASTVLM_CHECKPOINT_DIRS

    assert role in FASTVLM_CHECKPOINT_DIRS

    original_path = upload_receipts.Path

    def checked_path(value, *args, **kwargs):
        if isinstance(value, str) and value.strip().lower() in FASTVLM_CHECKPOINT_DIRS:
            pytest.fail("a closed caption role must not resolve a physical path or require a base")
        return original_path(value, *args, **kwargs)

    monkeypatch.setattr(upload_receipts, "Path", checked_path)
    expected = (Path("/frozen/input"), Path("/frozen/input/photo.jpg")) if field == "selector" else ()
    assert caption_selector_paths(field, role) == expected


@pytest.mark.parametrize("field", ["vlm_captioning_model", "vlmCaptioningModel", "selector"])
@pytest.mark.parametrize("value", ["./caption-model", "unknown-caption-role"])
def test_relative_caption_paths_and_unknown_roles_require_server_base(tmp_path, field, value):
    with pytest.raises(UploadOwnershipLost, match="explicit server-owned base"):
        caption_selector_paths(field, value)
    expected = {(tmp_path / value).resolve()}
    if field == "selector":
        expected.update((Path("/frozen/input"), Path("/frozen/input/photo.jpg")))
    assert set(caption_selector_paths(field, value, upload_read_base=tmp_path)) == expected


def frozen_selection_paths(selection, **kwargs):
    from transformation_portal.orchestrator.upload_receipts import _admitted_source_paths

    return _admitted_source_paths(json.dumps({"input_selection": selection, "nodes": []}).encode(), None, **kwargs)


@pytest.mark.parametrize("relative_root", [False, True])
def test_frozen_selected_members_resolve_against_selection_root(tmp_path, monkeypatch, relative_root):
    base, cwd = tmp_path / "server", tmp_path / "cwd"
    base.mkdir()
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    root = base / "uploads"
    selection = {
        "root": "uploads" if relative_root else str(root),
        "files": [{"id": "selected", "path": "upload_selected/input/photo.jpg"}],
    }
    paths = frozen_selection_paths(selection, upload_read_base=base if relative_root else None)
    assert paths == (root, root / "upload_selected/input/photo.jpg")


def test_relative_frozen_selection_requires_explicit_server_base():
    with pytest.raises(UploadOwnershipLost, match="explicit server-owned base"):
        frozen_selection_paths({"root": "uploads", "files": [{"id": "selected", "path": "photo.jpg"}]})


@pytest.mark.parametrize(
    "selection",
    [
        None,
        [],
        {"root": "/frozen/input"},
        {"root": "/frozen/input", "files": None},
        {"root": "/frozen/input", "files": {}},
        {"root": None, "files": []},
        {"root": "", "files": []},
        {"root": 7, "files": []},
        {"root": "/frozen/input", "files": [None]},
        {"root": "/frozen/input", "files": [{}]},
        *(
            {"root": "/frozen/input", "files": [{"path": path}]}
            for path in (7, "", "/absolute", "../escape", "./photo", "a\\b", "a\x00b")
        ),
    ],
)
def test_malformed_or_noncontained_frozen_members_fail_closed(selection):
    with pytest.raises(UploadOwnershipLost):
        frozen_selection_paths(selection)


@pytest.mark.parametrize("v6", [False, True])
@pytest.mark.parametrize("unified", [False, True])
def test_photographic_frozen_members_use_their_closed_binding_roots(tmp_path, v6, unified):
    from transformation_portal.orchestrator.photography_adapter import PhotographyBindings
    from transformation_portal.orchestrator.upload_receipts import _admitted_source_paths

    source, companions, materials = (tmp_path / name for name in ("sources", "companions", "materials"))
    descriptor = material_descriptor_fixture(materials, "upload_material/input/evidence.json")
    runtime = tmp_path / "runtime" / "python"
    bound = {
        "schema": "tp.job.photography.bindings.v1",
        "input_root": str(source),
        "companion_root": str(companions),
        "materials_root": str(materials),
        "runtime_python": str(runtime),
        "raw_python": None,
        "cache_root": None,
    }
    binding_bytes = PhotographyBindings.from_payload(bound).canonical_bytes
    if unified:
        binding_bytes = json.dumps(
            {
                "schema": "tp.job.lux_depth.bindings.v1",
                "pipeline": "lux-depth",
                "workflow": "process",
                "photography_bindings": bound,
            }
        ).encode()
    plan = {
        "schema": "tp.execution.plan.v4",
        "resources": {"max_input_bytes": 1024 * 1024, "max_pixels": 1},
        "inputs": [
            {
                "path": "upload_source/input/source.jpg",
                "companions": {"materials": {"masks": {"window": {"path": "upload_companion/input/window.png"}}}},
                "materials_v4": descriptor.receipt,
            }
        ],
        "companions_manifest": {"path": "upload_companion/input/companions.json"},
        "materials_manifest": {"path": "upload_material/input/materials.json"},
    }
    if v6:
        plan = {"schema": "tp.execution.plan.v5", "inference": plan}
    paths = _admitted_source_paths(
        json.dumps(plan).encode(), binding_bytes, upload_read_base=tmp_path / "unrelated-server-base"
    )
    assert set(paths) == {
        source,
        companions,
        materials,
        runtime,
        source / "upload_source/input/source.jpg",
        companions / "upload_companion/input/window.png",
        companions / "upload_companion/input/companions.json",
        descriptor.path,
        descriptor.mask,
        materials / "upload_material/input/materials.json",
    }


def test_frozen_material_read_set_snapshots_only_json_and_keeps_exact_mask_locators(tmp_path, monkeypatch):
    from transformation_portal.orchestrator import upload_receipts

    fixture = material_descriptor_fixture(tmp_path)
    observed = []
    original = upload_receipts.snapshot

    def snapshot(root, path, **kwargs):
        assert path == fixture.path and path.suffix == ".json"
        assert kwargs["maximum_bytes"] == fixture.receipt["size_bytes"] <= 1024 * 1024
        observed.append(path)
        return original(root, path, **kwargs)

    monkeypatch.setattr(upload_receipts, "snapshot", snapshot)
    assert upload_receipts._material_source_paths(
        tmp_path, fixture.receipt, {"max_input_bytes": 1024 * 1024, "max_pixels": 1}
    ) == (fixture.mask,)
    assert observed == [fixture.path]
    assert not fixture.mask.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "digest",
        "size",
        "oversized",
        "source",
        "content",
        "shape",
        "unknown_field",
        "regions",
        "mask_path",
        "mask_dtype",
        "mask_size",
        "mask_shape",
        "mask_hash",
        "aggregate_bytes",
    ],
)
def test_frozen_material_descriptors_fail_closed_on_identity_metadata_and_bounds(tmp_path, mutation):
    from transformation_portal.materials_v4.contracts import MaterialLimits
    from transformation_portal.orchestrator.upload_receipts import _material_source_paths

    fixture = material_descriptor_fixture(tmp_path)
    receipt, payload = fixture.receipt, fixture.payload
    resources = {"max_input_bytes": 1024 * 1024, "max_pixels": 1}
    if mutation == "missing":
        fixture.path.unlink()
    elif mutation == "digest":
        receipt["sha256"] = "b" * 64
    elif mutation == "size":
        receipt["size_bytes"] += 1
    elif mutation == "oversized":
        receipt["size_bytes"] = MaterialLimits().max_manifest_bytes + 1
    elif mutation == "aggregate_bytes":
        resources["max_input_bytes"] = receipt["size_bytes"] + payload["regions"][0]["mask"]["size_bytes"] - 1
    else:
        if mutation == "source":
            payload["source_sha256"] = "b" * 64
        elif mutation == "content":
            payload["content_sha256"] = "b" * 64
        elif mutation == "shape":
            payload["shape"] = [2, 2]
        elif mutation == "unknown_field":
            payload["extra"] = "unsupported"
        elif mutation == "regions":
            payload["regions"] *= MaterialLimits().max_regions + 1
        else:
            key, value = {
                "mask_path": ("path", "../outside.npy"),
                "mask_dtype": ("dtype", "float64"),
                "mask_size": ("size_bytes", True),
                "mask_shape": ("shape", [2, 2]),
                "mask_hash": ("file_sha256", "not-a-digest"),
            }[mutation]
            payload["regions"][0]["mask"][key] = value
        data = json.dumps(payload).encode()
        fixture.path.write_bytes(data)
        receipt["sha256"], receipt["size_bytes"] = hashlib.sha256(data).hexdigest(), len(data)
    with pytest.raises(UploadOwnershipLost):
        _material_source_paths(tmp_path, receipt, resources)


@pytest.mark.parametrize("budget", ["manifest", "input"])
def test_over_budget_material_receipts_are_rejected_before_opening_json(tmp_path, monkeypatch, budget):
    from transformation_portal.materials_v4.contracts import MaterialLimits
    from transformation_portal.orchestrator import upload_receipts

    fixture = material_descriptor_fixture(tmp_path)
    resources = {"max_input_bytes": 1024 * 1024, "max_pixels": 1}
    if budget == "manifest":
        fixture.receipt["size_bytes"] = MaterialLimits().max_manifest_bytes + 1
    else:
        resources["max_input_bytes"] = fixture.receipt["size_bytes"] - 1

    def unexpected(*_args, **_kwargs):
        pytest.fail("a frozen byte-budget rejection must precede descriptor I/O")

    monkeypatch.setattr(upload_receipts, "snapshot", unexpected)
    with pytest.raises(UploadOwnershipLost):
        upload_receipts._material_source_paths(tmp_path, fixture.receipt, resources)


@pytest.mark.asyncio
async def test_large_frozen_read_sets_use_one_typed_array_parameter(tmp_path):
    from sqlalchemy import Text
    from sqlalchemy.dialects.postgresql import ARRAY, dialect

    selected = tuple(tmp_path / "frozen" / f"member-{index:05d}.jpg" for index in range(4096))

    class Session:
        statement = None

        async def scalars(self, statement):
            self.statement = statement
            return SimpleNamespace(all=lambda: [])

        async def scalar(self, _statement):
            pytest.fail("no matched upload rows must not sample a ledger clock")

    session = Session()
    await bind_upload_references(
        session, JobRecord(id="job_large_readset", created_at=1.0, request={}), "tenant_a", frozen_read_paths=selected
    )
    statement = session.statement.compile(dialect=dialect())
    assert statement.params["upload_read_paths"] == [str(path) for path in selected]
    assert len(statement.params) < 16
    assert str(statement).count("unnest(") == 1
    array = statement.binds["upload_read_paths"].type
    assert isinstance(array, ARRAY) and isinstance(array.item_type, Text)


@pytest.mark.parametrize("pipeline", ["lux-depth", "lux-depth-v5", "lux-depth-v6"])
def test_legacy_material_manifest_retention_keeps_nested_untracked_batches(tmp_path, pipeline):
    import app

    root = tmp_path / "uploads"
    result = stage_upload_batch(upload_root=root, uploads=[IncomingUpload("nested/mask.npy", io.BytesIO(b"old"))], now=1.0)
    manifest = root / "legacy-materials.json"
    manifest.write_text("{}")
    payload = {"pipeline": pipeline, "args": {"materials_manifest": str(manifest)}}
    assert upload_source_paths(payload) == (manifest,)
    retained = set()
    app._add_staged_input_dir(retained, app.Job(id="job_legacy_material", created_at=1.0, request=payload))
    assert cleanup_expired_batches(root, now=1e12, ttl_seconds=1, retained_input_dirs=retained) == []
    assert result.input_dir.exists()
    assert str(root) in retained


@pytest.mark.parametrize(
    "payload",
    [
        {"pipeline": "unknown", "args": {"materials_manifest": "manifest.json"}},
        {"pipeline": "lux-depth-v5", "args": {"other_manifest": "manifest.json"}},
        {"pipeline": "lux-depth-v5", "args": {"materialsManifest": "manifest.json"}},
    ],
)
def test_retention_namespace_fallback_is_field_and_workflow_scoped(tmp_path, payload):
    assert upload_source_paths(payload, upload_read_base=tmp_path, retention_only=True) == ()


@pytest.mark.parametrize("mutation", ["outside", "contained_alias", "loop"])
def test_frozen_member_symlinks_cannot_change_containment_or_identity(tmp_path, mutation):
    root = tmp_path / "frozen"
    root.mkdir()
    member = root / "photo.jpg"
    if mutation == "loop":
        member.symlink_to("photo.jpg")
    else:
        target = (tmp_path if mutation == "outside" else root) / "actual.jpg"
        target.write_bytes(b"retained source")
        member.symlink_to(target)
    with pytest.raises(UploadOwnershipLost):
        frozen_selection_paths({"root": str(root), "files": [{"id": "selected", "path": "photo.jpg"}]})


@pytest.mark.asyncio
async def test_frozen_read_through_a_symlink_loop_cannot_bind_its_identity(tmp_path):
    (tmp_path / "loop").symlink_to("loop")

    class Session:
        async def scalars(self, _statement):
            pytest.fail("an unresolvable frozen read must fail before any ledger query")

        async def scalar(self, _statement):
            pytest.fail("an unresolvable frozen read must fail before any ledger query")

    with pytest.raises(UploadOwnershipLost):
        await bind_upload_references(
            Session(),
            JobRecord(id="job_loop_read", created_at=1.0, request={}),
            "tenant_a",
            frozen_read_paths=(tmp_path / "loop",),
        )


def test_upload_read_base_through_a_symlink_loop_is_not_server_owned(tmp_path):
    from transformation_portal.orchestrator.upload_receipts import _resolved_source_paths

    (tmp_path / "loop").symlink_to("loop", target_is_directory=True)
    selected = tmp_path / "selected"
    selected.mkdir()
    # Absolute reads never join the base, so only base resolution can fail.
    with pytest.raises(UploadOwnershipLost, match="server-owned base"):
        _resolved_source_paths([str(selected)], upload_read_base=tmp_path / "loop")


@pytest.mark.parametrize("absolute", [False, True])
def test_upload_read_selector_cannot_normalize_through_a_symlink_loop(tmp_path, absolute):
    (tmp_path / "loop").symlink_to("loop", target_is_directory=True)
    (tmp_path / "selected").mkdir()
    selector = Path("missing/../loop/../selected")
    payload = {"pipeline": "lux-depth-v5", "args": {"input_dir": str(tmp_path / selector if absolute else selector)}}

    with pytest.raises(UploadOwnershipLost, match="path resolution is unavailable"):
        upload_source_paths(payload, upload_read_base=tmp_path)


def test_photographic_frozen_members_keep_their_existing_portable_path_limits(tmp_path):
    from transformation_portal.orchestrator.photography_adapter import PhotographyBindings
    from transformation_portal.orchestrator.upload_receipts import _admitted_source_paths

    root, runtime = tmp_path / "source", tmp_path / "runtime" / "python"
    binding = PhotographyBindings.from_payload(
        {
            "schema": "tp.job.photography.bindings.v1",
            "input_root": str(root),
            "runtime_python": str(runtime),
            "raw_python": None,
            "cache_root": None,
            "companion_root": None,
            "materials_root": None,
        }
    )
    relative = "/".join([*(f"directory-{index}" for index in range(65)), "photo.jpg"])
    plan = {"schema": "tp.execution.plan.v4", "inputs": [{"path": relative}]}
    assert set(_admitted_source_paths(json.dumps(plan).encode(), binding.canonical_bytes)) == {
        root,
        runtime,
        root / relative,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("runtime", [False, True])
async def test_unresolvable_retained_upload_path_stops_all_cleanup(relative_read_app, monkeypatch, runtime):
    case = relative_read_app
    (case.root / "loop").symlink_to("loop", target_is_directory=True)
    payload = {"pipeline": "lux-depth-v3", "args": {"input_dir": "./loop/input"}}
    record = JobRecord(id="job_retained_loop", created_at=1.0, request=payload, effective_request=payload)

    class Repository:
        async def list(self, **_kwargs):
            return ([] if runtime else [record]), 0

    def unexpected(*_args, **_kwargs):
        pytest.fail("unresolvable retained paths must prevent both legacy and ledger cleanup")

    jobs = {record.id: case.app._job_from_record(record)} if runtime else {}
    monkeypatch.setattr(case.app, "JOBS", jobs)
    monkeypatch.setattr(case.app, "_job_repository", Repository)
    monkeypatch.setattr(case.app, "_resolved_portal_upload_root", lambda: case.root / "uploads")
    monkeypatch.setattr(case.app, "cleanup_expired_batches", unexpected)
    monkeypatch.setattr(case.app, "_upload_receipt_store", unexpected)
    monkeypatch.setenv("TP_ORCHESTRATOR_STATE_BACKEND", "postgres")
    assert await case.app._retained_staged_input_dirs() is None
    await case.app._cleanup_expired_upload_batches(1e12)


def test_upload_owner_is_composite_and_role_independent():
    actor = {"accessEmail": "Shared@example.test", "username": "Alice", "role": "operator"}
    assert upload_owner("tenant_a", actor) == upload_owner("tenant_a", {**actor, "role": "admin"})
    assert upload_owner("tenant_a", actor) != upload_owner("tenant_a", {**actor, "username": "bob"})
    assert upload_owner("tenant_a", actor) != upload_owner("tenant_b", actor)
    with pytest.raises(ValueError):
        upload_owner("tenant_a", {"username": "alice"})


@pytest.mark.parametrize(
    "bad", ["", "../upload", "A" * 100, "12345678-1234-4ABC-8DEF-123456789ABC", "00000000-0000-0000-0000-000000000000/"]
)
def test_request_id_is_bounded_canonical_uuid(bad):
    with pytest.raises(ValueError):
        validate_upload_request_id(bad)


def test_completion_seal_binds_request_generation_and_all_retained_source_bytes(tmp_path):
    item, result = sealed(tmp_path)
    digest, data = verify_completion(item)
    assert len(digest) == 64
    assert data == result.to_response_data()
    for wrong in [
        replace(item, request_id=str(uuid.uuid4())),
        replace(item, generation=uuid.uuid4().hex),
        replace(item, owner_sha256="b" * 64),
    ]:
        with pytest.raises(ValueError):
            verify_completion(wrong)
    (result.input_dir / "nested/test.txt").write_bytes(b"modified")
    with pytest.raises(ValueError, match="source changed"):
        verify_completion(item)


def test_receipt_without_completion_seal_cannot_recover_ready(tmp_path):
    item = reservation(tmp_path)
    initialize_batch(item)
    stage_upload_batch(
        upload_root=tmp_path,
        uploads=[IncomingUpload("test.txt", io.BytesIO(b"body"))],
        now=1.0,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    with pytest.raises(ValueError, match="incomplete"):
        verify_completion(item)


def test_legacy_cleanup_never_deletes_ledger_marked_ready_or_partial_batch(tmp_path):
    item, _ = sealed(tmp_path)
    partial = reservation(tmp_path)
    initialize_batch(partial)
    assert (Path(partial.batch_root) / MARKER_FILENAME).is_file()
    assert cleanup_expired_batches(tmp_path, now=1e12, ttl_seconds=1, retained_input_dirs=[]) == []
    assert Path(item.batch_root).exists() and Path(partial.batch_root).exists()


def test_external_batch_lock_is_exclusive_and_survives_batch_removal(tmp_path):
    item = reservation(tmp_path)
    first, second = UploadBatchLock(tmp_path, item.batch_id), UploadBatchLock(tmp_path, item.batch_id)
    try:
        assert first.acquire()
        assert not second.acquire()
        assert not Path(item.batch_root).exists()
        first.close()
        assert second.acquire()
    finally:
        first.close()
        second.close()


def test_symlinked_lock_directory_is_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / ".upload-locks").symlink_to(outside, target_is_directory=True)
    with pytest.raises(OSError):
        UploadBatchLock(tmp_path, reservation(tmp_path).batch_id)


@pytest.mark.asyncio
async def test_cancelled_owner_waits_for_thread_before_stream_or_lock_release(tmp_path):
    entered, finish = threading.Event(), threading.Event()
    stream = io.BytesIO(b"body")
    item = reservation(tmp_path)
    lock = UploadBatchLock(tmp_path, item.batch_id)
    assert lock.acquire()

    def work():
        entered.set()
        assert finish.wait(timeout=5)
        assert not stream.closed
        assert stream.read() == b"body"

    async def owner():
        try:
            await drain_owned(asyncio.to_thread(work))
        finally:
            stream.close()
            lock.close()

    task = asyncio.create_task(owner())
    await asyncio.to_thread(entered.wait, 5)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done() and not stream.closed
    contender = UploadBatchLock(tmp_path, item.batch_id)
    try:
        assert not contender.acquire()
        finish.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed
        assert contender.acquire()
    finally:
        finish.set()
        contender.close()


@pytest.mark.asyncio
async def test_receipt_writes_fail_closed_without_flag_or_managed_tenant(monkeypatch):
    from starlette.requests import Request

    import app

    request = Request({"type": "http", "method": "POST", "path": "/v1/uploads/staging", "headers": []})
    monkeypatch.delenv("TP_PORTAL_UPLOAD_RECEIPTS_ENABLED", raising=False)
    assert (await app._upload_receipt_context(request, write=True))[-1].status_code == 503
    monkeypatch.setenv("TP_PORTAL_UPLOAD_RECEIPTS_ENABLED", "1")

    async def no_managed_tenant(*args, **kwargs):
        return None, None

    monkeypatch.setattr(app, "_pilot_tenant_from_request", no_managed_tenant)
    assert (await app._upload_receipt_context(request, write=True))[-1].status_code == 403


@pytest.mark.asyncio
async def test_exact_receipt_read_remains_api_key_protected(monkeypatch):
    import httpx

    import app

    monkeypatch.setattr(app, "API_KEY_SECRET", "upload-test-key")
    monkeypatch.setattr(app, "ENFORCE_JOB_API_KEY", True)
    app.RATE_LIMIT_BUCKETS.clear()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://testserver") as client:
        response = await client.get("/v1/uploads/staging/receipts/" + str(uuid.uuid4()))
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "UNAUTHORIZED"


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["before_read", "during_verification"])
async def test_expired_receipt_read_never_returns_recovered_paths(tmp_path, monkeypatch, phase):
    from starlette.requests import Request

    import app
    from transformation_portal.ingest import upload_recovery
    from transformation_portal.orchestrator.upload_receipts import UploadReceiptExpired

    item = reservation(tmp_path) if phase == "before_read" else sealed(tmp_path)[0]
    clock = [11.0 if phase == "before_read" else 9.0]
    completions = []

    class Store:
        async def get(self, *_args):
            return item

        async def complete(self, *args):
            assert phase == "during_verification" and clock[0] >= item.expires_at
            completions.append(args)
            raise UploadReceiptExpired("staged upload expired")

    async def context(*_args, **_kwargs):
        return SimpleNamespace(tenant_id=item.tenant_id), item.owner_sha256, Store(), None

    async def authorized(*_args, **_kwargs):
        return None

    original_verify = upload_recovery.verify_completion

    def verify(*args):
        assert phase == "during_verification", "expired staging without a seal must not inspect the filesystem"
        result = original_verify(*args)
        clock[0] = 11.0
        return result

    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_resolved_portal_upload_root", lambda: tmp_path)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", authorized)
    monkeypatch.setattr(app, "_now", lambda: clock[0])
    monkeypatch.setattr(upload_recovery, "verify_completion", verify)
    request = Request({"type": "http", "method": "GET", "path": "/v1/uploads/staging/receipts/" + item.request_id})
    response = await app.get_portal_upload_receipt(request, item.request_id)
    assert response.status_code == 200
    assert json.loads(response.body)["data"] == {"request_id": item.request_id, "status": "expired", "receipt": None}
    assert response.headers["cache-control"] == "no-store"
    assert len(completions) == (phase == "during_verification")
    assert item.state == "staging"
    if phase == "before_read":
        assert not Path(item.batch_root).exists()
    else:
        assert Path(item.batch_root).is_dir()


@pytest.mark.asyncio
async def test_upload_completion_past_ttl_is_typed_without_replaying_or_deleting_body(tmp_path, monkeypatch):
    from starlette.requests import Request

    import app
    from transformation_portal.ingest import upload_recovery
    from transformation_portal.orchestrator.upload_receipts import UploadReceiptExpired

    clock = [1.0]
    stream = io.BytesIO(b"upload bytes")
    parsed = app._ParsedPortalUploadPayload([IncomingUpload("test.txt", stream)], None)
    calls = []

    class Store:
        current = None

        async def reserve(self, item):
            self.current = item
            calls.append("reserve")

        async def get(self, *_args):
            return self.current

        async def complete(self, item, *_args):
            assert clock[0] >= item.expires_at
            calls.append("complete")
            raise UploadReceiptExpired("staged upload expired")

    store = Store()

    async def context(*_args, **_kwargs):
        return SimpleNamespace(tenant_id="tenant_a"), "a" * 64, store, None

    async def authorized(*_args, **_kwargs):
        return None

    async def parse(*_args):
        calls.append("parse")
        return parsed

    original_verify = upload_recovery.verify_completion

    def verify(*args, **kwargs):
        result = original_verify(*args, **kwargs)
        clock[0] = 12.0
        return result

    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", authorized)
    monkeypatch.setattr(app, "_parse_portal_upload_multipart", parse)
    monkeypatch.setattr(app, "PORTAL_UPLOAD_TTL_SECONDS", 10)
    monkeypatch.setattr(app, "_now", lambda: clock[0])
    monkeypatch.setattr(upload_recovery, "verify_completion", verify)
    request = Request(
        {
            "type": "http",
            "method": "POST",
            "path": "/v1/uploads/staging",
            "headers": [(b"x-tp-upload-request-id", str(uuid.uuid4()).encode())],
        }
    )
    response = await app._stage_correlated_portal_upload(request, tmp_path)
    assert response.status_code == 409
    body = json.loads(response.body)
    assert body["error"]["code"] == "CONFLICT"
    assert body["error"]["details"]["reason"] == "upload_expired"
    assert body["data"] is None
    assert calls == ["reserve", "parse", "complete"]
    assert stream.closed and store.current.state == "staging"
    assert Path(store.current.batch_root).is_dir()
    lock = UploadBatchLock(tmp_path, store.current.batch_id)
    try:
        assert lock.acquire(), "the completed worker must release its lock without deleting the fenced batch"
    finally:
        lock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["body_limit", "direct_http", "provider", "mixed_group"])
async def test_correlated_chunked_upload_preserves_http_limit_and_resource_cleanup(tmp_path, monkeypatch, failure):
    import httpx

    import app

    calls = []
    streams = []

    class Store:
        current = None

        async def reserve(self, item):
            calls.append("reserve")
            if failure == "provider":
                raise RuntimeError("receipt provider unavailable")
            if failure == "mixed_group":
                raise ExceptionGroup(
                    "multiple request failures",
                    [app.HTTPException(status_code=413), RuntimeError("receipt provider unavailable")],
                )
            self.current = item

        async def get(self, *_args):
            return self.current

        async def complete(self, *_args):
            pytest.fail("an oversized body must not publish ready")

    store = Store()

    async def context(request, **_kwargs):
        assert "content-length" not in request.headers
        return SimpleNamespace(tenant_id="tenant_a"), "a" * 64, store, None

    async def authorized(*_args, **_kwargs):
        return None

    original_headers = app._parse_portal_upload_part_headers

    def track_headers(*args):
        part = original_headers(*args)
        if part.file_stream is not None:
            streams.append(part.file_stream)
        return part

    def unexpected_staging(*_args, **_kwargs):
        pytest.fail("the failed request must not start a staging worker")

    async def direct_http_rejection(*_args):
        raise app.HTTPException(status_code=413, detail="request body rejected")

    boundary = b"tp-chunked-limit"
    first = partial_multipart(boundary)

    async def chunks():
        yield first
        yield b"x" * 64
        yield b"\r\n--" + boundary + b"--\r\n"

    monkeypatch.setattr(app, "API_KEY_SECRET", "upload-test-key")
    monkeypatch.setattr(app, "ENFORCE_JOB_API_KEY", True)
    monkeypatch.setattr(app, "RATE_LIMIT_PER_MINUTE", 0)
    monkeypatch.setattr(app, "MAX_UPLOAD_REQUEST_BYTES", len(first) + 16)
    monkeypatch.setattr(app, "_portal_staged_uploads_enabled", lambda _actor: True)
    monkeypatch.setattr(app, "_resolved_portal_upload_root", lambda: tmp_path)
    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", authorized)
    monkeypatch.setattr(app, "_parse_portal_upload_part_headers", track_headers)
    monkeypatch.setattr(app, "stage_upload_batch", unexpected_staging)
    if failure == "direct_http":
        monkeypatch.setattr(app, "_parse_portal_upload_multipart", direct_http_rejection)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://testserver") as client:
        response = await client.post(
            "/v1/uploads/staging",
            content=chunks(),
            headers={
                "x-api-key": "upload-test-key",
                "X-TP-Upload-Request-ID": str(uuid.uuid4()),
                "content-type": "multipart/form-data; boundary=" + boundary.decode(),
            },
        )
    expected_status = 413 if failure in {"body_limit", "direct_http"} else 503
    assert response.status_code == expected_status
    assert response.json()["schema"] == "tp.orchestrator.error.v1"
    assert response.json()["error"]["code"] == ("REQUEST_TOO_LARGE" if expected_status == 413 else "SERVICE_UNAVAILABLE")
    assert response.json()["data"] is None
    assert calls == ["reserve"]
    if failure in {"provider", "mixed_group"}:
        assert streams == [] and store.current is None
    else:
        assert len(streams) == (2 if failure == "body_limit" else 0)
        assert all(stream.closed for stream in streams)
        assert store.current.state == "staging" and store.current.receipt_sha256 is None
        assert Path(store.current.batch_root).is_dir()
        lock = UploadBatchLock(tmp_path, store.current.batch_id)
        try:
            assert lock.acquire(), "parser failure must release the drained request's stable batch lock"
        finally:
            lock.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("correlated", [False, True])
async def test_cancelled_multipart_request_closes_custom_spools_before_staging(tmp_path, monkeypatch, correlated):
    from starlette.requests import Request

    import app

    streams = []
    waiting = asyncio.Event()
    blocked = asyncio.Event()
    received = False

    class Store:
        current = None

        async def reserve(self, item):
            self.current = item

        async def get(self, *_args):
            return self.current

        async def complete(self, *_args):
            pytest.fail("a cancelled parser must not publish ready")

    store = Store()

    async def context(*_args, **_kwargs):
        return SimpleNamespace(tenant_id="tenant_a"), "a" * 64, store, None

    async def authorized(*_args, **_kwargs):
        return None

    original_headers = app._parse_portal_upload_part_headers

    def track_headers(*args):
        part = original_headers(*args)
        if part.file_stream is not None:
            streams.append(part.file_stream)
        return part

    def unexpected_staging(*_args, **_kwargs):
        pytest.fail("cancellation before multipart completion must not start a worker")

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": partial_multipart(b"cancel-test"), "more_body": True}
        waiting.set()
        await blocked.wait()
        pytest.fail("the upload request should be cancelled while awaiting its body")

    monkeypatch.setattr(app, "_portal_staged_uploads_enabled", lambda _actor: True)
    monkeypatch.setattr(app, "_resolved_portal_upload_root", lambda: tmp_path)
    monkeypatch.setattr(app, "_upload_receipt_context", context)
    monkeypatch.setattr(app, "_pilot_enforce_request_paths", authorized)
    monkeypatch.setattr(app, "_parse_portal_upload_part_headers", track_headers)
    monkeypatch.setattr(app, "stage_upload_batch", unexpected_staging)
    headers = [(b"content-type", b"multipart/form-data; boundary=cancel-test")]
    if correlated:
        headers.append((b"x-tp-upload-request-id", str(uuid.uuid4()).encode()))
    request = Request({"type": "http", "method": "POST", "path": "/v1/uploads/staging", "headers": headers}, receive)
    task = asyncio.create_task(app.stage_portal_uploads(request))
    try:
        await asyncio.wait_for(waiting.wait(), 5)
        assert len(streams) == 2 and all(not stream.closed for stream in streams)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await request.close()
        assert all(stream.closed for stream in streams)
        if correlated:
            assert store.current.state == "staging" and store.current.receipt_sha256 is None
            assert Path(store.current.batch_root).is_dir()
            lock = UploadBatchLock(tmp_path, store.current.batch_id)
            try:
                assert lock.acquire(), "request cancellation must release its drained batch lock"
            finally:
                lock.close()
        else:
            assert store.current is None
    finally:
        if not task.done():
            task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        for stream in streams:
            stream.close()


def test_durable_seal_digest_rejects_rewritten_source_and_resealed_inventory(tmp_path):
    item, result = sealed(tmp_path)
    original_digest, _ = verify_completion(item)
    frozen = replace(item, receipt_sha256=original_digest)
    source = result.input_dir / "nested/test.txt"
    source.write_bytes(b"modified")
    baseline = json.loads(result.baseline_manifest_path.read_bytes())
    baseline["records"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    baseline["records"][0]["size_bytes"] = len(source.read_bytes())
    result.baseline_manifest_path.write_text(json.dumps(baseline))
    # A coherent rewrite of on-disk metadata still cannot change the database
    # completion identity pinned before the ready response.
    seal_completion(item, result)
    with pytest.raises(ValueError, match="durable seal digest"):
        verify_completion(frozen)


@pytest.mark.parametrize("status", ["pending", "unknown", "expired"])
def test_nonready_observations_cannot_include_upload_paths(tmp_path, status):
    from pydantic import ValidationError

    from transformation_portal.api.v1.uploads import UploadReceiptData

    item, result = sealed(tmp_path)
    with pytest.raises(ValidationError):
        UploadReceiptData(request_id=item.request_id, status=status, receipt=result.to_response_data())
    with pytest.raises(ValidationError):
        UploadReceiptData(request_id=item.request_id, status="ready", receipt=None)


@pytest.mark.parametrize(
    "mutation",
    [
        "extra_file",
        "empty_directory",
        "extra_symlink",
        "linked_ancestor",
        "linked_source",
        "linked_marker",
        "linked_seal",
        "hardlinked_source",
        "extra_root",
        "extra_metadata",
    ],
)
def test_completion_requires_exact_independent_inventory_at_every_level(tmp_path, mutation):
    from transformation_portal.ingest.upload_recovery import SEAL_FILENAME

    item, result = sealed(tmp_path)
    source = result.input_dir / "nested/test.txt"
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"original")
    if mutation == "extra_file":
        (result.input_dir / "unlisted.txt").write_bytes(b"not sealed")
    elif mutation == "empty_directory":
        (result.input_dir / "unlisted").mkdir()
    elif mutation == "extra_symlink":
        (result.input_dir / "unlisted.txt").symlink_to(outside)
    elif mutation == "linked_ancestor":
        (result.input_dir / "nested").rename(result.input_dir / "actual")
        (result.input_dir / "nested").symlink_to("actual", target_is_directory=True)
    elif mutation == "linked_source":
        source.unlink()
        source.symlink_to(outside)
    elif mutation == "hardlinked_source":
        source.unlink()
        os.link(outside, source)
    elif mutation in {"linked_marker", "linked_seal"}:
        path = Path(item.batch_root) / MARKER_FILENAME if mutation == "linked_marker" else result.portal_dir / SEAL_FILENAME
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
    elif mutation == "extra_root":
        (Path(item.batch_root) / "tmp").mkdir()
    else:
        (result.portal_dir / "unlisted.json").write_bytes(b"{}")
    with pytest.raises((ValueError, OSError)):
        verify_completion(item)


def test_recovery_uses_sealed_count_and_existing_configured_file_limit(tmp_path):
    item = reservation(tmp_path)
    initialize_batch(item)
    count = 4097  # A supported configured writer limit, not a new default.
    result = stage_upload_batch(
        upload_root=tmp_path,
        uploads=[IncomingUpload(f"file-{index:04d}.txt", io.BytesIO(b"x")) for index in range(count)],
        now=1.0,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    seal_completion(item, result)
    assert verify_completion(item, max_files=count)[1]["summary"]["file_count"] == count
    assert verify_completion(item)[1] == result.to_response_data()  # GET survives later config reduction.
    with pytest.raises(ValueError, match="inventory"):
        verify_completion(item, max_files=256)


@pytest.mark.parametrize("location", ["input", "metadata", "root"])
def test_completion_rejects_inventory_added_during_source_verification(tmp_path, monkeypatch, location):
    from transformation_portal.ingest import upload_recovery

    item, result = sealed(tmp_path)
    original = upload_recovery._source_digest

    def digest(*args):
        value = original(*args)
        directory = {"input": result.input_dir, "metadata": result.portal_dir, "root": result.batch_root}[location]
        (directory / "unlisted.txt").write_bytes(b"changed during verification")
        return value

    monkeypatch.setattr(upload_recovery, "_source_digest", digest)
    with pytest.raises(ValueError, match="directory changed"):
        verify_completion(item)


def test_metadata_capacity_is_checked_before_a_completion_can_be_sealed(tmp_path):
    from transformation_portal.ingest.upload_recovery import MAX_RECOVERY_METADATA_BYTES

    item, result = sealed(tmp_path)
    with result.capture_metadata_path.open("wb") as stream:
        stream.truncate(MAX_RECOVERY_METADATA_BYTES + 1)
    with pytest.raises(UploadStagingError) as error:
        seal_completion(item, result)
    assert error.value.status_code == 413 and error.value.reason == "upload_recovery_metadata_limit"


def test_completion_seal_capacity_is_a_typed_rejection_without_a_published_seal(tmp_path):
    from transformation_portal.ingest.upload_recovery import MAX_RECOVERY_METADATA_BYTES, SEAL_FILENAME

    item, result = sealed(tmp_path)
    seal_path = result.portal_dir / SEAL_FILENAME
    seal_path.unlink()
    oversized = replace(result, warnings=("x" * MAX_RECOVERY_METADATA_BYTES,))
    assert all(path.stat().st_size < MAX_RECOVERY_METADATA_BYTES for path in result.portal_dir.iterdir())
    with pytest.raises(UploadStagingError) as error:
        seal_completion(item, oversized)
    assert error.value.status_code == 413 and error.value.reason == "upload_recovery_metadata_limit"
    assert not seal_path.exists()


@pytest.mark.asyncio
async def test_empty_upload_read_set_never_touches_filesystem_or_database(monkeypatch):
    import tempfile

    from transformation_portal.orchestrator import upload_receipts

    def unexpected(*args, **kwargs):
        pytest.fail("an empty upload read set must not inspect filesystem paths")

    class Session:
        async def scalar(self, statement):
            pytest.fail("an empty upload read set must not query receipt rows")

        async def scalars(self, statement):
            pytest.fail("an empty upload read set must not query receipt rows")

    monkeypatch.setattr(upload_receipts, "Path", unexpected)
    monkeypatch.setattr(tempfile, "gettempdir", unexpected)
    await bind_upload_references(Session(), JobRecord(id="job", created_at=1.0, request={}), "")


@pytest.mark.asyncio
@pytest.mark.parametrize("tenant_id", [None, ""])
async def test_trusted_direct_admission_cannot_omit_tenant_scope_for_a_tracked_batch(tmp_path, monkeypatch, tenant_id):
    monkeypatch.setenv("TP_PORTAL_UPLOAD_ROOT", str(tmp_path))
    record = JobRecord(id="job", created_at=1.0, request={"args": {"input_dir": str(tmp_path / "upload_tracked" / "input")}})

    class Session:
        def __init__(self):
            self.references = []

        async def scalar(self, statement):
            return 2.0

        async def scalars(self, statement):
            return SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(
                        batch_id="upload_tracked",
                        batch_root=str(tmp_path / "upload_tracked"),
                        tenant_id="tenant_a",
                        state="ready",
                        expires_at=10.0,
                    )
                ]
            )

        def add(self, row):
            self.references.append(row)

    session = Session()
    with pytest.raises(UploadOwnershipLost, match="outside this tenant"):
        await bind_upload_references(session, record, tenant_id)
    assert not session.references


@pytest.mark.asyncio
async def test_reservation_conflict_target_only_covers_the_scoped_request_identity(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager

    from sqlalchemy.dialects import postgresql

    from transformation_portal.orchestrator.upload_receipts import PostgresUploadReceiptStore

    item = reservation(tmp_path)
    statements = []

    class Session:
        @asynccontextmanager
        async def begin(self):
            yield self

        async def scalar(self, statement):
            statements.append(str(statement.compile(dialect=postgresql.dialect())))
            return item.batch_id

    @asynccontextmanager
    async def session():
        yield Session()

    store = PostgresUploadReceiptStore(database_url="postgresql+asyncpg://unused/unused_test")
    monkeypatch.setattr(store, "_session", session)
    await store.reserve(item)
    assert len(statements) == 1
    assert "ON CONFLICT (tenant_id, owner_sha256, request_id) DO NOTHING" in statements[0]


@pytest.mark.parametrize("pipeline", ["lux-depth", "lux-depth-v5", "lux-depth-v6"])
def test_photographic_manifests_retain_their_batches_without_output_paths(tmp_path, pipeline):
    args = {name: str(tmp_path / name) for name in ("input_dir", "companions_manifest", "materials_manifest", "output_dir")}
    paths = upload_source_paths({"pipeline": pipeline, "args": args})
    assert set(paths) == {Path(args[name]) for name in ("input_dir", "companions_manifest", "materials_manifest")}


def test_read_selector_contract_tracks_current_frontdoor_path_aliases(tmp_path):
    import app
    from transformation_portal.orchestrator import upload_receipts

    assert upload_receipts._ARCHIVE_DEFAULT_COMMANDS == app.ARCHIVE_GATE_DEFAULT_COMMANDS
    for canonical, aliases, scope in app.PATH_FIELD_SPECS:
        if scope != app.PATH_SCOPE_INPUT:
            continue
        if canonical in {"policy_yaml", "rights_jsonl"}:
            command = "rights-apply" if canonical == "policy_yaml" else "manifest-build"
            request = {"pipeline": "archive-gate-a", "args": {"archive_command": command}}
        elif canonical in {"companions_manifest", "materials_manifest"}:
            request = {"pipeline": "lux-depth-v5", "args": {}}
        elif canonical == "archive_root":
            request = {"pipeline": "archive-gate-a", "args": {}}
        else:
            request = {"pipeline": "lux-depth-v3", "args": {}}
        for alias in aliases:
            request["args"] = {**request["args"], alias: str(tmp_path / alias)}
            assert tmp_path / alias in upload_source_paths(request), alias


@pytest.mark.parametrize("command", tuple(ARCHIVE_OPERATIONS))
def test_archive_selector_roles_include_every_read_alias_and_exclude_outputs(tmp_path, command):
    from transformation_portal.core.archive_execution_plan import ARCHIVE_OPERATIONS

    operation = ARCHIVE_OPERATIONS[command]
    args = {"archive_command": command, "input_dir": str(tmp_path / "input")}
    expected = {tmp_path / "input"}
    for name in operation.files + operation.directories + operation.outputs:
        parts = name.split("_")
        alias = parts[0] + "".join(part.title() for part in parts[1:])
        args[alias] = str(tmp_path / name)
        if name not in operation.outputs:
            expected.add(tmp_path / name)
    assert set(upload_source_paths({"pipeline": operation.pipeline, "args": args})) == expected


@pytest.mark.asyncio
async def test_all_distinct_input_alias_batches_are_locked_once_in_stable_order(tmp_path, monkeypatch):
    monkeypatch.setenv("TP_PORTAL_UPLOAD_ROOT", str(tmp_path))
    first, second = tmp_path / "upload_aaa" / "input", tmp_path / "upload_zzz" / "input"
    record = JobRecord(
        id="job",
        created_at=1.0,
        request={"args": {"inputDir": str(second), "archiveRoot": str(first / "nested")}},
        effective_request={"args": {"input_dir": str(first), "archive_root": str(second / "nested")}},
    )
    original_resolve = Path.resolve
    resolved_roots = []

    def resolve(path, *args, **kwargs):
        if path == tmp_path:
            resolved_roots.append(path)
        return original_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)

    class Session:
        def __init__(self):
            self.locked, self.references = [], []
            self.selection = None

        async def scalar(self, statement):
            assert self.locked == ["upload_aaa", "upload_zzz"], "sample expiry only after every matching row is locked"
            return 2.0

        async def scalars(self, statement):
            self.selection = str(statement)
            self.locked.extend(["upload_aaa", "upload_zzz"])
            return SimpleNamespace(
                all=lambda: [
                    SimpleNamespace(
                        batch_id=batch, batch_root=str(tmp_path / batch), tenant_id="tenant_a", state="ready", expires_at=10.0
                    )
                    for batch in self.locked
                ]
            )

        def add(self, row):
            self.references.append(row)

    session = Session()
    await bind_upload_references(session, record, "tenant_a")
    assert session.locked == ["upload_aaa", "upload_zzz"]
    assert [(row.batch_id, row.job_id) for row in session.references] == [("upload_aaa", "job"), ("upload_zzz", "job")]
    assert len(upload_source_paths(record.request, record.effective_request)) == 4
    assert not resolved_roots, "admission must use absolute ledger identities, independent of the configured root"
    assert "ORDER BY upload_receipts.batch_id" in session.selection and "FOR UPDATE" in session.selection


def test_retained_scan_preserves_all_aliases_even_when_effective_request_exists(tmp_path):
    import app

    retained = set()
    paths = [tmp_path / name for name in ("input-snake", "input-camel", "archive-snake", "archive-camel")]
    job = SimpleNamespace(
        request={"args": {"inputDir": str(paths[1]), "archiveRoot": str(paths[3])}},
        effective_request={"args": {"input_dir": str(paths[0]), "archive_root": str(paths[2])}},
    )
    app._add_staged_input_dir(retained, job)
    assert retained == {str(path.resolve()) for path in paths}


def test_legacy_nested_archive_selector_keeps_its_entire_staged_batch(tmp_path):
    result = stage_upload_batch(
        upload_root=tmp_path, uploads=[IncomingUpload("nested/source.txt", io.BytesIO(b"body"))], now=1.0
    )
    assert cleanup_expired_batches(tmp_path, now=1e12, ttl_seconds=1, retained_input_dirs=[result.input_dir / "nested"]) == []
    assert result.input_dir.exists()


@pytest.mark.parametrize("retained", ["root", "ancestor", "sibling"])
def test_retained_directory_overlap_protects_legacy_batches_without_pinning_siblings(tmp_path, retained):
    result = stage_upload_batch(upload_root=tmp_path, uploads=[IncomingUpload("source.txt", io.BytesIO(b"body"))], now=1.0)
    selector = {"root": tmp_path, "ancestor": tmp_path.parent, "sibling": tmp_path / "unrelated"}[retained]
    removed = cleanup_expired_batches(tmp_path, now=1e12, ttl_seconds=1, retained_input_dirs=[selector])
    assert removed == ([result.batch_id] if retained == "sibling" else [])
    assert result.input_dir.exists() is (retained != "sibling")


@pytest.mark.parametrize("lock_kind", ["regular", "dangling_symlink"])
def test_legacy_cleanup_never_claims_a_markerless_batch_with_permanent_ledger_lock_identity(tmp_path, lock_kind):
    from transformation_portal.ingest.upload_recovery import SEAL_FILENAME

    item, result = sealed(tmp_path)
    (result.batch_root / MARKER_FILENAME).unlink()
    (result.portal_dir / SEAL_FILENAME).unlink()
    if lock_kind == "regular":
        lock = UploadBatchLock(tmp_path, item.batch_id)
        lock.close()
    else:
        directory = tmp_path / ".upload-locks"
        directory.mkdir()
        (directory / (item.batch_id + ".lock")).symlink_to(tmp_path / "missing-lock-target")
    assert cleanup_expired_batches(tmp_path, now=1e12, ttl_seconds=1, retained_input_dirs=[]) == []
    assert result.input_dir.exists()


@pytest.mark.parametrize("state", ["ready", "staging"])
def test_markerless_deletion_requires_a_fenced_deleting_ledger_state(tmp_path, state):
    from transformation_portal.ingest.upload_recovery import delete_owned_batch

    item, result = sealed(tmp_path)
    digest, data = verify_completion(item)
    (result.batch_root / MARKER_FILENAME).unlink()
    item = replace(
        item,
        state=state,
        receipt_sha256=digest if state == "ready" else None,
        response_data=data if state == "ready" else None,
    )
    lock = UploadBatchLock(tmp_path, item.batch_id)
    assert lock.acquire()
    try:
        with pytest.raises(ValueError):
            delete_owned_batch(item, tmp_path)
        assert result.input_dir.exists()
    finally:
        lock.close()


@pytest.mark.parametrize("mutation", ["mismatched_marker", "symlink_root", "wrong_root"])
def test_deleting_retry_never_weakens_root_or_existing_marker_ownership(tmp_path, mutation):
    from transformation_portal.ingest import upload_recovery

    item, result = sealed(tmp_path)
    digest, data = verify_completion(item)
    item = replace(item, state="deleting", receipt_sha256=digest, response_data=data)
    if mutation == "mismatched_marker":
        marker = result.batch_root / MARKER_FILENAME
        payload = json.loads(marker.read_bytes())
        marker.write_bytes(upload_recovery.canonicalize_json({**payload, "owner_sha256": "b" * 64}))
    elif mutation == "symlink_root":
        outside = tmp_path / "outside"
        result.batch_root.rename(outside)
        result.batch_root.symlink_to(outside, target_is_directory=True)
    lock = UploadBatchLock(tmp_path, item.batch_id)
    assert lock.acquire()
    try:
        with pytest.raises(ValueError):
            upload_recovery.delete_owned_batch(item, tmp_path / "wrong" if mutation == "wrong_root" else tmp_path)
        assert result.input_dir.exists()
    finally:
        lock.close()


def test_verification_rejects_an_earlier_file_rewritten_while_a_later_file_is_hashed(tmp_path, monkeypatch):
    from transformation_portal.ingest import upload_recovery

    item = reservation(tmp_path)
    initialize_batch(item)
    result = stage_upload_batch(
        upload_root=tmp_path,
        uploads=[IncomingUpload("a.txt", io.BytesIO(b"original")), IncomingUpload("b.txt", io.BytesIO(b"original"))],
        now=item.created_at,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    seal_completion(item, result)
    original = upload_recovery._source_digest
    verified = []

    def digest(directory_fd, name, size):
        value = original(directory_fd, name, size)
        verified.append(name)
        if len(verified) == 2:
            (result.input_dir / verified[0]).write_bytes(b"modified")
        return value

    monkeypatch.setattr(upload_recovery, "_source_digest", digest)
    with pytest.raises(ValueError, match="changed"):
        verify_completion(item)
    assert len(verified) == 2


def test_source_identities_are_rechecked_at_the_end_of_complete_verification(tmp_path, monkeypatch):
    from transformation_portal.ingest import upload_recovery

    item, result = sealed(tmp_path)
    original = upload_recovery._verify_input_inventory

    def inventory(*args, **kwargs):
        value = original(*args, **kwargs)
        (result.input_dir / "nested/test.txt").write_bytes(b"modified")
        return value

    monkeypatch.setattr(upload_recovery, "_verify_input_inventory", inventory)
    with pytest.raises(ValueError, match="changed"):
        verify_completion(item)


@pytest.mark.parametrize("metadata", ["marker", "seal", "baseline", "capture", "receipt"])
def test_metadata_identities_survive_until_the_end_of_source_verification(tmp_path, monkeypatch, metadata):
    from transformation_portal.ingest import upload_recovery

    item, result = sealed(tmp_path)
    paths = {
        "marker": result.batch_root / MARKER_FILENAME,
        "seal": result.portal_dir / upload_recovery.SEAL_FILENAME,
        "baseline": result.baseline_manifest_path,
        "capture": result.capture_metadata_path,
        "receipt": result.upload_receipt_path,
    }
    target = paths[metadata]
    original = upload_recovery._source_digest

    def digest(*args, **kwargs):
        value = original(*args, **kwargs)
        target.write_bytes(target.read_bytes() + b" ")
        return value

    monkeypatch.setattr(upload_recovery, "_source_digest", digest)
    with pytest.raises(ValueError, match="changed"):
        verify_completion(item)


def test_verified_nested_directory_cannot_be_replaced_by_a_link_during_later_hashing(tmp_path, monkeypatch):
    from transformation_portal.ingest import upload_recovery

    item = reservation(tmp_path)
    initialize_batch(item)
    result = stage_upload_batch(
        upload_root=tmp_path,
        uploads=[
            IncomingUpload("one/deep/a.txt", io.BytesIO(b"original")),
            IncomingUpload("two/deep/b.txt", io.BytesIO(b"original")),
        ],
        now=item.created_at,
        reserved_batch_id=item.batch_id,
        cleanup_on_error=False,
    )
    seal_completion(item, result)
    original = upload_recovery._source_digest
    verified = []

    def digest(directory_fd, name, size):
        value = original(directory_fd, name, size)
        verified.append(name)
        if len(verified) == 2:
            earlier = result.input_dir / ("one" if verified[0] == "a.txt" else "two") / "deep"
            outside = tmp_path / "retained-directory"
            earlier.rename(outside)
            earlier.symlink_to(outside, target_is_directory=True)
        return value

    monkeypatch.setattr(upload_recovery, "_source_digest", digest)
    with pytest.raises(ValueError, match="changed"):
        verify_completion(item)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["dangling_symlink", "hardlink", "acquire_oserror", "acquire_valueerror"])
async def test_one_unavailable_cleanup_lock_does_not_starve_later_batches(tmp_path, monkeypatch, caplog, failure):
    from transformation_portal.ingest import upload_recovery

    bad, bad_result = sealed(tmp_path)
    good, good_result = sealed(tmp_path)
    lock_path = tmp_path / ".upload-locks" / (bad.batch_id + ".lock")
    if failure == "dangling_symlink":
        lock_path.parent.mkdir()
        lock_path.symlink_to(tmp_path / "missing-lock")
    else:
        lock = UploadBatchLock(tmp_path, bad.batch_id)
        lock.close()
        if failure == "hardlink":
            os.link(lock_path, tmp_path / "second-lock-name")
        else:
            inode = lock_path.stat().st_ino
            acquire = UploadBatchLock.acquire

            def failing_acquire(lock, *args, **kwargs):
                if os.fstat(lock.fd).st_ino == inode:
                    raise (OSError if failure == "acquire_oserror" else ValueError)("lock unavailable")
                return acquire(lock, *args, **kwargs)

            monkeypatch.setattr(UploadBatchLock, "acquire", failing_acquire)

    class Store:
        def __init__(self):
            self.claimed, self.finished = [], []

        async def cleanup_candidates(self, now):
            return [bad, good]

        async def claim_deletion(self, expected, now, retained):
            self.claimed.append(expected.batch_id)
            return replace(expected, state="deleting", generation=uuid.uuid4().hex)

        async def finish_deletion(self, expected):
            self.finished.append(expected.batch_id)

    store = Store()
    assert await upload_recovery.cleanup_correlated_batches(store, tmp_path, 20.0, set()) == [good.batch_id]
    assert store.claimed == store.finished == [good.batch_id]
    assert bad_result.input_dir.exists() and not good_result.batch_root.exists()
    assert bad.batch_id in caplog.text
