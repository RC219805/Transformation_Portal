"""Canonical qualification uses real authority contracts and bounded local services."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from tests.lux_depth_v5 import test_pipeline as controlled_pipeline
from tests.test_managed_materials_qualification import _committed_fixture
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v4.photography import output_srgb_icc
from transformation_portal.lux_depth_v6.managed import ManagedLuxDepthV6Request, prepare
from transformation_portal.orchestrator.artifact_store.generation import GenerationPublisher
from transformation_portal.orchestrator.lux_depth_adapter import UnifiedBindings
from transformation_portal.orchestrator.photography_adapter import PhotographyBindings
from transformation_portal.photographic_delivery import managed_qualification as managed

pytestmark = pytest.mark.unit
request_case = controlled_pipeline.request_case


@pytest.fixture(name="authority")
def fixture_authority(request, tmp_path):
    upstream_request = request.getfixturevalue("request_case")
    prepared = prepare(
        ManagedLuxDepthV6Request(upstream_request), publisher=GenerationPublisher(artifact_store=None, record_store=None)
    )
    bindings = PhotographyBindings.from_payload(
        {
            "schema": "tp.job.photography.bindings.v1",
            "input_root": str(prepared.inference.input_root),
            "runtime_python": sys.executable,
            "raw_python": None,
            "cache_root": None,
            "companion_root": None,
            "materials_root": None,
        }
    )
    envelope = UnifiedBindings(
        canonicalize_json(
            {
                "schema": "tp.job.lux_depth.bindings.v1",
                "pipeline": "lux-depth",
                "workflow": "process",
                "photography_bindings": bindings.to_payload(),
            }
        )
    )
    plan, carrier = tmp_path / "plan.json", tmp_path / "bindings.json"
    plan.write_bytes(prepared.plan.canonical_bytes)
    carrier.write_bytes(envelope.canonical_bytes)
    return SimpleNamespace(plan=plan, bindings=carrier, source=upstream_request.input_dir / "ramp.tif")


@pytest.mark.parametrize(
    "url,redis",
    [
        ("https://s3.amazonaws.com", False),
        ("http://minio.example:9000", False),
        ("http://localhost:9000", False),
        ("http://127.0.0.1:9000/path", False),
        ("http://secret:password@127.0.0.1:9000", False),
        ("http://127.0.0.1:9000?secret=private", False),
        ("redis://127.0.0.1:6379/16", True),
        ("redis://127.0.0.1:6379", True),
        ("redis://example.invalid:6379/0", True),
        ("redis://:secret@127.0.0.1:6379/0", True),
    ],
)
def test_qualification_refuses_non_loopback_or_credential_service_urls_without_echoing_values(url, redis):
    with pytest.raises(ValueError, match="loopback") as error:
        managed._loopback_url(url, redis=redis)
    assert url not in str(error.value) and "secret" not in str(error.value)


@pytest.mark.parametrize(
    "url,redis", [("http://127.0.0.1:19000", False), ("http://[::1]:19000/", False), ("redis://127.0.0.1:16379/3", True)]
)
def test_literal_loopback_endpoints_are_explicit_and_unchanged(url, redis):
    assert managed._loopback_url(url, redis=redis) == url


def test_unified_authority_retains_exact_plan_bindings_and_original_hash_without_inference(authority):
    found = managed._canonical_authority(authority.plan, authority.bindings)
    assert found["plan_sha256"] == hashlib.sha256(authority.plan.read_bytes()).hexdigest()
    assert found["bindings_sha256"] == hashlib.sha256(authority.bindings.read_bytes()).hexdigest()
    assert len(found["originals"]) == 1 and found["originals"][0]["path"] == str(authority.source)
    assert found["originals"][0]["bound"]["sha256"] == hashlib.sha256(authority.source.read_bytes()).hexdigest()
    assert "no_inference" in found["scope"]


@pytest.mark.parametrize("mutation", ["bytes", "symlink", "noncanonical", "infer"])
def test_unified_authority_rejects_changed_originals_or_unmeasured_family(authority, mutation):
    if mutation == "bytes":
        authority.source.write_bytes(b"changed original")
    elif mutation == "symlink":
        old = authority.source.with_name("preserved-original.tif")
        authority.source.rename(old)
        authority.source.symlink_to(old)
    elif mutation == "noncanonical":
        authority.plan.write_bytes(authority.plan.read_bytes() + b"\n")
    else:
        document = json.loads(authority.bindings.read_bytes())
        document["workflow"] = "infer"
        authority.bindings.write_bytes(canonicalize_json(document))
    with pytest.raises(ValueError):
        managed._canonical_authority(authority.plan, authority.bindings)


def _docker_projection(root):
    return {
        "mounts": [{"Destination": "/data", "Source": str(root), "Type": "bind", "RW": True}],
        "ports": {"9000/tcp": [{"HostIp": "127.0.0.1", "HostPort": "19000"}]},
        "image": "reviewed-minio:fixture",
        "image_id": "sha256:" + "a" * 64,
        "running": True,
        "entrypoint": "/usr/bin/docker-entrypoint.sh",
        "arguments": ["server", "/data", "--console-address", ":9001"],
    }


def test_canonical_original_hashing_stops_at_one_byte_beyond_recorded_length(tmp_path, monkeypatch):
    path = tmp_path / "original.jpg"
    path.write_bytes(b"original")
    reads = []

    @contextmanager
    def growing_stream(_path, _maximum):
        with path.open("rb") as stream:

            class Growing:
                def fileno(self):
                    return stream.fileno()

                def read(self, amount):
                    reads.append(amount)
                    assert len(reads) == 1, "A growing source must fail before an unbounded second read"
                    return b"originalx"

            yield Growing()

    monkeypatch.setattr(managed.offline, "_open_regular", growing_stream)
    with pytest.raises(ValueError, match="immutable byte bound"):
        managed._bound_file(path, hashlib.sha256(b"original").hexdigest(), 8)
    assert reads == [9] and path.read_bytes() == b"original"


def test_minio_receipt_proves_actual_mount_port_image_and_version_without_inspecting_secrets(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        output = (
            json.dumps(_docker_projection(tmp_path)).encode() if argv[1] == "inspect" else b"minio version RELEASE.fixture\n"
        )
        return SimpleNamespace(stdout=output)

    monkeypatch.setattr(managed.subprocess, "run", run)
    receipt = managed._minio_receipt("tp-native-canonical-minio-unit", tmp_path, "http://127.0.0.1:19000")
    assert receipt["publication_root"] == str(tmp_path) and receipt["image_id"] == "sha256:" + "a" * 64
    assert len(calls) == 2 and "Config.Env" not in calls[0][0][3]
    assert all(call[1]["timeout"] == 15 for call in calls)


@pytest.mark.parametrize("mutation", ["mount", "port", "read_only", "not_running", "image", "other_data", "other_entrypoint"])
def test_minio_receipt_rejects_unproven_filesystem_or_endpoint(tmp_path, monkeypatch, mutation):
    data = _docker_projection(tmp_path)
    if mutation == "mount":
        data["mounts"][0]["Source"] = str(tmp_path / "elsewhere")
    elif mutation == "port":
        data["ports"]["9000/tcp"][0]["HostPort"] = "9000"
    elif mutation == "read_only":
        data["mounts"][0]["RW"] = False
    elif mutation == "not_running":
        data["running"] = False
    elif mutation == "other_data":
        data["arguments"][1] = "/other-data"
    elif mutation == "other_entrypoint":
        data["entrypoint"] = "/bin/sh"
    else:
        data["image_id"] = "mutable-tag"
    monkeypatch.setattr(managed.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(stdout=json.dumps(data).encode()))
    with pytest.raises(ValueError, match="mapping"):
        managed._minio_receipt("tp-native-canonical-minio-unit", tmp_path, "http://127.0.0.1:19000")


def test_canonical_products_reconstruct_object_keys_and_fresh_verifier_paths(tmp_path):
    result, committed, source = _committed_fixture()
    result["provider"] = "canonical-s3"
    products = managed._committed_products(tmp_path, result, committed, source, 0)
    assert len(products) == 4
    for product in products:
        assert Path(product["path"]).is_relative_to(tmp_path / "verified-products" / result["job_id"])
        assert product["storage_job_id"] == result["job_id"]
        assert Path(product["path"]).as_posix().endswith(product["storage_path"])


@pytest.mark.asyncio
async def test_s3_product_verification_streams_exact_committed_object_and_closes_provider(tmp_path, monkeypatch):
    payload = b"independent committed bytes"
    target = tmp_path / "verified-products" / "job" / "master.tif"
    state = SimpleNamespace(closed=False, requested=[])

    class Store:
        async def open_bytes(self, job, key):
            state.requested.append((job, key))

            async def chunks():
                yield payload[:4]
                yield payload[4:]

            return chunks()

        async def close(self):
            state.closed = True

    monkeypatch.setattr(managed, "_s3_store", lambda _spec: Store())
    product = {
        "path": str(target),
        "storage_job_id": "job",
        "storage_path": "photographic-review/master.tif",
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }
    await managed._materialize_s3_products(tmp_path, {}, [product])
    assert target.read_bytes() == payload and state.closed
    assert state.requested == [("job", "photographic-review/master.tif")]
    with pytest.raises(ValueError, match="fresh"):
        await managed._materialize_s3_products(tmp_path, {}, [copy.deepcopy(product)])


def test_canonical_preflight_is_metadata_only_secret_free_and_never_launches(authority, tmp_path, monkeypatch):
    monkeypatch.setenv("TP_MANAGED_QUALIFICATION_DATABASE_URL", "postgresql+asyncpg://localhost/canonical_test")
    environment = {
        "TP_MANAGED_QUALIFICATION_REDIS_URL": "redis://127.0.0.1:16379/0",
        "TP_MANAGED_QUALIFICATION_S3_ENDPOINT": "http://127.0.0.1:19000",
        "TP_MANAGED_QUALIFICATION_S3_BUCKET": "canonical-test-fixture",
        "TP_MANAGED_QUALIFICATION_S3_ACCESS_KEY_ID": "private-test-access",
        "TP_MANAGED_QUALIFICATION_S3_SECRET_ACCESS_KEY": "private-test-secret",
    }
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(
        managed,
        "_manifest",
        lambda _path: [
            {
                "path": str(authority.source),
                "bound": {"sha256": "a" * 64, "size_bytes": 1000, "width": 4, "height": 3},
                "recipe": {"regions": [{"strength": 0}]},
            }
        ],
    )
    monkeypatch.setattr(managed, "runtime_identity", lambda: {"managed_profile_identity": "a" * 64})
    monkeypatch.setattr(managed, "_headroom", lambda *_args: (managed.offline.GIB, ["physical_scratch_headroom"]))
    monkeypatch.setattr(managed, "_minio_receipt", lambda *_args: {"image_id": "sha256:" + "a" * 64})
    monkeypatch.setattr(managed, "_run_process", lambda *_args, **_kwargs: pytest.fail("Preflight must not launch"))
    output = tmp_path / "evidence"
    output.mkdir()
    before = dict(os.environ)
    report = managed.qualify(
        authority.plan,
        output / "fresh",
        "a" * 64,
        provider="canonical-s3",
        canonical_plan=authority.plan,
        canonical_bindings=authority.bindings,
        s3_publication_root=tmp_path,
        s3_container="tp-native-canonical-minio-unit",
        preflight_only=True,
    )
    assert report["status"] == "blocked" and report["blocking_reasons"] == ["physical_scratch_headroom"]
    assert report["provider"] == "canonical-s3" and report["production_flags_changed"] is False
    assert report["canonical_scratch_plan"]["schema"] == "tp.photo.private_canonical_campaign_scratch.v1"
    assert os.environ == before
    raw = (output / "fresh" / "qualification.json").read_text()
    assert "private-test-secret" not in raw and "private-test-access" not in raw
    assert set(path.name for path in (output / "fresh").iterdir()) == {"qualification.json"}


def _unequal_scratch_sources():
    return [
        {"bound": {"size_bytes": 65000, "width": 100, "height": 100}},
        {"bound": {"size_bytes": 60000, "width": 200, "height": 100}},
    ]


def test_canonical_campaign_bound_retains_every_copy_and_only_one_paired_live_operation():
    from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES
    from transformation_portal.photographic_delivery.materials import (
        MAX_MATERIALS_DESCRIPTOR_BYTES,
        MAX_MATERIALS_OVERLAY_BYTES,
    )

    sources = _unequal_scratch_sources()
    authority = {"originals": [{"bound": {"size_bytes": 111}}, {"bound": {"size_bytes": 333}}]}
    plan = managed._canonical_scratch_plan(sources, 3, authority)
    products = [
        8 * row["bound"]["width"] * row["bound"]["height"]
        + 65536
        + MAX_PREVIEW_BYTES
        + MAX_MATERIALS_OVERLAY_BYTES
        + MAX_MATERIALS_DESCRIPTOR_BYTES
        for row in sources
    ]
    terms = plan["terms"]
    assert terms["base_metadata_bytes"] == 256 * managed.offline.MIB
    assert terms["retained_source_and_generation_snapshot_bytes"] == 3 * (65000 + 60000)
    assert terms["retained_original_copy_bytes"] == 111 + 333
    assert terms["all_s3_products_and_independent_verifier_copies_bytes"] == 2 * 3 * sum(products)
    assert terms["one_live_operation_transient_bytes"] == max(65000 + 4 * products[0], 60000 + 4 * products[1])
    assert terms["one_live_operation_transient_bytes"] != 65000 + 4 * max(products)
    assert plan["total_bytes"] == sum(terms.values())
    smaller = managed._canonical_scratch_plan(sources, 2, authority)
    assert plan["total_bytes"] - smaller["total_bytes"] == 2 * sum(products)
    assert all(
        smaller["terms"][key] == value
        for key, value in terms.items()
        if key != "all_s3_products_and_independent_verifier_copies_bytes"
    )


def test_canonical_product_bound_uses_actual_overlay_and_descriptor_caps(monkeypatch):
    from transformation_portal.photographic_delivery import materials

    monkeypatch.setattr(materials, "MAX_MATERIALS_OVERLAY_BYTES", 12345)
    monkeypatch.setattr(materials, "MAX_MATERIALS_DESCRIPTOR_BYTES", 9876)
    from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES

    row = _unequal_scratch_sources()[0]
    plan = managed._canonical_scratch_plan([row], 1, {"originals": []})
    assert plan["product_upper_bound_bytes"] == [100 * 100 * 8 + 65536 + MAX_PREVIEW_BYTES + 12345 + 9876]


@pytest.mark.parametrize("low_root", [None, "workspace", "publication"])
def test_canonical_headroom_checks_both_filesystems_at_unchanged_reserve_and_floor(tmp_path, monkeypatch, low_root):
    from transformation_portal.orchestrator.photographic_admission import ReviewAdmissionConfig

    sources = _unequal_scratch_sources()
    authority = {"originals": [{"bound": {"size_bytes": 2000}}]}
    publication = tmp_path / "publication"
    publication.mkdir()
    total = managed._canonical_scratch_plan(sources, 3, authority)["total_bytes"]
    config = ReviewAdmissionConfig(enabled=True)
    assert config.memory_bytes == 12 * managed.offline.GIB and config.scratch_bytes == 64 * managed.offline.GIB
    reserve = config.interactive_scratch_bytes + config.scratch_floor_bytes
    assert reserve == 4 * managed.offline.GIB
    observed = []

    def usage(path):
        observed.append(path)
        low = path == (tmp_path if low_root == "workspace" else publication if low_root == "publication" else None)
        return SimpleNamespace(free=total + reserve - int(low))

    monkeypatch.setattr(managed.shutil, "disk_usage", usage)
    monkeypatch.setitem(
        sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=100 * managed.offline.GIB))
    )
    scratch, reasons = managed._headroom(sources, 3, tmp_path, authority, publication)
    assert scratch == total and reasons == ([] if low_root is None else ["physical_scratch_headroom"])
    assert tmp_path in observed
    if low_root != "workspace":
        assert publication in observed


def test_retained_local_campaign_formula_is_exactly_unchanged(tmp_path, monkeypatch):
    from transformation_portal.lux_depth_v5.preview import MAX_PREVIEW_BYTES

    sources = _unequal_scratch_sources()
    expected = 256 * managed.offline.MIB + sum(
        2 * row["bound"]["size_bytes"]
        + 3 * 3 * (row["bound"]["width"] * row["bound"]["height"] * 8 + 65536 + 2 * MAX_PREVIEW_BYTES + 256 * 1024)
        for row in sources
    )
    monkeypatch.setattr(managed.shutil, "disk_usage", lambda _path: SimpleNamespace(free=100 * managed.offline.GIB))
    monkeypatch.setitem(
        sys.modules, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(available=100 * managed.offline.GIB))
    )
    assert managed._headroom(sources, 3, tmp_path) == (expected, [])


@pytest.mark.parametrize("scenario", sorted(managed.SCENARIOS))
def test_real_canonical_s3_managed_cohort_faults_and_independent_restart_verifier(authority, tmp_path, scenario):
    container = os.environ.get("TP_MANAGED_QUALIFICATION_S3_CONTAINER")
    publication = os.environ.get("TP_MANAGED_QUALIFICATION_S3_PUBLICATION_ROOT")
    if not container or not publication or not os.environ.get("TP_MANAGED_QUALIFICATION_DATABASE_URL"):
        pytest.skip("Canonical qualification requires owned loopback MinIO/Redis and a dedicated migrated *_test database")
    profile = managed._profile_identity()
    if profile is None:
        pytest.skip("The current source/runtime does not match the governed managed resource profile")
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
    del samples
    recipe = {
        "schema": "tp.photo.materials_recipe.v1",
        "input_stage": "display_referred",
        "regions": [
            {"id": f"preservation-{index}", "label": "stone", "region": rectangle, "strength": 0}
            for index, rectangle in enumerate(([0, 0, 0.5, 0.5], [0.5, 0, 1, 0.5], [0, 0.5, 0.5, 1], [0.5, 0.5, 1, 1]))
        ],
    }
    manifest = retained / "manifest.json"
    with source.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    manifest.write_bytes(
        canonicalize_json(
            {"schema": managed.MANIFEST_SCHEMA, "sources": [{"path": str(source), "sha256": digest, "recipe": recipe}]}
        )
    )
    report = managed.qualify(
        manifest,
        evidence / "fresh",
        profile,
        provider="canonical-s3",
        canonical_plan=authority.plan,
        canonical_bindings=authority.bindings,
        s3_publication_root=Path(publication),
        s3_container=container,
        repeats=2 if scenario == "success" else 1,
        scenario=scenario,
    )
    assert report["status"] == "measured", report["blocking_reasons"]
    result = report["complete_managed_result"]
    assert result["provider"] == "canonical-s3" and result["canonical_fixture_import"]["redis_locator_acquired"] is True
    expected = 2 if scenario == "success" else 1 if scenario == "cancel_transport" else 0
    assert len(result["results"]) == len(report["independent_verification"]["results"]) == expected
    assert all(
        row["products_verified"] == 5 and row["maximum_16bit_code_delta"] <= 1
        for row in report["independent_verification"]["results"]
    )
    assert all(
        result["final_admission"][key] == 0
        for key in ("active", "outstanding", "queued", "memory_reserved_bytes", "scratch_reserved_bytes")
    )
    if expected:
        assert result["transport_ownership_verified"] is True
    if scenario == "success":
        assert report["encoded_products_repeatable"] is True
        phases = [row["phase"] for row in result["phases"]]
        assert "same_tenant_inspection_completed_while_native_held" in phases
        assert "same_tenant_inspection_completed_while_outer_transport_blocked" in phases
