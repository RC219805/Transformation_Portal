"""Native V6 integration: real decode/plan/products/replay, fixture neural worker."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile
from PIL import Image

from tests.lux_depth_v6.test_depth_pro_backend import _metadata
from transformation_portal.depth.backends.depth_pro import DepthProBackend
from transformation_portal.ingest.canonical_json import canonicalize_json
from transformation_portal.lux_depth_v3.execution_evidence import ArtifactEvidenceError
from transformation_portal.lux_depth_v6 import depth_pro as native
from transformation_portal.lux_depth_v6.__main__ import main
from transformation_portal.lux_depth_v6.color import GradeRecipe
from transformation_portal.lux_depth_v6.plan import OutputLimits, digest
from transformation_portal.lux_depth_v6.source import SourceLimits

pytestmark = pytest.mark.unit


@pytest.fixture
def native_case(tmp_path, monkeypatch):
    monkeypatch.setattr(
        native,
        "require_process_supervisor",
        lambda: SimpleNamespace(Process=lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0))),
    )
    root = tmp_path / "originals"
    root.mkdir()
    Image.new("RGB", (5, 3), (80, 110, 145)).save(root / "kitchen.png")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"fixture-checkpoint")
    monkeypatch.setattr(
        native.backend,
        "_checkpoint_identity",
        lambda path: {
            "path": str(path),
            "sha256": DepthProBackend.EXPECTED_SHA256,
            "size_bytes": 18,
        },
    )

    def runtime(python, model, **kwargs):
        return {
            "schema": "tp.lux.depth_pro.runtime.v1",
            "python_executable": str(python),
            "checkpoint": {"path": str(model), "sha256": DepthProBackend.EXPECTED_SHA256, "size_bytes": 18},
            "python": "3.12 fixture",
            "python_prefix": "/fixture",
            "python_binary_sha256": "1" * 64,
            "python_binary_size_bytes": 1,
            "source_sha256": "2" * 64,
            "distributions": [
                {
                    "name": name,
                    "version": "1.0",
                    "direct_url_sha256": "3" * 64,
                    "record_sha256": "4" * 64,
                    "installed_files_sha256": "5" * 64,
                }
                for name in sorted(("depth-pro", "torch", "torchvision", "numpy", "pillow", "jsonschema"))
            ],
        }

    monkeypatch.setattr(native.backend, "runtime_identity", runtime)

    def infer(authority, model_input, shape, *, memory_mib, checkpoint):
        checkpoint()
        return np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1, _metadata(authority, shape)

    monkeypatch.setattr(native.backend, "infer", infer)
    return native.NativeDepthProRequest(
        root,
        tmp_path / "output",
        Path(sys.executable),
        checkpoint,
        non_commercial_ok=True,
        accept_license=True,
        input_color="srgb",
    )


def test_native_plan_and_replay_produce_honest_v6_products(native_case):
    prepared = native.prepare(native_case)
    assert not native_case.output_dir.exists()
    payload = prepared.plan.to_payload()
    assert payload["schema"] == "tp.lux.depth_pro.plan.v1"
    assert payload["backend"] == "depth_pro"
    result = native.run(prepared)
    verified = native.verify(result.output_root, source_root=native_case.input_dir, expected_plan_sha256=result.plan_sha256)
    assert verified.to_payload()["input_count"] == 1
    photograph = json.loads((result.output_root / "input-0000/photograph.json").read_bytes())
    assert photograph["production_acceptance"] == "not_established"
    assert tifffile.imread(result.output_root / "input-0000/delivery.tif").dtype == np.uint16
    assert (result.output_root / "input-0000/source.bin").read_bytes() == (native_case.input_dir / "kitchen.png").read_bytes()


def test_repeated_native_fixture_execution_is_byte_identical(native_case):
    first = native.run(native.prepare(native_case))
    second = native.run(native.prepare(replace(native_case, output_dir=native_case.output_dir.with_name("second"))))
    assert first.plan_sha256 == second.plan_sha256
    for path in first.output_root.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (second.output_root / path.relative_to(first.output_root)).read_bytes()


@pytest.mark.parametrize(
    "field,value",
    [("non_commercial_ok", False), ("accept_license", False), ("non_commercial_ok", 1), ("accept_license", "true")],
)
def test_native_requires_explicit_typed_acknowledgements(native_case, field, value):
    with pytest.raises(ValueError, match="acknowledgements"):
        native.prepare(replace(native_case, **{field: value}))
    assert not native_case.output_dir.exists()


@pytest.mark.parametrize("change", ["source", "selection", "runtime", "processing"])
def test_frozen_admission_rejects_change_before_publication(native_case, monkeypatch, change):
    prepared = native.prepare(native_case)
    if change == "source":
        Image.new("RGB", (5, 3), "red").save(native_case.input_dir / "kitchen.png")
    elif change == "selection":
        Image.new("RGB", (5, 3)).save(native_case.input_dir / "extra.png")
    elif change == "runtime":
        monkeypatch.setattr(native.backend, "runtime_identity", lambda *args, **kwargs: {"fixture": "changed"})
    else:
        monkeypatch.setattr(native, "processing_identity", lambda: {})
    with pytest.raises(ValueError):
        native.run(prepared)
    assert not native_case.output_dir.exists()


@pytest.mark.parametrize(
    "filename", ["master.npy", "estimated-depth-meters.npy", "native-depth.npy", "source.bin", "worker.json"]
)
def test_changed_products_cannot_be_reauthorized_by_inventory(native_case, filename):
    result = native.run(native.prepare(native_case))
    path = result.output_root / "input-0000" / filename
    if filename.endswith(".npy"):
        array = np.load(path, allow_pickle=False)
        array.flat[0] += 1
        np.save(path, array, allow_pickle=False)
    elif filename == "source.bin":
        path.write_bytes(path.read_bytes() + b"tampered")
    else:
        receipt = json.loads(path.read_bytes())
        receipt["execution_authority"]["candidate_id"] = "da3"
        path.write_bytes(canonicalize_json(receipt))
    evidence = json.loads(result.evidence_path.read_bytes())
    record = next(item for item in evidence["artifacts"] if item["path"] == f"input-0000/{filename}")
    record.update(size_bytes=path.stat().st_size, sha256=digest(path.read_bytes()))
    result.evidence_path.write_bytes(canonicalize_json(evidence))
    with pytest.raises(ValueError):
        native.verify(result.output_root, source_root=native_case.input_dir)


def test_verification_requires_exact_namespace_and_expected_plan(native_case):
    result = native.run(native.prepare(native_case))
    with pytest.raises(ValueError, match="expected exact"):
        native.verify(result.output_root, source_root=native_case.input_dir, expected_plan_sha256="0" * 64)
    (result.output_root / "extra").write_text("unplanned")
    with pytest.raises(ValueError, match="namespace"):
        native.verify(result.output_root, source_root=native_case.input_dir)


def test_native_cancellation_leaves_no_completion(native_case, monkeypatch):
    prepared = native.prepare(native_case)
    with pytest.raises(RuntimeError, match="cancelled"):
        native.run(prepared, cancellation=lambda: True)
    assert not native_case.output_dir.exists()

    def fail(*args, **kwargs):
        raise RuntimeError("fixture worker failure")

    monkeypatch.setattr(native.backend, "infer", fail)
    with pytest.raises(RuntimeError, match="worker failure"):
        native.run(prepared)
    assert native_case.output_dir.exists()
    assert not (native_case.output_dir / "evidence.json").exists()


@pytest.mark.parametrize(
    "limits", [SourceLimits(max_pixels=10), SourceLimits(memory_mib=1024), SourceLimits(max_input_bytes=5)]
)
def test_native_bounded_admission(native_case, limits):
    with pytest.raises(ValueError):
        native.prepare(replace(native_case, source_limits=limits))
    assert not native_case.output_dir.exists()


def test_native_output_reserve_and_tighter_replay_budget(native_case):
    with pytest.raises(ValueError, match="output byte admission"):
        native.prepare(replace(native_case, output_limits=OutputLimits(max_output_bytes=100)))
    result = native.run(native.prepare(native_case))
    with pytest.raises(ValueError, match="pixel/working-memory"):
        native.verify(result.output_root, source_root=native_case.input_dir, source_limits=SourceLimits(max_pixels=10))


def test_native_replay_binds_completion_after_long_replay(native_case, monkeypatch):
    result = native.run(native.prepare(native_case))
    replay = native._verify_artifacts

    def changed(*args, **kwargs):
        replay(*args, **kwargs)
        result.evidence_path.write_bytes(b"{}")

    monkeypatch.setattr(native, "_verify_artifacts", changed)
    with pytest.raises(ValueError, match="completion changed"):
        native.verify(result.output_root, source_root=native_case.input_dir)


def test_native_cli_plan_run_verify_and_retained_route_option_guard(native_case, capsys):
    common = [
        "--depth-backend",
        "depth-pro",
        "--input-dir",
        str(native_case.input_dir),
        "--output-dir",
        str(native_case.output_dir),
    ]
    model = [
        "--depth-pro-python",
        str(native_case.python_executable),
        "--depth-pro-checkpoint",
        str(native_case.checkpoint),
        "--non-commercial-ok",
        "--accept-apple-depth-pro-research-license",
        "--input-color",
        "srgb",
    ]
    assert main(common + model + ["--plan"]) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == native.PLAN_SCHEMA
    assert not native_case.output_dir.exists()
    assert main(common + model) == 0
    capsys.readouterr()
    assert main(common + ["--verify"]) == 0
    assert json.loads(capsys.readouterr().out)["verified"] is True
    assert main(common + model + ["--depth-refinement", "bilinear", "--depth-maps"]) == 1
    assert (
        main(["--input-dir", str(native_case.input_dir), "--output-dir", str(native_case.output_dir), "--non-commercial-ok"])
        == 1
    )


def test_native_plan_closed_and_grade_bound(native_case):
    prepared = native.prepare(replace(native_case, grade=GradeRecipe(exposure_stops=0.5)))
    payload = prepared.plan.to_payload()
    payload["extra"] = True
    with pytest.raises(ValueError, match="closed"):
        native.NativeDepthProPlan(canonicalize_json(payload))
    with pytest.raises(ValueError, match="canonical"):
        native.NativeDepthProPlan(prepared.canonical_plan_bytes + b"\n")


@pytest.mark.parametrize("field", ["python_executable", "checkpoint"])
def test_prepared_runtime_cannot_substitute_other_worker_paths(native_case, field):
    prepared = native.prepare(native_case)
    replacement = native_case.output_dir.with_name("other-runtime")
    replacement.write_bytes(b"different")
    with pytest.raises(ValueError, match="runtime paths"):
        native.run(replace(prepared, **{field: replacement}))
    assert not native_case.output_dir.exists()


@pytest.mark.parametrize("key", ["source_limits", "limits"])
def test_plan_requires_every_resource_key(native_case, key):
    payload = native.prepare(native_case).plan.to_payload()
    payload[key].pop(next(iter(payload[key])))
    with pytest.raises(ValueError, match="closed resource"):
        native.NativeDepthProPlan(canonicalize_json(payload))


def test_output_changed_during_final_runtime_scan_cannot_complete(native_case, monkeypatch):
    prepared = native.prepare(native_case)
    runtime = native.backend.runtime_identity
    calls = 0

    def scan(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            path = prepared.output_root / "input-0000/master.npy"
            values = np.load(path, allow_pickle=False)
            values[0, 0, 0] += 0.1
            np.save(path, values, allow_pickle=False)
        return runtime(*args, **kwargs)

    monkeypatch.setattr(native.backend, "runtime_identity", scan)
    with pytest.raises(ValueError, match="output bytes changed"):
        native.run(prepared)
    assert not (prepared.output_root / "evidence.json").exists()


def test_tighter_replay_rejects_replaced_source_before_decoding(native_case, monkeypatch):
    result = native.run(native.prepare(native_case))
    Image.new("RGB", (1000, 1000), "blue").save(native_case.input_dir / "kitchen.png")

    def forbid_decode(*args, **kwargs):
        raise AssertionError("Changed source must fail byte binding before decode")

    monkeypatch.setattr(native, "decode_master", forbid_decode)
    with pytest.raises(ValueError):
        native.verify(result.output_root, source_root=native_case.input_dir, source_limits=SourceLimits(max_pixels=15))


def test_batch_input_budget_is_enforced_before_next_decode(native_case, monkeypatch):
    first = native_case.input_dir / "kitchen.png"
    (native_case.input_dir / "second.png").write_bytes(first.read_bytes())
    decode = native.decode_master
    decoded = []

    def observe(*args, **kwargs):
        decoded.append(kwargs["source_name"])
        return decode(*args, **kwargs)

    monkeypatch.setattr(native, "decode_master", observe)
    with pytest.raises(ValueError):
        native.prepare(replace(native_case, source_limits=SourceLimits(max_input_bytes=first.stat().st_size + 1)))
    assert decoded == ["kitchen.png"]


def test_native_deadline_covers_source_revalidation(native_case, monkeypatch):
    prepared = native.prepare(replace(native_case, output_limits=OutputLimits(wall_time_seconds=1)))
    elapsed = 0.0
    snapshot = native.snapshot

    def slow_snapshot(*args, **kwargs):
        nonlocal elapsed
        result = snapshot(*args, **kwargs)
        elapsed = 2.0
        return result

    monkeypatch.setattr(native.time, "monotonic", lambda: elapsed)
    monkeypatch.setattr(native, "snapshot", slow_snapshot)
    with pytest.raises(RuntimeError, match="wall-time"):
        native.run(prepared)
    assert not prepared.output_root.exists()


@pytest.mark.parametrize("link", ["symlink", "hardlink"])
def test_native_original_aliases_fail_closed(native_case, link):
    source = native_case.input_dir / "kitchen.png"
    alias = native_case.input_dir / "alias.png"
    if link == "symlink":
        alias.symlink_to(source)
    else:
        alias.hardlink_to(source)
    with pytest.raises((ValueError, OSError, ArtifactEvidenceError)):
        native.prepare(native_case)
