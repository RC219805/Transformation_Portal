"""Native Materials crosses prepared parent and real V2 process boundaries."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import tifffile

from tests.lux_depth_v3.test_geometry_orchestrator_handoff import _depth_stage, _orchestrator, _v2_arguments
from tests.lux_depth_v3.test_native_materials_binding import prepared_native
from transformation_portal.lux_depth_v3.config import EnhanceConfig
from transformation_portal.lux_depth_v3.execution_lifecycle import prepare_lux_execution
from transformation_portal.lux_depth_v3.orchestrator import EnhanceOrchestrator

pytestmark = pytest.mark.unit


def _prepared(path, binding, *, preset="none"):
    return prepare_lux_execution(
        EnhanceConfig(
            model_key="da3-metric",
            depth_device="cpu",
            geometry_recipe="normalized_crop_support_v1",
            v2_input_color="linear_srgb",
            v2_preset=preset,
            output_bit_depth=16,
            enable_depth_cache=False,
            save_float_depth=True,
            native_materials_manifest=Path(binding["evidence_root"]) / binding["manifest"]["path"],
            native_materials_policy=Path(binding["evidence_root"]) / "policy.json",
        ),
        path.parent,
        [path],
    )


def test_prepared_native_stage_reaches_actual_child_without_image_carrier(prepared_native, tmp_path, monkeypatch):
    source, _, _, binding = prepared_native
    before = source.read_bytes()
    prepared = _prepared(source, binding)
    # Reuse only the controlled depth-model boundary. Parent preparation,
    # source/evidence admission, publication and V2 subprocess are production.
    _orchestrator(tmp_path / "controlled-model", monkeypatch)
    parent = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "native-output")
    parent.postprocessor.process = lambda result: result
    depth, metadata = _depth_stage(parent, source)
    arguments = _v2_arguments(parent, source, depth, metadata)
    arguments["native_materials_source"] = source.name
    result, _, report = parent._run_v2_stage(**arguments)
    assert result["status"] == "success"
    assert result["processing_version"] == "1.4.0"
    assert result["native_materials"]["binding_sha256"] == binding["binding_sha256"]
    pixels = tifffile.imread(result["output"])
    assert pixels.dtype == np.uint16 and pixels.shape == (31, 45, 3)
    assert json.loads(report.read_text())["native_materials"] == result["native_materials"]
    assert not list((parent.output_root / "temp").glob("native-materials-*"))
    assert not list((parent.output_root / "temp").glob("*_materials_v3_enhanced.*"))
    assert source.read_bytes() == before


@pytest.mark.parametrize("drift", ["source", "mask", "manifest"])
def test_parent_revalidates_native_evidence_before_output_or_model(prepared_native, tmp_path, monkeypatch, drift):
    source, _, _, binding = prepared_native
    prepared = _prepared(source, binding)
    root = Path(binding["evidence_root"])
    target = {"source": source, "mask": next(root.glob("mask-*.npy")), "manifest": root / binding["manifest"]["path"]}[drift]
    data = target.read_bytes()
    target.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
    output = tmp_path / "never-created"
    with pytest.raises(ValueError):
        EnhanceOrchestrator.from_prepared(prepared, output)
    assert not output.exists()


def test_native_execution_cannot_enter_legacy_constructor(prepared_native, tmp_path):
    source, _, _, binding = prepared_native
    config = _prepared(source, binding).runtime_config
    with pytest.raises(ValueError, match="prepared canonical"):
        EnhanceOrchestrator(config, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def _output_snapshot(root):
    return {path.relative_to(root).as_posix(): None if path.is_dir() else path.read_bytes() for path in root.rglob("*")}


def _native_parent(prepared_native, tmp_path, monkeypatch, *, preset="none"):
    source, _, _, binding = prepared_native
    prepared = _prepared(source, binding, preset=preset)
    _orchestrator(tmp_path / "controlled-model", monkeypatch)
    parent = EnhanceOrchestrator.from_prepared(prepared, tmp_path / "native-output")
    parent.postprocessor.process = lambda result: result
    return source, binding, parent


@pytest.mark.parametrize(
    "drift",
    [
        "rehash_policy",
        "remove_binding",
        "manifest_selector",
        "policy_selector",
        "geometry_recipe",
        "v2_input_color",
        "enable_v2",
        "enable_materials_v3",
        "v2_preset",
        "output_bit_depth",
    ],
)
def test_native_runtime_drift_fails_before_depth_child_or_output(prepared_native, tmp_path, monkeypatch, drift):
    from transformation_portal.lux_depth_v3.native_materials import native_materials_binding_hash

    source, _, parent = _native_parent(prepared_native, tmp_path, monkeypatch)
    depth, metadata = _depth_stage(parent, source)
    arguments = _v2_arguments(parent, source, depth, metadata)
    arguments["native_materials_source"] = source.name
    original_output = _output_snapshot(parent.output_root)
    if drift == "rehash_policy":
        binding = parent.config.native_materials_binding
        binding["policy"]["operations"][0]["strength"] *= 2
        binding["binding_sha256"] = native_materials_binding_hash(binding)
    elif drift == "remove_binding":
        parent.config.native_materials_binding = None
    elif drift.endswith("selector"):
        setattr(parent.config, "native_materials_" + drift.removesuffix("_selector"), tmp_path / "new-selector.json")
    else:
        value = {
            "geometry_recipe": "legacy_encoded_crop_stretch_v1",
            "v2_input_color": "srgb",
            "enable_v2": False,
            "enable_materials_v3": True,
            "v2_preset": "architectural",
            "output_bit_depth": 8,
        }[drift]
        setattr(parent.config, drift, value)

    def forbidden(*_args, **_kwargs):
        pytest.fail("Changed native authority reached model, child or output work")

    monkeypatch.setattr(parent.depth_backend, "compute", forbidden)
    monkeypatch.setattr(parent.v2_runner, "run", forbidden)
    operations = (
        lambda: _depth_stage(parent, source),
        lambda: parent._run_v2_stage(**arguments),
        lambda: parent.enhance_batch(source.parent),
        parent.compute_config_fingerprint,
        parent._build_run_card_config_fingerprint,
    )
    for operation in operations:
        with pytest.raises(ValueError, match="Native Materials"):
            operation()
    assert _output_snapshot(parent.output_root) == original_output


def test_legacy_runtime_cannot_introduce_native_binding_after_construction(prepared_native, tmp_path, monkeypatch):
    source, _, _, binding = prepared_native
    parent = _orchestrator(tmp_path / "legacy", monkeypatch)
    original_output = _output_snapshot(parent.output_root)
    parent.config.native_materials_binding = binding
    with pytest.raises(ValueError, match="authoritative V2 plan"):
        _depth_stage(parent, source)
    assert _output_snapshot(parent.output_root) == original_output


@pytest.mark.parametrize("selector", ["native_materials_policy", "native_materials_manifest"])
def test_mutated_unpaired_selectors_fail_before_runtime_path_resolution(prepared_native, monkeypatch, selector):
    from transformation_portal.lux_depth_v3 import execution_lifecycle

    source, _, _, _ = prepared_native
    config = EnhanceConfig(model_key="da3-metric")
    setattr(config, selector, source.parent / "missing.json")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Unpaired native selector reached runtime selection")

    monkeypatch.setattr(execution_lifecycle, "resolve_prepared_da3_python_executable", forbidden)
    with pytest.raises(ValueError, match="paired explicit"):
        prepare_lux_execution(config, source.parent, [source])


def test_legacy_constructor_rejects_mutated_policy_selector_before_output(tmp_path):
    config = EnhanceConfig()
    config.native_materials_policy = tmp_path / "policy.json"
    output = tmp_path / "never-created"
    with pytest.raises(ValueError, match="prepared canonical"):
        EnhanceOrchestrator(config, output)
    assert not output.exists()


def test_child_binding_snapshot_is_detached_from_mutable_config(prepared_native, tmp_path, monkeypatch):
    _, binding, parent = _native_parent(prepared_native, tmp_path, monkeypatch)
    selected = parent._validate_native_materials_authority()
    selected["policy"]["operations"][0]["strength"] *= 2
    assert parent.config.native_materials_binding == binding
    assert parent._validate_native_materials_authority() == binding


def test_native_batch_emits_source_bound_processing_identity_in_run_card(prepared_native, tmp_path, monkeypatch):
    import hashlib

    source, binding, parent = _native_parent(prepared_native, tmp_path, monkeypatch, preset="architectural")
    results = parent.enhance_batch(source.parent)
    assert [result["status"] for result in results] == ["ok"]
    run_cards = [path for path in parent.output_root.glob("run_card_*.json") if not path.name.endswith(".self.json")]
    assert len(run_cards) == 1
    card = json.loads(run_cards[0].read_text())
    fingerprint = card["config_fingerprint"]
    assert fingerprint["native_materials_sha256"] == binding["binding_sha256"]
    assert fingerprint["v2_processing_version"] == "1.4.0"
    canonical = json.loads(fingerprint["canonical_json"])
    assert canonical["native_materials_sha256"] == binding["binding_sha256"]
    assert fingerprint["sha256"] == hashlib.sha256(fingerprint["canonical_json"].encode()).hexdigest()
    reports = [json.loads(path.read_text()) for path in parent.v2_dir.rglob("*_report.json")]
    assert reports and all(report["native_materials"]["binding_sha256"] == binding["binding_sha256"] for report in reports)
    assert all(report["depth_map"].endswith(".npy") and report["depth_consumed"] for report in reports)


def test_source_changed_after_parent_construction_fails_before_depth_compute(prepared_native, tmp_path, monkeypatch):
    source, _, parent = _native_parent(prepared_native, tmp_path, monkeypatch)
    original_output = _output_snapshot(parent.output_root)
    data = source.read_bytes()
    source.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))

    def forbidden(*_args, **_kwargs):
        pytest.fail("A source outside the frozen native selection reached the model")

    monkeypatch.setattr(parent.depth_backend, "compute", forbidden)
    with pytest.raises(ValueError, match="source changed before depth"):
        _depth_stage(parent, source)
    assert _output_snapshot(parent.output_root) == original_output
