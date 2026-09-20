"""Closed photography-plan and materialized multi-input identity contracts."""

from __future__ import annotations

import copy
import json
from importlib.resources import files

import jsonschema
import pytest

from transformation_portal.core.execution_identity_v4 import MaterializedExecutionIdentityV4
from transformation_portal.core.execution_plan import ExecutionPlanError
from transformation_portal.core.execution_plan_v2 import (
    ExecutionPlanV2,
    digest_payload,
    parse_execution_plan,
    photography_nodes,
)
from transformation_portal.ingest.canonical_json import canonicalize_json

pytestmark = pytest.mark.unit


def payload():
    configuration = {"input_color": "srgb", "target_size": 518, "strength": 0.25, "clarity": 0.0, "preview_maps": False}
    result = {
        "schema": "tp.execution.plan.v2",
        "canonicalization": "tp.canonical.json.v1",
        "pipeline": "lux_depth_v4",
        "model": {
            "canonical_key": "da3_metric",
            "repo_id": "depth-anything/DA3METRIC-LARGE",
            "revision": "a" * 40,
            "license": "apache-2.0",
        },
        "device": "cpu",
        "inputs": [{"id": "input-0000", "path": "image.png", "sha256": "a" * 64, "size_bytes": 100}],
        "configuration": configuration,
        "resources": {
            "max_pixels": 1000000,
            "max_input_bytes": 1024,
            "max_output_bytes": 1024**3,
            "wall_time_seconds": 30,
            "memory_mib": 1024,
            "inference_slots": 1,
        },
        "nodes": photography_nodes(configuration),
    }
    return signed(result)


def signed(value):
    value = copy.deepcopy(value)
    value.pop("plan_fingerprint_sha256", None)
    value["plan_fingerprint_sha256"] = digest_payload(value)
    return value


def identity(plan, *, node_id="preprocess", input_id="input-0000", inputs=None, model=None):
    return MaterializedExecutionIdentityV4.from_plan(
        plan,
        node_id=node_id,
        input_id=input_id,
        inputs=inputs or {"source": "a" * 64},
        source_identity_sha256="b" * 64,
        runtime_identity_sha256="c" * 64,
        model_identity_sha256=model,
    )


def test_v2_roundtrip_is_immutable_and_version_dispatched():
    source = payload()
    plan = ExecutionPlanV2.from_payload(source)
    source["configuration"]["strength"] = 1.0
    projected = plan.to_payload()
    projected["configuration"]["strength"] = 0.0
    assert plan.to_payload()["configuration"]["strength"] == 0.25
    assert parse_execution_plan(plan.canonical_bytes) == plan


@pytest.mark.parametrize(
    "path",
    [
        ".",
        "../escape.png",
        "/absolute.png",
        "a/../b.png",
        "a\\b.png",
        "a\nb.png",
        "a\u202eb.png",
        "image.png ",
        "folder./image.png",
    ],
)
def test_plan_rejects_nonportable_input_paths(path):
    value = payload()
    value["inputs"][0]["path"] = path
    with pytest.raises(ExecutionPlanError):
        ExecutionPlanV2.from_payload(signed(value))


def test_plan_rejects_normalization_and_case_collisions():
    value = payload()
    value["inputs"] = [
        {"id": "input-0000", "path": "A.png", "sha256": "a" * 64, "size_bytes": 100},
        {"id": "input-0001", "path": "a.png", "sha256": "b" * 64, "size_bytes": 100},
    ]
    with pytest.raises(ExecutionPlanError):
        ExecutionPlanV2.from_payload(signed(value))


@pytest.mark.parametrize("field", ["memory_mib", "wall_time_seconds", "inference_slots"])
def test_resource_counts_are_exact_integers(field):
    value = payload()
    value["resources"][field] = float(value["resources"][field])
    with pytest.raises(ExecutionPlanError, match="exact integer"):
        ExecutionPlanV2.from_payload(signed(value))


def test_input_size_cannot_exceed_declared_budget():
    value = payload()
    value["resources"]["max_input_bytes"] = 99
    with pytest.raises(ExecutionPlanError, match="resource budget"):
        ExecutionPlanV2.from_payload(signed(value))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), {1}, (1,)])
def test_non_json_values_are_rejected_before_coercion(invalid):
    value = payload()
    value["configuration"]["strength"] = invalid
    with pytest.raises(ExecutionPlanError):
        ExecutionPlanV2.from_payload(value)


def test_packaged_schema_itself_rejects_executable_or_unknown_node_fields():
    schema = json.loads(files("transformation_portal.schemas.execution").joinpath("plan.v2.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    value = payload()
    value["nodes"][0]["module"] = "untrusted.module"
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(value)


def test_plan_rejects_validly_rehashed_port_and_config_drift():
    value = payload()
    value["nodes"][2]["inputs"]["master"] = "preprocess.proxy"
    with pytest.raises(ExecutionPlanError):
        ExecutionPlanV2.from_payload(signed(value))
    value = payload()
    value["nodes"][2]["configuration"]["strength"] = 0.5
    with pytest.raises(ExecutionPlanError, match="closed photography"):
        ExecutionPlanV2.from_payload(signed(value))


def test_v2_plan_rejects_duplicate_keys_and_noncanonical_carrier():
    plan = ExecutionPlanV2.from_payload(payload())
    with pytest.raises(ExecutionPlanError):
        parse_execution_plan(
            plan.canonical_bytes.replace(
                b'"schema":"tp.execution.plan.v2"', b'"schema":"tp.execution.plan.v2","schema":"tp.execution.plan.v2"'
            )
        )
    with pytest.raises(ExecutionPlanError, match="canonical"):
        ExecutionPlanV2(json.dumps(payload(), indent=2).encode())


def test_identity_is_factory_only_and_rejects_forged_plan_carriers():
    with pytest.raises(ExecutionPlanError, match="factory-only"):
        MaterializedExecutionIdentityV4()

    class DuckPlan:
        def to_payload(self):
            return payload()

    with pytest.raises(ExecutionPlanError, match="core-owned"):
        identity(DuckPlan())
    forged = object.__new__(ExecutionPlanV2)
    changed = payload()
    changed["device"] = "cuda"
    object.__setattr__(forged, "canonical_bytes", canonicalize_json(changed))
    with pytest.raises(ExecutionPlanError):
        identity(forged)


def test_identity_separates_planned_inputs_with_identical_bytes():
    value = payload()
    value["inputs"].append({"id": "input-0001", "path": "image.tiff", "sha256": "a" * 64, "size_bytes": 100})
    plan = ExecutionPlanV2.from_payload(signed(value))
    assert identity(plan).execution_identity_sha256 != identity(plan, input_id="input-0001").execution_identity_sha256
    with pytest.raises(ExecutionPlanError, match="prepared inventory"):
        identity(plan, input_id="input-9999")


def test_depth_identity_requires_model_and_model_free_stages_reject_it():
    plan = ExecutionPlanV2.from_payload(payload())
    with pytest.raises(ExecutionPlanError, match="materialized model"):
        identity(plan, node_id="depth", inputs={"proxy": "d" * 64})
    with pytest.raises(ExecutionPlanError, match="Model-free"):
        identity(plan, model="d" * 64)
    assert identity(plan, node_id="depth", inputs={"proxy": "d" * 64}, model="e" * 64).execution_identity_sha256


def test_identity_binds_every_named_consumed_artifact():
    plan = ExecutionPlanV2.from_payload(payload())
    with pytest.raises(ExecutionPlanError, match="every declared"):
        identity(plan, node_id="enhance", inputs={"master": "a" * 64})
    first = identity(plan, node_id="enhance", inputs={"master": "a" * 64, "depth": "b" * 64, "proxy": "d" * 64})
    second = identity(plan, node_id="enhance", inputs={"master": "a" * 64, "depth": "c" * 64, "proxy": "d" * 64})
    assert first.execution_identity_sha256 != second.execution_identity_sha256
    changed_proxy = identity(plan, node_id="enhance", inputs={"master": "a" * 64, "depth": "b" * 64, "proxy": "e" * 64})
    assert first.execution_identity_sha256 != changed_proxy.execution_identity_sha256


def test_serialized_identity_requires_independent_expected_authority():
    plan = ExecutionPlanV2.from_payload(payload())
    original = identity(plan)
    options = dict(
        expected_plan=plan,
        node_id="preprocess",
        input_id="input-0000",
        inputs={"source": "a" * 64},
        source_identity_sha256="b" * 64,
        runtime_identity_sha256="c" * 64,
    )
    assert MaterializedExecutionIdentityV4.from_payload(original.to_payload(), **options) == original
    changed = original.to_payload()
    changed["source_identity_sha256"] = "d" * 64
    with pytest.raises(ExecutionPlanError, match="independently materialized"):
        MaterializedExecutionIdentityV4.from_payload(changed, **options)


def with_calibration():
    value = payload()
    source = value["inputs"][0]
    source["companions"] = {
        "path": source["path"],
        "source_sha256": source["sha256"],
        "calibration": {
            "width": 20,
            "height": 10,
            "fx": 300,
            "fy": 300,
            "cx": 9.5,
            "cy": 4.5,
            "source": "measured camera fixture",
            "coordinate_space": "canonical_master",
        },
    }
    value["nodes"] = photography_nodes(value["configuration"], companions=True)
    value["companions_manifest"] = {"path": "companions.json", "sha256": "b" * 64, "size_bytes": 100}
    return signed(value)


def test_optional_inputs_require_explicit_named_identity_bindings():
    plan = ExecutionPlanV2.from_payload(with_calibration())
    with pytest.raises(ExecutionPlanError, match="every declared stage input"):
        identity(plan, node_id="depth", inputs={"proxy": "d" * 64}, model="e" * 64)
    assert identity(plan, node_id="depth", inputs={"proxy": "d" * 64, "calibration": "f" * 64}, model="e" * 64)


@pytest.mark.parametrize("mutation", ["hash", "path", "geometry", "provenance", "coordinate_space", "port"])
def test_companion_semantics_cannot_be_changed_by_rehashing(mutation):
    value = with_calibration()
    companion = value["inputs"][0]["companions"]
    if mutation == "hash":
        companion["source_sha256"] = "b" * 64
    elif mutation == "path":
        companion["path"] = "other.png"
    elif mutation == "geometry":
        companion["calibration"]["cx"] = 20
    elif mutation == "provenance":
        companion["calibration"]["source"] = "estimated"
    elif mutation == "coordinate_space":
        companion["calibration"]["coordinate_space"] = "original_exif"
    else:
        value["nodes"][1]["inputs"].pop("calibration")
    with pytest.raises(ExecutionPlanError):
        ExecutionPlanV2.from_payload(signed(value))
